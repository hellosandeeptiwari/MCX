"""
DHAN TOKEN MANAGER — Auto-renew DhanHQ access token before expiry

DhanHQ JWT tokens expire after 24 hours. This module:
  1. Reads current token from .env
  2. Decodes JWT to check expiry
  3. If expiring within RENEW_THRESHOLD_HOURS, calls /v2/RenewToken
  4. Atomically updates .env with the new token
  5. Optionally reloads os.environ so running process picks up new token

Usage:
    # As standalone script (cron / CLI):
    python dhan_token_manager.py

    # From code:
    from dhan_token_manager import ensure_token_fresh
    ensure_token_fresh()  # returns True if token is valid (renewed if needed)

Cron: 0 8 * * 1-5  (8:00 AM Mon-Fri, 45 min before OI backfill)
"""

import os
import sys
import json
import time
import base64
import logging
import tempfile
import requests
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger("dhan_token_manager")

# ── Config ──────────────────────────────────────────────────────────────────
RENEW_THRESHOLD_HOURS = 14   # Renew if < 14 hours left (generous margin)
RENEW_URL = "https://api.dhan.co/v2/RenewToken"
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

# ── JWT Helpers ─────────────────────────────────────────────────────────────

def decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification (we just need exp/iat)."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return {}
        payload = parts[1]
        # Add padding
        payload += '=' * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception as e:
        logger.warning(f"JWT decode failed: {e}")
        return {}


def token_expiry_info(token: str) -> dict:
    """Get token expiry details.
    
    Returns:
        {
            'valid': bool,
            'exp_ts': int (unix timestamp),
            'exp_dt': datetime,
            'hours_left': float,
            'expired': bool,
            'needs_renewal': bool  (< RENEW_THRESHOLD_HOURS left)
        }
    """
    payload = decode_jwt_payload(token)
    if not payload:
        return {'valid': False, 'expired': True, 'needs_renewal': True, 'hours_left': 0}
    
    exp_ts = payload.get('exp', 0)
    if not exp_ts:
        return {'valid': False, 'expired': True, 'needs_renewal': True, 'hours_left': 0}
    
    exp_dt = datetime.fromtimestamp(exp_ts)
    now = datetime.now()
    hours_left = (exp_dt - now).total_seconds() / 3600
    expired = exp_dt <= now
    
    return {
        'valid': True,
        'exp_ts': exp_ts,
        'exp_dt': exp_dt,
        'hours_left': hours_left,
        'expired': expired,
        'needs_renewal': expired or hours_left < RENEW_THRESHOLD_HOURS,
    }


# ── .env Read/Write ────────────────────────────────────────────────────────

def read_env_token(env_path: str = None) -> tuple:
    """Read DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN from .env file.
    
    Returns: (client_id, access_token)
    """
    env_path = env_path or ENV_FILE
    client_id = ''
    access_token = ''
    
    if not os.path.exists(env_path):
        logger.error(f".env not found: {env_path}")
        return client_id, access_token
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DHAN_CLIENT_ID='):
                client_id = line.split('=', 1)[1].strip()
            elif line.startswith('DHAN_ACCESS_TOKEN='):
                access_token = line.split('=', 1)[1].strip()
    
    return client_id, access_token


def update_env_token(new_token: str, env_path: str = None) -> bool:
    """Atomically update DHAN_ACCESS_TOKEN in .env file.
    
    Uses write-to-temp + rename for crash safety.
    Returns True on success.
    """
    env_path = env_path or ENV_FILE
    
    if not os.path.exists(env_path):
        logger.error(f".env not found: {env_path}")
        return False
    
    try:
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        updated = False
        new_lines = []
        for line in lines:
            if line.strip().startswith('DHAN_ACCESS_TOKEN='):
                new_lines.append(f'DHAN_ACCESS_TOKEN={new_token}\n')
                updated = True
            else:
                new_lines.append(line)
        
        if not updated:
            logger.error("DHAN_ACCESS_TOKEN line not found in .env")
            return False
        
        # Atomic write: write to temp file, then rename
        env_dir = os.path.dirname(env_path)
        fd, tmp_path = tempfile.mkstemp(dir=env_dir, prefix='.env.tmp.')
        try:
            with os.fdopen(fd, 'w') as f:
                f.writelines(new_lines)
            
            # On Windows, os.rename fails if target exists — use os.replace
            os.replace(tmp_path, env_path)
            logger.info(f"✅ .env updated with new token")
            return True
        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"⚠️ FALLBACK [dhan_token/cleanup_tmp]: {e}")
            raise e
    
    except Exception as e:
        logger.error(f"Failed to update .env: {e}")
        return False


# ── Token Renewal API ───────────────────────────────────────────────────────

def renew_token_api(client_id: str, current_token: str) -> str:
    """Call DhanHQ /v2/RenewToken to get a new token.
    
    Must be called while current token is still valid.
    Note: dhanClientId goes as a HEADER, not body (per DhanHQ docs).
    Only works for tokens generated from Dhan Web (not TOTP-generated).
    
    Returns: new access token string, or empty string on failure.
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'access-token': current_token,
        'dhanClientId': client_id,
    }
    
    try:
        resp = requests.post(RENEW_URL, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            new_token = data.get('accessToken', '')
            expiry_time = data.get('expiryTime', '?')
            if new_token:
                logger.info(f"🔑 Token renewed via RenewToken, new expiry: {expiry_time}")
                return new_token
            else:
                logger.error(f"Renew response missing accessToken: {data}")
        else:
            logger.warning(f"RenewToken failed: HTTP {resp.status_code} — {resp.text[:300]}")
    
    except Exception as e:
        logger.error(f"RenewToken request error: {e}")
    
    return ''


def generate_token_totp(client_id: str, pin: str, totp_secret: str) -> str:
    """Generate a brand new token using TOTP (no existing token needed).
    
    Requires TOTP to be enabled on the Dhan account.
    This works even after the old token has expired.
    
    Args:
        client_id: DhanHQ client ID
        pin: 6-digit Dhan PIN
        totp_secret: TOTP secret from authenticator setup
    
    Returns: new access token string, or empty string on failure.
    """
    try:
        import pyotp
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        return ''
    
    try:
        totp = pyotp.TOTP(totp_secret)
        totp_code = totp.now()
        
        url = (f"https://auth.dhan.co/app/generateAccessToken"
               f"?dhanClientId={client_id}&pin={pin}&totp={totp_code}")
        
        resp = requests.post(url, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            new_token = data.get('accessToken', '')
            expiry_time = data.get('expiryTime', '?')
            if new_token:
                logger.info(f"🔑 Token generated via TOTP, expiry: {expiry_time}")
                return new_token
            else:
                logger.error(f"TOTP generate response missing accessToken: {data}")
        else:
            logger.warning(f"TOTP generate failed: HTTP {resp.status_code} — {resp.text[:300]}")
    
    except Exception as e:
        logger.error(f"TOTP generate error: {e}")
    
    return ''


# ── Main Entry Point ───────────────────────────────────────────────────────

def ensure_token_fresh(env_path: str = None, force: bool = False) -> bool:
    """Check token freshness and renew if needed.
    
    Args:
        env_path: Path to .env file (default: auto-detect)
        force: Force renewal even if not yet due
    
    Returns:
        True if token is valid (present and not expired) after this call.
        False if token is expired AND renewal failed (manual intervention needed).
    """
    env_path = env_path or ENV_FILE
    
    # Ensure .env vars are loaded into os.environ (needed for DHAN_PIN, DHAN_TOTP_SECRET)
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        pass  # dotenv not available — rely on os.environ
    
    client_id, token = read_env_token(env_path)
    
    if not client_id or not token:
        logger.error("❌ DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not set in .env")
        return False
    
    info = token_expiry_info(token)
    
    if not info['valid']:
        logger.error("❌ Token is not a valid JWT")
        return False
    
    logger.info(f"🔑 Token status: {info['hours_left']:.1f}h remaining "
                f"(expires {info['exp_dt'].strftime('%Y-%m-%d %H:%M')})")
    
    if not info['needs_renewal'] and not force:
        logger.info(f"✅ Token fresh ({info['hours_left']:.1f}h left), no renewal needed")
        # Still update os.environ so running process has latest
        os.environ['DHAN_ACCESS_TOKEN'] = token
        return True
    
    # ── Renew ──
    new_token = ''
    
    if info['expired']:
        logger.warning("⚠️ Token EXPIRED — skipping RenewToken, trying TOTP...")
    else:
        reason = "forced" if force else f"{info['hours_left']:.1f}h left < {RENEW_THRESHOLD_HOURS}h threshold"
        logger.info(f"🔄 Renewing token ({reason})...")
        new_token = renew_token_api(client_id, token)
    
    # Fallback: try TOTP-based generation if RenewToken fails or token expired
    if not new_token:
        dhan_pin = os.environ.get('DHAN_PIN', '')
        totp_secret = os.environ.get('DHAN_TOTP_SECRET', '')
        if dhan_pin and totp_secret:
            logger.info("🔄 Trying TOTP-based token generation...")
            new_token = generate_token_totp(client_id, dhan_pin, totp_secret)
        else:
            logger.info("💡 Add DHAN_PIN and DHAN_TOTP_SECRET to .env for auto-renewal")
    
    if not new_token:
        if info['expired']:
            logger.error("❌ All renewal methods failed and token is EXPIRED")
            return False
        else:
            logger.warning(f"⚠️ Renewal failed — token still has "
                          f"{info['hours_left']:.1f}h left")
        # Token not yet expired, still usable
        os.environ['DHAN_ACCESS_TOKEN'] = token
        return not info['expired']
    
    # Verify new token is valid
    new_info = token_expiry_info(new_token)
    if not new_info['valid'] or new_info['expired']:
        logger.error("❌ Renewed token appears invalid/expired — keeping old token")
        os.environ['DHAN_ACCESS_TOKEN'] = token
        return not info['expired']
    
    # Persist to .env
    if update_env_token(new_token, env_path):
        logger.info(f"✅ Token renewed and saved! New expiry: "
                     f"{new_info['exp_dt'].strftime('%Y-%m-%d %H:%M')} "
                     f"({new_info['hours_left']:.1f}h)")
        # Update running process env
        os.environ['DHAN_ACCESS_TOKEN'] = new_token
        return True
    else:
        logger.warning("⚠️ Token renewed but failed to save to .env — "
                        "using new token in-memory only")
        os.environ['DHAN_ACCESS_TOKEN'] = new_token
        return True


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from dotenv import load_dotenv
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description='DhanHQ Token Manager')
    parser.add_argument('--force', action='store_true', help='Force renewal even if not due')
    parser.add_argument('--check', action='store_true', help='Only check status, do not renew')
    parser.add_argument('--env', default=None, help='Path to .env file')
    args = parser.parse_args()
    
    env_path = args.env or ENV_FILE
    
    # Load .env into os.environ first
    load_dotenv(env_path, override=True)
    
    if args.check:
        client_id, token = read_env_token(env_path)
        if not token:
            print("❌ No DHAN_ACCESS_TOKEN in .env")
            sys.exit(1)
        info = token_expiry_info(token)
        if not info['valid']:
            print("❌ Invalid JWT token")
            sys.exit(1)
        status = "EXPIRED" if info['expired'] else f"{info['hours_left']:.1f}h remaining"
        print(f"🔑 Token: {status} (expires {info['exp_dt'].strftime('%Y-%m-%d %H:%M')})")
        sys.exit(0 if not info['expired'] else 1)
    
    success = ensure_token_fresh(env_path, force=args.force)
    if success:
        print("✅ Dhan token is valid and fresh")
        sys.exit(0)
    else:
        print("❌ Dhan token expired — manual renewal needed at https://web.dhan.co/")
        sys.exit(1)
