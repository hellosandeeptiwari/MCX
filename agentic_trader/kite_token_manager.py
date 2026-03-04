"""
KITE TOKEN MANAGER — Headless auto-login for Zerodha Kite Connect

Zerodha access tokens expire daily (~6 AM next day). This module:
  1. Logs into Zerodha via HTTP (user_id + password)
  2. Submits TOTP for two-factor auth
  3. Captures the request_token from the redirect
  4. Exchanges it for an access_token via Kite Connect API
  5. Atomically updates .env with the new tokens
  6. Optionally restarts the titan-bot systemd service

Usage:
    # As standalone script (cron / CLI):
    python kite_token_manager.py

    # From code:
    from kite_token_manager import renew_kite_token
    renew_kite_token()

Cron: 45 8 * * 1-5  (8:45 AM IST Mon-Fri, before market open at 9:15)
"""

import os
import sys
import json
import time
import logging
import tempfile
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("kite_token_manager")

# ── Config ──────────────────────────────────────────────────────────────────
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'kite_token.log')

# Zerodha endpoints
KITE_LOGIN_URL = "https://kite.zerodha.com/api/login"
KITE_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
KITE_CONNECT_LOGIN = "https://kite.zerodha.com/connect/login"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ── .env Read/Write ────────────────────────────────────────────────────────

def read_env_values(env_path: str = None) -> dict:
    """Read Zerodha credentials from .env file."""
    env_path = env_path or ENV_FILE
    values = {}

    if not os.path.exists(env_path):
        logger.error(f".env not found: {env_path}")
        return values

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key in ('ZERODHA_API_KEY', 'ZERODHA_API_SECRET',
                           'ZERODHA_USER_ID', 'ZERODHA_PASSWORD',
                           'ZERODHA_TOTP_SECRET', 'ZERODHA_ACCESS_TOKEN',
                           'ZERODHA_REQUEST_TOKEN'):
                    values[key] = val

    return values


def update_env_tokens(access_token: str, request_token: str = None,
                      env_path: str = None) -> bool:
    """Atomically update ZERODHA_ACCESS_TOKEN (and optionally REQUEST_TOKEN) in .env.

    Uses write-to-temp + replace for crash safety.
    Returns True on success.
    """
    env_path = env_path or ENV_FILE

    if not os.path.exists(env_path):
        logger.error(f".env not found: {env_path}")
        return False

    try:
        with open(env_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        access_updated = False
        request_updated = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('ZERODHA_ACCESS_TOKEN='):
                new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
                access_updated = True
            elif request_token and stripped.startswith('ZERODHA_REQUEST_TOKEN='):
                new_lines.append(f'ZERODHA_REQUEST_TOKEN={request_token}\n')
                request_updated = True
            else:
                new_lines.append(line)

        if not access_updated:
            new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
        if request_token and not request_updated:
            new_lines.append(f'ZERODHA_REQUEST_TOKEN={request_token}\n')

        # Atomic write
        env_dir = os.path.dirname(env_path)
        fd, tmp_path = tempfile.mkstemp(dir=env_dir, prefix='.env.tmp.')
        try:
            with os.fdopen(fd, 'w') as f:
                f.writelines(new_lines)
            os.replace(tmp_path, env_path)
            logger.info("✅ .env updated with new Kite tokens")
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    except Exception as e:
        logger.error(f"Failed to update .env: {e}")
        return False


# ── Headless Zerodha Login ─────────────────────────────────────────────────

def generate_totp(secret: str) -> str:
    """Generate current TOTP code from base32 secret."""
    try:
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.now()
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        raise


def headless_login(api_key: str, api_secret: str, user_id: str,
                   password: str, totp_secret: str) -> dict:
    """Perform headless Zerodha login and return access_token + request_token.

    Flow:
      1. POST /api/login  (user_id + password) → request_id
      2. POST /api/twofa   (request_id + TOTP)  → redirect with request_token
      3. kite.generate_session(request_token)    → access_token

    Returns:
        {'access_token': str, 'request_token': str, 'user_name': str}
        or empty dict on failure.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'X-Kite-Version': '3',
    })

    # ── Step 1: Login with user_id + password ──
    logger.info(f"🔐 Step 1: Logging in as {user_id}...")
    login_payload = {
        'user_id': user_id,
        'password': password,
    }

    try:
        resp = session.post(KITE_LOGIN_URL, data=login_payload, timeout=30)
    except requests.RequestException as e:
        logger.error(f"Login request failed: {e}")
        return {}

    if resp.status_code != 200:
        logger.error(f"Login failed: HTTP {resp.status_code} — {resp.text[:300]}")
        return {}

    try:
        login_data = resp.json()
    except ValueError:
        logger.error(f"Login response not JSON: {resp.text[:300]}")
        return {}

    if login_data.get('status') != 'success':
        logger.error(f"Login failed: {login_data}")
        return {}

    request_id = login_data.get('data', {}).get('request_id', '')
    if not request_id:
        logger.error(f"No request_id in login response: {login_data}")
        return {}

    logger.info(f"✅ Step 1 done — got request_id")

    # ── Step 2: Submit TOTP ──
    logger.info("🔐 Step 2: Submitting TOTP...")
    totp_code = generate_totp(totp_secret)

    twofa_payload = {
        'user_id': user_id,
        'request_id': request_id,
        'twofa_value': totp_code,
        'twofa_type': 'totp',
    }

    try:
        resp = session.post(KITE_TWOFA_URL, data=twofa_payload,
                            timeout=30, allow_redirects=False)
    except requests.RequestException as e:
        logger.error(f"TOTP request failed: {e}")
        return {}

    if resp.status_code != 200:
        logger.error(f"TOTP failed: HTTP {resp.status_code} — {resp.text[:300]}")
        return {}

    try:
        twofa_data = resp.json()
    except ValueError:
        logger.error(f"TOTP response not JSON: {resp.text[:300]}")
        return {}

    if twofa_data.get('status') != 'success':
        logger.error(f"TOTP failed: {twofa_data}")
        return {}

    logger.info("✅ Step 2 done — TOTP accepted")

    # ── Step 3: Get request_token via Kite Connect redirect chain ──
    logger.info("🔐 Step 3: Getting request_token from Kite Connect...")
    # Use kite.trade (redirects through kite.zerodha.com/connect/login → /connect/finish → callback)
    connect_url = f"https://kite.trade/connect/login?api_key={api_key}&v=3"

    request_token = None
    try:
        # Follow redirect chain manually, hop by hop
        resp = session.get(connect_url, timeout=30, allow_redirects=False)
        location = resp.headers.get('Location', '')
        logger.info(f"  Redirect 1: HTTP {resp.status_code} → {location[:120]}")

        # Follow up to 5 redirects looking for request_token
        for hop in range(5):
            if 'request_token' in location:
                break
            if resp.status_code not in (301, 302, 303, 307):
                break
            if not location:
                break
            resp = session.get(location, timeout=30, allow_redirects=False)
            location = resp.headers.get('Location', '')
            logger.info(f"  Redirect {hop+2}: HTTP {resp.status_code} → {location[:120]}")

        # Parse request_token from the final redirect URL
        if 'request_token' in location:
            parsed = urlparse(location)
            params = parse_qs(parsed.query)
            request_token = params.get('request_token', [None])[0]

        # Fallback: check response body for request_token
        if not request_token and hasattr(resp, 'text') and 'request_token' in resp.text:
            import re
            match = re.search(r'request_token=([a-zA-Z0-9]+)', resp.text)
            if match:
                request_token = match.group(1)
                logger.info("Found request_token in response body")

    except requests.RequestException as e:
        logger.error(f"Connect redirect failed: {e}")
        return {}

    if not request_token:
        logger.error(f"Could not extract request_token. Last URL: {location[:200]}")
        return {}

    logger.info(f"✅ Step 3 done — got request_token: {request_token[:8]}...")

    # ── Step 4: Generate session (exchange request_token for access_token) ──
    logger.info("🔐 Step 4: Generating Kite session...")
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key, timeout=15)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data['access_token']
        kite.set_access_token(access_token)

        # Verify
        profile = kite.profile()
        user_name = profile.get('user_name', 'Unknown')
        logger.info(f"✅ Step 4 done — access_token obtained for {user_name}")

        return {
            'access_token': access_token,
            'request_token': request_token,
            'user_name': user_name,
        }

    except Exception as e:
        logger.error(f"generate_session failed: {e}")
        return {}


# ── Service Restart ────────────────────────────────────────────────────────

def restart_titan_bot():
    """Restart the titan-bot systemd service to pick up new token."""
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', 'restart', 'titan-bot'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info("🔄 titan-bot service restarted successfully")
            return True
        else:
            logger.warning(f"Service restart failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.info("systemctl not available (not on Linux?) — skipping restart")
        return False
    except Exception as e:
        logger.warning(f"Service restart error: {e}")
        return False


# ── Main Entry Point ───────────────────────────────────────────────────────

def renew_kite_token(env_path: str = None, restart_service: bool = True) -> bool:
    """Headless Kite token renewal.

    1. Reads credentials from .env
    2. Performs headless login (password + TOTP)
    3. Generates access_token via Kite Connect
    4. Saves to .env
    5. Optionally restarts titan-bot

    Returns True on success, False on failure.
    """
    env_path = env_path or ENV_FILE

    # Load .env into os.environ
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        pass

    vals = read_env_values(env_path)

    api_key = vals.get('ZERODHA_API_KEY', '') or os.environ.get('ZERODHA_API_KEY', '')
    api_secret = vals.get('ZERODHA_API_SECRET', '') or os.environ.get('ZERODHA_API_SECRET', '')
    user_id = vals.get('ZERODHA_USER_ID', '') or os.environ.get('ZERODHA_USER_ID', '')
    password = vals.get('ZERODHA_PASSWORD', '') or os.environ.get('ZERODHA_PASSWORD', '')
    totp_secret = vals.get('ZERODHA_TOTP_SECRET', '') or os.environ.get('ZERODHA_TOTP_SECRET', '')

    missing = []
    if not api_key:
        missing.append('ZERODHA_API_KEY')
    if not api_secret:
        missing.append('ZERODHA_API_SECRET')
    if not user_id:
        missing.append('ZERODHA_USER_ID')
    if not password:
        missing.append('ZERODHA_PASSWORD')
    if not totp_secret:
        missing.append('ZERODHA_TOTP_SECRET')

    if missing:
        logger.error(f"❌ Missing credentials in .env: {', '.join(missing)}")
        return False

    # Attempt with retries
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"🔑 Kite token renewal attempt {attempt}/{MAX_RETRIES}")

        result = headless_login(api_key, api_secret, user_id, password, totp_secret)

        if result and result.get('access_token'):
            access_token = result['access_token']
            request_token = result.get('request_token', '')
            user_name = result.get('user_name', 'Unknown')

            # Save to .env
            if update_env_tokens(access_token, request_token, env_path):
                # Also update os.environ for any running process
                os.environ['ZERODHA_ACCESS_TOKEN'] = access_token
                if request_token:
                    os.environ['ZERODHA_REQUEST_TOKEN'] = request_token

                logger.info(f"✅ Kite token renewed for {user_name} at "
                           f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Restart service
                if restart_service:
                    restart_titan_bot()

                return True
            else:
                logger.error("Token obtained but failed to save to .env")
                return False

        if attempt < MAX_RETRIES:
            logger.warning(f"Attempt {attempt} failed, retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    logger.error(f"❌ All {MAX_RETRIES} attempts failed — manual intervention needed")
    return False


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    # Setup logging
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode='a'),
        ]
    )

    parser = argparse.ArgumentParser(description='Zerodha Kite Token Manager')
    parser.add_argument('--no-restart', action='store_true',
                        help='Do not restart titan-bot after renewal')
    parser.add_argument('--env', default=None, help='Path to .env file')
    parser.add_argument('--check', action='store_true',
                        help='Only check if token works, do not renew')
    args = parser.parse_args()

    env_path = args.env or ENV_FILE

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        pass

    if args.check:
        vals = read_env_values(env_path)
        token = vals.get('ZERODHA_ACCESS_TOKEN', '')
        api_key = vals.get('ZERODHA_API_KEY', '')
        if not token or not api_key:
            print("❌ ZERODHA_ACCESS_TOKEN or ZERODHA_API_KEY not in .env")
            sys.exit(1)
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key, timeout=10)
            kite.set_access_token(token)
            profile = kite.profile()
            print(f"✅ Token valid — logged in as {profile.get('user_name', '?')}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Token invalid: {e}")
            sys.exit(1)

    success = renew_kite_token(
        env_path=env_path,
        restart_service=not args.no_restart
    )

    if success:
        print("✅ Kite token renewed successfully")
        sys.exit(0)
    else:
        print("❌ Kite token renewal FAILED — check logs at " + LOG_FILE)
        sys.exit(1)
