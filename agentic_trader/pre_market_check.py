#!/usr/bin/env python3
"""
TITAN v5 — Pre-Market Health Check
====================================
Run this BEFORE market open (9:00 AM IST) to catch issues early.
Can be triggered via cron, dashboard button, or manually via SSH.

Checks:
  1. Zerodha token validity (expires daily!)
  2. Dhan API connectivity
  3. Disk space / memory
  4. Log file writability
  5. SQLite DB integrity
  6. ML model loadability
  7. OI data directory
  8. Network latency to Kite API
  9. Bot service status
  10. Critical import chain

Exit codes:
  0 = All checks passed
  1 = Warnings only (non-critical)
  2 = Critical failure (bot WILL fail at market open)
"""

import json
import os
import sys
import time
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

CHECKS_PASSED = 0
CHECKS_WARNED = 0
CHECKS_FAILED = 0


def _ok(msg):
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  ✅ {msg}")


def _warn(msg):
    global CHECKS_WARNED
    CHECKS_WARNED += 1
    print(f"  ⚠️  {msg}")


def _fail(msg):
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  ❌ {msg}")


def check_zerodha_token():
    """Check if Zerodha access token is valid."""
    print("\n🔐 Zerodha Token:")
    try:
        from kiteconnect import KiteConnect
        api_key = os.environ.get('ZERODHA_API_KEY', '')
        token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
        if not api_key or not token:
            _fail("ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN not set in .env")
            return
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        profile = kite.profile()
        name = profile.get('user_name', '?')
        exchanges = profile.get('exchanges', [])
        _ok(f"Valid — {name}, exchanges: {exchanges}")
    except Exception as e:
        _fail(f"INVALID/EXPIRED — {e}")
        _fail("  → Token expires daily! Generate new one from:")
        _fail(f"    https://kite.zerodha.com/connect/login?api_key={api_key}&v=3")


def check_dhan_token():
    """Check Dhan API token."""
    print("\n🔑 Dhan Token:")
    client_id = os.environ.get('DHAN_CLIENT_ID', '')
    token = os.environ.get('DHAN_ACCESS_TOKEN', '')
    if not client_id or not token:
        _warn("DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not set (OI features degraded)")
        return
    # Check JWT expiry if possible
    try:
        import base64
        parts = token.split('.')
        if len(parts) == 3:
            payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
            data = json.loads(base64.b64decode(payload))
            exp = data.get('exp', 0)
            exp_dt = datetime.fromtimestamp(exp)
            if exp_dt < datetime.now():
                _fail(f"Dhan token EXPIRED at {exp_dt}")
            else:
                days_left = (exp_dt - datetime.now()).days
                if days_left < 2:
                    _warn(f"Dhan token expires in {days_left} days ({exp_dt})")
                else:
                    _ok(f"Valid until {exp_dt.strftime('%Y-%m-%d')} ({days_left} days)")
    except Exception as e:
        _warn(f"Cannot decode Dhan JWT: {e}")


def check_disk_memory():
    """Check disk and memory."""
    print("\n💾 Disk & Memory:")
    total, used, free = shutil.disk_usage('/')
    pct = used / total * 100
    if pct > 90:
        _fail(f"Disk {pct:.0f}% full ({free // (1024**3)}GB free)")
    elif pct > 75:
        _warn(f"Disk {pct:.0f}% full ({free // (1024**3)}GB free)")
    else:
        _ok(f"Disk {pct:.0f}% used ({free // (1024**3)}GB free)")

    try:
        with open('/proc/meminfo') as f:
            lines = f.readlines()
        total_kb = int([l for l in lines if 'MemTotal' in l][0].split()[1])
        avail_kb = int([l for l in lines if 'MemAvailable' in l][0].split()[1])
        avail_mb = avail_kb // 1024
        if avail_mb < 200:
            _fail(f"Low memory: {avail_mb}MB available")
        elif avail_mb < 500:
            _warn(f"Memory: {avail_mb}MB available (may be tight under load)")
        else:
            _ok(f"Memory: {avail_mb}MB available")
    except Exception:
        _ok("Memory check skipped (not Linux)")


def check_logs():
    """Check log files are writable and not too large."""
    print("\n📋 Log Files:")
    log_dir = Path(__file__).resolve().parent.parent / 'logs'
    if not log_dir.exists():
        _fail(f"Log directory missing: {log_dir}")
        return

    for logname in ['titan.log', 'titan-error.log', 'dashboard.log', 'dashboard-error.log']:
        logpath = log_dir / logname
        if logpath.exists():
            size_mb = logpath.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                _warn(f"{logname}: {size_mb:.1f}MB (consider rotation)")
            else:
                _ok(f"{logname}: {size_mb:.1f}MB")
        else:
            # Try to create it
            try:
                logpath.touch()
                _ok(f"{logname}: created (was missing)")
            except Exception:
                _warn(f"{logname}: missing and not writable")


def check_database():
    """Check SQLite database integrity."""
    print("\n🗄️  Database:")
    db_path = Path(__file__).parent / 'titan_state.db'
    if not db_path.exists():
        _warn("titan_state.db not found (will be created on first run)")
        return
    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] == 'ok':
            _ok(f"SQLite integrity OK ({db_path.stat().st_size // 1024}KB)")
        else:
            _fail(f"SQLite corruption: {result}")
        conn.close()
    except Exception as e:
        _fail(f"SQLite error: {e}")


def check_ml_models():
    """Check ML models are loadable."""
    print("\n🤖 ML Models:")
    model_dir = Path(__file__).parent / 'ml_models' / 'saved_models'
    if not model_dir.exists():
        _warn("ML model directory missing")
        return

    required = [
        'meta_gate_latest.json',
        'meta_direction_latest.json',
        'move_predictor_latest.json',
    ]
    for fname in required:
        fpath = model_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            _ok(f"{fname} ({size_mb:.1f}MB)")
        else:
            _fail(f"{fname} MISSING — ML predictions will fail")


def check_oi_directory():
    """Check OI data directories exist."""
    print("\n📊 OI Data:")
    data_dir = Path(__file__).parent / 'ml_models' / 'data'
    for dirname in ['futures_oi', 'oi_snapshots']:
        dirpath = data_dir / dirname
        if dirpath.exists():
            count = len(list(dirpath.iterdir()))
            _ok(f"{dirname}/ exists ({count} files)")
        else:
            _warn(f"{dirname}/ missing — creating...")
            try:
                dirpath.mkdir(parents=True, exist_ok=True)
                _ok(f"{dirname}/ created")
            except Exception as e:
                _fail(f"Cannot create {dirname}/: {e}")


def check_network():
    """Check network connectivity to trading APIs."""
    print("\n🌐 Network:")
    import urllib.request
    import urllib.error
    endpoints = [
        ('Kite API', 'https://api.kite.trade'),
        ('OpenAI API', 'https://api.openai.com/v1/models'),
    ]
    for name, url in endpoints:
        try:
            start = time.time()
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=5)
            latency = (time.time() - start) * 1000
            if latency > 500:
                _warn(f"{name}: {latency:.0f}ms (slow)")
            else:
                _ok(f"{name}: {latency:.0f}ms")
        except urllib.error.HTTPError as e:
            # HTTP error but server responded = network is fine
            latency = (time.time() - start) * 1000
            if e.code in (401, 403, 405, 421):
                _ok(f"{name}: reachable ({latency:.0f}ms, HTTP {e.code})")
            else:
                _warn(f"{name}: HTTP {e.code} ({latency:.0f}ms)")
        except Exception as e:
            _warn(f"{name}: {e}")


def check_service():
    """Check bot systemd service."""
    print("\n⚙️  Service:")
    try:
        r = subprocess.run(['systemctl', 'is-active', 'titan-bot'],
                          capture_output=True, text=True, timeout=3)
        if r.stdout.strip() == 'active':
            _ok("titan-bot service: ACTIVE")
        else:
            _warn(f"titan-bot service: {r.stdout.strip()}")
    except Exception:
        _warn("Cannot check service status (not systemd?)")

    try:
        r = subprocess.run(['systemctl', 'is-active', 'titan-dashboard'],
                          capture_output=True, text=True, timeout=3)
        if r.stdout.strip() == 'active':
            _ok("titan-dashboard service: ACTIVE")
        else:
            _warn(f"titan-dashboard service: {r.stdout.strip()}")
    except Exception:
        pass


def check_imports():
    """Check critical Python imports."""
    print("\n📦 Critical Imports:")
    modules = [
        ('config', 'from config import HARD_RULES'),
        ('state_db', 'from state_db import get_state_db'),
        ('trade_ledger', 'from trade_ledger import get_trade_ledger'),
        ('ml_models.predictor', 'from ml_models.predictor import MovePredictor'),
        ('options_trader', 'from options_trader import get_options_trader'),
        ('zerodha_tools', 'from zerodha_tools import ZerodhaTools'),
        ('exit_manager', 'from exit_manager import ExitManager'),
        ('execution_guard', 'from execution_guard import ExecutionGuard'),
    ]
    for name, stmt in modules:
        try:
            exec(stmt)
            _ok(name)
        except Exception as e:
            _fail(f"{name}: {e}")


def main():
    print("=" * 60)
    print("  TITAN v5 — Pre-Market Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 60)

    check_zerodha_token()
    check_dhan_token()
    check_disk_memory()
    check_logs()
    check_database()
    check_ml_models()
    check_oi_directory()
    check_network()
    check_service()
    check_imports()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {CHECKS_PASSED} passed, {CHECKS_WARNED} warnings, {CHECKS_FAILED} failures")

    if CHECKS_FAILED > 0:
        print("  ❌ CRITICAL ISSUES — FIX BEFORE MARKET OPEN!")
        print("=" * 60)
        return 2
    elif CHECKS_WARNED > 0:
        print("  ⚠️  Warnings present — review before market open")
        print("=" * 60)
        return 1
    else:
        print("  ✅ ALL CLEAR — Ready for trading!")
        print("=" * 60)
        return 0


if __name__ == '__main__':
    sys.exit(main())
