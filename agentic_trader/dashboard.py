"""
TITAN v5 â€” Monitoring Dashboard
================================
Real-time log viewer, P&L dashboard, trade history, system health.
Runs as a separate Flask/SSE service alongside the trading bot.
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    HARD_RULES, APPROVED_UNIVERSE, PAPER_MODE,
    TRADING_HOURS, TIER_1_OPTIONS, TIER_2_OPTIONS, TIER_3_OPTIONS,
    ZERODHA_API_KEY,
)
import config as _config_module
from state_db import get_state_db
from trade_ledger import get_trade_ledger

# â”€â”€ Lightweight Kite instance for LIVE exit orders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_dashboard_kite = None

def _get_dashboard_kite():
    """Lazy-init a KiteConnect instance for placing exit orders.
    Reuses the same access token as the bot (from .env).
    Auto-renews via kite_token_manager if token expired."""
    global _dashboard_kite
    from kiteconnect import KiteConnect

    # Fast path: re-validate cached instance
    if _dashboard_kite is not None:
        try:
            _dashboard_kite.profile()
            return _dashboard_kite
        except Exception:
            print("Dashboard cached Kite token expired, attempting renewal")
            _dashboard_kite = None

    # Try current env token
    token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
    if token:
        try:
            kite = KiteConnect(api_key=ZERODHA_API_KEY, timeout=15)
            kite.set_access_token(token)
            kite.profile()
            _dashboard_kite = kite
            return kite
        except Exception:
            print("Dashboard .env ZERODHA_ACCESS_TOKEN invalid, trying auto-renewal")

    # Auto-renew via kite_token_manager (same as bot)
    try:
        from kite_token_manager import renew_kite_token
        print("Dashboard: headless Kite token renewal via TOTP...")
        if renew_kite_token(restart_service=False):
            new_token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
            if new_token:
                kite = KiteConnect(api_key=ZERODHA_API_KEY, timeout=15)
                kite.set_access_token(new_token)
                kite.profile()
                _dashboard_kite = kite
                print("Dashboard Kite token auto-renewed successfully")
                return kite
    except ImportError:
        pass
    except Exception as e:
        print(f"Dashboard auto-renewal failed: {e}")

    return None


def _get_completed_order_fill_price(kite, order_id: str) -> float:
    """Return broker-reported average fill for a completed order."""
    if not kite or not order_id:
        return 0.0
    try:
        order_history = kite.order_history(order_id)
        for item in reversed(order_history):
            avg_price = float(item.get('average_price') or 0)
            if item.get('status') == 'COMPLETE' and avg_price > 0:
                return avg_price
    except Exception:
        pass
    return 0.0

# â”€â”€ App setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
TRADE_LEDGER_DIR = Path(__file__).parent / 'trade_ledger'

# â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _today() -> str:
    return datetime.now().strftime('%Y-%m-%d')


def _safe_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# â”€â”€ Server-Sent Events: live log tail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _tail_file(filepath: str, n=200):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


def _sse_log_stream(filepath: str):
    """Generator: yield new lines as SSE data events.
    
    Optimisations vs original:
    - 100ms poll instead of 500ms â†’ 5x faster log delivery
    - SSE heartbeat comment every 15s â†’ keeps connection alive through
      proxies / gunicorn timeout, prevents [Errno 110] TimeoutError
    - Handles log file rotation (re-open when truncated)
    """
    last_heartbeat = time.time()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(0, 2)
            while True:
                where = f.tell()
                line = f.readline()
                if line:
                    yield f"data: {json.dumps(line.rstrip())}\n\n"
                    last_heartbeat = time.time()
                else:
                    # Check for log rotation (file truncated / replaced)
                    try:
                        cur_size = os.path.getsize(filepath)
                        if cur_size < where:
                            # File was rotated â€” reopen from start
                            f.seek(0)
                            continue
                    except OSError:
                        pass
                    # SSE keepalive comment every 15s (invisible to EventSource)
                    if time.time() - last_heartbeat > 15:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.time()
                    time.sleep(0.1)
    except FileNotFoundError:
        yield f"data: {json.dumps('[log file not found]')}\n\n"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Server boot timestamp â€” changes on every restart (i.e. every deploy)
# Use file mtime so all gunicorn workers share the same version
# (avoids reload-loop when round-robin hits different worker boot times)
import time as _time
_SERVER_BOOT = str(int(os.path.getmtime(__file__)))

@app.route('/')
def index():
    resp = app.make_response(render_template('titan_dashboard.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/version')
def version():
    return jsonify({'v': _SERVER_BOOT})


# â”€â”€ Live log SSE streams â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/logs/stream')
def log_stream():
    logfile = str(LOG_DIR / 'titan.log')
    return Response(_sse_log_stream(logfile),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/logs/stream/error')
def error_log_stream():
    logfile = str(LOG_DIR / 'titan-error.log')
    return Response(_sse_log_stream(logfile),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/logs/recent')
def recent_logs():
    n = int(request.args.get('lines', 200))
    logfile = str(LOG_DIR / 'titan.log')
    lines = _tail_file(logfile, n)
    return jsonify({'lines': [l.rstrip() for l in lines]})


@app.route('/api/logs/errors')
def recent_errors():
    n = int(request.args.get('lines', 100))
    logfile = str(LOG_DIR / 'titan-error.log')
    lines = _tail_file(logfile, n)
    return jsonify({'lines': [l.rstrip() for l in lines]})


# â”€â”€ Smart Log Reader with Bookmarks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Bookmark file stores {file_key: {"line": N, "ts": "ISO"}} so the AI agent
# can call ?set_bookmark=true after reading, and next call with
# ?since_bookmark=true returns ONLY new lines since that position.

_LOG_BOOKMARK_FILE = LOG_DIR / 'log_bookmarks.json'

# All log files the system knows about
_LOG_FILES = {
    'titan':           LOG_DIR / 'titan.log',
    'titan_error':     LOG_DIR / 'titan-error.log',
    'dashboard_error': LOG_DIR / 'dashboard-error.log',
    'dashboard':       LOG_DIR / 'dashboard.log',
    'watchdog':        LOG_DIR / 'watchdog.log',
    'bot_debug':       Path(__file__).parent / 'bot_debug.log',
}


def _load_bookmarks() -> dict:
    try:
        if _LOG_BOOKMARK_FILE.exists():
            with open(_LOG_BOOKMARK_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_bookmarks(bm: dict):
    try:
        _LOG_BOOKMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_BOOKMARK_FILE, 'w') as f:
            json.dump(bm, f, indent=2)
    except Exception:
        pass


def _count_lines(filepath) -> int:
    """Fast line count without loading entire file into memory."""
    try:
        count = 0
        with open(filepath, 'rb') as f:
            for _ in f:
                count += 1
        return count
    except FileNotFoundError:
        return 0


def _read_lines_from(filepath, start_line: int, max_lines: int = 2000) -> list:
    """Read lines from start_line (1-based) up to max_lines."""
    result = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if i < start_line:
                    continue
                result.append(line.rstrip())
                if len(result) >= max_lines:
                    break
    except FileNotFoundError:
        pass
    return result


def _file_meta(filepath) -> dict:
    """Return metadata for a log file."""
    p = Path(filepath)
    if not p.exists():
        return {'exists': False, 'size': 0, 'total_lines': 0, 'modified': None}
    stat = p.stat()
    return {
        'exists': True,
        'size': stat.st_size,
        'size_human': f"{stat.st_size / 1024:.1f}KB" if stat.st_size < 1048576 else f"{stat.st_size / 1048576:.1f}MB",
        'total_lines': _count_lines(p),
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
    }


def _find_last_scan_cycle_line(filepath) -> int:
    """Find line number of last SCAN CYCLE marker."""
    last_scan_line = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if 'SCAN CYCLE @' in line or 'SCAN: scan_and_trade() ENTER' in line:
                    last_scan_line = i
    except FileNotFoundError:
        pass
    return last_scan_line


@app.route('/api/logs/smart')
def smart_logs():
    """Smart log reader with bookmark support.

    Query params:
      file          â€“ log key: titan|titan_error|dashboard_error|watchdog|bot_debug
                      (default: titan). Use "all" for metadata-only overview.
      lines         â€“ max lines to return (default: 300)
      since_bookmark â€“ if "true", return only new lines after saved bookmark
      set_bookmark  â€“ if "true", save current end-of-file as bookmark after read
      from_scan     â€“ if "true", return lines starting from last SCAN CYCLE
      grep          â€“ optional regex filter applied to returned lines
      tail          â€“ if "true" (default), read last N lines; if "false", read from bookmark/scan

    Returns:
      {meta: {file_key: {exists, size, total_lines, modified, bookmark_line}},
       file: str, lines: [...], start_line: int, end_line: int,
       bookmark_was: int|null, bookmark_now: int|null}
    """
    import re as _re

    file_key = request.args.get('file', 'titan')
    max_lines = int(request.args.get('lines', 300))
    since_bm = request.args.get('since_bookmark', '').lower() == 'true'
    set_bm = request.args.get('set_bookmark', '').lower() == 'true'
    from_scan = request.args.get('from_scan', '').lower() == 'true'
    grep_pat = request.args.get('grep', '')
    tail_mode = request.args.get('tail', 'true').lower() != 'false'

    bookmarks = _load_bookmarks()

    # â”€â”€ Always return metadata for ALL log files â”€â”€
    meta = {}
    for key, path in _LOG_FILES.items():
        fm = _file_meta(path)
        bm_line = bookmarks.get(key, {}).get('line', 0)
        fm['bookmark_line'] = bm_line
        fm['new_lines'] = max(0, fm['total_lines'] - bm_line) if fm['exists'] else 0
        meta[key] = fm

    # â”€â”€ If "all", return just metadata overview â”€â”€
    if file_key == 'all':
        if set_bm:
            for key in _LOG_FILES:
                if meta[key]['exists']:
                    bookmarks[key] = {
                        'line': meta[key]['total_lines'],
                        'ts': datetime.now().isoformat(),
                    }
            _save_bookmarks(bookmarks)
        return jsonify({'meta': meta, 'file': 'all', 'lines': [],
                        'note': 'Use ?file=<key> to read a specific log'})

    if file_key not in _LOG_FILES:
        return jsonify({'error': f'Unknown file key: {file_key}',
                        'available': list(_LOG_FILES.keys())}), 400

    log_path = _LOG_FILES[file_key]
    total = meta[file_key]['total_lines']
    bm_was = bookmarks.get(file_key, {}).get('line', 0)

    # â”€â”€ Decide start line â”€â”€
    if from_scan and file_key == 'titan':
        scan_line = _find_last_scan_cycle_line(log_path)
        start = scan_line if scan_line > 0 else max(1, total - max_lines + 1)
    elif since_bm and bm_was > 0:
        start = bm_was + 1
    elif tail_mode:
        start = max(1, total - max_lines + 1)
    else:
        start = 1

    lines = _read_lines_from(log_path, start, max_lines)
    end_line = start + len(lines) - 1 if lines else start

    # â”€â”€ Optional grep filter â”€â”€
    if grep_pat:
        try:
            pat = _re.compile(grep_pat, _re.IGNORECASE)
            lines = [l for l in lines if pat.search(l)]
        except _re.error:
            pass  # bad regex â€” return unfiltered

    # â”€â”€ Set bookmark â”€â”€
    bm_now = None
    if set_bm:
        bm_now = total
        bookmarks[file_key] = {'line': total, 'ts': datetime.now().isoformat()}
        _save_bookmarks(bookmarks)

    return jsonify({
        'meta': meta,
        'file': file_key,
        'start_line': start,
        'end_line': end_line,
        'total_lines': total,
        'returned': len(lines),
        'bookmark_was': bm_was or None,
        'bookmark_now': bm_now,
        'lines': lines,
    })


# â”€â”€ Bot control (start / stop / restart) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route('/api/bot/stop', methods=['POST'])
def bot_stop():
    try:
        r = subprocess.run(['sudo', 'systemctl', 'stop', 'titan-bot'],
                           capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0
        return jsonify({'ok': ok, 'action': 'stop', 'msg': r.stderr.strip() if not ok else 'Bot stopped'})
    except Exception as e:
        return jsonify({'ok': False, 'action': 'stop', 'msg': str(e)}), 500


@app.route('/api/bot/start', methods=['POST'])
def bot_start():
    try:
        r = subprocess.run(['sudo', 'systemctl', 'start', 'titan-bot'],
                           capture_output=True, text=True, timeout=10)
        ok = r.returncode == 0
        return jsonify({'ok': ok, 'action': 'start', 'msg': r.stderr.strip() if not ok else 'Bot started'})
    except Exception as e:
        return jsonify({'ok': False, 'action': 'start', 'msg': str(e)}), 500


@app.route('/api/bot/restart', methods=['POST'])
def bot_restart():
    try:
        r = subprocess.run(['sudo', 'systemctl', 'restart', 'titan-bot'],
                           capture_output=True, text=True, timeout=15)
        ok = r.returncode == 0
        return jsonify({'ok': ok, 'action': 'restart', 'msg': r.stderr.strip() if not ok else 'Bot restarted'})
    except Exception as e:
        return jsonify({'ok': False, 'action': 'restart', 'msg': str(e)}), 500


# ── Trading mode toggle (PAPER ↔ LIVE via .env) ─────────────────────
# We persist the mode in agentic_trader/.env (TRADING_MODE=PAPER|LIVE)
# and require an explicit restart to flip — paper_mode is baked into
# the trader stack at startup, so a runtime flip would leave the bot
# in a half-state. The endpoint returns whether a restart is needed.

_DOTENV_PATH = os.path.join(os.path.dirname(__file__), '.env')


def _read_trading_mode_from_env():
    """Return ('PAPER'|'LIVE', file_exists). Defaults to PAPER on any error."""
    try:
        if not os.path.exists(_DOTENV_PATH):
            return 'PAPER', False
        with open(_DOTENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('TRADING_MODE='):
                    val = line.split('=', 1)[1].strip().strip('"').strip("'").upper()
                    if val in ('PAPER', 'LIVE'):
                        return val, True
    except Exception:
        pass
    return 'PAPER', os.path.exists(_DOTENV_PATH)


def _write_trading_mode_to_env(new_mode: str) -> bool:
    """Atomically rewrite TRADING_MODE=<new_mode> in .env. Adds the line if
    missing. Returns True on success."""
    new_mode = (new_mode or '').upper()
    if new_mode not in ('PAPER', 'LIVE'):
        return False
    try:
        lines = []
        if os.path.exists(_DOTENV_PATH):
            with open(_DOTENV_PATH, 'r') as f:
                lines = f.readlines()
        replaced = False
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped.startswith('TRADING_MODE='):
                lines[i] = f'TRADING_MODE={new_mode}\n'
                replaced = True
                break
        if not replaced:
            lines.append(f'TRADING_MODE={new_mode}\n')
        tmp = _DOTENV_PATH + '.tmp'
        with open(tmp, 'w') as f:
            f.writelines(lines)
        os.replace(tmp, _DOTENV_PATH)
        return True
    except Exception as e:
        print(f"⚠️ .env write failed: {e}")
        return False


@app.route('/api/mode', methods=['GET'])
def mode_get():
    """Return current trading mode read from .env."""
    mode, exists = _read_trading_mode_from_env()
    return jsonify({'ok': True, 'mode': mode, 'env_present': exists})


@app.route('/api/mode', methods=['POST'])
def mode_set():
    """Set TRADING_MODE in .env and (optionally) restart the bot.

    Body:
      {"mode": "LIVE"|"PAPER", "restart": true|false}

    Returns:
      {ok, prev, new, restarted, msg}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        new_mode = (data.get('mode') or '').upper()
        do_restart = bool(data.get('restart', True))
        if new_mode not in ('PAPER', 'LIVE'):
            return jsonify({'ok': False, 'msg': "mode must be 'PAPER' or 'LIVE'"}), 400
        prev, _ = _read_trading_mode_from_env()
        if prev == new_mode:
            return jsonify({'ok': True, 'prev': prev, 'new': new_mode,
                            'restarted': False, 'msg': f'Already in {new_mode} mode'})
        if not _write_trading_mode_to_env(new_mode):
            return jsonify({'ok': False, 'msg': '.env write failed'}), 500
        restarted = False
        msg = f'Mode changed: {prev} → {new_mode}. Restart required to take effect.'
        if do_restart:
            try:
                r = subprocess.run(['sudo', 'systemctl', 'restart', 'titan-bot'],
                                   capture_output=True, text=True, timeout=15)
                restarted = (r.returncode == 0)
                msg = (f'Mode changed: {prev} → {new_mode}. Bot restarted.'
                       if restarted else
                       f'Mode written but restart failed: {r.stderr.strip()}')
            except Exception as e:
                msg = f'Mode written but restart failed: {e}'
        return jsonify({'ok': True, 'prev': prev, 'new': new_mode,
                        'restarted': restarted, 'msg': msg})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ── Trading pause toggle (block new entries / reverses) ─────────────
# Pause does NOT touch existing positions — exit_manager continues
# managing trailing SL / target / EOD. Only NEW entries (scan_and_trade)
# and auto_pilot reverses are blocked. Implemented as a flag file the
# bot polls each tick, so toggle effect is near-instant (<1s).

TRADING_PAUSE_FLAG = os.path.join(os.path.dirname(__file__), 'trading_paused.flag')


@app.route('/api/pause', methods=['GET'])
def pause_get():
    paused = os.path.exists(TRADING_PAUSE_FLAG)
    reason = ''
    ts = ''
    if paused:
        try:
            with open(TRADING_PAUSE_FLAG, 'r') as f:
                d = json.load(f) or {}
                reason = d.get('reason') or ''
                ts = d.get('ts') or ''
        except Exception:
            pass
    return jsonify({'ok': True, 'paused': paused, 'reason': reason, 'ts': ts})


@app.route('/api/pause', methods=['POST'])
def pause_set():
    """Body: {"paused": true|false, "reason": "optional note"}"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        want = bool(data.get('paused'))
        reason = (data.get('reason') or '').strip()[:200]
        if want:
            payload = {'reason': reason or 'Paused via dashboard',
                       'ts': datetime.now().isoformat(timespec='seconds')}
            tmp = TRADING_PAUSE_FLAG + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp, TRADING_PAUSE_FLAG)
            return jsonify({'ok': True, 'paused': True,
                            'msg': '⏸  New entries/reverses paused. Existing positions still managed.'})
        else:
            try:
                if os.path.exists(TRADING_PAUSE_FLAG):
                    os.remove(TRADING_PAUSE_FLAG)
            except Exception:
                pass
            return jsonify({'ok': True, 'paused': False, 'msg': '▶  Trading resumed.'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500




AUTOPILOT_STATE_PATH = os.path.join(os.path.dirname(__file__), 'auto_pilot_state.json')
AUTOPILOT_DISABLE_FLAG = os.path.join(os.path.dirname(__file__), 'auto_pilot_disabled.flag')


def _autopilot_read():
    try:
        if os.path.exists(AUTOPILOT_STATE_PATH):
            with open(AUTOPILOT_STATE_PATH, 'r') as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _autopilot_write(state):
    try:
        tmp = AUTOPILOT_STATE_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, AUTOPILOT_STATE_PATH)
        return True
    except Exception as e:
        print(f"⚠️ autopilot write failed: {e}")
        return False


@app.route('/api/auto_pilot', methods=['GET', 'POST'])
def auto_pilot():
    """Read or toggle the auto-pilot agent running in titan-bot.

    POST body: {"enabled": true|false}
    Response: current state incl. daily counters and recent decisions.
    """
    state = _autopilot_read()
    today = _today()
    # Day-rollover: reset counters if file is from yesterday
    if state.get('date') != today:
        state['date'] = today
        state['counters'] = {}
        state['last_action_ts'] = {}

    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        on = bool(data.get('enabled'))
        state['enabled'] = on
        # Also toggle the physical kill flag so a crashed/restarting bot
        # honors the setting at boot.
        try:
            if on and os.path.exists(AUTOPILOT_DISABLE_FLAG):
                os.remove(AUTOPILOT_DISABLE_FLAG)
            elif not on:
                with open(AUTOPILOT_DISABLE_FLAG, 'w') as f:
                    f.write(datetime.now().isoformat())
        except Exception:
            pass
        _autopilot_write(state)

    # Always normalize counters for display
    counters = state.get('counters', {}) or {}
    return jsonify({
        'ok': True,
        'enabled': bool(state.get('enabled', False)) and not os.path.exists(AUTOPILOT_DISABLE_FLAG),
        'date': state.get('date'),
        'counters': counters,
        'limits': {
            'max_reverses_per_symbol': 7,
            'max_adds_per_symbol': 3,
        },
        'recent_decisions': (state.get('last_decision') or [])[-15:],
        'scope': 'NAKED_OPTION only',
        'aggression': 'Aggressive',
        'llm': 'gpt-5.2',
    })


# ── Reverse trade: EXIT current position + OPEN opposite ───────

@app.route('/api/reverse_trade', methods=['POST'])
def reverse_trade():
    """Reverse a position: exit current, then open opposite (PE↔CE, same strike/lots).
    Fetches live market LTP via Kite API, calculates SL/target like the bot does."""
    try:
        import re, random
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get('symbol', '').strip()
        underlying = data.get('underlying', '').strip()
        direction = data.get('direction', '').strip()  # reversed direction
        quantity = int(data.get('quantity', 0))
        is_option = data.get('is_option', False)
        option_type = data.get('option_type', '')
        strike = data.get('strike', 0)
        expiry = data.get('expiry', '')
        lots = int(data.get('lots', 1))

        if not symbol or not direction or quantity <= 0:
            return jsonify({'ok': False, 'msg': 'Missing required fields'}), 400

        # ── THROTTLE: reject rapid repeat clicks on the same symbol ──
        # Rationale: bot consumes manual_entry/exit signal files on its next
        # save cycle (~2-5s). Stacking multiple reverse pairs faster than that
        # causes symbol-matched signal consumption to drop positions (see
        # _consume_manual_exit_signals symbol-only matching).
        global _REVERSE_LOCKS, _REVERSE_LAST_TS
        try:
            _REVERSE_LOCKS
        except NameError:
            import threading as _th
            _REVERSE_LOCKS = {}   # {symbol: threading.Lock}
            _REVERSE_LAST_TS = {} # {symbol: epoch_seconds of last accepted click}
        import threading as _th
        _now = time.time()
        _COOLDOWN = 5.0  # seconds between accepted reverse clicks per symbol
        _last = _REVERSE_LAST_TS.get(symbol, 0.0)
        if _now - _last < _COOLDOWN:
            _wait = round(_COOLDOWN - (_now - _last), 1)
            return jsonify({
                'ok': False,
                'msg': f'Too fast — wait {_wait}s before reversing {symbol} again (bot is still syncing the last flip).'
            }), 429
        _lock = _REVERSE_LOCKS.setdefault(symbol, _th.Lock())
        if not _lock.acquire(blocking=False):
            return jsonify({
                'ok': False,
                'msg': f'Reverse for {symbol} already in flight — please wait.'
            }), 429
        _REVERSE_LAST_TS[symbol] = _now
        try:
            return _reverse_trade_impl(data, symbol, underlying, direction, quantity,
                                       is_option, option_type, strike, expiry, lots)
        finally:
            _lock.release()
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Reverse trade failed: {str(e)}'}), 500


def _reverse_trade_impl(data, symbol, underlying, direction, quantity,
                        is_option, option_type, strike, expiry, lots):
    """Actual reverse logic — called under per-symbol lock from reverse_trade()."""
    import re, random
    try:
        # Dashboard pause: block manual reverse when paused.
        if os.path.exists(TRADING_PAUSE_FLAG):
            return jsonify({'ok': False,
                            'msg': '⏸  Trading is paused — resume from Overview to place orders.'}), 409
        db = get_state_db()
        today = _today()

        # ── STEP 1: EXIT the current position ──
        positions, realized_pnl, paper_capital = db.load_active_trades(today)
        target_pos = None
        remaining = []
        for pos in positions:
            pos_sym = pos.get('symbol') or pos.get('option_symbol') or ''
            if pos_sym == symbol and pos.get('status', 'OPEN') == 'OPEN' and target_pos is None:
                target_pos = pos
            else:
                remaining.append(pos)

        # GUARD: don't create a ghost reverse if the original position is gone.
        # This happens when the dashboard row is stale (user clicks reverse twice
        # before auto-refresh fires, or after the bot auto-exited on SL/target).
        # Without this guard every stale click would silently spawn a new BUY,
        # piling up unwanted positions after a few clicks.
        if target_pos is None:
            return jsonify({
                'ok': False,
                'msg': f'Position {symbol} no longer open (already exited/flipped). Refresh and try again.'
            }), 409

        exit_pnl = 0
        exit_price = 0
        exit_msg = ''
        if target_pos:
            # Get current LTP for P&L calc
            live = db.load_live_pnl() or {}
            lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
            ltp_exit = 0
            if isinstance(lp, dict):
                ltp_exit = lp.get('ltp', 0)
            elif isinstance(lp, (int, float)):
                ltp_exit = float(lp)

            entry_price = target_pos.get('avg_price') or target_pos.get('entry_price') or 0
            exit_qty = abs(target_pos.get('quantity', 0))
            side = target_pos.get('side') or target_pos.get('direction', 'BUY')

            if ltp_exit > 0 and entry_price > 0:
                if side in ('BUY', 'LONG'):
                    exit_pnl = (ltp_exit - entry_price) * exit_qty
                else:
                    exit_pnl = (entry_price - ltp_exit) * exit_qty
            else:
                exit_pnl = target_pos.get('unrealized_pnl', 0)

            exit_price = ltp_exit if ltp_exit > 0 else entry_price

            # Write exit signal for bot (in-memory cleanup)
            exit_signal = {
                'symbol': symbol,
                'exit_price': round(exit_price, 2),
                'pnl': round(exit_pnl, 2),
                'exit_time': datetime.now().isoformat(),
                'exit_type': 'AUTOPILOT_REVERSE_EXIT',
                'direction': side,
                'quantity': exit_qty,
                'entry_price': round(entry_price, 2),
                'order_id': target_pos.get('order_id', ''),
                'trade_id': target_pos.get('trade_id', ''),
                'trade': target_pos,
                'live_exit_placed': False,
            }
            exit_pending = []
            if MANUAL_EXIT_FILE.exists():
                try:
                    exit_pending = json.loads(MANUAL_EXIT_FILE.read_text())
                except Exception:
                    exit_pending = []
            exit_pending.append(exit_signal)
            MANUAL_EXIT_FILE.write_text(json.dumps(exit_pending, indent=2, default=str))

            # Log EXIT in trade_ledger
            try:
                ledger = get_trade_ledger()
                _underlying = target_pos.get('underlying', '')
                if not _underlying:
                    m = re.match(r'(?:NFO:)?([A-Z]+)\d', symbol.replace('NFO:', ''))
                    _underlying = f"NSE:{m.group(1)}" if m else symbol
                _hold_mins = 0
                try:
                    from dateutil.parser import parse as _dp
                    _hold_mins = int((datetime.now() - _dp(target_pos.get('timestamp', ''))).total_seconds() / 60)
                except Exception:
                    pass
                _entry_price = float(target_pos.get('avg_price') or target_pos.get('entry_price') or 0)
                _exit_qty_log = abs(int(target_pos.get('quantity', 0) or 0))
                _pnl_pct = (round(exit_pnl, 2) / (_entry_price * _exit_qty_log) * 100) if _entry_price > 0 and _exit_qty_log > 0 else 0
                ledger.log_exit(
                    symbol=symbol,
                    underlying=_underlying,
                    direction=target_pos.get('direction') or target_pos.get('side') or 'BUY',
                    source=target_pos.get('setup_type', target_pos.get('strategy_type', '')),
                    sector=target_pos.get('sector', ''),
                    exit_type='AUTOPILOT_REVERSE_EXIT',
                    entry_price=_entry_price,
                    exit_price=round(exit_price, 2),
                    quantity=_exit_qty_log,
                    pnl=round(exit_pnl, 2),
                    pnl_pct=round(_pnl_pct, 2),
                    smart_score=float(target_pos.get('smart_score', 0) or 0),
                    final_score=float(target_pos.get('entry_score', 0) or 0),
                    dr_score=float(target_pos.get('dr_score', 0) or 0),
                    strategy_type=target_pos.get('strategy_type', ''),
                    exit_reason='Reverse trade exit from dashboard UI',
                    hold_minutes=_hold_mins,
                    entry_time=target_pos.get('timestamp', ''),
                    order_id=target_pos.get('order_id', ''),
                    trade_id=target_pos.get('trade_id', ''),
                )
            except Exception:
                pass

            realized_pnl += exit_pnl
            exit_msg = f'Exited {symbol} P&L={exit_pnl:+,.0f} | '

        # ── STEP 2: BUILD reverse position (PE↔CE) ──
        rev_symbol = symbol
        if is_option and option_type in ('CE', 'PE'):
            orig_type = 'PE' if option_type == 'CE' else 'CE'
            rev_symbol = re.sub(rf'{orig_type}$', option_type, symbol)

        # Fetch REAL market LTP via Kite API
        market_ltp = 0
        kite = _get_dashboard_kite()
        if not kite:
            # Still save the exit
            if target_pos:
                db.save_active_trades(remaining, realized_pnl, paper_capital)
            return jsonify({'ok': False, 'msg': f'{exit_msg}Kite API not available for reverse'}), 503

        nfo_symbol = f'NFO:{rev_symbol}' if not rev_symbol.startswith('NFO:') else rev_symbol
        try:
            ltp_data = kite.ltp([nfo_symbol])
            if nfo_symbol in ltp_data:
                market_ltp = ltp_data[nfo_symbol]['last_price']
        except Exception as e:
            print(f"\u26a0\ufe0f Reverse trade LTP fetch failed for {nfo_symbol}: {e}")

        if market_ltp <= 0:
            if target_pos:
                db.save_active_trades(remaining, realized_pnl, paper_capital)
            return jsonify({'ok': False, 'msg': f'{exit_msg}Could not fetch price for {rev_symbol}'}), 400

        # ── Capital-matched sizing (Apr 30) ──
        # Re-size the reverse leg so its premium-spend ≈ original leg's spend.
        # Without this, flipping CE↔PE at very different premiums silently
        # changes deployed capital (a ₹150 CE → ₹40 PE flip would deploy only
        # 27% of original notional; the reverse case deploys 3.75×).
        # Bounds: ≥1 lot, ≤ max(2 × original_lots, 10) — caps blow-ups when
        # the opposite leg is dirt cheap.
        try:
            _orig_entry = float(target_pos.get('avg_price') or target_pos.get('entry_price') or 0) if target_pos else 0
            _orig_qty = abs(int(target_pos.get('quantity', 0) or 0)) if target_pos else 0
            _orig_lots = max(1, int(target_pos.get('lots', lots) or lots)) if target_pos else lots
            _lot_size = max(1, int(_orig_qty / _orig_lots)) if (_orig_qty > 0 and _orig_lots > 0) else max(1, int(quantity / max(1, lots)))
            _orig_capital = _orig_entry * _orig_qty
            if _orig_capital > 0 and market_ltp > 0 and _lot_size > 0:
                _ideal_lots = round(_orig_capital / (market_ltp * _lot_size))
                _max_lots = max(_orig_lots * 2, 10)
                _new_lots = max(1, min(int(_ideal_lots), _max_lots))
                _new_qty = _lot_size * _new_lots
                if _new_lots != lots or _new_qty != quantity:
                    print(
                        f"💰 REVERSE capital-match {symbol}→{rev_symbol}: "
                        f"orig ₹{_orig_entry:.2f}×{_orig_qty}=₹{_orig_capital:,.0f} | "
                        f"new ₹{market_ltp:.2f}×{_new_qty} ({_new_lots} lots vs orig {_orig_lots}) "
                        f"=₹{market_ltp * _new_qty:,.0f}",
                        flush=True,
                    )
                lots = _new_lots
                quantity = _new_qty
        except Exception as _ce:
            print(f"⚠️ reverse capital-match sizing failed: {_ce}; falling back to {lots} lots", flush=True)

        stoploss_premium = round(market_ltp * 0.72, 2)
        target_premium = round(market_ltp * 1.60, 2)
        # Tick-noise floor: for cheap options (premium < ₹5), the 28% relative
        # SL collapses to 1-3 ticks (e.g. ₹0.05 entry → ₹0.036 SL). Enforce a
        # minimum SL distance of ₹0.50 OR 28%, whichever is wider, so normal
        # bid/ask noise can't trigger SL_HIT immediately. Cap target at 60%
        # symmetrically to keep R-multiple math sensible.
        _min_sl_dist = 0.50
        if (market_ltp - stoploss_premium) < _min_sl_dist:
            stoploss_premium = round(max(0.05, market_ltp - _min_sl_dist), 2)
        total_premium = round(market_ltp * quantity, 2)
        max_loss = round((market_ltp - stoploss_premium) * quantity, 2)

        paper_id = f'PAPER_REV_{random.randint(100000, 999999)}'
        trade_id = f'REV_{datetime.now().strftime("%H%M%S")}_{random.randint(1000,9999)}'

        new_pos = {
            'symbol': rev_symbol,
            'underlying': underlying or symbol,
            'quantity': quantity,
            'lots': lots,
            'avg_price': market_ltp,
            'side': 'BUY',
            'direction': direction,
            'option_type': option_type if is_option else '',
            'strike': strike if is_option else 0,
            'expiry': expiry if is_option else '',
            'stop_loss': stoploss_premium,
            'target': target_premium,
            'order_id': paper_id,
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'is_option': is_option,
            'total_premium': total_premium,
            'max_loss': max_loss,
            'setup_type': 'REVERSE_SHADOW',
            'strategy_type': 'NAKED_OPTION',
            'rationale': f'Reverse of {symbol} \u2192 {rev_symbol} @ \u20b9{market_ltp:.2f} (same strike/lots)',
            'entry_score': 0,
            'score_tier': 'manual',
            'smart_score': 0,
            'lot_multiplier': 1.0,
            'sector': '',
            'trigger_type': 'AUTOPILOT_REVERSE',
            'is_sniper': False,
            'delta': 0, 'theta': 0, 'iv': 0,
            # Manual setup flag: tells exit_manager to skip partial-profit /
            # trail-to-breakeven logic so the position rides on its original
            # hard SL until target / SL / manual exit. Without this, a brief
            # +25% spike that retraces will trigger SL_HIT at entry price.
            'manual_setup': True,
        }

        # ── Log ENTRY for the new reverse position to trade_ledger FIRST ──
        # Done before writing the signal file so new_pos carries the
        # _ledger_logged flag; bot's consumer uses that flag to skip
        # re-logging and avoid duplicate ENTRY rows in Trade History.
        try:
            _ledger = get_trade_ledger()
            _rev_underlying = new_pos.get('underlying') or symbol
            if not _rev_underlying.startswith('NSE:') and not _rev_underlying.startswith('NFO:'):
                _m = re.match(r'(?:NFO:)?([A-Z]+)\d', rev_symbol.replace('NFO:', ''))
                _rev_underlying = f"NSE:{_m.group(1)}" if _m else rev_symbol
            _ledger.log_entry(
                symbol=rev_symbol,
                underlying=_rev_underlying,
                direction=direction or 'BUY',
                source='AUTOPILOT_REVERSE',
                strategy_type='NAKED_OPTION',
                score_tier='manual',
                option_symbol=rev_symbol if is_option else '',
                strike=strike if is_option else 0,
                option_type=option_type if is_option else '',
                expiry=expiry if is_option else '',
                entry_price=round(market_ltp, 2),
                quantity=quantity,
                lots=lots,
                lot_multiplier=1.0,
                stop_loss=stoploss_premium,
                target=target_premium,
                total_premium=total_premium,
                rationale=new_pos.get('rationale', f'Reverse of {symbol}'),
                order_id=paper_id,
                trade_id=trade_id,
            )
            new_pos['_ledger_logged'] = True
        except Exception as _e:
            print(f"⚠️ reverse_trade log_entry failed: {_e}")

        # Write entry signal for bot (in-memory injection)
        signal_file = os.path.join(os.path.dirname(__file__), 'manual_entry_requests.json')
        pending = []
        if os.path.exists(signal_file):
            try:
                with open(signal_file, 'r') as sf:
                    pending = json.loads(sf.read().strip() or '[]')
            except Exception:
                pending = []
        pending.append(new_pos)
        with open(signal_file, 'w') as sf:
            json.dump(pending, sf)

        # Save to state_db: remaining (without exited) + new reverse position
        remaining.append(new_pos)
        db.save_active_trades(remaining, realized_pnl, paper_capital)

        # Seed live_pnl IMMEDIATELY so the dashboard renders the flipped

        # Seed live_pnl IMMEDIATELY so the dashboard renders the flipped
        # position with its correct entry LTP instead of the old symbol's
        # stale cached value. Without this the row shows no PnL or wrong
        # PnL until the bot's next ticker cycle (~2-5s).
        try:
            _live = db.load_live_pnl() or {}
            _snaps = []
            _total = 0.0
            _exited_sym = symbol if target_pos else None
            for _k, _v in _live.items():
                if _k.startswith('_') or not isinstance(_v, dict):
                    continue
                if _exited_sym and _k == _exited_sym:
                    continue  # drop old symbol from cache
                _snaps.append({
                    'symbol': _k,
                    'ltp': _v.get('ltp', 0),
                    'unrealized_pnl': _v.get('unrealized_pnl', 0),
                })
                _total += _v.get('unrealized_pnl', 0) or 0
            # Add seed entry for the NEW reverse position
            _snaps.append({'symbol': rev_symbol, 'ltp': market_ltp, 'unrealized_pnl': 0.0})
            db.save_live_pnl(_snaps, round(_total, 2))
        except Exception:
            pass

        # Freeze the displayed PnL at 0 for a short window so the row
        # opens at exactly ₹0 and only starts moving once the bot has
        # rebased avg_price to a fresh LTP. Key under both forms so the
        # endpoint matches whichever symbol shape the active_trades row
        # ends up persisted as.
        try:
            import time as _tt_rev
            _freeze_until = _tt_rev.time() + _REVERSE_FREEZE_SECS
            _bare_rev = rev_symbol.replace('NFO:', '')
            _REVERSE_FREEZE_UNTIL[_bare_rev] = _freeze_until
            _REVERSE_FREEZE_UNTIL[f'NFO:{_bare_rev}'] = _freeze_until
        except Exception:
            pass

        return jsonify({
            'ok': True,
            'msg': f'{exit_msg}Opened {rev_symbol} @ \u20b9{market_ltp:.2f} | SL \u20b9{stoploss_premium:.2f} | TGT \u20b9{target_premium:.2f}',
            'order_id': paper_id,
            'ltp': market_ltp,
            'stop_loss': stoploss_premium,
            'target': target_premium,
        })

    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Reverse trade failed: {str(e)}'}), 500


# ── Add-lot (+1): strengthen existing position with one more lot at LTP ─

_ADD_LOT_LOCKS = {}
_ADD_LOT_LAST_TS = {}

@app.route('/api/add_lot', methods=['POST'])
def add_lot():
    """Strengthen an existing option position by buying one more lot of the
    SAME strike / SAME option type at current LTP.

    Merges the new lot into the existing position using weighted-average
    entry price. Keeps a single row in active_trades (and one ENTRY record
    in trade_ledger) so the eventual EXIT pairs cleanly with the merged
    position and P&L reflects the full averaged cost basis.
    """
    try:
        import re as _re, random as _rand, threading as _th
        data = request.get_json(force=True, silent=True) or {}
        symbol = (data.get('symbol') or '').strip()
        if not symbol:
            return jsonify({'ok': False, 'msg': 'symbol required'}), 400

        # Per-symbol cooldown + in-flight lock (mirrors reverse_trade policy)
        _now = time.time()
        _COOLDOWN = 3.0
        _last = _ADD_LOT_LAST_TS.get(symbol, 0.0)
        if _now - _last < _COOLDOWN:
            _wait = round(_COOLDOWN - (_now - _last), 1)
            return jsonify({
                'ok': False,
                'msg': f'Too fast — wait {_wait}s before adding another lot to {symbol}.'
            }), 429
        _lock = _ADD_LOT_LOCKS.setdefault(symbol, _th.Lock())
        if not _lock.acquire(blocking=False):
            return jsonify({'ok': False, 'msg': f'Add-lot for {symbol} already in flight.'}), 429
        _ADD_LOT_LAST_TS[symbol] = _now
        try:
            db = get_state_db()
            today = _today()
            positions, realized_pnl, paper_capital = db.load_active_trades(today)

            target_pos = None
            remaining = []
            for pos in positions:
                pos_sym = pos.get('symbol') or pos.get('option_symbol') or ''
                if pos_sym == symbol and pos.get('status', 'OPEN') == 'OPEN' and target_pos is None:
                    target_pos = pos
                else:
                    remaining.append(pos)

            if target_pos is None:
                return jsonify({
                    'ok': False,
                    'msg': f'Position {symbol} not open — cannot add lot. Refresh and try again.'
                }), 409

            old_qty = int(target_pos.get('quantity', 0) or 0)
            old_lots = int(target_pos.get('lots', 1) or 1)
            old_avg = float(target_pos.get('avg_price') or target_pos.get('entry_price') or 0)
            if old_qty <= 0 or old_lots <= 0 or old_avg <= 0:
                return jsonify({'ok': False, 'msg': f'{symbol} has invalid size/price — cannot add lot.'}), 400

            lot_size = max(1, int(round(old_qty / old_lots)))

            # ── +1 sizing revamp (2026-05-05) ──
            # Cap at TWO adds per position. Lot count per add scales with
            # the *initial* size so big positions get meaningful adds.
            #   target_total_lots = max(2 × initial, initial + 2)
            #   total_to_add = target - initial
            #   1st add = ceil(total_to_add / 2)
            #   2nd add = total_to_add - 1st_add
            # Examples:
            #   initial 1 → 1, 1   (final 3)
            #   initial 2 → 1, 1   (final 4)
            #   initial 3 → 2, 1   (final 6)
            #   initial 5 → 3, 2   (final 10)
            prior_adds = list(target_pos.get('add_lot_history') or [])
            prior_added_lots = sum(int(a.get('lots_added', 0) or 0) for a in prior_adds)
            initial_lots = max(1, old_lots - prior_added_lots)
            target_total_lots = max(2 * initial_lots, initial_lots + 2)
            total_to_add = max(0, target_total_lots - initial_lots)
            import math as _math
            first_add_lots = max(1, _math.ceil(total_to_add / 2))
            second_add_lots = max(1, total_to_add - first_add_lots)
            adds_so_far = len(prior_adds)
            if adds_so_far >= 2:
                return jsonify({
                    'ok': False,
                    'msg': (f'{symbol} already has {adds_so_far} +1 adds (cap=2). '
                            f'Initial={initial_lots}, current={old_lots} lots. '
                            f'No further adds allowed for this position.')
                }), 409
            add_lots = first_add_lots if adds_so_far == 0 else second_add_lots
            add_qty = add_lots * lot_size

            # Fetch live LTP via Kite
            kite = _get_dashboard_kite()
            if not kite:
                return jsonify({'ok': False, 'msg': 'Kite API unavailable for add-lot'}), 503
            nfo_symbol = f'NFO:{symbol}' if not symbol.startswith('NFO:') else symbol
            market_ltp = 0
            try:
                ltp_data = kite.ltp([nfo_symbol])
                if nfo_symbol in ltp_data:
                    market_ltp = float(ltp_data[nfo_symbol].get('last_price', 0) or 0)
            except Exception as e:
                print(f"⚠️ Add-lot LTP fetch failed for {nfo_symbol}: {e}")

            if market_ltp <= 0:
                return jsonify({'ok': False, 'msg': f'Could not fetch LTP for {symbol}'}), 400

            new_qty = old_qty + add_qty
            new_lots = old_lots + add_lots
            new_avg = round(((old_avg * old_qty) + (market_ltp * add_qty)) / new_qty, 2)

            # Recompute SL/target on merged avg using the same 72%/160% bands
            new_sl = round(new_avg * 0.72, 2)
            new_tgt = round(new_avg * 1.60, 2)
            new_total_premium = round(new_avg * new_qty, 2)
            new_max_loss = round((new_avg - new_sl) * new_qty, 2)

            merged = dict(target_pos)
            merged.update({
                'quantity': new_qty,
                'lots': new_lots,
                'avg_price': new_avg,
                'entry_price': new_avg,
                'stop_loss': new_sl,
                'target': new_tgt,
                'total_premium': new_total_premium,
                'max_loss': new_max_loss,
                'status': 'OPEN',
            })
            # Track adds for audit
            adds = list(merged.get('add_lot_history') or [])
            adds.append({
                'time': datetime.now().isoformat(),
                'ltp': round(market_ltp, 2),
                'qty_added': add_qty,
                'lots_added': add_lots,
                'new_avg': new_avg,
            })
            merged['add_lot_history'] = adds

            # Signal the bot to merge its in-memory paper_position
            signal_file = os.path.join(os.path.dirname(__file__), 'manual_add_lot_requests.json')
            pending = []
            if os.path.exists(signal_file):
                try:
                    with open(signal_file, 'r') as sf:
                        pending = json.loads(sf.read().strip() or '[]')
                except Exception:
                    pending = []
            pending.append({
                'symbol': symbol,
                'add_qty': add_qty,
                'add_lots': add_lots,
                'add_price': round(market_ltp, 2),
                'new_qty': new_qty,
                'new_lots': new_lots,
                'new_avg': new_avg,
                'new_stop_loss': new_sl,
                'new_target': new_tgt,
                'new_total_premium': new_total_premium,
                'new_max_loss': new_max_loss,
                'time': datetime.now().isoformat(),
            })
            with open(signal_file, 'w') as sf:
                json.dump(pending, sf)

            # Persist merged position to DB immediately
            remaining.append(merged)
            db.save_active_trades(remaining, realized_pnl, paper_capital)

            return jsonify({
                'ok': True,
                'msg': (f'Added {add_lots} lot(s) to {symbol} @ ₹{market_ltp:.2f} | '
                        f'Qty {old_qty}→{new_qty} | Lots {old_lots}→{new_lots} | '
                        f'Avg ₹{old_avg:.2f}→₹{new_avg:.2f} | '
                        f'SL ₹{new_sl:.2f} | TGT ₹{new_tgt:.2f}'),
                'ltp': market_ltp,
                'new_qty': new_qty,
                'new_lots': new_lots,
                'new_avg': new_avg,
                'new_stop_loss': new_sl,
                'new_target': new_tgt,
            })
        finally:
            _lock.release()
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Add-lot failed: {str(e)}'}), 500


# ── Candles endpoint: lightweight chart data from Kite historical API ──

_CANDLES_CACHE = {}  # {(symbol, interval): (ts, payload)}
_CANDLES_TTL = 2     # 2s cache — matches 3s client poll, still dedupes bursts
_INSTRUMENT_TOKEN_CACHE = {}  # {'NSE:RELIANCE': 738561}
_INSTRUMENT_TOKEN_CACHE_TS = 0

@app.route('/api/candles', methods=['GET'])
def candles():
    """Return recent OHLC candles for a symbol. Used by dashboard chart modal.

    Query params:
      symbol   e.g. NSE:RELIANCE or MCX:CRUDEOIL (prefix required)
      interval 1m|3m|5m|15m|30m|60m|D  (default 5m)

    Response: {ok, candles: [{t,o,h,l,c,v}, ...]}
    """
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        iv_raw = (request.args.get('interval') or '5m').strip()
        if not symbol or ':' not in symbol:
            return jsonify({'ok': False, 'msg': 'symbol=NSE:XYZ required'}), 400

        iv_map = {
            '1': 'minute', '1m': 'minute',
            '3': '3minute', '3m': '3minute',
            '5': '5minute', '5m': '5minute',
            '15': '15minute', '15m': '15minute',
            '30': '30minute', '30m': '30minute',
            '60': '60minute', '60m': '60minute', '1h': '60minute',
            'd': 'day', 'D': 'day', '1d': 'day',
        }
        kite_iv = iv_map.get(iv_raw.lower()) or iv_map.get(iv_raw) or '5minute'

        # Cache check (30s TTL — fast successive opens return instantly)
        import time as _t
        now = _t.time()
        ck = (symbol, kite_iv)
        hit = _CANDLES_CACHE.get(ck)
        if hit and (now - hit[0]) < _CANDLES_TTL:
            return jsonify(hit[1])

        kite = _get_dashboard_kite()
        if not kite:
            return jsonify({'ok': False, 'msg': 'Kite not available'}), 503

        # Resolve instrument_token (cached 1h)
        global _INSTRUMENT_TOKEN_CACHE_TS
        tok = _INSTRUMENT_TOKEN_CACHE.get(symbol)
        if not tok:
            # Cheapest path: kite.ltp() returns instrument_token in payload
            try:
                lp = kite.ltp([symbol])
                if symbol in lp:
                    tok = lp[symbol].get('instrument_token')
                    if tok:
                        _INSTRUMENT_TOKEN_CACHE[symbol] = tok
            except Exception as e:
                return jsonify({'ok': False, 'msg': f'ltp lookup failed: {e}'}), 400

        if not tok:
            return jsonify({'ok': False, 'msg': f'No instrument_token for {symbol}'}), 404

        # Window: last N bars — keep small for speed
        from datetime import datetime as _dt, timedelta as _td
        now_dt = _dt.now()
        if kite_iv == 'day':
            frm = now_dt - _td(days=120)
        elif kite_iv == '60minute':
            frm = now_dt - _td(days=15)
        elif kite_iv == '30minute':
            frm = now_dt - _td(days=8)
        elif kite_iv == '15minute':
            frm = now_dt - _td(days=5)
        elif kite_iv == '5minute':
            frm = now_dt - _td(days=2)
        elif kite_iv == '3minute':
            frm = now_dt - _td(days=2)
        else:  # 1minute — today's session only (9:15 IST)
            frm = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            if frm > now_dt:
                # Pre-open — show previous trading day session
                frm = frm - _td(days=1)

        try:
            raw = kite.historical_data(tok, frm, now_dt, kite_iv)
        except Exception as e:
            return jsonify({'ok': False, 'msg': f'historical_data failed: {e}'}), 502

        out = []
        for r in raw[-400:]:  # cap 400 bars for payload size
            _ts = r.get('date')
            # Lightweight-charts expects UNIX seconds (UTC)
            try:
                _unix = int(_ts.timestamp())
            except Exception:
                continue
            out.append({
                't': _unix,
                'o': r.get('open', 0),
                'h': r.get('high', 0),
                'l': r.get('low', 0),
                'c': r.get('close', 0),
                'v': r.get('volume', 0),
            })

        payload = {'ok': True, 'symbol': symbol, 'interval': kite_iv, 'candles': out}
        _CANDLES_CACHE[ck] = (now, payload)
        return jsonify(payload)

    except Exception as e:
        return jsonify({'ok': False, 'msg': f'candles error: {e}'}), 500


# ── Live tick endpoint: <1s price feed for chart & position-less streaming ──
_LTP_CACHE = {}  # {symbol: (ts, ltp)}
_LTP_TTL = 0.8   # 800ms — many clients share same ping

# Per-symbol "freeze PnL at 0" window after a manual Reverse click.
# Without this, the live-PnL endpoint instantly recomputes
# (fresh_kite_ltp - entry_price) * qty using a Kite LTP fetched
# milliseconds after click — the market has already ticked, so PnL
# is non-zero on the very first poll. Freezing for ~6s covers the
# bot's rebase cycle (1–3s) plus a small safety margin, after which
# avg_price has been re-anchored to the just-observed LTP and PnL
# accumulates from real post-entry price moves.
_REVERSE_FREEZE_UNTIL = {}  # {symbol_no_prefix: epoch_ts_until}
_REVERSE_FREEZE_SECS = 6.0

def _reverse_freeze_active(sym: str) -> bool:
    if not sym:
        return False
    import time as _tt
    _now = _tt.time()
    bare = sym.replace('NFO:', '')
    until = _REVERSE_FREEZE_UNTIL.get(bare) or _REVERSE_FREEZE_UNTIL.get(sym)
    if until and until > _now:
        return True
    if until and until <= _now:
        # Lazy cleanup
        _REVERSE_FREEZE_UNTIL.pop(bare, None)
        _REVERSE_FREEZE_UNTIL.pop(sym, None)
    return False

@app.route('/api/ltp', methods=['GET'])
def live_ltp():
    """Ultra-fast LTP lookup for charting — <100ms round trip.

    Query:  ?symbol=NSE:RELIANCE  (or comma-separated for batch)
    Return: {ok, ts, ltp: {'NSE:RELIANCE': 1234.5, ...}}

    Uses kite.ltp() which is ~30-80ms per call. Shared 800ms cache across
    all dashboard clients makes 1s-poll-per-chart effectively free.
    """
    try:
        raw = (request.args.get('symbol') or '').strip().upper()
        if not raw:
            return jsonify({'ok': False, 'msg': 'symbol required'}), 400
        syms = [s.strip() for s in raw.split(',') if s.strip() and ':' in s]
        if not syms:
            return jsonify({'ok': False, 'msg': 'need NSE:XYZ format'}), 400

        import time as _tl
        now = _tl.time()
        out = {}
        fetch = []
        for s in syms:
            hit = _LTP_CACHE.get(s)
            if hit and (now - hit[0]) < _LTP_TTL:
                out[s] = hit[1]
            else:
                fetch.append(s)

        if fetch:
            kite = _get_dashboard_kite()
            if not kite:
                return jsonify({'ok': False, 'msg': 'Kite unavailable'}), 503
            try:
                data = kite.ltp(fetch)
                for s in fetch:
                    px = (data.get(s) or {}).get('last_price')
                    if px is not None:
                        _LTP_CACHE[s] = (now, float(px))
                        out[s] = float(px)
            except Exception as e:
                return jsonify({'ok': False, 'msg': f'kite.ltp failed: {e}'}), 502

        return jsonify({'ok': True, 'ts': int(now * 1000), 'ltp': out})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# â”€â”€ Manual position exit (dashboard â†’ signal file â†’ bot) â”€â”€â”€â”€â”€

MANUAL_EXIT_FILE = Path(__file__).parent / 'manual_exit_requests.json'

@app.route('/api/exit_position', methods=['POST'])
def exit_position():
    """Exit a position at market price via dashboard.
    
    Flow:
    1. Find position in state_db
    2. Get current LTP from live_pnl bridge
    3. Calculate realized P&L
    4. Write signal file for bot to process (in-memory cleanup + LIVE order)
    5. Remove from state_db immediately (so UI refreshes)
    6. Log EXIT in trade_ledger
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get('symbol', '').strip()
        if not symbol:
            return jsonify({'ok': False, 'msg': 'Missing symbol'}), 400

        db = get_state_db()
        today = _today()
        positions, realized_pnl, paper_capital = db.load_active_trades(today)

        # Find the matching position
        target = None
        remaining = []
        for pos in positions:
            pos_sym = pos.get('symbol') or pos.get('option_symbol') or ''
            if pos_sym == symbol and pos.get('status', 'OPEN') == 'OPEN':
                target = pos
            else:
                remaining.append(pos)

        if not target:
            return jsonify({'ok': False, 'msg': f'Position {symbol} not found or already closed'}), 404

        # Get current LTP from live_pnl bridge
        live = db.load_live_pnl() or {}
        lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
        ltp = 0
        if isinstance(lp, dict):
            ltp = lp.get('ltp', 0)
        elif isinstance(lp, (int, float)):
            ltp = float(lp)

        # Calculate P&L
        entry_price = target.get('avg_price') or target.get('entry_price') or 0
        qty = abs(target.get('quantity', 0))
        # Use 'side' (actual transaction side: BUY/SELL) NOT 'direction' (market view: BUY=bullish, SELL=bearish)
        # For PE options: direction=SELL but side=BUY â†’ must use side for correct P&L
        direction = target.get('side') or target.get('direction', 'BUY')

        # For debit spreads, use net premium as entry
        if target.get('is_debit_spread') or target.get('is_credit_spread'):
            entry_price = target.get('net_premium', entry_price)
            # For spreads, unrealized_pnl from live_pnl is the most accurate
            pnl = 0
            lp_spread = live.get(symbol)
            if isinstance(lp_spread, dict) and lp_spread.get('unrealized_pnl') is not None:
                pnl = lp_spread['unrealized_pnl']
            elif target.get('unrealized_pnl'):
                pnl = target['unrealized_pnl']
        elif ltp > 0 and entry_price > 0:
            if direction in ('BUY', 'LONG'):
                pnl = (ltp - entry_price) * qty
            else:
                pnl = (entry_price - ltp) * qty
        else:
            pnl = target.get('unrealized_pnl', 0)

        exit_price = ltp if ltp > 0 else entry_price

        # 1ï¸âƒ£  LIVE MODE: Place real exit order immediately from dashboard
        live_exit_placed = False
        live_exit_msg = ''
        if not PAPER_MODE:
            kite = _get_dashboard_kite()
            if kite:
                try:
                    # Cancel pending SL-M order first
                    sl_order_id = target.get('sl_order_id')
                    if sl_order_id and not str(sl_order_id).startswith('PAPER_'):
                        try:
                            kite.cancel_order(variety='regular', order_id=sl_order_id)
                        except Exception:
                            pass  # May have already triggered

                    # Determine legs for spreads/condors
                    legs = []
                    is_spread = target.get('is_credit_spread') or target.get('is_debit_spread') or target.get('is_iron_condor')
                    if target.get('is_iron_condor'):
                        for pfx, act in [('sold_ce','BUY'),('sold_pe','BUY'),('hedge_ce','SELL'),('hedge_pe','SELL')]:
                            s = target.get(f'{pfx}_symbol')
                            if s: legs.append((s, act))
                    elif target.get('is_credit_spread'):
                        if target.get('sold_symbol'): legs.append((target['sold_symbol'], 'BUY'))
                        if target.get('hedge_symbol'): legs.append((target['hedge_symbol'], 'SELL'))
                    elif target.get('is_debit_spread'):
                        syms = (target.get('symbol') or '').split('|')
                        if len(syms) == 2:
                            legs.append((syms[0], 'SELL'))
                            legs.append((syms[1], 'BUY'))
                    elif '|' in symbol:
                        syms = symbol.split('|')
                        if len(syms) == 2:
                            legs.append((syms[0], 'SELL' if direction in ('BUY','LONG') else 'BUY'))
                            legs.append((syms[1], 'BUY' if direction in ('BUY','LONG') else 'SELL'))
                    else:
                        exit_side = 'SELL' if direction in ('BUY', 'LONG') else 'BUY'
                        legs.append((symbol, exit_side))

                    order_ids = []
                    # Place exit order for each leg
                    for leg_sym, leg_action in legs:
                        exch, tsym = leg_sym.split(':')
                        tx = kite.TRANSACTION_TYPE_SELL if leg_action == 'SELL' else kite.TRANSACTION_TYPE_BUY
                        order_id = kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=exch,
                            tradingsymbol=tsym,
                            transaction_type=tx,
                            quantity=qty,
                            product=kite.PRODUCT_MIS,
                            order_type=kite.ORDER_TYPE_MARKET,
                            validity=kite.VALIDITY_DAY,
                            tag='TITAN_MANUAL'
                        )
                        order_ids.append(order_id)
                        live_exit_msg += f' order:{order_id}'

                    if len(order_ids) == 1:
                        fill_price = _get_completed_order_fill_price(kite, order_ids[0])
                        if fill_price > 0:
                            exit_price = fill_price
                            if direction in ('BUY', 'LONG'):
                                pnl = (exit_price - entry_price) * qty
                            else:
                                pnl = (entry_price - exit_price) * qty

                    live_exit_placed = True
                except Exception as e:
                    live_exit_msg = f'âš ï¸ LIVE exit order failed: {e}'
                    print(f"   ðŸš¨ Dashboard LIVE exit failed for {symbol}: {e}")

        # 2ï¸âƒ£  Write signal file for bot (in-memory cleanup + sync)
        signal = {
            'symbol': symbol,
            'exit_price': round(exit_price, 2),
            'pnl': round(pnl, 2),
            'exit_time': datetime.now().isoformat(),
            'exit_type': 'MANUAL_DASHBOARD',
            'direction': direction,
            'quantity': qty,
            'entry_price': round(entry_price, 2),
            'trade': target,
            'live_exit_placed': live_exit_placed,  # bot skips placing order if True
        }
        pending = []
        if MANUAL_EXIT_FILE.exists():
            try:
                pending = json.loads(MANUAL_EXIT_FILE.read_text())
            except Exception:
                pending = []
        pending.append(signal)
        MANUAL_EXIT_FILE.write_text(json.dumps(pending, indent=2, default=str))

        # 2ï¸âƒ£  Remove from state_db immediately (UI refreshes)
        new_realized = realized_pnl + pnl
        db.save_active_trades(remaining, new_realized, paper_capital)

        # 3ï¸âƒ£  Log EXIT in trade_ledger
        try:
            ledger = get_trade_ledger()
            import re
            _underlying = target.get('underlying', '')
            if not _underlying:
                m = re.match(r'(?:NFO:)?([A-Z]+)\d', symbol.replace('NFO:', ''))
                _underlying = f"NSE:{m.group(1)}" if m else symbol
            _hold_mins = 0
            try:
                _et = target.get('timestamp', '')
                if _et:
                    _hold_mins = int((datetime.now() - datetime.fromisoformat(_et)).total_seconds() / 60)
            except Exception:
                pass
            _pnl_pct = (pnl / (entry_price * qty) * 100) if entry_price > 0 and qty > 0 else 0
            ledger.log_exit(
                symbol=symbol,
                underlying=_underlying,
                direction=direction,
                source=target.get('setup_type', target.get('strategy_type', '')),
                sector=target.get('sector', ''),
                exit_type='MANUAL_DASHBOARD',
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl,
                pnl_pct=_pnl_pct,
                smart_score=target.get('smart_score', 0),
                final_score=target.get('entry_score', 0),
                dr_score=target.get('dr_score', 0),
                exit_reason='Manual exit from dashboard UI',
                hold_minutes=_hold_mins,
                entry_time=target.get('timestamp', ''),
            )
        except Exception as e:
            print(f"âš ï¸ Trade ledger log failed for manual exit: {e}")

        _mode_label = 'LIVE' if not PAPER_MODE else 'PAPER'
        _exit_msg = f'[{_mode_label}] Exited {symbol} @ â‚¹{exit_price:.2f} | P&L: â‚¹{pnl:+,.2f}'
        if live_exit_placed:
            _exit_msg += ' | Broker order placed âœ…'
        elif not PAPER_MODE:
            _exit_msg += f' | {live_exit_msg}'

        return jsonify({
            'ok': True,
            'msg': _exit_msg,
            'symbol': symbol,
            'exit_price': round(exit_price, 2),
            'pnl': round(pnl, 2),
            'live_exit_placed': live_exit_placed,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'msg': f'Exit failed: {str(e)}'}), 500


@app.route('/api/exit_all', methods=['POST'])
def exit_all_positions():
    """Exit ALL open positions at market price â€” emergency kill switch.

    Iterates every OPEN position, places market exit orders (live mode),
    writes manual-exit signals for the bot, and clears active_trades.
    Returns a summary of successes and failures.
    """
    try:
        db = get_state_db()
        today = _today()
        positions, realized_pnl, paper_capital = db.load_active_trades(today)

        open_positions = [p for p in positions if p.get('status', 'OPEN') == 'OPEN']
        if not open_positions:
            return jsonify({'ok': True, 'msg': 'No open positions to exit', 'results': []})

        live = db.load_live_pnl() or {}
        kite = None
        if not PAPER_MODE:
            kite = _get_dashboard_kite()

        results = []
        total_pnl = 0
        pending_signals = []
        if MANUAL_EXIT_FILE.exists():
            try:
                pending_signals = json.loads(MANUAL_EXIT_FILE.read_text())
            except Exception:
                pending_signals = []

        for pos in open_positions:
            symbol = pos.get('symbol') or pos.get('option_symbol') or ''
            entry_price = pos.get('avg_price') or pos.get('entry_price') or 0
            qty = abs(pos.get('quantity', 0))
            direction = pos.get('side') or pos.get('direction', 'BUY')

            # LTP from live_pnl
            lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
            ltp = 0
            if isinstance(lp, dict):
                ltp = lp.get('ltp', 0)
            elif isinstance(lp, (int, float)):
                ltp = float(lp)

            # P&L calculation â€” same logic as single exit
            if pos.get('is_debit_spread') or pos.get('is_credit_spread'):
                entry_price = pos.get('net_premium', entry_price)
                pnl = 0
                lp_spread = live.get(symbol)
                if isinstance(lp_spread, dict) and lp_spread.get('unrealized_pnl') is not None:
                    pnl = lp_spread['unrealized_pnl']
                elif pos.get('unrealized_pnl'):
                    pnl = pos['unrealized_pnl']
            elif ltp > 0 and entry_price > 0:
                if direction in ('BUY', 'LONG'):
                    pnl = (ltp - entry_price) * qty
                else:
                    pnl = (entry_price - ltp) * qty
            else:
                pnl = pos.get('unrealized_pnl', 0)

            exit_price = ltp if ltp > 0 else entry_price

            # Place LIVE exit orders
            live_exit_placed = False
            live_exit_msg = ''
            if not PAPER_MODE and kite:
                try:
                    sl_order_id = pos.get('sl_order_id')
                    if sl_order_id and not str(sl_order_id).startswith('PAPER_'):
                        try:
                            kite.cancel_order(variety='regular', order_id=sl_order_id)
                        except Exception:
                            pass

                    legs = []
                    if pos.get('is_iron_condor'):
                        for pfx, act in [('sold_ce','BUY'),('sold_pe','BUY'),('hedge_ce','SELL'),('hedge_pe','SELL')]:
                            s = pos.get(f'{pfx}_symbol')
                            if s: legs.append((s, act))
                    elif pos.get('is_credit_spread'):
                        if pos.get('sold_symbol'): legs.append((pos['sold_symbol'], 'BUY'))
                        if pos.get('hedge_symbol'): legs.append((pos['hedge_symbol'], 'SELL'))
                    elif pos.get('is_debit_spread'):
                        syms = (pos.get('symbol') or '').split('|')
                        if len(syms) == 2:
                            legs.append((syms[0], 'SELL'))
                            legs.append((syms[1], 'BUY'))
                    elif '|' in symbol:
                        syms = symbol.split('|')
                        if len(syms) == 2:
                            legs.append((syms[0], 'SELL' if direction in ('BUY','LONG') else 'BUY'))
                            legs.append((syms[1], 'BUY' if direction in ('BUY','LONG') else 'SELL'))
                    else:
                        exit_side = 'SELL' if direction in ('BUY', 'LONG') else 'BUY'
                        legs.append((symbol, exit_side))

                    order_ids = []
                    for leg_sym, leg_action in legs:
                        exch, tsym = leg_sym.split(':')
                        tx = kite.TRANSACTION_TYPE_SELL if leg_action == 'SELL' else kite.TRANSACTION_TYPE_BUY
                        order_id = kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=exch,
                            tradingsymbol=tsym,
                            transaction_type=tx,
                            quantity=qty,
                            product=kite.PRODUCT_MIS,
                            order_type=kite.ORDER_TYPE_MARKET,
                            validity=kite.VALIDITY_DAY,
                            tag='TITAN_EXITALL'
                        )
                        order_ids.append(order_id)

                    if len(order_ids) == 1:
                        fill_price = _get_completed_order_fill_price(kite, order_ids[0])
                        if fill_price > 0:
                            exit_price = fill_price
                            if direction in ('BUY', 'LONG'):
                                pnl = (exit_price - entry_price) * qty
                            else:
                                pnl = (entry_price - exit_price) * qty
                    live_exit_placed = True
                except Exception as e:
                    live_exit_msg = str(e)

            # Write signal for bot
            signal = {
                'symbol': symbol,
                'exit_price': round(exit_price, 2),
                'pnl': round(pnl, 2),
                'exit_time': datetime.now().isoformat(),
                'exit_type': 'MANUAL_EXIT_ALL',
                'direction': direction,
                'quantity': qty,
                'entry_price': round(entry_price, 2),
                'trade': pos,
                'live_exit_placed': live_exit_placed,
            }
            pending_signals.append(signal)

            # Log in trade ledger
            try:
                import re as _re
                ledger = get_trade_ledger()
                _underlying = pos.get('underlying', '')
                if not _underlying:
                    m = _re.match(r'(?:NFO:)?([A-Z]+)\d', symbol.replace('NFO:', ''))
                    _underlying = f"NSE:{m.group(1)}" if m else symbol
                _hold_mins = 0
                try:
                    _et = pos.get('timestamp', '')
                    if _et:
                        _hold_mins = int((datetime.now() - datetime.fromisoformat(_et)).total_seconds() / 60)
                except Exception:
                    pass
                _pnl_pct = (pnl / (entry_price * qty) * 100) if entry_price > 0 and qty > 0 else 0
                ledger.log_exit(
                    symbol=symbol, underlying=_underlying, direction=direction,
                    source=pos.get('setup_type', pos.get('strategy_type', '')),
                    sector=pos.get('sector', ''),
                    exit_type='MANUAL_EXIT_ALL',
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=qty, pnl=pnl, pnl_pct=_pnl_pct,
                    smart_score=pos.get('smart_score', 0),
                    final_score=pos.get('entry_score', 0),
                    dr_score=pos.get('dr_score', 0),
                    exit_reason='Manual EXIT ALL from dashboard UI',
                    hold_minutes=_hold_mins,
                    entry_time=pos.get('timestamp', ''),
                )
            except Exception as e:
                print(f"âš ï¸ Trade ledger log failed (exit_all) for {symbol}: {e}")

            total_pnl += pnl
            status = 'âœ…' if live_exit_placed or PAPER_MODE else f'âš ï¸ {live_exit_msg}'
            results.append({'symbol': symbol, 'pnl': round(pnl, 2), 'status': status})

        # Write all signals at once
        MANUAL_EXIT_FILE.write_text(json.dumps(pending_signals, indent=2, default=str))

        # Clear all active trades from state_db
        new_realized = realized_pnl + total_pnl
        db.save_active_trades([], new_realized, paper_capital)

        _mode = 'LIVE' if not PAPER_MODE else 'PAPER'
        msg = f'[{_mode}] Exited {len(results)} positions | Net P&L: â‚¹{total_pnl:+,.2f}'
        print(f"ðŸš¨ EXIT ALL: {msg}")

        return jsonify({
            'ok': True,
            'msg': msg,
            'results': results,
            'total_pnl': round(total_pnl, 2),
            'count': len(results),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'msg': f'Exit All failed: {str(e)}'}), 500


# â”€â”€ Helpers: enrich positions with exit-state LTP & unrealized P&L â”€â”€

def _enrich_positions(positions: list, db) -> list:
    """Merge live P&L data into each position dict so the dashboard
    can display current LTP, unrealized P&L, underlying symbol, etc."""
    import re

    # Primary source: live_pnl table (updated every scan cycle by the bot)
    live = db.load_live_pnl() or {}
    # Exit-manager state (trailing SL, breakeven, current SL). Merged in-line
    # so the UI doesn't depend on a separate /api/exits round-trip succeeding.
    try:
        exit_states = db.load_exit_states() or {}
    except Exception:
        exit_states = {}

    for pos in positions:
        sym = pos.get('symbol') or pos.get('option_symbol') or ''
        # Tag exchange â€” all state_db positions are NSE/NFO
        if not pos.get('exchange'):
            pos['exchange'] = 'NSE'
        # Derive underlying from NFO symbol  e.g. NFO:DLF26MAR590PE -> DLF
        if not pos.get('underlying'):
            m = re.match(r'(?:NFO:)?([A-Z]+)\d', sym.replace('NFO:', ''))
            pos['underlying'] = f"NSE:{m.group(1)}" if m else ''

        # Merge live LTP & unrealized P&L from bot's scan cycle
        lp = live.get(sym) or live.get(sym.replace('NFO:', ''))
        if isinstance(lp, dict):
            pos['ltp'] = lp.get('ltp', 0)
            pos['unrealized_pnl'] = lp.get('unrealized_pnl', 0)
            pos['ltp_updated'] = lp.get('last_updated', '')
        elif isinstance(lp, (int, float)):
            pos['ltp'] = float(lp)

        # Merge exit-manager trailing state inline. UI prefers these fields
        # over a separate /api/exits lookup so trailing pill / row tint render
        # even if the second API call fails or is cached stale.
        es = exit_states.get(sym) or exit_states.get(sym.replace('NFO:', ''))
        if isinstance(es, dict):
            cs = es.get('current_sl')
            if cs is not None:
                pos['current_sl'] = cs
            pos['trailing_active'] = bool(es.get('trailing_active', False))
            pos['breakeven_applied'] = bool(es.get('breakeven_applied', False))
            hp = es.get('highest_price')
            if hp is not None:
                pos['highest_price'] = hp
    return positions


# â”€â”€ News Targets (Early Bird Mode D) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/news_targets')
def get_news_targets():
    """Return pre-market news scan results for dashboard display."""
    try:
        scan_file = os.path.join(os.path.dirname(__file__), 'news_scan_results.json')
        if not os.path.exists(scan_file):
            return jsonify({'targets': [], 'scan_time': None})
        with open(scan_file, 'r') as f:
            data = json.load(f)
        # Check if scan is from today
        scan_time = data.get('scan_time', '')
        today = _today()
        if scan_time and not scan_time.startswith(today):
            return jsonify({'targets': [], 'scan_time': None, 'stale': True})
        # Check which targets have been bought (have active positions)
        db = get_state_db()
        positions, _, _ = db.load_active_trades(today)
        open_syms = set()
        for p in positions:
            sym = (p.get('symbol') or '').replace('NSE:', '')
            underlying = (p.get('underlying') or '').replace('NSE:', '')
            open_syms.add(sym)
            open_syms.add(underlying)
        targets = data.get('targets', [])
        for t in targets:
            t['bought'] = t.get('symbol', '') in open_syms
        # Filter out acted/bought news â€” once traded, don't show in dashboard
        targets = [t for t in targets if not t.get('bought')]
        return jsonify({
            'targets': targets,
            'scan_time': scan_time,
            'lookback_hours': data.get('lookback_hours', 18),
            'total_scanned': data.get('total_scanned', len(targets)),
            'tradeable_count': data.get('tradeable_count', 0),
        })
    except Exception as e:
        return jsonify({'targets': [], 'scan_time': None, 'error': str(e)})


# â”€â”€ Manual news trade (dashboard button â†’ real Kite order) â”€â”€â”€

@app.route('/api/place_news_trade', methods=['POST'])
def place_news_trade():
    """Place a REAL option order from a news target via Kite API.
    Finds ITM-1 strike from Kite instruments, places market order."""
    try:
        import random, time as _time
        from options_trader import FNO_LOT_SIZES

        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get('symbol', '').strip()
        sentiment = data.get('sentiment', '').strip().upper()
        confidence = int(data.get('confidence', 0))

        if not symbol or sentiment not in ('BULLISH', 'BEARISH'):
            return jsonify({'ok': False, 'msg': 'Missing symbol or invalid sentiment'}), 400

        direction = 'BUY' if sentiment == 'BULLISH' else 'SELL'
        option_type = 'CE' if sentiment == 'BULLISH' else 'PE'

        kite = _get_dashboard_kite()
        if not kite:
            return jsonify({'ok': False, 'msg': 'Kite API not available'}), 503

        # â”€â”€ Get underlying LTP to find ATM strike â”€â”€
        nse_sym = f'NSE:{symbol}'
        try:
            ltp_data = kite.ltp([nse_sym])
            underlying_price = ltp_data.get(nse_sym, {}).get('last_price', 0)
        except Exception as e:
            return jsonify({'ok': False, 'msg': f'Cannot fetch price for {symbol}: {e}'}), 400

        if underlying_price <= 0:
            return jsonify({'ok': False, 'msg': f'No market price for {symbol}'}), 400

        # â”€â”€ Find ATM strike gap â”€â”€
        if symbol in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
            strike_gap = 50
        elif underlying_price > 5000:
            strike_gap = 100
        elif underlying_price > 2500:
            strike_gap = 50
        elif underlying_price > 1000:
            strike_gap = 20
        elif underlying_price > 500:
            strike_gap = 10
        else:
            strike_gap = 5

        atm_strike = round(underlying_price / strike_gap) * strike_gap

        # ITM-1: BULLISH â†’ CE 1 strike below ATM, BEARISH â†’ PE 1 strike above ATM
        if sentiment == 'BULLISH':
            target_strike = atm_strike - strike_gap
        else:
            target_strike = atm_strike + strike_gap

        # â”€â”€ Find correct trading symbol from Kite instruments â”€â”€
        from datetime import date
        today_date = date.today()
        nfo_instruments = kite.instruments('NFO')

        # Filter to this symbol's options with matching type
        candidates = []
        for inst in nfo_instruments:
            if (inst.get('name') == symbol and
                inst.get('instrument_type') == option_type and
                inst.get('expiry') is not None):
                exp = inst['expiry']
                if isinstance(exp, str):
                    exp = datetime.strptime(exp, '%Y-%m-%d').date()
                if exp >= today_date:
                    candidates.append({
                        'tradingsymbol': inst['tradingsymbol'],
                        'strike': float(inst.get('strike', 0)),
                        'expiry': exp,
                        'lot_size': int(inst.get('lot_size', 1)),
                    })

        if not candidates:
            return jsonify({'ok': False, 'msg': f'No NFO instruments found for {symbol} {option_type}'}), 400

        # Find nearest expiry
        nearest_expiry = min(set(c['expiry'] for c in candidates))
        # Filter to nearest expiry
        candidates = [c for c in candidates if c['expiry'] == nearest_expiry]
        # Find the contract closest to our target strike
        candidates.sort(key=lambda c: abs(c['strike'] - target_strike))
        chosen = candidates[0]

        tradingsymbol = chosen['tradingsymbol']
        actual_strike = chosen['strike']
        lot_size = chosen['lot_size']
        expiry_str = str(nearest_expiry)

        # Override lot_size from our known map if available
        lot_size = FNO_LOT_SIZES.get(symbol, lot_size)

        # â”€â”€ Get option LTP â”€â”€
        nfo_key = f'NFO:{tradingsymbol}'
        try:
            ld = kite.ltp([nfo_key])
            opt_ltp = ld.get(nfo_key, {}).get('last_price', 0)
        except Exception:
            opt_ltp = 0

        if opt_ltp <= 0:
            return jsonify({'ok': False, 'msg': f'No market price for {tradingsymbol}'}), 400

        # â”€â”€ Sizing: minimum â‚¹50k notional budget per manual news trade â”€â”€
        # If single-lot premium < 50k, scale up lots; cap at 10 lots to avoid
        # runaway sizing on ultra-cheap options. Caller may override via
        # data['lots'] if explicitly provided.
        _MIN_BUDGET = 50000.0
        _MAX_LOTS = 10
        _premium_per_lot = float(opt_ltp) * float(lot_size)
        try:
            _override_lots = int(data.get('lots') or 0)
        except Exception:
            _override_lots = 0
        if _override_lots > 0:
            lots = min(_override_lots, _MAX_LOTS)
        elif _premium_per_lot <= 0:
            lots = 1
        else:
            import math as _math
            lots = int(_math.ceil(_MIN_BUDGET / _premium_per_lot))
            lots = max(1, min(lots, _MAX_LOTS))
        quantity = lot_size * lots
        print(f"ðŸ“° NEWS MANUAL SIZING: {symbol} premium/lot=â‚¹{_premium_per_lot:.0f} â†’ {lots} lot(s) = â‚¹{_premium_per_lot*lots:.0f} notional")

        # â”€â”€ Place REAL Kite order (or paper if PAPER_MODE) â”€â”€
        fill_price = opt_ltp
        order_tag = 'TITAN_NEWS'

        if PAPER_MODE:
            order_id = f'OPTION_PAPER_{random.randint(100000, 999999)}'
            is_live = False
            print(f"ðŸ“° NEWS MANUAL (PAPER): BUY {tradingsymbol} x{quantity} @ â‚¹{opt_ltp:.2f}")
        else:
            # REAL LIVE ORDER
            try:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange='NFO',
                    tradingsymbol=tradingsymbol,
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=quantity,
                    product=kite.PRODUCT_MIS,
                    order_type=kite.ORDER_TYPE_MARKET,
                    tag=order_tag,
                )
                order_id = str(order_id)
                is_live = True
                print(f"ðŸ“° NEWS MANUAL (LIVE): BUY {tradingsymbol} x{quantity} order_id={order_id}")

                # Wait for fill and get actual price
                _time.sleep(0.5)
                try:
                    order_history = kite.order_history(order_id)
                    for oh in reversed(order_history):
                        if oh.get('status') == 'COMPLETE' and oh.get('average_price', 0) > 0:
                            fill_price = oh['average_price']
                            break
                except Exception:
                    pass  # Use LTP as fallback

            except Exception as e:
                return jsonify({'ok': False, 'msg': f'Kite order failed: {str(e)}'}), 500

        # â”€â”€ SL/Target: 28% SL, 60% target â”€â”€
        stoploss = round(fill_price * 0.72, 2)
        target = round(fill_price * 1.60, 2)
        total_premium = round(fill_price * quantity, 2)
        max_loss = round((fill_price - stoploss) * quantity, 2)

        trade_id = f'NEWSMAN_{datetime.now().strftime("%H%M%S")}_{random.randint(1000,9999)}'

        pos = {
            'symbol': f'NFO:{tradingsymbol}',
            'underlying': nse_sym,
            'quantity': quantity,
            'lots': lots,
            'avg_price': fill_price,
            'side': 'BUY',
            'direction': direction,
            'option_type': option_type,
            'strike': int(actual_strike),
            'expiry': expiry_str,
            'stop_loss': stoploss,
            'target': target,
            'order_id': order_id,
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'is_option': True,
            'is_live': is_live,
            'total_premium': total_premium,
            'max_loss': max_loss,
            'setup_type': 'WATCHER_EARLYBIRD_D_UP' if sentiment == 'BULLISH' else 'WATCHER_EARLYBIRD_D_DOWN',
            'strategy_type': 'NAKED_OPTION',
            'rationale': f'Manual news: {sentiment} {symbol} conf={confidence} â†’ {option_type} {int(actual_strike)} @ â‚¹{fill_price:.2f}',
            'entry_score': confidence,
            'score_tier': 'premium' if confidence >= 80 else 'standard',
            'smart_score': confidence,
            'lot_multiplier': 1.5,
            'sector': '',
            'trigger_type': 'MANUAL_NEWS',
            'is_sniper': False,
            'delta': 0, 'theta': 0, 'iv': 0,
        }

        # ── Log ENTRY to trade_ledger BEFORE writing the signal file ──
        # CRITICAL: dashboard writes to state_db immediately; the bot's
        # _sync_positions_from_db (3s tick) picks it up from DB before the
        # signal consumer runs, dedupes the signal, and the ENTRY never
        # gets logged. That orphans the eventual EXIT in Trade History.
        # Log here directly and tag _ledger_logged=True so the bot won't
        # double-log when it consumes the signal.
        try:
            from trade_ledger import get_trade_ledger as _gtl
            _gtl().log_entry(
                symbol=pos['symbol'],
                underlying=nse_sym,
                direction=direction,
                source=pos['setup_type'],
                strategy_type='NAKED_OPTION',
                score_tier=pos['score_tier'],
                smart_score=confidence,
                final_score=confidence,
                option_symbol=pos['symbol'],
                strike=int(actual_strike),
                option_type=option_type,
                expiry=expiry_str,
                entry_price=round(fill_price, 2),
                quantity=quantity,
                lots=lots,
                lot_multiplier=1.5,
                stop_loss=stoploss,
                target=target,
                total_premium=total_premium,
                rationale=pos['rationale'],
                order_id=order_id,
                trade_id=trade_id,
            )
            pos['_ledger_logged'] = True
        except Exception as _e:
            print(f"⚠️ news_trade log_entry failed: {_e}")

        # ── Write to signal file so the bot injects into its in-memory positions ──
        signal_file = os.path.join(os.path.dirname(__file__), 'manual_entry_requests.json')
        pending = []
        if os.path.exists(signal_file):
            try:
                with open(signal_file, 'r') as sf:
                    pending = json.loads(sf.read().strip() or '[]')
            except Exception:
                pending = []
        pending.append(pos)
        with open(signal_file, 'w') as sf:
            json.dump(pending, sf)

        # Also write to state_db for immediate dashboard display
        db = get_state_db()
        today = _today()
        positions, realized_pnl, paper_capital = db.load_active_trades(today)
        positions.append(pos)
        db.save_active_trades(positions, realized_pnl, paper_capital)

        mode_label = 'PAPER' if PAPER_MODE else 'LIVE'
        return jsonify({
            'ok': True,
            'msg': f'[{mode_label}] BUY {tradingsymbol} @ â‚¹{fill_price:.2f} | SL â‚¹{stoploss:.2f} | TGT â‚¹{target:.2f} | {lots}L',
            'order_id': order_id,
            'ltp': fill_price,
            'stop_loss': stoploss,
            'target': target,
        })

    except Exception as e:
        return jsonify({'ok': False, 'msg': f'News trade failed: {str(e)}'}), 500


# â”€â”€ System status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/status')
def get_status():
    db = get_state_db()
    today = _today()

    positions, realized_pnl, paper_capital = db.load_active_trades(today)
    # Drop already-closed rows (e.g. SYNC_MISSING_FROM_DB) so the UI never
    # shows ghost rows whose Exit/Reverse buttons would 404.
    positions = [p for p in positions if (p.get('status') or 'OPEN') == 'OPEN']
    positions = _enrich_positions(positions, db)

    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions)

    risk_state = db.load_risk_state(today) or {}
    data_health = db.load_data_health(today) or {}

    svc_active = False
    svc_uptime = ''
    try:
        r = subprocess.run(['systemctl', 'is-active', 'titan-bot'],
                           capture_output=True, text=True, timeout=3)
        svc_active = r.stdout.strip() == 'active'
        if svc_active:
            r2 = subprocess.run(
                ['systemctl', 'show', 'titan-bot', '--property=ActiveEnterTimestamp'],
                capture_output=True, text=True, timeout=3)
            svc_uptime = r2.stdout.strip().split('=', 1)[-1]
    except Exception:
        pass

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'mode': 'PAPER' if PAPER_MODE else 'LIVE',
        'service_active': svc_active,
        'service_uptime': svc_uptime,
        'capital': paper_capital,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': total_unrealized,
        'open_positions': len(positions),
        'positions': positions[:50],
        'risk_state': risk_state,
        'data_health': {
            'stale': data_health.get('stale_counters', '{}'),
            'halted': data_health.get('halted_symbols', '[]'),
        },
        'config': {
            'risk_per_trade': HARD_RULES.get('RISK_PER_TRADE', 0.07),
            'max_daily_loss': HARD_RULES.get('MAX_DAILY_LOSS', 0.20),
            'max_positions': HARD_RULES.get('MAX_POSITIONS', 80),
            'capital': HARD_RULES.get('CAPITAL', 500_000),
            'universe_count': len(APPROVED_UNIVERSE),
        }
    })


@app.route('/api/pnl_live')
def pnl_live():
    """Ultra-lightweight real-time P&L patch feed.
    Returns only {symbol: {ltp, upnl}} for open positions, plus totals.
    Meant for 1-2s client-side polling to patch cells in-place without
    touching systemctl / ledger / risk state. Typical response <2KB.

    Freshness strategy:
      1. Preferred: bot's live_pnl table (updated each scan cycle).
      2. Stale (>3s old): fresh batched kite.ltp() call, cached 800ms.
      3. Kite fetch fails: use _LAST_GOOD_LTP from previous successful call
         (prevents P&L collapsing to 0 when upstream hiccups).
    """
    try:
        db = get_state_db()
        today = _today()
        positions, realized_pnl, _cap = db.load_active_trades(today)
        # Only stream live P&L for still-open rows; closed rows must not appear.
        positions = [p for p in (positions or []) if (p.get('status') or 'OPEN') == 'OPEN']
        live = db.load_live_pnl() or {}

        import time as _t
        now_wall = _t.time()
        last_iso = live.get('_last_updated')
        stale = True
        if last_iso:
            try:
                from datetime import datetime as _dt
                last_dt = _dt.fromisoformat(last_iso.replace('Z', ''))
                age = (_dt.now() - last_dt).total_seconds()
                stale = age > 3.0
            except Exception:
                stale = True

        pos_syms = []
        for p in positions or []:
            s = p.get('symbol') or p.get('option_symbol') or ''
            if s:
                pos_syms.append(s)

        fresh_ltps = {}
        kite_err = None
        # Always attempt fresh Kite LTP for every open position symbol
        # (bounded by the 800ms _LTP_CACHE below). This guarantees that
        # immediately after a manual Reverse / +1 / news-buy the dashboard
        # shows real-time market price instead of the bot's last-cycle
        # snapshot (which can be up to 3-5s stale and may even be missing
        # the freshly-added symbol until the bot's next save cycle).
        if pos_syms:
            try:
                to_fetch = []
                for s in pos_syms:
                    hit = _LTP_CACHE.get(s)
                    if hit and (now_wall - hit[0]) < 0.8:
                        fresh_ltps[s] = hit[1]
                    else:
                        to_fetch.append(s)
                if to_fetch:
                    kite = _get_dashboard_kite()
                    if kite:
                        lookup = []
                        key_map = {}
                        for s in to_fetch:
                            k = s if ':' in s else ('NFO:' + s)
                            lookup.append(k)
                            key_map[k] = s
                        data = kite.ltp(lookup) or {}
                        for k, v in data.items():
                            lp = (v or {}).get('last_price')
                            if lp is not None:
                                orig = key_map.get(k, k)
                                _LTP_CACHE[orig] = (now_wall, float(lp))
                                fresh_ltps[orig] = float(lp)
                    else:
                        kite_err = 'no_kite'
            except Exception as _e:
                kite_err = str(_e)[:80]

        # Last-good per-symbol cache (module-scoped; survives across requests)
        # Stored as {sym: (ts, ltp)} so we can expire stuck values after 30s.
        global _LAST_GOOD_LTP
        try:
            _LAST_GOOD_LTP
        except NameError:
            _LAST_GOOD_LTP = {}
        _LAST_GOOD_TTL = 30.0  # don't serve a cached LTP older than this

        out = {}
        total_unreal = 0.0
        for p in positions or []:
            sym = p.get('symbol') or p.get('option_symbol') or ''
            if not sym:
                continue
            entry = p.get('avg_price') or p.get('entry_price') or p.get('net_premium', 0) or 0
            qty = abs(p.get('quantity', 0) or 0)
            # P&L uses SIDE (actual transaction) not DIRECTION (market view).
            # For bearish-via-long-put trades: side='BUY', direction='SELL' — using direction would flip the sign.
            s = (p.get('side') or ('BUY' if p.get('direction') in ('BUY', 'LONG') else 'SELL')).upper()
            d = p.get('direction') or ('LONG' if p.get('side') == 'BUY' else 'SHORT')
            spread = p.get('is_debit_spread') or p.get('is_credit_spread') or p.get('is_iron_condor')

            # Candidate LTPs in priority order
            ltp = 0.0
            unreal = 0.0
            db_lp = live.get(sym) or live.get(sym.replace('NFO:', ''))
            db_ltp = 0.0
            if isinstance(db_lp, dict):
                db_ltp = float(db_lp.get('ltp') or 0)
            elif isinstance(db_lp, (int, float)):
                db_ltp = float(db_lp)

            lg = _LAST_GOOD_LTP.get(sym)
            lg_fresh = isinstance(lg, tuple) and (now_wall - lg[0]) < _LAST_GOOD_TTL

            if sym in fresh_ltps and fresh_ltps[sym] > 0:
                ltp = float(fresh_ltps[sym])          # 1. Fresh Kite LTP
            elif not stale and db_ltp > 0:
                ltp = db_ltp                          # 2. Recent bot-written LTP
            elif lg_fresh:
                ltp = float(lg[1])                    # 3. Last good (<30s)
            elif db_ltp > 0:
                ltp = db_ltp                          # 4. Stale DB as last resort

            if ltp > 0 and sym in fresh_ltps:
                _LAST_GOOD_LTP[sym] = (now_wall, ltp)  # only remember fresh fetches

            # Compute unrealized (SIDE-based to match bot's authoritative formula)
            if spread and isinstance(db_lp, dict):
                unreal = float(db_lp.get('unrealized_pnl') or 0)
            elif ltp > 0 and entry > 0 and qty > 0:
                if s == 'BUY':
                    unreal = (ltp - entry) * qty
                else:
                    unreal = (entry - ltp) * qty

            # Reverse-click freeze: pin PnL to exactly 0 (and LTP=entry)
            # for a brief window after the user clicks Reverse, so the
            # row opens at ₹0 instead of showing the tick-drift between
            # click-time LTP and the next Kite poll.
            if _reverse_freeze_active(sym) and entry > 0:
                ltp = float(entry)
                unreal = 0.0

            premium = p.get('total_premium') or (entry * qty) or 0
            pnl_pct = (unreal / premium * 100) if premium > 0 else 0
            out[sym] = {
                'ltp': round(ltp, 2),
                'upnl': round(unreal, 2),
                'pct': round(pnl_pct, 2),
            }
            total_unreal += unreal
        return jsonify({
            'ok': True,
            'ts': int(time.time()),
            'stale': stale,
            'fresh_count': len(fresh_ltps),
            'kite_err': kite_err,
            'positions': out,
            'total_unrealized': round(total_unreal, 2),
            'realized_pnl': round(realized_pnl or 0, 2),
            'net_pnl': round((realized_pnl or 0) + total_unreal, 2),
        })
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# â”€â”€ Trade ledger â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route('/api/trade_summary')
def trade_summary():
    """One-call comprehensive trade summary for today.
    Includes: positions, P&L, ledger events, risk state â€” everything."""
    import re as _re
    db = get_state_db()
    today = _today()
    positions, realized_pnl, capital = db.load_active_trades(today)
    live = db.load_live_pnl() or {}

    # â”€â”€ Enrich open positions â”€â”€
    open_positions = []
    total_unreal = 0
    for p in positions:
        sym = p.get('symbol', '')
        entry = p.get('avg_price') or p.get('entry_price') or p.get('net_premium', 0)
        qty = abs(p.get('quantity', 0))
        d = p.get('direction', '')
        # SIDE-based P&L (direction-based would flip sign for bearish-via-long-put trades)
        s = (p.get('side') or ('BUY' if d in ('BUY', 'LONG') else 'SELL')).upper()
        spread = p.get('is_debit_spread') or p.get('is_credit_spread') or p.get('is_iron_condor')

        lp = live.get(sym) or live.get(sym.replace('NFO:', ''))
        ltp = 0
        unreal = 0
        if isinstance(lp, dict):
            ltp = lp.get('ltp', 0)
            unreal = lp.get('unrealized_pnl', 0)
        elif isinstance(lp, (int, float)):
            ltp = float(lp)

        if unreal == 0 and ltp > 0 and entry > 0 and not spread:
            if s == 'BUY':
                unreal = (ltp - entry) * qty
            else:
                unreal = (entry - ltp) * qty

        # Reverse-click freeze: pin to 0 for a brief post-click window.
        if _reverse_freeze_active(sym) and entry > 0:
            ltp = float(entry)
            unreal = 0

        total_unreal += unreal
        pnl_pct = (unreal / (entry * qty) * 100) if entry > 0 and qty > 0 else 0

        # Hold time
        hold_mins = 0
        try:
            ts = p.get('timestamp', '')
            if ts:
                hold_mins = int((datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 60)
        except Exception:
            pass

        # Format entry time for display (HH:MM)
        _entry_time_str = ''
        try:
            if ts:
                _entry_time_str = datetime.fromisoformat(ts).strftime('%H:%M')
        except Exception:
            pass

        open_positions.append({
            'symbol': sym,
            'direction': d,
            'quantity': qty,
            'entry_price': round(entry, 2),
            'ltp': round(ltp, 2),
            'unrealized_pnl': round(unreal, 2),
            'pnl_pct': round(pnl_pct, 1),
            'stop_loss': p.get('stop_loss', 0),
            'target': p.get('target', 0),
            'setup': p.get('setup_type', p.get('strategy_type', '')),
            'trigger_type': p.get('trigger_type', ''),
            'score': p.get('smart_score', p.get('entry_score', 0)),
            'hold_minutes': hold_mins,
            'entry_time': _entry_time_str,
            'is_spread': bool(spread),
            'status': 'winning' if unreal >= 0 else 'losing',
        })

    # â”€â”€ Ledger events â”€â”€
    ledger = get_trade_ledger()
    ledger_entries = []
    ledger_exits = []
    try:
        ledger_dir = Path(__file__).parent / 'trade_ledger'
        ledger_file = ledger_dir / f'trade_ledger_{today}.jsonl'
        if ledger_file.exists():
            import json as _json
            for line in ledger_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ev = _json.loads(line)
                    if ev.get('event') == 'ENTRY':
                        ledger_entries.append({
                            'symbol': ev.get('symbol', ''),
                            'direction': ev.get('direction', ''),
                            'entry_price': ev.get('entry_price', 0),
                            'quantity': ev.get('quantity', 0),
                            'source': ev.get('source', ''),
                            'smart_score': ev.get('smart_score', 0),
                            'time': str(ev.get('timestamp', ''))[:19],
                        })
                    elif ev.get('event') == 'EXIT':
                        ledger_exits.append({
                            'symbol': ev.get('symbol', ''),
                            'exit_type': ev.get('exit_type', ''),
                            'entry_price': ev.get('entry_price', 0),
                            'exit_price': ev.get('exit_price', 0),
                            'pnl': ev.get('pnl', 0),
                            'quantity': ev.get('quantity', 0),
                            'time': str(ev.get('timestamp', ''))[:19],
                        })
                except Exception:
                    pass
    except Exception:
        pass

    # â”€â”€ Risk state â”€â”€
    risk = db.load_risk_state(today) or {}

    # â”€â”€ Build summary â”€â”€
    net_pnl = realized_pnl + total_unreal
    winners = sum(1 for p in open_positions if p['status'] == 'winning')
    losers = sum(1 for p in open_positions if p['status'] == 'losing')

    return jsonify({
        'date': today,
        'capital': capital,
        'realized_pnl': round(realized_pnl, 2),
        'unrealized_pnl': round(total_unreal, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round(net_pnl / capital * 100, 2) if capital > 0 else 0,
        'open_positions': open_positions,
        'open_count': len(open_positions),
        'open_winners': winners,
        'open_losers': losers,
        'ledger_entries': ledger_entries,
        'ledger_exits': ledger_exits,
        'total_entries_today': len(ledger_entries),
        'total_exits_today': len(ledger_exits),
        'risk': {
            'daily_loss_pct': risk.get('daily_loss_pct', 0),
            'circuit_breaker': risk.get('circuit_breaker', False),
        },
    })


@app.route('/api/trades/today')
def trades_today():
    ledger = get_trade_ledger()
    summary = ledger.daily_summary()
    return jsonify(summary)


@app.route('/api/trades/history')
def trades_history():
    days = int(request.args.get('days', 30))
    ledger = get_trade_ledger()
    results = []
    for i in range(days):
        d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        s = ledger.daily_summary(d)
        if s.get('total_trades', 0) > 0 or i == 0:
            s_lite = {k: v for k, v in s.items() if k != 'trades'}
            s_lite['trade_count'] = len(s.get('trades', []))
            results.append(s_lite)
    return jsonify(results)


@app.route('/api/trades/day/<date_str>')
def trades_for_day(date_str):
    ledger = get_trade_ledger()
    summary = ledger.daily_summary(date_str)
    return jsonify(summary)


# â”€â”€ Scan decisions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/scans')
def scan_decisions():
    db = get_state_db()
    date_str = request.args.get('date', _today())
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 200))
    decisions = db.get_scan_decisions(date_str, symbol, limit)
    return jsonify(decisions)


# â”€â”€ Slippage log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/slippage')
def slippage():
    db = get_state_db()
    limit = int(request.args.get('limit', 100))
    records = db.load_slippage_log(limit)
    return jsonify(records)


# â”€â”€ Orders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/orders')
def orders():
    db = get_state_db()
    placed_ids, records = db.load_order_records()
    return jsonify({'placed_ids': list(placed_ids), 'records': records})


# â”€â”€ Exit states â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/exits')
def exit_states():
    db = get_state_db()
    states = db.load_exit_states()
    return jsonify(states)


# â”€â”€ Ledger dates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/ledger-dates')
def ledger_dates():
    dates = []
    if TRADE_LEDGER_DIR.exists():
        for f in sorted(TRADE_LEDGER_DIR.glob('trade_ledger_*.jsonl'), reverse=True):
            d = f.stem.replace('trade_ledger_', '')
            dates.append(d)
    return jsonify(dates[:60])


# â”€â”€ P&L calendar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/pnl-calendar')
def pnl_calendar():
    days = int(request.args.get('days', 30))
    db = get_state_db()
    result = {}
    for i in range(days):
        d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        pnl = db.get_daily_realized_pnl(d)
        if pnl != 0:
            result[d] = round(pnl, 2)
    return jsonify(result)


# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/config')
def config_route():
    return jsonify({
        'hard_rules': HARD_RULES,
        'universe': APPROVED_UNIVERSE,
        'tier1': TIER_1_OPTIONS,
        'tier2': TIER_2_OPTIONS,
        'tier3': TIER_3_OPTIONS,
        'trading_hours': TRADING_HOURS,
        'paper_mode': PAPER_MODE,
    })


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SETTINGS â€” Single source of truth via settings_manager
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from settings_manager import settings as _sm

# Sync defaults on dashboard startup (populate missing keys from config.py)
_sm.sync_defaults()

# All strategy config dicts that can be toggled on/off or adjusted
_STRATEGY_CONFIGS = [
    'BREAKOUT_WATCHER', 'ELITE_AUTO_FIRE', 'DOWN_RISK_GATING',
    'GMM_SNIPER', 'GMM_CONTRARIAN', 'TEST_GMM', 'TEST_XGB',
    'SNIPER_OI_UNWINDING', 'SNIPER_PCR_EXTREME', 'ARBTR_CONFIG',
    'IRON_CONDOR_CONFIG', 'CREDIT_SPREAD_CONFIG', 'DEBIT_SPREAD_CONFIG',
    'GCR_CONFIG', 'ML_DIRECTION_CONFLICT',
]


def _load_settings() -> dict:
    """Load saved settings from settings_manager."""
    return _sm.get_all()


def _save_settings(settings: dict):
    """Save settings via settings_manager (updates json + config in-memory)."""
    _sm.set_many(settings)


def _get_current_settings() -> dict:
    """Build full settings view from settings_manager (single source of truth)."""
    saved = _sm.get_all()

    # Capital & Risk
    capital_risk = {
        'CAPITAL': saved.get('CAPITAL', HARD_RULES.get('CAPITAL', 500000)),
        'RISK_PER_TRADE': saved.get('RISK_PER_TRADE', HARD_RULES.get('RISK_PER_TRADE', 0.07)),
        'MAX_DAILY_LOSS': saved.get('MAX_DAILY_LOSS', HARD_RULES.get('MAX_DAILY_LOSS', 0.20)),
        'MAX_POSITIONS': saved.get('MAX_POSITIONS', HARD_RULES.get('MAX_POSITIONS', 80)),
        'REENTRY_COOLDOWN_MINUTES': saved.get('REENTRY_COOLDOWN_MINUTES', HARD_RULES.get('REENTRY_COOLDOWN_MINUTES', 30)),
        'MIN_OPTION_PREMIUM': saved.get('MIN_OPTION_PREMIUM', HARD_RULES.get('MIN_OPTION_PREMIUM', 3.0)),
        'PORTFOLIO_PROFIT_TARGET': saved.get('PORTFOLIO_PROFIT_TARGET', HARD_RULES.get('PORTFOLIO_PROFIT_TARGET', 0.15)),
    }

    # Strategy toggles (enabled/disabled)
    strategies = {}
    for name in _STRATEGY_CONFIGS:
        cfg = getattr(_config_module, name, {})
        default_enabled = cfg.get('enabled', True) if isinstance(cfg, dict) else True
        strategies[name] = saved.get(f'strategy_{name}', default_enabled)

    # Watcher-specific tunables (from settings_manager)
    watcher = {
        'min_score': saved.get('watcher_min_score', 35),
        'max_trades_per_scan': saved.get('watcher_max_trades_per_scan', 2),
        'max_triggers_per_batch': saved.get('watcher_max_triggers_per_batch', 6),
        'sustain_seconds': saved.get('watcher_sustain_seconds', 53),
        'vix_hard_block_above': saved.get('watcher_vix_hard_block_above', saved.get('watcher_vix_hard_block', 32.0)),
        'momentum_exit_enabled': saved.get('watcher_momentum_exit', True),
    }

    # Lot multipliers per strategy
    lot_multipliers = {}
    for name in ['GMM_SNIPER', 'ARBTR_CONFIG', 'SNIPER_OI_UNWINDING', 'SNIPER_PCR_EXTREME',
                  'DOWN_RISK_GATING', 'GMM_CONTRARIAN']:
        cfg = getattr(_config_module, name, {})
        if isinstance(cfg, dict):
            if name == 'DOWN_RISK_GATING':
                default_mult = cfg.get('all_agree_lot_multiplier', 1.5)
            else:
                default_mult = cfg.get('lot_multiplier', 1.0)
        else:
            default_mult = 1.0
        lot_multipliers[name] = saved.get(f'lots_{name}', default_mult)

    # Trading hours
    th = getattr(_config_module, 'TRADING_HOURS', {})
    hours = {
        'start': saved.get('hours_start', th.get('start', '09:15')),
        'end': saved.get('hours_end', th.get('end', '15:25')),
        'no_new_after': saved.get('hours_no_new_after', th.get('no_new_after', '15:10')),
    }

    # Kill switch
    kill_switch = saved.get('kill_switch', False)

    # Global lot multiplier (scale ALL positions)
    global_lot_multiplier = saved.get('global_lot_multiplier', 1.0)

    return {
        'capital_risk': capital_risk,
        'strategies': strategies,
        'watcher': watcher,
        'lot_multipliers': lot_multipliers,
        'hours': hours,
        'kill_switch': kill_switch,
        'global_lot_multiplier': global_lot_multiplier,
        'last_updated': saved.get('_last_updated', None),
    }


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Return current settings (defaults merged with overrides)."""
    return jsonify(_get_current_settings())


@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings overrides to titan_settings.json.
    Bot reads this file periodically and applies overrides."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        saved = _load_settings()

        # Capital & Risk
        cr = data.get('capital_risk', {})
        if 'CAPITAL' in cr:
            saved['CAPITAL'] = max(50000, min(10000000, int(cr['CAPITAL'])))
        if 'RISK_PER_TRADE' in cr:
            saved['RISK_PER_TRADE'] = max(0.01, min(0.15, float(cr['RISK_PER_TRADE'])))
        if 'MAX_DAILY_LOSS' in cr:
            saved['MAX_DAILY_LOSS'] = max(0.05, min(0.50, float(cr['MAX_DAILY_LOSS'])))
        if 'MAX_POSITIONS' in cr:
            saved['MAX_POSITIONS'] = max(1, min(200, int(cr['MAX_POSITIONS'])))
        if 'REENTRY_COOLDOWN_MINUTES' in cr:
            saved['REENTRY_COOLDOWN_MINUTES'] = max(0, min(120, int(cr['REENTRY_COOLDOWN_MINUTES'])))
        if 'MIN_OPTION_PREMIUM' in cr:
            saved['MIN_OPTION_PREMIUM'] = max(1.0, min(50.0, float(cr['MIN_OPTION_PREMIUM'])))
        if 'PORTFOLIO_PROFIT_TARGET' in cr:
            saved['PORTFOLIO_PROFIT_TARGET'] = max(0.05, min(0.50, float(cr['PORTFOLIO_PROFIT_TARGET'])))

        # Strategy toggles
        strats = data.get('strategies', {})
        for name in _STRATEGY_CONFIGS:
            if name in strats:
                saved[f'strategy_{name}'] = bool(strats[name])

        # Watcher tunables
        wt = data.get('watcher', {})
        if 'min_score' in wt:
            saved['watcher_min_score'] = max(20, min(80, int(wt['min_score'])))
        if 'max_trades_per_scan' in wt:
            saved['watcher_max_trades_per_scan'] = max(1, min(10, int(wt['max_trades_per_scan'])))
        if 'max_triggers_per_batch' in wt:
            saved['watcher_max_triggers_per_batch'] = max(1, min(20, int(wt['max_triggers_per_batch'])))
        if 'sustain_seconds' in wt:
            saved['watcher_sustain_seconds'] = max(10, min(300, int(wt['sustain_seconds'])))
        if 'vix_hard_block_above' in wt:
            saved['watcher_vix_hard_block_above'] = max(15, min(50, float(wt['vix_hard_block_above'])))
        if 'momentum_exit_enabled' in wt:
            saved['watcher_momentum_exit'] = bool(wt['momentum_exit_enabled'])

        # Lot multipliers
        lots = data.get('lot_multipliers', {})
        for name in lots:
            if name in _STRATEGY_CONFIGS or name in ['GMM_SNIPER', 'ARBTR_CONFIG',
                    'SNIPER_OI_UNWINDING', 'SNIPER_PCR_EXTREME', 'DOWN_RISK_GATING', 'GMM_CONTRARIAN']:
                saved[f'lots_{name}'] = max(0.5, min(10.0, float(lots[name])))

        # Trading hours
        hrs = data.get('hours', {})
        for key in ['start', 'end', 'no_new_after']:
            if key in hrs:
                val = str(hrs[key]).strip()
                # Basic HH:MM validation
                if len(val) == 5 and val[2] == ':':
                    saved[f'hours_{key}'] = val

        # Global lot multiplier
        if 'global_lot_multiplier' in data:
            saved['global_lot_multiplier'] = max(0.25, min(5.0, float(data['global_lot_multiplier'])))

        # Kill switch
        if 'kill_switch' in data:
            saved['kill_switch'] = bool(data['kill_switch'])

        saved['_last_updated'] = datetime.now().isoformat()
        _save_settings(saved)

        return jsonify({'ok': True, 'msg': 'Settings saved. Bot will pick up changes within 30 seconds.'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Save failed: {e}'}), 500


@app.route('/api/settings/kill-switch', methods=['POST'])
def kill_switch():
    """Emergency kill switch: stop all trading immediately."""
    try:
        saved = _load_settings()
        saved['kill_switch'] = True
        saved['_last_updated'] = datetime.now().isoformat()
        saved['_kill_switch_activated'] = datetime.now().isoformat()
        _save_settings(saved)

        # Also stop the bot service immediately
        try:
            subprocess.run(['sudo', 'systemctl', 'stop', 'titan-bot'],
                           capture_output=True, text=True, timeout=10)
        except Exception:
            pass

        return jsonify({'ok': True, 'msg': 'ðŸš¨ KILL SWITCH ACTIVATED â€” Bot stopped, all trading halted.'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Kill switch failed: {e}'}), 500


@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset all settings to config.py defaults."""
    try:
        from settings_manager import SETTINGS_FILE as _sf
        if _sf.exists():
            _sf.unlink()
        # Re-sync defaults from config.py
        _sm.sync_defaults()
        _sm.apply_all()
        return jsonify({'ok': True, 'msg': 'Settings reset to defaults. Restart bot to apply.'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Reset failed: {e}'}), 500


# â”€â”€ Pre-market health check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/api/health-check', methods=['POST'])
def health_check():
    """Run pre-market health check and return results."""
    try:
        r = subprocess.run(
            [sys.executable, 'pre_market_check.py'],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent),
        )
        lines = (r.stdout + r.stderr).strip().split('\n')
        passed = sum(1 for l in lines if 'âœ…' in l)
        warned = sum(1 for l in lines if 'âš ' in l)
        failed = sum(1 for l in lines if 'âŒ' in l)
        return jsonify({
            'ok': failed == 0,
            'passed': passed,
            'warned': warned,
            'failed': failed,
            'output': lines,
            'exit_code': r.returncode,
        })
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'msg': 'Health check timed out'}), 504
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# â”€â”€ Watchdog alerts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
WATCHDOG_ALERTS_FILE = LOG_DIR / 'watchdog_alerts.json'

@app.route('/api/alerts')
def watchdog_alerts():
    """Return watchdog alerts for today."""
    try:
        if WATCHDOG_ALERTS_FILE.exists():
            with open(WATCHDOG_ALERTS_FILE, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({'count': 0, 'critical': 0, 'warnings': 0, 'alerts': [],
                        'date': _today(), 'updated': None})
    except Exception as e:
        return jsonify({'count': 0, 'alerts': [], 'error': str(e)})


@app.route('/api/alerts/watchdog-log')
def watchdog_log():
    """Return recent watchdog log lines."""
    n = int(request.args.get('lines', 100))
    logfile = str(LOG_DIR / 'watchdog.log')
    lines = _tail_file(logfile, n)
    return jsonify({'lines': [l.rstrip() for l in lines]})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    print(f"\n{'='*56}")
    print(f"  TITAN v5 â€” Monitoring Dashboard")
    print(f"  http://{host}:{port}")
    print(f"  Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    print(f"{'='*56}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_dashboard(debug=True)
