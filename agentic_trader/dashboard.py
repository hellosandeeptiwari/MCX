"""
TITAN v5 — Monitoring Dashboard
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
    TRADING_HOURS, TIER_1_OPTIONS, TIER_2_OPTIONS,
    ZERODHA_API_KEY,
)
from state_db import get_state_db
from trade_ledger import get_trade_ledger

# ── Lightweight Kite instance for LIVE exit orders ────────────
_dashboard_kite = None

def _get_dashboard_kite():
    """Lazy-init a KiteConnect instance for placing exit orders.
    Reuses the same access token as the bot (from .env)."""
    global _dashboard_kite
    if _dashboard_kite is not None:
        return _dashboard_kite
    try:
        from kiteconnect import KiteConnect
        token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
        if not token:
            return None
        kite = KiteConnect(api_key=ZERODHA_API_KEY, timeout=15)
        kite.set_access_token(token)
        kite.profile()  # validate token
        _dashboard_kite = kite
        return kite
    except Exception as e:
        print(f"⚠️ Dashboard KiteConnect init failed: {e}")
        return None

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
TRADE_LEDGER_DIR = Path(__file__).parent / 'trade_ledger'

# ── Utility ──────────────────────────────────────────────────

def _today() -> str:
    return datetime.now().strftime('%Y-%m-%d')


def _safe_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ── Server-Sent Events: live log tail ────────────────────────

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
    - 100ms poll instead of 500ms → 5x faster log delivery
    - SSE heartbeat comment every 15s → keeps connection alive through
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
                            # File was rotated — reopen from start
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


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

# Server boot timestamp — changes on every restart (i.e. every deploy)
import time as _time
_SERVER_BOOT = str(int(_time.time()))

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


# ── Live log SSE streams ─────────────────────────────────────
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


# ── Smart Log Reader with Bookmarks ──────────────────────────
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
      file          – log key: titan|titan_error|dashboard_error|watchdog|bot_debug
                      (default: titan). Use "all" for metadata-only overview.
      lines         – max lines to return (default: 300)
      since_bookmark – if "true", return only new lines after saved bookmark
      set_bookmark  – if "true", save current end-of-file as bookmark after read
      from_scan     – if "true", return lines starting from last SCAN CYCLE
      grep          – optional regex filter applied to returned lines
      tail          – if "true" (default), read last N lines; if "false", read from bookmark/scan

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

    # ── Always return metadata for ALL log files ──
    meta = {}
    for key, path in _LOG_FILES.items():
        fm = _file_meta(path)
        bm_line = bookmarks.get(key, {}).get('line', 0)
        fm['bookmark_line'] = bm_line
        fm['new_lines'] = max(0, fm['total_lines'] - bm_line) if fm['exists'] else 0
        meta[key] = fm

    # ── If "all", return just metadata overview ──
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

    # ── Decide start line ──
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

    # ── Optional grep filter ──
    if grep_pat:
        try:
            pat = _re.compile(grep_pat, _re.IGNORECASE)
            lines = [l for l in lines if pat.search(l)]
        except _re.error:
            pass  # bad regex — return unfiltered

    # ── Set bookmark ──
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


# ── Bot control (start / stop / restart) ─────────────────────

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


# ── Manual position exit (dashboard → signal file → bot) ─────

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
        direction = target.get('direction') or target.get('side', 'BUY')

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

        # 1️⃣  LIVE MODE: Place real exit order immediately from dashboard
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
                        live_exit_msg += f' order:{order_id}'

                    live_exit_placed = True
                except Exception as e:
                    live_exit_msg = f'⚠️ LIVE exit order failed: {e}'
                    print(f"   🚨 Dashboard LIVE exit failed for {symbol}: {e}")

        # 2️⃣  Write signal file for bot (in-memory cleanup + sync)
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

        # 2️⃣  Remove from state_db immediately (UI refreshes)
        new_realized = realized_pnl + pnl
        db.save_active_trades(remaining, new_realized, paper_capital)

        # 3️⃣  Log EXIT in trade_ledger
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
            print(f"⚠️ Trade ledger log failed for manual exit: {e}")

        _mode_label = 'LIVE' if not PAPER_MODE else 'PAPER'
        _exit_msg = f'[{_mode_label}] Exited {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+,.2f}'
        if live_exit_placed:
            _exit_msg += ' | Broker order placed ✅'
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


# ── Helpers: enrich positions with exit-state LTP & unrealized P&L ──

def _enrich_positions(positions: list, db) -> list:
    """Merge live P&L data into each position dict so the dashboard
    can display current LTP, unrealized P&L, underlying symbol, etc."""
    import re

    # Primary source: live_pnl table (updated every scan cycle by the bot)
    live = db.load_live_pnl() or {}

    for pos in positions:
        sym = pos.get('symbol') or pos.get('option_symbol') or ''
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
    return positions


# ── System status ────────────────────────────────────────────
@app.route('/api/status')
def get_status():
    db = get_state_db()
    today = _today()

    positions, realized_pnl, paper_capital = db.load_active_trades(today)
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


# ── Trade ledger ─────────────────────────────────────────────

@app.route('/api/trade_summary')
def trade_summary():
    """One-call comprehensive trade summary for today.
    Includes: positions, P&L, ledger events, risk state — everything."""
    import re as _re
    db = get_state_db()
    today = _today()
    positions, realized_pnl, capital = db.load_active_trades(today)
    live = db.load_live_pnl() or {}

    # ── Enrich open positions ──
    open_positions = []
    total_unreal = 0
    for p in positions:
        sym = p.get('symbol', '')
        entry = p.get('avg_price') or p.get('entry_price') or p.get('net_premium', 0)
        qty = abs(p.get('quantity', 0))
        d = p.get('direction', '')
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
            if d in ('BUY', 'LONG'):
                unreal = (ltp - entry) * qty
            else:
                unreal = (entry - ltp) * qty

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
            'score': p.get('smart_score', p.get('entry_score', 0)),
            'hold_minutes': hold_mins,
            'is_spread': bool(spread),
            'status': 'winning' if unreal >= 0 else 'losing',
        })

    # ── Ledger events ──
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

    # ── Risk state ──
    risk = db.load_risk_state(today) or {}

    # ── Build summary ──
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


# ── Scan decisions ───────────────────────────────────────────
@app.route('/api/scans')
def scan_decisions():
    db = get_state_db()
    date_str = request.args.get('date', _today())
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 200))
    decisions = db.get_scan_decisions(date_str, symbol, limit)
    return jsonify(decisions)


# ── Slippage log ─────────────────────────────────────────────
@app.route('/api/slippage')
def slippage():
    db = get_state_db()
    limit = int(request.args.get('limit', 100))
    records = db.load_slippage_log(limit)
    return jsonify(records)


# ── Orders ───────────────────────────────────────────────────
@app.route('/api/orders')
def orders():
    db = get_state_db()
    placed_ids, records = db.load_order_records()
    return jsonify({'placed_ids': list(placed_ids), 'records': records})


# ── Exit states ──────────────────────────────────────────────
@app.route('/api/exits')
def exit_states():
    db = get_state_db()
    states = db.load_exit_states()
    return jsonify(states)


# ── Ledger dates ─────────────────────────────────────────────
@app.route('/api/ledger-dates')
def ledger_dates():
    dates = []
    if TRADE_LEDGER_DIR.exists():
        for f in sorted(TRADE_LEDGER_DIR.glob('trade_ledger_*.jsonl'), reverse=True):
            d = f.stem.replace('trade_ledger_', '')
            dates.append(d)
    return jsonify(dates[:60])


# ── P&L calendar ─────────────────────────────────────────────
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


# ── Config ───────────────────────────────────────────────────
@app.route('/api/config')
def config_route():
    return jsonify({
        'hard_rules': HARD_RULES,
        'universe': APPROVED_UNIVERSE,
        'tier1': TIER_1_OPTIONS,
        'tier2': TIER_2_OPTIONS,
        'trading_hours': TRADING_HOURS,
        'paper_mode': PAPER_MODE,
    })


# ── Pre-market health check ─────────────────────────────────
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
        passed = sum(1 for l in lines if '✅' in l)
        warned = sum(1 for l in lines if '⚠' in l)
        failed = sum(1 for l in lines if '❌' in l)
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


# ── Watchdog alerts ──────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    print(f"\n{'='*56}")
    print(f"  TITAN v5 — Monitoring Dashboard")
    print(f"  http://{host}:{port}")
    print(f"  Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    print(f"{'='*56}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_dashboard(debug=True)
