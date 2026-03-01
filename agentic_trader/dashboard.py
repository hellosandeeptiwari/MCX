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
)
from state_db import get_state_db
from trade_ledger import get_trade_ledger

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

@app.route('/')
def index():
    return render_template('titan_dashboard.html')


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


# ── System status ────────────────────────────────────────────
@app.route('/api/status')
def get_status():
    db = get_state_db()
    today = _today()

    positions, realized_pnl, paper_capital = db.load_active_trades(today)
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
