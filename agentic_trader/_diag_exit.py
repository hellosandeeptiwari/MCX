"""Diagnose manual exit P&L tracking"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from state_db import get_state_db
from trade_ledger import get_trade_ledger
from datetime import datetime

db = get_state_db()
today = datetime.now().strftime('%Y-%m-%d')

# 1. State DB
positions, realized_pnl, cap = db.load_active_trades(today)
print(f"=== STATE DB ===")
print(f"Positions: {len(positions)}")
print(f"Realized P&L: {realized_pnl}")
print(f"Capital: {cap}")
for p in positions:
    sym = p.get('symbol', '')
    st = p.get('status', '?')
    entry = p.get('avg_price') or p.get('entry_price', 0)
    print(f"  {sym} status={st} entry={entry}")

# 2. Live P&L bridge
live = db.load_live_pnl() or {}
print(f"\n=== LIVE P&L BRIDGE ({len(live)} entries) ===")
for sym, lp in list(live.items())[:10]:
    if isinstance(lp, dict):
        print(f"  {sym}: ltp={lp.get('ltp',0)} upnl={lp.get('unrealized_pnl',0)}")
    else:
        print(f"  {sym}: raw={lp}")

# 3. Daily state from daily_state table
try:
    with db._lock:
        ds = db._conn.execute(
            "SELECT realized_pnl, paper_capital, last_updated FROM daily_state WHERE date = ?", (today,)
        ).fetchone()
    if ds:
        print(f"\n=== DAILY STATE ===")
        print(f"Realized P&L: {ds['realized_pnl']}")
        print(f"Capital: {ds['paper_capital']}")
        print(f"Last updated: {ds['last_updated']}")
except Exception as e:
    print(f"daily_state error: {e}")

# 4. Trade ledger entries today (EXITs)
print(f"\n=== TRADE LEDGER TODAY ===")
ledger = get_trade_ledger()
summary = ledger.daily_summary()
print(f"Total trades: {summary.get('total_trades', 0)}")
print(f"Wins: {summary.get('wins', 0)}, Losses: {summary.get('losses', 0)}")
print(f"Net P&L: {summary.get('net_pnl', 0)}")
for t in summary.get('trades', []):
    if t.get('event') == 'EXIT':
        print(f"  EXIT: {t.get('symbol', '')} type={t.get('exit_type', '')} pnl={t.get('pnl', 0)} reason={t.get('exit_reason', '')}")

# 5. Check for manual exit signal file
signal_file = os.path.join(os.path.dirname(__file__), 'manual_exit_requests.json')
if os.path.exists(signal_file):
    print(f"\n=== PENDING MANUAL EXIT SIGNALS ===")
    with open(signal_file) as f:
        signals = json.load(f)
    for s in signals:
        print(f"  {s.get('symbol', '')} pnl={s.get('pnl', 0)} time={s.get('exit_time', '')}")
else:
    print(f"\n=== No pending manual exit signals ===")

# 6. Check get_daily_realized_pnl
daily_rpnl = db.get_daily_realized_pnl(today)
print(f"\n=== get_daily_realized_pnl() = {daily_rpnl} ===")
