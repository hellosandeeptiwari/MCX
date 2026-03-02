"""Quick diagnostic for trade ledger data."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_ledger import get_trade_ledger

ledger = get_trade_ledger()
summary = ledger.daily_summary()
total = summary.get('total_trades', 0)
print(f"daily_summary total_trades: {total}")
if summary.get('trades'):
    for t in summary['trades'][:5]:
        print(f"  {t.get('underlying','?')} pnl={t.get('total_pnl',0):.0f} result={t.get('final_result','?')}")

# Raw check
ledger_dir = os.path.join(os.path.dirname(__file__), 'trade_ledger')
from datetime import datetime
fname = os.path.join(ledger_dir, f'trade_ledger_{datetime.now().strftime("%Y-%m-%d")}.jsonl')
if os.path.exists(fname):
    with open(fname) as f:
        lines = f.readlines()
    print(f"\nRaw ledger file: {len(lines)} lines")
    events = {}
    for l in lines:
        d = json.loads(l)
        ev = d.get('event', '?')
        events[ev] = events.get(ev, 0) + 1
    print(f"Events: {events}")
    print("\nLast 3 lines:")
    for l in lines[-3:]:
        d = json.loads(l)
        print(f"  event={d.get('event')} symbol={d.get('symbol','?')} ts={d.get('ts','?')[:19]}")
else:
    print(f"No ledger file: {fname}")

# Also check state_db
from state_db import get_state_db
db = get_state_db()
today = datetime.now().strftime('%Y-%m-%d')
positions, pnl, cap = db.load_active_trades(today)
print(f"\nstate_db: {len(positions)} active positions, realized_pnl={pnl:.2f}, capital={cap:.0f}")
for p in positions[:3]:
    print(f"  {p.get('symbol','?')} qty={p.get('quantity',0)} avg={p.get('avg_price',0)}")
