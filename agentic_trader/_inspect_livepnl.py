"""Inspect live_pnl vs active positions."""
from state_db import get_state_db
from datetime import datetime

db = get_state_db()
today = datetime.now().strftime('%Y-%m-%d')
positions, pnl, cap = db.load_active_trades(today)
live = db.load_live_pnl() or {}

print(f"Active positions: {len(positions)}  realized_pnl={pnl}  capital={cap}")
print(f"live_pnl keys: {len(live)}")
print()
print("--- Positions vs live_pnl ---")
for p in positions:
    sym = p.get('symbol', '')
    lp = live.get(sym) or live.get(sym.replace('NFO:', ''))
    if isinstance(lp, dict):
        ltp = lp.get('ltp', 0)
        upnl = lp.get('unrealized_pnl', 0)
        ts = lp.get('last_updated', '')
        print(f"  {sym:40} ltp={ltp:>8}  upnl={upnl:>10}  ts={ts}")
    else:
        print(f"  {sym:40} NO_LIVE_ENTRY  (lp={lp})")

print()
print("--- live_pnl keys ---")
for k, v in list(live.items())[:20]:
    print(f"  {k}: {v}")
