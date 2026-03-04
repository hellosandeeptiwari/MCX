#!/usr/bin/env python3
"""Check current state_db P&L values."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_db import get_state_db
from datetime import date

db = get_state_db()
today = str(date.today())
positions, realized_pnl, capital = db.load_active_trades(today)
print(f"realized_pnl={realized_pnl}")
print(f"capital={capital}")
print(f"positions_count={len(positions)}")

# Also check trade ledger for the wrong manual exits
from trade_ledger import get_trade_ledger
ledger = get_trade_ledger()
exits = ledger.get_exits(today)
print("\n--- Manual exits ---")
for e in exits:
    if e.get('exit_type') == 'MANUAL_DASHBOARD':
        sym = e.get('symbol','')
        pnl = e.get('pnl', 0)
        direction = e.get('direction','')
        side = e.get('side','')
        print(f"  {sym}: pnl={pnl}, direction={direction}, side={side}")
