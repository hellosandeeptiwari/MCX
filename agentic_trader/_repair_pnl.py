#!/usr/bin/env python3
"""Fix state_db capital to match: base_capital(500000) + realized_pnl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_db import get_state_db
from datetime import date

BASE_CAPITAL = 500000
db = get_state_db()
today = str(date.today())
positions, realized_pnl, capital = db.load_active_trades(today)
print(f"BEFORE: realized_pnl={realized_pnl}, capital={capital}")

correct_capital = BASE_CAPITAL + realized_pnl
print(f"CORRECT: capital = {BASE_CAPITAL} + ({realized_pnl}) = {correct_capital}")

db.save_active_trades(positions, realized_pnl, correct_capital)
_, rp2, cap2 = db.load_active_trades(today)
print(f"AFTER: realized_pnl={rp2}, capital={cap2}")
