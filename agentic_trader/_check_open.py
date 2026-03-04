#!/usr/bin/env python3
"""Check OPEN trades and ORB_BREAKOUT results"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_ledger import get_trade_ledger

tl = get_trade_ledger()
trades = tl.get_trades_with_pnl('2026-03-04')

print("=== ALL OPEN TRADES ===")
for t in trades:
    if t.get('status','') == 'OPEN':
        print(f"  {t['symbol']:40s}  src={t.get('source','?'):15s}  entry=Rs{t.get('entry_price',0)}")

print("\n=== ALL ORB_BREAKOUT TRADES ===")
for t in trades:
    if 'ORB' in t.get('source',''):
        pnl = t.get('realized_pnl', 0) or 0
        print(f"  {t['symbol']:40s}  status={t.get('status','?'):12s}  pnl={pnl:+.0f}  dir={t.get('direction','?')}  entry=Rs{t.get('entry_price',0)}")

print("\n=== PAPER POSITIONS (active_trades.json) ===")
import json
tf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'active_trades.json')
if os.path.exists(tf):
    with open(tf) as f:
        data = json.load(f)
    print(f"  Date: {data.get('date')}")
    for t in data.get('active_trades', []):
        if t.get('status','') == 'OPEN':
            print(f"  {t['symbol']:40s}  status={t.get('status','?')}")
else:
    print("  active_trades.json not found")

# Check SQLite state
print("\n=== SQLITE STATE ===")
try:
    from state_db import get_state_db
    positions, pnl, cap = get_state_db().load_active_trades('2026-03-04')
    open_pos = [p for p in positions if p.get('status','') == 'OPEN']
    print(f"  Total positions in DB: {len(positions)}, OPEN: {len(open_pos)}, realized_pnl: {pnl}")
    for p in open_pos:
        print(f"  {p['symbol']:40s}  status={p.get('status','?')}")
except Exception as e:
    print(f"  Error: {e}")
