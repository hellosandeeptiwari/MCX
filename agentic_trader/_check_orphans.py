#!/usr/bin/env python3
"""Find orphaned trades (ENTRY without EXIT)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ledger_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger')
ledger_file = os.path.join(ledger_dir, 'trade_ledger_2026-03-04.jsonl')

entries = {}  # symbol -> entry record
exits = {}   # symbol -> exit record

with open(ledger_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        sym = rec.get('symbol', '')
        ev = rec.get('event', '')
        if ev == 'ENTRY':
            entries[sym] = rec
        elif ev == 'EXIT':
            exits[sym] = rec

print("=== ORPHANED TRADES (ENTRY without EXIT) ===")
orphaned = []
for sym, entry in entries.items():
    if sym not in exits:
        src = entry.get('source', '?')
        direction = entry.get('direction', '?')
        price = entry.get('entry_price', 0)
        qty = entry.get('quantity', 0)
        ts = entry.get('ts', '?')
        orphaned.append((ts, sym, src, direction, price, qty))

orphaned.sort()
total_premium = 0
for ts, sym, src, direction, price, qty in orphaned:
    prem = price * qty
    total_premium += prem
    print(f"  {ts[11:19]}  {sym:42s}  {src:15s}  {direction:4s}  entry=Rs{price}  qty={qty}  premium=Rs{prem:,.0f}")

print(f"\n  TOTAL orphaned: {len(orphaned)} trades, total premium at risk: Rs{total_premium:,.0f}")

print(f"\n=== EXITED TRADES ({len(exits)}) ===")
for sym, ex in exits.items():
    pnl = ex.get('realized_pnl', 0)
    reason = ex.get('exit_reason', '?')
    src = entries.get(sym, {}).get('source', '?')
    print(f"  {sym:42s}  {src:15s}  pnl=Rs{pnl:+,.0f}  reason={reason}")
