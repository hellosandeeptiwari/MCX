#!/usr/bin/env python3
"""Analyze today's trades - March 17, 2026"""
import json, os
from datetime import datetime

BASE = '/home/ubuntu/titan/agentic_trader'

# Load closed trades
closed = []
try:
    with open(os.path.join(BASE, 'closed_trades.json')) as f:
        closed = json.load(f)
except: pass

# Load active trades
active = []
try:
    with open(os.path.join(BASE, 'active_trades.json')) as f:
        active = json.load(f)
except: pass

# Load exit manager state
exit_state = {}
try:
    with open(os.path.join(BASE, 'exit_manager_state.json')) as f:
        exit_state = json.load(f)
except: pass

print("=" * 80)
print("TITAN v5 - TRADE ANALYSIS - March 17, 2026")
print("=" * 80)

# Today's closed trades
today_closed = [t for t in closed if t.get('exit_time', '').startswith('2026-03-17') or t.get('entry_time', '').startswith('2026-03-17')]

print(f"\nCLOSED TRADES TODAY: {len(today_closed)}")
print("-" * 80)

total_pnl = 0
winners = 0
losers = 0

for t in sorted(today_closed, key=lambda x: x.get('entry_time', '')):
    sym = t.get('symbol', t.get('sym', 'UNKNOWN'))
    direction = t.get('direction', '?')
    entry_price = t.get('entry_price', 0)
    exit_price = t.get('exit_price', 0)
    entry_time = t.get('entry_time', '?')
    exit_time = t.get('exit_time', '?')
    pnl = t.get('pnl', t.get('realized_pnl', 0))
    pnl_pct = t.get('pnl_pct', 0)
    trade_type = t.get('trade_type', t.get('setup_type', '?'))
    score = t.get('smart_score', t.get('score', '?'))
    exit_reason = t.get('exit_reason', '?')
    qty = t.get('quantity', t.get('qty', '?'))
    opt_sym = t.get('option_symbol', '?')
    opt_entry = t.get('option_entry_price', '?')
    opt_exit = t.get('option_exit_price', '?')
    
    if pnl is None: pnl = 0
    if pnl_pct is None: pnl_pct = 0
    
    total_pnl += pnl
    if pnl > 0: winners += 1
    elif pnl < 0: losers += 1
    
    icon = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BE"
    print(f"\n  [{icon}] {sym} ({direction}) - {trade_type}")
    print(f"     Entry: Rs{entry_price} @ {entry_time}")
    print(f"     Exit:  Rs{exit_price} @ {exit_time} [{exit_reason}]")
    print(f"     Option: {opt_sym} | Entry: Rs{opt_entry} | Exit: Rs{opt_exit}")
    print(f"     Score: {score} | Qty: {qty}")
    print(f"     P&L: Rs{pnl:,.0f} ({pnl_pct:+.2f}%)")

print(f"\n{'=' * 80}")
n = len(today_closed)
be = n - winners - losers
print(f"CLOSED SUMMARY: {winners}W / {losers}L / {be}BE | Net P&L: Rs{total_pnl:,.0f}")
if n > 0:
    win_rate = winners / n * 100
    avg_pnl = total_pnl / n
    print(f"Win Rate: {win_rate:.0f}% | Avg P&L/trade: Rs{avg_pnl:,.0f}")
print(f"{'=' * 80}")

# Active positions
print(f"\nACTIVE POSITIONS (still open): {len(active)}")
print("-" * 80)

for t in active:
    sym = t.get('symbol', t.get('sym', 'UNKNOWN'))
    direction = t.get('direction', '?')
    entry_price = t.get('entry_price', 0)
    entry_time = t.get('entry_time', '?')
    trade_type = t.get('trade_type', t.get('setup_type', '?'))
    score = t.get('smart_score', t.get('score', '?'))
    qty = t.get('quantity', t.get('qty', '?'))
    opt_sym = t.get('option_symbol', '?')
    opt_entry = t.get('option_entry_price', '?')
    sl = t.get('stop_loss', '?')
    target = t.get('target', '?')
    
    trailing = False
    be_active = False
    for k, v in exit_state.items():
        if sym in k or (isinstance(opt_sym, str) and opt_sym in k):
            trailing = v.get('trailing_active', False)
            be_active = v.get('breakeven_active', False)
            break
    
    trail_str = " [TRAILING]" if trailing else " [BREAKEVEN]" if be_active else ""
    print(f"\n  {sym} ({direction}) - {trade_type}{trail_str}")
    print(f"     Entry: Rs{entry_price} @ {entry_time}")
    print(f"     Option: {opt_sym} @ Rs{opt_entry}")
    print(f"     SL: Rs{sl} | Target: Rs{target} | Score: {score} | Qty: {qty}")

# Trade type breakdown
print(f"\n\nTRADE TYPE BREAKDOWN:")
print("-" * 80)
all_trades = today_closed + active
type_counts = {}
type_pnl = {}
for t in all_trades:
    tt = t.get('trade_type', t.get('setup_type', 'unknown'))
    type_counts[tt] = type_counts.get(tt, 0) + 1
    pnl = t.get('pnl', t.get('realized_pnl', 0)) or 0
    type_pnl[tt] = type_pnl.get(tt, 0) + pnl

for tt in sorted(type_counts.keys()):
    print(f"  {tt}: {type_counts[tt]} trades | Closed P&L: Rs{type_pnl[tt]:,.0f}")

# Score distribution
print(f"\n\nSCORE DISTRIBUTION:")
print("-" * 80)
scores = []
for t in all_trades:
    s = t.get('smart_score', t.get('score', None))
    if s is not None:
        try: scores.append(float(s))
        except: pass

if scores:
    scores.sort()
    print(f"  Min: {min(scores):.0f} | Max: {max(scores):.0f} | Avg: {sum(scores)/len(scores):.0f} | Median: {scores[len(scores)//2]:.0f}")
    
    print(f"\n  Score vs Outcome (closed only):")
    for t in sorted(today_closed, key=lambda x: x.get('smart_score', x.get('score', 0)) or 0, reverse=True):
        s = t.get('smart_score', t.get('score', '?'))
        sym = t.get('symbol', t.get('sym', '?'))
        pnl = t.get('pnl', t.get('realized_pnl', 0)) or 0
        icon = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BE"
        print(f"    [{icon}] Score {s:>5} -> {sym:<20} P&L: Rs{pnl:,.0f}")

print(f"\n{'=' * 80}")
print("END OF ANALYSIS")
