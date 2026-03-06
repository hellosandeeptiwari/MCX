#!/usr/bin/env python3
"""Detailed analysis of a single trading day."""
import json, sys
from collections import defaultdict
from datetime import datetime

f = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-05.jsonl'

entries = {}  # symbol -> entry record
exits = []
scans = []

for line in open(f):
    try:
        t = json.loads(line.strip())
    except:
        continue
    ev = t.get('event', '')
    if ev == 'ENTRY':
        sym = t.get('symbol', '')
        entries[sym] = t
    elif ev == 'EXIT':
        exits.append(t)
    elif ev == 'SCAN':
        scans.append(t)

# Sort exits by time
exits.sort(key=lambda x: x.get('ts', ''))

print("=" * 80)
print("TITAN v5 — MARCH 05, 2026 — DETAILED TRADE ANALYSIS")
print("=" * 80)

total_pnl = sum(t.get('pnl', 0) for t in exits)
wins = [t for t in exits if t.get('pnl', 0) >= 0]
losses = [t for t in exits if t.get('pnl', 0) < 0]

print(f"Trades: {len(exits)} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {len(wins)/len(exits)*100:.0f}%")
print(f"Total PnL: Rs {total_pnl:+,.0f}")
print(f"Avg Win: Rs {sum(t['pnl'] for t in wins)/max(len(wins),1):+,.0f} | Avg Loss: Rs {sum(t['pnl'] for t in losses)/max(len(losses),1):+,.0f}")
print()

# By source
src_data = defaultdict(list)
for t in exits:
    src_data[t.get('source', '?')].append(t)

print("--- BY SOURCE ---")
for src, trades in sorted(src_data.items(), key=lambda x: sum(t.get('pnl',0) for t in x[1]), reverse=True):
    pnl = sum(t.get('pnl', 0) for t in trades)
    w = len([t for t in trades if t.get('pnl', 0) >= 0])
    print(f"  {src:20s}: {len(trades):2d} trades | {w}W/{len(trades)-w}L | Rs {pnl:+10,.0f}")
print()

# Chronological trade log
print("--- CHRONOLOGICAL TRADE LOG ---")
print(f"{'#':>2} {'Time':>8} {'Exit':>8} {'Sym':15s} {'Dir':4s} {'Type':3s} {'Source':15s} {'Entry':>8} {'Exit':>8} {'PnL':>10} {'Hold':>5} {'Exit Reason':30s}")
print("-" * 140)

for i, t in enumerate(exits, 1):
    sym = t.get('underlying', '?').replace('NSE:', '')
    opt_sym = t.get('symbol', '')
    # Determine option type
    if 'CE' in opt_sym and 'PE' not in opt_sym:
        otype = 'CE'
    elif 'PE' in opt_sym:
        otype = 'PE'
    else:
        otype = '?'
    
    entry_time = t.get('entry_time', '')
    if entry_time:
        entry_time = entry_time[11:19]
    exit_time = t.get('ts', '')[11:19]
    
    pnl = t.get('pnl', 0)
    marker = '+' if pnl >= 0 else '-'
    
    # Direction
    direction = t.get('direction', '?')
    if direction == 'BUY':
        dir_label = 'BUY'  # Buying CE = bullish, buying PE = bearish
    else:
        dir_label = 'SELL'
    
    src = t.get('source', '?')
    entry_px = t.get('entry_price', 0)
    exit_px = t.get('exit_price', 0)
    hold = t.get('hold_minutes', 0)
    exit_reason = t.get('exit_reason', t.get('exit_type', '?'))[:30]
    
    print(f"{i:>2} {entry_time:>8} {exit_time:>8} {sym:15s} {dir_label:4s} {otype:3s} {src:15s} {entry_px:>8.2f} {exit_px:>8.2f} Rs{pnl:>+9,.0f} {hold:>4}m {exit_reason}")

print("-" * 140)
print(f"{'':>78} TOTAL: Rs{total_pnl:>+9,.0f}")
print()

# Timeline analysis
print("--- TIME BLOCKS ---")
blocks = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
for t in exits:
    entry_time = t.get('entry_time', t.get('ts', ''))
    if entry_time:
        hour = entry_time[11:13]
        block = f"{hour}:00-{hour}:59"
        blocks[block]['trades'] += 1
        blocks[block]['pnl'] += t.get('pnl', 0)
        if t.get('pnl', 0) >= 0:
            blocks[block]['wins'] += 1

for block in sorted(blocks.keys()):
    d = blocks[block]
    print(f"  Entries {block}: {d['trades']} trades | {d['wins']}W/{d['trades']-d['wins']}L | Rs {d['pnl']:+,.0f}")
print()

# Biggest winners and losers
print("--- TOP 3 WINNERS ---")
for t in sorted(exits, key=lambda x: x.get('pnl', 0), reverse=True)[:3]:
    sym = t.get('underlying', '?').replace('NSE:', '')
    pnl = t.get('pnl', 0)
    src = t.get('source', '?')
    hold = t.get('hold_minutes', 0)
    entry = t.get('entry_price', 0)
    exit_p = t.get('exit_price', 0)
    pnl_pct = t.get('pnl_pct', 0)
    print(f"  {sym}: Rs {pnl:+,.0f} ({pnl_pct:+.1f}%) | {src} | {hold}min | {entry:.2f} -> {exit_p:.2f}")

print()
print("--- TOP 3 LOSERS ---")
for t in sorted(exits, key=lambda x: x.get('pnl', 0))[:3]:
    sym = t.get('underlying', '?').replace('NSE:', '')
    pnl = t.get('pnl', 0)
    src = t.get('source', '?')
    hold = t.get('hold_minutes', 0)
    entry = t.get('entry_price', 0)
    exit_p = t.get('exit_price', 0)
    pnl_pct = t.get('pnl_pct', 0)
    exit_reason = t.get('exit_reason', t.get('exit_type', '?'))[:50]
    print(f"  {sym}: Rs {pnl:+,.0f} ({pnl_pct:+.1f}%) | {src} | {hold}min | {entry:.2f} -> {exit_p:.2f} | {exit_reason}")

print()

# Exit type analysis
print("--- EXIT TYPE ANALYSIS ---")
exit_types = defaultdict(lambda: {'count': 0, 'pnl': 0})
for t in exits:
    et = t.get('exit_type', t.get('exit_reason', '?'))
    if 'MANUAL' in str(et).upper():
        et_key = 'MANUAL'
    elif 'SL' in str(et).upper() or 'STOP' in str(et).upper():
        et_key = 'STOP_LOSS'
    elif 'TARGET' in str(et).upper():
        et_key = 'TARGET'
    elif 'TRAIL' in str(et).upper():
        et_key = 'TRAILING'
    elif 'EOD' in str(et).upper() or 'CLOSE' in str(et).upper():
        et_key = 'EOD_CLOSE'
    elif 'RISK' in str(et).upper() or 'GOVERNOR' in str(et).upper():
        et_key = 'RISK_GOV'
    else:
        et_key = str(et)[:25]
    exit_types[et_key]['count'] += 1
    exit_types[et_key]['pnl'] += t.get('pnl', 0)

for et, d in sorted(exit_types.items(), key=lambda x: -x[1]['count']):
    print(f"  {et:25s}: {d['count']:2d} trades | Rs {d['pnl']:+,.0f}")

# Strategy analysis  
print()
print("--- STRATEGY TYPE ---")
strat_data = defaultdict(lambda: {'count': 0, 'pnl': 0})
for t in exits:
    # Check entry for strategy
    sym = t.get('symbol', '')
    entry = entries.get(sym, {})
    strat = entry.get('strategy_type', t.get('strategy_type', 'UNKNOWN'))
    if '|' in sym:
        strat = 'DEBIT_SPREAD'
    strat_data[strat]['count'] += 1
    strat_data[strat]['pnl'] += t.get('pnl', 0)

for st, d in sorted(strat_data.items(), key=lambda x: -x[1]['count']):
    print(f"  {st:20s}: {d['count']:2d} trades | Rs {d['pnl']:+,.0f}")
