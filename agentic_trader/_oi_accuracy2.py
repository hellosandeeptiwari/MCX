#!/usr/bin/env python3
"""Analyze OI signal accuracy vs trade outcomes for March 12, 2026."""
import json, re

entries = {}
exits = {}
with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-12.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except:
            continue
        ev = r.get('event', '')
        sym = r.get('underlying', r.get('symbol', ''))
        if ev == 'ENTRY':
            entries[sym] = r
        elif ev == 'EXIT':
            exits[sym] = r

# Get OI cross-val penalties from the log — for TRADED stocks only
traded_stocks = set(s.replace('NSE:', '') for s in entries.keys())

oi_penalties = {}  # stock -> {penalty, oi_dir, scored_dir, timestamp}
with open('/home/ubuntu/titan/logs/titan.log') as f:
    for line in f:
        if 'OI cross-val' not in line or 'penalised' not in line:
            continue
        m = re.search(r'\[(\d+:\d+:\d+)', line)
        ts = m.group(1) if m else ''
        
        m2 = re.search(r'(\w+) penalised (-\d+) \(OI (\w+) vs scored (\w+)\)', line)
        if not m2:
            continue
        stock = m2.group(1)
        penalty = int(m2.group(2))
        oi_dir = m2.group(3)
        scored_dir = m2.group(4)
        
        if stock in traded_stocks:
            # Keep the one closest to trade time (last one before trade)
            oi_penalties[stock] = {
                'penalty': penalty,
                'oi_dir': oi_dir,
                'scored_dir': scored_dir,
                'ts': ts
            }

# Also get OI from the OI: tag in place_option_order lines for traded stocks
oi_placement = {}
with open('/home/ubuntu/titan/logs/titan.log') as f:
    for line in f:
        if 'place_option_order' not in line:
            continue
        for stock in traded_stocks:
            if stock in line:
                m = re.search(r'OI:(\w+)\(PCR:([\d.]+)', line)
                if m:
                    oi_placement[stock] = {
                        'signal': m.group(1),
                        'pcr': float(m.group(2))
                    }

print("=" * 105)
print("OI SIGNAL ACCURACY REPORT - March 12, 2026")
print("=" * 105)
print()

print("TRADES WITH OI CONFLICT (OI penalised because it disagreed with trade direction):")
print("-" * 105)
print(f"  {'Stock':15s} {'Dir':5s} {'Score':6s} {'Source':14s} {'OI':10s} {'Penalty':8s} {'P&L':>10s} {'Result':7s} {'OI Verdict':18s}")
print("-" * 105)

oi_right = 0
oi_wrong = 0
conflict_pnl = 0
non_conflict_pnl = 0

for sym in sorted(entries.keys()):
    e = entries[sym]
    x = exits.get(sym)
    direction = e.get('direction', '')
    score = e.get('final_score', 0)
    src = e.get('source', '')
    
    pnl = x.get('total_pnl') or x.get('pnl', 0) if x else None
    
    stock = sym.replace('NSE:', '')
    oi_info = oi_penalties.get(stock)
    
    if pnl is not None:
        result = 'WIN' if pnl > 0 else 'LOSS'
        pnl_str = f"{pnl:+,.0f}"
    else:
        result = 'OPEN'
        pnl_str = 'OPEN'
    
    if oi_info:
        penalty = oi_info['penalty']
        oi_dir = oi_info['oi_dir']
        oi_verdict = ''
        if result == 'LOSS':
            oi_verdict = 'OI WAS RIGHT'
            oi_right += 1
        elif result == 'WIN':
            oi_verdict = 'OI WAS WRONG'
            oi_wrong += 1
        else:
            oi_verdict = '(pending)'
        
        conflict_pnl += pnl or 0
        print(f"  {stock:15s} {direction:5s} {score:6.1f} {src:14s} {oi_dir:10s} {penalty:+4d}     {pnl_str:>10s} {result:7s} {oi_verdict}")

print()
print()
print("TRADES WITHOUT OI CONFLICT (no OI penalty applied):")
print("-" * 105)
print(f"  {'Stock':15s} {'Dir':5s} {'Score':6s} {'Source':14s} {'OI signal':12s} {'P&L':>10s} {'Result':7s}")
print("-" * 105)

no_conflict_wins = 0
no_conflict_losses = 0

for sym in sorted(entries.keys()):
    e = entries[sym]
    x = exits.get(sym)
    stock = sym.replace('NSE:', '')
    if stock in oi_penalties:
        continue  # Already shown above
    
    direction = e.get('direction', '')
    score = e.get('final_score', 0)
    src = e.get('source', '')
    pnl = x.get('total_pnl') or x.get('pnl', 0) if x else None
    
    if pnl is not None:
        result = 'WIN' if pnl > 0 else 'LOSS'
        pnl_str = f"{pnl:+,.0f}"
        non_conflict_pnl += pnl
        if pnl > 0:
            no_conflict_wins += 1
        else:
            no_conflict_losses += 1
    else:
        result = 'OPEN'
        pnl_str = 'OPEN'
    
    placement = oi_placement.get(stock)
    oi_str = f"{placement['signal']}(PCR:{placement['pcr']:.2f})" if placement else 'N/A'
    print(f"  {stock:15s} {direction:5s} {score:6.1f} {src:14s} {oi_str:12s} {pnl_str:>10s} {result:7s}")

print()
print()
print("=" * 105)
print("SUMMARY")
print("=" * 105)
conflict_count = oi_right + oi_wrong
print(f"\n  Trades where OI CONFLICTED with direction: {conflict_count}")
if conflict_count > 0:
    accuracy = oi_right / conflict_count * 100
    print(f"    OI was RIGHT (trade lost):  {oi_right}  ({accuracy:.0f}%)")
    print(f"    OI was WRONG (trade won):   {oi_wrong}  ({100-accuracy:.0f}%)")
    print(f"    OI ACCURACY:                {accuracy:.0f}%")
    print(f"    Total P&L on these trades:  {conflict_pnl:+,.0f}")

print(f"\n  Trades with NO OI conflict: {no_conflict_wins + no_conflict_losses}")
print(f"    Wins: {no_conflict_wins}  Losses: {no_conflict_losses}")
print(f"    Total P&L:  {non_conflict_pnl:+,.0f}")

total_trades = conflict_count + no_conflict_wins + no_conflict_losses
total_pnl = conflict_pnl + non_conflict_pnl
print(f"\n  TOTAL: {total_trades} trades, P&L = {total_pnl:+,.0f}")
if conflict_count > 0:
    print(f"\n  KEY INSIGHT: When OI said 'opposite direction' and we traded anyway,")
    print(f"  OI was correct {accuracy:.0f}% of the time. Those trades lost {abs(conflict_pnl):,.0f}")
