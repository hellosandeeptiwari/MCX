#!/usr/bin/env python3
"""Quick performance analysis of Titan trade ledger."""
import json, glob, sys
from collections import defaultdict
from datetime import datetime

wins = losses = total_pnl = 0
manual = auto = sl_hits = target_hits = trail = breakeven = 0
sources = defaultdict(float)
daily_pnl = defaultdict(float)
daily_trades = defaultdict(int)
pnl_list = []
hold_minutes_list = []
watcher_trades = 0
scanner_trades = 0

for f in sorted(glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/*.jsonl')):
    date = f.split('_')[-1].replace('.jsonl', '')
    for line in open(f):
        try:
            t = json.loads(line.strip())
        except:
            continue
        if t.get('event') != 'EXIT':
            continue
        
        pnl = t.get('pnl', 0)
        total_pnl += pnl
        pnl_list.append(pnl)
        daily_pnl[date] += pnl
        daily_trades[date] += 1
        
        src = t.get('source', '?')
        sources[src] += pnl
        
        hm = t.get('hold_minutes', 0)
        if hm > 0:
            hold_minutes_list.append(hm)
        
        if pnl >= 0:
            wins += 1
        else:
            losses += 1
        
        et = t.get('exit_type', '')
        if 'MANUAL' in et:
            manual += 1
        elif 'SL' in et or 'STOP' in et:
            sl_hits += 1
        elif 'TARGET' in et:
            target_hits += 1
        elif 'TRAIL' in et:
            trail += 1
        elif 'BREAKEVEN' in et:
            breakeven += 1
        else:
            auto += 1

total = wins + losses
if total == 0:
    print("No closed trades found.")
    sys.exit()

avg_win = sum(p for p in pnl_list if p >= 0) / max(wins, 1)
avg_loss = sum(p for p in pnl_list if p < 0) / max(losses, 1)
biggest_win = max(pnl_list)
biggest_loss = min(pnl_list)
avg_hold = sum(hold_minutes_list) / len(hold_minutes_list) if hold_minutes_list else 0

print("=" * 60)
print("TITAN v5 — PAPER TRADING PERFORMANCE REPORT")
print("=" * 60)
print(f"Period: {min(daily_pnl.keys())} to {max(daily_pnl.keys())} ({len(daily_pnl)} trading days)")
print(f"Total closed trades: {total}")
print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wins/total*100:.1f}%")
print(f"Total PnL: Rs {total_pnl:+,.0f}")
print(f"Avg PnL/trade: Rs {total_pnl/total:+,.0f}")
print(f"Avg Win: Rs {avg_win:+,.0f} | Avg Loss: Rs {avg_loss:+,.0f}")
print(f"Profit Factor: {abs(avg_win * wins / (avg_loss * losses)):.2f}" if avg_loss != 0 else "Profit Factor: N/A")
print(f"Biggest Win: Rs {biggest_win:+,.0f} | Biggest Loss: Rs {biggest_loss:+,.0f}")
print(f"Avg Hold Time: {avg_hold:.0f} min")
print()
print("--- Exit Types ---")
print(f"  Manual: {manual} | SL Hit: {sl_hits} | Target: {target_hits} | Trail: {trail} | Breakeven: {breakeven} | Other: {auto}")
print()
print("--- Daily Breakdown ---")
for d in sorted(daily_pnl.keys()):
    print(f"  {d}: Rs {daily_pnl[d]:+10,.0f} ({daily_trades[d]} trades)")
print()
print("--- PnL by Source ---")
src_counts = defaultdict(int)
for f in sorted(glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/*.jsonl')):
    for line in open(f):
        try:
            t = json.loads(line.strip())
            if t.get('event') == 'EXIT':
                src_counts[t.get('source', '?')] += 1
        except:
            pass

for s, p in sorted(sources.items(), key=lambda x: -x[1]):
    cnt = src_counts.get(s, 0)
    wr = sum(1 for f2 in sorted(glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/*.jsonl'))
             for line in open(f2) 
             for t in [json.loads(line.strip())] 
             if t.get('event') == 'EXIT' and t.get('source') == s and t.get('pnl', 0) >= 0)
    print(f"  {s}: Rs {p:+,.0f} ({cnt} trades, {wr/max(cnt,1)*100:.0f}% win)")
