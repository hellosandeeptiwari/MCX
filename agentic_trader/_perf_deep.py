#!/usr/bin/env python3
"""Deep dive into trade sources."""
import json, glob
from collections import defaultdict

data = defaultdict(list)
for f in sorted(glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/*.jsonl')):
    for line in open(f):
        try:
            t = json.loads(line.strip())
            if t.get('event') == 'EXIT':
                data[t.get('source', '?')].append(t)
        except:
            pass

for source in ['WATCHER', 'TEST_XGB', 'GMM_SNIPER', 'TEST_GMM', 'VWAP_TREND']:
    trades = data.get(source, [])
    if not trades:
        continue
    wins = [t for t in trades if t.get('pnl', 0) >= 0]
    losses = [t for t in trades if t.get('pnl', 0) < 0]
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    
    print(f"=== {source} ({len(trades)} trades, {len(wins)}W/{len(losses)}L, Rs {total_pnl:+,}) ===")
    for t in trades:
        sym = t.get('underlying', '?').replace('NSE:', '')
        pnl = t.get('pnl', 0)
        et = t.get('exit_type', '')
        hm = t.get('hold_minutes', 0)
        sc = t.get('final_score', 0)
        date = t.get('ts', '')[:10]
        ot = t.get('option_type', t.get('symbol', '').split('|')[0][-2:])
        marker = 'W' if pnl >= 0 else 'L'
        print(f"  [{marker}] {date} {sym:15s} {ot:3s} Rs {pnl:>+8,} | {hm:>3}min | score={sc}")
    print()
