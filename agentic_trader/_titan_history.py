"""Analyze TITAN_CORE (ORB + Elite Auto-Fire) trade history across all ledger days."""
import json, os, glob
from collections import defaultdict

ledger_files = sorted(glob.glob('trade_ledger/trade_ledger_*.jsonl'))
print(f'Ledger files: {len(ledger_files)}')
for f in ledger_files:
    print(f'  {os.path.basename(f)}')

all_entries = []
all_exits = []

for lf in ledger_files:
    day = os.path.basename(lf).replace('trade_ledger_','').replace('.jsonl','')
    with open(lf) as f:
        for line in f:
            d = json.loads(line)
            d['_day'] = day
            ev = d.get('event','')
            r = d.get('rationale','')
            
            if ev == 'ENTRY':
                is_titan = False
                if 'ELITE AUTO-FIRE' in r:
                    d['_sub'] = 'ELITE'
                    is_titan = True
                elif 'ORB' in r and 'TEST_' not in r and 'SNIPER' not in r:
                    d['_sub'] = 'ORB'
                    is_titan = True
                elif 'Score' in r and 'TEST_' not in r and 'SNIPER' not in r and 'GMM_' not in r and 'ALL_AGREE' not in r:
                    d['_sub'] = 'CORE_SCORE'
                    is_titan = True
                if is_titan:
                    all_entries.append(d)
            elif ev == 'EXIT':
                all_exits.append(d)

# Build exit lookup by order_id
exit_by_oid = {}
for ex in all_exits:
    oid = ex.get('order_id','')
    if oid:
        exit_by_oid[oid] = ex

# Also build by underlying+day for matching
exit_by_sym_day = defaultdict(list)
for ex in all_exits:
    key = (ex.get('underlying',''), ex.get('_day',''))
    exit_by_sym_day[key].append(ex)

print(f'\n{"="*120}')
print(f'TITAN_CORE ENTRIES: {len(all_entries)} total')
print(f'{"="*120}')

# Print all entries with matched exits
total_pnl = 0
wins = 0
losses = 0
flat = 0
open_count = 0
by_day_stats = defaultdict(lambda: {'entries':0, 'pnl':0, 'wins':0, 'losses':0})
by_sub = defaultdict(lambda: {'entries':0, 'pnl':0, 'wins':0, 'losses':0})

for e in all_entries:
    day = e['_day']
    sub = e.get('_sub','?')
    sym = e.get('underlying','?').replace('NSE:','')
    ts = e.get('timestamp','')
    time_part = ts.split('T')[1][:5] if 'T' in ts else '?'
    direction = e.get('direction','?')
    otype = e.get('option_type','?')
    score = e.get('entry_score', e.get('smart_score', 0))
    tier = e.get('score_tier','?')
    prem = e.get('total_premium',0)
    rat = e.get('rationale','')[:65]
    oid = e.get('order_id','')
    
    # Find matching exit
    ex = exit_by_oid.get(oid)
    if not ex:
        # Try by underlying+day
        candidates = exit_by_sym_day.get((e.get('underlying',''), day), [])
        if candidates:
            ex = candidates[0]
    
    pnl_str = ''
    exit_reason = ''
    pnl = 0
    if ex:
        pnl = ex.get('realized_pnl', 0)
        exit_reason = ex.get('exit_reason','')[:45]
        pnl_str = f'Rs{pnl:+,.0f}'
        total_pnl += pnl
        if pnl > 0: 
            wins += 1
            by_day_stats[day]['wins'] += 1
            by_sub[sub]['wins'] += 1
        elif pnl < 0: 
            losses += 1
            by_day_stats[day]['losses'] += 1
            by_sub[sub]['losses'] += 1
        else: flat += 1
        by_day_stats[day]['pnl'] += pnl
        by_sub[sub]['pnl'] += pnl
    else:
        pnl_str = 'OPEN'
        open_count += 1
    
    by_day_stats[day]['entries'] += 1
    by_sub[sub]['entries'] += 1
    
    print(f'{day} {time_part} | {sub:12s} | {sym:15s} | {direction} {otype} | score={score:>3.0f} {tier:>8s} | Rs{prem:>7,.0f} | {pnl_str:>12s} | {exit_reason}')

print(f'\n{"="*120}')
print(f'SUMMARY')
print(f'{"="*120}')
print(f'Total entries : {len(all_entries)}')
print(f'Closed        : {wins + losses + flat} (W:{wins} L:{losses} Flat:{flat})')
print(f'Open          : {open_count}')
print(f'Realized PnL  : Rs {total_pnl:+,.0f}')
if wins + losses > 0:
    wr = wins / (wins + losses) * 100
    print(f'Win Rate      : {wr:.0f}% ({wins}W / {losses}L)')
    
    # Avg win/loss
    win_pnls = []
    loss_pnls = []
    for e in all_entries:
        oid = e.get('order_id','')
        ex = exit_by_oid.get(oid)
        if ex:
            p = ex.get('realized_pnl',0)
            if p > 0: win_pnls.append(p)
            elif p < 0: loss_pnls.append(p)
    
    if win_pnls:
        print(f'Avg Win       : Rs {sum(win_pnls)/len(win_pnls):+,.0f}')
    if loss_pnls:
        print(f'Avg Loss      : Rs {sum(loss_pnls)/len(loss_pnls):+,.0f}')
    if win_pnls and loss_pnls:
        pf = sum(win_pnls) / abs(sum(loss_pnls))
        print(f'Profit Factor : {pf:.2f}')

print(f'\n--- By Day ---')
for d in sorted(by_day_stats.keys()):
    s = by_day_stats[d]
    print(f'  {d}: {s["entries"]} entries, {s["wins"]}W/{s["losses"]}L, PnL=Rs{s["pnl"]:+,.0f}')

print(f'\n--- By Sub-Type ---')
for sub in sorted(by_sub.keys()):
    s = by_sub[sub]
    print(f'  {sub:12s}: {s["entries"]} entries, {s["wins"]}W/{s["losses"]}L, PnL=Rs{s["pnl"]:+,.0f}')
