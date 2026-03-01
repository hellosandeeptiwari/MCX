"""Full TITAN_CORE history with correct PnL from 'pnl' field."""
import json, os, glob
from collections import defaultdict

ledger_files = sorted(glob.glob('trade_ledger/trade_ledger_*.jsonl'))

# Collect all events per day
day_events = {}
for lf in ledger_files:
    day = os.path.basename(lf).replace('trade_ledger_','').replace('.jsonl','')
    entries = []
    exits = []
    with open(lf) as f:
        for line in f:
            d = json.loads(line)
            d['_day'] = day
            if d.get('event') == 'ENTRY':
                entries.append(d)
            elif d.get('event') == 'EXIT':
                exits.append(d)
    day_events[day] = {'entries': entries, 'exits': exits}

# Classify TITAN_CORE entries
def is_titan_core(rationale):
    r = rationale
    if 'ELITE AUTO-FIRE' in r:
        return 'ELITE'
    if 'TEST_' in r or 'SNIPER' in r or 'GMM_' in r or 'ALL_AGREE' in r:
        return None
    if 'ORB' in r:
        return 'ORB'
    if 'Score' in r and ('breakdown' in r.lower() or 'breakout' in r.lower() or 'bearish' in r.lower() or 'bullish' in r.lower()):
        return 'CORE_SCORE'
    return None

# Match entries to exits by order_id
all_titan = []
for day, evts in sorted(day_events.items()):
    exit_by_oid = {}
    exit_by_sym = defaultdict(list)
    for ex in evts['exits']:
        oid = ex.get('order_id','')
        if oid:
            exit_by_oid[oid] = ex
        sym = ex.get('underlying','')
        exit_by_sym[sym].append(ex)
    
    for e in evts['entries']:
        rat = e.get('rationale','')
        sub = is_titan_core(rat)
        if sub is None:
            continue
        
        oid = e.get('order_id','')
        sym = e.get('underlying','')
        
        # Find exit
        ex = exit_by_oid.get(oid)
        if not ex and sym in exit_by_sym:
            # Match by symbol — take first unmatched
            for candidate in exit_by_sym[sym]:
                ex = candidate
                break
        
        pnl = ex.get('pnl', 0) if ex else None
        pnl_pct = ex.get('pnl_pct', 0) if ex else None
        exit_reason = ex.get('exit_reason','') if ex else 'OPEN'
        
        all_titan.append({
            'day': day,
            'sub': sub,
            'sym': sym.replace('NSE:',''),
            'direction': e.get('direction',''),
            'option_type': e.get('option_type',''),
            'score': e.get('entry_score', e.get('smart_score', 0)),
            'tier': e.get('score_tier','?'),
            'premium': e.get('total_premium', 0),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason[:55],
            'rationale': rat[:70],
            'status': 'CLOSED' if ex else 'OPEN',
        })

# Print
print("="*140)
print("TITAN_CORE (ORB + Elite Auto-Fire) — FULL HISTORY")
print("="*140)

header = f"{'Day':12s} {'Sub':12s} {'Symbol':15s} {'Dir':5s} {'Opt':3s} {'Score':>5s} {'Tier':>8s} {'Premium':>10s} {'PnL':>10s} {'PnL%':>7s} {'Status':>7s} {'Exit Reason'}"
print(header)
print("-"*140)

total_pnl = 0
wins = 0
losses = 0
by_day = defaultdict(lambda: {'e':0,'w':0,'l':0,'pnl':0})
by_sub = defaultdict(lambda: {'e':0,'w':0,'l':0,'pnl':0})
by_tier = defaultdict(lambda: {'e':0,'w':0,'l':0,'pnl':0})
win_pnls = []
loss_pnls = []

for t in all_titan:
    pnl_str = f"Rs{t['pnl']:+,.0f}" if t['pnl'] is not None else 'OPEN'
    pct_str = f"{t['pnl_pct']:+.1f}%" if t['pnl_pct'] is not None else ''
    prem_str = f"Rs{t['premium']:>7,.0f}" if t['premium'] > 0 else '-'
    
    print(f"{t['day']:12s} {t['sub']:12s} {t['sym']:15s} {t['direction']:5s} {t['option_type']:3s} {t['score']:>5.0f} {t['tier']:>8s} {prem_str:>10s} {pnl_str:>10s} {pct_str:>7s} {t['status']:>7s} {t['exit_reason']}")
    
    if t['pnl'] is not None:
        total_pnl += t['pnl']
        if t['pnl'] > 0:
            wins += 1
            win_pnls.append(t['pnl'])
        elif t['pnl'] < 0:
            losses += 1
            loss_pnls.append(t['pnl'])
        
        by_day[t['day']]['pnl'] += t['pnl']
        if t['pnl'] > 0: by_day[t['day']]['w'] += 1
        elif t['pnl'] < 0: by_day[t['day']]['l'] += 1
        
        by_sub[t['sub']]['pnl'] += t['pnl']
        if t['pnl'] > 0: by_sub[t['sub']]['w'] += 1
        elif t['pnl'] < 0: by_sub[t['sub']]['l'] += 1
    
        tier = t['tier'] if t['tier'] not in ('','?','unknown') else 'unknown'
        by_tier[tier]['pnl'] += t['pnl']
        if t['pnl'] > 0: by_tier[tier]['w'] += 1
        elif t['pnl'] < 0: by_tier[tier]['l'] += 1
    
    by_day[t['day']]['e'] += 1
    by_sub[t['sub']]['e'] += 1
    tier = t['tier'] if t['tier'] not in ('','?','unknown') else 'unknown'
    by_tier[tier]['e'] += 1

open_count = sum(1 for t in all_titan if t['status'] == 'OPEN')

print(f"\n{'='*140}")
print("SUMMARY")
print(f"{'='*140}")
print(f"Total Entries : {len(all_titan)}")
print(f"Closed        : {wins + losses} ({wins}W / {losses}L)")
print(f"Open          : {open_count}")
print(f"Realized PnL  : Rs {total_pnl:+,.0f}")
if wins + losses > 0:
    wr = wins / (wins + losses) * 100
    print(f"Win Rate      : {wr:.0f}%")
if win_pnls:
    print(f"Avg Win       : Rs {sum(win_pnls)/len(win_pnls):+,.0f}")
if loss_pnls:
    print(f"Avg Loss      : Rs {sum(loss_pnls)/len(loss_pnls):+,.0f}")
if win_pnls and loss_pnls:
    pf = sum(win_pnls) / abs(sum(loss_pnls))
    print(f"Profit Factor : {pf:.2f}")

print(f"\n--- By Day ---")
for d in sorted(by_day.keys()):
    s = by_day[d]
    print(f"  {d}: {s['e']} entries, {s['w']}W/{s['l']}L, PnL = Rs {s['pnl']:+,.0f}")

print(f"\n--- By Sub-Type ---")
for sub in sorted(by_sub.keys()):
    s = by_sub[sub]
    print(f"  {sub:12s}: {s['e']} entries, {s['w']}W/{s['l']}L, PnL = Rs {s['pnl']:+,.0f}")

print(f"\n--- By Score Tier ---")
for tier in sorted(by_tier.keys()):
    s = by_tier[tier]
    print(f"  {tier:12s}: {s['e']} entries, {s['w']}W/{s['l']}L, PnL = Rs {s['pnl']:+,.0f}")

# Exit reason breakdown
print(f"\n--- Exit Reason Breakdown ---")
exit_reasons = defaultdict(lambda: {'count':0, 'pnl':0})
for t in all_titan:
    if t['status'] == 'OPEN':
        continue
    reason = t['exit_reason']
    if 'IV crush' in reason: bucket = 'IV_CRUSH'
    elif 'Credit spread' in reason: bucket = 'CREDIT_SPREAD_EXIT'
    elif 'Debit spread SL' in reason: bucket = 'DEBIT_SPREAD_SL'
    elif 'Debit spread target' in reason or '80%+' in reason: bucket = 'DEBIT_SPREAD_TARGET'
    elif 'Debit spread auto-exit' in reason: bucket = 'DEBIT_SPREAD_EOD'
    elif 'Speed gate' in reason or 'GREEKS EXIT' in reason: bucket = 'GREEKS_SPEED_EXIT'
    elif 'follow-through' in reason: bucket = 'NO_FOLLOW_THROUGH'
    elif 'BOS' in reason: bucket = 'BOS_EXIT'
    elif 'SL hit' in reason: bucket = 'STOP_LOSS'
    elif 'target' in reason.lower(): bucket = 'TARGET_HIT'
    else: bucket = 'OTHER'
    
    exit_reasons[bucket]['count'] += 1
    exit_reasons[bucket]['pnl'] += t['pnl'] or 0

for bucket in sorted(exit_reasons.keys(), key=lambda b: exit_reasons[b]['pnl']):
    s = exit_reasons[bucket]
    print(f"  {bucket:25s}: {s['count']} trades, PnL = Rs {s['pnl']:+,.0f}")
