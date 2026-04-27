#!/usr/bin/env python3
"""Cross-reference today's losing trades against gate-pass data from logs.
For each loser: dump entry data (score, P(move), trigger, conviction, OI, etc.)
and exit data (exit_type, hold_time, pnl) to find which gates SHOULD have blocked."""

import json, os, glob
from datetime import datetime, timedelta
from collections import defaultdict

LEDGER_DIR = '/home/ubuntu/titan/agentic_trader/trade_ledger'
LOG_DIR = '/home/ubuntu/titan/agentic_trader/logs'

today_str = datetime.now().strftime('%Y-%m-%d')
ledger_file = os.path.join(LEDGER_DIR, f'trade_ledger_{today_str}.jsonl')

# Load all trades
trades = []
with open(ledger_file) as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

# Separate entries and exits
entries = [t for t in trades if t.get('event') == 'ENTRY']
exits_map = {}  # trade_id -> exit record
for t in trades:
    if t.get('event') == 'EXIT':
        tid = t.get('trade_id', '')
        exits_map[tid] = t

print(f"=== CROSS-REFERENCE: LOSING TRADES vs GATE DATA ({today_str}) ===")
print(f"Total entries: {len(entries)}, Total exits: {len(exits_map)}")
print()

# Build combined view: entry + exit
combined = []
for e in entries:
    tid = e.get('trade_id', '')
    ex = exits_map.get(tid)
    pnl = ex.get('realized_pnl', 0) if ex else None
    combined.append({
        'trade_id': tid,
        'symbol': e.get('symbol', '').replace('NSE:', ''),
        'direction': e.get('direction', ''),
        'trigger_type': e.get('trigger_type', ''),
        'setup_type': e.get('setup_type', ''),
        'entry_time': e.get('timestamp', ''),
        'score': e.get('score', 0),
        'ml_move_prob': e.get('ml_move_prob', 0),
        'b2_conviction': e.get('b2_conviction', 0),
        'move_pct': e.get('move_pct', 0),
        'oi_confirms': e.get('oi_confirms', False),
        'oi_disagrees': e.get('oi_disagrees', False),
        'spike_plus_surge': e.get('spike_plus_surge', False),
        'entry_price': e.get('entry_price', 0),
        'exit_type': ex.get('exit_type', 'OPEN') if ex else 'OPEN',
        'exit_time': ex.get('timestamp', '') if ex else '',
        'pnl': pnl,
        'hold_mins': 0,
        # Grab any additional entry context
        'raw_entry': e,
        'raw_exit': ex,
    })
    if ex and e.get('timestamp') and ex.get('timestamp'):
        try:
            et = datetime.fromisoformat(e['timestamp'])
            xt = datetime.fromisoformat(ex['timestamp'])
            combined[-1]['hold_mins'] = (xt - et).total_seconds() / 60
        except:
            pass

# Sort by PnL ascending (worst losers first)
losers = [c for c in combined if c['pnl'] is not None and c['pnl'] < 0]
losers.sort(key=lambda x: x['pnl'])

print(f"LOSING TRADES: {len(losers)} (sorted worst first)")
print("=" * 120)

for i, t in enumerate(losers):
    print(f"\n{'─'*120}")
    print(f"#{i+1} | {t['symbol']} | PnL: ₹{t['pnl']:,.0f} | Dir: {t['direction']} | Trigger: {t['trigger_type']}")
    print(f"   Entry: {t['entry_time'][:19]} | Exit: {t['exit_time'][:19] if t['exit_time'] else 'OPEN'} | Hold: {t['hold_mins']:.0f}min")
    print(f"   Score: {t['score']:.0f} | P(move): {t['ml_move_prob']:.2f} | B2 Conviction: {t['b2_conviction']}/4 | Move%: {t['move_pct']:+.2f}%")
    print(f"   OI confirms: {t['oi_confirms']} | OI disagrees: {t['oi_disagrees']} | spike+surge: {t['spike_plus_surge']}")
    print(f"   Exit Type: {t['exit_type']}")
    
    # Vulnerability analysis
    vulns = []
    if t['ml_move_prob'] < 0.55:
        vulns.append(f"LOW_PMOVE({t['ml_move_prob']:.2f}<0.55)")
    if t['b2_conviction'] <= 2:
        vulns.append(f"WEAK_CONVICTION({t['b2_conviction']}/4)")
    if t['oi_disagrees']:
        vulns.append("OI_DISAGREES_AT_ENTRY")
    if not t['oi_confirms']:
        vulns.append("NO_OI_CONFIRM")
    if t['score'] < 45:
        vulns.append(f"LOW_SCORE({t['score']:.0f})")
    if abs(t['move_pct']) < 0.5:
        vulns.append(f"TINY_MOVE({t['move_pct']:+.2f}%)")
    if t['hold_mins'] > 60:
        vulns.append(f"OVERHELD({t['hold_mins']:.0f}min)")
    if t['exit_type'] in ('OPTION_SPEED_GATE', 'OI_STRENGTH_COLLAPSE'):
        vulns.append(f"EXIT_SUGGESTS_BAD_ENTRY({t['exit_type']})")
    
    print(f"   ⚠️ VULNERABILITIES: {' | '.join(vulns) if vulns else 'NONE DETECTED'}")

# Aggregate stats
print(f"\n\n{'='*120}")
print("AGGREGATE VULNERABILITY ANALYSIS")
print("="*120)

vuln_counts = defaultdict(int)
vuln_pnl = defaultdict(float)
trigger_pnl = defaultdict(list)
exit_type_pnl = defaultdict(list)
pmove_buckets = {'<0.50': [], '0.50-0.55': [], '0.55-0.60': [], '0.60+': []}
conviction_buckets = {1: [], 2: [], 3: [], 4: []}

for t in losers:
    trigger_pnl[t['trigger_type']].append(t['pnl'])
    exit_type_pnl[t['exit_type']].append(t['pnl'])
    
    pm = t['ml_move_prob']
    if pm < 0.50: pmove_buckets['<0.50'].append(t['pnl'])
    elif pm < 0.55: pmove_buckets['0.50-0.55'].append(t['pnl'])
    elif pm < 0.60: pmove_buckets['0.55-0.60'].append(t['pnl'])
    else: pmove_buckets['0.60+'].append(t['pnl'])
    
    bc = t['b2_conviction']
    if bc in conviction_buckets: conviction_buckets[bc].append(t['pnl'])

    if t['ml_move_prob'] < 0.55: vuln_counts['LOW_PMOVE'] += 1; vuln_pnl['LOW_PMOVE'] += t['pnl']
    if t['b2_conviction'] <= 2: vuln_counts['WEAK_CONVICTION'] += 1; vuln_pnl['WEAK_CONVICTION'] += t['pnl']
    if t['oi_disagrees']: vuln_counts['OI_DISAGREES'] += 1; vuln_pnl['OI_DISAGREES'] += t['pnl']
    if not t['oi_confirms']: vuln_counts['NO_OI_CONFIRM'] += 1; vuln_pnl['NO_OI_CONFIRM'] += t['pnl']
    if t['score'] < 45: vuln_counts['LOW_SCORE'] += 1; vuln_pnl['LOW_SCORE'] += t['pnl']
    if abs(t['move_pct']) < 0.5: vuln_counts['TINY_MOVE'] += 1; vuln_pnl['TINY_MOVE'] += t['pnl']
    if t['hold_mins'] > 60: vuln_counts['OVERHELD'] += 1; vuln_pnl['OVERHELD'] += t['pnl']

print("\n--- By Vulnerability Type ---")
for v in sorted(vuln_pnl.keys(), key=lambda k: vuln_pnl[k]):
    print(f"  {v:25s}: {vuln_counts[v]:3d} trades, ₹{vuln_pnl[v]:>10,.0f} total loss")

print("\n--- P(move) Distribution of Losers ---")
for bucket, pnls in sorted(pmove_buckets.items()):
    if pnls:
        print(f"  P(move) {bucket:12s}: {len(pnls):3d} trades, ₹{sum(pnls):>10,.0f} total, avg ₹{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- B2 Conviction Distribution of Losers ---")
for conv, pnls in sorted(conviction_buckets.items()):
    if pnls:
        print(f"  Conviction {conv}/4: {len(pnls):3d} trades, ₹{sum(pnls):>10,.0f} total, avg ₹{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- By Trigger Type ---")
for trig in sorted(trigger_pnl.keys(), key=lambda k: sum(trigger_pnl[k])):
    pnls = trigger_pnl[trig]
    print(f"  {trig:25s}: {len(pnls):3d} trades, ₹{sum(pnls):>10,.0f} total, avg ₹{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- By Exit Type ---")
for et in sorted(exit_type_pnl.keys(), key=lambda k: sum(exit_type_pnl[k])):
    pnls = exit_type_pnl[et]
    print(f"  {et:30s}: {len(pnls):3d} trades, ₹{sum(pnls):>10,.0f} total, avg ₹{sum(pnls)/len(pnls):>8,.0f}")

# Also show ALL trades P(move) vs outcome for comparison
print(f"\n\n{'='*120}")
print("ALL TRADES: P(move) vs OUTCOME (winners vs losers)")
print("="*120)
winners = [c for c in combined if c['pnl'] is not None and c['pnl'] >= 0]
print(f"Winners: {len(winners)}, Losers: {len(losers)}")
print(f"Win rate: {len(winners)/(len(winners)+len(losers))*100:.1f}%")
print(f"Avg winner P(move): {sum(w['ml_move_prob'] for w in winners)/max(len(winners),1):.3f}")
print(f"Avg loser P(move):  {sum(t['ml_move_prob'] for t in losers)/max(len(losers),1):.3f}")
print(f"Avg winner Score:   {sum(w['score'] for w in winners)/max(len(winners),1):.1f}")
print(f"Avg loser Score:    {sum(t['score'] for t in losers)/max(len(losers),1):.1f}")
print(f"Avg winner B2:      {sum(w['b2_conviction'] for w in winners)/max(len(winners),1):.2f}")
print(f"Avg loser B2:       {sum(t['b2_conviction'] for t in losers)/max(len(losers),1):.2f}")

# Check if there's a clear threshold that separates
print("\n--- What thresholds would have filtered? ---")
for pm_thresh in [0.52, 0.55, 0.58, 0.60]:
    filtered_losers = [t for t in losers if t['ml_move_prob'] < pm_thresh]
    filtered_winners = [w for w in winners if w['ml_move_prob'] < pm_thresh]
    saved = sum(t['pnl'] for t in filtered_losers)
    lost_wins = sum(w['pnl'] for w in filtered_winners)
    print(f"  P(move)≥{pm_thresh:.2f}: Would block {len(filtered_losers)} losers (save ₹{abs(saved):,.0f}) + {len(filtered_winners)} winners (lose ₹{lost_wins:,.0f}) → net ₹{abs(saved)-lost_wins:+,.0f}")

for sc_thresh in [40, 45, 50]:
    filtered_losers = [t for t in losers if t['score'] < sc_thresh]
    filtered_winners = [w for w in winners if w['score'] < sc_thresh]
    saved = sum(t['pnl'] for t in filtered_losers)
    lost_wins = sum(w['pnl'] for w in filtered_winners)
    print(f"  Score≥{sc_thresh}: Would block {len(filtered_losers)} losers (save ₹{abs(saved):,.0f}) + {len(filtered_winners)} winners (lose ₹{lost_wins:,.0f}) → net ₹{abs(saved)-lost_wins:+,.0f}")

print("\nDone.")
