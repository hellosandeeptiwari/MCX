#!/usr/bin/env python3
"""Cross-reference today's losing trades against gate-pass data from entry logs.
For each loser: dump entry data (score, P(move), trigger, conviction, OI, etc.)
and exit data (exit_type, hold_time, pnl) to find which gates SHOULD have blocked."""

import json, os
from datetime import datetime
from collections import defaultdict

LEDGER = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-02.jsonl'

trades = []
with open(LEDGER) as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

entries_by_tid = {}
exits_by_tid = {}
for t in trades:
    tid = t.get('trade_id', '')
    if t.get('event') == 'ENTRY':
        entries_by_tid[tid] = t
    elif t.get('event') == 'EXIT':
        exits_by_tid[tid] = t

print(f"=== CROSS-REFERENCE: LOSING TRADES vs GATE DATA (2026-04-02) ===")
print(f"Total entries: {len(entries_by_tid)}, Total exits: {len(exits_by_tid)}")
print()

# Build combined view
combined = []
for tid, ex in exits_by_tid.items():
    en = entries_by_tid.get(tid, {})
    pnl = ex.get('pnl', 0)
    # Entry fields
    ml_move_prob = en.get('ml_move_prob', 0)
    final_score = en.get('final_score', 0) or ex.get('final_score', 0)
    smart_score = en.get('smart_score', 0) or ex.get('smart_score', 0)
    direction = en.get('direction', '') or ex.get('direction', '')
    source = en.get('source', '') or ex.get('source', '')
    # Try to get trigger info from entry's extra field
    extra = en.get('extra', {}) or {}
    trigger_type = extra.get('trigger_type', '')
    b2_conviction = extra.get('b2_conviction', 0)
    oi_signal = en.get('oi_signal', '')
    oi_confirms = extra.get('oi_confirms', False)
    oi_disagrees = extra.get('oi_disagrees', False)
    move_pct = extra.get('move_pct', 0)
    ml_confidence = en.get('ml_confidence', 0)
    ml_direction = en.get('ml_direction', '')
    xgb_disagrees = en.get('xgb_disagrees', False)
    gate_prob = en.get('gate_prob', 0)
    strategy_type = en.get('strategy_type', '') or ex.get('strategy_type', '')
    
    symbol = (en.get('underlying', '') or ex.get('underlying', '') or 
              en.get('symbol', '') or ex.get('symbol', ''))
    
    combined.append({
        'trade_id': tid,
        'symbol': symbol.replace('NSE:', ''),
        'option': ex.get('symbol', ''),
        'direction': direction,
        'source': source,
        'trigger_type': trigger_type,
        'strategy_type': strategy_type,
        'entry_time': en.get('ts', ''),
        'exit_time': ex.get('ts', ''),
        'final_score': final_score,
        'smart_score': smart_score,
        'ml_move_prob': ml_move_prob,
        'ml_confidence': ml_confidence,
        'ml_direction': ml_direction,
        'xgb_disagrees': xgb_disagrees,
        'gate_prob': gate_prob,
        'b2_conviction': b2_conviction,
        'oi_signal': oi_signal,
        'oi_confirms': oi_confirms,
        'oi_disagrees': oi_disagrees,
        'move_pct': move_pct,
        'entry_price': en.get('entry_price', 0),
        'exit_price': ex.get('exit_price', 0),
        'exit_type': ex.get('exit_type', ''),
        'exit_reason': ex.get('exit_reason', ''),
        'pnl': pnl,
        'pnl_pct': ex.get('pnl_pct', 0),
        'hold_mins': ex.get('hold_minutes', 0),
        'r_multiple': ex.get('r_multiple', 0),
        'max_favorable': ex.get('max_favorable', 0),
        'candles_held': ex.get('candles_held', 0),
        'lot_mult': en.get('lot_multiplier', 1),
        'delta': en.get('delta', 0),
        'iv': en.get('iv', 0),
        'is_sniper': en.get('is_sniper', False),
        'dr_score': en.get('dr_score', 0) or ex.get('dr_score', 0),
        'score_tier': en.get('score_tier', '') or ex.get('score_tier', ''),
    })

# Sort by PnL ascending (worst losers first)
losers = [c for c in combined if c['pnl'] < 0]
winners = [c for c in combined if c['pnl'] >= 0]
losers.sort(key=lambda x: x['pnl'])

print(f"LOSING TRADES: {len(losers)} | WINNING TRADES: {len(winners)}")
print(f"Total loss: Rs{sum(t['pnl'] for t in losers):,.0f} | Total win: Rs{sum(t['pnl'] for t in winners):,.0f}")
print(f"Net PnL: Rs{sum(t['pnl'] for t in combined):,.0f}")
print("=" * 130)

for i, t in enumerate(losers):
    print(f"\n{'─'*130}")
    print(f"#{i+1} | {t['symbol']:15s} | PnL: Rs{t['pnl']:>8,.0f} ({t['pnl_pct']:+.1f}%) | Dir: {t['direction']} | Src: {t['source']}")
    print(f"   Trigger: {t['trigger_type']:20s} | Strategy: {t['strategy_type']}")
    print(f"   Entry: {t['entry_time'][:19]} | Exit: {t['exit_time'][:19]} | Hold: {t['hold_mins']:.0f}min | Candles: {t['candles_held']}")
    print(f"   FinalScore: {t['final_score']:.0f} | SmartScore: {t['smart_score']:.0f} | ScoreTier: {t['score_tier']} | DR: {t['dr_score']:.0f}")
    print(f"   P(move): {t['ml_move_prob']:.3f} | ML Dir: {t['ml_direction']} | ML Conf: {t['ml_confidence']:.2f} | XGB disagrees: {t['xgb_disagrees']} | gate_prob: {t['gate_prob']:.2f}")
    print(f"   B2 Conviction: {t['b2_conviction']}/4 | Move%: {t['move_pct']:+.2f}% | OI: {t['oi_signal']} | OI confirms: {t['oi_confirms']} | OI disagrees: {t['oi_disagrees']}")
    print(f"   Entry$: {t['entry_price']:.1f} | Exit$: {t['exit_price']:.1f} | Delta: {t['delta']:.3f} | IV: {t['iv']:.1f}% | R-mult: {t['r_multiple']:.2f} | MaxFav: {t['max_favorable']:.1f}%")
    print(f"   Exit Type: {t['exit_type']} | Exit Reason: {t['exit_reason'][:80] if t['exit_reason'] else ''}")
    
    # Vulnerability analysis
    vulns = []
    if t['ml_move_prob'] > 0 and t['ml_move_prob'] < 0.55:
        vulns.append(f"LOW_PMOVE({t['ml_move_prob']:.2f})")
    if t['b2_conviction'] > 0 and t['b2_conviction'] <= 2:
        vulns.append(f"WEAK_CONVICTION({t['b2_conviction']}/4)")
    if t['oi_disagrees']:
        vulns.append("OI_DISAGREES_AT_ENTRY")
    if not t['oi_confirms'] and t['oi_signal']:
        vulns.append(f"OI_NOT_CONFIRMING({t['oi_signal']})")
    if t['final_score'] > 0 and t['final_score'] < 45:
        vulns.append(f"LOW_SCORE({t['final_score']:.0f})")
    if t['move_pct'] != 0 and abs(t['move_pct']) < 0.5:
        vulns.append(f"TINY_MOVE({t['move_pct']:+.2f}%)")
    if t['hold_mins'] > 60:
        vulns.append(f"OVERHELD({t['hold_mins']:.0f}min)")
    if t['exit_type'] in ('OPTION_SPEED_GATE', 'OI_STRENGTH_COLLAPSE'):
        vulns.append(f"BAD_EXIT_SIGNAL({t['exit_type']})")
    if t['xgb_disagrees']:
        vulns.append("XGB_DISAGREES")
    if t['ml_confidence'] > 0 and t['ml_confidence'] < 0.55:
        vulns.append(f"LOW_ML_CONF({t['ml_confidence']:.2f})")
    if t['r_multiple'] < -1.0:
        vulns.append(f"BLOWN_SL(R={t['r_multiple']:.1f})")
    if t['pnl_pct'] < -20:
        vulns.append(f"LARGE_PCT_LOSS({t['pnl_pct']:.0f}%)")
    
    print(f"   ** VULNS: {' | '.join(vulns) if vulns else 'NONE DETECTED'}")

# Aggregate stats
print(f"\n\n{'='*130}")
print("AGGREGATE VULNERABILITY ANALYSIS")
print("="*130)

vuln_counts = defaultdict(int)
vuln_pnl = defaultdict(float)
trigger_pnl = defaultdict(list)
exit_type_pnl = defaultdict(list)
source_pnl = defaultdict(list)
pmove_buckets = {'<0.50': [], '0.50-0.55': [], '0.55-0.60': [], '0.60+': [], 'no_data': []}
conviction_buckets = {0: [], 1: [], 2: [], 3: [], 4: []}

for t in losers:
    trigger_pnl[t['trigger_type'] or 'unknown'].append(t['pnl'])
    exit_type_pnl[t['exit_type']].append(t['pnl'])
    source_pnl[t['source']].append(t['pnl'])
    
    pm = t['ml_move_prob']
    if pm <= 0: pmove_buckets['no_data'].append(t['pnl'])
    elif pm < 0.50: pmove_buckets['<0.50'].append(t['pnl'])
    elif pm < 0.55: pmove_buckets['0.50-0.55'].append(t['pnl'])
    elif pm < 0.60: pmove_buckets['0.55-0.60'].append(t['pnl'])
    else: pmove_buckets['0.60+'].append(t['pnl'])
    
    bc = t['b2_conviction']
    if bc in conviction_buckets: conviction_buckets[bc].append(t['pnl'])

    if pm > 0 and pm < 0.55: vuln_counts['LOW_PMOVE'] += 1; vuln_pnl['LOW_PMOVE'] += t['pnl']
    if bc > 0 and bc <= 2: vuln_counts['WEAK_CONVICTION'] += 1; vuln_pnl['WEAK_CONVICTION'] += t['pnl']
    if t['oi_disagrees']: vuln_counts['OI_DISAGREES'] += 1; vuln_pnl['OI_DISAGREES'] += t['pnl']
    if not t['oi_confirms']: vuln_counts['NO_OI_CONFIRM'] += 1; vuln_pnl['NO_OI_CONFIRM'] += t['pnl']
    if t['final_score'] > 0 and t['final_score'] < 45: vuln_counts['LOW_SCORE'] += 1; vuln_pnl['LOW_SCORE'] += t['pnl']
    if t['move_pct'] != 0 and abs(t['move_pct']) < 0.5: vuln_counts['TINY_MOVE'] += 1; vuln_pnl['TINY_MOVE'] += t['pnl']
    if t['hold_mins'] > 60: vuln_counts['OVERHELD'] += 1; vuln_pnl['OVERHELD'] += t['pnl']
    if t['xgb_disagrees']: vuln_counts['XGB_DISAGREES'] += 1; vuln_pnl['XGB_DISAGREES'] += t['pnl']
    if t['exit_type'] in ('OPTION_SPEED_GATE', 'OI_STRENGTH_COLLAPSE'):
        vuln_counts['BAD_EXIT'] += 1; vuln_pnl['BAD_EXIT'] += t['pnl']

print("\n--- By Vulnerability Type (total PnL impact) ---")
for v in sorted(vuln_pnl.keys(), key=lambda k: vuln_pnl[k]):
    print(f"  {v:25s}: {vuln_counts[v]:3d} trades, Rs{vuln_pnl[v]:>10,.0f} total loss")

print("\n--- P(move) Distribution of Losers ---")
for bucket in ['no_data', '<0.50', '0.50-0.55', '0.55-0.60', '0.60+']:
    pnls = pmove_buckets[bucket]
    if pnls:
        print(f"  P(move) {bucket:12s}: {len(pnls):3d} trades, Rs{sum(pnls):>10,.0f} total, avg Rs{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- B2 Conviction Distribution of Losers ---")
for conv in sorted(conviction_buckets.keys()):
    pnls = conviction_buckets[conv]
    if pnls:
        print(f"  Conviction {conv}/4: {len(pnls):3d} trades, Rs{sum(pnls):>10,.0f} total, avg Rs{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- By Source ---")
for src in sorted(source_pnl.keys(), key=lambda k: sum(source_pnl[k])):
    pnls = source_pnl[src]
    print(f"  {src:25s}: {len(pnls):3d} trades, Rs{sum(pnls):>10,.0f} total, avg Rs{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- By Trigger Type ---")
for trig in sorted(trigger_pnl.keys(), key=lambda k: sum(trigger_pnl[k])):
    pnls = trigger_pnl[trig]
    print(f"  {trig:25s}: {len(pnls):3d} trades, Rs{sum(pnls):>10,.0f} total, avg Rs{sum(pnls)/len(pnls):>8,.0f}")

print("\n--- By Exit Type ---")
for et in sorted(exit_type_pnl.keys(), key=lambda k: sum(exit_type_pnl[k])):
    pnls = exit_type_pnl[et]
    print(f"  {et:30s}: {len(pnls):3d} trades, Rs{sum(pnls):>10,.0f} total, avg Rs{sum(pnls)/len(pnls):>8,.0f}")

# Same stats for winners to compare
print(f"\n\n{'='*130}")
print("WINNER vs LOSER COMPARISON")
print("="*130)
if winners:
    print(f"Winners: {len(winners)}, Losers: {len(losers)}")
    print(f"Win rate: {len(winners)/(len(winners)+len(losers))*100:.1f}%")
    
    w_pm = [w['ml_move_prob'] for w in winners if w['ml_move_prob'] > 0]
    l_pm = [t['ml_move_prob'] for t in losers if t['ml_move_prob'] > 0]
    w_sc = [w['final_score'] for w in winners if w['final_score'] > 0]
    l_sc = [t['final_score'] for t in losers if t['final_score'] > 0]
    w_b2 = [w['b2_conviction'] for w in winners if w['b2_conviction'] > 0]
    l_b2 = [t['b2_conviction'] for t in losers if t['b2_conviction'] > 0]
    
    if w_pm: print(f"Avg winner P(move): {sum(w_pm)/len(w_pm):.3f} (n={len(w_pm)})")
    if l_pm: print(f"Avg loser P(move):  {sum(l_pm)/len(l_pm):.3f} (n={len(l_pm)})")
    if w_sc: print(f"Avg winner Score:   {sum(w_sc)/len(w_sc):.1f} (n={len(w_sc)})")
    if l_sc: print(f"Avg loser Score:    {sum(l_sc)/len(l_sc):.1f} (n={len(l_sc)})")
    if w_b2: print(f"Avg winner B2:      {sum(w_b2)/len(w_b2):.2f} (n={len(w_b2)})")
    if l_b2: print(f"Avg loser B2:       {sum(l_b2)/len(l_b2):.2f} (n={len(l_b2)})")
    
    # Time analysis
    w_times = []
    l_times = []
    for w in winners:
        if w['entry_time']:
            try:
                h = int(w['entry_time'][11:13])
                w_times.append(h)
            except: pass
    for t in losers:
        if t['entry_time']:
            try:
                h = int(t['entry_time'][11:13])
                l_times.append(h)
            except: pass
    
    print(f"\nEntry hour distribution:")
    for h in range(9, 16):
        wc = w_times.count(h)
        lc = l_times.count(h)
        if wc + lc > 0:
            print(f"  {h:02d}:00 - {h+1:02d}:00: {wc} wins, {lc} losses ({wc/(wc+lc)*100:.0f}% win rate)")

# Threshold analysis
print(f"\n\n{'='*130}")
print("WHAT-IF THRESHOLD ANALYSIS")
print("="*130)
all_trades = winners + losers

for pm_thresh in [0.52, 0.55, 0.58, 0.60]:
    fl = [t for t in losers if 0 < t['ml_move_prob'] < pm_thresh]
    fw = [w for w in winners if 0 < w['ml_move_prob'] < pm_thresh]
    saved = abs(sum(t['pnl'] for t in fl))
    lost_w = sum(w['pnl'] for w in fw)
    net = saved - lost_w
    print(f"  P(move)>={pm_thresh:.2f}: Block {len(fl)} losers (save Rs{saved:,.0f}) + {len(fw)} winners (lose Rs{lost_w:,.0f}) -> net Rs{net:+,.0f}")

for sc_thresh in [40, 45, 50, 55]:
    fl = [t for t in losers if 0 < t['final_score'] < sc_thresh]
    fw = [w for w in winners if 0 < w['final_score'] < sc_thresh]
    saved = abs(sum(t['pnl'] for t in fl))
    lost_w = sum(w['pnl'] for w in fw)
    net = saved - lost_w
    print(f"  Score>={sc_thresh}: Block {len(fl)} losers (save Rs{saved:,.0f}) + {len(fw)} winners (lose Rs{lost_w:,.0f}) -> net Rs{net:+,.0f}")

# B2 conviction analysis
for b2_thresh in [2, 3]:
    fl = [t for t in losers if 0 < t['b2_conviction'] < b2_thresh]
    fw = [w for w in winners if 0 < w['b2_conviction'] < b2_thresh]
    saved = abs(sum(t['pnl'] for t in fl))
    lost_w = sum(w['pnl'] for w in fw)
    net = saved - lost_w
    print(f"  B2>={b2_thresh}: Block {len(fl)} losers (save Rs{saved:,.0f}) + {len(fw)} winners (lose Rs{lost_w:,.0f}) -> net Rs{net:+,.0f}")

print("\nDone.")
