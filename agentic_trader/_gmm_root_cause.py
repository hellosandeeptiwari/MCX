#!/usr/bin/env python3
"""Analyze TEST_GMM trades across ALL days to find root cause of losers."""
import json, os, re, glob

ledger_dir = os.path.join(os.path.dirname(__file__), 'trade_ledger')
files = sorted(glob.glob(os.path.join(ledger_dir, 'trade_ledger_*.jsonl')))

all_entries = []
all_exits = {}

for fpath in files:
    day = os.path.basename(fpath).replace('trade_ledger_', '').replace('.jsonl', '')
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except:
                continue
            if rec.get('source') != 'TEST_GMM':
                continue
            if rec.get('event') == 'ENTRY':
                rec['_day'] = day
                all_entries.append(rec)
            elif rec.get('event') == 'EXIT':
                key = rec.get('underlying', '') + '_' + day
                all_exits[key] = rec

print(f"Total TEST_GMM entries across all days: {len(all_entries)}")
print(f"Total TEST_GMM exits across all days: {len(all_exits)}")

# Match entries with exits
trades = []
for e in all_entries:
    sym = e.get('underlying', '')
    day = e.get('_day', '')
    key = sym + '_' + day
    exit_rec = all_exits.get(key, {})
    pnl = exit_rec.get('pnl', exit_rec.get('realized_pnl', 0))
    
    rationale = e.get('rationale', '')
    up_m = re.search(r'UP=([\d.]+)', rationale)
    dn_m = re.search(r'DN=([\d.]+)', rationale)
    gap_m = re.search(r'gap=([\d.]+)', rationale)
    
    trades.append({
        'day': day,
        'sym': sym.replace('NSE:', ''),
        'time': e.get('ts', '')[11:16],
        'dir': e.get('direction', '?'),
        'up': float(up_m.group(1)) if up_m else e.get('ml_up_score', 0),
        'dn': float(dn_m.group(1)) if dn_m else e.get('ml_down_score', 0),
        'gap': float(gap_m.group(1)) if gap_m else 0,
        'pnl': pnl,
        'dr_score': e.get('dr_score', 0),
        'gate_prob': e.get('gate_prob', e.get('ml_move_prob', 0)),
        'smart_score': e.get('smart_score', e.get('pre_score', 0)),
        'has_exit': bool(exit_rec),
    })

# Only analyze completed trades (with exits)
completed = [t for t in trades if t['has_exit']]
winners = [t for t in completed if t['pnl'] >= 0]
losers = [t for t in completed if t['pnl'] < 0]

print(f"\nCompleted: {len(completed)} | Winners: {len(winners)} | Losers: {len(losers)}")
total_pnl = sum(t['pnl'] for t in completed)
print(f"Total PnL: {total_pnl:+,.0f}")

print(f"\n{'='*90}")
print(f"{'DAY':12s} {'SYM':15s} {'TIME':5s} {'DIR':4s} {'UP':>7s} {'DN':>7s} {'GAP':>7s} {'GATE':>6s} {'SCORE':>6s} {'PNL':>10s}")
print(f"{'-'*90}")
for t in sorted(completed, key=lambda x: (x['day'], x['time'])):
    tag = 'W' if t['pnl'] >= 0 else 'L'
    print(f"{t['day']:12s} {t['sym']:15s} {t['time']:5s} {t['dir']:4s} "
          f"{t['up']:7.4f} {t['dn']:7.4f} {t['gap']:7.4f} {t['gate_prob']:6.3f} "
          f"{t['smart_score']:6.1f} {t['pnl']:+10,.0f} {tag}")

# Statistical comparison
print(f"\n{'='*90}")
print("STATISTICAL COMPARISON (COMPLETED TRADES ONLY):")
for label, group in [("WINNERS", winners), ("LOSERS", losers)]:
    if not group:
        print(f"\n{label}: (none)")
        continue
    avg = lambda key: sum(t[key] for t in group) / len(group)
    print(f"\n{label} (n={len(group)}, total PnL={sum(t['pnl'] for t in group):+,.0f}):")
    print(f"  UP:    avg={avg('up'):.4f}  range=[{min(t['up'] for t in group):.4f}, {max(t['up'] for t in group):.4f}]")
    print(f"  DN:    avg={avg('dn'):.4f}  range=[{min(t['dn'] for t in group):.4f}, {max(t['dn'] for t in group):.4f}]")
    print(f"  GAP:   avg={avg('gap'):.4f}  range=[{min(t['gap'] for t in group):.4f}, {max(t['gap'] for t in group):.4f}]")
    print(f"  GATE:  avg={avg('gate_prob'):.4f}  range=[{min(t['gate_prob'] for t in group):.4f}, {max(t['gate_prob'] for t in group):.4f}]")
    print(f"  SCORE: avg={avg('smart_score'):.1f}  range=[{min(t['smart_score'] for t in group):.1f}, {max(t['smart_score'] for t in group):.1f}]")
    
    # Time distribution
    morning = [t for t in group if t['time'] < '12:00']
    afternoon = [t for t in group if t['time'] >= '12:00']
    print(f"  TIME:  morning={len(morning)}, afternoon={len(afternoon)}")
    
    # Direction distribution  
    buys = [t for t in group if t['dir'] == 'BUY']
    sells = [t for t in group if t['dir'] == 'SELL']
    print(f"  DIR:   BUY={len(buys)}, SELL={len(sells)}")

# What UP score range separates winners from losers?
print(f"\n{'='*90}")
print("OPTIMAL THRESHOLD ANALYSIS:")
if winners and losers:
    # Try different UP thresholds
    print("\nUP score caps (lower UP = cleaner signal for PUT):")
    for up_cap in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]:
        w = [t for t in winners if t['up'] <= up_cap]
        l = [t for t in losers if t['up'] <= up_cap]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  UP≤{up_cap:.2f}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Try different DN thresholds
    print("\nDN score floors (higher DN = stronger signal):")
    for dn_floor in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
        w = [t for t in winners if t['dn'] >= dn_floor]
        l = [t for t in losers if t['dn'] >= dn_floor]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  DN≥{dn_floor:.2f}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Try different GAP thresholds
    print("\nGAP (divergence) floors:")
    for gap_floor in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35]:
        w = [t for t in winners if t['gap'] >= gap_floor]
        l = [t for t in losers if t['gap'] >= gap_floor]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  gap≥{gap_floor:.2f}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Gate prob analysis
    print("\nGATE (P(move)) floors:")
    for gf in [0.0, 0.30, 0.40, 0.50, 0.55, 0.60]:
        w = [t for t in winners if t['gate_prob'] >= gf]
        l = [t for t in losers if t['gate_prob'] >= gf]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  gate≥{gf:.2f}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Smart score analysis
    print("\nSmart score floors:")
    for sf in [0, 20, 30, 40, 50, 55, 60]:
        w = [t for t in winners if t['smart_score'] >= sf]
        l = [t for t in losers if t['smart_score'] >= sf]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  score≥{sf}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Time cutoff
    print("\nTime cutoffs:")
    for tc in ['10:00', '10:30', '11:00', '11:30', '12:00', '13:00', '14:00']:
        w = [t for t in winners if t['time'] <= tc]
        l = [t for t in losers if t['time'] <= tc]
        w_pnl = sum(t['pnl'] for t in w)
        l_pnl = sum(t['pnl'] for t in l)
        print(f"  before {tc}: {len(w)}W/{len(l)}L  net={w_pnl+l_pnl:+,.0f}")
    
    # Combined: winner-aligned thresholds
    print(f"\n{'='*90}")
    print("COMBINED FILTER (aligned to winner numbers):")
    # Find the winner profile ranges
    w_up_max = max(t['up'] for t in winners)
    w_dn_min = min(t['dn'] for t in winners)
    w_gap_min = min(t['gap'] for t in winners)
    w_gate_min = min(t['gate_prob'] for t in winners)
    w_score_min = min(t['smart_score'] for t in winners)
    w_latest = max(t['time'] for t in winners)
    
    print(f"  Winner ranges:")
    print(f"    UP:    [{min(t['up'] for t in winners):.4f} - {w_up_max:.4f}]")
    print(f"    DN:    [{w_dn_min:.4f} - {max(t['dn'] for t in winners):.4f}]")
    print(f"    GAP:   [{w_gap_min:.4f} - {max(t['gap'] for t in winners):.4f}]")
    print(f"    GATE:  [{w_gate_min:.4f} - {max(t['gate_prob'] for t in winners):.4f}]")
    print(f"    SCORE: [{w_score_min:.1f} - {max(t['smart_score'] for t in winners):.1f}]")
    print(f"    TIME:  [earliest={min(t['time'] for t in winners)}, latest={w_latest}]")
    
    # Apply winner-aligned filter to all trades
    aligned = [t for t in completed 
               if t['up'] <= w_up_max 
               and t['dn'] >= w_dn_min 
               and t['gap'] >= w_gap_min]
    a_w = [t for t in aligned if t['pnl'] >= 0]
    a_l = [t for t in aligned if t['pnl'] < 0]
    a_pnl = sum(t['pnl'] for t in aligned)
    print(f"\n  Aligned filter (UP≤{w_up_max:.4f} DN≥{w_dn_min:.4f} gap≥{w_gap_min:.4f}):")
    print(f"    {len(a_w)}W/{len(a_l)}L  net={a_pnl:+,.0f}")
    for t in sorted(aligned, key=lambda x: (x['day'], x['time'])):
        tag = 'W' if t['pnl'] >= 0 else 'L'
        print(f"    {t['day']} {t['sym']:15s} {t['time']} UP={t['up']:.4f} DN={t['dn']:.4f} gap={t['gap']:.4f} PnL={t['pnl']:+,.0f} {tag}")
