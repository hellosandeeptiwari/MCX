#!/usr/bin/env python3
"""Analyze TEST_GMM sweet spot - Mar 16 + Mar 17, 2026"""
import json, os

BASE = '/home/ubuntu/titan/agentic_trader/trade_ledger'
gmm_exits = []

for day in ['2026-03-16', '2026-03-17']:
    path = os.path.join(BASE, f'trade_ledger_{day}.jsonl')
    if not os.path.exists(path): continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if d.get('event') == 'EXIT' and d.get('source') == 'TEST_GMM':
                d['_day'] = day
                gmm_exits.append(d)

print("=" * 110)
print(f"TEST_GMM ANALYSIS — Mar 16 + Mar 17 ({len(gmm_exits)} trades)")
print("=" * 110)

# Sort by score
gmm_exits.sort(key=lambda x: x.get('smart_score', 0) or 0, reverse=True)

print(f"\n{'Day':<12} {'Sym':<18} {'Dir':<5} {'Score':>6} {'Gate':>6} {'DR':>6} {'P&L':>10} {'P&L%':>7} {'Exit Type':<35} {'Hold':>5} {'MaxR':>5}")
print("-" * 110)

total_pnl = 0
for d in gmm_exits:
    und = d.get('underlying', d.get('symbol','?'))
    if und.startswith('NSE:'): und = und[4:]
    pnl = d.get('pnl', 0) or 0
    total_pnl += pnl
    icon = 'W' if pnl > 0 else 'L' if pnl < 0 else '-'
    gate = 0
    # extract gate from rationale
    rat = d.get('rationale', '') or ''
    import re
    gm = re.search(r'gate=([\d.]+)', rat)
    if gm: gate = float(gm.group(1))
    
    print(f"{d['_day']:<12} {und:<18} {d.get('direction','?'):<5} {d.get('smart_score',0):>6.1f} {gate:>6.2f} {d.get('dr_score',0):>6.3f} {pnl:>+10,.0f} {d.get('pnl_pct',0):>+6.1f}% {d.get('exit_type','?'):<35} {d.get('hold_minutes',0):>5} {d.get('r_multiple',0):>5.2f} {icon}")

# Score bucketing
print(f"\n\n{'=' * 110}")
print("SCORE BUCKET ANALYSIS")
print("=" * 110)

buckets = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
for lo, hi in buckets:
    b = [d for d in gmm_exits if lo <= (d.get('smart_score', 0) or 0) < hi]
    if not b: continue
    w = sum(1 for d in b if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in b if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in b)
    wr = w/len(b)*100 if b else 0
    avg = pnl/len(b) if b else 0
    print(f"  Score {lo:>2}-{hi:<3}: {len(b):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net P&L: Rs {pnl:>+10,.0f} | Avg: Rs {avg:>+8,.0f}")

# Gate prob bucketing
print(f"\n\nGATE PROBABILITY ANALYSIS")
print("=" * 110)
for d in gmm_exits:
    rat = d.get('rationale', '') or ''
    gm = re.search(r'gate=([\d.]+)', rat)
    d['_gate'] = float(gm.group(1)) if gm else 0

gate_buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]
for lo, hi in gate_buckets:
    b = [d for d in gmm_exits if lo <= d['_gate'] < hi]
    if not b: continue
    w = sum(1 for d in b if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in b if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in b)
    wr = w/len(b)*100 if b else 0
    avg = pnl/len(b) if b else 0
    print(f"  Gate {lo:.2f}-{hi:.2f}: {len(b):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net P&L: Rs {pnl:>+10,.0f} | Avg: Rs {avg:>+8,.0f}")

# DR score bucketing
print(f"\n\nDR SCORE ANALYSIS")
print("=" * 110)
dr_buckets = [(0, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50)]
for lo, hi in dr_buckets:
    b = [d for d in gmm_exits if lo <= (d.get('dr_score', 0) or 0) < hi]
    if not b: continue
    w = sum(1 for d in b if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in b if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in b)
    wr = w/len(b)*100 if b else 0
    avg = pnl/len(b) if b else 0
    print(f"  DR {lo:.2f}-{hi:.2f}: {len(b):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net P&L: Rs {pnl:>+10,.0f} | Avg: Rs {avg:>+8,.0f}")

# Hold time analysis
print(f"\n\nHOLD TIME vs OUTCOME")
print("=" * 110)
hold_buckets = [(0, 30), (30, 60), (60, 120), (120, 200), (200, 400)]
for lo, hi in hold_buckets:
    b = [d for d in gmm_exits if lo <= (d.get('hold_minutes', 0) or 0) < hi]
    if not b: continue
    w = sum(1 for d in b if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in b if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in b)
    wr = w/len(b)*100 if b else 0
    print(f"  Hold {lo:>3}-{hi:<3}min: {len(b):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net P&L: Rs {pnl:>+10,.0f}")

# Entry time analysis
print(f"\n\nENTRY TIME vs OUTCOME")
print("=" * 110)
time_buckets = [('09:00','10:00'), ('10:00','11:00'), ('11:00','12:00'), ('12:00','13:00'), ('13:00','14:00'), ('14:00','15:00')]
for lo, hi in time_buckets:
    b = [d for d in gmm_exits if lo <= (d.get('entry_time','')[11:16] or '00:00') < hi]
    if not b: continue
    w = sum(1 for d in b if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in b if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in b)
    wr = w/len(b)*100 if b else 0
    print(f"  Entry {lo}-{hi}: {len(b):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net P&L: Rs {pnl:>+10,.0f}")

# Winners vs Losers comparison
winners = [d for d in gmm_exits if (d.get('pnl',0) or 0) > 0]
losers = [d for d in gmm_exits if (d.get('pnl',0) or 0) < 0]

print(f"\n\nWINNER vs LOSER PROFILE")
print("=" * 110)
if winners:
    avg_score_w = sum(d.get('smart_score',0) or 0 for d in winners) / len(winners)
    avg_gate_w = sum(d.get('_gate',0) for d in winners) / len(winners)
    avg_dr_w = sum(d.get('dr_score',0) or 0 for d in winners) / len(winners)
    avg_hold_w = sum(d.get('hold_minutes',0) or 0 for d in winners) / len(winners)
    print(f"  WINNERS ({len(winners)}): Avg Score={avg_score_w:.1f} | Avg Gate={avg_gate_w:.3f} | Avg DR={avg_dr_w:.3f} | Avg Hold={avg_hold_w:.0f}min")

if losers:
    avg_score_l = sum(d.get('smart_score',0) or 0 for d in losers) / len(losers)
    avg_gate_l = sum(d.get('_gate',0) for d in losers) / len(losers)
    avg_dr_l = sum(d.get('dr_score',0) or 0 for d in losers) / len(losers)
    avg_hold_l = sum(d.get('hold_minutes',0) or 0 for d in losers) / len(losers)
    print(f"  LOSERS  ({len(losers)}): Avg Score={avg_score_l:.1f} | Avg Gate={avg_gate_l:.3f} | Avg DR={avg_dr_l:.3f} | Avg Hold={avg_hold_l:.0f}min")

# Simulated filters
print(f"\n\nFILTER SIMULATION — What if we applied score floors?")
print("=" * 110)
for floor in [25, 30, 35, 40, 45, 50]:
    passed = [d for d in gmm_exits if (d.get('smart_score',0) or 0) >= floor]
    if not passed: continue
    w = sum(1 for d in passed if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in passed if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in passed)
    wr = w/len(passed)*100 if passed else 0
    blocked = len(gmm_exits) - len(passed)
    blocked_pnl = sum(d.get('pnl',0) or 0 for d in gmm_exits if (d.get('smart_score',0) or 0) < floor)
    print(f"  Score >= {floor}: {len(passed):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net: Rs {pnl:>+10,.0f} | Blocked: {blocked} trades (P&L: Rs {blocked_pnl:>+10,.0f})")

# Gate probability filter
print(f"\n\nFILTER SIMULATION — Gate probability floors?")
print("=" * 110)
for floor in [0.52, 0.55, 0.58, 0.60, 0.65]:
    passed = [d for d in gmm_exits if d['_gate'] >= floor]
    if not passed: continue
    w = sum(1 for d in passed if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in passed if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in passed)
    wr = w/len(passed)*100 if passed else 0
    blocked = len(gmm_exits) - len(passed)
    blocked_pnl = sum(d.get('pnl',0) or 0 for d in gmm_exits if d['_gate'] < floor)
    print(f"  Gate >= {floor:.2f}: {len(passed):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net: Rs {pnl:>+10,.0f} | Blocked: {blocked} trades (P&L: Rs {blocked_pnl:>+10,.0f})")

# Combined filter
print(f"\n\nCOMBINED FILTER SIMULATION")
print("=" * 110)
combos = [(30, 0.55), (35, 0.55), (40, 0.55), (35, 0.58), (40, 0.58), (40, 0.60), (45, 0.55), (45, 0.58)]
for sf, gf in combos:
    passed = [d for d in gmm_exits if (d.get('smart_score',0) or 0) >= sf and d['_gate'] >= gf]
    if not passed: continue
    w = sum(1 for d in passed if (d.get('pnl',0) or 0) > 0)
    l = sum(1 for d in passed if (d.get('pnl',0) or 0) < 0)
    pnl = sum(d.get('pnl',0) or 0 for d in passed)
    wr = w/len(passed)*100 if passed else 0
    blocked = len(gmm_exits) - len(passed)
    print(f"  Score>={sf} + Gate>={gf:.2f}: {len(passed):>2} trades | {w}W/{l}L | WR={wr:>5.1f}% | Net: Rs {pnl:>+10,.0f} | Blocked: {blocked}")

print(f"\n{'=' * 110}")
print(f"TOTAL TEST_GMM: {len(gmm_exits)} trades | Net P&L: Rs {total_pnl:,.0f}")
print(f"{'=' * 110}")
