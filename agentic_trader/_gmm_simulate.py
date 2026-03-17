#!/usr/bin/env python3
"""Simulate various TEST_GMM threshold combinations to find optimal."""
import json, os, re, glob

ledger_dir = os.path.join(os.path.dirname(__file__), 'trade_ledger')
files = sorted(glob.glob(os.path.join(ledger_dir, 'trade_ledger_*.jsonl')))

trades = []
for fpath in files:
    day = os.path.basename(fpath).replace('trade_ledger_', '').replace('.jsonl', '')
    entries = {}
    exits = {}
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
            sym = rec.get('underlying', '')
            if rec.get('event') == 'ENTRY':
                entries[sym] = rec
            elif rec.get('event') == 'EXIT':
                exits[sym] = rec
    
    for sym, e in entries.items():
        exit_rec = exits.get(sym, {})
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
            'up': float(up_m.group(1)) if up_m else 0,
            'dn': float(dn_m.group(1)) if dn_m else 0,
            'gap': float(gap_m.group(1)) if gap_m else 0,
            'pnl': pnl,
            'gate_prob': e.get('gate_prob', e.get('ml_move_prob', 0)),
            'smart_score': e.get('smart_score', e.get('pre_score', 0)),
        })

completed = [t for t in trades if True]  # all have exits since we matched
print(f"Total completed TEST_GMM trades: {len(completed)}")
w_count = sum(1 for t in completed if t['pnl'] >= 0)
l_count = sum(1 for t in completed if t['pnl'] < 0)
total_pnl = sum(t['pnl'] for t in completed)
print(f"Baseline: {w_count}W/{l_count}L, PnL={total_pnl:+,.0f}")

# Simulate different filter combos
print(f"\n{'='*80}")
print("SIMULATION RESULTS:")
print(f"{'Filter':<55s} {'W':>3s} {'L':>3s} {'Win%':>5s} {'Net PnL':>10s} {'AvgW':>8s} {'AvgL':>8s}")
print(f"{'-'*80}")

def sim(label, filt):
    passed = [t for t in completed if filt(t)]
    w = [t for t in passed if t['pnl'] >= 0]
    l = [t for t in passed if t['pnl'] < 0]
    net = sum(t['pnl'] for t in passed)
    wr = len(w)/(len(w)+len(l))*100 if (w or l) else 0
    avg_w = sum(t['pnl'] for t in w)/len(w) if w else 0
    avg_l = sum(t['pnl'] for t in l)/len(l) if l else 0
    print(f"{label:<55s} {len(w):>3d} {len(l):>3d} {wr:>5.1f} {net:>+10,.0f} {avg_w:>+8,.0f} {avg_l:>+8,.0f}")

# Current (broken): DN≥0.30, gap≥0.30 (way too tight, selects losers)
sim("CURRENT: dn≥0.30 gap≥0.30", lambda t: t['dn'] >= 0.30 and t['gap'] >= 0.30)

# Baseline: no filter
sim("NO FILTER (all trades)", lambda t: True)

# Time only
sim("Time only: before 12:00", lambda t: t['time'] < '12:00')
sim("Time only: before 13:00", lambda t: t['time'] < '13:00')
sim("Time only: before 14:00", lambda t: t['time'] < '14:00')

# Smart score only
sim("Smart ≥ 15", lambda t: t['smart_score'] >= 15)
sim("Smart ≥ 20", lambda t: t['smart_score'] >= 20)
sim("Smart ≥ 25", lambda t: t['smart_score'] >= 25)

# DN cap (block over-confident)
sim("DN ≤ 0.34 (cap extreme)", lambda t: t['dn'] <= 0.34)
sim("DN ≤ 0.32", lambda t: t['dn'] <= 0.32)
sim("DN ≤ 0.30", lambda t: t['dn'] <= 0.30)

# Gap alignment
sim("gap ≤ 0.28 (winner max)", lambda t: t['gap'] <= 0.28)

# Combined: winner-aligned
sim("DN≤0.34 + smart≥20", 
    lambda t: t['dn'] <= 0.34 and t['smart_score'] >= 20)

sim("DN≤0.34 + smart≥20 + time<14:00", 
    lambda t: t['dn'] <= 0.34 and t['smart_score'] >= 20 and t['time'] < '14:00')

sim("DN≤0.34 + smart≥15 + time<14:00", 
    lambda t: t['dn'] <= 0.34 and t['smart_score'] >= 15 and t['time'] < '14:00')

sim("DN 0.24-0.34 + smart≥20", 
    lambda t: 0.24 <= t['dn'] <= 0.34 and t['smart_score'] >= 20)

sim("DN 0.24-0.34 + smart≥20 + time<14:00", 
    lambda t: 0.24 <= t['dn'] <= 0.34 and t['smart_score'] >= 20 and t['time'] < '14:00')

# The real fix: use winner's actual numbers
sim("WINNER ALIGNED: dn 0.245-0.343, gap 0.13-0.28", 
    lambda t: 0.245 <= t['dn'] <= 0.343 and 0.13 <= t['gap'] <= 0.28)

sim("WINNER ALIGNED + smart≥20", 
    lambda t: 0.245 <= t['dn'] <= 0.343 and 0.13 <= t['gap'] <= 0.28 and t['smart_score'] >= 20)

sim("WINNER ALIGNED + smart≥20 + time<14:00", 
    lambda t: 0.245 <= t['dn'] <= 0.343 and 0.13 <= t['gap'] <= 0.28 and t['smart_score'] >= 20 and t['time'] < '14:00')

# What about just capping DN?
sim("dn_max=0.35 (block over-confident)", lambda t: t['dn'] <= 0.35)
sim("dn_max=0.35 + smart≥20", lambda t: t['dn'] <= 0.35 and t['smart_score'] >= 20)

# Using floor correctly: low floor, high cap
sim("DN 0.24-0.35 + gap 0.13-0.28 + smart≥20", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and 0.13 <= t['gap'] <= 0.28 and t['smart_score'] >= 20)

sim("DN 0.24-0.35 + gap≥0.13 + smart≥20", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and t['gap'] >= 0.13 and t['smart_score'] >= 20)

sim("DN 0.24-0.35 + gap≥0.13 + smart≥20 + time<14:00", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and t['gap'] >= 0.13 and t['smart_score'] >= 20 and t['time'] < '14:00')

# Pure research: what if we just lower the floor?
sim("DN≥0.24 gap≥0.13 (original intent)", 
    lambda t: t['dn'] >= 0.24 and t['gap'] >= 0.13)

# Best combo candidates  
print(f"\n{'='*80}")
print("BEST CANDIDATES:")

sim("★ DN∈[0.24,0.35] gap∈[0.13,0.29] smart≥20", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and 0.13 <= t['gap'] <= 0.29 and t['smart_score'] >= 20)

sim("★ DN∈[0.24,0.35] smart≥20 time<14:00", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and t['smart_score'] >= 20 and t['time'] < '14:00')

sim("★ DN∈[0.24,0.35] smart≥20", 
    lambda t: 0.24 <= t['dn'] <= 0.35 and t['smart_score'] >= 20)
