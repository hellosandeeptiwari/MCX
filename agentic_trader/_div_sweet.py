import json, os, glob, re
from collections import defaultdict

entries = {}  # order_id -> entry data
exits = {}    # order_id -> exit data

for f in sorted(glob.glob('trade_ledger/trade_ledger_2026-03-*.jsonl')):
    for line in open(f):
        t = json.loads(line.strip())
        ev = t.get('event', '')
        oid = t.get('order_id', '')
        src = t.get('source', '')
        
        if ev == 'ENTRY' and src == 'TEST_GMM' and oid:
            # Parse divergence from rationale
            rat = t.get('rationale', '')
            m = re.search(r'UP=([\d.]+)\s+DN=([\d.]+)\s+gap=([\d.]+)\s+gate=([\d.]+)', rat)
            if m:
                t['_up'] = float(m.group(1))
                t['_dn'] = float(m.group(2))
                t['_gap'] = float(m.group(3))
                t['_gate'] = float(m.group(4))
            side = 'PUT' if 'PUT' in rat else 'CALL' if 'CALL' in rat else '?'
            t['_side'] = side
            entries[oid] = t
        
        if ev == 'EXIT' and (src == 'TEST_GMM' or t.get('setup') == 'TEST_GMM') and oid:
            exits[oid] = t

# Also check for partial exits - group by order_id
exit_groups = defaultdict(list)
for f in sorted(glob.glob('trade_ledger/trade_ledger_2026-03-*.jsonl')):
    for line in open(f):
        t = json.loads(line.strip())
        if t.get('event') == 'EXIT' and (t.get('source') == 'TEST_GMM' or t.get('setup') == 'TEST_GMM'):
            oid = t.get('order_id', '')
            if oid:
                exit_groups[oid].append(t)

# Build matched trades
trades = []
for oid, entry in entries.items():
    if oid not in exit_groups:
        continue
    total_pnl = sum(e.get('pnl', 0) for e in exit_groups[oid])
    last_exit = max(exit_groups[oid], key=lambda e: e.get('ts', ''))
    trades.append({
        'sym': entry.get('underlying', '').replace('NSE:', ''),
        'side': entry.get('_side', '?'),
        'up': entry.get('_up', 0),
        'dn': entry.get('_dn', 0),
        'gap': entry.get('_gap', 0),
        'gate': entry.get('_gate', 0),
        'smart': entry.get('smart_score', 0),
        'pnl': total_pnl,
        'entry_time': entry.get('ts', '')[:16],
        'exit_type': last_exit.get('exit_type', ''),
        'hold_min': last_exit.get('hold_minutes', 0),
        'date': entry.get('ts', '')[:10],
    })

trades.sort(key=lambda x: x['gap'], reverse=True)

print(f"Matched {len(trades)} TEST_GMM trades with P&L (from {len(entries)} entries, {len(exit_groups)} exit groups)")
print(f"\n{'SYM':<14} {'SIDE':<6} {'UP':>6} {'DN':>6} {'GAP':>6} {'GATE':>5} {'SMART':>6} {'PNL':>9} {'HOLD':>5} {'EXIT':<16} {'DATE'}")
print('-' * 105)
for t in trades:
    w = 'WIN' if t['pnl'] > 0 else 'LOSS'
    print(f"{t['sym']:<14} {t['side']:<6} {t['up']:>6.3f} {t['dn']:>6.3f} {t['gap']:>6.3f} {t['gate']:>5.2f} {t['smart']:>6.1f} {t['pnl']:>+9.0f} {t['hold_min']:>5} {t['exit_type']:<16} {t['date']} {w}")

# === DIVERGENCE GAP BUCKETS ===
print(f"\n{'='*60}")
print("DIVERGENCE GAP BUCKETS")
print(f"{'='*60}")
buckets: defaultdict[str, dict[str, int | float]] = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
for t in trades:
    g = t['gap']
    if g < 0.15: b = '<0.15'
    elif g < 0.17: b = '0.15-0.17'
    elif g < 0.19: b = '0.17-0.19'
    elif g < 0.21: b = '0.19-0.21'
    elif g < 0.23: b = '0.21-0.23'
    elif g < 0.25: b = '0.23-0.25'
    elif g < 0.27: b = '0.25-0.27'
    elif g < 0.29: b = '0.27-0.29'
    else: b = '0.29+'
    if t['pnl'] > 0: buckets[b]['w'] += 1
    else: buckets[b]['l'] += 1
    buckets[b]['pnl'] += t['pnl']

print(f"\n{'BUCKET':<12} {'W':>3} {'L':>3} {'TOT':>4} {'WR%':>6} {'PNL':>10} {'AVG':>8}")
print('-' * 55)
order = ['<0.15', '0.15-0.17', '0.17-0.19', '0.19-0.21', '0.21-0.23', '0.23-0.25', '0.25-0.27', '0.27-0.29', '0.29+']
for b in order:
    if b not in buckets: continue
    d = buckets[b]
    tot = d['w'] + d['l']
    wr = d['w'] / tot * 100 if tot else 0
    avg = d['pnl'] / tot if tot else 0
    marker = ' <<<' if wr >= 50 and tot >= 2 else ''
    print(f"{b:<12} {d['w']:>3} {d['l']:>3} {tot:>4} {wr:>5.1f}% {d['pnl']:>+10.0f} {avg:>+8.0f}{marker}")

# === CUMULATIVE FROM MINIMUM ===
print(f"\n{'='*60}")
print("CUMULATIVE: div >= threshold")
print(f"{'='*60}")
for th in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28]:
    w = sum(1 for t in trades if t['gap'] >= th and t['pnl'] > 0)
    l = sum(1 for t in trades if t['gap'] >= th and t['pnl'] <= 0)
    pnl = sum(t['pnl'] for t in trades if t['gap'] >= th)
    tot = w + l
    if tot == 0: continue
    wr = w / tot * 100
    avg = pnl / tot
    marker = ' <<<' if wr >= 50 and tot >= 3 else ''
    print(f"  gap >= {th:.2f}: {w:>2}W/{l:>2}L  WR={wr:>5.1f}%  PNL={pnl:>+10.0f}  avg={avg:>+6.0f}  (n={tot}){marker}")

# === CUMULATIVE FROM MAXIMUM ===
print(f"\n{'='*60}")
print("CUMULATIVE: div <= threshold")
print(f"{'='*60}")
for th in [0.15, 0.17, 0.19, 0.20, 0.21, 0.22, 0.23, 0.25, 0.27, 0.29]:
    w = sum(1 for t in trades if t['gap'] <= th and t['pnl'] > 0)
    l = sum(1 for t in trades if t['gap'] <= th and t['pnl'] <= 0)
    pnl = sum(t['pnl'] for t in trades if t['gap'] <= th)
    tot = w + l
    if tot == 0: continue
    wr = w / tot * 100
    avg = pnl / tot
    print(f"  gap <= {th:.2f}: {w:>2}W/{l:>2}L  WR={wr:>5.1f}%  PNL={pnl:>+10.0f}  avg={avg:>+6.0f}  (n={tot})")

# === BY SIDE (CE vs PE) ===
print(f"\n{'='*60}")
print("BY SIDE: PUT vs CALL")
print(f"{'='*60}")
for side in ['PUT', 'CALL']:
    subset = [t for t in trades if t['side'] == side]
    sw = sum(1 for t in subset if t['pnl'] > 0)
    sl = sum(1 for t in subset if t['pnl'] <= 0)
    sp = sum(t['pnl'] for t in subset)
    tot = sw + sl
    wr = sw / tot * 100 if tot else 0
    print(f"\n{side}: {sw}W/{sl}L  WR={wr:.0f}%  PNL={sp:+.0f}  (n={tot})")
    for th in [0.15, 0.18, 0.20, 0.22, 0.25]:
        w2 = sum(1 for t in subset if t['gap'] >= th and t['pnl'] > 0)
        l2 = sum(1 for t in subset if t['gap'] >= th and t['pnl'] <= 0)
        p2 = sum(t['pnl'] for t in subset if t['gap'] >= th)
        t2 = w2 + l2
        if t2 == 0: continue
        wr2 = w2 / t2 * 100
        print(f"  gap>={th:.2f}: {w2}W/{l2}L WR={wr2:.0f}% PNL={p2:+.0f} (n={t2})")

# === INTERACTION: GAP x SMART x GATE ===
print(f"\n{'='*60}")
print("INTERACTIONS: div + smart + gate")
print(f"{'='*60}")
for div_th in [0.15, 0.18, 0.20, 0.22]:
    for smart_th in [20, 30, 40, 45]:
        w = sum(1 for t in trades if t['gap'] >= div_th and t['smart'] >= smart_th and t['pnl'] > 0)
        l = sum(1 for t in trades if t['gap'] >= div_th and t['smart'] >= smart_th and t['pnl'] <= 0)
        pnl = sum(t['pnl'] for t in trades if t['gap'] >= div_th and t['smart'] >= smart_th)
        tot = w + l
        if tot < 2: continue
        wr = w / tot * 100
        marker = ' <<<' if wr >= 50 else ''
        print(f"  gap>={div_th:.2f} + smart>={smart_th}: {w}W/{l}L WR={wr:.0f}% PNL={pnl:+.0f} (n={tot}){marker}")

# Gate prob interaction
print()
for div_th in [0.15, 0.18, 0.20]:
    for gate_th in [0.40, 0.50, 0.55, 0.60]:
        w = sum(1 for t in trades if t['gap'] >= div_th and t['gate'] >= gate_th and t['pnl'] > 0)
        l = sum(1 for t in trades if t['gap'] >= div_th and t['gate'] >= gate_th and t['pnl'] <= 0)
        pnl = sum(t['pnl'] for t in trades if t['gap'] >= div_th and t['gate'] >= gate_th)
        tot = w + l
        if tot < 2: continue
        wr = w / tot * 100
        marker = ' <<<' if wr >= 50 else ''
        print(f"  gap>={div_th:.2f} + gate>={gate_th:.2f}: {w}W/{l}L WR={wr:.0f}% PNL={pnl:+.0f} (n={tot}){marker}")

# === SWEET SPOT SUMMARY ===
print(f"\n{'='*60}")
print("SWEET SPOT SUMMARY")
print(f"{'='*60}")
# Find the gap range with best WR% * trade count
best = None
best_score = -999999
for lo in [0.13, 0.15, 0.17, 0.19, 0.20, 0.21, 0.22]:
    for hi in [0.25, 0.27, 0.29, 0.35]:
        if lo >= hi: continue
        subset = [t for t in trades if lo <= t['gap'] <= hi]
        w = sum(1 for t in subset if t['pnl'] > 0)
        l = sum(1 for t in subset if t['pnl'] <= 0)
        tot = w + l
        if tot < 3: continue
        wr = w / tot * 100
        pnl = sum(t['pnl'] for t in subset)
        score = pnl  # optimize for total PNL
        if score > best_score:
            best_score = score
            best = (lo, hi, w, l, tot, wr, pnl)

if best:
    lo, hi, w, l, tot, wr, pnl = best
    print(f"\nBest gap range: [{lo:.2f}, {hi:.2f}]")
    print(f"  {w}W/{l}L  WR={wr:.1f}%  PNL={pnl:+.0f}  (n={tot})")
