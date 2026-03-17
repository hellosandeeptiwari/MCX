import json, os
from collections import defaultdict

rows = []
for f in ['trade_ledger/trade_ledger_2026-03-16.jsonl','trade_ledger/trade_ledger_2026-03-17.jsonl']:
    if not os.path.exists(f):
        continue
    for line in open(f):
        t = json.loads(line.strip())
        if t.get('setup_type') != 'TEST_GMM':
            continue
        ml = t.get('ml_data', {}) or {}
        gmm = ml.get('gmm_model', {}) or {}
        up = gmm.get('up_score', ml.get('up_score', 0))
        dn = gmm.get('down_score', ml.get('down_score', 0))
        div = gmm.get('divergence_score', 0)
        pnl = t.get('pnl', 0)
        side = gmm.get('gmm_action', '')
        sym = t.get('underlying', '').replace('NSE:', '')
        entry = t.get('entry_time', '')[:16]
        smart = ml.get('smart_score', ml.get('p_score', 0))
        rows.append((sym, side, up, dn, div, pnl, entry, smart))

rows.sort(key=lambda x: x[4], reverse=True)

print(f"{'SYM':<14} {'SIDE':<22} {'UP':>6} {'DN':>6} {'DIV':>6} {'PNL':>9} {'SMART':>6} {'ENTRY':<16}")
print('-' * 90)
for r in rows:
    w = 'WIN' if r[5] > 0 else 'LOSS'
    print(f"{r[0]:<14} {r[1]:<22} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>6.3f} {r[5]:>+9.0f} {r[7]:>6.1f} {r[6]:<16} {w}")

# Bucket analysis by divergence gap
print("\n=== DIVERGENCE GAP BUCKETS ===")
buckets = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
for r in rows:
    d = r[4]
    if d < 0.15:
        b = '<0.15'
    elif d < 0.18:
        b = '0.15-0.18'
    elif d < 0.20:
        b = '0.18-0.20'
    elif d < 0.22:
        b = '0.20-0.22'
    elif d < 0.25:
        b = '0.22-0.25'
    elif d < 0.28:
        b = '0.25-0.28'
    else:
        b = '0.28+'
    if r[5] > 0:
        buckets[b]['w'] += 1
    else:
        buckets[b]['l'] += 1
    buckets[b]['pnl'] += r[5]

print(f"{'BUCKET':<12} {'W':>3} {'L':>3} {'TOT':>4} {'WR%':>6} {'PNL':>10}")
print('-' * 45)
for b in ['<0.15', '0.15-0.18', '0.18-0.20', '0.20-0.22', '0.22-0.25', '0.25-0.28', '0.28+']:
    if b not in buckets:
        continue
    d = buckets[b]
    tot = d['w'] + d['l']
    wr = d['w'] / tot * 100 if tot else 0
    print(f"{b:<12} {d['w']:>3} {d['l']:>3} {tot:>4} {wr:>5.1f}% {d['pnl']:>+10.0f}")

# Cumulative from top
print("\n=== CUMULATIVE (div >= threshold) ===")
thresholds = [0.13, 0.15, 0.17, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28]
for th in thresholds:
    w = sum(1 for r in rows if r[4] >= th and r[5] > 0)
    l = sum(1 for r in rows if r[4] >= th and r[5] <= 0)
    pnl = sum(r[5] for r in rows if r[4] >= th)
    tot = w + l
    if tot == 0:
        continue
    wr = w / tot * 100
    print(f"div >= {th:.2f}: {w}W/{l}L  WR={wr:>5.1f}%  PNL={pnl:>+10.0f}  (n={tot})")

# Cumulative from bottom (div <= threshold)
print("\n=== CUMULATIVE (div <= threshold) ===")
for th in thresholds:
    w = sum(1 for r in rows if r[4] <= th and r[5] > 0)
    l = sum(1 for r in rows if r[4] <= th and r[5] <= 0)
    pnl = sum(r[5] for r in rows if r[4] <= th)
    tot = w + l
    if tot == 0:
        continue
    wr = w / tot * 100
    print(f"div <= {th:.2f}: {w}W/{l}L  WR={wr:>5.1f}%  PNL={pnl:>+10.0f}  (n={tot})")

# By side (CE vs PE)
print("\n=== BY SIDE ===")
for side_key in ['PUT', 'CALL']:
    subset = [r for r in rows if side_key in r[1]]
    sw = sum(1 for r in subset if r[5] > 0)
    sl = sum(1 for r in subset if r[5] <= 0)
    sp = sum(r[5] for r in subset)
    tot = sw + sl
    wr = sw / tot * 100 if tot else 0
    print(f"\n{side_key}: {sw}W/{sl}L  WR={wr:.0f}%  PNL={sp:+.0f}")
    # Divergence buckets per side
    for th in [0.15, 0.18, 0.20, 0.22, 0.25]:
        w2 = sum(1 for r in subset if r[4] >= th and r[5] > 0)
        l2 = sum(1 for r in subset if r[4] >= th and r[5] <= 0)
        p2 = sum(r[5] for r in subset if r[4] >= th)
        t2 = w2 + l2
        if t2 == 0: continue
        wr2 = w2 / t2 * 100
        print(f"  div>={th:.2f}: {w2}W/{l2}L WR={wr2:.0f}% PNL={p2:+.0f}")

# Interaction: divergence + smart_score
print("\n=== DIV x SMART SCORE INTERACTION ===")
for div_th in [0.15, 0.18, 0.20, 0.22]:
    for smart_th in [20, 30, 40, 45]:
        w = sum(1 for r in rows if r[4] >= div_th and r[7] >= smart_th and r[5] > 0)
        l = sum(1 for r in rows if r[4] >= div_th and r[7] >= smart_th and r[5] <= 0)
        pnl = sum(r[5] for r in rows if r[4] >= div_th and r[7] >= smart_th)
        tot = w + l
        if tot < 2: continue
        wr = w / tot * 100
        print(f"div>={div_th:.2f} + smart>={smart_th}: {w}W/{l}L WR={wr:.0f}% PNL={pnl:+.0f} (n={tot})")
