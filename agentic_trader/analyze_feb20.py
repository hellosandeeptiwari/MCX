"""Analyze Feb 20 trades: scores, DR, source, PnL"""
import json, re
from collections import defaultdict
from datetime import datetime

with open('scan_decisions.json') as f:
    scans = json.load(f)
with open('trade_history.json') as f:
    trades = json.load(f)

placed_types = ['GMM_BOOST_PLACED', 'GMM_SNIPER_PLACED', 'MODEL_TRACKER_PLACED', 'ML_OVERRIDE', 'PLACED']
placed = [d for d in scans if '2026-02-20' in d.get('timestamp','') and d.get('outcome','') in placed_types]
feb20_trades = [t for t in trades if '2026-02-20' in t.get('timestamp','')]

def parse_val(reason, key):
    m = re.search(key + r'=([0-9.]+)', reason)
    return float(m.group(1)) if m else None

src_map = {
    'GMM_BOOST_PLACED': 'GMM_BOOST',
    'GMM_SNIPER_PLACED': 'SNIPER',
    'MODEL_TRACKER_PLACED': 'MODEL_TRACKER',
    'ML_OVERRIDE': 'ML_OVERRIDE',
    'PLACED': 'SCORE'
}

trade_by_und = defaultdict(list)
for t in feb20_trades:
    und = t.get('underlying', t.get('symbol',''))
    trade_by_und[und].append(t)

total_pnl = 0
wins = 0
losses = 0
source_stats = defaultdict(lambda: {'count':0, 'pnl':0, 'wins':0, 'losses':0})
all_rows = []

header = f"{'#':>2} | {'Time':>8} | {'Symbol':>14} | {'Dir':>4} | {'Score':>5} | {'Smart':>5} | {'DR':>5} | {'Gate':>4} | {'Source':>14} | {'PnL':>9} | {'Result':>15}"
print(header)
print("-" * len(header))

for i, s in enumerate(placed, 1):
    sym = s['symbol']
    ts = s['timestamp'][11:19]
    d = s.get('direction', '?')
    sc = s.get('score', 0)
    reason = s.get('reason', '')
    dr = parse_val(reason, 'dr')
    smart = parse_val(reason, 'smart') or parse_val(reason, 'score')
    gate = parse_val(reason, 'gate')
    source = src_map.get(s.get('outcome',''), s.get('outcome',''))

    matched = trade_by_und.get(sym, [])
    best = None
    for t in matched:
        try:
            st = datetime.fromisoformat(s['timestamp'])
            tt = datetime.fromisoformat(t['timestamp'])
            diff = abs((tt - st).total_seconds())
            if diff < 60 and t.get('direction','') == d:
                if best is None or diff < best[1]:
                    best = (t, diff)
        except:
            pass

    pnl = 0
    result = 'NO_MATCH'
    option_sym = ''
    if best:
        pnl = best[0].get('pnl', 0)
        result = best[0].get('result', '')[:15]
        option_sym = best[0].get('symbol', '')
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        source_stats[source]['count'] += 1
        source_stats[source]['pnl'] += pnl
        if pnl > 0:
            source_stats[source]['wins'] += 1
        else:
            source_stats[source]['losses'] += 1

    sym_s = sym.replace('NSE:', '')
    dr_s = f"{dr:.3f}" if dr is not None else "  N/A"
    sm_s = f"{smart:.1f}" if smart is not None else "  N/A"
    gt_s = f"{gate:.2f}" if gate is not None else " N/A"
    pnl_s = f"{pnl:+,.0f}" if best else "    N/M"

    print(f"{i:>2} | {ts:>8} | {sym_s:>14} | {d:>4} | {sc:>5.1f} | {sm_s:>5} | {dr_s:>5} | {gt_s:>4} | {source:>14} | {pnl_s:>9} | {result:>15}")
    all_rows.append({'sym': sym_s, 'dir': d, 'sc': sc, 'smart': smart, 'dr': dr, 'gate': gate, 'source': source, 'pnl': pnl if best else None, 'result': result})

total = wins + losses
print("-" * len(header))
print()
print(f"TOTAL ENTRIES: {len(placed)} scan decisions placed")
print(f"MATCHED TRADES: {total} | Wins: {wins} | Losses: {losses} | WR: {wins/total*100:.1f}%")
print(f"TOTAL PnL: Rs {total_pnl:+,.0f}")
print()

# Source breakdown
print("=" * 65)
print("SOURCE-WISE PERFORMANCE")
print("=" * 65)
print(f"  {'Source':>14} | {'Count':>5} | {'Wins':>4} | {'Loss':>4} | {'WR%':>5} | {'PnL':>12}")
print(f"  {'-'*14}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*12}")
for src in ['GMM_BOOST', 'SNIPER', 'MODEL_TRACKER', 'ML_OVERRIDE', 'SCORE']:
    if src in source_stats:
        s = source_stats[src]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        print(f"  {src:>14} | {s['count']:>5} | {s['wins']:>4} | {s['losses']:>4} | {wr:>5.1f} | Rs {s['pnl']:>+10,.0f}")

# Top winners and losers
print()
print("=" * 65)
print("TOP 5 WINNERS")
print("=" * 65)
matched_rows = [r for r in all_rows if r['pnl'] is not None]
sorted_win = sorted(matched_rows, key=lambda x: x['pnl'], reverse=True)[:5]
for r in sorted_win:
    print(f"  {r['sym']:>14} {r['dir']:>4} | Score={r['sc']:.1f} Smart={r['smart'] or 0:.1f} DR={r['dr'] or 0:.3f} | {r['source']:>14} | PnL=Rs {r['pnl']:+,.0f}")

print()
print("=" * 65)
print("TOP 5 LOSERS")
print("=" * 65)
sorted_loss = sorted(matched_rows, key=lambda x: x['pnl'])[:5]
for r in sorted_loss:
    print(f"  {r['sym']:>14} {r['dir']:>4} | Score={r['sc']:.1f} Smart={r['smart'] or 0:.1f} DR={r['dr'] or 0:.3f} | {r['source']:>14} | PnL=Rs {r['pnl']:+,.0f}")

# Also show unmatched trades from trade_history (no scan match)
matched_underlyings = set()
for s in placed:
    matched_underlyings.add(s['symbol'])

unmatched_trades = []
for t in feb20_trades:
    und = t.get('underlying', t.get('symbol',''))
    if und not in matched_underlyings:
        unmatched_trades.append(t)

if unmatched_trades:
    print()
    print(f"NOTE: {len(unmatched_trades)} trades in trade_history had no matching scan decision")
    for t in unmatched_trades[:5]:
        sym = t.get('underlying', t.get('symbol',''))
        print(f"  {sym} {t.get('direction','')} pnl={t.get('pnl',0):+,.0f} result={t.get('result','')}")

# Also show DR distribution
print()
print("=" * 65)
print("DR SCORE DISTRIBUTION")
print("=" * 65)
dr_vals = [r['dr'] for r in all_rows if r['dr'] is not None]
clean = len([d for d in dr_vals if d < 0.15])
flagged = len([d for d in dr_vals if d >= 0.19])
sniper_ok = len([d for d in dr_vals if d < 0.10])
print(f"  CLEAN (DR < 0.15):     {clean}/{len(dr_vals)} ({clean/len(dr_vals)*100:.0f}%)")
print(f"  FLAGGED (DR >= 0.19):  {flagged}/{len(dr_vals)} ({flagged/len(dr_vals)*100:.0f}%)")
print(f"  Sniper eligible (<0.10): {sniper_ok}/{len(dr_vals)} ({sniper_ok/len(dr_vals)*100:.0f}%)")

# PnL by DR bucket
dr_buckets = {'CLEAN (<0.15)': [], 'MID (0.15-0.19)': [], 'FLAGGED (>=0.19)': []}
for r in matched_rows:
    if r['dr'] is not None:
        if r['dr'] < 0.15:
            dr_buckets['CLEAN (<0.15)'].append(r['pnl'])
        elif r['dr'] < 0.19:
            dr_buckets['MID (0.15-0.19)'].append(r['pnl'])
        else:
            dr_buckets['FLAGGED (>=0.19)'].append(r['pnl'])

print()
print("PnL by DR bucket:")
for bucket, pnls in dr_buckets.items():
    if pnls:
        avg = sum(pnls)/len(pnls)
        total = sum(pnls)
        w = len([p for p in pnls if p > 0])
        print(f"  {bucket}: {len(pnls)} trades, WR={w/len(pnls)*100:.0f}%, Avg=Rs {avg:+,.0f}, Total=Rs {total:+,.0f}")
    else:
        print(f"  {bucket}: 0 trades")
