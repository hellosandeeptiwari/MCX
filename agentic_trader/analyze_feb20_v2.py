"""Feb 20, 2026 — Full Trade Report with Partial Exit Merging"""
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

with open('trade_history.json') as f:
    trades = json.load(f)
with open('scan_decisions.json') as f:
    scans = json.load(f)

feb20 = [t for t in trades if t.get('timestamp','').startswith('2026-02-20')]
placed = [s for s in scans if s.get('timestamp','').startswith('2026-02-20') and 'PLACED' in s.get('outcome','')]

# --- Step 1: Group trade_history by order_id to merge partial exits ---
order_groups = defaultdict(list)
for t in feb20:
    oid = t.get('order_id','')
    order_groups[oid].append(t)

merged_trades = []
for oid, legs in order_groups.items():
    base = legs[0]
    total_pnl = sum(l.get('pnl', 0) or 0 for l in legs)
    exit_types = []
    for l in legs:
        et = l.get('exit_type') or ''
        if not et:
            ed = l.get('exit_detail', {})
            if isinstance(ed, dict):
                et = ed.get('exit_type', '')
        exit_types.append(et)
    final_exit = exit_types[-1] if exit_types else ''
    # If we have a non-partial as final, use it; else pick last
    for et in reversed(exit_types):
        if et and 'PARTIAL' not in et.upper():
            final_exit = et
            break

    merged_trades.append({
        'order_id': oid,
        'symbol': base.get('symbol',''),
        'underlying': base.get('underlying',''),
        'direction': base.get('direction',''),
        'timestamp': base.get('timestamp',''),
        'pnl': total_pnl,
        'exit_type': final_exit,
        'legs': len(legs),
        'quantity': base.get('quantity', 0),
        'entry_price': base.get('entry_price'),
        'exit_detail': base.get('exit_detail', {}),
    })

print(f'Trade history records for Feb 20: {len(feb20)}')
print(f'Merged trades (unique order_ids): {len(merged_trades)}')
print(f'Total PnL (merged): Rs {sum(t["pnl"] for t in merged_trades):+,.0f}')

# --- Step 2: Match scan_decisions to merged trades ---
import re

def parse_ts(s):
    try: return datetime.fromisoformat(s)
    except: return None

def parse_reason(reason_str):
    """Extract score, dr, gate from reason string like 'Smart pick: score=69.9, dr=0.000, gate=0.87, ...'"""
    info = {}
    if not reason_str:
        return info
    m = re.search(r'score=([\d.]+)', reason_str)
    if m: info['smart_score'] = float(m.group(1))
    m = re.search(r'dr=([\d.]+)', reason_str)
    if m: info['dr_score'] = float(m.group(1))
    m = re.search(r'gate=([\d.]+)', reason_str)
    if m: info['gate_prob'] = float(m.group(1))
    m = re.search(r'gmm_action=(\w+)', reason_str)
    if m: info['gmm_action'] = m.group(1)
    m = re.search(r'sector=(\w+)', reason_str)
    if m: info['sector'] = m.group(1)
    return info

for mt in merged_trades:
    mt['source'] = 'UNKNOWN'
    mt['smart_score'] = None
    mt['pre_score'] = None
    mt['dr_score'] = None
    mt['gate_prob'] = None
    mt['sector'] = None
    mt_ts = parse_ts(mt['timestamp'])
    mt_und = mt['underlying']  # e.g. "NSE:ABB"
    mt_dir = mt['direction']

    best_match = None
    best_delta = timedelta(seconds=120)

    for s in placed:
        # scan_decisions uses 'symbol' field (e.g. "NSE:AUBANK"), trade_history uses 'underlying'
        s_sym = s.get('symbol','')
        s_dir = s.get('direction','')
        s_ts = parse_ts(s.get('timestamp',''))
        if not s_ts: continue
        if s_sym == mt_und and s_dir == mt_dir:
            delta = abs(mt_ts - s_ts)
            if delta < best_delta:
                best_delta = delta
                best_match = s

    if best_match:
        outcome = best_match.get('outcome','')
        setup = best_match.get('setup', '')
        if 'SNIPER' in outcome or setup == 'GMM_SNIPER':
            mt['source'] = 'SNIPER'
        elif 'MODEL_TRACKER' in outcome or setup == 'MODEL_TRACKER':
            mt['source'] = 'MODEL_TRACKER'
        elif 'GMM_BOOST' in outcome or setup == 'GMM_BOOST':
            mt['source'] = 'GMM_BOOST'
        else:
            mt['source'] = setup or outcome
        mt['pre_score'] = best_match.get('score')
        # Parse reason string for detailed scores
        reason_info = parse_reason(best_match.get('reason', ''))
        mt['smart_score'] = reason_info.get('smart_score', best_match.get('score'))
        mt['dr_score'] = reason_info.get('dr_score')
        mt['gate_prob'] = reason_info.get('gate_prob')
        mt['sector'] = reason_info.get('sector')

# Sort by timestamp
merged_trades.sort(key=lambda x: x['timestamp'])

# --- Step 3: Print report ---
total_pnl = sum(t['pnl'] for t in merged_trades)
winners = [t for t in merged_trades if t['pnl'] > 0]
losers = [t for t in merged_trades if t['pnl'] <= 0]

print()
print('=' * 130)
print(f'  TITAN TRADE REPORT  —  February 20, 2026')
print('=' * 130)
print(f'  Total Unique Trades: {len(merged_trades)}  |  Winners: {len(winners)}  |  Losers: {len(losers)}  |  Win Rate: {len(winners)/len(merged_trades)*100:.1f}%')
print(f'  Total PnL: Rs {total_pnl:+,.0f}')
if winners:
    avg_w = sum(w['pnl'] for w in winners) / len(winners)
    print(f'  Avg Winner: Rs {avg_w:+,.0f}', end='')
if losers:
    avg_l = sum(l['pnl'] for l in losers) / len(losers)
    print(f'  |  Avg Loser: Rs {avg_l:+,.0f}')
else:
    print()
print('=' * 130)

# Source breakdown
source_stats = defaultdict(lambda: {'count':0, 'wins':0, 'pnl':0})
for t in merged_trades:
    s = t['source']
    source_stats[s]['count'] += 1
    source_stats[s]['pnl'] += t['pnl']
    if t['pnl'] > 0:
        source_stats[s]['wins'] += 1

print()
print('  SOURCE BREAKDOWN:')
print(f'  {"Source":>15} | {"Count":>5} | {"Wins":>4} | {"WR%":>6} | {"PnL":>14}')
print(f'  {"-"*15}+{"-"*7}+{"-"*6}+{"-"*8}+{"-"*15}')
for src in ['GMM_BOOST','SNIPER','MODEL_TRACKER','UNKNOWN']:
    st = source_stats.get(src)
    if st and st['count'] > 0:
        wr = st['wins']/st['count']*100
        print(f'  {src:>15} | {st["count"]:>5} | {st["wins"]:>4} | {wr:>5.1f}% | Rs {st["pnl"]:>+11,.0f}')

print()
print('  TRADE DETAILS:')
print(f'  {"#":>3} | {"Time":>8} | {"Underlying":>14} | {"Dir":>4} | {"Score":>5} | {"DR":>6} | {"Source":>14} | {"PnL":>11} | {"Exit Type":>15} | {"Legs":>4}')
print(f'  {"-"*3}+{"-"*10}+{"-"*16}+{"-"*6}+{"-"*7}+{"-"*8}+{"-"*16}+{"-"*13}+{"-"*17}+{"-"*6}')
for i, t in enumerate(merged_trades, 1):
    ts = t['timestamp'][11:19] if len(t['timestamp']) > 19 else ''
    und = t['underlying'].replace('NSE:','')[:14]
    sc = f"{t['smart_score']:.0f}" if t['smart_score'] is not None else '-'
    dr = f"{t['dr_score']:.3f}" if t['dr_score'] is not None else '-'
    src = t['source'][:14]
    ext = (t['exit_type'] or '-')[:15]
    print(f'  {i:>3} | {ts:>8} | {und:>14} | {t["direction"]:>4} | {sc:>5} | {dr:>6} | {src:>14} | Rs {t["pnl"]:>+9,.0f} | {ext:>15} | {t["legs"]:>4}')

print('=' * 130)

# Exit type breakdown
print()
exit_counts = Counter()
for t in merged_trades:
    exit_counts[t['exit_type'] or 'UNKNOWN'] += 1
print('  EXIT TYPE BREAKDOWN:')
for et, c in exit_counts.most_common():
    pnl_et = sum(t['pnl'] for t in merged_trades if (t['exit_type'] or 'UNKNOWN') == et)
    print(f'    {et}: {c} trades, PnL Rs {pnl_et:+,.0f}')

# Score distribution
print()
scored = [t for t in merged_trades if t['smart_score'] is not None]
if scored:
    print('  SCORE DISTRIBUTION:')
    brackets = [(0,59,'<60 (Block)'),(60,63,'60-63'),(64,67,'64-67 (Std)'),(68,100,'68+ (Premium)')]
    for lo, hi, label in brackets:
        in_bracket = [t for t in scored if lo <= (t['smart_score'] or 0) <= hi]
        if in_bracket:
            w = len([t for t in in_bracket if t['pnl'] > 0])
            pnl_b = sum(t['pnl'] for t in in_bracket)
            wr = w/len(in_bracket)*100
            print(f'    {label}: {len(in_bracket)} trades, W:{w} L:{len(in_bracket)-w}, WR:{wr:.0f}%, PnL: Rs {pnl_b:+,.0f}')
