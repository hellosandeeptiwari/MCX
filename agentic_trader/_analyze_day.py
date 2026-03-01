import json
from collections import Counter

entries = []
exits = []

with open('trade_ledger/trade_ledger_2026-02-25.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d['event'] == 'ENTRY':
            entries.append(d)
        elif d['event'] == 'EXIT':
            exits.append(d)

print(f'=== ENTRIES: {len(entries)} ===')
for i, e in enumerate(entries, 1):
    sniper = ' [SNIPER]' if e.get('is_sniper') else ''
    und = e['underlying']
    dirn = e['direction']
    ot = e['option_type']
    strike = e['strike']
    ep = e['entry_price']
    qty = e['quantity']
    lots = e['lots']
    prem = e['total_premium']
    sl = e['stop_loss']
    tgt = e['target']
    src = e['source']
    smart = e['smart_score']
    dr = e['dr_score']
    ts = e['ts'][11:16]
    print(f'{i:2d}. {ts} {und:20s} {dirn:4s} {ot:2s} strike={strike:>8} entry={ep:>8} qty={qty:>6} lots={lots} prem={prem:>9.0f} SL={sl:>7} tgt={tgt:>7} src={src}{sniper} smart={smart} dr={dr}')

print(f'\n=== EXITS: {len(exits)} ===')
total_pnl = 0
winners = 0
losers = 0
for i, e in enumerate(exits, 1):
    pnl = e['pnl']
    total_pnl += pnl
    if pnl > 0:
        winners += 1
    else:
        losers += 1
    hold = e.get('hold_minutes', '?')
    sniper = ' [SNIPER]' if e.get('is_sniper') else ''
    und = e.get('underlying', '?')
    dirn = e.get('direction', '?')
    ot = e.get('option_type', '?')
    ep = e.get('entry_price', 0)
    xp = e.get('exit_price', 0)
    pnl_pct = e.get('pnl_pct', 0)
    exit_type = e.get('exit_type', '?')
    src = e.get('source', '?')
    ts = e['ts'][11:16]
    print(f'{i:2d}. {ts} {und:20s} {dirn:4s} entry={ep:>8} exit={xp:>8} PnL={pnl:>+10.1f} ({pnl_pct:>+6.1f}%) hold={hold}min type={exit_type} src={src}{sniper}')

print(f'\n=== SUMMARY ===')
print(f'Total PnL: {total_pnl:,.1f}')
print(f'Winners: {winners} | Losers: {losers} | Win Rate: {winners/(winners+losers)*100:.1f}%')
if winners > 0:
    print(f'Avg Winner: {sum(e["pnl"] for e in exits if e["pnl"]>0)/winners:,.1f}')
if losers > 0:
    print(f'Avg Loser: {sum(e["pnl"] for e in exits if e["pnl"]<=0)/losers:,.1f}')

# Exit type breakdown
exit_types = Counter(e['exit_type'] for e in exits)
print(f'\nExit Type Breakdown:')
for et, cnt in exit_types.most_common():
    et_pnl = sum(e['pnl'] for e in exits if e['exit_type'] == et)
    et_wins = sum(1 for e in exits if e['exit_type'] == et and e['pnl'] > 0)
    print(f'  {et}: {cnt} trades, PnL={et_pnl:>+10,.1f}, W:{et_wins}/L:{cnt-et_wins}')

# Source breakdown
sources = Counter(e['source'] for e in exits)
print(f'\nSource Breakdown:')
for src, cnt in sources.most_common():
    src_pnl = sum(e['pnl'] for e in exits if e['source'] == src)
    src_wins = sum(1 for e in exits if e['source'] == src and e['pnl'] > 0)
    print(f'  {src}: {cnt} trades, PnL={src_pnl:>+10,.1f}, W:{src_wins}/L:{cnt-src_wins}')

# Sector breakdown
sectors = Counter(e['sector'] for e in exits)
print(f'\nSector Breakdown:')
for sec, cnt in sectors.most_common():
    sec_pnl = sum(e['pnl'] for e in exits if e['sector'] == sec)
    sec_wins = sum(1 for e in exits if e['sector'] == sec and e['pnl'] > 0)
    print(f'  {sec}: {cnt} trades, PnL={sec_pnl:>+10,.1f}, W:{sec_wins}/L:{cnt-sec_wins}')

# Average hold time
avg_hold = sum(e.get('hold_minutes', 0) for e in exits) / len(exits)
print(f'\nAvg Hold Time: {avg_hold:.0f} min')

# Sniper trades
sniper_exits = [e for e in exits if e.get('is_sniper')]
if sniper_exits:
    sniper_pnl = sum(e['pnl'] for e in sniper_exits)
    sniper_wins = sum(1 for e in sniper_exits if e['pnl'] > 0)
    print(f'\nSniper Trades: {len(sniper_exits)}, PnL={sniper_pnl:>+,.1f}, W:{sniper_wins}/L:{len(sniper_exits)-sniper_wins}')
else:
    print('\nSniper Trades: 0')

# Direction breakdown  
print(f'\nDirection Breakdown:')
for d in ['BUY', 'SELL']:
    d_exits = [e for e in exits if e['direction'] == d]
    if d_exits:
        d_pnl = sum(e['pnl'] for e in d_exits)
        d_wins = sum(1 for e in d_exits if e['pnl'] > 0)
        print(f'  {d}: {len(d_exits)} trades, PnL={d_pnl:>+10,.1f}, W:{d_wins}/L:{len(d_exits)-d_wins}')

# Best and worst trades
best = max(exits, key=lambda e: e['pnl'])
worst = min(exits, key=lambda e: e['pnl'])
print(f'\nBest Trade:  {best["underlying"]} {best["direction"]} PnL={best["pnl"]:>+,.1f} ({best["exit_type"]})')
print(f'Worst Trade: {worst["underlying"]} {worst["direction"]} PnL={worst["pnl"]:>+,.1f} ({worst["exit_type"]})')

# IV Crush analysis
iv_crush = [e for e in exits if 'IV_CRUSH' in e.get('exit_type', '')]
if iv_crush:
    ic_pnl = sum(e['pnl'] for e in iv_crush)
    print(f'\nIV Crush Exits: {len(iv_crush)} trades, PnL={ic_pnl:>+,.1f}')

# Timeline analysis - when were trades opened vs closed
print(f'\nTimeline:')
print(f'  First entry:  {entries[0]["ts"][11:16]} - {entries[0]["underlying"]}')
print(f'  Last entry:   {entries[-1]["ts"][11:16]} - {entries[-1]["underlying"]}')
print(f'  First exit:   {exits[0]["ts"][11:16]} - {exits[0]["underlying"]}')
print(f'  Last exit:    {exits[-1]["ts"][11:16]} - {exits[-1]["underlying"]}')
