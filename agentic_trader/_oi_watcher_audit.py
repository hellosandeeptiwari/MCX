import json, glob, os

# Read all ledger files
entries = {}  # order_id -> entry record
exits = {}   # order_id -> exit record

for f in sorted(glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-*.jsonl')):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line or 'OI_WATCHER' not in line:
                continue
            try:
                t = json.loads(line)
            except:
                continue
            event = t.get('event', '')
            if event == 'ENTRY' and t.get('source') == 'OI_WATCHER':
                oid = t.get('order_id', '')
                entries[oid] = t
            elif event == 'EXIT':
                oid = t.get('order_id', '')
                exits[oid] = t

print(f'=== OI_WATCHER TRADE HISTORY ===')
print(f'Entries: {len(entries)} | Exits: {len(exits)}')
print()

total_pnl = 0
wins = 0
losses = 0

for oid, entry in sorted(entries.items(), key=lambda x: x[1].get('ts', '')):
    sym = entry.get('underlying', '') or entry.get('symbol', '')
    ep = entry.get('entry_price', 0)
    direction = entry.get('direction', '')
    lots = entry.get('lots', 0)
    qty = entry.get('quantity', 0)
    oi_sig = entry.get('oi_signal', '')
    rat = entry.get('rationale', '')
    et = entry.get('ts', '')[:19]
    score = entry.get('smart_score', 0)
    opt_sym = entry.get('option_symbol', '')
    
    ex = exits.get(oid, {})
    if ex:
        xp = ex.get('exit_price', 0)
        pnl = ex.get('pnl', 0)
        xr = ex.get('exit_type', '') or ex.get('reason', '')
        held = ex.get('held_minutes', '-')
        xt = ex.get('ts', '')[:19]
        pp = ((xp - ep) / ep * 100) if ep and xp else 0
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        
        wl = 'W' if pnl > 0 else 'L'
        print(f'{et} | {sym:15s} | {direction:4s} | {oi_sig:16s} | e={ep:7.2f} x={xp:7.2f} | {pp:+6.1f}% | PnL={pnl:+8,.0f} | held={held}min | {xr} | {wl}')
    else:
        print(f'{et} | {sym:15s} | {direction:4s} | {oi_sig:16s} | e={ep:7.2f} | STILL OPEN | score={score}')

print(f'\n=== SUMMARY ===')
print(f'Total PnL: Rs{total_pnl:+,.0f}')
print(f'Wins: {wins} | Losses: {losses} | Win%: {wins/(wins+losses)*100:.0f}%' if (wins+losses) > 0 else 'No closed trades')
print(f'Avg PnL: Rs{total_pnl/(wins+losses):+,.0f}' if (wins+losses) > 0 else '')

# Group by exit reason
from collections import Counter
exit_reasons = Counter()
exit_pnl = {}
for oid, ex in exits.items():
    xr = ex.get('exit_type', '') or ex.get('reason', '')
    pnl = ex.get('pnl', 0)
    exit_reasons[xr] += 1
    exit_pnl[xr] = exit_pnl.get(xr, 0) + pnl

print(f'\n=== EXIT REASON BREAKDOWN ===')
for reason, count in exit_reasons.most_common():
    print(f'  {reason:45s} count={count:2d} PnL={exit_pnl.get(reason,0):+8,.0f}')
