import json

entries = []
exits = []
conversions = []
with open('trade_ledger_2026-03-10.jsonl') as f:
    for line in f:
        d = json.loads(line.strip())
        ev = d.get('event')
        if ev == 'ENTRY':
            entries.append(d)
        elif ev == 'EXIT':
            exits.append(d)
        elif ev == 'CONVERSION':
            conversions.append(d)

print('=== ENTRIES (31) ===')
for e in entries:
    sym = e.get('underlying','').replace('NSE:','')
    src = e.get('source','')
    score = e.get('smart_score',0)
    dr = e.get('dr_score',0)
    prob = e.get('gate_prob',0) or e.get('ml_move_prob',0)
    ep = e.get('entry_price',0)
    sl = e.get('stop_loss',0)
    lots = e.get('lots',0)
    prem = e.get('total_premium',0)
    opt = e.get('option_symbol','').replace('NFO:','')
    ts = e.get('ts','')[11:19]
    d = e.get('direction','')
    print(f'{ts} {sym:14s} {d:4s} {src:15s} Score={score:5.1f} P(move)={prob:.2f} DR={dr:.3f} Entry={ep:8.2f} SL={sl:8.2f} Lots={lots} Prem={prem:>7,.0f} [{opt}]')

print()
print('=== EXITS (32) ===')
total_pnl = 0
wins = 0
losses = 0
for x in exits:
    sym = x.get('underlying','').replace('NSE:','')
    pnl = x.get('pnl',0)
    pnl_pct = x.get('pnl_pct',0)
    ep = x.get('entry_price',0)
    xp = x.get('exit_price',0)
    reason = x.get('exit_reason','')[:65]
    etype = x.get('exit_type','')
    score = x.get('smart_score',0)
    hold = x.get('hold_minutes',0)
    src = x.get('source','')
    ts = x.get('ts','')[11:19]
    total_pnl += pnl
    if pnl > 0: wins += 1
    else: losses += 1
    icon = 'W' if pnl > 0 else 'L'
    print(f'{ts} {sym:14s} {icon} PnL={pnl:+9,.0f} ({pnl_pct:+6.1f}%) Ep={ep:8.2f} Xp={xp:8.2f} Hold={hold:3d}m Score={score:5.1f} [{etype[:15]}] {reason}')

print()
print('=== CONVERSIONS (8) ===')
for c in conversions:
    sym = c.get('underlying','').replace('NSE:','')
    ctype = c.get('conversion_type','')
    net = c.get('net_debit',0)
    width = c.get('spread_width',0)
    ts = c.get('ts','')[11:19]
    print(f'{ts} {sym:14s} {ctype} net_debit={net:.1f} width={width:.0f}')

print()
print('=== SUMMARY ===')
print(f'Entries: {len(entries)} | Exits: {len(exits)} | Conversions: {len(conversions)}')
if wins + losses > 0:
    print(f'Wins: {wins} | Losses: {losses} | Win Rate: {wins/(wins+losses)*100:.0f}%')
print(f'Total PnL: Rs {total_pnl:+,.0f}')
w_pnls = [x['pnl'] for x in exits if x['pnl'] > 0]
l_pnls = [x['pnl'] for x in exits if x['pnl'] <= 0]
if w_pnls:
    print(f'Avg Win: Rs {sum(w_pnls)/len(w_pnls):+,.0f} | Best: Rs {max(w_pnls):+,.0f}')
if l_pnls:
    print(f'Avg Loss: Rs {sum(l_pnls)/len(l_pnls):+,.0f} | Worst: Rs {min(l_pnls):+,.0f}')

# By source
print()
print('=== PnL BY SOURCE ===')
by_source = {}
for x in exits:
    src = x.get('source','UNKNOWN')
    by_source.setdefault(src, []).append(x['pnl'])
for src, pnls in sorted(by_source.items(), key=lambda kv: sum(kv[1]), reverse=True):
    w = sum(1 for p in pnls if p > 0)
    l = sum(1 for p in pnls if p <= 0)
    print(f'{src:15s}: Rs {sum(pnls):+9,.0f} | {w}W/{l}L | Trades: {len(pnls)}')

# Still open
print()
print('=== STILL OPEN (no exit in ledger) ===')
exit_oids = {x.get('order_id','') for x in exits}
for e in entries:
    oid = e.get('order_id','')
    if oid not in exit_oids:
        sym = e.get('underlying','').replace('NSE:','')
        src = e.get('source','')
        ep = e.get('entry_price',0)
        score = e.get('smart_score',0)
        ts = e.get('ts','')[11:19]
        d = e.get('direction','')
        print(f'{ts} {sym:14s} {d:4s} {src:15s} Score={score:5.1f} Entry={ep:8.2f}')

# By exit type
print()
print('=== PnL BY EXIT TYPE ===')
by_exit = {}
for x in exits:
    et = x.get('exit_type','UNKNOWN')
    by_exit.setdefault(et, []).append(x['pnl'])
for et, pnls in sorted(by_exit.items(), key=lambda kv: sum(kv[1]), reverse=True):
    print(f'{et:20s}: Rs {sum(pnls):+9,.0f} | {len(pnls)} trades')

# Score distribution of losers
print()
print('=== LOSERS DETAIL ===')
for x in sorted(exits, key=lambda z: z['pnl']):
    if x['pnl'] < 0:
        sym = x.get('underlying','').replace('NSE:','')
        pnl = x['pnl']
        score = x.get('smart_score',0)
        hold = x.get('hold_minutes',0)
        src = x.get('source','')
        etype = x.get('exit_type','')
        reason = x.get('exit_reason','')[:70]
        print(f'{sym:14s} PnL={pnl:+9,.0f} Score={score:5.1f} Hold={hold:3d}m [{src}/{etype}] {reason}')
