import json, sys, os
from datetime import datetime

ledger_path = os.path.join(os.path.dirname(__file__), 'trade_ledger', 'trade_ledger_2026-03-06.jsonl')
entries, exits = [], []
scans = []
with open(ledger_path) as f:
    for line in f:
        try:
            r = json.loads(line.strip())
            ev = r.get('event','')
            if ev == 'ENTRY': entries.append(r)
            elif ev in ('EXIT','MANUAL_EXIT'): exits.append(r)
            elif ev == 'SCAN': scans.append(r)
        except: pass

# Build trade map — match exits to entries by base symbol (before '|' for debit spreads)
trade_map = {}
for e in entries:
    key = e.get('symbol','') or e.get('option_symbol','')
    trade_map[key] = {'entry': e, 'exit': None}
for x in exits:
    raw_key = x.get('symbol','') or x.get('option_symbol','')
    base_key = raw_key.split('|')[0]  # debit spread exits have "LEG1|LEG2" format
    if raw_key in trade_map:
        trade_map[raw_key]['exit'] = x
    elif base_key in trade_map:
        trade_map[base_key]['exit'] = x  # match debit-spread exit to original entry
    else:
        trade_map[raw_key] = {'entry': None, 'exit': x}

total_pnl = 0
wins = losses = opens = 0
by_source = {}
by_sector = {}

print()
print("=" * 130)
print(f"{'Time':>5} | {'Symbol':<30} | {'Dir':<4} | {'Source':<18} | {'Entry':>8} | {'Exit':>8} | {'Qty':>4} | {'P&L':>10} | {'P&L%':>6} | {'Hold':>5} | {'Exit Reason':<22}")
print("=" * 130)

for sym, t in sorted(trade_map.items(), key=lambda x: (x[1]['entry'] or x[1]['exit'] or {}).get('ts','')):
    e = t['entry']
    x = t['exit']
    entry_time = (e or {}).get('ts','')[11:16] if e else '?'
    symbol = sym.replace('NFO:','').replace('NSE:','')[:30]
    direction = (e or x or {}).get('direction','?')
    source = (e or x or {}).get('source','?')[:18]
    entry_price = (e or {}).get('entry_price', 0)
    exit_price = (x or {}).get('exit_price', 0)
    qty = (e or x or {}).get('quantity', 0)
    sector = (e or x or {}).get('sector', '?')
    pnl = (x or {}).get('pnl', 0) if x else 0
    pnl_pct = (x or {}).get('pnl_pct', 0) if x else 0
    hold_min = (x or {}).get('hold_minutes', '-') if x else '-'
    exit_reason = (x or {}).get('exit_reason', 'OPEN')[:22] if x else 'STILL OPEN'
    smart_score = (e or {}).get('smart_score', 0)
    dr_score = (e or {}).get('dr_score', 0)
    
    if x:
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
        by_source.setdefault(source, {'pnl':0, 'wins':0, 'losses':0, 'trades':0})
        by_source[source]['pnl'] += pnl
        by_source[source]['trades'] += 1
        if pnl > 0: by_source[source]['wins'] += 1
        else: by_source[source]['losses'] += 1
        by_sector.setdefault(sector, {'pnl':0, 'trades':0})
        by_sector[sector]['pnl'] += pnl
        by_sector[sector]['trades'] += 1
    else:
        opens += 1
    
    s = '+' if pnl > 0 else ('-' if pnl < 0 else ' ')
    pnl_str = f"{s}Rs{abs(pnl):,.0f}"
    print(f"{entry_time:>5} | {symbol:<30} | {direction:<4} | {source:<18} | {entry_price:>8.1f} | {exit_price:>8.1f} | {qty:>4} | {pnl_str:>10} | {pnl_pct:>5.1f}% | {str(hold_min):>4}m | {exit_reason:<22}")

print("=" * 130)
print()

# Summary
total_trades = wins + losses
print(f"  TRADES: {len(trade_map)} total | {wins} wins | {losses} losses | {opens} still open")
print(f"  WIN RATE: {wins*100/total_trades if total_trades else 0:.0f}%")
print(f"  NET P&L (closed): Rs {total_pnl:+,.1f}")
print()

# By source
print("  BY STRATEGY:")
for src, data in sorted(by_source.items(), key=lambda x: x[1]['pnl'], reverse=True):
    wr = data['wins']*100/data['trades'] if data['trades'] else 0
    print(f"    {src:<20} {data['trades']} trades | {data['wins']}W/{data['losses']}L ({wr:.0f}%) | Rs {data['pnl']:+,.1f}")

# By sector
print()
print("  BY SECTOR:")
for sec, data in sorted(by_sector.items(), key=lambda x: x[1]['pnl'], reverse=True):
    print(f"    {sec:<20} {data['trades']} trades | Rs {data['pnl']:+,.1f}")

# ALL_AGREE analysis
print()
print("  ALL_AGREE SCAN ACTIVITY:")
aa_scans = [s for s in scans if 'ALL_AGREE' in s.get('outcome','')]
outcomes = {}
for s in aa_scans:
    o = s.get('outcome','')
    outcomes[o] = outcomes.get(o, 0) + 1
for o, c in sorted(outcomes.items()):
    print(f"    {o}: {c} events")
    
# Stocks that passed ALL_AGREE
aa_passed = [s for s in aa_scans if s.get('outcome') == 'ALL_AGREE']
if aa_passed:
    print()
    print("  ALL_AGREE QUALIFIED:")
    for s in aa_passed:
        sym = s.get('symbol','').replace('NSE:','')
        print(f"    {s.get('ts','')[11:16]} {sym:<20} score={s.get('score',0):.0f} {s.get('reason','')}")

# Watcher trades would need log parsing
print()
print("  KEY OBSERVATIONS:")
if total_pnl > 0:
    print(f"    + Profitable day: Rs {total_pnl:+,.1f}")
elif total_pnl < 0:
    print(f"    - Loss day: Rs {total_pnl:+,.1f}")
    
biggest_win = max([t['exit']['pnl'] for t in trade_map.values() if t['exit'] and t['exit']['pnl'] > 0], default=0)
biggest_loss = min([t['exit']['pnl'] for t in trade_map.values() if t['exit'] and t['exit']['pnl'] < 0], default=0)
if biggest_win:
    bw_sym = [sym for sym, t in trade_map.items() if t['exit'] and t['exit']['pnl'] == biggest_win][0]
    print(f"    Best trade: {bw_sym.replace('NFO:','')} Rs {biggest_win:+,.1f}")
if biggest_loss:
    bl_sym = [sym for sym, t in trade_map.items() if t['exit'] and t['exit']['pnl'] == biggest_loss][0]
    print(f"    Worst trade: {bl_sym.replace('NFO:','')} Rs {biggest_loss:+,.1f}")

avg_hold = [t['exit']['hold_minutes'] for t in trade_map.values() if t['exit'] and isinstance(t['exit'].get('hold_minutes'), (int,float))]
if avg_hold:
    print(f"    Avg hold time: {sum(avg_hold)/len(avg_hold):.0f} min")
