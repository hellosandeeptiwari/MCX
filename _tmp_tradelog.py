import json, os
tlog = '/home/ubuntu/titan/agentic_trader/trade_log.json'
with open(tlog) as f:
    events = json.load(f)

# Today's exits only
today_exits = [e for e in events if isinstance(e, dict) and e.get('event')=='EXIT' and (e.get('ts') or '').startswith('2026-04-23')]
print(f'total EXIT events today: {len(today_exits)}')
total = 0
reverse_count = 0
for e in today_exits:
    et = e.get('exit_type','')
    pnl = e.get('pnl') or 0
    total += pnl
    sym = e.get('symbol','')
    print(f'  {e.get("ts")[:19]} {sym:<45} pnl={pnl:>10.2f} type={et}')
    if 'REVERSE' in et.upper() or 'MANUAL' in et.upper():
        reverse_count += 1
print(f'\nSum of pnl from trade_log EXIT events: {total:+,.2f}')
print(f'Reverse/Manual exits: {reverse_count}')
