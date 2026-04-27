import json, pprint
h = json.load(open('/home/ubuntu/titan/agentic_trader/trade_history.json'))
rows = h if isinstance(h, list) else h.get('trades', h.get('history', []))
print('TYPE', type(h).__name__, 'COUNT', len(rows))
sl = [r for r in rows if 'SL' in str(r.get('exit_type', r.get('exit_reason', ''))).upper()]
print('SL rows in tail:')
for r in (sl[-6:] if sl else rows[-6:]):
    pprint.pprint(r)
    print('---')
