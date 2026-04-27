"""Inspect trade history tail."""
import json
th = json.load(open('trade_history.json'))
print('Total records:', len(th))
print()
print('--- Last 15 records ---')
for t in th[-15:]:
    print(t.get('symbol'), '| status=', t.get('status'),
          '| entry=', t.get('entry_price'), '| exit=', t.get('exit_price'),
          '| order_id=', t.get('order_id'), '| exit_order_id=', t.get('exit_order_id'),
          '| paper=', t.get('paper_trade'), '| exit_time=', t.get('exit_time'))
print()
print('--- Today Apr 21 matches ---')
for t in th:
    et = str(t.get('exit_time', '')) + str(t.get('entry_time', ''))
    if '2026-04-21' in et:
        print(t.get('symbol'), 'entry=', t.get('entry_price'), 'exit=', t.get('exit_price'),
              'order_id=', t.get('order_id'), 'exit_order_id=', t.get('exit_order_id'),
              'paper=', t.get('paper_trade'))
