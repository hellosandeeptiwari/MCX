"""List all SELL COMPLETE orders today."""
import os, json
from kiteconnect import KiteConnect

env = {}
for line in open('.env'):
    line = line.strip()
    if '=' in line and not line.startswith('#'):
        k_, v_ = line.split('=', 1)
        env[k_.strip()] = v_.strip().strip('"').strip("'")

k = KiteConnect(api_key=env.get('ZERODHA_API_KEY'))
k.set_access_token(env.get('ZERODHA_ACCESS_TOKEN'))

print("--- ALL SELL COMPLETE orders ---")
for o in k.orders():
    if o.get('transaction_type') == 'SELL' and o.get('status') == 'COMPLETE':
        print(o.get('tradingsymbol'), 'avg=', o.get('average_price'),
              'qty=', o.get('quantity'), 'id=', o.get('order_id'),
              'ts=', str(o.get('order_timestamp')))

# Also check trade_history.json
print("\n--- trade_history.json today's EXITED ---")
th = json.load(open('trade_history.json'))
for t in th[-40:]:
    sym = t.get('symbol', '')
    if any(x in sym for x in ['VEDL26APR760CE', 'FORTIS26APR880CE', 'AUBANK26APR1020CE', 'VBL26APR480CE', 'MAZDOCK26APR2700PE', 'ICICIPRULI26APR545CE']):
        print(sym, 'entry=', t.get('entry_price'), 'exit=', t.get('exit_price'),
              'exit_order_id=', t.get('exit_order_id'), 'order_id=', t.get('order_id'),
              'status=', t.get('status'))
