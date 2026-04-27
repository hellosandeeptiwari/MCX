"""Audit today's MANUAL_DASHBOARD exits vs. broker fills."""
import json, os, sys
from kiteconnect import KiteConnect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Load from .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
env = {}
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k_, v_ = line.split('=', 1)
            env[k_.strip()] = v_.strip().strip('"').strip("'")
api_key = env.get('ZERODHA_API_KEY') or os.environ.get('ZERODHA_API_KEY')
access_token = env.get('ZERODHA_ACCESS_TOKEN') or os.environ.get('ZERODHA_ACCESS_TOKEN')
k = KiteConnect(api_key=api_key)
k.set_access_token(access_token)

ledger = 'trade_ledger/trade_ledger_2026-04-21.jsonl'
exits = []
for line in open(ledger):
    try:
        r = json.loads(line)
        if r.get('event') == 'EXIT':
            exits.append(r)
    except Exception:
        pass

orders = k.orders()
sell_orders = [o for o in orders if o.get('status') == 'COMPLETE' and o.get('transaction_type') == 'SELL']

print(f"{'SYMBOL':<30} {'REC_EXIT':>10} {'BROKER_AVG':>12} {'QTY':>8} {'DIFF':>10} {'MATCH':>6}")
print('-' * 90)
for e in exits:
    sym = e['symbol'].split(':', 1)[-1]
    rec = float(e.get('exit_price') or 0)
    qty = int(e.get('quantity') or 0)
    # find matching sell order with same tradingsymbol & quantity
    cand = [o for o in sell_orders if o.get('tradingsymbol') == sym]
    if not cand:
        print(f"{sym:<30} {rec:>10.2f} {'NO_ORDER':>12} {qty:>8} {'-':>10} {'NO':>6}")
        continue
    # prefer matching qty
    match = next((o for o in cand if int(o.get('quantity') or 0) == qty), cand[-1])
    avg = float(match.get('average_price') or 0)
    diff = rec - avg
    ok = 'YES' if abs(diff) < 0.05 else 'NO'
    print(f"{sym:<30} {rec:>10.2f} {avg:>12.2f} {qty:>8} {diff:>+10.2f} {ok:>6}  ts={e['ts'][-8:]}")
