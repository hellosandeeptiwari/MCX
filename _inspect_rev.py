import json, pprint
h = json.load(open('/home/ubuntu/titan/agentic_trader/trade_history.json'))
rows = h if isinstance(h, list) else h.get('trades', h.get('history', []))
for r in rows:
    sym = str(r.get('symbol', ''))
    src = str(r.get('source','')) + str(r.get('trigger_type','')) + str(r.get('setup_type',''))
    if ('RELIANCE' in sym or 'IDFC' in sym) and 'REVERSE' in src.upper():
        keys = ['symbol','side','direction','avg_price','entry_price','exit_price','stop_loss','target','quantity','pnl','result','status','setup_type','trigger_type','manual_setup','timestamp','closed_at','exit_detail','option_type']
        pprint.pprint({k: r.get(k) for k in keys})
        print('---')
