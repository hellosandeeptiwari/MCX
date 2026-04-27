import json
with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-02.jsonl') as f:
    for line in f:
        d = json.loads(line.strip())
        if d.get('event') == 'EXIT':
            print('KEYS:', list(d.keys()))
            pnl_fields = {k:d[k] for k in d if 'pnl' in k.lower() or 'profit' in k.lower() or 'realized' in k.lower() or 'loss' in k.lower()}
            print('PNL fields:', pnl_fields)
            print('Symbol:', d.get('symbol',''), 'exit_type:', d.get('exit_type',''))
            print('source:', d.get('source',''))
            break
