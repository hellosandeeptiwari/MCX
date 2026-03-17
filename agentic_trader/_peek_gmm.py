import json, os
for f in ['trade_ledger/trade_ledger_2026-03-17.jsonl']:
    if not os.path.exists(f):
        continue
    for line in open(f):
        t = json.loads(line)
        if t.get('setup_type') == 'TEST_GMM':
            print(json.dumps(t, indent=2)[:3000])
            break
