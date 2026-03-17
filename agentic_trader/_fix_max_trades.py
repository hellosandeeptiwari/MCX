import json
p = '/home/ubuntu/titan/agentic_trader/titan_settings.json'
with open(p) as f:
    d = json.load(f)
d['watcher_max_trades_per_scan'] = 5
with open(p, 'w') as f:
    json.dump(d, f, indent=2)
print(f"watcher_max_trades_per_scan = {d['watcher_max_trades_per_scan']}")
