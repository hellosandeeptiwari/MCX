import json
f = '/home/ubuntu/titan/agentic_trader/titan_settings.json'
d = json.load(open(f))
d['watcher_slow_grind_pct'] = 0.85
json.dump(d, open(f, 'w'), indent=2)
print('slow_grind_pct set to', d['watcher_slow_grind_pct'])
