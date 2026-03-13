import json
f = '/home/ubuntu/titan/agentic_trader/titan_settings.json'
d = json.load(open(f))
# Remove legacy short key if full key exists
if 'watcher_vix_hard_block_above' in d and 'watcher_vix_hard_block' in d:
    del d['watcher_vix_hard_block']
    print("Removed legacy watcher_vix_hard_block")
json.dump(d, open(f, 'w'), indent=4)
print(f"vix_hard_block_above = {d.get('watcher_vix_hard_block_above')}")
