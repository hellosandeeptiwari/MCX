import json
with open("titan_settings.json", "r") as f:
    s = json.load(f)
s["watcher_early_market_min_dir_conf"] = 40
with open("titan_settings.json", "w") as f:
    json.dump(s, f, indent=2)
print(f"watcher_early_market_min_dir_conf = {s['watcher_early_market_min_dir_conf']}")
