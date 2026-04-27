import json
p = "titan_settings.json"
d = json.load(open(p))
d["watcher_early_market_min_score"] = 35
json.dump(d, open(p, "w"), indent=4)
print("Done: early_market_min_score=35")
