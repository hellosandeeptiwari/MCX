import json
p = "titan_settings.json"
d = json.load(open(p))
d["watcher_grind_min_score"] = 40
json.dump(d, open(p, "w"), indent=4)
print("Added watcher_grind_min_score=40")
