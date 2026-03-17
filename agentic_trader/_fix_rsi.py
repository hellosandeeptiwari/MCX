import json
f = "/home/ubuntu/titan/agentic_trader/titan_settings.json"
d = json.load(open(f))
d["watcher_rsi_extreme_pe_max"] = 24
json.dump(d, open(f, "w"), indent=2)
print("DONE:", d["watcher_rsi_extreme_pe_max"])
