import json

with open("titan_settings.json", "r") as f:
    s = json.load(f)

s["watcher_price_spike_pct"] = 0.6
s["watcher_price_spike_pct_open"] = 1.0
s["watcher_price_spike_open_until"] = "09:45"
s["watcher_sustain_recheck_pct"] = 0.4
s["watcher_slow_grind_pct"] = 0.75

with open("titan_settings.json", "w") as f:
    json.dump(s, f, indent=2)

print(f"price_spike_pct       = {s['watcher_price_spike_pct']}")
print(f"price_spike_pct_open  = {s['watcher_price_spike_pct_open']}")
print(f"price_spike_open_until= {s['watcher_price_spike_open_until']}")
print(f"sustain_recheck       = {s['watcher_sustain_recheck_pct']}")
print(f"slow_grind_pct        = {s['watcher_slow_grind_pct']}")
print("DONE — settings will hot-reload in ~30s")
