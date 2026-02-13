import json, os
f = "trade_history.json"
if os.path.exists(f):
    h = json.load(open(f))
    before = len(h)
    h = [t for t in h if "2026-02-11" not in t.get("closed_at", "")]
    with open(f, "w") as fp:
        json.dump(h, fp, indent=2)
    print(f"trade_history.json: removed {before - len(h)} today entries, kept {len(h)}")
else:
    print("No trade_history.json")
