import json

data = json.load(open("/home/ubuntu/titan/agentic_trader/scan_decisions.json"))

# Find VBL and RVNL decisions (today only)
for d in data:
    sym = d.get("symbol","")
    ts = d.get("timestamp","")
    if ("VBL" in sym or "RVNL" in sym) and "2026-03-04" in ts:
        print(json.dumps(d))

# Also show all AUTO_FIRED decisions today
print("\n=== ALL AUTO-FIRED TODAY ===")
for d in data:
    ts = d.get("timestamp","")
    outcome = d.get("outcome","")
    if "2026-03-04" in ts and "AUTO_FIRED" in outcome:
        print(json.dumps(d))
