import json, sys

try:
    data = json.load(open("/home/ubuntu/titan/agentic_trader/scan_decisions.json"))
except:
    print("No scan_decisions.json found")
    sys.exit(0)

for d in data:
    sym = d.get("symbol","")
    if "VBL" in sym or "RVNL" in sym:
        print(json.dumps(d, indent=2))
        print("---")
