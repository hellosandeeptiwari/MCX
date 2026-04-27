import json, urllib.request
def get(p):
    return json.loads(urllib.request.urlopen("http://localhost:5000"+p).read())
h24 = get("/api/trades/day/2026-04-24")
print("=== /api/trades/day/2026-04-24 (Trade History tab on Apr 24) ===")
for k,v in h24.items():
    if not isinstance(v,(list,dict)):
        print(f"  {k}: {v}")
print()
print("  by_source counts:")
for src,info in (h24.get("by_source") or {}).items():
    print(f"    {src:25s} count={info['count']:3d} wins={info['wins']:3d} pnl={info['pnl']:+,.0f}")
print()
print("  trades in summary:", len(h24.get("trades", [])))


