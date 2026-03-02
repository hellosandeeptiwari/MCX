#!/usr/bin/env python3
"""Quick check of /api/status enriched fields."""
import json, urllib.request

resp = urllib.request.urlopen("http://localhost:5000/api/status")
d = json.loads(resp.read())

print(f"unrealized_pnl: {d.get('unrealized_pnl')}")
print(f"realized_pnl:   {d.get('realized_pnl')}")
print(f"open_positions:  {d.get('open_positions')}")
print()
for p in d.get("positions", []):
    print(f"  sym={p.get('symbol')}")
    print(f"    underlying={p.get('underlying')}, strike={p.get('strike')}, type={p.get('option_type')}")
    print(f"    ltp={p.get('ltp')}, unrealized_pnl={p.get('unrealized_pnl')}")
    print(f"    dir={p.get('direction')}, entry={p.get('avg_price')}, sl={p.get('stop_loss')}, target={p.get('target')}")
    print()
