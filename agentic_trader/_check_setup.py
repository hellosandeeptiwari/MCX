#!/usr/bin/env python3
import json, urllib.request
d = json.loads(urllib.request.urlopen("http://localhost:5000/api/status").read())
for p in d.get("positions", []):
    print(f"sym={p.get('symbol')}")
    print(f"  setup_type={p.get('setup_type')}")
    print(f"  strategy_type={p.get('strategy_type')}")
    print(f"  smart_score={p.get('smart_score')}")
    print(f"  entry_score={p.get('entry_score')}")
    print(f"  p_score={p.get('p_score')}")
    print(f"  ml_scored_direction={p.get('ml_scored_direction')}")
    print(f"  rationale (first 60)={str(p.get('rationale',''))[:60]}")
    print(f"  ltp={p.get('ltp')}, upnl={p.get('unrealized_pnl')}")
    print("---")
