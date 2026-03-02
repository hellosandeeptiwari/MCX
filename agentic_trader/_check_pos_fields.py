#!/usr/bin/env python3
import json, urllib.request
d = json.loads(urllib.request.urlopen("http://localhost:5000/api/status").read())
for p in d.get("positions", []):
    fields = {k: p.get(k) for k in [
        "symbol", "strategy_type", "source", "smart_score", "pre_score",
        "final_score", "dr_score", "gate_prob", "ml_direction", "ml_move_prob",
        "gmm_action", "score_tier", "is_sniper", "rationale"
    ]}
    print(json.dumps(fields, indent=2))
    print("---")
