#!/usr/bin/env python3
import sys, json
sys.path.insert(0, "/home/ubuntu/titan/agentic_trader")
from state_db import get_state_db
db = get_state_db()
es = db.load_exit_states()
print(f"exit_states: {len(es)} keys")
for k, v in es.items():
    ph = v.get("premium_history", [])
    print(f"  {k}: premium_history len={len(ph)}, last3={ph[-3:] if ph else 'EMPTY'}")
    print(f"    entry_price={v.get('entry_price')}, side={v.get('side')}")
