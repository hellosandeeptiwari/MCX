#!/usr/bin/env python3
"""Check all exit_state fields for LTP proxy data."""
import sys, json
sys.path.insert(0, "/home/ubuntu/titan/agentic_trader")
from state_db import get_state_db
db = get_state_db()
es = db.load_exit_states()
for k, v in es.items():
    print(f"\n{k}:")
    for field in ["entry_price", "side", "current_sl", "target", "highest_price",
                   "lowest_price", "candles_since_entry", "max_favorable_move",
                   "max_premium_gain_pct", "premium_history", "underlying_history",
                   "quantity"]:
        print(f"  {field}: {v.get(field)}")
