#!/usr/bin/env python3
import sqlite3
c = sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db")
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)
if 'live_pnl' in tables:
    cnt = c.execute("SELECT count(*) FROM live_pnl").fetchone()[0]
    print(f"live_pnl rows: {cnt}")
    if cnt > 0:
        for r in c.execute("SELECT * FROM live_pnl").fetchall():
            print(f"  {r}")
else:
    print("live_pnl table NOT FOUND")
