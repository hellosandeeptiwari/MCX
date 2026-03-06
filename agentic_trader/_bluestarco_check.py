#!/usr/bin/env python3
"""Check BLUESTARCO duplicate entries in trade ledger."""
import sqlite3

db = sqlite3.connect('state.db')
db.row_factory = sqlite3.Row
rows = db.execute(
    "SELECT * FROM trade_ledger WHERE symbol LIKE '%BLUESTARCO%' AND date >= '2026-03-05' ORDER BY timestamp"
).fetchall()

print(f"Total BLUESTARCO ledger rows: {len(rows)}\n")
for r in rows:
    d = dict(r)
    print(f"  {d.get('record_type','?'):6s} | {d.get('symbol','?'):30s} | "
          f"ts={d.get('timestamp','?')} | price={d.get('entry_price', d.get('exit_price','?'))} | "
          f"pnl={d.get('pnl','?')} | exit_type={d.get('exit_type','?')} | "
          f"setup={d.get('setup_type','?')} | score={d.get('score','?')}")

# Also check decisions log for BLUESTARCO
print("\n--- Decision log ---")
try:
    drows = db.execute(
        "SELECT * FROM watcher_decisions WHERE symbol LIKE '%BLUESTARCO%' AND date >= '2026-03-05' ORDER BY timestamp"
    ).fetchall()
    for r in drows:
        d = dict(r)
        print(f"  {d.get('decision','?'):25s} | score={d.get('score','?')} | "
              f"reason={d.get('reason','?')} | dir={d.get('direction','?')} | ts={d.get('timestamp','?')}")
except Exception as e:
    print(f"  (no decisions table or error: {e})")

db.close()
