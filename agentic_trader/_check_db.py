#!/usr/bin/env python3
"""Inspect SQLite databases for trade storage."""
import sqlite3, json

# Check titan_state.db
conn = sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print("Tables in titan_state.db:", [t[0] for t in tables])

for tbl in tables:
    name = tbl[0]
    cur.execute(f"SELECT COUNT(*) FROM {name}")
    cnt = cur.fetchone()[0]
    print(f"  {name}: {cnt} rows")
    cur.execute(f"PRAGMA table_info({name})")
    cols = [c[1] for c in cur.fetchall()]
    print(f"    columns: {cols}")
    if cnt > 0 and "date" in str(cols).lower():
        cur.execute(f"SELECT DISTINCT date FROM {name} ORDER BY date DESC LIMIT 5")
        dates = cur.fetchall()
        print(f"    recent dates: {[d[0] for d in dates]}")

conn.close()

# Check state.db
print()
conn2 = sqlite3.connect("/home/ubuntu/titan/agentic_trader/state.db")
cur2 = conn2.cursor()
cur2.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables2 = cur2.fetchall()
print("Tables in state.db:", [t[0] for t in tables2])
for tbl in tables2:
    name = tbl[0]
    cur2.execute(f"SELECT COUNT(*) FROM {name}")
    cnt = cur2.fetchone()[0]
    print(f"  {name}: {cnt} rows")
    cur2.execute(f"PRAGMA table_info({name})")
    cols = [c[1] for c in cur2.fetchall()]
    print(f"    columns: {cols}")

conn2.close()

# Look for today's traded data via titan_state
print("\n=== Checking today's data ===")
conn3 = sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db")
cur3 = conn3.cursor()
try:
    cur3.execute("SELECT date, data FROM daily_state WHERE date = '2026-03-18'")
    row = cur3.fetchone()
    if row:
        data = json.loads(row[1])
        trades = data.get("active_trades", data.get("trades", []))
        print(f"Today: {len(trades)} trades in daily_state")
        snipers = [t for t in trades if t.get("setup_type") == "GMM_SNIPER"]
        print(f"  GMM_SNIPER: {len(snipers)}")
        for t in snipers:
            sym = t.get("underlying", "?")
            dr = t.get("direction", "?")
            et = str(t.get("entry_time", ""))[:19]
            pnl = t.get("pnl", 0)
            status = t.get("status", "?")
            print(f"    {sym} {dr} {status} {et} P/L:{pnl}")
    else:
        print("No data for 2026-03-18")
except Exception as e:
    print(f"Query error: {e}")
conn3.close()
