import sqlite3, json

db = sqlite3.connect(r'C:\Users\SandeepTiwari\MCX\titan_state_remote.db')
db.row_factory = sqlite3.Row

# Check active trade for COLPAL
print("=== ACTIVE TRADE: COLPAL ===")
rows = db.execute("SELECT * FROM active_trades WHERE date='2026-03-11' AND trade_json LIKE '%COLPAL%'").fetchall()
for r in rows:
    t = json.loads(r['trade_json'])
    print(json.dumps(t, indent=2, default=str))
