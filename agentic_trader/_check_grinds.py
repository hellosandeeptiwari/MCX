import sqlite3
conn = sqlite3.connect('titan_state.db')
c = conn.cursor()

# Today's grind trades
print("=== TODAY'S GRIND TRADES ===")
c.execute('''SELECT underlying, direction, source, pnl, smart_score, entry_time, exit_time, exit_type 
             FROM trades WHERE date='2026-03-18' AND source LIKE '%GRIND%' ORDER BY pnl''')
for r in c.fetchall():
    print(r)

print("\n=== GRIND SUMMARY TODAY ===")
c.execute('''SELECT source, COUNT(*), ROUND(SUM(pnl),0), ROUND(AVG(pnl),0) FROM trades WHERE date='2026-03-18' AND source LIKE '%GRIND%' GROUP BY source''')
for r in c.fetchall():
    print(f"  {r[0]}: {r[1]} trades, total={r[2]}, avg={r[3]}")

# All-time grind stats
print("\n=== ALL-TIME GRIND STATS ===")
c.execute('''SELECT source, COUNT(*), ROUND(SUM(pnl),0), ROUND(AVG(pnl),0),
             SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
             SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses
             FROM trades WHERE source LIKE '%GRIND%' GROUP BY source''')
for r in c.fetchall():
    print(f"  {r[0]}: {r[1]} trades, total={r[2]}, avg={r[3]}, W:{r[4]} L:{r[5]}")

# Recent grind trades with move_pct context
print("\n=== RECENT GRIND TRADES (last 5 days) ===")
c.execute('''SELECT date, underlying, direction, source, pnl, smart_score, entry_time, exit_type
             FROM trades WHERE source LIKE '%GRIND%' ORDER BY date DESC, entry_time DESC LIMIT 20''')
for r in c.fetchall():
    print(r)

conn.close()
