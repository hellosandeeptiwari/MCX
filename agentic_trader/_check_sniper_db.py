#!/usr/bin/env python3
"""Get today's GMM_SNIPER trades from the trades table."""
import sqlite3, json

conn = sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db")
cur = conn.cursor()

# Get today's GMM_SNIPER trades
cur.execute("""
    SELECT underlying, direction, entry_time, exit_time, exit_type, pnl, status, 
           smart_score, strategy_type, lot_multiplier, hold_minutes
    FROM trades
    WHERE date = '2026-03-18' AND strategy_type = 'GMM_SNIPER'
    ORDER BY entry_time
""")
rows = cur.fetchall()
print(f"Today 2026-03-18 GMM_SNIPER trades: {len(rows)}")
for r in rows:
    print(f"  {r[0]:20s} {r[1]:5s} entry={r[2]} exit={r[3]} exit_type={r[4]} P/L={r[5]} status={r[6]} smart={r[7]} lots={r[9]}")

print()

# All strategy types today
cur.execute("""
    SELECT strategy_type, COUNT(*), SUM(COALESCE(pnl, 0))
    FROM trades
    WHERE date = '2026-03-18'
    GROUP BY strategy_type
    ORDER BY COUNT(*) DESC
""")
rows = cur.fetchall()
print("Today's trade breakdown by strategy:")
for r in rows:
    print(f"  {r[0]:25s}: {r[1]} trades, P/L: {r[2]}")

print()

# Total today
cur.execute("SELECT COUNT(*), SUM(COALESCE(pnl, 0)) FROM trades WHERE date = '2026-03-18'")
total = cur.fetchone()
print(f"Total today: {total[0]} trades, P/L: {total[1]}")

# Previous day comparison
cur.execute("""
    SELECT strategy_type, COUNT(*)
    FROM trades
    WHERE date = '2026-03-17' AND strategy_type = 'GMM_SNIPER'
""")
prev = cur.fetchone()
print(f"\nYesterday 2026-03-17 GMM_SNIPER: {prev[1]} trades")

# All GMM_SNIPER across dates
cur.execute("""
    SELECT date, COUNT(*), SUM(COALESCE(pnl, 0))
    FROM trades
    WHERE strategy_type = 'GMM_SNIPER'
    GROUP BY date
    ORDER BY date
""")
rows = cur.fetchall()
print("\nGMM_SNIPER by date:")
for r in rows:
    print(f"  {r[0]}: {r[1]} trades, P/L: {r[2]}")

conn.close()
