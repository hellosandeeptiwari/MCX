#!/usr/bin/env python3
import sqlite3
conn = sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db")
cur = conn.cursor()

# Find VEDL trades today
cur.execute("SELECT underlying, direction, source, strategy_type, is_sniper, entry_time, pnl, status, smart_score FROM trades WHERE date = '2026-03-18' AND underlying LIKE '%VEDL%'")
vedl = cur.fetchall()
print(f"VEDL trades today: {len(vedl)}")
for r in vedl:
    print(f"  {r}")

# All snipers today (is_sniper flag)
cur.execute("SELECT underlying, direction, source, strategy_type, is_sniper, entry_time, exit_time, pnl, status, smart_score, exit_type FROM trades WHERE date = '2026-03-18' AND is_sniper = 1")
snipers = cur.fetchall()
print(f"\nSniper trades today (is_sniper=1): {len(snipers)}")
for r in snipers:
    print(f"  {r}")

# All sources today
cur.execute("SELECT DISTINCT source FROM trades WHERE date = '2026-03-18'")
print(f"\nSources today: {[r[0] for r in cur.fetchall()]}")

# Check by source containing GMM or SNIPER
cur.execute("SELECT underlying, direction, source, strategy_type, entry_time, pnl, status FROM trades WHERE date = '2026-03-18' AND (source LIKE '%SNIPER%' OR source LIKE '%GMM%')")
gmm = cur.fetchall()
print(f"\nGMM/SNIPER source trades: {len(gmm)}")
for r in gmm:
    print(f"  {r}")

conn.close()
