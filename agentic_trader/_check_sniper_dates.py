#!/usr/bin/env python3
"""Check GMM_SNIPER trades across all dates in trade_history."""
import json, sys
sys.path.insert(0, "/home/ubuntu/titan/agentic_trader")

# Check trade history
try:
    with open("/home/ubuntu/titan/agentic_trader/trade_history.json") as f:
        history = json.load(f)
    snipers = [t for t in history if t.get("setup_type") == "GMM_SNIPER"]
    print(f"trade_history total trades: {len(history)}")
    print(f"trade_history GMM_SNIPER total: {len(snipers)}")
    for t in snipers:
        et = str(t.get("entry_time", ""))[:19]
        sym = t.get("underlying", "?")
        dr = t.get("direction", "?")
        st = t.get("status", "?")
        pnl = t.get("pnl", 0)
        print(f"  {sym:15s} {dr:5s} {st:8s} {et} P/L:{pnl}")
except Exception as e:
    print(f"History error: {e}")

print()

# Check all dates
try:
    dates = set()
    for t in history:
        et = str(t.get("entry_time", ""))[:10]
        if et.startswith("2026"):
            dates.add(et)
    print(f"All dates in trade_history: {sorted(dates)}")
except Exception as e:
    print(f"Date check error: {e}")

print()

# Check state DB for all dates
try:
    from state_db import get_state_db
    db = get_state_db()
    import sqlite3
    conn = sqlite3.connect(db._db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM daily_state ORDER BY date")
    rows = cursor.fetchall()
    print(f"Dates in state DB: {[r[0] for r in rows]}")
    
    # Get today's trades from state DB
    cursor.execute("SELECT data FROM daily_state WHERE date = '2026-03-18'")
    row = cursor.fetchone()
    if row:
        data = json.loads(row[0])
        trades = data.get("active_trades", [])
        sniper_trades = [t for t in trades if t.get("setup_type") == "GMM_SNIPER"]
        print(f"\nState DB 2026-03-18: {len(trades)} total trades, {len(sniper_trades)} snipers")
        for t in sniper_trades:
            et = str(t.get("entry_time", ""))[:19]
            sym = t.get("underlying", "?")
            dr = t.get("direction", "?")
            pnl = t.get("pnl", 0)
            print(f"  {sym:15s} {dr:5s} {et} P/L:{pnl}")
    conn.close()
except Exception as e:
    print(f"State DB error: {e}")

# Also check VEDL specifically
print("\n=== VEDL trades in history ===")
try:
    vedl = [t for t in history if "VEDL" in str(t.get("underlying", ""))]
    for t in vedl:
        et = str(t.get("entry_time", ""))[:19]
        print(f"  {t.get('underlying','?')} {t.get('direction','?')} {t.get('setup_type','?')} {et} P/L:{t.get('pnl',0)} {t.get('status','?')}")
except Exception as e:
    print(f"VEDL check error: {e}")
