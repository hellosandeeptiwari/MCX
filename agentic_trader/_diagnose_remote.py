#!/usr/bin/env python3
"""Diagnose watcher performance - run on EC2"""
import sqlite3, json, os

BASE = "/home/ubuntu/titan/agentic_trader"
TODAY = "2026-03-10"

# 1. Explore databases
for db_name in ["titan_state.db", "state.db"]:
    db_path = os.path.join(BASE, db_name)
    if not os.path.exists(db_path):
        continue
    print(f"\n=== {db_name} ===")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Tables: {tables}")
    for tbl in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
        count = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {count} rows, cols={cols}")
        # Try to find today's trades
        for col in cols:
            if "time" in col.lower() or "date" in col.lower() or "stamp" in col.lower():
                try:
                    today_rows = conn.execute(f"SELECT * FROM {tbl} WHERE {col} LIKE ?", (f"%{TODAY}%",)).fetchall()
                    if today_rows:
                        print(f"\n  >>> {len(today_rows)} rows today in {tbl} (filtered by {col}):")
                        for row in today_rows:
                            print(f"    {dict(row)}")
                except Exception as e:
                    pass
                break
    conn.close()

# 2. Watcher state
ws_path = os.path.join(BASE, "watcher_state.json")
if os.path.exists(ws_path):
    ws = json.load(open(ws_path))
    print(f"\n=== WATCHER STATE ===")
    print(f"Keys: {list(ws.keys()) if isinstance(ws, dict) else 'list'}")
    if isinstance(ws, dict):
        for k, v in ws.items():
            if isinstance(v, dict):
                print(f"  {k}: {list(v.keys())[:10]}...")
            elif isinstance(v, list):
                print(f"  {k}: {len(v)} items")
            else:
                print(f"  {k}: {v}")

# 3. Check recent logs
log_dir = "/home/ubuntu/titan/logs"
if os.path.exists(log_dir):
    import glob
    logs = sorted(glob.glob(os.path.join(log_dir, "*.log")), key=os.path.getmtime, reverse=True)
    print(f"\n=== LOG FILES ===")
    for l in logs[:5]:
        sz = os.path.getsize(l)
        print(f"  {os.path.basename(l)}: {sz/1024:.0f}KB")

# 4. Read recent log lines for watcher activity
for log_name in ["titan_bot.log", "titan-bot.log", "bot.log"]:
    log_path = os.path.join(log_dir, log_name)
    if os.path.exists(log_path):
        print(f"\n=== RECENT WATCHER LOG LINES from {log_name} ===")
        import subprocess
        result = subprocess.run(["grep", "-i", "watcher\|WME\|momentum.*exit\|GATE.*CHECK\|trigger.*bonus\|placed.*order\|ENTRY\|EXIT", log_path], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        # Get only today's lines
        today_lines = [l for l in lines if TODAY in l]
        if today_lines:
            print(f"  {len(today_lines)} watcher-related lines today")
            for l in today_lines[-50:]:
                print(f"  {l}")
        else:
            # Show last 30 lines anyway
            print(f"  No today lines, last 30 watcher lines:")
            for l in lines[-30:]:
                print(f"  {l}")
        break

# 5. Also check stdout/journal
print("\n=== CHECKING systemd journal (last 100 watcher lines) ===")
import subprocess
try:
    result = subprocess.run(
        ["journalctl", "-u", "titan-bot", "--since", "today", "--no-pager", "-o", "cat"],
        capture_output=True, text=True, timeout=10
    )
    all_lines = result.stdout.strip().split("\n")
    watcher_lines = [l for l in all_lines if any(kw in l.lower() for kw in ["watcher", "wme", "momentum exit", "gate check", "trigger bonus", "placed", "🌊", "📊", "breakout", "entry signal"])]
    print(f"Total journal lines today: {len(all_lines)}")
    print(f"Watcher-related lines: {len(watcher_lines)}")
    for l in watcher_lines[-80:]:
        print(f"  {l}")
    
    # Also show trade placement and exit lines
    trade_lines = [l for l in all_lines if any(kw in l for kw in ["TRADE PLACED", "EXIT", "P&L", "pnl", "CLOSED", "OPEN position"])]
    if trade_lines:
        print(f"\n=== TRADE PLACEMENT/EXIT LINES ({len(trade_lines)}) ===")
        for l in trade_lines[-30:]:
            print(f"  {l}")
except Exception as e:
    print(f"  Journal error: {e}")
