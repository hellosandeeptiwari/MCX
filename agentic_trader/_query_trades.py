#!/usr/bin/env python3
"""Quick query: find today's trades, especially 'all_3'."""
import sqlite3, json, os, sys

# Check state DB
db_path = os.path.join(os.path.dirname(__file__), 'titan_state.db')
if os.path.exists(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"=== DB Tables: {tables} ===")
    for t in tables:
        cols = [d[1] for d in db.execute(f"PRAGMA table_info({t})").fetchall()]
        cnt = db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {cnt} rows, cols={cols[:10]}")
    
    # Search for today's trades
    for t in tables:
        cols = [d[1] for d in db.execute(f"PRAGMA table_info({t})").fetchall()]
        date_cols = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower()]
        if date_cols:
            for dc in date_cols:
                try:
                    rows = db.execute(f"SELECT * FROM {t} WHERE {dc} LIKE '2026-03-02%' OR {dc} LIKE '%2026-03-02%'").fetchall()
                    if rows:
                        print(f"\n=== {t} (filtered by {dc}): {len(rows)} rows ===")
                        for r in rows:
                            d = dict(r)
                            print(json.dumps(d, indent=2, default=str))
                except:
                    pass
    db.close()
else:
    print(f"No DB at {db_path}")

# Check trade ledger
ledger_path = os.path.join(os.path.dirname(__file__), 'trade_ledger', 'trade_ledger_2026-03-02.jsonl')
if os.path.exists(ledger_path):
    print(f"\n=== Trade Ledger ({ledger_path}) ===")
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Print summary
                sym = entry.get('symbol', '?')
                action = entry.get('action', entry.get('event', '?'))
                setup = entry.get('setup_type', entry.get('setup', ''))
                pnl = entry.get('pnl', entry.get('realized_pnl', ''))
                ts = entry.get('timestamp', entry.get('time', ''))
                trade_id = entry.get('trade_id', '')
                print(f"  {ts} | {sym} | {action} | setup={setup} | pnl={pnl} | id={trade_id}")
            except:
                print(f"  RAW: {line[:200]}")

# Check for 'all_3' pattern in logs
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
if os.path.isdir(log_dir):
    today_logs = [f for f in os.listdir(log_dir) if '2026-03-02' in f or 'titan' in f.lower()]
    for lf in sorted(today_logs)[:5]:
        fp = os.path.join(log_dir, lf)
        try:
            with open(fp) as f:
                content = f.read()
                if 'all_3' in content.lower() or 'all3' in content.lower():
                    print(f"\n=== Found 'all_3' in {lf} ===")
                    for i, line in enumerate(content.split('\n')):
                        if 'all_3' in line.lower() or 'all3' in line.lower():
                            print(f"  L{i}: {line[:300]}")
        except:
            pass

# Check bot log
bot_log = '/home/ubuntu/titan/agentic_trader/titan_bot.log'
if os.path.exists(bot_log):
    import subprocess
    result = subprocess.run(['grep', '-i', 'all_3\|all3', bot_log], capture_output=True, text=True)
    if result.stdout:
        print(f"\n=== 'all_3' in titan_bot.log ===")
        for line in result.stdout.strip().split('\n')[:30]:
            print(f"  {line[:300]}")

# Search all recent log files for all_3
for root, dirs, files in os.walk('/home/ubuntu/titan/agentic_trader'):
    for f in files:
        if f.endswith('.log') or f.endswith('.jsonl'):
            fp = os.path.join(root, f)
            try:
                mtime = os.path.getmtime(fp)
                import time
                if time.time() - mtime < 86400:  # last 24h
                    with open(fp) as fh:
                        for i, line in enumerate(fh):
                            if 'all_3' in line.lower() or 'all3' in line.lower():
                                print(f"\n  Found in {fp} L{i}: {line.strip()[:300]}")
            except:
                pass
