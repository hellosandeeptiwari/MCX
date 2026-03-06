#!/usr/bin/env python3
"""Check BLUESTARCO duplicate entries."""
import sqlite3, os, json, glob

# 1. Find all db files
print("=== DB files ===")
for f in glob.glob("*.db") + glob.glob("*.sqlite*"):
    print(f"  {f}")

# 2. List tables in state.db
db = sqlite3.connect('state.db')
tables = [t[0] for t in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"\nstate.db tables: {tables}")

# 3. Search for BLUESTARCO in each table
for t in tables:
    cols = [c[1] for c in db.execute(f"PRAGMA table_info({t})").fetchall()]
    # find text columns that might have symbol
    for col in cols:
        try:
            rows = db.execute(f"SELECT * FROM {t} WHERE {col} LIKE '%BLUESTARCO%'").fetchall()
            if rows:
                print(f"\n--- {t}.{col}: {len(rows)} BLUESTARCO rows ---")
                col_names = [c[1] for c in db.execute(f"PRAGMA table_info({t})").fetchall()]
                for r in rows:
                    row_dict = dict(zip(col_names, r))
                    print(f"  {row_dict}")
        except:
            pass
db.close()

# 4. Check JSON ledger files
print("\n=== JSON ledger files ===")
for f in glob.glob("*ledger*") + glob.glob("*trades*json"):
    print(f"  {f}")
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            bs = [d for d in data if 'BLUESTARCO' in str(d)]
        elif isinstance(data, dict):
            bs = {k:v for k,v in data.items() if 'BLUESTARCO' in str(k) or 'BLUESTARCO' in str(v)}
        else:
            bs = []
        if bs:
            print(f"    BLUESTARCO entries: {json.dumps(bs, indent=2, default=str)[:2000]}")
    except Exception as e:
        print(f"    Error: {e}")

# 5. Check logs
print("\n=== Log BLUESTARCO ===")
log_path = "/home/ubuntu/titan/logs/titan.log"
if os.path.exists(log_path):
    import subprocess
    result = subprocess.run(["grep", "-i", "BLUESTARCO", log_path], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    print(f"  Total log lines with BLUESTARCO: {len(lines)}")
    # Show ENTRY/EXIT/FIRED lines
    for l in lines:
        if any(k in l.upper() for k in ['ENTRY', 'EXIT', 'FIRED', 'PLACED', 'TRADE']):
            print(f"  {l[:200]}")
