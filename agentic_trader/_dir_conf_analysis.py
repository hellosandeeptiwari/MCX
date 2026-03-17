"""Analyze dir_conf distribution: blocked vs entered vs won/lost."""
import json, glob, os

ledger_dir = "/home/ubuntu/titan/agentic_trader/trade_ledger"
files = sorted(glob.glob(os.path.join(ledger_dir, "trade_ledger_2026-03-*.jsonl")))

# Collect all events
blocked_dir_conf = []  # (ts, symbol, dir_conf, score, reason)
entries = []  # (ts, symbol, dir_conf, score, direction, pnl)

for f in files:
    day_entries = {}  # trade_id -> entry record
    day_exits = {}   # trade_id -> exit record
    
    for line in open(f):
        try:
            r = json.loads(line.strip())
        except:
            continue
        
        ev = r.get("event", "")
        
        # Blocked by EARLY_DIR
        if ev == "SCAN" and "WATCHER_EARLY_DIR" in r.get("outcome", ""):
            reason = r.get("reason", "")
            # Extract dir_conf from reason like "Early market dir_conf=37%<45%"
            import re
            m = re.search(r'dir_conf=(\d+)', reason)
            dc = int(m.group(1)) if m else 0
            blocked_dir_conf.append((r.get("ts","")[:16], r.get("symbol",""), dc, r.get("score",0), reason))
        
        # Entries with direction_confidence (only recent files)
        if ev == "ENTRY":
            # dir_conf might not be stored — check
            dc = r.get("direction_confidence", r.get("dir_conf", -1))
            entries.append({
                "ts": r.get("ts","")[:16],
                "symbol": r.get("underlying", r.get("symbol", "")),
                "dir_conf": dc,
                "score": r.get("smart_score", r.get("final_score", 0)),
                "direction": r.get("direction", ""),
                "source": r.get("source", ""),
                "pnl": None  # will fill from exits
            })

print("=" * 70)
print("BLOCKED BY A2-EARLY-DIR (all days)")
print("=" * 70)
if blocked_dir_conf:
    for ts, sym, dc, score, reason in sorted(blocked_dir_conf, key=lambda x: x[2]):
        print(f"  {ts}  {sym:20s}  dir_conf={dc:3d}%  score={score:.0f}")
    print(f"\nTotal blocked: {len(blocked_dir_conf)}")
    dcs = [x[2] for x in blocked_dir_conf]
    print(f"Dir_conf range: {min(dcs)}-{max(dcs)}%")
    print(f"Dir_conf avg: {sum(dcs)/len(dcs):.1f}%")
    scores = [x[3] for x in blocked_dir_conf]
    print(f"Score range: {min(scores):.0f}-{max(scores):.0f}")
    # Bucket by dir_conf
    for bucket_min, bucket_max in [(0,20),(20,30),(30,40),(40,50)]:
        n = sum(1 for dc in dcs if bucket_min <= dc < bucket_max)
        if n > 0:
            ss = [x[3] for x in blocked_dir_conf if bucket_min <= x[2] < bucket_max]
            print(f"  dir_conf {bucket_min}-{bucket_max}%: {n} blocked, avg score={sum(ss)/len(ss):.0f}")
else:
    print("  No EARLY_DIR blocks found in ledger.")

print()
print("=" * 70) 
print("TODAY'S BLOCKS (Mar 17)")
print("=" * 70)
today_blocks = [x for x in blocked_dir_conf if "2026-03-17" in x[0]]
for ts, sym, dc, score, reason in today_blocks:
    print(f"  {ts}  {sym:20s}  dir_conf={dc:3d}%  score={score:.0f}")
print(f"Total today: {len(today_blocks)}")

# Also show today's log for watcher blocks
print()
print("=" * 70)
print("TODAY'S WATCHER LOG - EARLY_DIR blocks from titan.log")  
print("=" * 70)
import subprocess
result = subprocess.run(
    ["grep", "-i", "EARLY-DIR", "/home/ubuntu/titan/agentic_trader/bot_debug.log"],
    capture_output=True, text=True
)
for line in result.stdout.strip().split("\n")[-30:]:
    if line:
        print(f"  {line.strip()}")
