"""Root cause: VOLUME_SURGE missing from ledger.
Hypothesis: watcher_debug.log spans multiple DAYS (not cleared between days).
Check how many separate sessions exist in the log, and which dates' ledgers have those order_ids.
"""
import re, os, glob, json
from collections import defaultdict

LOG = '/home/ubuntu/titan/agentic_trader/watcher_debug.log'
LEDGER_DIR = '/home/ubuntu/titan/agentic_trader/trade_ledger'

# 1. Find all time resets (backward jumps) in watcher_debug.log
print("=" * 70)
print("STEP 1: Identify separate sessions via time resets")
print("=" * 70)
prev_time = None
sessions = []
current_session_start = None
current_session_lines = 0
session_first_trade = None
session_last_trade = None
time_pat = re.compile(r'^\[(\d{2}:\d{2}:\d{2}\.\d+)\]')

with open(LOG, 'r') as f:
    for i, line in enumerate(f, 1):
        m = time_pat.match(line)
        if m:
            t = m.group(1)
            if prev_time and t < prev_time and (prev_time[:2] != '09' or t[:2] != '09'):
                # Time went backwards = new session
                sessions.append({
                    'start_line': current_session_start,
                    'start_time': sessions[-1]['end_time'] if sessions else '??',
                    'end_time': prev_time,
                    'lines': current_session_lines,
                    'first_trade': session_first_trade,
                    'last_trade': session_last_trade,
                })
                current_session_start = i
                current_session_lines = 0
                session_first_trade = None
                session_last_trade = None
            if current_session_start is None:
                current_session_start = i
            prev_time = t
            current_session_lines += 1
            
            if 'TRADE PLACED' in line:
                if session_first_trade is None:
                    session_first_trade = t
                session_last_trade = t

# Last session
if current_session_start:
    sessions.append({
        'start_line': current_session_start,
        'start_time': sessions[-1]['end_time'] if sessions else '??',
        'end_time': prev_time,
        'lines': current_session_lines,
        'first_trade': session_first_trade,
        'last_trade': session_last_trade,
    })

print(f"Found {len(sessions)} distinct sessions in watcher_debug.log\n")
for i, s in enumerate(sessions):
    print(f"  Session {i+1}: lines ~{s['start_line']} | time {s.get('start_time','?')[:8]} → {s['end_time'][:8]} | "
          f"{s['lines']} lines | trades: {s.get('first_trade','none')[:8] if s.get('first_trade') else 'none'} → {s.get('last_trade','none')[:8] if s.get('last_trade') else 'none'}")

# 2. Extract ALL TRADE PLACED order IDs with their session number
print("\n" + "=" * 70)
print("STEP 2: Extract order IDs per session, find VOLUME_SURGE ones")
print("=" * 70)

trade_pat = re.compile(r'TRADE PLACED.*?order=([\w]+).*?(?:setup|trigger)=(\S+)')
session_boundaries = []
prev_time2 = None
session_idx = 0

order_sessions = {}  # order_id -> session_idx
vol_surge_orders = []

with open(LOG, 'r') as f:
    line_num = 0
    for line in f:
        line_num += 1
        m = time_pat.match(line)
        if m:
            t = m.group(1)
            if prev_time2 and t < prev_time2 and (prev_time2[:2] != '09' or t[:2] != '09'):
                session_idx += 1
            prev_time2 = t
        
        if 'TRADE PLACED' in line:
            tm = trade_pat.search(line)
            if tm:
                oid = tm.group(1)
                setup = tm.group(2)
                order_sessions[oid] = session_idx
                if 'VOLUME' in setup.upper() or 'VOLUME' in line.upper():
                    vol_surge_orders.append((oid, session_idx, line.strip()[:120]))

print(f"\nTotal TRADE PLACED entries: {len(order_sessions)}")
print(f"VOLUME_SURGE orders: {len(vol_surge_orders)}")
print(f"\nVolume surge per session:")
vol_by_session = defaultdict(int)
for _, sidx, _ in vol_surge_orders:
    vol_by_session[sidx] += 1
for sidx in sorted(vol_by_session):
    print(f"  Session {sidx+1}: {vol_by_session[sidx]} VOLUME_SURGE orders")

# 3. Check ALL available ledger dates for these order IDs
print("\n" + "=" * 70)
print("STEP 3: Search ALL ledger dates for VOLUME_SURGE order IDs")
print("=" * 70)

ledger_files = sorted(glob.glob(os.path.join(LEDGER_DIR, 'trade_ledger_*.jsonl')))
print(f"\nAvailable ledger files: {len(ledger_files)}")
for lf in ledger_files:
    basename = os.path.basename(lf)
    size = os.path.getsize(lf)
    print(f"  {basename} ({size:,} bytes)")

# Build set of ALL order_ids across ALL ledger files
all_ledger_oids = {}  # order_id -> date
for lf in ledger_files:
    date_str = os.path.basename(lf).replace('trade_ledger_', '').replace('.jsonl', '')
    with open(lf, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                oid = rec.get('order_id', '')
                if oid:
                    all_ledger_oids[oid] = date_str
            except:
                pass

print(f"\nTotal unique order_ids across ALL ledger files: {len(all_ledger_oids)}")

# Check each volume surge order
found_count = 0
not_found_count = 0
found_dates = defaultdict(int)
print(f"\nVOLUME_SURGE order_id lookup across ALL dates:")
for oid, sidx, snippet in vol_surge_orders[:30]:  # Show first 30
    if oid in all_ledger_oids:
        found_count += 1
        found_dates[all_ledger_oids[oid]] += 1
        print(f"  ✅ {oid} → FOUND in {all_ledger_oids[oid]} (session {sidx+1})")
    else:
        not_found_count += 1
        print(f"  ❌ {oid} → NOT FOUND in any ledger (session {sidx+1})")

if len(vol_surge_orders) > 30:
    remaining = vol_surge_orders[30:]
    for oid, sidx, snippet in remaining:
        if oid in all_ledger_oids:
            found_count += 1
            found_dates[all_ledger_oids[oid]] += 1
        else:
            not_found_count += 1

print(f"\n--- SUMMARY ---")
print(f"VOLUME_SURGE orders FOUND in ledger: {found_count}")
print(f"VOLUME_SURGE orders NOT FOUND anywhere: {not_found_count}")
if found_dates:
    print(f"Found in dates: {dict(found_dates)}")

# 4. Also check ALL 81 non-volume orders
print("\n" + "=" * 70)
print("STEP 4: ALL watcher orders — check which dates they appear in")
print("=" * 70)
all_found = 0
all_missing = 0
found_by_date = defaultdict(int)
missing_by_session = defaultdict(int)
for oid, sidx in order_sessions.items():
    if oid in all_ledger_oids:
        all_found += 1
        found_by_date[all_ledger_oids[oid]] += 1
    else:
        all_missing += 1
        missing_by_session[sidx] += 1

print(f"ALL watcher trades FOUND in ledger: {all_found} / {len(order_sessions)}")
print(f"Found by ledger date: {dict(found_by_date)}")
print(f"Missing by session: {dict({f'session_{k+1}': v for k, v in missing_by_session.items()})}")

# 5. Check if watcher_debug.log was created on a previous date
import subprocess
result = subprocess.run(['stat', LOG], capture_output=True, text=True)
print(f"\n--- WATCHER LOG FILE STAT ---")
print(result.stdout[:500])
