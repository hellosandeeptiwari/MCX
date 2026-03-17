"""Why did VOLUME_SURGE not fire on Mar 17 (Session 9)?
Session 9 starts at line ~13730 in watcher_debug.log.
Check what happened to VOLUME_SURGE gate checks today.
"""
import re
from collections import Counter

LOG = '/home/ubuntu/titan/agentic_trader/watcher_debug.log'

# Session 9 starts at line 13730
START_LINE = 13730
time_pat = re.compile(r'^\[(\d{2}:\d{2}:\d{2}\.\d+)\]')

lines = []
with open(LOG, 'r') as f:
    for i, line in enumerate(f, 1):
        if i >= START_LINE:
            lines.append(line.rstrip())

print(f"Session 9 (Mar 17): {len(lines)} lines from line {START_LINE}")

# Count VOLUME_SURGE mentions
vol_mentions = [l for l in lines if 'VOLUME_SURGE' in l]
print(f"\nVOLUME_SURGE mentions in today's session: {len(vol_mentions)}")

# Categorize VOLUME_SURGE lines
categories = Counter()
for l in vol_mentions:
    if 'TRADE PLACED' in l:
        categories['TRADE PLACED'] += 1
    elif 'TRADE FAIL' in l or 'FAILED' in l:
        categories['TRADE FAILED'] += 1
    elif 'GATE CHECK' in l:
        categories['GATE CHECK'] += 1
    elif 'LATE-DECAY' in l:
        categories['LATE-DECAY'] += 1
    elif 'DUPLICATE' in l:
        categories['DUPLICATE BLOCKED'] += 1
    elif 'VETO' in l or 'BLOCKED' in l or 'REJECT' in l:
        categories['BLOCKED/REJECTED'] += 1
    elif 'scanning' in l.lower() or 'detected' in l.lower() or 'trigger' in l.lower():
        categories['DETECTION'] += 1
    else:
        categories['OTHER'] += 1

print(f"\nVOLUME_SURGE breakdown today:")
for cat, cnt in categories.most_common():
    print(f"  {cat}: {cnt}")

# Show first 40 VOLUME_SURGE lines
print(f"\n--- First 40 VOLUME_SURGE lines ---")
for l in vol_mentions[:40]:
    print(l[:200])

# Also check: what DID fire as TRADE PLACED today?
placed_today = [l for l in lines if 'TRADE PLACED' in l]
print(f"\n--- ALL TRADE PLACED today ({len(placed_today)}) ---")
for l in placed_today:
    print(l[:200])

# Check GATE CHECK lines for VOLUME_SURGE specifically
gate_vol = [l for l in lines if 'GATE CHECK' in l and 'VOLUME_SURGE' in l]
print(f"\n--- VOLUME_SURGE GATE CHECKS today ({len(gate_vol)}) ---")
for l in gate_vol[:20]:
    print(l[:250])

# Check for VOLUME_SURGE followed by failure/block
print(f"\n--- VOLUME_SURGE blocks/failures ---")
block_lines = [l for l in vol_mentions if any(x in l.upper() for x in ['FAIL', 'BLOCK', 'REJECT', 'SKIP', 'VETO', 'MAX_TRADES', 'RISK', 'DUPLICATE'])]
for l in block_lines[:30]:
    print(l[:250])

# Check if volume_surge triggers are even being detected by kite_ticker
print(f"\n--- Lines mentioning vol_surge or volume from kite_ticker signals ---")
vol_detect = [l for l in lines if 'volume' in l.lower() and ('surge' in l.lower() or 'spike' in l.lower())]
for l in vol_detect[:20]:
    print(l[:200])
