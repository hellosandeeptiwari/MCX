#!/usr/bin/env python3
"""Patch: Change aggressive breadth from 2/6 to 3/6 + raise exposure cap to 200%.

1) oi_watcher_engine.py: _ag_min_breadth = 2 → 3
2) risk_governor.py: max_total_exposure_pct = 100.0 → 200.0 (PAPER mode needs room)
"""
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/titan/agentic_trader'

# ============================================================
# FIX 1: oi_watcher_engine.py — breadth 2/6 → 3/6
# ============================================================
oi_path = f'{BASE}/oi_watcher_engine.py'
with open(oi_path, 'rb') as f:
    oi_data = f.read()

old_breadth = b'_ag_min_breadth = 2'
new_breadth = b'_ag_min_breadth = 3'

if old_breadth in oi_data:
    oi_data = oi_data.replace(old_breadth, new_breadth, 1)
    with open(oi_path, 'wb') as f:
        f.write(oi_data)
    print("FIX 1: _ag_min_breadth = 2 -> 3 (breadth gate now 3/6)")
else:
    if new_breadth in oi_data:
        print("FIX 1: SKIP — already set to 3")
    else:
        print("ERROR: Could not find _ag_min_breadth = 2")
        sys.exit(1)

# Also update the comment
old_comment = b'# Gate 2: Breadth \xe2\x80\x94 2/6 for all entries'
new_comment = b'# Gate 2: Breadth \xe2\x80\x94 3/6 for all entries (Apr 20, raised from 2)'
with open(oi_path, 'rb') as f:
    oi_data2 = f.read()
if old_comment in oi_data2:
    oi_data2 = oi_data2.replace(old_comment, new_comment, 1)
    with open(oi_path, 'wb') as f:
        f.write(oi_data2)
    print("  Updated comment to reflect 3/6")

# ============================================================
# FIX 2: risk_governor.py — exposure 100% → 200%
# ============================================================
rg_path = f'{BASE}/risk_governor.py'
with open(rg_path, 'rb') as f:
    rg_data = f.read()

old_exp = b'max_total_exposure_pct: float = 100.0'
new_exp = b'max_total_exposure_pct: float = 200.0'

if old_exp in rg_data:
    rg_data = rg_data.replace(old_exp, new_exp, 1)
    with open(rg_path, 'wb') as f:
        f.write(rg_data)
    print("FIX 2: max_total_exposure_pct = 100.0 -> 200.0 (unblock watcher trades)")
else:
    if b'max_total_exposure_pct: float = 200.0' in rg_data:
        print("FIX 2: SKIP — already set to 200.0")
    else:
        print("ERROR: Could not find max_total_exposure_pct = 100.0")
        sys.exit(1)

print("\nDone. Compile both files then restart titan-bot.")
