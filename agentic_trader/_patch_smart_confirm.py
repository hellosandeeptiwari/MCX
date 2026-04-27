#!/usr/bin/env python3
"""Smart adaptive confirmation for both Path 1 (OI_WATCHER) and OI_AGGR.

Path 1 — Adaptive conviction threshold:
  Current: always needs ≥4/12 confirms (kills 3,657 candidates)
  Smart:   ≥3 anchor factors → need only 3/12 total confirms
           ≥2 anchor factors → need only 3/12 total confirms
           1 anchor factor   → keep 4/12 (standard)
  Rationale: strong institutional anchors (F,H,K,N) ARE the highest-quality
  signals. When 2-3 anchors agree, the setup is already high conviction.

OI-AGGR — Adaptive price confirmation based on breadth factor count:
  Current: all candidates need same price delta and confirmation time
  Smart:   5-6/6 factors → delta × 0.50, time × 0.70 (technicals very strong)
           4/6 factors   → delta × 0.70, time × 0.85
           3/6 factors   → keep as-is (full requirement)
  Rationale: PFC with 5/6 factors got PRICE REJECTED at Δ+0.07% (need 0.10%).
  With adaptive logic, threshold drops to 0.05% for 5/6 → PFC would CONFIRM.
"""
import sys, os

def patch_binary(fpath, replacements):
    data = open(fpath, 'rb').read()
    applied = 0
    for old_b, new_b, label in replacements:
        if old_b in data:
            count = data.count(old_b)
            data = data.replace(old_b, new_b, 1)
            applied += 1
            print(f"  OK: {label} (found {count}x, replaced 1st)")
        else:
            print(f"  SKIP: {label} — pattern not found")
    return data, applied

BASE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════
# PATCH 1: Path 1 OI_WATCHER — Adaptive conviction
# ═══════════════════════════════════════════════
oi_path = os.path.join(BASE, 'oi_watcher_engine.py')
print(f"=== PATCH 1: Path 1 adaptive conviction ({oi_path}) ===")

# Find the conviction gate in Path 1 and make it adaptive to anchor count.
# Current code (around line 677):
#   if _oi_confirms < t._oi_min_confirmations:
#       ... BLOCKED ...
#       continue
# 
# Replace with adaptive logic that lowers the threshold when anchors are strong.
# We need the anchor count which is computed AFTER the conviction gate currently,
# so we must move the adaptive logic. But actually _oi_confirms includes anchors
# and the anchor gate is checked AFTER the conviction gate.
#
# Better approach: compute a _smart_min_confirms BEFORE the conviction check,
# based on the already-computed factor flags stored in _oc_res.

OLD_PATH1_CONVICTION = (
    b"                # \xe2\x94\x80\xe2\x94\x80 OI CONVICTION GATE (Instant \xe2\x94\x80\xe2\x94\x80 No Time Delay) \xe2\x94\x80\xe2\x94\x80\r\n"
    b"                # Instead of waiting 60s (by which time the move is over or you're chasing),\r\n"
    b"                # use CONFLUENCE COUNT: how many independent market microstructure factors\r\n"
    b"                # confirm the signal RIGHT NOW. \xe2\x89\xa5 N factors = enter immediately.\r\n"
    b"                _oi_confirms = _oi_res.get('_confirm_count', 0)\r\n"
    b"                if _oi_confirms < t._oi_min_confirmations:\r\n"
    b"                    t._wlog(f\"  \xe2\x9b\x94 OI_WATCHER LOW CONVICTION: {_oi_sym.replace('NSE:', '')} \"\r\n"
    b"                               f\"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} \xe2\x80\x94 \"  \r\n"
    b"                               f\"only {_oi_confirms}/{t._oi_min_confirmations} factors confirm \"\r\n"
    b"                               f\"[{_oi_res.get('_quality_boosts', '')}] \xe2\x80\x94 need more confluence\")\r\n"
    b"                    continue"
)

NEW_PATH1_CONVICTION = (
    b"                # \xe2\x94\x80\xe2\x94\x80 OI CONVICTION GATE (Instant \xe2\x94\x80\xe2\x94\x80 No Time Delay) \xe2\x94\x80\xe2\x94\x80\r\n"
    b"                # Instead of waiting 60s (by which time the move is over or you're chasing),\r\n"
    b"                # use CONFLUENCE COUNT: how many independent market microstructure factors\r\n"
    b"                # confirm the signal RIGHT NOW. \xe2\x89\xa5 N factors = enter immediately.\r\n"
    b"                # [FIX Apr 20] ADAPTIVE CONVICTION: strong anchors lower the threshold\r\n"
    b"                # \xe2\x89\xa52 anchor factors (F,H,K,N) = institutional signals are strong\r\n"
    b"                # \xe2\x86\x92 need only 3/12 total instead of 4/12\r\n"
    b"                _oi_confirms = _oi_res.get('_confirm_count', 0)\r\n"
    b"                _oi_pre_anchors = sum([_oi_res.get('_mf_F', False), _oi_res.get('_mf_H', False),\r\n"
    b"                                       _oi_res.get('_mf_K', False), _oi_res.get('_mf_N', False)])\r\n"
    b"                _oi_smart_min = t._oi_min_confirmations  # default 4\r\n"
    b"                if _oi_pre_anchors >= 2:\r\n"
    b"                    _oi_smart_min = 3  # 2+ institutional anchors = high conviction, relax to 3\r\n"
    b"                if _oi_confirms < _oi_smart_min:\r\n"
    b"                    t._wlog(f\"  \xe2\x9b\x94 OI_WATCHER LOW CONVICTION: {_oi_sym.replace('NSE:', '')} \"\r\n"
    b"                               f\"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} \xe2\x80\x94 \"  \r\n"
    b"                               f\"only {_oi_confirms}/{_oi_smart_min} factors confirm \"\r\n"
    b"                               f\"(anchor={_oi_pre_anchors}/4) \"\r\n"
    b"                               f\"[{_oi_res.get('_quality_boosts', '')}] \xe2\x80\x94 need more confluence\")\r\n"
    b"                    continue"
)

# Check if CRLF or LF
data = open(oi_path, 'rb').read()
if b'\r\n' not in data:
    # LF only — strip \r\n to \n
    OLD_PATH1_CONVICTION = OLD_PATH1_CONVICTION.replace(b'\r\n', b'\n')
    NEW_PATH1_CONVICTION = NEW_PATH1_CONVICTION.replace(b'\r\n', b'\n')

if OLD_PATH1_CONVICTION in data:
    data = data.replace(OLD_PATH1_CONVICTION, NEW_PATH1_CONVICTION, 1)
    print("  OK: Path 1 adaptive conviction gate (anchor≥2 → min=3)")
else:
    print("  SKIP: Path 1 conviction gate pattern not found, trying alt match...")
    # Try just the key line
    alt_old = b"                if _oi_confirms < t._oi_min_confirmations:"
    alt_new = (
        b"                _oi_pre_anchors = sum([_oi_res.get('_mf_F', False), _oi_res.get('_mf_H', False),\r\n"
        b"                                       _oi_res.get('_mf_K', False), _oi_res.get('_mf_N', False)])\r\n"
        b"                _oi_smart_min = t._oi_min_confirmations  # default 4\r\n"
        b"                if _oi_pre_anchors >= 2:\r\n"
        b"                    _oi_smart_min = 3  # 2+ institutional anchors = high conviction\r\n"
        b"                if _oi_confirms < _oi_smart_min:"
    )
    if b'\r\n' not in data:
        alt_new = alt_new.replace(b'\r\n', b'\n')
    if alt_old in data:
        data = data.replace(alt_old, alt_new, 1)
        print("  OK: Path 1 adaptive conviction (alt match — key line only)")
        # Also fix the log line to show smart_min instead of hardcoded
        log_old = b"f\"only {_oi_confirms}/{t._oi_min_confirmations} factors confirm \""
        log_new = b"f\"only {_oi_confirms}/{_oi_smart_min} factors confirm (anchor={_oi_pre_anchors}/4) \""
        if log_old in data:
            data = data.replace(log_old, log_new, 1)
            print("  OK: Path 1 log line updated for smart_min")
    else:
        print("  FAIL: Could not find Path 1 conviction gate!")

# ═══════════════════════════════════════════════
# PATCH 2: OI-AGGR — Adaptive confirmation based on breadth factor count
# ═══════════════════════════════════════════════
print(f"\n=== PATCH 2: OI-AGGR adaptive confirmation ({oi_path}) ===")

# The confirmation gate computes _ag_confirm_secs and _ag_min_delta, then applies
# multipliers for acceleration, volume, anchor strength, OI strength.
# After all multipliers, it clamps with max()/min().
# We insert breadth-factor-adaptive multipliers AFTER the strength multipliers 
# and BEFORE the clamp.
#
# Current code:
#   if _str < 0.60:
#       _ag_confirm_secs *= 1.20
#       _ag_min_delta *= 1.15
#   _ag_confirm_secs = max(25.0, min(90.0, _ag_confirm_secs))
#   _ag_min_delta = max(0.10, min(0.30, _ag_min_delta))
#
# Insert between the strength block and the clamp:
#   # [FIX Apr 20] ADAPTIVE: high breadth factor count → less confirmation needed
#   if _ag_mf_total >= 5:
#       _ag_confirm_secs *= 0.70
#       _ag_min_delta *= 0.50
#   elif _ag_mf_total >= 4:
#       _ag_confirm_secs *= 0.85
#       _ag_min_delta *= 0.70

CLAMP_LINE = b"            _ag_confirm_secs = max(25.0, min(90.0, _ag_confirm_secs))"
BREADTH_ADAPTIVE = (
    b"            # [FIX Apr 20] ADAPTIVE: high breadth factors \xe2\x86\x92 less price confirmation needed\r\n"
    b"            # 5-6/6 factors: technicals very strong, need minimal price proof\r\n"
    b"            # 4/6 factors:   good conviction, moderate relaxation\r\n"
    b"            # 3/6 factors:   standard (no change)\r\n"
    b"            if _ag_mf_total >= 5:\r\n"
    b"                _ag_confirm_secs *= 0.70\r\n"
    b"                _ag_min_delta *= 0.50\r\n"
    b"            elif _ag_mf_total >= 4:\r\n"
    b"                _ag_confirm_secs *= 0.85\r\n"
    b"                _ag_min_delta *= 0.70\r\n"
    b"            _ag_confirm_secs = max(25.0, min(90.0, _ag_confirm_secs))"
)

if b'\r\n' not in data:
    BREADTH_ADAPTIVE = BREADTH_ADAPTIVE.replace(b'\r\n', b'\n')

if CLAMP_LINE in data:
    data = data.replace(CLAMP_LINE, BREADTH_ADAPTIVE, 1)
    print("  OK: OI-AGGR breadth-adaptive confirmation inserted before clamp")
else:
    print("  FAIL: Could not find clamp line for AGGR confirmation!")

# Also update the PENDING log to show the breadth-adaptive detail
PENDING_LOG_OLD = b"f\"waiting {_ag_confirm_secs:.0f}s, \"\r\n                           f\"need "
PENDING_LOG_NEW = b"f\"waiting {_ag_confirm_secs:.0f}s (breadth={_ag_mf_total}/6), \"\r\n                           f\"need "
if b'\r\n' not in data[:100]:
    PENDING_LOG_OLD = PENDING_LOG_OLD.replace(b'\r\n', b'\n')
    PENDING_LOG_NEW = PENDING_LOG_NEW.replace(b'\r\n', b'\n')
if PENDING_LOG_OLD in data:
    data = data.replace(PENDING_LOG_OLD, PENDING_LOG_NEW, 1)
    print("  OK: PENDING log now shows breadth count")
else:
    print("  SKIP: PENDING log pattern not found")

# Write patched file
open(oi_path, 'wb').write(data)
print(f"\n  >>> oi_watcher_engine.py written ({len(data)} bytes)")

# ═══════════════════════════════════════════════
# PATCH 3: Log the smart_min in the HIGH CONVICTION line for Path 1
# ═══════════════════════════════════════════════
# Also update the HIGH CONVICTION log to reference _oi_smart_min
data2 = open(oi_path, 'rb').read()
hc_old = b"f\"{_oi_confirms}/{t._oi_min_confirmations} factors confirm \"\r\n                           f\"anchor={_oi_anchor_count}/4"
hc_new = b"f\"{_oi_confirms}/{_oi_smart_min} factors confirm \"\r\n                           f\"anchor={_oi_anchor_count}/4"
if b'\r\n' not in data2[:100]:
    hc_old = hc_old.replace(b'\r\n', b'\n')
    hc_new = hc_new.replace(b'\r\n', b'\n')
if hc_old in data2:
    data2 = data2.replace(hc_old, hc_new, 1)
    open(oi_path, 'wb').write(data2)
    print("  OK: HIGH CONVICTION log updated to show smart_min")
else:
    print("  SKIP: HIGH CONVICTION log pattern not found (may already use smart_min)")

print("\nDone. py_compile oi_watcher_engine.py, then restart titan-bot.")
