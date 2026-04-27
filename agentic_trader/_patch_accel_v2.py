#!/usr/bin/env python3
"""
ACCELERATION EARLY-FIRE PATCH v2 for oi_watcher_engine.py
=========================================================
Problem: OI watcher correctly identifies spike-causing factors but enters
LATE because it waits for factor COUNT to accumulate. By time 5-7 factors
confirm, price already moved 1-4% and consolidation follows → loss.

Solution: Score factor INTENSITY as a continuous metric (0-100). When
institutional factors show extreme strength (volume surge 5x+, writer 85%+,
FOIDH at day high), enter early with fewer total factors.

Changes:
  1. Store raw writer_ratio in _oc_res/AGGR _res for intensity scoring
  2. Path 1: ACCELERATION EARLY-FIRE before conviction gate — score ≥60 → fire with 2 factors
  3. Path 1: Relax breadth gate from 4→2 when acceleration active
  4. AGGR: Init _ag_vol_surge_ratio + store raw intensities in _res
  5. AGGR: FastTrack — extreme intensity → confirm in 8-12s instead of 25-35s
"""
import sys
import shutil

FILE = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'

with open(FILE, 'r') as f:
    code = f.read()

shutil.copy2(FILE, FILE + '.bak_accel_v2')
print(f"✅ Backup: {FILE}.bak_accel_v2")

patches_applied = 0
patches_total = 6

# ═══════════════════════════════════════════════════════════
# PATCH 1: Store writer ratio in Path 1 _oc_res
# After line ~519: _oc_res['_mf_K_strong'] = _oc_K_strong_confirmed
# ═══════════════════════════════════════════════════════════
anchor1 = "                _oc_res['_mf_K_strong'] = _oc_K_strong_confirmed\n                _oc_res['_mf_M'] = _oc_vol_surge_aligned"
replace1 = "                _oc_res['_mf_K_strong'] = _oc_K_strong_confirmed\n                _oc_res['_accel_writer'] = _oc_writer_ratio if _oc_writer_ratio is not None else 0.0\n                _oc_res['_mf_M'] = _oc_vol_surge_aligned"

if anchor1 in code:
    code = code.replace(anchor1, replace1, 1)
    patches_applied += 1
    print("✅ Patch 1: Stored _accel_writer in Path 1 _oc_res")
else:
    print("❌ Patch 1 FAILED: anchor not found")
    print(f"   Looking for: _oc_res['_mf_K_strong'] = _oc_K_strong_confirmed\\n                _oc_res['_mf_M']...")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# PATCH 2: ACCELERATION EARLY-FIRE in Path 1 conviction gate
# Insert between _oi_smart_min adjustment and LOW CONVICTION check
# ═══════════════════════════════════════════════════════════
anchor2 = """                if _oi_pre_anchors >= 2:
                    _oi_smart_min = 3  # 2+ institutional anchors = high conviction
                if _oi_confirms < _oi_smart_min:"""

replace2 = """                if _oi_pre_anchors >= 2:
                    _oi_smart_min = 3  # 2+ institutional anchors = high conviction
                # ── ACCELERATION EARLY-FIRE (Apr 20) ──
                # Score factor INTENSITY: when key institutional factors show extreme
                # strength, enter early with fewer factors — catch spikes at inception.
                _oi_accel_score = 0
                _oi_accel_early_fire = False
                _oi_accel_vol = _oi_res.get('_vol_surge_ratio', 1.0)
                _oi_accel_wr = _oi_res.get('_accel_writer', 0.0)
                _oi_accel_px = abs(_oi_res.get('_mf_C_price_chg', 0))
                # Volume surge: 3x→15, 5x→25, 8x+→35  (max 35)
                if _oi_accel_vol >= 8.0: _oi_accel_score += 35
                elif _oi_accel_vol >= 5.0: _oi_accel_score += 25
                elif _oi_accel_vol >= 3.0: _oi_accel_score += 15
                # Writer conviction: 85%+→30, 75%+→20, 65%+→10  (max 30)
                if _oi_accel_wr >= 0.85: _oi_accel_score += 30
                elif _oi_accel_wr >= 0.75: _oi_accel_score += 20
                elif _oi_accel_wr >= 0.65: _oi_accel_score += 10
                # FOIDH: strong(≥75%)→25, confirmed→15  (max 25)
                if _oi_res.get('_mf_K_strong', False): _oi_accel_score += 25
                elif _oi_res.get('_mf_K', False): _oi_accel_score += 15
                # Futures buildup: strong→15, confirmed→8  (max 15)
                if _oi_res.get('_mf_F_strong', False): _oi_accel_score += 15
                elif _oi_res.get('_mf_F', False): _oi_accel_score += 8
                # Price freshness: 0.15-0.80%→10 (just starting)  (max 10)
                if 0.15 <= _oi_accel_px <= 0.80: _oi_accel_score += 10
                # EARLY FIRE: score ≥ 60 + price < 1.5% + at least 2 factors
                if _oi_accel_score >= 60 and _oi_accel_px < 1.50 and _oi_confirms >= 2:
                    _oi_accel_early_fire = True
                    _oi_smart_min = min(_oi_smart_min, _oi_confirms)
                    t._wlog(f"  ⚡ ACCEL EARLY-FIRE: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} score={_oi_accel_score} "
                               f"vol={_oi_accel_vol:.1f}x wr={_oi_accel_wr:.0%} "
                               f"K✦={_oi_res.get('_mf_K_strong', False)} "
                               f"px={_oi_accel_px:.2f}% confirms={_oi_confirms} "
                               f"— min reduced {t._oi_min_confirmations}→{_oi_confirms}")
                elif _oi_accel_score >= 30:
                    t._wlog(f"  📊 ACCEL SCORE: {_oi_sym.replace('NSE:', '')} "
                               f"score={_oi_accel_score}/60 "
                               f"vol={_oi_accel_vol:.1f}x wr={_oi_accel_wr:.0%} "
                               f"px={_oi_accel_px:.2f}%")
                if _oi_confirms < _oi_smart_min:"""

if anchor2 in code:
    code = code.replace(anchor2, replace2, 1)
    patches_applied += 1
    print("✅ Patch 2: Inserted ACCELERATION EARLY-FIRE in Path 1")
else:
    print("❌ Patch 2 FAILED: anchor not found")
    # Debug: check if anchor parts exist
    lines = anchor2.strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip() in code:
            print(f"   Line {i} found: {line.strip()[:60]}")
        else:
            print(f"   Line {i} MISSING: {line.strip()[:60]}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# PATCH 3: Relax breadth gate for acceleration early-fire
# Change: if _oi_rest_count < 4: → dynamic threshold
# ═══════════════════════════════════════════════════════════
anchor3 = """                # Gate 2: At least 4 from the remaining 9 factors incl B2 (breadth)
                if _oi_rest_count < 4:"""

replace3 = """                # Gate 2: At least 4 from the remaining 9 factors incl B2 (breadth)
                # Apr 20: Relax to 2 when acceleration early-fire (intensity proves conviction)
                _oi_breadth_min = 2 if _oi_accel_early_fire else 4
                if _oi_rest_count < _oi_breadth_min:"""

if anchor3 in code:
    code = code.replace(anchor3, replace3, 1)
    patches_applied += 1
    print("✅ Patch 3: Relaxed breadth gate for acceleration early-fire")
else:
    print("❌ Patch 3 FAILED: anchor not found")
    sys.exit(1)

# Also update the breadth gate log to show dynamic threshold
anchor3b = '                               f"only {_oi_rest_count}/9 rest-pool factors confirm (need ≥4) "'
replace3b = '                               f"only {_oi_rest_count}/9 rest-pool factors confirm (need ≥{_oi_breadth_min}) "'
if anchor3b in code:
    code = code.replace(anchor3b, replace3b, 1)
    print("   ✅ Patch 3b: Updated breadth gate log")

# ═══════════════════════════════════════════════════════════
# PATCH 4: Initialize _ag_vol_surge_ratio in AGGR path
# ═══════════════════════════════════════════════════════════
anchor4 = "            _ag_mf_M = False   # Volume surge alignment"
replace4 = "            _ag_mf_M = False   # Volume surge alignment\n            _ag_vol_surge_ratio = 0.0  # Apr 20: init for acceleration scoring"

if anchor4 in code:
    code = code.replace(anchor4, replace4, 1)
    patches_applied += 1
    print("✅ Patch 4: Initialized _ag_vol_surge_ratio in AGGR")
else:
    print("❌ Patch 4 FAILED: anchor not found")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# PATCH 5: Store raw intensities in AGGR _res dict
# ═══════════════════════════════════════════════════════════
anchor5 = "            _res['_ag_mf_M'] = _ag_mf_M\n            # Apr 15 RCA: Store evaluability flags"
replace5 = "            _res['_ag_mf_M'] = _ag_mf_M\n            _res['_accel_writer'] = _ag_writer_ratio if _ag_writer_ratio is not None else 0.0\n            _res['_accel_vol_surge'] = _ag_vol_surge_ratio\n            # Apr 15 RCA: Store evaluability flags"

if anchor5 in code:
    code = code.replace(anchor5, replace5, 1)
    patches_applied += 1
    print("✅ Patch 5: Stored acceleration intensities in AGGR _res")
else:
    print("❌ Patch 5 FAILED: anchor not found")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# PATCH 6: AGGR acceleration fast-track for pending confirmation
# When institutional intensity extreme → confirm in 8-12s
# ═══════════════════════════════════════════════════════════
anchor6 = "            _ag_confirm_secs = max(25.0, min(90.0, _ag_confirm_secs))\n            _ag_min_delta = max(0.10, min(0.30, _ag_min_delta))"

replace6 = """            # ── AGGR ACCELERATION FAST-TRACK (Apr 20) ──
            # When institutional intensity extreme → near-instant confirmation
            _ag_accel_wr = _res.get('_accel_writer', 0.0)
            _ag_accel_vs = _res.get('_accel_vol_surge', 0.0)
            _ag_accel_foidh = _res.get('_ag_mf_K', False)
            _ag_accel_intense = (_ag_accel_wr >= 0.75 and
                                 (_ag_accel_vs >= 3.0 or _ag_accel_foidh))
            if _ag_accel_intense:
                _ag_confirm_secs *= 0.35  # ~12s instead of 35s
                _ag_min_delta *= 0.40     # ~0.06% instead of 0.15%
                t._wlog(f"  ⚡ AGGR ACCEL FAST-TRACK: {_sym.replace('NSE:', '')} "
                           f"wr={_ag_accel_wr:.0%} vol={_ag_accel_vs:.1f}x "
                           f"K={_ag_accel_foidh} → confirm={_ag_confirm_secs:.0f}s "
                           f"delta={_ag_min_delta:.2f}%")
            # Floor: allow lower floors when acceleration is active
            _ag_min_secs_floor = 8.0 if _ag_accel_intense else 25.0
            _ag_min_delta_floor = 0.05 if _ag_accel_intense else 0.10
            _ag_confirm_secs = max(_ag_min_secs_floor, min(90.0, _ag_confirm_secs))
            _ag_min_delta = max(_ag_min_delta_floor, min(0.30, _ag_min_delta))"""

if anchor6 in code:
    code = code.replace(anchor6, replace6, 1)
    patches_applied += 1
    print("✅ Patch 6: Added AGGR acceleration fast-track")
else:
    print("❌ Patch 6 FAILED: anchor not found")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# Write patched file
# ═══════════════════════════════════════════════════════════
with open(FILE, 'w') as f:
    f.write(code)

print(f"\n{'='*60}")
print(f"✅ ALL {patches_applied}/{patches_total} patches applied successfully!")
print(f"File: {FILE}")
print(f"Backup: {FILE}.bak_accel_v2")
print(f"{'='*60}")
print(f"\nNext: python3 -m py_compile {FILE}")
