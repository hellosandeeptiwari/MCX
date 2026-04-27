#!/usr/bin/env python3
"""
ACCELERATION EARLY-FIRE PATCH for oi_watcher_engine.py
=====================================================
Problem: OI watcher correctly identifies spike-causing factors but enters
LATE because it waits for factor COUNT to accumulate. By time 5-7 factors
confirm, price already moved 1-4% and consolidation follows → loss.

Example from Apr 18:
  PIIND:  price=-3.73% at entry, VSRG=5x, FOIDH=100% → too late
  JINDALSTEL: price=+0.94%, VSRG=9.3x → entered tail of spike
  KPITTECH: price=-2.05%, FOIDH=90% → late 

Fix: Score factor INTENSITY (not just count). When key institutional
acceleration signals are extreme, bypass count-based gates and fire early.

Changes:
1. Store raw intensity values (writer_ratio, vol_surge_ratio, foidh_pct)
   in _oc_res / _res for both Path 1 and AGGR
2. Path 1: Acceleration early-fire bypass before conviction gate
   - Computes accel_score from intensity of writer, volume, FOIDH, price freshness
   - If accel_score >= 60 AND >=1 anchor AND >=2 confirms → reduce min_confirms
3. AGGR: When extreme institutional intensity detected, slash confirmation
   hold time (45s → 15s) and price delta (0.15% → 0.05%)
"""

import sys
import re

TARGET = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'

def patch():
    with open(TARGET, 'r') as f:
        code = f.read()
    
    original_len = len(code)
    patches_applied = 0
    
    # ================================================================
    # PATCH 1: Store raw intensity values in Path 1 _oc_res
    # Insert after: _oc_res['_vol_surge_ratio'] = _oc_vol_surge_ratio
    # ================================================================
    anchor1 = "_oc_res['_vol_surge_ratio'] = _oc_vol_surge_ratio"
    if anchor1 in code:
        insert1 = """
                # [Apr 20] Store raw intensity values for acceleration scoring
                _oc_res['_accel_writer_ratio'] = _oc_writer_ratio if _oc_writer_ratio is not None else 0.0
                _oc_res['_accel_vol_surge'] = _oc_vol_surge_ratio
                _oc_res['_accel_foidh_pct'] = _oc_fk_pos * 100 if '_oc_fk_pos' in dir() and isinstance(_oc_fk_pos, (int, float)) else 0.0"""
        # Actually _oc_fk_pos is a local var that may not exist if the K block didn't run
        # Safer: store it explicitly after the K block
        # Instead, just use a try/except approach
        insert1 = """
                # [Apr 20] Store raw intensity values for acceleration scoring
                _oc_res['_accel_writer_ratio'] = _oc_writer_ratio if _oc_writer_ratio is not None else 0.0
                _oc_res['_accel_vol_surge'] = _oc_vol_surge_ratio"""
        code = code.replace(anchor1, anchor1 + insert1, 1)
        patches_applied += 1
        print(f"  OK: Patch 1a — stored writer_ratio + vol_surge in Path 1 _oc_res")
    else:
        print(f"  FAIL: Patch 1a — anchor not found: {anchor1[:60]}")
        return False

    # Store FOIDH % after K block — need to capture _oc_fk_pos
    # Insert after _oc_K_strong_confirmed is set to True (the first time)
    # Best anchor: right after _mf_K_strong is stored
    anchor1b = "_oc_res['_mf_K_strong'] = _oc_K_strong_confirmed"
    if anchor1b in code:
        insert1b = """
                # [Apr 20] Store raw FOIDH position for acceleration scoring
                _oc_res['_accel_foidh_pct'] = getattr(_oc_fk_pos, '__float__', lambda: 0.0)() * 100 if '_oc_fk_pos' in dir() else 0.0"""
        # Hmm, _oc_fk_pos is a local — need to store it at computation time
        # Actually simpler: initialize _oc_fk_pos = 0.0 at the top of K block 
        # and then it's always available. Let me take a different approach.
        pass  # Will handle below
    
    # Better approach for FOIDH: initialize _oc_fk_pos at start of factor K
    # and store after all K computation
    anchor_k_init = "_oc_K_confirmed = False\n                _oc_K_strong_confirmed = False"
    if anchor_k_init in code:
        replacement_k_init = "_oc_K_confirmed = False\n                _oc_K_strong_confirmed = False\n                _oc_fk_pos_raw = 0.0  # [Apr 20] Raw FOIDH position for acceleration scoring"
        code = code.replace(anchor_k_init, replacement_k_init, 1)
        patches_applied += 1
        print(f"  OK: Patch 1b — initialized _oc_fk_pos_raw in Path 1")
    else:
        print(f"  WARN: Patch 1b — K init anchor not found, trying alt")

    # Now store _oc_fk_pos_raw where _oc_fk_pos is computed
    # After: _oc_fk_pos = (_oc_fk_oi - _oc_fk_low) / _oc_fk_range
    anchor_fk = "_oc_fk_pos = (_oc_fk_oi - _oc_fk_low) / _oc_fk_range"
    if anchor_fk in code:
        code = code.replace(
            anchor_fk,
            anchor_fk + "\n                                _oc_fk_pos_raw = _oc_fk_pos  # [Apr 20] Store for accel scoring",
            1  # Only first occurrence (Path 1)
        )
        patches_applied += 1
        print(f"  OK: Patch 1c — captured _oc_fk_pos_raw in Path 1")
    else:
        print(f"  WARN: Patch 1c — _oc_fk_pos anchor not found")

    # Store FOIDH in _oc_res after _mf_K_strong
    if anchor1b in code:
        insert1b_v2 = "\n                _oc_res['_accel_foidh_pct'] = _oc_fk_pos_raw * 100 if _oc_fk_pos_raw > 0 else 0.0"
        code = code.replace(anchor1b, anchor1b + insert1b_v2, 1)
        patches_applied += 1
        print(f"  OK: Patch 1d — stored _accel_foidh_pct in Path 1 _oc_res")

    # ================================================================
    # PATCH 2: Store raw intensity values in AGGR _res
    # Insert after: _res['_ag_mf_K'] = _ag_mf_K
    # ================================================================
    anchor2 = "_res['_ag_mf_K'] = _ag_mf_K\n            _res['_ag_mf_M'] = _ag_mf_M"
    if anchor2 in code:
        insert2 = """
            # [Apr 20] Store raw intensity values for acceleration fast-track
            _res['_ag_accel_writer_ratio'] = _ag_writer_ratio if _ag_writer_ratio is not None else 0.0
            _res['_ag_accel_vol_surge'] = _ag_vol_surge_ratio if '_ag_vol_surge_ratio' in dir() else 0.0
            _res['_ag_accel_foidh_pct'] = _ag_fk_pos_raw * 100 if '_ag_fk_pos_raw' in dir() and _ag_fk_pos_raw > 0 else 0.0"""
        code = code.replace(anchor2, anchor2 + insert2, 1)
        patches_applied += 1
        print(f"  OK: Patch 2a — stored intensity values in AGGR _res")
    else:
        print(f"  FAIL: Patch 2a — anchor not found")

    # Initialize _ag_fk_pos_raw and _ag_vol_surge_ratio at start of AGGR factors
    # AGGR K block: _ag_mf_K = False
    anchor2b = "_ag_mf_K = False   # Futures conviction"
    if anchor2b in code:
        code = code.replace(
            anchor2b,
            anchor2b + "\n            _ag_fk_pos_raw = 0.0   # [Apr 20] Raw FOIDH position for acceleration\n            _ag_vol_surge_ratio = 0.0  # [Apr 20] Raw volume surge for acceleration",
            1
        )
        patches_applied += 1
        print(f"  OK: Patch 2b — initialized _ag_fk_pos_raw + _ag_vol_surge_ratio in AGGR")
    else:
        print(f"  WARN: Patch 2b — AGGR K init anchor not found")

    # Capture _ag_fk_pos → _ag_fk_pos_raw in AGGR K block
    anchor2c = "_ag_fk_pos = (_ag_fk_oi - _ag_fk_low) / _ag_fk_range"
    if anchor2c in code:
        code = code.replace(
            anchor2c,
            anchor2c + "\n                            _ag_fk_pos_raw = _ag_fk_pos  # [Apr 20] Store for accel scoring",
            1
        )
        patches_applied += 1
        print(f"  OK: Patch 2c — captured _ag_fk_pos_raw in AGGR")

    # ================================================================
    # PATCH 3: ACCELERATION EARLY-FIRE in Path 1
    # Insert BEFORE the conviction gate at:
    #   "# ── OI CONVICTION GATE (Instant — No Time Delay) ──"
    # This allows high-intensity signals to fire with fewer factors
    # ================================================================
    anchor3 = "                # ── OI CONVICTION GATE (Instant — No Time Delay) ──"
    if anchor3 not in code:
        # Try alternate anchor
        anchor3 = "# ── OI CONVICTION GATE"
        
    if anchor3 in code:
        accel_block = """                # ── ACCELERATION EARLY-FIRE (Apr 20) ──
                # When key institutional factors are at extreme intensity,
                # don't wait for all factors to accumulate — enter early in the spike.
                # Score: writer conviction (institutional skin) + volume surge 
                #   (real-time flow) + FOIDH (OI at day high = active positioning)
                # If score >= 60 AND price is fresh (< 1.0%), relax min confirmations.
                _oi_accel_score = 0
                _oi_accel_writer = _oi_res.get('_accel_writer_ratio', 0) or 0
                _oi_accel_vol = _oi_res.get('_accel_vol_surge', 0) or 0
                _oi_accel_foidh = _oi_res.get('_accel_foidh_pct', 0) or 0
                _oi_accel_price = abs(_oi_mf_C_chg)
                # Writer conviction: 85%+ = institutions locked in with margin
                if _oi_accel_writer >= 0.90:
                    _oi_accel_score += 30
                elif _oi_accel_writer >= 0.80:
                    _oi_accel_score += 20
                elif _oi_accel_writer >= 0.70:
                    _oi_accel_score += 10
                # Volume surge: real-time institutional footprint
                if _oi_accel_vol >= 8.0:
                    _oi_accel_score += 35
                elif _oi_accel_vol >= 5.0:
                    _oi_accel_score += 25
                elif _oi_accel_vol >= 3.0:
                    _oi_accel_score += 15
                # FOIDH: OI at day-high = institutions actively adding NOW
                if _oi_accel_foidh >= 95:
                    _oi_accel_score += 25
                elif _oi_accel_foidh >= 85:
                    _oi_accel_score += 15
                elif _oi_accel_foidh >= 75:
                    _oi_accel_score += 8
                # Price freshness: early in spike = best entry, extended = stale
                # 0.15-0.80% = fresh (the spike is just starting)
                # 0.80-1.5% = ok but getting late  
                # >1.5% = already extended, no bonus
                _oi_accel_fresh = 0.15 <= _oi_accel_price <= 0.80
                if _oi_accel_fresh:
                    _oi_accel_score += 10
                _oi_accel_early_fire = False
                # EARLY FIRE: accel_score >= 60 + at least 1 anchor + at least 2 total factors
                # + price not already extended (< 1.5%)
                if (_oi_accel_score >= 60 and _oi_anchor_count >= 1 
                        and _oi_confirms >= 2 and _oi_accel_price < 1.5):
                    _oi_accel_early_fire = True
                    t._wlog(f"  ⚡ OI_WATCHER ACCEL EARLY-FIRE: {_oi_sym.replace('NSE:', '')} "
                               f"accel={_oi_accel_score} (W={_oi_accel_writer:.0%} "
                               f"V={_oi_accel_vol:.1f}x FOIDH={_oi_accel_foidh:.0f}% "
                               f"Px={_oi_accel_price:.2f}%{'⚡' if _oi_accel_fresh else ''}) — "
                               f"reduced min_confirms {_oi_smart_min}→{_oi_confirms} factors")

"""
        code = code.replace(anchor3, accel_block + anchor3, 1)
        patches_applied += 1
        print(f"  OK: Patch 3 — inserted ACCELERATION EARLY-FIRE block in Path 1")
    else:
        print(f"  FAIL: Patch 3 — conviction gate anchor not found")
        return False

    # ================================================================
    # PATCH 4: Make conviction gate respect accel early-fire
    # The conviction gate checks _oi_confirms < _oi_smart_min
    # We need to skip the LOW CONVICTION block when accel_early_fire is True
    # Anchor: "⛔ OI_WATCHER LOW CONVICTION:"
    # ================================================================
    anchor4 = '                    t._wlog(f"  ⛔ OI_WATCHER LOW CONVICTION:'
    if anchor4 in code:
        # Find the full block: the log + continue statement
        # We need to wrap the continue with: if not _oi_accel_early_fire
        # Find pattern: "if _oi_pre_anchors >= 2:\n                    _oi_smart_min = 3"
        # Then the else block has the ⛔ log + continue
        # Actually, let's find the exact block:
        # Line ~683: t._wlog(f"  ⛔ OI_WATCHER LOW CONVICTION: ...
        # Line ~686: continue
        # We need to gate this with: if not _oi_accel_early_fire:
        
        # Find: "⛔ OI_WATCHER LOW CONVICTION:" ... continue
        # The continue after LOW CONVICTION
        low_conv_pattern = re.search(
            r'(                    t\._wlog\(f"  ⛔ OI_WATCHER LOW CONVICTION:.*?\n.*?continue)',
            code, re.DOTALL
        )
        if low_conv_pattern:
            old_block = low_conv_pattern.group(1)
            new_block = old_block.replace(
                '                    t._wlog(f"  ⛔ OI_WATCHER LOW CONVICTION:',
                '                    if not _oi_accel_early_fire:\n                        t._wlog(f"  ⛔ OI_WATCHER LOW CONVICTION:'
            ).replace(
                '                    continue',
                '                        continue'
            )
            code = code.replace(old_block, new_block, 1)
            patches_applied += 1
            print(f"  OK: Patch 4a — LOW CONVICTION gate respects accel_early_fire")
        else:
            print(f"  WARN: Patch 4a — LOW CONVICTION continue pattern not found")
    
    # Also gate ANCHOR GATE with accel_early_fire
    anchor4b = '                    t._wlog(f"  ⛔ OI_WATCHER ANCHOR GATE:'
    if anchor4b in code:
        anchor_gate_pattern = re.search(
            r'(                    t\._wlog\(f"  ⛔ OI_WATCHER ANCHOR GATE:.*?\n.*?continue)',
            code, re.DOTALL
        )
        if anchor_gate_pattern:
            old_ablock = anchor_gate_pattern.group(1)
            new_ablock = old_ablock.replace(
                '                    t._wlog(f"  ⛔ OI_WATCHER ANCHOR GATE:',
                '                    if not _oi_accel_early_fire:\n                        t._wlog(f"  ⛔ OI_WATCHER ANCHOR GATE:'
            ).replace(
                '                    continue',
                '                        continue'
            )
            code = code.replace(old_ablock, new_ablock, 1)
            patches_applied += 1
            print(f"  OK: Patch 4b — ANCHOR GATE respects accel_early_fire")
    
    # Gate BREADTH GATE with accel_early_fire
    anchor4c = '                      t._wlog(f"  ⛔ OI_WATCHER BREADTH GATE:'
    if anchor4c in code:
        breadth_gate_pattern = re.search(
            r'(                      t\._wlog\(f"  ⛔ OI_WATCHER BREADTH GATE:.*?\n.*?continue)',
            code, re.DOTALL
        )
        if breadth_gate_pattern:
            old_bblock = breadth_gate_pattern.group(1)
            new_bblock = old_bblock.replace(
                '                      t._wlog(f"  ⛔ OI_WATCHER BREADTH GATE:',
                '                      if not _oi_accel_early_fire:\n                          t._wlog(f"  ⛔ OI_WATCHER BREADTH GATE:'
            ).replace(
                '                      continue',
                '                          continue'
            )
            code = code.replace(old_bblock, new_bblock, 1)
            patches_applied += 1
            print(f"  OK: Patch 4c — BREADTH GATE respects accel_early_fire")

    # ================================================================
    # PATCH 5: AGGR path acceleration fast-track
    # When institutional intensity is extreme, slash confirmation hold time
    # Insert after the existing adaptive confirmation adjustments
    # Anchor: the min_delta floor line
    #   _ag_min_delta = max(0.10, min(0.30, _ag_min_delta))
    # ================================================================
    anchor5 = "_ag_min_delta = max(0.10, min(0.30, _ag_min_delta))"
    if anchor5 in code:
        fast_track_block = """
            # [Apr 20] ACCELERATION FAST-TRACK: extreme institutional intensity = enter faster
            # When writer conviction is high AND (volume surging OR FOIDH extreme),
            # the spike is institutional-driven — don't wait 45s for confirmation.
            _ag_accel_writer = _res.get('_ag_accel_writer_ratio', 0) or 0
            _ag_accel_vol = _res.get('_ag_accel_vol_surge', 0) or 0
            _ag_accel_foidh = _res.get('_ag_accel_foidh_pct', 0) or 0
            _ag_accel_intense = (
                _ag_accel_writer >= 0.75 and 
                (_ag_accel_vol >= 3.0 or _ag_accel_foidh >= 85)
            )
            if _ag_accel_intense:
                _ag_confirm_secs = max(10.0, _ag_confirm_secs * 0.35)  # 45s → ~16s
                _ag_min_delta = max(0.05, _ag_min_delta * 0.40)        # 0.15% → ~0.06%
                # Re-apply floor after acceleration adjustment
"""
        code = code.replace(anchor5, anchor5 + fast_track_block, 1)
        patches_applied += 1
        print(f"  OK: Patch 5 — AGGR acceleration fast-track inserted")
    else:
        print(f"  FAIL: Patch 5 — AGGR floor anchor not found")

    # Also update the PENDING log to show acceleration status
    anchor5b = "f\"waiting {_ag_confirm_secs:.0f}s (breadth={_ag_mf_total}/6), \""
    if anchor5b in code:
        code = code.replace(
            anchor5b,
            "f\"waiting {_ag_confirm_secs:.0f}s (breadth={_ag_mf_total}/6){' ⚡ACCEL' if _ag_accel_intense else ''}, \"",
            1
        )
        patches_applied += 1
        print(f"  OK: Patch 5b — AGGR pending log shows acceleration status")

    # ================================================================
    # VALIDATION
    # ================================================================
    if patches_applied < 5:
        print(f"\n  ⚠️ Only {patches_applied} patches applied (expected 8+). Aborting.")
        return False
    
    # Write patched file
    with open(TARGET, 'w') as f:
        f.write(code)
    new_len = len(code)
    print(f"\n  >>> oi_watcher_engine.py written ({new_len} bytes, +{new_len - original_len} bytes)")
    print(f"  >>> {patches_applied} patches applied successfully")
    print(f"\npy_compile oi_watcher_engine.py, then restart.")
    return True


if __name__ == '__main__':
    ok = patch()
    sys.exit(0 if ok else 1)
