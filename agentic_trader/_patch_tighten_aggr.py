#!/usr/bin/env python3
"""Tighten ALL technical gates in the OI-AGGR pipeline (inside breadth gate).

Targets: oi_watcher_engine.py + autonomous_trader.py (config thresholds)

Factor thresholds (oi_watcher_engine.py):
  C  Price confirmation:    0.15% -> 0.25%
  F  Futures OI buildup:    moderate 0.30->0.40, strong 0.75->0.85
  G  Futures basis:         0.10% -> 0.15%  (both buy & sell)
  H  OI concentration ATM:  2.5% -> 2.0%
  I  PCR shift rate:        0.015 -> 0.020
  K1 Futures day-high:      0.75->0.80 (strong), 0.60->0.65 (moderate)
  K2 Orderbook imbalance:   0.55/0.45 -> 0.58/0.42
  M  Volume surge:          2.2x->2.5x (strong), 1.60x->1.80x (moderate), elevated 1.4x->1.5x
  SC/LU eff floor:          0.65 -> 0.70
  LB/SB eff floor:          0.55 -> 0.60

Config thresholds (autonomous_trader.py):
  _oi_confirm_seconds:          35 -> 45
  _oi_confirm_min_price_delta:  0.15 -> 0.20
  _oi_aggr_strong_str:          0.45 -> 0.55
  _oi_aggr_accel_min_str:       0.25 -> 0.35

Post-confirm floor clamps (oi_watcher_engine.py):
  max(0.08, ...) -> max(0.10, ...)
  max(20.0, ...) -> max(25.0, ...)
"""
import sys, os

def patch_binary(fpath, replacements):
    """Apply byte-level replacements to a file."""
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

# ── PATCH 1: oi_watcher_engine.py ──
oi_path = os.path.join(BASE, 'oi_watcher_engine.py')
print(f"=== Patching {oi_path} ===")

oi_replacements = [
    # (C) Price confirmation: 0.15 -> 0.25
    (b"_oi_aggr_min_price_delta', 0.15)", b"_oi_aggr_min_price_delta', 0.25)", "C: price confirm 0.15->0.25%"),

    # (F) Futures OI moderate: 0.30 -> 0.40
    (b"abs(_ag_fut_buildup) >= 0.30  # Apr 9: tightened", b"abs(_ag_fut_buildup) >= 0.40  # Apr 20: tightened", "F: futures moderate 0.30->0.40"),
    # (F) Futures OI strong: 0.75 -> 0.85
    (b"abs(_ag_fut_buildup) >= 0.75", b"abs(_ag_fut_buildup) >= 0.85", "F: futures strong 0.75->0.85"),

    # (G) Futures basis: 0.10 -> 0.15 (buy side)
    (b"_ag_basis_pct > 0.10) or   # Apr 7: tightened 0.05", b"_ag_basis_pct > 0.15) or   # Apr 20: tightened 0.10", "G: basis buy 0.10->0.15"),
    # (G) Futures basis: 0.10 -> 0.15 (sell side)
    (b"_ag_basis_pct < -0.10)      # Apr 7: tightened", b"_ag_basis_pct < -0.15)      # Apr 20: tightened", "G: basis sell 0.10->0.15"),
    # (G) Futures basis disagree thresholds
    (b"_ag_basis_pct < -0.10) or", b"_ag_basis_pct < -0.15) or", "G: basis disagree buy"),
    (b"_ag_basis_pct > 0.10)", b"_ag_basis_pct > 0.15)", "G: basis disagree sell"),

    # (H) OI concentration ATM: 2.5 -> 2.0
    (b"_ag_stk_dist <= 2.5:  # Apr 7: tightened 3.0", b"_ag_stk_dist <= 2.0:  # Apr 20: tightened 2.5", "H: ATM distance 2.5->2.0%"),

    # (I) PCR shift rate: 0.015 -> 0.020
    (b"_ag_pcr_rate >= 0.015) or   # tightened 0.01", b"_ag_pcr_rate >= 0.020) or   # Apr 20: tightened 0.015", "I: PCR shift buy 0.015->0.020"),
    (b"_ag_pcr_rate <= -0.015)", b"_ag_pcr_rate <= -0.020)", "I: PCR shift sell confirm"),
    (b"_ag_pcr_rate <= -0.015) or", b"_ag_pcr_rate <= -0.020) or", "I: PCR shift oppose buy"),
    (b"_ag_pcr_rate >= 0.015)", b"_ag_pcr_rate >= 0.020)", "I: PCR shift oppose sell"),

    # (K1) Futures day-high: 0.75 -> 0.80
    (b"_ag_fk_pos >= 0.75:  # Apr 7: tightened 0.70", b"_ag_fk_pos >= 0.80:  # Apr 20: tightened 0.75", "K1: day-high 0.75->0.80"),
    # (K1) Futures moderate: 0.60 -> 0.65
    (b"_ag_fk_pos >= 0.60:  # Apr 7: tightened 0.55", b"_ag_fk_pos >= 0.65:  # Apr 20: tightened 0.60", "K1: day-high moderate 0.60->0.65"),
    # (K2) Orderbook imbalance: 0.55/0.45 -> 0.58/0.42
    (b"_ag_fk_imb > 0.55) or", b"_ag_fk_imb > 0.58) or", "K2: orderbook buy 0.55->0.58"),
    (b"_ag_fk_imb < 0.45)", b"_ag_fk_imb < 0.42)", "K2: orderbook sell 0.45->0.42"),

    # (M) Volume surge strong: 2.2x -> 2.5x
    (b"_ag_vol_surge_ratio >= 2.2 and _ag_v_elevated >= 2:  # Apr 9: tightened 2.0",
     b"_ag_vol_surge_ratio >= 2.5 and _ag_v_elevated >= 2:  # Apr 20: tightened 2.2",
     "M: volume strong 2.2->2.5x"),
    # (M) Volume surge moderate: 1.60x -> 1.80x
    (b"_ag_vol_surge_ratio >= 1.60 and _ag_v_elevated >= 1:  # Apr 9: tightened 1.46",
     b"_ag_vol_surge_ratio >= 1.80 and _ag_v_elevated >= 1:  # Apr 20: tightened 1.60",
     "M: volume moderate 1.60->1.80x"),
    # (M) Volume elevated threshold: 1.4x -> 1.5x
    (b"_vd >= _ag_vavg * 1.4)", b"_vd >= _ag_vavg * 1.5)", "M: volume elevated 1.4->1.5x"),

    # SC/LU effective floor: 0.65 -> 0.70
    (b"('SHORT_COVERING', 'LONG_UNWINDING') and _eff_str < 0.65:",
     b"('SHORT_COVERING', 'LONG_UNWINDING') and _eff_str < 0.70:",
     "SC/LU floor 0.65->0.70"),
    # LB/SB effective floor: 0.55 -> 0.60
    (b"('LONG_BUILDUP', 'SHORT_BUILDUP') and _eff_str < 0.55:",
     b"('LONG_BUILDUP', 'SHORT_BUILDUP') and _eff_str < 0.60:",
     "LB/SB floor 0.55->0.60"),

    # Post-confirm floor clamps: 0.08 -> 0.10
    (b"_ag_min_delta = max(0.08, min(0.30,", b"_ag_min_delta = max(0.10, min(0.30,", "confirm delta floor 0.08->0.10"),
    # Post-confirm time floor: 20.0 -> 25.0
    (b"_ag_confirm_secs = max(20.0, min(90.0,", b"_ag_confirm_secs = max(25.0, min(90.0,", "confirm time floor 20->25s"),
]

oi_data, oi_count = patch_binary(oi_path, oi_replacements)
if oi_count > 0:
    open(oi_path, 'wb').write(oi_data)
    print(f"\n  >>> oi_watcher_engine.py: {oi_count}/{len(oi_replacements)} patches applied\n")
else:
    print("\n  >>> oi_watcher_engine.py: NO patches applied!\n")

# ── PATCH 2: autonomous_trader.py (config thresholds) ──
at_path = os.path.join(BASE, 'autonomous_trader.py')
print(f"=== Patching {at_path} ===")

at_replacements = [
    # Confirm seconds: 35 -> 45
    (b"self._oi_confirm_seconds = 35                      # Apr 15: relaxed 50",
     b"self._oi_confirm_seconds = 45                      # Apr 20: tightened 35",
     "confirm_seconds 35->45"),
    # Confirm delta: 0.15 -> 0.20
    (b"self._oi_confirm_min_price_delta = 0.15            # Apr 15: relaxed 0.30",
     b"self._oi_confirm_min_price_delta = 0.20            # Apr 20: tightened 0.15",
     "confirm_delta 0.15->0.20"),
    # Strong strength: 0.45 -> 0.55
    (b"self._oi_aggr_strong_str = 0.45      # Strength",
     b"self._oi_aggr_strong_str = 0.55      # Apr 20: tightened 0.45, Strength",
     "strong_str 0.45->0.55"),
    # Accel min strength: 0.25 -> 0.35
    (b"self._oi_aggr_accel_min_str = 0.25   # Lower strength floor when acceleration",
     b"self._oi_aggr_accel_min_str = 0.35   # Apr 20: tightened 0.25, Lower strength floor when acceleration",
     "accel_min 0.25->0.35"),
]

at_data, at_count = patch_binary(at_path, at_replacements)
if at_count > 0:
    open(at_path, 'wb').write(at_data)
    print(f"\n  >>> autonomous_trader.py: {at_count}/{len(at_replacements)} patches applied\n")
else:
    print("\n  >>> autonomous_trader.py: NO patches applied!\n")

print(f"TOTAL: {oi_count + at_count} patches applied")
print("Next: py_compile both files, then restart titan-bot")
