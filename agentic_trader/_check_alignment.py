#!/usr/bin/env python3
"""Check predictor metadata feature alignment with down-risk detector."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Check MovePredictor metadata feature names
meta_paths = [
    "ml_models/saved_models/move_predictor_meta.json",
    "ml_models/saved_models/move_predictor_latest_meta.json",
]
meta_features = []
for meta_path in meta_paths:
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        mf = meta.get('feature_names', [])
        print(f"MovePredictor metadata ({meta_path}): {len(mf)} features")
        if mf:
            print(f"  Last 15: {mf[-15:]}")
        if not meta_features:
            meta_features = mf
if not meta_features:
    print("MovePredictor metadata NOT FOUND at any path")
    # Also check what predictor.py actually loads
    try:
        from ml_models.predictor import MovePredictor
        p = MovePredictor.__new__(MovePredictor)
        print("  (would fall back to get_feature_names())")
    except:
        pass

# 2. Check get_feature_names()
from ml_models.feature_engineering import get_feature_names
live_features = get_feature_names()
print(f"\nget_feature_names() returns: {len(live_features)}")
print(f"  Last 15: {live_features[-15:]}")

# 3. Check down_risk feature files
for label, fname in [("superset", "down_risk_feature_names.json"),
                     ("UP", "down_risk_up_feature_names.json"),
                     ("DOWN", "down_risk_down_feature_names.json")]:
    fpath = f"ml_models/saved_models/{fname}"
    if os.path.exists(fpath):
        with open(fpath) as f:
            fns = json.load(f)
        print(f"\nDR {label} feature_names: {len(fns)}")
    else:
        print(f"\nDR {label} feature_names: FILE MISSING")

# 4. Alignment check
print("\n=== ALIGNMENT ANALYSIS ===")
predictor_width = len(meta_features) if meta_features else len(live_features)
print(f"Predictor X width (what gets passed to DR): {predictor_width}")
print(f"DR superset width: {len(live_features)}")

# Check if predictor features match get_feature_names
if meta_features:
    missing_in_meta = [f for f in live_features if f not in meta_features]
    extra_in_meta = [f for f in meta_features if f not in live_features]
    if missing_in_meta:
        print(f"\nFEATURES in get_feature_names() but NOT in predictor metadata ({len(missing_in_meta)}):")
        for f in missing_in_meta:
            print(f"  - {f}")
    if extra_in_meta:
        print(f"\nFEATURES in predictor metadata but NOT in get_feature_names() ({len(extra_in_meta)}):")
        for f in extra_in_meta:
            print(f"  - {f}")
    if not missing_in_meta and not extra_in_meta:
        print("\nPredictor metadata MATCHES get_feature_names() perfectly")
else:
    print("\nNo metadata features — predictor falls back to get_feature_names()")

# 5. Simulate the flow
print("\n=== SIMULATION ===")
print(f"predictor.py builds X with {predictor_width} columns")

# Load DR superset
dr_super_path = "ml_models/saved_models/down_risk_feature_names.json"
if os.path.exists(dr_super_path):
    with open(dr_super_path) as f:
        dr_superset = json.load(f)
else:
    dr_superset = live_features

# UP detector feature mapping
up_path = "ml_models/saved_models/down_risk_up_feature_names.json"
if os.path.exists(up_path):
    with open(up_path) as f:
        up_features = json.load(f)
    # Simulate _select_features for UP
    issues = []
    for fname in up_features:
        if fname in dr_superset:
            idx = dr_superset.index(fname)
            if idx >= predictor_width:
                issues.append(f"  UP needs '{fname}' at superset idx {idx}, but X only has {predictor_width} cols")
        else:
            issues.append(f"  UP feature '{fname}' NOT in superset")
    if issues:
        print(f"\nUP DETECTOR ISSUES ({len(issues)}):")
        for i in issues:
            print(i)
    else:
        print(f"UP detector ({len(up_features)} features): OK - all map within X width")

# DOWN detector feature mapping
down_path = "ml_models/saved_models/down_risk_down_feature_names.json"
if os.path.exists(down_path):
    with open(down_path) as f:
        down_features = json.load(f)
    # Simulate _select_features for DOWN
    issues = []
    for fname in down_features:
        if fname in dr_superset:
            idx = dr_superset.index(fname)
            if idx >= predictor_width:
                issues.append(f"  DOWN needs '{fname}' at superset idx {idx}, but X only has {predictor_width} cols → ZERO-FILLED!")
        else:
            issues.append(f"  DOWN feature '{fname}' NOT in superset")
    if issues:
        print(f"\nDOWN DETECTOR ISSUES ({len(issues)}):")
        for i in issues:
            print(i)
    else:
        print(f"DOWN detector ({len(down_features)} features): OK - all map within X width")

print("\n=== DONE ===")
