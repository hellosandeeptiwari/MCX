#!/usr/bin/env python3
"""Quick test: verify hybrid model loads correctly on EC2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.down_risk_detector import DownRiskDetector

d = DownRiskDetector()
d.load()

up_dim = d.up_detector.vae.encoder[0].in_features
down_dim = d.down_detector.vae.encoder[0].in_features

print(f"UP input_dim:   {up_dim}")
print(f"DOWN input_dim: {down_dim}")

# Check per-detector feature names
up_fn = getattr(d.up_detector, 'feature_names', None)
down_fn = getattr(d.down_detector, 'feature_names', None)
print(f"UP feature_names:   {len(up_fn) if up_fn else 'MISSING'}")
print(f"DOWN feature_names: {len(down_fn) if down_fn else 'MISSING'}")

# Verify consistency
ok = True
if up_dim != 87:
    print(f"FAIL: UP dim should be 87, got {up_dim}")
    ok = False
if down_dim != 98:
    print(f"FAIL: DOWN dim should be 98, got {down_dim}")
    ok = False
if up_fn and len(up_fn) != 87:
    print(f"FAIL: UP feature_names should be 87, got {len(up_fn)}")
    ok = False
if down_fn and len(down_fn) != 98:
    print(f"FAIL: DOWN feature_names should be 98, got {len(down_fn)}")
    ok = False

if ok:
    print("\nHYBRID VERIFICATION SUCCESS!")
else:
    print("\nHYBRID VERIFICATION FAILED!")
    sys.exit(1)
