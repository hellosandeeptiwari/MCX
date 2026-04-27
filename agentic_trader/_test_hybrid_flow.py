#!/usr/bin/env python3
"""End-to-end test: verify predictor→DR detector feature flow works with hybrid model."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.feature_engineering import get_feature_names
from ml_models.down_risk_detector import DownRiskDetector

print("=== END-TO-END HYBRID FEATURE FLOW TEST ===\n")

# 1. Load hybrid detector
det = DownRiskDetector()
if not det.load():
    print("FAIL: Could not load detector")
    sys.exit(1)

assert det.up_detector.vae is not None, "UP VAE not loaded"
assert det.down_detector.vae is not None, "DOWN VAE not loaded"
up_dim = det.up_detector.vae.encoder[0].in_features
down_dim = det.down_detector.vae.encoder[0].in_features
print(f"Detector loaded: UP={up_dim}, DOWN={down_dim}")

# 2. Simulate what predictor.py now does: build X with get_feature_names() width
dr_feature_names = get_feature_names()
print(f"get_feature_names() → {len(dr_feature_names)} features")

# Create a fake feature array (98 features, non-zero to detect zero-fill bugs)
np.random.seed(42)
X_full = np.random.randn(1, len(dr_feature_names)).astype(np.float32)
print(f"X_full shape: {X_full.shape}")

# 3. Test UP prediction (should remap 98 → 87)
up_result = det.predict_single(X_full, 'UP')
print(f"\nUP prediction: score={up_result['anomaly_score'][0]:.4f}, "
      f"flag={up_result['down_risk_flag'][0]}, bucket={up_result['confidence_bucket'][0]}")

# 4. Test DOWN prediction (should stay at 98)
down_result = det.predict_single(X_full, 'DOWN')
print(f"\nDOWN prediction: score={down_result['anomaly_score'][0]:.4f}, "
      f"flag={down_result['down_risk_flag'][0]}, bucket={down_result['confidence_bucket'][0]}")

# 5. Verify both paths produce valid scores
assert 0 <= up_result['anomaly_score'][0] <= 1, "UP score out of range"
assert 0 <= down_result['anomaly_score'][0] <= 1, "DOWN score out of range"
print(f"\nBoth scores in [0,1] range ✓")

# 6. Simulate OLD predictor bug: X with only 61 features
print("\n=== OLD BUG SIMULATION (61-wide X) ===")
X_narrow = np.random.randn(1, 61).astype(np.float32)
up_narrow = det.predict_single(X_narrow, 'UP')
down_narrow = det.predict_single(X_narrow, 'DOWN')
print(f"With 61-wide X:")
print(f"  UP score: {up_narrow['anomaly_score'][0]:.4f}")
print(f"  DOWN score: {down_narrow['anomaly_score'][0]:.4f}")

print("\n=== ALL TESTS PASSED ===")
