"""Quick sanity check before paper trading."""
import json, sys

# 1. Check model files match v5
meta_up = json.load(open('ml_models/saved_models/down_risk_up_meta.json'))
meta_down = json.load(open('ml_models/saved_models/down_risk_down_meta.json'))
feat = json.load(open('ml_models/saved_models/down_risk_feature_names.json'))
print(f"UP meta timestamp:   {meta_up['timestamp']}")
print(f"DOWN meta timestamp: {meta_down['timestamp']}")
print(f"Model input_dim:     {meta_up['config']['input_dim']}")
print(f"Feature names count: {len(feat)}")

# 2. Check feature_engineering returns matching count
from ml_models.feature_engineering import get_feature_names
live_feats = get_feature_names()
print(f"Live feature count:  {len(live_feats)}")

# 3. Check overlap
model_set = set(feat)
live_set = set(live_feats)
missing = model_set - live_set
extra = live_set - model_set
print(f"Missing from live:   {missing if missing else 'NONE'}")
print(f"Extra in live:       {extra if extra else 'NONE'}")

# 4. Quick import test
from ml_models.down_risk_detector import DownRiskDetector
d = DownRiskDetector()
loaded = d.load()
print(f"Model loads OK:      {loaded}")

# 5. Verdict
ok = True
if meta_up['timestamp'] != '2026-02-20 22:03:49':
    print("FAIL: UP model not v5"); ok = False
if meta_down['timestamp'] != '2026-02-20 22:03:49':
    print("FAIL: DOWN model not v5"); ok = False
if missing:
    print(f"FAIL: Model needs features not in live pipeline: {missing}"); ok = False
if not loaded:
    print("FAIL: Model failed to load"); ok = False

if ok:
    print("\n=== ALL CHECKS PASSED — READY FOR PAPER TRADING ===")
else:
    print("\n=== ISSUES FOUND — DO NOT TRADE ===")
    sys.exit(1)
