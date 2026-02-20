"""Audit: Internal XGB regime classifier quality & leakage check."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
from ml_models.down_risk_detector import load_detector_dataset, DetectorConfig
from sklearn.metrics import accuracy_score, classification_report

config = DetectorConfig()
print("Loading dataset (this takes ~2 min for 207 stocks)...")
combined, feature_names = load_detector_dataset(config)

true_regime = combined['label_idx'].map({0: 'DOWN', 1: 'FLAT', 2: 'UP'}).values
pred_regime = combined['xgb_regime'].values

acc = accuracy_score(true_regime, pred_regime)
print(f'\n=== INTERNAL XGB REGIME CLASSIFIER METRICS ===')
print(f'Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)')
print(classification_report(true_regime, pred_regime, digits=3))

# Confusion matrix
print("=== Confusion (True -> Predicted) ===")
for true_r in ['UP', 'DOWN', 'FLAT']:
    mask_true = true_regime == true_r
    total = mask_true.sum()
    parts = []
    for pred_r in ['UP', 'DOWN', 'FLAT']:
        n = (pred_regime[mask_true] == pred_r).sum()
        pct = n / total * 100 if total > 0 else 0
        parts.append(f'{pred_r}:{n}({pct:.1f}%)')
    sep = ' | '
    print(f'  True {true_r} ({total}) -> {sep.join(parts)}')

# Leakage check: what's really in the UP bucket?
up_mask = pred_regime == 'UP'
n_up = up_mask.sum()
true_in_up = true_regime[up_mask]
print(f'\n=== LEAKAGE: What is REALLY in the UP bucket? ===')
print(f'UP bucket: {n_up} samples')
for r in ['UP', 'DOWN', 'FLAT']:
    n = (true_in_up == r).sum()
    pct = n / n_up * 100
    print(f'  Actually {r}: {n} ({pct:.1f}%)')

# y_down8 crash rate by predicted regime
print(f'\n=== Crash rate (y_down8) by predicted regime ===')
for r in ['UP', 'DOWN', 'FLAT']:
    mask = pred_regime == r
    if mask.sum() > 0:
        rate = combined.loc[mask, 'y_down8'].mean()
        print(f'  {r}: {rate:.4f} ({rate*100:.2f}%) - {mask.sum()} samples')

print("\nDone.")
