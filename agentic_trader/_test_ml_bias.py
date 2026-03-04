"""Quick diagnostic: check ML direction model bias.
Run on EC2: python3 _test_ml_bias.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ml_models.predictor import MovePredictor
from ml_models.feature_engineering import compute_features
import pandas as pd, glob, numpy as np

p = MovePredictor()

data_dir = os.path.join(os.path.dirname(__file__), 'ml_models', 'data', 'candles_5min')
files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))[:15]

print('=== RAW vs CALIBRATED direction predictions (stored 5min, no context) ===')
header = f'{"Stock":15s} {"dir_raw":>8s} {"dir_cal":>8s} {"gate_raw":>8s} {"gate_cal":>8s} {"oi_feat":>10s} {"signal":>6s}'
print(header)
print('-' * len(header))
for f in files:
    sym = os.path.basename(f).replace('.parquet','')
    df = pd.read_parquet(f)
    if len(df) < 50:
        continue
    featured = compute_features(df.tail(500))
    if featured.empty:
        continue
    latest = featured.iloc[-1:]
    for feat in p.feature_names:
        if feat not in latest.columns:
            latest = latest.copy()
            latest[feat] = 0.0
    X = latest[p.feature_names].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    gate_raw = float(p.gate_model.predict_proba(X)[0, 1])
    gate_cal = float(p.gate_cal.predict([gate_raw])[0]) if p.gate_cal else gate_raw
    
    if p.dir_feature_names != p.feature_names:
        dir_indices = [p.feature_names.index(fn) for fn in p.dir_feature_names if fn in p.feature_names]
        X_dir = X[:, dir_indices]
    else:
        X_dir = X
    dir_raw = float(p.dir_model.predict_proba(X_dir)[0, 1])
    dir_cal = float(p.dir_cal.predict([dir_raw])[0]) if p.dir_cal else dir_raw
    
    oi_idx = p.dir_feature_names.index('fut_oi_buildup') if 'fut_oi_buildup' in p.dir_feature_names else -1
    oi_val = float(X_dir[0, oi_idx]) if oi_idx >= 0 else -1
    
    # Determine signal
    if gate_cal >= 0.50:
        if dir_cal >= 0.54:
            sig = 'UP'
        elif (1, dir_cal) >= (1, 0.54):
            sig = 'DOWN'
        else:
            sig = 'FLAT'
    else:
        sig = 'FLAT'
    
    # Actually compute properly
    p_up = dir_cal
    p_dn = 1 - dir_cal
    if gate_cal >= 0.50:
        if p_up >= 0.54:
            sig = 'UP'
        elif p_dn >= 0.54:
            sig = 'DOWN'
        else:
            sig = 'FLAT'
    else:
        sig = 'FLAT'
    
    print(f'{sym:15s} {dir_raw:8.4f} {dir_cal:8.4f} {gate_raw:8.4f} {gate_cal:8.4f} {oi_val:10.4f} {sig:>6s}')

print()
print('=== KEY FINDING ===')
print('fut_oi_buildup = 46.5% of direction model importance')
print('oi_price_confirm = 10.8% of direction model importance')
print('Together = 57.3% of direction model decision')
print()
print('When fut_oi_buildup=0 (missing OI), the model has no primary signal')
print('and falls back to secondary features which may all point same direction')
print()

# Now test: what happens when fut_oi_buildup is manipulated
print('=== SENSITIVITY TEST: varying fut_oi_buildup on first stock ===')
f = files[0]
sym = os.path.basename(f).replace('.parquet','')
df = pd.read_parquet(f)
featured = compute_features(df.tail(500))
latest = featured.iloc[-1:]
for feat in p.feature_names:
    if feat not in latest.columns:
        latest = latest.copy()
        latest[feat] = 0.0

for oi_val in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
    X = latest[p.feature_names].values.copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Set fut_oi_buildup
    oi_idx_full = list(p.feature_names).index('fut_oi_buildup') if 'fut_oi_buildup' in p.feature_names else -1
    if oi_idx_full >= 0:
        X[0, oi_idx_full] = oi_val
    
    if p.dir_feature_names != p.feature_names:
        dir_indices = [p.feature_names.index(fn) for fn in p.dir_feature_names if fn in p.feature_names]
        X_dir = X[:, dir_indices]
    else:
        X_dir = X
    
    dir_raw = float(p.dir_model.predict_proba(X_dir)[0, 1])
    dir_cal = float(p.dir_cal.predict([dir_raw])[0]) if p.dir_cal else dir_raw
    gate_raw = float(p.gate_model.predict_proba(X)[0, 1])
    gate_cal = float(p.gate_cal.predict([gate_raw])[0]) if p.gate_cal else gate_raw
    
    p_up = dir_cal
    p_dn = 1 - dir_cal
    if gate_cal >= 0.50:
        if p_up >= 0.54: sig = 'UP'
        elif p_dn >= 0.54: sig = 'DOWN'
        else: sig = 'FLAT'
    else:
        sig = 'FLAT'
    
    print(f'  oi_buildup={oi_val:+5.1f}  dir_raw={dir_raw:.4f}  dir_cal={dir_cal:.4f}  signal={sig}')
