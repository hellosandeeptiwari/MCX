"""Quick diagnostic: show actual calibrated probability distribution for ATR 1.5 model."""
import os, sys
from dotenv import load_dotenv
load_dotenv()

from ml_models.predictor import MovePredictor
mp = MovePredictor()
print(f"Model ready: {mp.ready}, type: {mp.model_type}")

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

kite = KiteConnect(api_key=os.getenv('ZERODHA_API_KEY'))
kite.set_access_token(os.getenv('ZERODHA_ACCESS_TOKEN'))

test_stocks = [
    'INFY', 'ITC', 'RELIANCE', 'HDFCBANK', 'TCS', 'WIPRO', 'SBIN',
    'ICICIBANK', 'ETERNAL', 'HINDALCO', 'BSE', 'KFINTECH', 'HAL',
    'BEL', 'GMRAIRPORT', 'PERSISTENT', 'COFORGE', 'TATAELXSI',
    'INDUSINDBK', 'LT', 'SWIGGY', 'VEDL', 'AMBER', 'MCX'
]

instruments = kite.instruments('NSE')
token_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}

import time
results = []
for sym in test_stocks:
    tok = token_map.get(sym)
    if not tok:
        continue
    time.sleep(0.4)
    data = kite.historical_data(tok, datetime.now() - timedelta(days=120), datetime.now(), 'day')
    if len(data) < 50:
        continue
    df = pd.DataFrame(data)
    
    # Also get raw (uncalibrated) probabilities
    from ml_models.feature_engineering import compute_features
    featured = compute_features(df)
    if featured.empty or len(featured) < 2:
        continue
    latest = featured.iloc[-1:]
    for feat in mp.feature_names:
        if feat not in latest.columns:
            latest = latest.copy()
            latest[feat] = 0.0
    X = latest[mp.feature_names].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Raw probabilities (before calibration)
    gate_raw = float(mp.gate_model.predict_proba(X)[0, 1])
    
    if mp.dir_feature_names != mp.feature_names:
        dir_indices = [mp.feature_names.index(f) for f in mp.dir_feature_names if f in mp.feature_names]
        X_dir = X[:, dir_indices]
    else:
        X_dir = X
    dir_raw = float(mp.dir_model.predict_proba(X_dir)[0, 1])
    
    # Calibrated
    pred = mp.predict(df)
    if pred:
        results.append({
            'sym': sym,
            'gate_raw': gate_raw,
            'dir_raw': dir_raw,
            'p_move': pred.get('ml_p_move', 0),
            'p_up': pred.get('ml_p_up_given_move', 0),
            'p_down': pred.get('ml_p_down_given_move', 0),
            'signal': pred.get('ml_signal'),
            'boost': pred.get('ml_score_boost', 0),
        })

# Print table
header = f"{'Sym':<15} {'gate_raw':>9} {'dir_raw':>8} {'p_move':>7} {'p_up|mv':>8} {'p_dn|mv':>8} {'signal':>6} {'boost':>5}"
print("\n" + header)
print("-" * len(header))
for r in sorted(results, key=lambda x: x['p_move'], reverse=True):
    print(f"{r['sym']:<15} {r['gate_raw']:>9.4f} {r['dir_raw']:>8.4f} {r['p_move']:>7.4f} {r['p_up']:>8.4f} {r['p_down']:>8.4f} {r['signal']:>6} {r['boost']:>5}")

# Stats
moves = [r['p_move'] for r in results]
ups = [r['p_up'] for r in results]
gate_raws = [r['gate_raw'] for r in results]
dir_raws = [r['dir_raw'] for r in results]

print(f"\n--- CALIBRATED ---")
print(f"p_move    min:{min(moves):.4f} max:{max(moves):.4f} mean:{np.mean(moves):.4f} median:{np.median(moves):.4f}")
print(f"p_up|mv   min:{min(ups):.4f} max:{max(ups):.4f} mean:{np.mean(ups):.4f} median:{np.median(ups):.4f}")

print(f"\n--- RAW (before calibration) ---")
print(f"gate_raw  min:{min(gate_raws):.4f} max:{max(gate_raws):.4f} mean:{np.mean(gate_raws):.4f} median:{np.median(gate_raws):.4f}")
print(f"dir_raw   min:{min(dir_raws):.4f} max:{max(dir_raws):.4f} mean:{np.mean(dir_raws):.4f} median:{np.median(dir_raws):.4f}")

sigs = [r['signal'] for r in results]
print(f"\nSignals: UP={sigs.count('UP')} DOWN={sigs.count('DOWN')} FLAT={sigs.count('FLAT')}")
print(f"Stocks >= 0.35 p_move: {sum(1 for m in moves if m >= 0.35)}/{len(moves)}")
print(f"Stocks >= 0.30 p_move: {sum(1 for m in moves if m >= 0.30)}/{len(moves)}")
print(f"p_up >= 0.50: {sum(1 for u in ups if u >= 0.50)}/{len(ups)}")
print(f"p_up >= 0.48: {sum(1 for u in ups if u >= 0.48)}/{len(ups)}")
