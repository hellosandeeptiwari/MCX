"""Debug: What regime adjustment is the model actually computing?"""
import sys, os
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
os.chdir('/home/ubuntu/titan/agentic_trader')

from dotenv import load_dotenv
load_dotenv('/home/ubuntu/titan/agentic_trader/.env')

import pandas as pd
import numpy as np

# Load the predictor
from ml_models.predictor import MovePredictor
pred = MovePredictor()

# Load NIFTY data (same as the bot has)
nifty_5min = pd.read_parquet('ml_models/data/candles_5min/NIFTY50.parquet')
nifty_daily = pd.read_parquet('ml_models/data/candles_daily/NIFTY50.parquet')
print(f"NIFTY 5min: {len(nifty_5min)} rows, last={nifty_5min['date'].max()}")
print(f"NIFTY daily: {len(nifty_daily)} rows, last={nifty_daily['date'].max()}")

# Pick a test stock
test_syms = ['RELIANCE', 'TATASTEEL', 'SBIN', 'HDFCBANK']
from ml_models.feature_engineering import compute_features

for sym in test_syms:
    path5 = f'ml_models/data/candles_5min/{sym}.parquet'
    if not os.path.exists(path5):
        print(f"\n{sym}: no 5min data")
        continue
    
    df = pd.read_parquet(path5)
    
    # Load OI if available
    oi_path = f'ml_models/data/futures_oi/{sym}_futures_oi.parquet'
    oi_df = pd.read_parquet(oi_path) if os.path.exists(oi_path) else None
    
    features = compute_features(df, symbol=sym, oi_df=oi_df, nifty_5min_df=nifty_5min, nifty_daily_df=nifty_daily)
    if features is None or len(features) == 0:
        print(f"\n{sym}: no features")
        continue
    
    last = features.iloc[-1:]
    
    # Print key NIFTY features  
    nifty_cols = [c for c in features.columns if 'nifty' in c.lower()]
    print(f"\n{'='*60}")
    print(f"{sym} — Last candle: {features['date'].iloc[-1]}")
    print(f"NIFTY features in the feature vector:")
    for c in nifty_cols:
        v = last[c].values[0]
        print(f"  {c:25s} = {v:+.4f}")
    
    # Run prediction
    result = pred.predict(features)
    if result:
        print(f"Signal: {result.get('ml_signal')} | P(UP|MOVE)={result.get('ml_p_up_given_move', 'N/A')} | Boost={result.get('ml_boost', 0)}")
        print(f"P(move)={result.get('ml_prob_flat', 'N/A')} (flat)")
        
        # Now manually call regime adjust to see details
        X = pred._prepare_X(last)
        if X is not None:
            # Get raw direction P(UP|MOVE) without adjustment
            if pred.dir_feature_names != pred.feature_names:
                dir_indices = [pred.feature_names.index(f) for f in pred.dir_feature_names if f in pred.feature_names]
                X_dir = X[:, dir_indices]
            else:
                X_dir = X
            dir_raw = float(pred.dir_model.predict_proba(X_dir)[0, 1])
            if pred.dir_cal is not None:
                p_up_raw = float(pred.dir_cal.predict([dir_raw])[0])
            else:
                p_up_raw = dir_raw
            
            # Get regime adjustment
            p_up_adj, shift = pred._market_regime_adjust(X, p_up_raw)
            print(f"RAW P(UP|MOVE) = {p_up_raw:.4f}")
            print(f"REGIME SHIFT   = {shift:+.4f}")
            print(f"ADJ P(UP|MOVE) = {p_up_adj:.4f}")
            
            # Also print what the regime score components would be
            def _get(name, default=0.0):
                try:
                    idx = pred.feature_names.index(name)
                    return float(X[0, idx])
                except (ValueError, IndexError):
                    return default
            
            nifty_roc = _get('nifty_roc_6')
            nifty_rsi = _get('nifty_rsi_14')
            nifty_trend = _get('nifty_daily_trend')
            nifty_d_rsi = _get('nifty_daily_rsi')
            nifty_slope = _get('nifty_ema9_slope')
            nifty_bb = _get('nifty_bb_position')
            print(f"Regime inputs: roc={nifty_roc:+.4f} rsi={nifty_rsi:.1f} trend={nifty_trend:+.4f} d_rsi={nifty_d_rsi:.1f} slope={nifty_slope:+.4f} bb={nifty_bb:.4f}")
    else:
        print(f"Prediction returned empty")
