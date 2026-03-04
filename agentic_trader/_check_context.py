"""Check context data freshness and what raw direction probs look like with context"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ml_models.predictor import MovePredictor
from ml_models.feature_engineering import compute_features
import pandas as pd, glob, numpy as np

p = MovePredictor()

# 1. Check context data dates
print('=== CONTEXT DATA FRESHNESS ===')
for name in ['NIFTY50_daily', 'SECTOR_IT', 'SECTOR_BANKS']:
    path = f'ml_models/data/candles_daily/{name}.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f'  {name}: {len(df)} rows, last date: {df["date"].iloc[-1]}')

nifty_5m = None
nifty_daily = None
for p2 in ['ml_models/data/candles_5min/NIFTY50.parquet', 'ml_models/data/candles_5min/NIFTY_50.parquet']:
    if os.path.exists(p2):
        nifty_5m = pd.read_parquet(p2)
        print(f'  Nifty 5min: {len(nifty_5m)} rows, last: {nifty_5m["date"].iloc[-1]}')
        break

np2 = 'ml_models/data/candles_daily/NIFTY50_daily.parquet'
if os.path.exists(np2):
    nifty_daily = pd.read_parquet(np2)

# 2. Test with context data on 5 stocks
print('\n=== WITH CONTEXT DATA (nifty_5min + nifty_daily + futures_oi) ===')
data_dir = 'ml_models/data/candles_5min'
ups, downs, flats = 0, 0, 0
dir_raws = []
stocks_tested = 0
for fname in sorted(os.listdir(data_dir))[:40]:
    if not fname.endswith('.parquet') or 'NIFTY' in fname or 'SECTOR' in fname:
        continue
    sym = fname.replace('.parquet', '')
    df = pd.read_parquet(os.path.join(data_dir, fname))
    if len(df) < 50:
        continue
    
    # Load OI
    oi_path = f'ml_models/data/futures_oi/{sym}_futures_oi.parquet'
    oi_df = pd.read_parquet(oi_path) if os.path.exists(oi_path) else None
    
    pred = p.predict(df.tail(500), futures_oi_df=oi_df, 
                     nifty_5min_df=nifty_5m, nifty_daily_df=nifty_daily)
    if pred:
        sig = pred.get('ml_signal', '?')
        if sig == 'UP': ups += 1
        elif sig == 'DOWN': downs += 1
        else: flats += 1
        stocks_tested += 1
        
        # Get raw direction prob
        featured = compute_features(df.tail(500), futures_oi_df=oi_df,
                                    nifty_5min_df=nifty_5m, nifty_daily_df=nifty_daily)
        if not featured.empty:
            latest = featured.iloc[-1:]
            for feat in p.feature_names:
                if feat not in latest.columns:
                    latest = latest.copy()
                    latest[feat] = 0.0
            X = latest[p.feature_names].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if p.dir_feature_names != p.feature_names:
                dir_indices = [p.feature_names.index(fn) for fn in p.dir_feature_names if fn in p.feature_names]
                X_dir = X[:, dir_indices]
            else:
                X_dir = X
            dr = float(p.dir_model.predict_proba(X_dir)[0, 1])
            dc = float(p.dir_cal.predict([dr])[0]) if p.dir_cal else dr
            dir_raws.append(dr)
            if stocks_tested <= 10:
                oi_val = pred.get('_features_array', [[]])[0][p.feature_names.index('fut_oi_buildup')] if 'fut_oi_buildup' in p.feature_names else 0
                print(f'  {sym:15s} sig={sig:5s} dir_raw={dr:.4f} dir_cal={dc:.4f} p_move={pred.get("ml_p_move",0):.3f} oi_build={oi_val:.2f}')

print(f'\nTotals ({stocks_tested} stocks): UP={ups} DOWN={downs} FLAT={flats}')
print(f'dir_raw stats: min={min(dir_raws):.4f} max={max(dir_raws):.4f} mean={np.mean(dir_raws):.4f} std={np.std(dir_raws):.4f}')
unique_sigs = len(set(round(r, 3) for r in dir_raws))
print(f'Unique raw values (rounded to 3dp): {unique_sigs}')
