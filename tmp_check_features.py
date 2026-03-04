"""Check what NIFTY and OI feature values the model is seeing TODAY."""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from ml_models.feature_engineering import compute_features

# Load recent NIFTY 5min data
nifty_5m_path = '/home/ubuntu/titan/agentic_trader/ml_models/data/NIFTY_5min.parquet'
if os.path.exists(nifty_5m_path):
    nifty_5m = pd.read_parquet(nifty_5m_path)
    print(f'NIFTY 5min: {len(nifty_5m)} rows, last={nifty_5m["date"].max()}')
    last_nifty = nifty_5m.tail(3)
    for _, r in last_nifty.iterrows():
        print(f'  {r["date"]} close={r["close"]:.1f}')
else:
    print('NO NIFTY 5min file')

# Load NIFTY daily
nifty_d_path = '/home/ubuntu/titan/agentic_trader/ml_models/data/NIFTY_daily.parquet'
if os.path.exists(nifty_d_path):
    nifty_d = pd.read_parquet(nifty_d_path)
    print(f'\nNIFTY daily: {len(nifty_d)} rows, last={nifty_d["date"].max()}')

# Check a few stocks' OI features  
oi_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data/futures_oi'
for sym in ['RELIANCE', 'TATASTEEL', 'SBIN', 'HDFCBANK', 'INFY']:
    path = os.path.join(oi_dir, f'{sym}_futures_oi.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        last = df.iloc[-1]
        print(f'\n{sym} last OI ({last.get("date","?")}):')
        print(f'  buildup={last.get("fut_oi_buildup",0)} basis={last.get("fut_basis_pct",0):.3f}% vol_ratio={last.get("fut_vol_ratio",0):.2f} trend={last.get("fut_oi_5d_trend",0):.1f}%')

# Count how many stocks have buildup > 0 
print('\n\n=== OI BUILDUP DISTRIBUTION (latest date) ===')
import glob
files = glob.glob(os.path.join(oi_dir, '*_futures_oi.parquet'))
buildups = {}
for f in files:
    sym = os.path.basename(f).replace('_futures_oi.parquet', '')
    df = pd.read_parquet(f)
    if len(df) > 0:
        b = df.iloc[-1].get('fut_oi_buildup', 0)
        buildups[sym] = b

total = len(buildups)
long_buildup = sum(1 for v in buildups.values() if v == 1.0)     # +1
short_covering = sum(1 for v in buildups.values() if v == 0.5)   # +0.5
short_buildup = sum(1 for v in buildups.values() if v == -1.0)   # -1
long_unwind = sum(1 for v in buildups.values() if v == -0.5)     # -0.5
zero = sum(1 for v in buildups.values() if v == 0)

print(f'Total stocks with OI: {total}')
print(f'  Long buildup (+1):  {long_buildup} ({100*long_buildup/total:.0f}%)')
print(f'  Short covering (+0.5): {short_covering} ({100*short_covering/total:.0f}%)')
print(f'  Short buildup (-1): {short_buildup} ({100*short_buildup/total:.0f}%)')
print(f'  Long unwinding (-0.5): {long_unwind} ({100*long_unwind/total:.0f}%)')
print(f'  Zero/neutral: {zero} ({100*zero/total:.0f}%)')
print(f'\nBULLISH OI (+1 or +0.5): {long_buildup + short_covering} ({100*(long_buildup+short_covering)/total:.0f}%)')
print(f'BEARISH OI (-1 or -0.5): {short_buildup + long_unwind} ({100*(short_buildup+long_unwind)/total:.0f}%)')
