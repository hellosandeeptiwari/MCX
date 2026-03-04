"""Test the intraday gap-fill fix for dhan_futures_oi.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhan_futures_oi import FuturesOIFetcher, DATA_DIR
import pandas as pd

f = FuturesOIFetcher()
print(f"DhanHQ ready: {f.ready}")

# Test with SBIN
print("\n=== SBIN (with gap-fill) ===")
df = f.fetch_daily_futures_oi('SBIN', months_back=1)
if df is not None:
    print(f"Rows: {len(df)}")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")
    print("Last 5 rows:")
    for _, row in df.tail(5).iterrows():
        print(f"  {row['date']}  close={row['close']:.2f}  OI={row['fut_oi']:,.0f}")

# Test with RELIANCE
print("\n=== RELIANCE (with gap-fill) ===")
df2 = f.fetch_daily_futures_oi('RELIANCE', months_back=1)
if df2 is not None:
    print(f"Rows: {len(df2)}")
    print(f"Range: {df2['date'].min()} to {df2['date'].max()}")
    print("Last 5 rows:")
    for _, row in df2.tail(5).iterrows():
        print(f"  {row['date']}  close={row['close']:.2f}  OI={row['fut_oi']:,.0f}")

# Full features test
print("\n=== SBIN features (with gap-fill) ===")
features = f.compute_daily_features('SBIN', months_back=1)
if features is not None:
    print(f"Rows: {len(features)}, Range: {features['date'].min()} to {features['date'].max()}")
    print("Last 3:")
    for _, row in features.tail(3).iterrows():
        print(f"  {row['date'].date()} buildup={row['fut_oi_buildup']:.1f} basis={row['fut_basis_pct']:.3f} oi_chg={row['fut_oi_change_pct']:.1f}%")

# Quick backfill test (3 stocks only)
print("\n=== Mini backfill (3 stocks) ===")
results = f.backfill_all(months_back=1, symbols=['SBIN', 'RELIANCE', 'HDFCBANK'])
for sym, r in results.items():
    print(f"  {sym}: {r}")

# Verify the saved files
print("\n=== Verify saved files ===")
for sym in ['SBIN', 'RELIANCE', 'HDFCBANK']:
    path = os.path.join(DATA_DIR, f'{sym}_futures_oi.parquet')
    if os.path.exists(path):
        saved = pd.read_parquet(path)
        print(f"  {sym}: {len(saved)} rows, last={saved['date'].max()}")
