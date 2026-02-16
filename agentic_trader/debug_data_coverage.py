"""Debug: Check data coverage gap between candles and futures OI."""
import pandas as pd
import os
import glob
import random

# 1. Check how much 5-min candle data we have (training data range)
candle_dir = 'ml_models/data/candles_5min'
files = glob.glob(os.path.join(candle_dir, '*.parquet'))
print('=== 5-MIN CANDLE DATA (TRAINING INPUT) ===')
for f in sorted(files)[:5]:
    df = pd.read_parquet(f)
    sym = os.path.basename(f).replace('.parquet', '')
    days = (df['date'].max() - df['date'].min()).days
    print(f"  {sym}: {len(df)} candles, {df['date'].min().date()} to {df['date'].max().date()} = {days} days")
print(f"  ... {len(files)} total files")

# random sample
random.seed(42)
sample = random.sample(files, min(5, len(files)))
for f in sample:
    df = pd.read_parquet(f)
    sym = os.path.basename(f).replace('.parquet', '')
    print(f"  {sym}: {len(df)} candles, {df['date'].min().date()} to {df['date'].max().date()}")

# 2. Check futures OI data range
oi_dir = 'ml_models/data/futures_oi'
oi_files = glob.glob(os.path.join(oi_dir, '*_futures_oi.parquet'))
print(f"\n=== FUTURES OI DATA ===")
print(f"  Total files: {len(oi_files)}")
for f in sorted(oi_files)[:5]:
    df = pd.read_parquet(f)
    sym = os.path.basename(f).replace('_futures_oi.parquet', '')
    days = (df['date'].max() - df['date'].min()).days
    nz_oi = (df['fut_oi_change_pct'] != 0).sum()
    nz_bu = (df['fut_oi_buildup'] != 0).sum()
    print(f"  {sym}: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()} = {days} days")
    print(f"    Non-zero fut_oi_change_pct: {nz_oi}/{len(df)}, buildup: {nz_bu}/{len(df)}")

# 3. DATE OVERLAP - critical issue
sample_candle = pd.read_parquet(sorted(files)[0])
sample_oi = pd.read_parquet(sorted(oi_files)[0])
candle_dates = set(sample_candle['date'].dt.date.unique())
oi_dates = set(sample_oi['date'].dt.date.unique())
overlap = candle_dates & oi_dates
print(f"\n=== DATE OVERLAP ({os.path.basename(sorted(files)[0])}) ===")
print(f"  Candle dates: {min(candle_dates)} to {max(candle_dates)} ({len(candle_dates)} unique days)")
print(f"  OI dates: {min(oi_dates)} to {max(oi_dates)} ({len(oi_dates)} unique days)")
print(f"  Overlap: {len(overlap)} days")
print(f"  Candle days WITHOUT OI: {len(candle_dates - oi_dates)}")
print(f"  OI days WITHOUT candles: {len(oi_dates - candle_dates)}")

# 4. Check DhanHQ historical API limits
print("\n=== DHAN API LIMITS CHECK ===")
print("  Current backfill used months_back=6")
print("  DhanHQ historical API supports up to 5 YEARS of daily data")
print("  We should be fetching much more!")

# 5. Check what the backfill code limited to
print("\n=== CHECKING BACKFILL CODE LIMITS ===")
with open('dhan_futures_oi.py', 'r') as f:
    content = f.read()
    
# Find months_back references
for i, line in enumerate(content.split('\n'), 1):
    if 'months_back' in line.lower() or 'from_date' in line.lower() or 'year' in line.lower():
        print(f"  Line {i}: {line.strip()}")
