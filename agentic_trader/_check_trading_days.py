"""Check if Feb 27/28 were trading days"""
import pandas as pd, glob, os

# Check nifty 5min data
nifty_dir = 'ml_models/data/nifty_5min'
files = sorted(glob.glob(os.path.join(nifty_dir, '*.parquet')))
if files:
    df = pd.read_parquet(files[-1])
    print(f"Nifty 5min file: {files[-1]}")
    print(f"  Range: {df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "")
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    # Check for Feb 27 data
    for col in ['date', 'datetime']:
        if col in df.columns:
            feb27 = df[pd.to_datetime(df[col]).dt.date == pd.Timestamp('2026-02-27').date()]
            print(f"  Feb 27 rows: {len(feb27)}")
            feb28 = df[pd.to_datetime(df[col]).dt.date == pd.Timestamp('2026-02-28').date()]
            print(f"  Feb 28 rows: {len(feb28)}")
            mar02 = df[pd.to_datetime(df[col]).dt.date == pd.Timestamp('2026-03-02').date()]
            print(f"  Mar 02 rows: {len(mar02)}")
            break
    else:
        # Try index
        try:
            idx = pd.to_datetime(df.index)
            feb27 = df[idx.date == pd.Timestamp('2026-02-27').date()]
            print(f"  Feb 27 rows (from index): {len(feb27)}")
            feb28 = df[idx.date == pd.Timestamp('2026-02-28').date()]
            print(f"  Feb 28 rows (from index): {len(feb28)}")
            mar02 = df[idx.date == pd.Timestamp('2026-03-02').date()]
            print(f"  Mar 02 rows (from index): {len(mar02)}")
        except:
            print("  Could not parse index as dates")
else:
    print("No nifty 5min files")

# Check daily stock data
print("\n=== Daily Stock Data ===")
daily_dir = 'ml_models/data/daily'
if os.path.exists(daily_dir):
    daily_files = sorted(glob.glob(os.path.join(daily_dir, '*.parquet')))[:3]
    for f in daily_files:
        df = pd.read_parquet(f)
        sym = os.path.basename(f).replace('.parquet', '')
        if 'date' in df.columns:
            print(f"  {sym}: last date = {df['date'].max()}")
else:
    print("  No daily dir")

# Check kite token date for reference
print("\n=== Today's Date Context ===")
from datetime import datetime
print(f"  Now: {datetime.now()}")
print(f"  Weekday: {datetime.now().strftime('%A')}")
