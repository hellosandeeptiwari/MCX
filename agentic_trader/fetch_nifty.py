"""Fetch NIFTY50 5-min + daily candles for market context features."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.data_fetcher import get_kite_client, fetch_candles, fetch_daily_candles

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data')
CANDLE_5MIN_DIR = os.path.join(DATA_DIR, 'candles_5min')
DAILY_DIR = os.path.join(DATA_DIR, 'candles_daily')

kite = get_kite_client()
if not kite:
    print("ERROR: No Kite client available")
    sys.exit(1)

print("Kite authenticated")

# Fetch NIFTY 50 5-min candles
print("Fetching NIFTY 50 5-min candles (365 days)...")
df5 = fetch_candles(kite, "NIFTY 50", days=365, interval="5minute")
if len(df5) > 0:
    days_5 = df5['date'].dt.date.nunique()
    print(f"  Got {len(df5):,} candles across {days_5} days")
    path5 = os.path.join(CANDLE_5MIN_DIR, "NIFTY50.parquet")
    df5.to_parquet(path5, index=False)
    print(f"  Saved: {path5}")
else:
    print("  ERROR: No 5-min data returned")

# Fetch NIFTY 50 daily candles
print("Fetching NIFTY 50 daily candles (2000 days)...")
dfd = fetch_daily_candles(kite, "NIFTY 50", days=2000)
if len(dfd) > 0:
    print(f"  Got {len(dfd):,} daily candles")
    pathd = os.path.join(DAILY_DIR, "NIFTY50.parquet")
    dfd.to_parquet(pathd, index=False)
    print(f"  Saved: {pathd}")
else:
    print("  ERROR: No daily data returned")

print("Done!")
