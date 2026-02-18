"""
Fetch NIFTY sector index 5-min + daily candles for sector context features.

Downloads candles for all major sector indices (NIFTY METAL, NIFTY IT, etc.)
and saves them as parquet files alongside stock candles.

These are used by `feature_engineering.py` to compute sector-relative features
so the ML model can detect sector-incongruent signals (e.g., buying JINDALSTEL CE
when NIFTY METAL is tanking).

Usage:
    python fetch_sector_indices.py              # Fetch all sector indices (365 days)
    python fetch_sector_indices.py --days 500   # Fetch 500 days
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.data_fetcher import get_kite_client, fetch_candles, fetch_daily_candles

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data')
CANDLE_5MIN_DIR = os.path.join(DATA_DIR, 'candles_5min')
DAILY_DIR = os.path.join(DATA_DIR, 'candles_daily')

# Sector indices to fetch (must match STOCK_SECTOR_MAP in feature_engineering.py)
# Format: (Kite symbol, parquet filename)
SECTOR_INDICES = [
    ("NIFTY METAL",    "SECTOR_METAL"),
    ("NIFTY IT",       "SECTOR_IT"),
    ("NIFTY BANK",     "SECTOR_BANK"),
    ("NIFTY AUTO",     "SECTOR_AUTO"),
    ("NIFTY PHARMA",   "SECTOR_PHARMA"),
    ("NIFTY ENERGY",   "SECTOR_ENERGY"),
    ("NIFTY FMCG",     "SECTOR_FMCG"),
    ("NIFTY REALTY",   "SECTOR_REALTY"),
    ("NIFTY INFRA",    "SECTOR_INFRA"),
]


def main():
    parser = argparse.ArgumentParser(description='Fetch sector index candles')
    parser.add_argument('--days', type=int, default=365, help='Days of 5-min history')
    parser.add_argument('--daily-days', type=int, default=2000, help='Days of daily history')
    args = parser.parse_args()

    kite = get_kite_client()
    if not kite:
        print("ERROR: No Kite client available. Run Titan first to authenticate.")
        sys.exit(1)

    print("Kite authenticated")
    os.makedirs(CANDLE_5MIN_DIR, exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    success = 0
    failed = 0

    for kite_symbol, filename in SECTOR_INDICES:
        # 5-min candles
        print(f"\nFetching {kite_symbol} 5-min candles ({args.days} days)...")
        try:
            df5 = fetch_candles(kite, kite_symbol, days=args.days, interval="5minute")
            if len(df5) > 0:
                days_count = df5['date'].dt.date.nunique()
                path5 = os.path.join(CANDLE_5MIN_DIR, f"{filename}.parquet")
                df5.to_parquet(path5, index=False)
                print(f"  ✅ {len(df5):,} candles across {days_count} days → {path5}")
                success += 1
            else:
                print(f"  ❌ No 5-min data for {kite_symbol}")
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1

        # Daily candles (longer history for daily context)
        print(f"Fetching {kite_symbol} daily candles ({args.daily_days} days)...")
        try:
            dfd = fetch_daily_candles(kite, kite_symbol, days=args.daily_days)
            if len(dfd) > 0:
                pathd = os.path.join(DAILY_DIR, f"{filename}.parquet")
                dfd.to_parquet(pathd, index=False)
                print(f"  ✅ {len(dfd):,} daily candles → {pathd}")
            else:
                print(f"  ⚠  No daily data for {kite_symbol}")
        except Exception as e:
            print(f"  ⚠  Daily error: {e}")

    print(f"\n{'='*50}")
    print(f"Done: {success} succeeded, {failed} failed")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
