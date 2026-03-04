"""Quick diagnostic: Why is Futures OI data stuck at Feb 26?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhan_futures_oi import FuturesOIFetcher, DATA_DIR
import pandas as pd
from datetime import datetime

fetcher = FuturesOIFetcher()
print(f"=== DhanHQ OI Fetcher ===")
print(f"  ready: {fetcher.ready}")
print(f"  client_id: {'***' + fetcher.client_id[-4:] if fetcher.client_id else 'MISSING'}")
print(f"  access_token: {'***' + fetcher.access_token[-4:] if fetcher.access_token else 'MISSING'}")

if not fetcher.ready:
    print("FATAL: DhanHQ credentials not configured!")
    sys.exit(1)

# Test 1: instrument master
print(f"\n=== Instrument Master ===")
contracts = fetcher.load_instrument_master()
print(f"  Total symbols with contracts: {len(contracts)}")

# Test nearest contract for SBIN
print(f"\n=== Nearest Contract (SBIN) ===")
nearest = fetcher.get_nearest_contract('SBIN')
print(f"  {nearest}")

# Test with RELIANCE too
nearest_rel = fetcher.get_nearest_contract('RELIANCE')
print(f"  RELIANCE: {nearest_rel}")

# Test 2: Raw API call for SBIN
print(f"\n=== Raw Daily Fetch: SBIN ===")
fut_df = fetcher.fetch_daily_futures_oi('SBIN', months_back=1)
if fut_df is not None:
    print(f"  Rows: {len(fut_df)}")
    print(f"  Date range: {fut_df['date'].min()} to {fut_df['date'].max()}")
    print(f"  Last 5 dates:")
    for _, row in fut_df.tail(5).iterrows():
        print(f"    {row['date'].date()} OI={row['fut_oi']:,.0f} Close={row['close']:.2f}")
else:
    print("  FAILED: no data returned")

# Test 3: Full features for SBIN
print(f"\n=== Compute Features: SBIN ===")
features = fetcher.compute_daily_features('SBIN', months_back=1)
if features is not None:
    print(f"  Rows: {len(features)}")
    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Last 3 rows:")
    for _, row in features.tail(3).iterrows():
        print(f"    {row['date'].date()} buildup={row['fut_oi_buildup']:.1f} basis={row['fut_basis_pct']:.3f}")
else:
    print("  FAILED")

# Test 4: Check existing parquet
print(f"\n=== Current Stored Data (SBIN) ===")
sbin_path = os.path.join(DATA_DIR, 'SBIN_futures_oi.parquet')
if os.path.exists(sbin_path):
    stored = pd.read_parquet(sbin_path)
    print(f"  Rows: {len(stored)}, Date range: {stored['date'].min()} to {stored['date'].max()}")
else:
    print("  No stored data")

# Test 5: Check contract cache for rollover info
cache_file = os.path.join(DATA_DIR, 'futures_contracts.json')
if os.path.exists(cache_file):
    import json
    mtime = os.path.getmtime(cache_file)
    age_hours = (datetime.now().timestamp() - mtime) / 3600
    print(f"\n=== Contract Cache ===")
    print(f"  Age: {age_hours:.1f} hours")
    with open(cache_file) as f:
        cc = json.load(f)
    sbin_ctrs = cc.get('SBIN', [])
    print(f"  SBIN contracts: {len(sbin_ctrs)}")
    for c in sbin_ctrs[:4]:
        print(f"    {c['sym']} expiry={c['expiry'][:10]} id={c['id']}")
