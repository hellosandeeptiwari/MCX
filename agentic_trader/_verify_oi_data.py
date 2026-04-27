"""Quick verification of OI feature data quality before training."""
import pandas as pd
import numpy as np
import os

oi = pd.read_parquet('ml_models/data/options_oi/all_options_oi_features.parquet')
print("=" * 60)
print("DATA QUALITY VERIFICATION")
print("=" * 60)

# 1. Shape
print(f"\n1. SHAPE: {oi.shape[0]:,} rows x {oi.shape[1]} columns")
print(f"   Columns: {list(oi.columns)}")
dates = oi["trade_date"]
print(f"   Date range: {dates.min()} to {dates.max()}")
print(f"   Unique dates: {dates.nunique()}")
syms = oi["symbol"]
print(f"   Unique symbols: {syms.nunique()}")

# 2. NaN/Inf check
print(f"\n2. NaN/Inf CHECK:")
numeric_cols = oi.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    nan_ct = oi[col].isna().sum()
    inf_ct = np.isinf(oi[col]).sum()
    zero_ct = (oi[col] == 0).sum()
    pct_zero = zero_ct / len(oi) * 100
    status = "WARN" if nan_ct > 0 or inf_ct > 0 else "OK"
    print(f"   {col}: NaN={nan_ct}, Inf={inf_ct}, zeros={zero_ct} ({pct_zero:.1f}%) [{status}]")

# 3. Value distributions
print(f"\n3. VALUE DISTRIBUTIONS:")
for col in numeric_cols:
    d = oi[col].describe()
    print(f"   {col}: min={d['min']:.4f}, p25={d['25%']:.4f}, med={d['50%']:.4f}, p75={d['75%']:.4f}, max={d['max']:.4f}")

# 4. Symbol coverage vs candle data
candle_dir = "ml_models/data/5min"
if os.path.exists(candle_dir):
    candle_files = [f.replace(".parquet", "") for f in os.listdir(candle_dir) if f.endswith(".parquet")]
    oi_symbols = set(oi["symbol"].unique())
    candle_symbols = set(candle_files)
    overlap = oi_symbols & candle_symbols
    only_oi = oi_symbols - candle_symbols
    only_candle = candle_symbols - oi_symbols
    print(f"\n4. SYMBOL COVERAGE:")
    print(f"   OI symbols: {len(oi_symbols)}, Candle symbols: {len(candle_symbols)}")
    print(f"   Overlap: {len(overlap)} ({len(overlap)/len(candle_symbols)*100:.1f}% of candle)")
    print(f"   Only in OI (no candle): {len(only_oi)}")
    print(f"   Only in Candle (no OI): {len(only_candle)}")
    if only_candle and len(only_candle) <= 20:
        print(f"   Missing from OI: {sorted(only_candle)}")

    # Date coverage
    sample = candle_files[0]
    cdf = pd.read_parquet(f"{candle_dir}/{sample}.parquet")
    if "date" in cdf.columns:
        candle_dates = set(pd.to_datetime(cdf["date"]).dt.date.unique())
    elif "timestamp" in cdf.columns:
        candle_dates = set(pd.to_datetime(cdf["timestamp"]).dt.date.unique())
    else:
        candle_dates = set()
    oi_dates = set(pd.to_datetime(oi["trade_date"]).dt.date.unique())
    print(f"\n5. DATE COVERAGE (sample {sample}):")
    print(f"   Candle dates: {len(candle_dates)}")
    print(f"   OI dates:     {len(oi_dates)}")
    if candle_dates:
        date_overlap = candle_dates & oi_dates
        print(f"   Overlap:      {len(date_overlap)} ({len(date_overlap)/len(candle_dates)*100:.1f}% of candle)")
        candle_only = sorted(candle_dates - oi_dates)
        if len(candle_only) <= 15:
            print(f"   Candle-only dates: {candle_only}")
        else:
            print(f"   Candle-only: {len(candle_only)} dates missing from OI (first 5: {candle_only[:5]})")

# 6. Per-stock day counts
print(f"\n6. STOCKS WITH FEW DAYS (< 100):")
per_stock = oi.groupby("symbol")["trade_date"].nunique()
thin = per_stock[per_stock < 100]
if len(thin):
    print(f"   {len(thin)} stocks with < 100 days: {dict(thin.nlargest(10))}")
else:
    print(f"   All {len(per_stock)} stocks have 100+ days ✅")

print(f"\n   Median days/stock: {per_stock.median():.0f}")
print(f"   Min days/stock: {per_stock.min()} ({per_stock.idxmin()})")
print(f"   Max days/stock: {per_stock.max()} ({per_stock.idxmax()})")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
