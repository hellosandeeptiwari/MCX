"""Check futures OI data values"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd

for sym in ['LT', 'CAMS', 'BAJAJFINSV', 'ADANIGREEN', 'RELIANCE']:
    path = f'ml_models/data/futures_oi/{sym}_futures_oi.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f'\n=== {sym} ({len(df)} rows) ===')
        print(f'Columns: {df.columns.tolist()}')
        print(df.tail(3).to_string())
        if 'fut_oi_buildup' in df.columns:
            print(f'Last fut_oi_buildup: {df["fut_oi_buildup"].iloc[-1]:.4f}')
    else:
        print(f'{sym}: no OI file')
