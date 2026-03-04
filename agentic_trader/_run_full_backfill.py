"""Run full OI backfill for all stocks"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dhan_futures_oi import FuturesOIFetcher

f = FuturesOIFetcher()
if not f.ready:
    print("FATAL: DhanHQ not ready")
    sys.exit(1)

r = f.backfill_all(months_back=1)
ok = sum(1 for v in r.values() if v['status'] == 'ok')
failed = sum(1 for v in r.values() if v['status'] == 'failed')
no_contract = sum(1 for v in r.values() if v['status'] == 'no_contract')
print(f"\nSUMMARY: {ok} OK, {failed} failed, {no_contract} no_contract out of {len(r)} total")

# Verify a few
import pandas as pd
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'futures_oi')
for sym in ['SBIN', 'RELIANCE', 'HDFCBANK', 'TCS', 'TATASTEEL']:
    path = os.path.join(DATA_DIR, f'{sym}_futures_oi.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  {sym}: {len(df)} rows, last={df['date'].max()}")
