import json, os, glob
from dhan_futures_oi import load_all_futures_oi_daily
d = load_all_futures_oi_daily() or {}
print('Total symbols in futures_oi_data:', len(d))
k = 'NSE:MANAPPURAM'
print('MANAPPURAM key present:', k in d)
if k in d:
    df = d[k]
    print('rows:', len(df))
    if len(df):
        last = df.iloc[-1].to_dict()
        print('last row keys:', list(last.keys()))
        print('fut_oi_buildup:', last.get('fut_oi_buildup'))
        print('date:', last.get('date'))
else:
    matches = [s for s in d if 'MANAPPURAM' in s.upper()]
    print('MANAPPURAM-like keys:', matches[:5])
    print('sample keys:', list(d.keys())[:10])
# Check options OI history
try:
    from download_oi_history import load_all_options_oi_daily
    od = load_all_options_oi_daily() or {}
    print('\nOptions OI symbols:', len(od))
    ok = [s for s in od if 'MANAPPURAM' in s.upper()]
    print('MANAPPURAM options OI keys:', ok[:5])
except Exception as e:
    print('options OI load error:', e)
# List local parquet files
print('\nParquet files for MANAPPURAM:')
for p in glob.glob('/home/ubuntu/titan/agentic_trader/data/futures_oi/*MANAPPURAM*'):
    print(' ', p, os.path.getsize(p))
for p in glob.glob('/home/ubuntu/titan/agentic_trader/data/**/*MANAPPURAM*', recursive=True):
    print(' ', p)
