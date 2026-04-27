import os, glob, pandas as pd
from pathlib import Path
from download_oi_history import DATA_DIR as OPT_DIR, load_all_options_oi_daily
print('options_oi DATA_DIR:', OPT_DIR, 'exists=', OPT_DIR.exists())
if OPT_DIR.exists():
    files = list(OPT_DIR.glob('*_options_oi.parquet'))
    print('parquet files:', len(files))
    for f in files[:5]:
        print(' ', f.name, f.stat().st_size)
od = load_all_options_oi_daily() or {}
print('loaded symbols:', len(od))

# Time a single analyze() call to see why aggr times out
from oi_analyzer import OIAnalyzer  # may be different name
