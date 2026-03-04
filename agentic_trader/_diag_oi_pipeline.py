"""Diagnose OI data pipeline: why is it stale at Feb 26?"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd

# 1. Check stored parquet dates
print('=== STORED OI PARQUET FILES ===')
files = sorted(glob.glob('ml_models/data/futures_oi/*.parquet'))
print(f'Total files: {len(files)}')
for f in files[:5]:
    df = pd.read_parquet(f)
    sym = os.path.basename(f).replace('_futures_oi.parquet','')
    print(f'  {sym}: {len(df)} rows, {df.date.min()} to {df.date.max()}')

# 2. Check live OI log files
print('\n=== LIVE OI LOG FILES ===')
oi_log_dir = 'oi_logs'
if os.path.isdir(oi_log_dir):
    for f in sorted(os.listdir(oi_log_dir))[-5:]:
        path = os.path.join(oi_log_dir, f)
        print(f'  {f}: {os.path.getsize(path)} bytes')
else:
    print(f'  {oi_log_dir}/ not found')

# 3. Check if dhan_futures_oi fetcher exists and when it last ran
print('\n=== DHAN OI FETCHER ===')
for p in ['dhan_futures_oi.py', 'dhan_oi_fetcher.py', 'nse_oi_fetcher.py']:
    if os.path.exists(p):
        stat = os.stat(p)
        print(f'  {p}: {stat.st_size} bytes, modified {pd.Timestamp(stat.st_mtime, unit="s")}')

# 4. Check crontab for OI fetch jobs
print('\n=== CHECKING CRONTAB ===')
import subprocess
try:
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True, timeout=5)
    oi_lines = [l for l in r.stdout.split('\n') if 'oi' in l.lower() or 'futures' in l.lower() or 'backfill' in l.lower()]
    if oi_lines:
        for l in oi_lines:
            print(f'  {l}')
    else:
        print('  No OI-related cron jobs found')
    # Also check root crontab
    r2 = subprocess.run(['sudo', 'crontab', '-l'], capture_output=True, text=True, timeout=5)
    oi_lines2 = [l for l in r2.stdout.split('\n') if 'oi' in l.lower() or 'futures' in l.lower() or 'backfill' in l.lower()]
    if oi_lines2:
        print('  ROOT crontab:')
        for l in oi_lines2:
            print(f'    {l}')
except Exception as e:
    print(f'  crontab check failed: {e}')

# 5. Check systemd timers
print('\n=== SYSTEMD TIMERS ===')
try:
    r = subprocess.run(['systemctl', 'list-timers', '--all'], capture_output=True, text=True, timeout=5)
    for l in r.stdout.split('\n'):
        if 'oi' in l.lower() or 'futures' in l.lower() or 'backfill' in l.lower() or 'fetch' in l.lower():
            print(f'  {l}')
    if not any('oi' in l.lower() or 'fetch' in l.lower() for l in r.stdout.split('\n')):
        print('  No OI-related timers found')
except Exception as e:
    print(f'  timer check failed: {e}')

# 6. Check how the bot fetches OI during scan
print('\n=== BOT OI FETCH MECHANISM ===')
import subprocess
r = subprocess.run(['grep', '-n', 'futures_oi', 'autonomous_trader.py'], capture_output=True, text=True)
lines = r.stdout.strip().split('\n')
for l in lines[:15]:
    print(f'  {l.strip()}')
print(f'  ... ({len(lines)} total matches)')
