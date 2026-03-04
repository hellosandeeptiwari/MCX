"""Check if NIFTY data is actually reaching the model predictions."""
import sys, os, glob
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

# Check what NIFTY data files exist
data_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data'
nifty_files = glob.glob(os.path.join(data_dir, '*[Nn][Ii][Ff][Tt]*'))
print('NIFTY files in ml_models/data/:')
for f in nifty_files:
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.0f} KB)')
    
# Check all parquet/csv files in data dir 
all_files = glob.glob(os.path.join(data_dir, '*.parquet')) + glob.glob(os.path.join(data_dir, '*.csv'))
print(f'\nAll data files ({len(all_files)}):')
for f in sorted(all_files):
    print(f'  {os.path.basename(f)} ({os.path.getsize(f)/1024:.0f} KB)')

# Check the autonomous_trader.py - how does it pass NIFTY data to predictor?
# Search for nifty_5min in the main bot
import re
bot_path = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'
with open(bot_path) as fh:
    content = fh.read()

# Find where predict_for_titan is called
matches = [(i+1, line.strip()) for i, line in enumerate(content.split('\n')) 
           if 'predict_for_titan' in line or 'nifty_5min' in line or 'nifty_daily' in line]
print(f'\nReferences to predict_for_titan/nifty_5min/nifty_daily in autonomous_trader.py:')
for lineno, line in matches[:30]:
    print(f'  L{lineno}: {line[:120]}')

# Also check where NIFTY candles are loaded
matches2 = [(i+1, line.strip()) for i, line in enumerate(content.split('\n'))
            if 'nifty' in line.lower() and ('candle' in line.lower() or 'fetch' in line.lower() or 'load' in line.lower() or 'kite' in line.lower())]
print(f'\nNIFTY candle loading:')
for lineno, line in matches2[:20]:
    print(f'  L{lineno}: {line[:120]}')
