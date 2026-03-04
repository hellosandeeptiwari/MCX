"""Check exactly how NIFTY data flows to the predictor in autonomous_trader.py"""
import re

bot_path = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'
with open(bot_path) as fh:
    lines = fh.readlines()

# Find predict_for_titan calls and nifty_5min references
print('=== predict_for_titan CALLS ===')
for i, line in enumerate(lines):
    if 'predict_for_titan' in line and not line.strip().startswith('#'):
        # Print context
        start = max(0, i-2)
        end = min(len(lines), i+8)
        print(f'\n--- L{i+1} ---')
        for j in range(start, end):
            marker = '>>>' if j == i else '   '
            print(f'{marker} L{j+1}: {lines[j].rstrip()}')

print('\n\n=== NIFTY DATA LOADING/FETCHING ===')
for i, line in enumerate(lines):
    ll = line.lower()
    if ('nifty' in ll and ('candle' in ll or 'fetch' in ll or 'kite' in ll or '_5min' in ll or '_daily' in ll)) and not line.strip().startswith('#'):
        print(f'  L{i+1}: {line.strip()[:140]}')

print('\n\n=== self._nifty or self.nifty ===')
for i, line in enumerate(lines):
    if re.search(r'self[._]nifty', line, re.IGNORECASE) and not line.strip().startswith('#'):
        print(f'  L{i+1}: {line.strip()[:140]}')
