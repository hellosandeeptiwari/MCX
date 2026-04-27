import sys
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from config import FNO_LOT_SIZES
syms = ['PFC', 'AUBANK', 'POWERINDIA', 'KAYNES', 'RVNL', 'KEI', 'COFORGE', 'CDSL', 'TATATECH']
for s in syms:
    lot = FNO_LOT_SIZES.get(s)
    print(f"  {s}: {lot}")
# Also check similar keys
matches = [k for k in sorted(FNO_LOT_SIZES.keys()) if 'PFC' in k]
print(f"\nKeys containing PFC: {matches}")
matches2 = [k for k in sorted(FNO_LOT_SIZES.keys()) if 'AUBANK' in k or 'AU ' in k]
print(f"Keys containing AUBANK: {matches2}")
print(f"\nTotal FNO symbols: {len(FNO_LOT_SIZES)}")
