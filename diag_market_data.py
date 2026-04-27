"""Diagnostic: Compare market_data dicts for INTRADAY_NEWS batch stocks.
Run on EC2 to check if get_market_data returns identical data for different stocks.
"""
import sys, os
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
os.chdir('/home/ubuntu/titan/agentic_trader')

# Load .env
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/titan/agentic_trader/.env')

from zerodha_tools import ZerodhaTools

tools = ZerodhaTools()
symbols = ['NSE:ANGELONE', 'NSE:WIPRO', 'NSE:HDFCBANK', 'NSE:APOLLOHOSP']

md = tools.get_market_data(symbols, force_fresh=True)

# Compare key fields
keys_to_check = ['ltp', 'open', 'high', 'low', 'rsi_14', 'adx', 'vwap', 'ema_9', 'ema_21',
                  'orb_signal', 'orb_high', 'orb_low', 'volume_regime', 'price_vs_vwap',
                  'vwap_slope', 'vwap_change_pct', 'ema_regime', 'ema_spread',
                  'htf_alignment', 'follow_through_candles', 'range_expansion_ratio',
                  'change_pct', 'volume', 'volume_ratio']

print("=== MARKET DATA COMPARISON ===")
for k in keys_to_check:
    vals = {s.replace('NSE:', ''): md.get(s, {}).get(k, '?') for s in symbols if isinstance(md.get(s), dict)}
    all_same = len(set(str(v) for v in vals.values())) <= 1
    flag = " ⚠️ IDENTICAL" if all_same and len(vals) > 1 else ""
    print(f"  {k:30s}: {vals}{flag}")

# Count how many fields are identical across all stocks
md_dicts = [md[s] for s in symbols if isinstance(md.get(s), dict)]
if len(md_dicts) > 1:
    all_keys = set(md_dicts[0].keys())
    identical_count = 0
    differ_count = 0
    for k in sorted(all_keys):
        vals = [str(d.get(k, '')) for d in md_dicts]
        if len(set(vals)) <= 1:
            identical_count += 1
        else:
            differ_count += 1
    print(f"\n=== SUMMARY: {identical_count} identical fields, {differ_count} different fields out of {len(all_keys)} total ===")
    print(f"Identical fields:")
    for k in sorted(all_keys):
        vals = [str(d.get(k, '')) for d in md_dicts]
        if len(set(vals)) <= 1:
            print(f"  {k}: {md_dicts[0].get(k, '')}")
