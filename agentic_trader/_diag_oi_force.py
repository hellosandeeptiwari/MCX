"""Force-refresh and deep test DhanHQ OI API"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhan_futures_oi import FuturesOIFetcher, DATA_DIR
import pandas as pd

f = FuturesOIFetcher()
print(f"Token ends: ...{f.access_token[-6:]}")

# Force refresh instrument master
cache = os.path.join(DATA_DIR, 'futures_contracts.json')
if os.path.exists(cache):
    os.remove(cache)
    print("Deleted stale contract cache")

contracts = f.load_instrument_master(force_refresh=True)
sbin = contracts.get('SBIN', [])
print(f"\nSBIN contracts: {len(sbin)}")
for c in sbin:
    print(f"  {c}")

# Fetch daily with fresh master
print("\n=== SBIN daily (fresh master) ===")
df = f.fetch_daily_futures_oi('SBIN', months_back=1)
if df is not None:
    print(f"Rows: {len(df)}")
    print(f"Range: {df['date'].min()} to {df['date'].max()}")
    for _, row in df.tail(5).iterrows():
        print(f"  {row['date']}  close={row['close']:.2f}  OI={row['fut_oi']:,.0f}")
else:
    print("No data returned")

# Try intraday for Feb 27
print("\n=== SBIN INTRADAY Feb 27 ===")
sec_id = sbin[0]['id'] if sbin else '52030'
st, d = f._request('POST', '/charts/intraday', json_body={
    'securityId': sec_id,
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'interval': '5',
    'oi': True,
    'fromDate': '2026-02-27 09:15:00',
    'toDate': '2026-02-27 15:30:00'
})
print(f"Status: {st}")
if d.get('timestamp'):
    dates = pd.to_datetime(d['timestamp'], unit='s')
    print(f"Rows: {len(dates)}, {dates.min()} to {dates.max()}")
else:
    resp_str = json.dumps(d) if isinstance(d, dict) else str(d)
    print(f"Response: {resp_str[:400]}")

# Try intraday for today Mar 2
print("\n=== SBIN INTRADAY Today (Mar 2) ===")
st2, d2 = f._request('POST', '/charts/intraday', json_body={
    'securityId': sec_id,
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'interval': '5',
    'oi': True,
    'fromDate': '2026-03-02 09:15:00',
    'toDate': '2026-03-02 15:30:00'
})
print(f"Status: {st2}")
if d2.get('timestamp'):
    dates2 = pd.to_datetime(d2['timestamp'], unit='s')
    print(f"Rows: {len(dates2)}, {dates2.min()} to {dates2.max()}")
    oi_list = d2.get('open_interest', [])
    if oi_list:
        print(f"Last OI: {oi_list[-1]:,}")
else:
    resp_str2 = json.dumps(d2) if isinstance(d2, dict) else str(d2)
    print(f"Response: {resp_str2[:400]}")

# Also check SPOT equity for Feb 27
print("\n=== SBIN SPOT EQUITY Feb 25-Mar 2 ===")
st3, d3 = f._request('POST', '/charts/historical', json_body={
    'securityId': '3045',
    'exchangeSegment': 'NSE_EQ',
    'instrument': 'EQUITY',
    'fromDate': '2026-02-25',
    'toDate': '2026-03-02',
})
print(f"Status: {st3}")
if d3.get('timestamp'):
    dates3 = pd.to_datetime(d3['timestamp'], unit='s')
    print(f"Rows: {len(dates3)}")
    for dt in dates3:
        print(f"  {dt}")
else:
    resp_str3 = json.dumps(d3) if isinstance(d3, dict) else str(d3)
    print(f"Response: {resp_str3[:400]}")

# Try historical for the FULL range including today
print("\n=== SBIN FUTURES DAILY Feb 25-Mar 2 (historical) ===")
st4, d4 = f._request('POST', '/charts/historical', json_body={
    'securityId': sec_id,
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'oi': True,
    'fromDate': '2026-02-25',
    'toDate': '2026-03-02',
})
print(f"Status: {st4}")
if d4.get('timestamp'):
    dates4 = pd.to_datetime(d4['timestamp'], unit='s')
    oi4 = d4.get('open_interest', [])
    close4 = d4.get('close', [])
    for i, dt in enumerate(dates4):
        oi_val = oi4[i] if i < len(oi4) else '?'
        cl_val = close4[i] if i < len(close4) else '?'
        print(f"  {dt}  close={cl_val}  OI={oi_val}")
else:
    resp_str4 = json.dumps(d4) if isinstance(d4, dict) else str(d4)
    print(f"Response: {resp_str4[:400]}")
