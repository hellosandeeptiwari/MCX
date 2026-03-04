"""Fetch NIFTY March 2 data directly from Kite API to compare with stored parquet."""
import sys, os
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from dotenv import load_dotenv
load_dotenv('/home/ubuntu/titan/agentic_trader/.env')

from kiteconnect import KiteConnect

api_key = os.environ.get('ZERODHA_API_KEY', '')
access_token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Verify connection
try:
    p = kite.profile()
    uname = p["user_name"]
    print(f"Connected as: {uname}")
except Exception as e:
    print(f"Auth error: {e}")
    sys.exit(1)

from datetime import datetime, timedelta

# Fetch NIFTY 50 5-min candles for March 2
# NIFTY 50 instrument token = 256265
print('=== KITE API: NIFTY 50 5-min candles for March 2, 2026 ===')
try:
    data = kite.historical_data(
        instrument_token=256265,
        from_date=datetime(2026, 3, 2),
        to_date=datetime(2026, 3, 2, 23, 59),
        interval='5minute'
    )
    print(f'Candles returned: {len(data)}')
    if data:
        print(f'\nFirst candle: {data[0]}')
        print(f'Last candle:  {data[-1]}')
        open_price = data[0]['open']
        close_price = data[-1]['close']
        day_return = (close_price - open_price) / open_price * 100
        print(f'\nDay open:  {open_price}')
        print(f'Day close: {close_price}')
        print(f'Day return: {day_return:.2f}%')
        
        # Show all candles
        print(f'\nAll {len(data)} candles:')
        for c in data:
            dt, o, h, l, cl, v = c["date"], c["open"], c["high"], c["low"], c["close"], c["volume"]
            print(f'  {dt}  O={o:.1f}  H={h:.1f}  L={l:.1f}  C={cl:.1f}  V={v}')
except Exception as e:
    print(f'ERROR: {e}')

# Also fetch March 3 and March 4
for d in ['2026-03-03', '2026-03-04']:
    print(f'\n=== KITE API: NIFTY 50 5-min for {d} ===')
    try:
        dt = datetime.strptime(d, '%Y-%m-%d')
        data = kite.historical_data(
            instrument_token=256265,
            from_date=dt,
            to_date=dt + timedelta(hours=23, minutes=59),
            interval='5minute'
        )
        print(f'Candles: {len(data)}')
        if data:
            first = data[0]
            last = data[-1]
            print(f'  First: {first["date"]} O={first["open"]:.1f} C={first["close"]:.1f}')
            print(f'  Last:  {last["date"]} O={last["open"]:.1f} C={last["close"]:.1f}')
            day_ret = (last['close'] - first['open']) / first['open'] * 100
            print(f'  Day return: {day_ret:.2f}%')
    except Exception as e:
        print(f'ERROR: {e}')

# Also fetch NIFTY daily for last 3 months to see how much history is available
print(f'\n\n=== KITE API: NIFTY 50 DAILY (last 120 days) ===')
try:
    data = kite.historical_data(
        instrument_token=256265,
        from_date=datetime(2025, 11, 1),
        to_date=datetime(2026, 3, 4, 23, 59),
        interval='day'
    )
    print(f'Daily candles returned: {len(data)}')
    if data:
        first = data[0]
        last = data[-1]
        print(f'First: {first["date"]} C={first["close"]:.1f}')
        print(f'Last:  {last["date"]} C={last["close"]:.1f}')
        # Show last 10
        print('\nLast 10 daily candles:')
        for c in data[-10:]:
            dt, o, h, l, cl = c["date"], c["open"], c["high"], c["low"], c["close"]
            print(f'  {dt}  O={o:.1f}  H={h:.1f}  L={l:.1f}  C={cl:.1f}')
except Exception as e:
    print(f'ERROR: {e}')
