"""Map all data sources available from Kite + Titan for ML model planning"""
import re

# 1. What indicators does Titan compute?
with open('zerodha_tools.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

matches = re.findall(r"result\['(\w+)'\]", content)
unique_keys = sorted(set(matches))
print(f'=== INDICATOR KEYS FROM _calculate_indicators() ({len(unique_keys)} total) ===')
for i, k in enumerate(unique_keys):
    print(f'  {i+1:>2}. {k}')

# 2. Kite data sources
print('\n=== DATA YOU CAN GET FROM KITE API ===')
print()
print('1. quote() — REAL-TIME SNAPSHOT (per stock)')
print('   last_price, last_quantity, average_price, volume')
print('   buy_quantity, sell_quantity (total pending)')
print('   ohlc (open, high, low, close)')
print('   oi (open interest), oi_day_high, oi_day_low')
print('   depth: 5-level bid/ask (price, quantity, orders per level)')
print('   net_change')
print()
print('2. historical_data() — CANDLE DATA (stored)')
print('   date, open, high, low, close, volume')
print('   Intervals: minute, 3min, 5min, 10min, 15min, 30min, 60min, day')
print('   Max lookback: 60d(1min), 100d(3-5min), 200d(15-30min), 2000d(day)')
print()
print('3. instruments() — STOCK/OPTION UNIVERSE')
print('   instrument_token, tradingsymbol, name, expiry, strike')
print('   tick_size, lot_size, instrument_type (EQ/FUT/CE/PE)')
print('   segment, exchange')
print()
print('4. WebSocket TICKS — REAL-TIME STREAMING')
print('   last_price, last_quantity, avg_price, volume')
print('   buy_quantity, sell_quantity, ohlc, oi, change')
print('   depth (5-level bid/ask), exchange timestamp')
print()
print('5. positions() / orders() / trades() — EXECUTION DATA')
print('   fill_price, fill_quantity, order_timestamp')
print('   status, rejection_reason, average_price')
print()
print('6. option_chain via instruments() — OPTION UNIVERSE')
print('   All strikes × all expiries for any underlying')
print('   Can call quote() on any option to get: LTP, OI, bid/ask depth, volume')
print('   NO historical option prices (only current snapshot)')

# 3. What Titan already records per trade
print('\n=== WHAT TITAN RECORDS PER TRADE (trade_history.json) ===')
print('  26 entry_metadata fields:')
fields = [
    'entry_score', 'score_tier', 'acceleration_score',
    'adx', 'rsi', 'follow_through_candles', 'range_expansion_ratio',
    'orb_signal', 'orb_strength_pct', 'volume_ratio', 'volume_regime',
    'vwap_position', 'htf_alignment', 'trend_state', 'change_pct',
    'microstructure_score', 'microstructure_block',
    'spot_price', 'strategy_type', 'cap_pct_used',
    'delta', 'theta', 'iv', 'strike', 'expiry',
    'pnl', 'result', 'exit_time', 'closed_at'
]
for f in fields:
    print(f'    {f}')
