import json, sqlite3

db = sqlite3.connect('titan_state.db')
db.row_factory = sqlite3.Row

# Check trade flags
for row in db.execute("SELECT trade_json FROM active_trades WHERE date = '2026-03-11'"):
    val = row['trade_json']
    if not val:
        continue
    try:
        t = json.loads(val)
    except Exception:
        continue
    trades = [t] if isinstance(t, dict) else t if isinstance(t, list) else []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        sym = trade.get('symbol', '')
        if 'COLPAL' in sym:
            keys = ['symbol','direction','side','is_debit_spread','hedged_from_tie','hedge_unwound',
                    'buy_symbol','sell_symbol','buy_premium','sell_premium','net_premium','net_debit',
                    'spread_width','avg_price','entry_price','quantity','setup_type','strategy_type',
                    'underlying_entry_ltp','underlying_symbol','timestamp']
            for k in keys:
                print(f'  {k}: {trade.get(k, "N/A")}')
            print('--- hedge/spread/tie keys ---')
            for k, v in trade.items():
                kl = k.lower()
                if any(x in kl for x in ['hedge','tie','unwind','spread','debit']):
                    print(f'  [{k}]: {v}')

# Check live_pnl for COLPAL
print('\n--- Live PnL ---')
for r in db.execute("SELECT symbol, ltp, unrealized_pnl FROM live_pnl WHERE date = '2026-03-11'"):
    if 'COLPAL' in r['symbol']:
        print(f"  {r['symbol']} | LTP: {r['ltp']} | uPnL: {r['unrealized_pnl']}")

# Check unwind calc
print('\n--- Unwind Check ---')
buy_entry = 65.0
sell_entry = 33.5
recovery_pct = 85
threshold = buy_entry * (recovery_pct / 100)
print(f'  Buy entry: {buy_entry}, Threshold (85%): {threshold}')
print(f'  Buy leg needs LTP >= {threshold} to trigger unwind')
print(f'  Sell entry: {sell_entry}')
print(f'  Dashboard shows LTP:32 which is the SPREAD value, NOT the buy leg LTP')

