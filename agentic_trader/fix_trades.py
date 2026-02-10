import json
from datetime import datetime

# === Fix active_trades.json ===
with open('active_trades.json', 'r') as f:
    data = json.load(f)

# Load existing trade history
try:
    with open('trade_history.json', 'r') as f:
        history = json.load(f)
except:
    history = []

new_active = []
for t in data['active_trades']:
    sym = t.get('symbol', '')
    status = t.get('status', 'OPEN')
    
    if status == 'OPEN':
        # Keep open trades as-is
        new_active.append(t)
        print(f"  KEEP OPEN: {sym}")
    elif status in ('OPTION_SPEED_GATE', 'SL_HIT', 'STOPLOSS_HIT', 'TARGET_HIT'):
        # Move legitimately closed trades to history
        t['result'] = status
        t['closed_at'] = t.get('exit_time', datetime.now().isoformat())
        history.append(t)
        print(f"  -> HISTORY: {sym} ({status}, pnl={t.get('pnl', 0):.0f})")
    else:
        new_active.append(t)
        print(f"  KEEP OTHER: {sym} ({status})")

# Re-add TATASTEEL as OPEN (it was phantom-exited due to quote failure)
tatasteel_open = {
    "symbol": "NFO:TATASTEEL26FEB208CE",
    "underlying": "NSE:TATASTEEL",
    "quantity": 5500,
    "lots": 1,
    "avg_price": 5.09,
    "side": "BUY",
    "direction": "BUY",
    "option_type": "CE",
    "strike": 208.0,
    "expiry": "2026-02-24",
    "stop_loss": 3.56,
    "target": 7.635,
    "order_id": "OPTION_PAPER_718129",
    "timestamp": "2026-02-10T10:11:45.425578",
    "status": "OPEN",
    "is_option": True,
    "total_premium": 27995.0,
    "max_loss": 27995.0,
    "breakeven": 213.09,
    "delta": 0.529,
    "theta": -0.196,
    "iv": 0.302,
    "rationale": "VWAP_TREND"
}
new_active.append(tatasteel_open)
print(f"  + RE-ADDED: NFO:TATASTEEL26FEB208CE as OPEN (was phantom exit)")

data['active_trades'] = new_active
data['last_updated'] = datetime.now().isoformat()

with open('active_trades.json', 'w') as f:
    json.dump(data, f, indent=2)

with open('trade_history.json', 'w') as f:
    json.dump(history, f, indent=2)

open_count = sum(1 for t in new_active if t.get('status') == 'OPEN')
print(f"\nResult: {len(new_active)} active ({open_count} OPEN), {len(history)} in history")
print(f"Realized P&L: {data['realized_pnl']}")
print("DONE")
