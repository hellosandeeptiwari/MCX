import json, os

# Check trade_history.json (source of truth)
f = 'trade_history.json'
if os.path.exists(f):
    d = json.load(open(f))
    today = [t for t in d if '2026-02-10' in t.get('closed_at', '')]
    total = sum(t.get('pnl', 0) for t in today)
    print(f"Today closed trades: {len(today)}, Total realized PnL: {total:.2f}")
    for i, t in enumerate(today):
        sym = t.get('symbol', '?')
        status = t.get('result', '?')
        pnl = t.get('pnl', 0)
        print(f"  {i+1:2d}. {sym:40s} {status:20s} pnl={pnl:+10.2f}")
else:
    print("No trade_history.json found")

# Check active_trades.json
print()
f2 = 'active_trades.json'
if os.path.exists(f2):
    d2 = json.load(open(f2))
    print(f"active_trades.json realized_pnl: {d2.get('realized_pnl', 0):.2f}")
    
# Check risk_state.json
print()
f3 = 'risk_state.json'
if os.path.exists(f3):
    d3 = json.load(open(f3))
    print(f"risk_state.json daily_pnl: {d3.get('daily_pnl', 0):.2f}")
    print(f"  trades: {d3.get('trades_today',0)}, W:{d3.get('wins_today',0)} L:{d3.get('losses_today',0)}")
