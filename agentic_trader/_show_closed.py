import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = json.load(open('active_trades.json'))
trades = data.get('active_trades', []) if isinstance(data, dict) else data
closed = [t for t in trades if isinstance(t, dict) and t.get('status') == 'CLOSED']

if not closed:
    print("No squared-off trades yet today.")
else:
    print(f"{'Symbol':35s} {'Side':5s} {'Entry':>8s} {'Exit':>8s} {'P&L':>10s} {'Reason'}")
    print("-" * 90)
    total = 0
    for t in closed:
        sym = t.get('symbol', '?')
        side = t.get('side', '?')
        entry = t.get('avg_price', t.get('entry_price', 0))
        exit_p = t.get('exit_price', 0)
        pnl = t.get('pnl', 0)
        reason = t.get('exit_reason', '?')
        total += pnl
        print(f"{sym:35s} {side:5s} {entry:8.2f} {exit_p:8.2f} {pnl:+10.0f} {reason}")
    print("-" * 90)
    print(f"{'TOTAL':49s} {total:+10.0f}")
