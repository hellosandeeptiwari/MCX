import json
today = [t for t in json.load(open('trade_history.json')) if '2026-02-10' in t.get('closed_at','')]
losers = [t for t in today if t['pnl'] < 0 and t['underlying'] != 'NSE:BSE']

print("NON-BSE LOSING TRADES — ENTRY STRATEGY ANALYSIS")
print("=" * 70)
for t in losers:
    sym = t['symbol']
    pnl = t['pnl']
    ts = t['timestamp'][11:16]
    rat = t.get('rationale', '?')
    iv = t.get('iv', 0) * 100
    delta = t.get('delta', 0)
    theta = t.get('theta', 0)
    premium = t.get('total_premium', 0)
    print(f"\n{sym}")
    print(f"  Time: {ts}  PnL: {pnl:+,.0f}  Exit: {t['result']}")
    print(f"  Rationale: {rat}")
    print(f"  IV: {iv:.1f}%  Delta: {delta:.2f}  Theta: {theta:.2f}/day")
    print(f"  Premium: {premium:,.0f}  Qty: {t['quantity']}")
    # Theta cost over hold time
    from datetime import datetime
    hold_hrs = (datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(t['timestamp'])).total_seconds() / 3600
    theta_cost = abs(theta) * t['quantity'] * (hold_hrs / 6.5)  # theta per trading day
    print(f"  Held: {hold_hrs:.1f}hrs  Theta cost: ~{theta_cost:,.0f}")
    # Was entry near day high (chasing)?
    entry_pct_from_breakeven = abs(t['avg_price'] - t['exit_price']) / t['avg_price'] * 100
    print(f"  Entry→Exit move: {entry_pct_from_breakeven:.1f}%")
