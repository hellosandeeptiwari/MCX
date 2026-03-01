import json

data = json.load(open('active_trades.json'))
trades = data.get('active_trades', [])
closed = data.get('closed_trades', [])

print("=" * 70)
print("CLOSED TRADES (Realized P&L)")
print("=" * 70)
total_closed = 0
for t in closed:
    pnl = t.get('realized_pnl', 0)
    sym = t.get('symbol', '?')
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    qty = t.get('quantity', 0)
    reason = t.get('exit_reason', '?')
    total_closed += pnl
    print(f"  {sym}")
    print(f"    Entry: {entry:.2f} | Exit: {exit_p:.2f} | Qty: {qty}")
    print(f"    P&L: Rs {pnl:,.0f} | Reason: {reason}")
    print()

print(f"  TOTAL REALIZED: Rs {total_closed:,.0f}")
print()

print("=" * 70)
print(f"OPEN TRADES ({len(trades)} positions)")
print("=" * 70)
for t in trades:
    sym = t.get('symbol', '?')
    entry = t.get('avg_price', 0)
    qty = t.get('quantity', 0)
    prem = t.get('total_premium', 0)
    iv = t.get('iv', 0) or 0
    ivp = iv * 100 if iv < 1 else iv
    setup = t.get('setup_type', '?')
    score = t.get('entry_score', 0) or 0
    delta = t.get('delta', 0) or 0
    theta = t.get('theta', 0) or 0
    sl = t.get('stop_loss', 0)
    tgt = t.get('target', 0)
    print(f"  {sym}")
    print(f"    Entry: Rs{entry:.2f} | Qty: {qty} | Premium: Rs{prem:,.0f}")
    print(f"    IV: {ivp:.1f}% | Delta: {delta:.2f} | Theta: {theta:.2f}")
    print(f"    SL: Rs{sl:.2f} | Target: Rs{tgt:.2f}")
    print(f"    Setup: {setup} | Score: {score:.0f}")
    print()

print("=" * 70)
print("ALL JSON KEYS:", list(data.keys()))

# Check for daily_pnl in the file
if 'daily_pnl' in data:
    print("Daily PnL:", json.dumps(data['daily_pnl'], indent=2))
if 'daily_stats' in data:
    print("Daily Stats:", json.dumps(data['daily_stats'], indent=2))
