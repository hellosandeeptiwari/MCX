import json

# Load trade history (closed trades)
history = json.load(open('trade_history.json'))
# Load active trades
active = json.load(open('active_trades.json'))

# Filter to option trades only (today + recent)
option_trades = [t for t in history if 'CE' in t.get('symbol','') or 'PE' in t.get('symbol','')]
today_trades = [t for t in history if '2026-02-10' in t.get('timestamp', '')]

print("=" * 100)
print("TODAY'S CLOSED OPTION TRADES")
print("=" * 100)
print(f"{'Symbol':40s} | {'Entry':>8s} | {'Exit':>8s} | {'%Move':>7s} | {'PnL':>9s} | Exit Reason")
print("-" * 100)

for t in today_trades:
    sym = t.get('symbol', '?')
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    pnl = t.get('pnl', 0)
    status = t.get('status', t.get('result', '?'))
    if entry > 0 and exit_p > 0:
        pct = (exit_p - entry) / entry * 100
    else:
        pct = 0
    print(f"{sym:40s} | {entry:8.2f} | {exit_p:8.2f} | {pct:+6.1f}% | {pnl:+9.0f} | {status}")

print()
print("CURRENTLY OPEN:")
for t in active.get('active_trades', []):
    sym = t.get('symbol', '?')
    entry = t.get('avg_price', 0)
    sl = t.get('stop_loss', 0)
    tgt = t.get('target', 0)
    sl_pct = (sl - entry) / entry * 100 if entry > 0 else 0
    tgt_pct = (tgt - entry) / entry * 100 if entry > 0 else 0
    print(f"  {sym:40s} | Entry: {entry:.2f} | SL: {sl:.2f} ({sl_pct:+.0f}%) | Target: {tgt:.2f} ({tgt_pct:+.0f}%)")

print()
print("EXIT REASON BREAKDOWN (today):")
reasons = {}
for t in today_trades:
    r = t.get('status', t.get('result', 'UNKNOWN'))
    reasons[r] = reasons.get(r, {'count': 0, 'pnl': 0})
    reasons[r]['count'] += 1
    reasons[r]['pnl'] += t.get('pnl', 0)
for r, data in sorted(reasons.items(), key=lambda x: -x[1]['count']):
    print(f"  {r:30s}: {data['count']} trades, PnL: {data['pnl']:+,.0f}")

print()
total_realized = sum(t.get('pnl', 0) for t in today_trades)
winners = [t for t in today_trades if t.get('pnl', 0) > 0]
losers = [t for t in today_trades if t.get('pnl', 0) <= 0]
print(f"Today: {len(today_trades)} closed | Winners: {len(winners)} | Losers: {len(losers)}")
print(f"Total realized: {total_realized:+,.0f}")
if winners:
    print(f"Avg winner: {sum(t.get('pnl',0) for t in winners)/len(winners):+,.0f}")
if losers:
    print(f"Avg loser: {sum(t.get('pnl',0) for t in losers)/len(losers):+,.0f}")
