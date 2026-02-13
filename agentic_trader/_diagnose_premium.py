"""Diagnose why 65-74 Premium band loses money despite 73% win rate"""
import json, statistics

trades = json.load(open('trade_history.json'))

# Filter to scored trades in Premium band (65-74)
premium = []
for t in trades:
    score = t.get('entry_score', 0) or 0
    if 65 <= score <= 74:
        premium.append(t)

print("=" * 80)
print("  PREMIUM BAND (65-74) DEEP DIVE")
print(f"  {len(premium)} trades, Win Rate: {sum(1 for t in premium if (t.get('pnl',0) or 0) > 0)/len(premium)*100:.1f}%")
print(f"  Total PnL: {sum(t.get('pnl',0) or 0 for t in premium):+,.0f}")
print("=" * 80)

# Show every trade
print(f"\n{'Symbol':<28s} {'Score':>5s} {'PnL':>10s} {'R':>6s} {'Exit':>20s} {'Type':>14s} {'Candles':>7s}")
print("-" * 95)
for t in sorted(premium, key=lambda x: x.get('pnl', 0)):
    ed = t.get('exit_detail', {}) or {}
    sym = (t.get('symbol', '') or t.get('underlying', ''))[:27]
    score = t.get('entry_score', 0) or 0
    pnl = t.get('pnl', 0) or 0
    r = ed.get('r_multiple_achieved', 0) or 0
    exit_type = ed.get('exit_type', t.get('result', '?'))
    stype = (t.get('strategy_type', 'EQ') or 'EQ')[:14]
    candles = ed.get('candles_held', 0) or 0
    marker = "<<<< BIG LOSER" if pnl < -2000 else ""
    print(f"{sym:<28s} {score:>5.0f} {pnl:>+10,.0f} {r:>+6.2f} {exit_type:>20s} {stype:>14s} {candles:>7d}  {marker}")

# Stats on winners vs losers
winners = [t for t in premium if (t.get('pnl', 0) or 0) > 0]
losers = [t for t in premium if (t.get('pnl', 0) or 0) < 0]

print(f"\n--- WINNER ANALYSIS ({len(winners)} trades) ---")
if winners:
    wpnls = [t.get('pnl', 0) or 0 for t in winners]
    print(f"  Total: {sum(wpnls):+,.0f}")
    print(f"  Avg: {statistics.mean(wpnls):+,.0f}")
    print(f"  Median: {statistics.median(wpnls):+,.0f}")
    print(f"  Range: {min(wpnls):+,.0f} to {max(wpnls):+,.0f}")

print(f"\n--- LOSER ANALYSIS ({len(losers)} trades) ---")
if losers:
    lpnls = [t.get('pnl', 0) or 0 for t in losers]
    print(f"  Total: {sum(lpnls):+,.0f}")
    print(f"  Avg: {statistics.mean(lpnls):+,.0f}")
    print(f"  Median: {statistics.median(lpnls):+,.0f}")
    print(f"  Range: {min(lpnls):+,.0f} to {max(lpnls):+,.0f}")
    print(f"\n  PAYOFF RATIO: {statistics.mean(wpnls)/abs(statistics.mean(lpnls)):.2f}")
    print(f"  → Need >{1/(len(winners)/len(premium)):.2f} to break even, got {statistics.mean(wpnls)/abs(statistics.mean(lpnls)):.2f}")

# Compare with other bands
print("\n" + "=" * 80)
print("  ALL BANDS COMPARISON")
print("=" * 80)
bands = [
    ("Sub-45", 0, 44),
    ("Base (45-54)", 45, 54),
    ("Standard (55-64)", 55, 64),
    ("Premium (65-74)", 65, 74),
    ("Elite (75+)", 75, 100),
    ("Unscored (0)", 0, 0),
]

for label, lo, hi in bands:
    if label == "Unscored (0)":
        band = [t for t in trades if (t.get('entry_score', 0) or 0) == 0]
    else:
        band = [t for t in trades if lo <= (t.get('entry_score', 0) or 0) <= hi]
    if not band:
        continue
    
    bpnl = sum(t.get('pnl', 0) or 0 for t in band)
    bwins = sum(1 for t in band if (t.get('pnl', 0) or 0) > 0)
    bwr = bwins / len(band) * 100 if band else 0
    
    wpnls = [t.get('pnl', 0) or 0 for t in band if (t.get('pnl', 0) or 0) > 0]
    lpnls = [t.get('pnl', 0) or 0 for t in band if (t.get('pnl', 0) or 0) < 0]
    avg_w = statistics.mean(wpnls) if wpnls else 0
    avg_l = statistics.mean(lpnls) if lpnls else 0
    payoff = avg_w / abs(avg_l) if avg_l != 0 else float('inf')
    
    print(f"  {label:<20s}  N={len(band):>3d}  WR={bwr:>5.1f}%  PnL={bpnl:>+10,.0f}  AvgW={avg_w:>+8,.0f}  AvgL={avg_l:>+8,.0f}  Payoff={payoff:.2f}")


# Check position sizes — are Premium losers taking bigger positions?
print("\n" + "=" * 80)
print("  POSITION SIZE ANALYSIS FOR PREMIUM BAND")
print("=" * 80)
for t in sorted(premium, key=lambda x: x.get('pnl', 0)):
    ed = t.get('exit_detail', {}) or {}
    sym = (t.get('symbol', '') or t.get('underlying', ''))[:27]
    pnl = t.get('pnl', 0) or 0
    qty = t.get('quantity', 0) or 0
    avg_price = t.get('avg_price', 0) or 0
    exit_price = t.get('exit_price', 0) or 0
    position_value = qty * avg_price
    risk = t.get('risk_amount', 0) or t.get('position_risk', 0) or 0
    print(f"  {sym:<27s} PnL={pnl:>+8,.0f}  Qty={qty:>5d}  Entry={avg_price:>8.1f}  PositionVal={position_value:>10,.0f}  Risk={risk:>8,.0f}")
