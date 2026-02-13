"""Analyze today's capital utilization and position sizing issues"""

trades = [
    {'sym': 'IDEA (equity)',     'qty': 8196, 'entry': 11.68, 'exit': 11.69, 'pnl': 82},
    {'sym': 'ASHOKLEY 205PE',    'qty': 5000, 'entry': 6.00,  'exit': 6.62,  'pnl': 3100},
    {'sym': 'TITAN 4220PE',      'qty': 175,  'entry': 78.65, 'exit': 65.75, 'pnl': -2258},
    {'sym': 'IREDA 126PE',       'qty': 3450, 'entry': 4.44,  'exit': 4.53,  'pnl': 310},
    {'sym': 'BANDHANBNK 170CE',  'qty': 3600, 'entry': 4.15,  'exit': 3.69,  'pnl': -1656},
    {'sym': 'BHEL 260PE',        'qty': 2625, 'entry': 7.10,  'exit': 6.40,  'pnl': -1837},
    {'sym': 'IOC 180CE',         'qty': 4875, 'entry': 3.58,  'exit': 4.07,  'pnl': 2389},
    {'sym': 'SBIN 1185CE',       'qty': 750,  'entry': 19.00, 'exit': 17.80, 'pnl': -900},
    {'sym': 'ASHOKLEY 207.5PE',  'qty': 5000, 'entry': 6.70,  'exit': 6.41,  'pnl': -1450},
]

total_cap = 500000

print("=" * 75)
print("CAPITAL UTILIZATION ANALYSIS - Feb 11")
print("=" * 75)

print(f"\n{'Trade':<22} {'Premium':>10} {'%Cap':>6} {'PnL':>8} {'Return':>8}")
print("-" * 60)

for t in trades:
    premium = t['qty'] * t['entry']
    pct = premium / total_cap * 100
    ret = t['pnl'] / premium * 100 if premium > 0 else 0
    print(f"{t['sym']:<22} Rs{premium:>8,.0f} {pct:>5.1f}% {t['pnl']:>+8,} {ret:>+6.1f}%")

premiums = [t['qty'] * t['entry'] for t in trades]
wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] <= 0]

print(f"\n{'PROBLEM DIAGNOSIS':=^60}")
print(f"\n1. TINY POSITION SIZES:")
print(f"   Avg premium/trade:  Rs {sum(premiums)/len(premiums):,.0f}")
print(f"   Avg % of capital:   {sum(premiums)/len(premiums)/total_cap*100:.1f}%")
print(f"   Total deployed:     Rs {sum(premiums):,.0f} ({sum(premiums)/total_cap*100:.0f}%)")
print(f"   => Each trade uses ~3-6% of capital = PATHETICALLY SMALL")

print(f"\n2. POOR WIN/LOSS RATIO:")
avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0
print(f"   Avg win:  Rs {avg_win:,.0f}")
print(f"   Avg loss: Rs {avg_loss:,.0f}")
print(f"   Ratio:    {avg_win/avg_loss:.2f}x  (NEED 2.0x+)")

print(f"\n3. TOO MANY TRADES (SCATTERED CAPITAL):")
print(f"   9 trades x 1 lot = scattered across 9 positions")
print(f"   Better: 3 trades x 3 lots = concentrated on best setups")

print(f"\n{'WHAT NEEDS TO CHANGE':=^60}")
print("""
CURRENT vs NEEDED:
  risk_per_trade:     2.5% (Rs 12,500) -> 5% (Rs 25,000)
  min_lots (premium): 1 lot            -> 2-3 lots
  max_positions:      unlimited        -> 3-4 max
  target %:           50%              -> 80-100% (runners)
  SL %:               30%              -> 25% (tighter)
  time_stop:          7 candles        -> 12+ candles
  
EXPECTED RESULT:
  3 trades x 3 lots x Rs 50K premium = Rs 150K deployed (30% of capital)
  If 2 win at 80%: 2 x Rs 40K = +Rs 80K
  If 1 loses at 25%: 1 x Rs 12.5K = -Rs 12.5K
  Net: +Rs 67.5K (but with 60% accuracy, more like +Rs 20-30K/day)
  
  Even with 50% win rate and 2:1 R:R:
  3 trades: 1.5 wins x Rs 30K - 1.5 losses x Rs 12.5K = +Rs 26,250
""")
