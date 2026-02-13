"""Analyze why debit spreads never triggered today"""
import re

with open('trade_decisions.log', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find chasing percentages (= intraday move from open)
chasing = re.findall(r'for NSE:(\w+).*?CHASING: ([\d.]+)% intraday move', content, re.DOTALL)

# Also find the indicator snapshots for move_pct
moves = {}
for sym, pct in chasing:
    pct_f = float(pct)
    if sym not in moves or pct_f > moves[sym]:
        moves[sym] = pct_f

print("=" * 70)
print("WHY DEBIT SPREADS NEVER TRIGGERED TODAY")
print("=" * 70)
print(f"\nCurrent DEBIT_SPREAD_CONFIG['min_move_pct'] = 2.5%")
print(f"Stocks need to move >2.5% intraday to qualify\n")

print(f"{'Symbol':<15} {'Max Move %':>10} {'Passes 2.5%?':>12} {'Passes 1.5%?':>12}")
print("-" * 55)

pass_25 = 0
pass_15 = 0
for sym, pct in sorted(moves.items(), key=lambda x: -x[1]):
    p25 = "YES" if pct >= 2.5 else "no"
    p15 = "YES" if pct >= 1.5 else "no"
    if pct >= 2.5: pass_25 += 1
    if pct >= 1.5: pass_15 += 1
    print(f"{sym:<15} {pct:>9.1f}% {p25:>12} {p15:>12}")

print(f"\nSummary: {pass_25}/{len(moves)} stocks pass 2.5%, {pass_15}/{len(moves)} pass 1.5%")

# What about all the non-chasing stocks? They moved < 1% 
# Find all scored symbols
all_scored = re.findall(r'for NSE:(\w+)', content)
all_unique = set(all_scored)
no_chasing = all_unique - set(moves.keys())
print(f"\nStocks with NO chasing penalty (moved < threshold): {len(no_chasing)}")
for s in sorted(no_chasing):
    print(f"  {s}: < 1% move (no chasing penalty triggered)")

print(f"\n{'DIAGNOSIS':=^70}")
print("""
PROBLEM: min_move_pct=2.5% is too restrictive for Indian equities.
- Most FnO stocks move 0.5-2% intraday on average
- Only extreme days see 2.5%+ moves
- Result: ZERO debit spreads all day

ALSO: The cascade tries debit spread ONLY after credit spread fails.
This means debit spreads are purely a fallback, never proactively sought.

FIX NEEDED:
1. Lower min_move_pct to 1.5% (allows more qualifying stocks)
2. Add smart candle filters instead of arbitrary move threshold
3. Proactively scan for debit spread opportunities (not just fallback)
4. Use follow-through + ADX + ORB as quality filters (like naked buys)
5. Better R:R: target 80% of max-profit, SL 30% (not 50/40)
""")
