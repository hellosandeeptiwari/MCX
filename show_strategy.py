"""
MCX OPTIONS STRATEGY RECOMMENDATIONS
=====================================
"""
import json
import os

os.system('cls' if os.name == 'nt' else 'clear')

# Load live data
with open('mcx_options_live.json', 'r') as f:
    data = json.load(f)

spot = data['spot']
chain = {float(k): v for k, v in data['chain'].items()}
LOT = 625
BUDGET = 300000

# Find ATM
strikes = sorted(chain.keys())
atm = min(strikes, key=lambda x: abs(x - spot))
atm_ce = chain[atm]['CE']['ltp']
atm_pe = chain[atm]['PE']['ltp']

print()
print("*" * 55)
print("*       MCX OPTIONS STRATEGIES - Feb 6, 2026        *")
print("*" * 55)
print()
print(f"  LIVE DATA:")
print(f"  ----------")
print(f"  Spot Price  : Rs.{spot:,.2f}")
print(f"  ATM Strike  : {int(atm)}")
print(f"  Expiry      : {data['expiry']}")
print()
print("*" * 55)
print("*         BEST BUY STRATEGIES (Rs.3L Budget)        *")
print("*" * 55)
print()

# Strategy 1
straddle = atm_ce + atm_pe
straddle_cost = straddle * LOT
lots = int(BUDGET / straddle_cost)
print(f"  1. LONG STRADDLE (Neutral - Big Move)")
print(f"     Buy {int(atm)} CE @ Rs.{atm_ce:.2f}")
print(f"     Buy {int(atm)} PE @ Rs.{atm_pe:.2f}")
print(f"     Cost/Lot : Rs.{straddle_cost:,.0f}")
print(f"     Lots     : {lots}")
print(f"     Total    : Rs.{straddle_cost * lots:,.0f}")
print()

# Strategy 2
ce_cost = atm_ce * LOT
lots_ce = int(BUDGET / ce_cost)
print(f"  2. BUY {int(atm)} CE (Bullish)")
print(f"     Premium  : Rs.{atm_ce:.2f}")
print(f"     Cost/Lot : Rs.{ce_cost:,.0f}")
print(f"     Lots     : {lots_ce}")
print(f"     Total    : Rs.{ce_cost * lots_ce:,.0f}")
print()

# Strategy 3
pe_cost = atm_pe * LOT
lots_pe = int(BUDGET / pe_cost)
print(f"  3. BUY {int(atm)} PE (Bearish) << BEST VALUE >>")
print(f"     Premium  : Rs.{atm_pe:.2f}")
print(f"     Cost/Lot : Rs.{pe_cost:,.0f}")
print(f"     Lots     : {lots_pe}")
print(f"     Total    : Rs.{pe_cost * lots_pe:,.0f}")
print()

print("*" * 55)
print("*                  RECOMMENDATION                   *")
print("*" * 55)
print()
print(f"  >> Buy {int(atm)} PE @ Rs.{atm_pe:.2f} (4 lots)")
print(f"  >> Total Investment: Rs.{pe_cost * lots_pe:,.0f}")
print(f"  >> If MCX drops 10%: Profit Rs.1,40,000+")
print()
print("*" * 55)
print()
input("Press Enter to exit...")
