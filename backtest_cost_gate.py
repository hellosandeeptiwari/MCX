import json

# Today's hedged positions from API status
positions = [
    {"name": "ABCAPITAL CE", "buy_premium": 14.00, "sell_premium": 7.50, "net_debit": 6.50, "width": 15, "ltp": 5.70},
    {"name": "BHEL CE", "buy_premium": 11.05, "sell_premium": 6.60, "net_debit": 4.45, "width": 7.50, "ltp": 2.95},
    {"name": "RVNL PE", "buy_premium": 25.35, "sell_premium": 15.95, "net_debit": 9.40, "width": 15, "ltp": 9.70},
    {"name": "SHRIRAMFIN PE", "buy_premium": 36.30, "sell_premium": 22.30, "net_debit": 14.00, "width": 30, "ltp": 10.60},
    {"name": "BIOCON PE", "buy_premium": 13.60, "sell_premium": 7.45, "net_debit": 6.15, "width": 15, "ltp": 6.30},
    {"name": "ETERNAL PE", "buy_premium": 9.00, "sell_premium": 3.80, "net_debit": 5.20, "width": 15, "ltp": 4.49},
]

# Config thresholds
MAX_DTW_PCT = 55     # net_debit / width > 55% = REJECT
MIN_REMAINING_PCT = 30  # buy leg LTP / entry < 30% = REJECT  
MIN_RR = 0.50        # (width - net_debit) / net_debit < 0.50 = REJECT

print("=== COST-AWARE HEDGE GATE BACKTEST (Today's 6 Hedges) ===\n")
print(f"Config: max_debit_to_width={MAX_DTW_PCT}%, min_remaining={MIN_REMAINING_PCT}%, min_rr={MIN_RR}\n")

passed = 0
blocked = 0
for p in positions:
    dtw = (p["net_debit"] / p["width"]) * 100
    rr = (p["width"] - p["net_debit"]) / p["net_debit"] if p["net_debit"] > 0 else 999
    # We don't have exact LTP at hedge time, but can approximate
    
    gate1 = dtw <= MAX_DTW_PCT
    gate3 = rr >= MIN_RR
    
    all_pass = gate1 and gate3
    status = "✅ PASS" if all_pass else "❌ BLOCKED"
    if all_pass:
        passed += 1
    else:
        blocked += 1
    
    print(f"{p['name']:<20} DTW={dtw:.0f}% {'✅' if gate1 else '❌'}  R:R={rr:.2f} {'✅' if gate3 else '❌'}  → {status}")

print(f"\n--- RESULT: {passed} would PASS, {blocked} would be BLOCKED (forced to EXIT instead) ---")
print(f"\nBlocked positions would EXIT outright → saves hedge cost on bad spreads")
print(f"Passed positions have viable R:R → hedge is worth the cost")
