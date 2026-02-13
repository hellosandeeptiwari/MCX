"""Simulate the 5 new candle-data-driven gates on today's 37 scored trades"""

# All 37 scored entries from trade_decisions.log with candle data
# Format: (sym, score, direction, time, ema_regime, vwap_slope, volume, orb_signal, orb_strength, orb_hold, ft_candles, adx, rsi, range_exp, pnl, result)
trades = [
    # Morning entries
    ("INFY",       63, "SELL", "09:33", "EXPANDING", "FLAT",    "NORMAL",    "BREAKOUT_DOWN",   0,  1, 0, 22.6, 20.9, 0,    0,     "??"),
    ("ICICIBANK",  68, "SELL", "09:43", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_DOWN",   0,  2, 2, 25.9, 63.7, 0,    0,     "??"),
    ("LT",         91, "SELL", "09:44", "EXPANDING", "FALLING", "HIGH",      "BREAKOUT_DOWN",   0,  3, 3, 40.3, 72.6, 0,    0,     "??"),
    ("MARUTI",     82, "BUY",  "09:44", "EXPANDING", "RISING",  "HIGH",      "BREAKOUT_UP",     0,  2, 3, 59.0, 42.3, 0,    0,     "??"),
    ("TITAN",      74, "SELL", "09:44", "EXPANDING", "FALLING", "EXPLOSIVE", "INSIDE_ORB",      0,  0, 0, 29.1, 73.8, 0,    -2258, "LOSS"),
    ("IOC",        79, "BUY",  "09:49", "EXPANDING", "RISING",  "HIGH",      "BREAKOUT_UP",     0,  1, 3, 31.7, 79.3, 0,    2389,  "WIN"),
    ("TCS",        76, "SELL", "09:58", "EXPANDING", "FALLING", "NORMAL",    "BREAKOUT_DOWN",   0,  6, 5, 10.4, 34.8, 0,    0,     "??"),
    ("ASHOKLEY",   94, "BUY",  "09:58", "EXPANDING", "RISING",  "EXPLOSIVE", "BREAKOUT_UP",     0,  2, 4, 42.1, 78.3, 0,    3100,  "WIN"),
    ("IREDA",      98, "SELL", "09:58", "EXPANDING", "FALLING", "EXPLOSIVE", "BREAKOUT_DOWN",   0,  6, 4, 42.8, 44.1, 0,    310,   "WIN"),
    # Mid-morning
    ("INFY",       72, "SELL", "10:13", "EXPANDING", "FLAT",    "NORMAL",    "BREAKOUT_DOWN",   0,  9, 7, 22.6, 20.3, 0,    0,     "??"),
    ("SBIN",       85, "BUY",  "10:47", "EXPANDING", "RISING",  "HIGH",      "BREAKOUT_UP",     0,  6, 0, 27.6, 70.1, 0,    -900,  "LOSS"),
    ("BAJFINANCE", 72, "BUY",  "10:47", "EXPANDING", "RISING",  "LOW",       "BREAKOUT_UP",     0,  5, 1, 29.9, 58.1, 0,    0,     "??"),
    ("INFY",       67, "SELL", "10:47", "EXPANDING", "FLAT",    "NORMAL",    "BREAKOUT_DOWN",   0, 16, 7, 22.6, 20.4, 0,    0,     "??"),
    ("BAJFINANCE", 60, "BUY",  "10:53", "NORMAL",   "RISING",  "LOW",       "BREAKOUT_UP",     0,  6, 1, 29.9, 57.8, 0,    0,     "??"),
    ("INFY",       64, "SELL", "10:53", "EXPANDING", "FLAT",    "NORMAL",    "BREAKOUT_DOWN",   0, 17, 7, 22.6, 20.4, 0,    0,     "??"),
    # Late morning
    ("TMPV",       84, "BUY",  "11:08", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_UP",     0,  4, 3, 27.4, 74.9, 0,    0,     "??"),
    ("INFY",       65, "SELL", "11:24", "EXPANDING", "FLAT",    "NORMAL",    "BREAKOUT_DOWN",   0, 23, 7, 22.6, 20.4, 0,    0,     "??"),
    ("IOC",        58, "BUY",  "11:31", "NORMAL",   "FLAT",    "NORMAL",    "BREAKOUT_UP",     0, 22, 5, 31.8, 79.5, 0,    2389,  "WIN"),
    ("SBIN",       69, "BUY",  "11:51", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_UP",     0, 19, 0, 27.7, 70.5, 0,    -900,  "LOSS"),
    ("TATASTEEL",  78, "SELL", "11:52", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_DOWN",   0, 22, 6, 30.5, 65.3, 0,    0,     "??"),
    ("IOC",        56, "BUY",  "11:57", "NORMAL",   "FLAT",    "NORMAL",    "BREAKOUT_UP",     0, 27, 5, 31.8, 79.5, 0,    2389,  "WIN"),
    ("BANDHANBNK", 82, "BUY",  "11:57", "EXPANDING", "FLAT",    "EXPLOSIVE", "BREAKOUT_UP",     0, 19, 0, 34.2, 81.6, 0,    -1656, "LOSS"),
    # Afternoon
    ("MARUTI",     50, "BUY",  "12:28", "NORMAL",   "FLAT",    "NORMAL",    "BREAKOUT_UP",     0, 35, 4, 58.8, 43.4, 0,    0,     "??"),
    ("LT",         70, "SELL", "12:40", "COMPRESSED","FLAT",    "HIGH",      "BREAKOUT_DOWN",   0, 38, 6, 40.3, 73.3, 0,    0,     "??"),
    ("ITC",        76, "SELL", "12:50", "EXPANDING", "FALLING", "HIGH",      "BREAKOUT_DOWN",   0, 41, 3, 56.6, 48.9, 0,    0,     "??"),
    ("FEDERALBNK", 72, "BUY",  "12:55", "EXPANDING", "FLAT",    "EXPLOSIVE", "BREAKOUT_UP",     0, 38, 0, 45.1, 61.5, 0,    0,     "??"),
    ("SBIN",       69, "BUY",  "13:01", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_UP",     0, 33, 0, 27.7, 71.1, 0,    -900,  "LOSS"),
    ("IOC",        56, "BUY",  "13:02", "NORMAL",   "FLAT",    "NORMAL",    "BREAKOUT_UP",     0, 40, 5, 31.8, 79.3, 0,    2389,  "WIN"),
    ("BANDHANBNK", 72, "BUY",  "13:14", "EXPANDING", "FLAT",    "HIGH",      "BREAKOUT_UP",     0, 34, 0, 34.2, 81.7, 0,    -1656, "LOSS"),
    ("TITAN",      58, "SELL", "13:53", "EXPANDING", "FLAT",    "EXPLOSIVE", "BREAKOUT_DOWN",   0, 37, 2, 29.1, 66.2, 0,    -2258, "LOSS"),
    ("ASHOKLEY",   79, "SELL", "13:58", "EXPANDING", "FALLING", "EXPLOSIVE", "BREAKOUT_DOWN",   0,  1, 1, 42.2, 68.3, 0,    3100,  "WIN"),
    ("IREDA",      57, "SELL", "13:58", "NORMAL",   "FLAT",    "HIGH",      "BREAKOUT_DOWN",   0, 54, 8, 42.9, 43.0, 0,    310,   "WIN"),
    ("BANDHANBNK", 55, "BUY",  "14:05", "NORMAL",   "FLAT",    "HIGH",      "BREAKOUT_UP",     0, 45, 0, 34.2, 81.7, 0,    -1656, "LOSS"),
    ("BHEL",       70, "SELL", "14:09", "EXPANDING", "FLAT",    "EXPLOSIVE", "BREAKOUT_DOWN",   0, 21, 2, 36.3, 54.1, 0,    -1837, "LOSS"),
    ("IOC",        63, "BUY",  "14:09", "COMPRESSED","FLAT",    "NORMAL",    "BREAKOUT_UP",     0, 53, 5, 31.8, 79.6, 0,    2389,  "WIN"),
    ("SBIN",       65, "BUY",  "14:31", "EXPANDING", "RISING",  "HIGH",      "BREAKOUT_UP",     0, 51, 0, 27.9, 71.9, 0,    -900,  "LOSS"),
    ("ASHOKLEY",   86, "SELL", "14:31", "EXPANDING", "FALLING", "EXPLOSIVE", "BREAKOUT_DOWN",   0,  8, 1, 42.2, 70.2, 0,    3100,  "WIN"),
]

# New gate constants
BLOCK = 60
STD = 65
PREMIUM = 70
FT_MIN_PREMIUM = 2
FT_MIN_STANDARD = 1
ADX_MIN_PREMIUM = 30
ADX_MIN_STANDARD = 25
ORB_STR_OVEREXT = 100
RANGE_EXP_OVEREXT = 0.60
MAX_SYMBOL_LOSSES = 1

# Simulate
symbol_losses = {}
print(f"{'='*130}")
print(f"SIMULATION: 5 New Candle-Smart Gates on Today's 37 Scored Entries")  
print(f"{'='*130}")
print(f"\n{'#':>2} {'Sym':<12} {'Score':>5} {'FT':>3} {'ADX':>5} {'ORB%':>5} {'RE':>5} {'Old':>8} {'NEW':>8} {'Gates Triggered':<60}")
print("-" * 130)

old_yes = 0
old_no = 0
new_yes = 0
new_no = 0
old_pnl = 0
new_pnl = 0

for i, t in enumerate(trades):
    sym, score, direction, time_, ema, vwap, vol, orb, orb_str, orb_hold, ft, adx, rsi, range_exp, pnl, result = t
    
    # Old system: just score >= BLOCK (60)
    old_would_trade = score >= BLOCK
    
    # New system: apply all 5 candle gates
    new_score = score
    gates_hit = []
    new_would_trade = score >= BLOCK
    
    if new_would_trade:
        # Determine tier
        if score >= PREMIUM:
            tier = "premium"
        elif score >= STD:
            tier = "standard"
        else:
            tier = "base"
        
        # Gate 8: Follow-through
        if tier == "premium" and ft < FT_MIN_PREMIUM:
            new_score -= 8
            gates_hit.append(f"FT={ft}<{FT_MIN_PREMIUM}")
            if ft == 0:
                new_would_trade = False
                gates_hit.append("BLOCKED:zero-FT")
        elif tier == "standard" and ft < FT_MIN_STANDARD:
            new_score -= 5
            gates_hit.append(f"FT={ft}<{FT_MIN_STANDARD}")
        
        # Gate 9: ADX
        if tier == "premium" and adx < ADX_MIN_PREMIUM:
            new_score -= 5
            gates_hit.append(f"ADX={adx:.0f}<{ADX_MIN_PREMIUM}")
        elif tier == "standard" and adx < ADX_MIN_STANDARD:
            new_score -= 3
            gates_hit.append(f"ADX={adx:.0f}<{ADX_MIN_STANDARD}")
        elif adx >= 40:
            new_score += 3
            gates_hit.append(f"ADX+3({adx:.0f})")
        
        # Gate 10: ORB overextension
        if orb_str > ORB_STR_OVEREXT:
            new_score -= 8
            gates_hit.append(f"ORB={orb_str:.0f}>{ORB_STR_OVEREXT}")
        
        # Gate 11: Range expansion
        if range_exp > RANGE_EXP_OVEREXT:
            new_score -= 5
            gates_hit.append(f"RE={range_exp:.1f}>{RANGE_EXP_OVEREXT}")
        
        # Gate 12: Re-entry prevention
        losses = symbol_losses.get(sym, 0)
        if losses >= MAX_SYMBOL_LOSSES:
            new_would_trade = False
            gates_hit.append(f"BLOCKED:lost-{losses}x")
        
        # Re-check score after penalties
        if new_score < BLOCK:
            new_would_trade = False
            gates_hit.append(f"BLOCKED:score-dropped-{new_score:.0f}")
    
    # Track results
    if old_would_trade:
        old_yes += 1
        old_pnl += pnl
    else:
        old_no += 1
    
    if new_would_trade:
        new_yes += 1
        new_pnl += pnl
    else:
        new_no += 1
    
    # Record loss for re-entry tracking (simulate real-time)
    if result == "LOSS":
        symbol_losses[sym] = symbol_losses.get(sym, 0) + 1
    elif result == "WIN":
        symbol_losses[sym] = max(0, symbol_losses.get(sym, 0) - 1)
    
    # Print
    old_str = "TRADE" if old_would_trade else "skip"
    new_str = "TRADE" if new_would_trade else "SKIP"
    
    if old_would_trade and not new_would_trade and pnl < 0:
        marker = " ✅ SAVED"
    elif old_would_trade and not new_would_trade and pnl > 0:
        marker = " ❌ MISSED WIN"
    elif not old_would_trade and new_would_trade:
        marker = " 🆕 NEW ENTRY"
    elif old_would_trade and new_would_trade and pnl != 0:
        marker = f" {'🟢' if pnl > 0 else '🔴'}"
    else:
        marker = ""
    
    gates_str = ", ".join(gates_hit) if gates_hit else "-"
    pnl_str = f"{pnl:>+7,}" if pnl != 0 else "     -"
    
    print(f"{i+1:>2} {sym:<12} {score:>5} {ft:>3} {adx:>5.1f} {orb_str:>5.0f} {range_exp:>5.2f} {old_str:>8} {new_str:>8} {gates_str:<55} {pnl_str}{marker}")

# Summary
actual_trades = [t for t in trades if t[14] != 0]  # trades with known PnL
saved_losses = sum(1 for t in trades if t[0] in trades and t[14] < 0)

print(f"\n{'='*130}")
print(f"{'SUMMARY':=^80}")
print(f"\n  Old system: {old_yes} trades allowed, {old_no} blocked")
print(f"  New system: {new_yes} trades allowed, {new_no} blocked")
print(f"  Trades blocked by new gates: {old_yes - new_yes}")

# Calculate saved losses
old_losses_taken = sum(1 for t in trades if t[1] >= BLOCK and t[14] < 0)
new_losses_taken = 0
symbol_losses2 = {}
new_trades_pnl = []
for t in trades:
    sym, score, *_, pnl, result = t[0], t[1], *t[2:14], t[14], t[15]
    ft, adx, orb_str, range_exp = t[10], t[11], t[8], t[13]
    
    if score < BLOCK:
        continue
    
    # Apply new gates
    tier = "premium" if score >= PREMIUM else "standard" if score >= STD else "base"
    blocked = False
    if tier == "premium" and ft == 0:
        blocked = True
    if symbol_losses2.get(sym, 0) >= MAX_SYMBOL_LOSSES:
        blocked = True
    
    if not blocked and pnl != 0:
        new_trades_pnl.append(pnl)
    
    if result == "LOSS":
        if not blocked:
            new_losses_taken += 1
        symbol_losses2[sym] = symbol_losses2.get(sym, 0) + 1
    elif result == "WIN":
        symbol_losses2[sym] = max(0, symbol_losses2.get(sym, 0) - 1)

print(f"\n  Old system losses taken: {old_losses_taken}")
print(f"  New system losses taken: {new_losses_taken}")
print(f"  Losses prevented: {old_losses_taken - new_losses_taken}")

# The key question
known_old = [t for t in trades if t[1] >= BLOCK and t[14] != 0]
known_old_pnl = sum(t[14] for t in known_old)
known_new_pnl = sum(new_trades_pnl)
print(f"\n  Old system P&L (known trades): Rs {known_old_pnl:+,}")
print(f"  New system P&L (known trades): Rs {known_new_pnl:+,}")
print(f"  Improvement: Rs {known_new_pnl - known_old_pnl:+,}")
