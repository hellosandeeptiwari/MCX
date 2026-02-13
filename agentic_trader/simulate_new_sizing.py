"""Simulate today's 9 trades under OLD vs NEW sizing to show the impact"""

trades = [
    {'sym': 'IDEA (equity)',     'qty': 8196, 'entry': 11.68, 'exit': 11.69, 'pnl': 82,    'score': 55, 'lot_size': 8196},
    {'sym': 'ASHOKLEY 205PE',    'qty': 5000, 'entry': 6.00,  'exit': 6.62,  'pnl': 3100,  'score': 79, 'lot_size': 5000},
    {'sym': 'TITAN 4220PE',      'qty': 175,  'entry': 78.65, 'exit': 65.75, 'pnl': -2258, 'score': 58, 'lot_size': 175},
    {'sym': 'IREDA 126PE',       'qty': 3450, 'entry': 4.44,  'exit': 4.53,  'pnl': 310,   'score': 61, 'lot_size': 3450},
    {'sym': 'BANDHANBNK 170CE',  'qty': 3600, 'entry': 4.15,  'exit': 3.69,  'pnl': -1656, 'score': 63, 'lot_size': 3600},
    {'sym': 'BHEL 260PE',        'qty': 2625, 'entry': 7.10,  'exit': 6.40,  'pnl': -1837, 'score': 62, 'lot_size': 2625},
    {'sym': 'IOC 180CE',         'qty': 4875, 'entry': 3.58,  'exit': 4.07,  'pnl': 2389,  'score': 73, 'lot_size': 4875},
    {'sym': 'SBIN 1185CE',       'qty': 750,  'entry': 19.00, 'exit': 17.80, 'pnl': -900,  'score': 66, 'lot_size': 750},
    {'sym': 'ASHOKLEY 207.5PE',  'qty': 5000, 'entry': 6.70,  'exit': 6.41,  'pnl': -1450, 'score': 64, 'lot_size': 5000},
]

CAPITAL = 500000
BLOCK = 60
STD = 65
PREMIUM = 70

print("=" * 85)
print("SIMULATION: OLD vs NEW SIZING (Feb 11 trades)")
print("=" * 85)

# NEW THRESHOLDS + NEW SIZING
# Trades below BLOCK (60) are BLOCKED
# score_tier determines risk budget and lots

def calc_new_lots(t):
    """Calculate lots under new tiered system"""
    score = t['score']
    entry = t['entry']
    lot_size = t['lot_size']
    premium_per_lot = entry * lot_size
    
    if score < BLOCK:
        return 0, "BLOCKED", "blocked"
    
    if score >= PREMIUM:
        tier = "premium"
        risk_pct = 0.05
        min_lots = 2
        cap_pct = 0.25
    elif score >= STD:
        tier = "standard"
        risk_pct = 0.035
        min_lots = 1
        cap_pct = 0.20
    else:
        tier = "base"
        risk_pct = 0.02
        min_lots = 1
        cap_pct = 0.15
    
    # Risk-based sizing (30% SL = actual max loss)
    max_loss = CAPITAL * risk_pct
    sl_loss_per_lot = premium_per_lot * 0.30
    lots = max(1, int(max_loss / sl_loss_per_lot))
    
    # Minimum for premium
    if tier == "premium" and lots < min_lots and premium_per_lot * min_lots < CAPITAL * cap_pct:
        lots = min_lots
    
    # Cap
    if lots * premium_per_lot > CAPITAL * cap_pct:
        lots = max(1, int(CAPITAL * cap_pct / premium_per_lot))
    
    return lots, tier, "active"

print(f"\n{'Trade':<22} {'Score':>5} {'Old':>6} {'New':>6} {'Old PnL':>10} {'New PnL':>10} {'Tier':<10}")
print("-" * 80)

old_total = 0
new_total = 0
new_trade_count = 0

for t in trades:
    old_lots = 1
    old_pnl = t['pnl']
    old_total += old_pnl
    
    new_lots, tier, status = calc_new_lots(t)
    
    if status == "blocked":
        new_pnl = 0
        new_lots_str = "BLOCK"
    else:
        # PnL scales linearly with lots
        move_per_share = t['exit'] - t['entry']
        new_pnl = int(move_per_share * t['lot_size'] * new_lots)
        new_lots_str = f"{new_lots}L"
        new_trade_count += 1
    
    new_total += new_pnl
    
    marker = ""
    if new_pnl > old_pnl + 500:
        marker = " ⬆️"
    elif new_pnl < old_pnl - 500:
        marker = " ⬇️"
    elif status == "blocked":
        marker = " 🚫"
    
    print(f"{t['sym']:<22} {t['score']:>5} {old_lots:>5}L {new_lots_str:>6} {old_pnl:>+10,} {new_pnl:>+10,} {tier:<10}{marker}")

print(f"\n{'TOTALS':<22} {'':>5} {'9':>5}T {new_trade_count:>5}T {old_total:>+10,} {new_total:>+10,}")
print(f"\n{'='*85}")
print(f"OLD SYSTEM: 9 trades, all 1 lot each = Rs {old_total:+,} ({old_total/CAPITAL*100:+.2f}%)")
print(f"NEW SYSTEM: {new_trade_count} trades, tiered sizing  = Rs {new_total:+,} ({new_total/CAPITAL*100:+.2f}%)")
print(f"DIFFERENCE: Rs {new_total - old_total:+,}")

print(f"\n{'KEY CHANGES SUMMARY':=^85}")
print("""
1. POSITION SIZING: Risk per trade 2% → 5% (premium), 3.5% (standard), 2% (base)
   - Assumed max loss: 50% → 30% (matches actual SL) → 2-3x more lots
   - Premium tier: Minimum 2 lots guaranteed
   - Multiplier: int() → round() — 1.5x actually gives 2 lots now

2. CAPITAL CAP: 15% flat → 25%/20%/15% tiered
   - Premium: up to Rs 1,25,000 per trade (25% of Rs 5L)
   - Standard: up to Rs 1,00,000 per trade (20% of Rs 5L)

3. TARGET/SL RATIO (R:R):
   - Premium: 80% target / 28% SL = 2.86:1 R:R (was 1.67:1)
   - Standard: 60% target / 28% SL = 2.14:1 R:R
   - With trailing stop at 0.5R, winners can run even further

4. MAX POSITIONS: 6 → 4 (fewer, bigger, higher conviction only)

5. TIME STOP: 7 candles / 0.5R → 10 candles / 0.3R (trades get more breathing room)

6. MAX DAILY LOSS: 3% → 5% (Rs 15K → Rs 25K) — room for bigger positions
""")
