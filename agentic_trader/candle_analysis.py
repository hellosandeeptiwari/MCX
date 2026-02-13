"""Extract all executed trades from trade_decisions.log for candle analysis"""
import json, re

with open('trade_decisions.log', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Split by the delimiter
blocks = content.split('=' * 70)
executed = []
for b in blocks:
    if 'EXECUTED' in b and '2026-02-11' in b:
        executed.append(b.strip())

print(f'Found {len(executed)} executed trades today\n')

# Known outcomes from paper trading
outcomes = {
    'IDEA': {'pnl': 82, 'result': 'WIN', 'exit': 'MANUAL'},
    'ASHOKLEY': {'pnl': 3100, 'result': 'WIN', 'exit': 'TARGET'},
    'TITAN': {'pnl': -2258, 'result': 'LOSS', 'exit': 'TIME_STOP'},
    'IREDA': {'pnl': 310, 'result': 'WIN', 'exit': 'TIME_STOP'},
    'BANDHANBNK': {'pnl': -1656, 'result': 'LOSS', 'exit': 'TIME_STOP'},
    'BHEL': {'pnl': -1837, 'result': 'LOSS', 'exit': 'TIME_STOP'},
    'IOC': {'pnl': 2389, 'result': 'WIN', 'exit': 'TRAILING'},
    'SBIN': {'pnl': -900, 'result': 'LOSS', 'exit': 'TIME_STOP'},
    'ASHOKLEY26FEB207': {'pnl': -1450, 'result': 'LOSS', 'exit': 'TIME_STOP'},
}

# Parse each trade
trades = []
for block in executed:
    # Extract key data
    sym_match = re.search(r'for NSE:(\w+)', block)
    score_match = re.search(r'Score: (\d+)/100', block)
    dir_match = re.search(r'Direction: (\w+)', block)
    time_match = re.search(r'Time: ([\d\- :]+)', block)
    
    # Extract indicators
    ind_match = re.search(r"Indicators: ({.*?})", block, re.DOTALL)
    
    # Extract key signals
    signals = {
        'ema_regime': re.search(r"'ema_regime': '(\w+)'", block),
        'ema_spread': re.search(r"'ema_spread': np.float64\(([\d.]+)\)", block),
        'vwap_slope': re.search(r"'vwap_slope': '(\w+)'", block),
        'volume_regime': re.search(r"'volume_regime': '(\w+)'", block),
        'orb_signal': re.search(r"'orb_signal': '(\w+)'", block),
        'orb_strength': re.search(r"'orb_strength': np.float64\(([\d.]+)\)", block),
        'orb_hold': re.search(r"'orb_hold_candles': (\d+)", block),
        'adx': re.search(r"'adx': ([\d.]+)", block),
        'follow_through': re.search(r"'follow_through_candles': (\d+)", block),
        'range_expansion': re.search(r"'range_expansion_ratio': np.float64\(([\d.]+)\)", block),
        'vwap_steepening': re.search(r"'vwap_slope_steepening': (\w+)", block),
        'rsi': re.search(r"'rsi_14': np.float64\(([\d.]+)\)", block),
    }
    
    sym = sym_match.group(1) if sym_match else '??'
    score = int(score_match.group(1)) if score_match else 0
    direction = dir_match.group(1) if dir_match else '??'
    trade_time = time_match.group(1) if time_match else '??'
    
    # Get outcome
    outcome = None
    for key in outcomes:
        if key in sym:
            outcome = outcomes[key]
            break
    
    trade = {
        'sym': sym,
        'score': score,
        'direction': direction,
        'time': trade_time,
        'ema_regime': signals['ema_regime'].group(1) if signals['ema_regime'] else '??',
        'ema_spread': float(signals['ema_spread'].group(1)) if signals['ema_spread'] else 0,
        'vwap_slope': signals['vwap_slope'].group(1) if signals['vwap_slope'] else '??',
        'volume': signals['volume_regime'].group(1) if signals['volume_regime'] else '??',
        'orb': signals['orb_signal'].group(1) if signals['orb_signal'] else '??',
        'orb_strength': float(signals['orb_strength'].group(1)) if signals['orb_strength'] else 0,
        'orb_hold': int(signals['orb_hold'].group(1)) if signals['orb_hold'] else 0,
        'adx': float(signals['adx'].group(1)) if signals['adx'] else 0,
        'follow_through': int(signals['follow_through'].group(1)) if signals['follow_through'] else 0,
        'range_expansion': float(signals['range_expansion'].group(1)) if signals['range_expansion'] else 0,
        'vwap_steep': signals['vwap_steepening'].group(1) if signals['vwap_steepening'] else '??',
        'rsi': float(signals['rsi'].group(1)) if signals['rsi'] else 0,
        'pnl': outcome['pnl'] if outcome else 0,
        'result': outcome['result'] if outcome else '??',
        'exit_type': outcome['exit'] if outcome else '??',
    }
    
    # Check for specific scoring flags in the block
    trade['has_chasing'] = 'CHASING' in block
    trade['chasing_pct'] = 0
    chase_match = re.search(r'CHASING: ([\d.]+)% intraday move', block)
    if chase_match:
        trade['chasing_pct'] = float(chase_match.group(1))
    
    trade['has_day_range_warning'] = 'day range' in block.lower()
    day_range_match = re.search(r'(\d+)% of day range', block)
    trade['day_range_pct'] = int(day_range_match.group(1)) if day_range_match else 0
    
    trade['counter_trend'] = 'COUNTER-TREND' in block
    trade['vwap_aligned'] = 'VWAP aligned' in block
    trade['ema_expanding'] = 'EXPANDING' in trade['ema_regime']
    
    trades.append(trade)

# Print analysis
print(f"{'='*120}")
print(f"{'Sym':<14} {'Score':>5} {'Dir':<5} {'Time':<18} {'EMA':<11} {'VWAP':<8} {'Vol':<10} {'ORB':<14} {'Hold':>4} {'FT':>3} {'ADX':>5} {'RSI':>5} {'PnL':>8} {'Result':<5} {'Exit':<10}")
print(f"{'-'*120}")

wins = [t for t in trades if t['result'] == 'WIN']
losses = [t for t in trades if t['result'] == 'LOSS']

for t in trades:
    marker = '+' if t['result'] == 'WIN' else '-'
    print(f"{marker} {t['sym']:<12} {t['score']:>5} {t['direction']:<5} {t['time']:<18} {t['ema_regime']:<11} {t['vwap_slope']:<8} {t['volume']:<10} {t['orb']:<14} {t['orb_hold']:>4} {t['follow_through']:>3} {t['adx']:>5.1f} {t['rsi']:>5.1f} {t['pnl']:>+8,} {t['result']:<5} {t['exit_type']:<10}")

print(f"\n{'='*120}")
print(f"\nWINNERS vs LOSERS CANDLE PATTERN COMPARISON:")
print(f"{'='*80}")

def avg(lst, key):
    vals = [t[key] for t in lst if isinstance(t[key], (int, float))]
    return sum(vals) / len(vals) if vals else 0

print(f"\n{'Metric':<30} {'Winners':>12} {'Losers':>12} {'Edge':>12}")
print(f"{'-'*70}")
print(f"{'Avg Score':<30} {avg(wins, 'score'):>12.1f} {avg(losses, 'score'):>12.1f} {avg(wins,'score')-avg(losses,'score'):>+12.1f}")
print(f"{'Avg ADX':<30} {avg(wins, 'adx'):>12.1f} {avg(losses, 'adx'):>12.1f} {avg(wins,'adx')-avg(losses,'adx'):>+12.1f}")
print(f"{'Avg RSI':<30} {avg(wins, 'rsi'):>12.1f} {avg(losses, 'rsi'):>12.1f} {avg(wins,'rsi')-avg(losses,'rsi'):>+12.1f}")
print(f"{'Avg EMA Spread':<30} {avg(wins, 'ema_spread'):>12.2f} {avg(losses, 'ema_spread'):>12.2f} {avg(wins,'ema_spread')-avg(losses,'ema_spread'):>+12.2f}")
print(f"{'Avg ORB Strength':<30} {avg(wins, 'orb_strength'):>12.1f} {avg(losses, 'orb_strength'):>12.1f} {avg(wins,'orb_strength')-avg(losses,'orb_strength'):>+12.1f}")
print(f"{'Avg ORB Hold Candles':<30} {avg(wins, 'orb_hold'):>12.1f} {avg(losses, 'orb_hold'):>12.1f} {avg(wins,'orb_hold')-avg(losses,'orb_hold'):>+12.1f}")
print(f"{'Avg Follow-Through':<30} {avg(wins, 'follow_through'):>12.1f} {avg(losses, 'follow_through'):>12.1f} {avg(wins,'follow_through')-avg(losses,'follow_through'):>+12.1f}")
print(f"{'Avg Range Expansion':<30} {avg(wins, 'range_expansion'):>12.2f} {avg(losses, 'range_expansion'):>12.2f} {avg(wins,'range_expansion')-avg(losses,'range_expansion'):>+12.2f}")
print(f"{'Avg Chase %':<30} {avg(wins, 'chasing_pct'):>12.1f} {avg(losses, 'chasing_pct'):>12.1f} {avg(wins,'chasing_pct')-avg(losses,'chasing_pct'):>+12.1f}")

# Categorical
print(f"\n{'Categorical Patterns':<30} {'Winners':>12} {'Losers':>12}")
print(f"{'-'*55}")

def pct(lst, key, val):
    count = sum(1 for t in lst if t.get(key) == val)
    return f"{count}/{len(lst)} ({count/len(lst)*100:.0f}%)"

print(f"{'EMA EXPANDING':<30} {pct(wins, 'ema_regime', 'EXPANDING'):>12} {pct(losses, 'ema_regime', 'EXPANDING'):>12}")
print(f"{'EMA COMPRESSED':<30} {pct(wins, 'ema_regime', 'COMPRESSED'):>12} {pct(losses, 'ema_regime', 'COMPRESSED'):>12}")
print(f"{'VWAP Aligned':<30} {pct(wins, 'vwap_aligned', True):>12} {pct(losses, 'vwap_aligned', True):>12}")
print(f"{'EXPLOSIVE Volume':<30} {pct(wins, 'volume', 'EXPLOSIVE'):>12} {pct(losses, 'volume', 'EXPLOSIVE'):>12}")
print(f"{'ORB BREAKOUT':<30} {sum(1 for t in wins if 'BREAKOUT' in t['orb']):>5}/{len(wins):>1} {sum(1 for t in losses if 'BREAKOUT' in t['orb']):>10}/{len(losses):>1}")
print(f"{'Has Chasing Penalty':<30} {pct(wins, 'has_chasing', True):>12} {pct(losses, 'has_chasing', True):>12}")
print(f"{'VWAP Steepening':<30} {pct(wins, 'vwap_steep', 'True'):>12} {pct(losses, 'vwap_steep', 'True'):>12}")

# Time analysis
print(f"\n{'TIME OF ENTRY ANALYSIS':=^80}")
for t in trades:
    time_parts = t['time'].strip().split(' ')
    if len(time_parts) >= 2:
        hour_min = time_parts[1]
    else:
        hour_min = t['time']
    marker = 'WIN' if t['result'] == 'WIN' else 'LOSS'
    print(f"  {t['sym']:<14} entered at {hour_min:<10} -> {marker:<5} ({t['pnl']:>+7,})")

print(f"\n{'ACTIONABLE INSIGHTS':=^80}")
print("""
Look for these SMART PATTERNS to size up or block:

1. EMA EXPANDING + VWAP Aligned + Follow-Through >= 1 = HIGH CONVICTION
   -> Size up 2-3x
   
2. ORB Hold >= 3 candles means breakout is CONFIRMED
   -> Early ORB trades (hold < 2) are risky
   
3. EXPLOSIVE volume WITHOUT EMA expansion = EXHAUSTION
   -> Should REDUCE size, not increase
   
4. Chasing > 1.5% = DON'T trade
   -> Every chasing trade lost money
   
5. TIME-OF-DAY: Morning entries (before 10:30) vs afternoon
   -> Compare win rates
""")
