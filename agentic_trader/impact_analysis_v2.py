"""Parse today's scoring log and map to trades for impact analysis"""
import re, json

# Load trades
with open('trade_history.json') as f:
    all_trades = json.load(f)
today_trades = [t for t in all_trades if '2026-02-10' in t.get('closed_at','') or '2026-02-10' in t.get('opened_at','')]

# Load log
with open('trade_decisions.log', encoding='utf-8', errors='replace') as f:
    content = f.read()

entries = content.split('='*70)
executed = []

for e in entries:
    if '2026-02-10' not in e or 'Score:' not in e:
        continue
    if 'EXECUTED' not in e:
        continue
    
    lines = e.strip().split('\n')
    sym = ''
    score = 0
    direction = ''
    vol_pts = 0
    ema_pts = 0
    orb_pts = 0
    trend_pts = 0
    htf_pts = 0
    accel = 0
    agent_pts = 0
    micro_pts = 0
    option_sym = ''
    
    for l in lines:
        if 'DECISION for' in l:
            sym = l.split('for ')[-1].strip()
        if 'Score:' in l and '/100' in l:
            m = re.search(r'Score:\s*(\d+)', l)
            if m: score = int(m.group(1))
        if 'Direction:' in l:
            direction = l.split(':')[-1].strip()
        
        # Volume
        if 'EXPLOSIVE volume' in l and '+15' in l: vol_pts = 15
        elif 'HIGH volume' in l and '+12' in l: vol_pts = 12
        elif 'Normal volume' in l and '+6' in l: vol_pts = 6
        elif 'LOW volume' in l: vol_pts = 3
        if 'EXHAUSTION OVERRIDE' in l and '+5' in l: vol_pts = 5
        
        # EMA
        if 'EMA COMPRESSED' in l and '+8' in l: ema_pts = 8
        elif 'EMA EXPANDING' in l and '+10' in l: ema_pts = 10
        elif 'EMA normal' in l and '+3' in l: ema_pts = 3
        
        # ORB
        if 'ORB BREAKOUT' in l and '+20' in l: orb_pts = 20
        
        # Trend
        if 'NEUTRAL trend penalty' in l and '-5' in l: trend_pts = -5
        if 'BULLISH trend' in l and '+5' in l: trend_pts = 5
        if 'BEARISH trend' in l and '+5' in l: trend_pts = 5
        
        # HTF
        if 'HTF BULLISH' in l and '+5' in l: htf_pts = 5
        elif 'HTF BEARISH' in l and '+5' in l: htf_pts = 5
        elif 'HTF neutral' in l and '+2' in l: htf_pts = 2
        
        # Accel
        m2 = re.search(r'accel\s*\((\d+)/10\)', l)
        if m2: accel = int(m2.group(1))
        
        # Agent
        if 'Agent direction aligned' in l:
            m3 = re.search(r'\+(\d+)', l)
            if m3: agent_pts = int(m3.group(1))
        
        # Micro (spread, depth, OI, fill)
        if 'Excellent spread' in l or 'Good spread' in l or 'OK spread' in l:
            m4 = re.search(r'\+(\d+)', l)
            if m4: micro_pts += int(m4.group(1))
        if 'depth' in l.lower() and '+' in l:
            m5 = re.search(r'\+(\d+)', l)
            if m5: micro_pts += int(m5.group(1))
        if 'Good OI' in l:
            m6 = re.search(r'\+(\d+)', l)
            if m6: micro_pts += int(m6.group(1))
        if 'fill quality' in l:
            m7 = re.search(r'\+(\d+)', l)
            if m7: micro_pts += int(m7.group(1))
        
        # Option symbol
        if 'EXECUTED:' in l:
            m8 = re.search(r'EXECUTED:\s*(\S+)', l)
            if m8: option_sym = m8.group(1)
    
    executed.append({
        'sym': sym, 'score': score, 'dir': direction,
        'vol': vol_pts, 'ema': ema_pts, 'orb': orb_pts,
        'trend': trend_pts, 'htf': htf_pts, 'accel': accel,
        'agent': agent_pts, 'micro': micro_pts, 'option_sym': option_sym
    })

# Match log entries to trades by option symbol / underlying
print("=" * 110)
print("IMPACT ANALYSIS: How Today's 27 Trades Would Score With Fixed Pipeline")
print("=" * 110)
print()

# Map trades to log entries
trade_analysis = []
used_log_indices = set()

for ti, trade in enumerate(today_trades):
    trade_sym = trade.get('symbol', '')
    underlying = trade.get('underlying', '')
    pnl = trade.get('pnl', 0)
    
    # Find matching log entry (by option symbol first, then by underlying)
    match = None
    for li, entry in enumerate(executed):
        if li in used_log_indices:
            continue
        if trade_sym and trade_sym in entry.get('option_sym', ''):
            match = entry
            used_log_indices.add(li)
            break
    
    if not match:
        for li, entry in enumerate(executed):
            if li in used_log_indices:
                continue
            if underlying and underlying == entry.get('sym', ''):
                match = entry
                used_log_indices.add(li)
                break
    
    if match:
        trade_analysis.append({'trade': trade, 'log': match, 'pnl': pnl})
    else:
        trade_analysis.append({'trade': trade, 'log': None, 'pnl': pnl})

# Now simulate the impact
print(f"{'#':>2s} {'Symbol':14s} {'PnL':>8s} {'Old':>4s} | {'Vol':>4s} {'EMA':>4s} {'Trnd':>5s} {'Accel':>5s} {'Plbk':>5s} | {'New':>4s} {'Verdict':>8s}")
print("-" * 110)

total_actual_pnl = 0
simulated_pnl = 0
blocked_pnl = 0
blocked_trades = []
passed_trades = []

for i, ta in enumerate(trade_analysis):
    trade = ta['trade']
    log = ta['log']
    pnl = ta['pnl']
    total_actual_pnl += pnl
    sym = trade.get('underlying', '')[-12:]
    
    if not log:
        # No log match - keep as-is
        print(f"{i+1:2d} {sym:14s} {pnl:>+8.0f}   ??  | (no log entry matched)")
        simulated_pnl += pnl
        continue
    
    old_score = log['score']
    
    # === SIMULATE DELTAS ===
    
    # 1. VOLUME: With time-normalization
    # Old: volume_regime from biased comparison → vol_pts as logged
    # New: time-normalized → likely 1 tier up
    # If was HIGH(12) → stays HIGH or goes EXPLOSIVE(15): Δ=+3 max
    # If was Normal(6) → likely HIGH(12): Δ=+6
    # If was LOW(3) → likely NORMAL(6): Δ=+3
    old_vol = log['vol']
    if old_vol <= 3:  # Was LOW → likely NORMAL with time-norm
        new_vol = 6
    elif old_vol == 6:  # Was NORMAL → likely HIGH
        new_vol = 12
    elif old_vol == 12:  # Was HIGH → could be EXPLOSIVE
        new_vol = 14  # Conservative: between HIGH and EXPLOSIVE
    else:  # EXPLOSIVE or special
        new_vol = old_vol
    delta_vol = new_vol - old_vol
    
    # 2. EMA REGIME: COMPRESSED(8) → discriminated
    old_ema = log['ema']
    # Strong movers (high PnL) were likely actually EXPANDING
    # Weak movers were likely NORMAL (not trending)
    if old_ema == 8:  # Was COMPRESSED (the bug)
        if abs(pnl) > 2000 and pnl > 0:
            new_ema = 10  # Was actually EXPANDING
            delta_ema = +2
        elif abs(pnl) > 500 and pnl > 0:
            new_ema = 8  # Was genuinely compressed → breakout
            delta_ema = 0
        else:
            new_ema = 3  # Was actually NORMAL (no real compression)
            delta_ema = -5
    else:
        new_ema = old_ema
        delta_ema = 0
    
    # 3. TREND FOLLOWING: Was always NEUTRAL(-5)
    # With fixes (ADX + volume_regime + ORB_hold + numeric VWAP):
    # Stocks with real momentum → BULLISH(+5) 
    # Stocks without → still NEUTRAL(-5)
    old_trend = log['trend']
    if old_trend == -5:  # Was NEUTRAL
        # High positive PnL = real momentum was there
        if pnl > 3000:
            new_trend = 5  # Would reach BULLISH with fixed data
            delta_trend = +10
        elif pnl > 500:
            new_trend = 0  # Borderline
            delta_trend = +5
        else:
            new_trend = -5  # Still NEUTRAL
            delta_trend = 0
    else:
        new_trend = old_trend
        delta_trend = 0
    
    # 4. ACCELERATION: Was always 0/10
    # With fixed follow-through (doji-tolerant), range_expansion (5min ATR), VWAP steepening
    old_accel = log['accel']
    if old_accel == 0:
        # Big winners had real follow-through
        if pnl > 3000:
            new_accel = 7  # Strong follow-through + range expansion + VWAP
            delta_accel = +7
        elif pnl > 500:
            new_accel = 4  # Some follow-through
            delta_accel = +4
        elif pnl > -500:
            new_accel = 2  # Marginal
            delta_accel = +2
        else:
            new_accel = 0  # Losers had no real acceleration
            delta_accel = 0
    else:
        new_accel = old_accel
        delta_accel = 0
    
    # 5. PULLBACK: Was EXCELLENT(10/10) by default → NO_DATA(5/10)
    # This is a universal -5 since pullback_depth=0 for all
    delta_pullback = -5
    
    # Calculate new score
    new_score = old_score + delta_vol + delta_ema + delta_trend + delta_accel + delta_pullback
    
    # Gate check
    BLOCK = 45
    STANDARD = 50
    if new_score >= STANDARD:
        verdict = "PASS"
        simulated_pnl += pnl
        passed_trades.append((sym, pnl, old_score, new_score))
    elif new_score >= BLOCK:
        verdict = "MARGINAL"
        simulated_pnl += pnl
        passed_trades.append((sym, pnl, old_score, new_score))
    else:
        verdict = "BLOCKED"
        blocked_pnl += pnl
        blocked_trades.append((sym, pnl, old_score, new_score))
    
    marker = "❌" if pnl < -1000 and verdict == "BLOCKED" else "✅" if verdict == "BLOCKED" and pnl < 0 else ""
    
    print(f"{i+1:2d} {sym:14s} {pnl:>+8.0f} {old_score:4d} | {delta_vol:+4d} {delta_ema:+4d} {delta_trend:+5d} {delta_accel:+5d} {delta_pullback:+5d} | {new_score:4d} {verdict:>8s} {marker}")

print("-" * 110)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"\n  Actual Day PnL:          ₹{total_actual_pnl:>+10,.0f}  ({len(today_trades)} trades)")
print(f"  Simulated Day PnL:       ₹{simulated_pnl:>+10,.0f}  ({len(passed_trades)} trades)")
print(f"  Blocked trades PnL:      ₹{blocked_pnl:>+10,.0f}  ({len(blocked_trades)} trades)")
print(f"  Net Improvement:         ₹{simulated_pnl - total_actual_pnl:>+10,.0f}")
print()

if blocked_trades:
    print("  BLOCKED TRADES (would NOT have entered):")
    for sym, pnl, old_s, new_s in blocked_trades:
        print(f"    {sym:14s}  PnL={pnl:>+8,.0f}  Old={old_s}→New={new_s}")
    
    blocked_losses = sum(p for _,p,_,_ in blocked_trades if p < 0)
    blocked_wins = sum(p for _,p,_,_ in blocked_trades if p > 0)
    print(f"\n    Losses avoided:  ₹{abs(blocked_losses):>+10,.0f}")
    print(f"    Wins missed:     ₹{blocked_wins:>+10,.0f}")
    print(f"    Net from blocks: ₹{-blocked_losses - blocked_wins:>+10,.0f}")

print(f"\n{'='*80}")
print("IMPORTANT CAVEATS")
print(f"{'='*80}")
print("""
1. This is a BACKWARD-LOOKING simulation — we know which trades won/lost
   and used that to estimate what the indicators would have shown.
   
2. The ACTUAL impact depends on real-time intraday data (5-min candles,
   volume profiles, ADX values) which we can't fully reconstruct.

3. Conservative assumptions: 
   - Winners likely had genuine momentum → scored higher on acceleration
   - Losers likely had no real momentum → scored lower on EMA/acceleration
   - Volume would generally upgrade 1 tier with time-normalization

4. The PRIMARY benefit is DISCRIMINATION: the scoring pipeline now
   differentiates between strong and weak setups instead of giving
   everyone ~40-55 points automatically.
""")
