"""
BOS/SWEEP REPLAY v2 — Sensitivity Analysis
=============================================
Tests multiple penalty levels to find optimal sweep penalty.
Uses real Kite API candle data.
"""
import json, os, sys
from datetime import datetime

with open('trade_history.json') as f:
    trades = json.load(f)

today = [t for t in trades if '2026-02-16' in str(t.get('timestamp', ''))]

# Try Kite API
kite = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    from kiteconnect import KiteConnect
    api_key = os.environ.get('ZERODHA_API_KEY', '')
    access_token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
    if api_key and access_token:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        kite.profile()
except:
    kite = None


def detect_bos_sweep_kite(symbol, entry_time_str, kite_client):
    try:
        entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
        from_dt = entry_time.replace(hour=9, minute=15, second=0)
        to_dt = entry_time
        sym_clean = symbol.replace('NSE:', '')
        token_map = {i['tradingsymbol']: i['instrument_token']
                     for i in kite_client.instruments('NSE')
                     if i['tradingsymbol'] == sym_clean}
        if sym_clean not in token_map:
            return 'NONE', 'NONE'
        inst_token = token_map[sym_clean]
        candles = kite_client.historical_data(inst_token, from_dt, to_dt, '3minute')
        if len(candles) < 8:
            return 'NONE', 'NONE'
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        closes = [c['close'] for c in candles]
        n = len(highs)
        swing_high = 0
        for si in range(n - 2, max(2, n - 16), -1):
            if (highs[si] > highs[si-1] and highs[si] > highs[si-2] and
                (highs[si] > highs[si-3] if si >= 3 else True) and
                si + 1 < n and highs[si] > highs[si+1]):
                swing_high = highs[si]
                break
        swing_low = 0
        for si in range(n - 2, max(2, n - 16), -1):
            if (lows[si] < lows[si-1] and lows[si] < lows[si-2] and
                (lows[si] < lows[si-3] if si >= 3 else True) and
                si + 1 < n and lows[si] < lows[si+1]):
                swing_low = lows[si]
                break
        cur_high = highs[-1]
        cur_low = lows[-1]
        cur_close = closes[-1]
        bos = 'NONE'
        sweep = 'NONE'
        if swing_high > 0 and cur_high > swing_high:
            if cur_close > swing_high:
                bos = 'BOS_HIGH'
            else:
                sweep = 'SWEEP_HIGH'
        if swing_low > 0 and cur_low < swing_low:
            if cur_close < swing_low:
                bos = 'BOS_LOW'
            else:
                sweep = 'SWEEP_LOW'
        return bos, sweep
    except:
        return 'NONE', 'NONE'


# Pre-detect all signals once
signals = []
for t in today:
    sym = t.get('underlying', t.get('symbol', '?'))
    if kite:
        bos, sweep = detect_bos_sweep_kite(sym, t.get('timestamp', ''), kite)
    else:
        bos, sweep = 'NONE', 'NONE'
    signals.append((bos, sweep))

# First: Show which trades had signals
print("=" * 80)
print("  TRADES WITH BOS/SWEEP SIGNALS DETECTED (Real Candle Data)")
print("=" * 80)
for i, t in enumerate(today):
    bos, sweep = signals[i]
    if bos == 'NONE' and sweep == 'NONE':
        continue
    sym = t.get('underlying', '?').replace('NSE:', '')
    opt_type = t.get('option_type', '?')
    meta = t.get('entry_metadata', {}) or {}
    score = meta.get('entry_score', 0) or 0
    pnl = t.get('pnl', 0) or 0
    result = t.get('result', '?')
    direction = t.get('direction', 'BUY' if opt_type == 'CE' else 'SELL')
    vwap = meta.get('vwap_position', '?')
    sig = bos if bos != 'NONE' else sweep
    alignment = ''
    if 'HIGH' in sig and direction in ('BUY', 'HOLD'):
        alignment = 'ALIGNED' if 'BOS' in sig else 'TRAP'
    elif 'LOW' in sig and direction in ('SELL',):
        alignment = 'ALIGNED' if 'BOS' in sig else 'TRAP'
    else:
        alignment = 'CONFLICT' if 'BOS' in sig else 'TRAP'
    
    print(f"  {sym:<16} {opt_type:<3} Score={score:>5.1f}  PnL={pnl:>+10,.0f}  "
          f"Signal={sig:<14} Dir={direction:<4} VWAP={vwap:<12} → {alignment}")

# Now test different penalty scenarios
print("\n" + "=" * 80)
print("  SENSITIVITY ANALYSIS: Different Sweep Penalty Levels")
print("=" * 80)

BLOCK_THRESHOLD = 56
STANDARD_THRESHOLD = 65

scenarios = [
    ("Current (-8/-12)", -8, -12, +5, -3),
    ("Medium (-12/-18)", -12, -18, +5, -3),
    ("Strong (-15/-20)", -15, -20, +5, -3),
    ("Score Cap (60)", 'CAP60', 'CAP60', +5, -3),
    ("Score Cap (55)", 'CAP55', 'CAP55', +5, -3),
    ("Gate: Block", 'GATE', 'GATE', +5, -3),
]

orig_pnl = sum(t.get('pnl', 0) or 0 for t in today)
orig_loss = sum(t.get('pnl', 0) or 0 for t in today if (t.get('pnl', 0) or 0) < 0)

print(f"\n  Original PnL: Rs {orig_pnl:>10,.0f}   |   Original Gross Loss: Rs {orig_loss:>10,.0f}")
print(f"  {'Scenario':<25} {'Adj PnL':>12} {'Delta':>10} {'Loss':>12} {'Saved':>10} {'Blocked':>8} {'Sized↓':>8} {'Boost':>8}")
print("  " + "-" * 96)

for name, base_pen, amp_pen, bos_reward, bos_conflict in scenarios:
    adj_pnl = 0
    adj_loss = 0
    n_blocked = 0
    n_sized = 0
    n_boosted = 0
    
    for i, t in enumerate(today):
        bos, sweep = signals[i]
        meta = t.get('entry_metadata', {}) or {}
        old_score = meta.get('entry_score', 0) or 0
        pnl = t.get('pnl', 0) or 0
        opt_type = t.get('option_type', '?')
        direction = t.get('direction', 'BUY' if opt_type == 'CE' else 'SELL')
        vwap = meta.get('vwap_position', 'AT_VWAP')
        
        # Calculate adjustment
        adj = 0
        is_gate_block = False
        
        if bos != 'NONE':
            aligned = (('HIGH' in bos and direction in ('BUY', 'HOLD')) or
                      ('LOW' in bos and direction in ('SELL',)))
            adj = bos_reward if aligned else bos_conflict
        
        if sweep != 'NONE':
            if base_pen == 'GATE':
                is_gate_block = True
            elif base_pen == 'CAP60':
                adj = min(0, 60 - old_score) if old_score > 60 else 0
            elif base_pen == 'CAP55':
                adj = min(0, 55 - old_score) if old_score > 55 else 0
            else:
                below_vwap = vwap in ('BELOW_VWAP', 'AT_VWAP')
                is_amplified = (('HIGH' in sweep and below_vwap) or
                              ('LOW' in sweep and not below_vwap))
                adj = amp_pen if is_amplified else base_pen
        
        new_score = old_score + adj
        
        if is_gate_block and old_score >= BLOCK_THRESHOLD:
            # Gate block: trade not taken
            new_pnl = 0
            n_blocked += 1
        elif sweep != 'NONE' and new_score < BLOCK_THRESHOLD and old_score >= BLOCK_THRESHOLD:
            new_pnl = 0
            n_blocked += 1
        elif sweep != 'NONE' and new_score < STANDARD_THRESHOLD and old_score >= STANDARD_THRESHOLD:
            new_pnl = pnl * 0.75
            n_sized += 1
        elif adj > 0:
            new_pnl = pnl
            n_boosted += 1
        else:
            new_pnl = pnl
        
        adj_pnl += new_pnl
        if new_pnl < 0:
            adj_loss += new_pnl
    
    delta = adj_pnl - orig_pnl
    saved = adj_loss - orig_loss
    marker = " ◄" if adj_pnl > orig_pnl else ""
    print(f"  {name:<25} Rs {adj_pnl:>9,.0f} {delta:>+10,.0f} Rs {adj_loss:>9,.0f} {saved:>+10,.0f}    {n_blocked:>4}    {n_sized:>4}    {n_boosted:>4}{marker}")

# Detailed analysis of what Gate Block would do
print("\n" + "=" * 80)
print("  GATE BLOCK DETAIL: Which trades get blocked if SWEEP = hard block?")
print("=" * 80)
for i, t in enumerate(today):
    bos, sweep = signals[i]
    if sweep == 'NONE':
        continue
    sym = t.get('underlying', '?').replace('NSE:', '')
    opt_type = t.get('option_type', '?')
    meta = t.get('entry_metadata', {}) or {}
    score = meta.get('entry_score', 0) or 0
    pnl = t.get('pnl', 0) or 0
    result = t.get('result', '?')
    save_or_cost = "SAVES" if pnl < 0 else "COSTS"
    print(f"  {sym:<16} {opt_type:<3} Score={score:>5.1f}  PnL={pnl:>+10,.0f}  "
          f"{sweep}  → BLOCKED → {save_or_cost} Rs {abs(pnl):>10,.0f}")

# Net gate impact
sweep_trades = [(i, t) for i, t in enumerate(today) if signals[i][1] != 'NONE']
gate_saved = sum(t.get('pnl', 0) or 0 for _, t in sweep_trades if (t.get('pnl', 0) or 0) < 0)
gate_cost = sum(t.get('pnl', 0) or 0 for _, t in sweep_trades if (t.get('pnl', 0) or 0) > 0)
gate_net = -(gate_saved + gate_cost)  # negative of total PnL of blocked trades
print(f"\n  Gate Block Net Impact: Blocks Rs {gate_cost:>+,.0f} winners + avoids Rs {gate_saved:>+,.0f} losers")
print(f"  Net: Rs {-gate_cost - (-gate_saved):>+,.0f}")

# Best approach analysis
print("\n" + "=" * 80)
print("  RECOMMENDATION")
print("=" * 80)
print("""
  The current -8/-12 penalty is TOO WEAK for today's data:
  - GODREJCP (Score 81, -₹23K loss): SWEEP detected but 81-8=73, still trades
  - AXISBANK (Score 67, +₹18K win): SWEEP detected, sized down, LOST ₹4.5K profit
  
  Net: Feature HURT today by -₹5,453.
  
  Score Cap approach (cap sweep at 60) is more surgical:
  - GODREJCP 81 → 60: drops tier, trades at 0.75x → saves ₹5,750
  - AXISBANK 67 → 60: drops below STANDARD, trades at 0.75x → same size reduction
  
  Options for tuning:
  1. Keep current penalties, observe over more days (safe but slow)
  2. Increase sweep penalty to -15/-20 (still may not block high-score trades)
  3. Cap sweep-detected trades at max score 60 (surgical, effective)
  4. Hard gate block all sweep trades (aggressive, blocks winners too)
""")
