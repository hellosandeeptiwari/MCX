"""
BOS/SWEEP REPLAY on Feb 16 — Actual Impact Simulation
=======================================================
Uses stored trade metadata + Zerodha historical data (if available)
to simulate what BOS/Sweep scoring would have done to each trade.

If Kite API not available, falls back to metadata-based estimation
using: orb_strength, range_expansion, vwap_position, spot vs R1/S1.
"""
import json, os, sys
from datetime import datetime, timedelta

# Load trades
with open('trade_history.json') as f:
    trades = json.load(f)

today = [t for t in trades if '2026-02-16' in str(t.get('timestamp', ''))]

# Try fetching candle data via Kite API
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
        # Test with a quick call
        kite.profile()
        print("[OK] Kite API connected - using real candle data\n")
except Exception as e:
    kite = None
    print(f"[WARN] Kite API unavailable ({str(e)[:60]}) - using metadata estimation\n")


def detect_bos_sweep_kite(symbol, entry_time_str, kite_client):
    """Fetch 3-min candles and run actual BOS/Sweep detection"""
    try:
        entry_time = datetime.fromisoformat(entry_time_str.replace('Z', ''))
        from_dt = entry_time.replace(hour=9, minute=15, second=0)
        to_dt = entry_time

        sym_clean = symbol.replace('NSE:', '')
        token_map = {i['tradingsymbol']: i['instrument_token']
                     for i in kite_client.instruments('NSE')
                     if i['tradingsymbol'] == sym_clean}
        if sym_clean not in token_map:
            return 'NONE', 'NONE', 0, 0

        inst_token = token_map[sym_clean]
        candles = kite_client.historical_data(inst_token, from_dt, to_dt, '3minute')

        if len(candles) < 8:
            return 'NONE', 'NONE', 0, 0

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

        return bos, sweep, swing_high, swing_low

    except Exception as e:
        return 'NONE', 'NONE', 0, 0


def estimate_bos_sweep_metadata(trade):
    """Estimate BOS/Sweep from stored metadata when no candle data"""
    meta = trade.get('entry_metadata', {}) or {}
    direction = trade.get('direction', '')
    opt_type = trade.get('option_type', '?')
    rationale = str(trade.get('rationale', ''))

    orb_strength = meta.get('orb_strength_pct', 0)
    range_exp = meta.get('range_expansion_ratio', 0)
    vwap_pos = meta.get('vwap_position', '')
    rsi = meta.get('rsi', 50)
    spot = meta.get('spot_price', 0)
    ft = meta.get('follow_through_candles', 0)
    orb_sig = meta.get('orb_signal', '')

    bos = 'NONE'
    sweep = 'NONE'
    confidence = 'LOW'

    if opt_type == 'CE' or direction == 'BUY':
        near_resistance = ('R1' in rationale or 'resistance' in rationale.lower() or
                          'near R' in rationale)
        wick_heavy = orb_strength > 40 and range_exp < 0.1
        weak_body = range_exp < 0.05 and orb_strength > 30

        if near_resistance and (wick_heavy or weak_body):
            sweep = 'SWEEP_HIGH'
            confidence = 'HIGH'
        elif wick_heavy and rsi > 65:
            sweep = 'SWEEP_HIGH'
            confidence = 'MEDIUM'
        elif orb_sig == 'BREAKOUT_UP' and ft >= 2 and range_exp >= 0.1:
            bos = 'BOS_HIGH'
            confidence = 'HIGH'
        elif orb_sig == 'BREAKOUT_UP' and ft >= 1:
            bos = 'BOS_HIGH'
            confidence = 'MEDIUM'

    elif opt_type == 'PE' or direction == 'SELL':
        near_support = ('S1' in rationale or 'support' in rationale.lower() or
                       'near S' in rationale)
        wick_heavy = orb_strength > 40 and range_exp < 0.1
        weak_body = range_exp < 0.05 and orb_strength > 30

        if near_support and (wick_heavy or weak_body):
            sweep = 'SWEEP_LOW'
            confidence = 'HIGH'
        elif wick_heavy and rsi < 35:
            sweep = 'SWEEP_LOW'
            confidence = 'MEDIUM'
        elif orb_sig == 'BREAKOUT_DOWN' and ft >= 2 and range_exp >= 0.1:
            bos = 'BOS_LOW'
            confidence = 'HIGH'
        elif orb_sig == 'BREAKOUT_DOWN' and ft >= 1:
            bos = 'BOS_LOW'
            confidence = 'MEDIUM'

    return bos, sweep, confidence


def calc_score_adjustment(bos, sweep, direction, vwap_pos):
    """Calculate the score adjustment from BOS/Sweep"""
    adj = 0
    reason = ''

    if bos == 'BOS_HIGH':
        if direction in ('BUY', 'HOLD'):
            adj = +5
            reason = 'BOS_HIGH confirmed breakout (+5)'
        else:
            adj = -3
            reason = 'BOS_HIGH conflicts with SELL (-3)'
    elif bos == 'BOS_LOW':
        if direction in ('SELL', 'HOLD'):
            adj = +5
            reason = 'BOS_LOW confirmed breakdown (+5)'
        else:
            adj = -3
            reason = 'BOS_LOW conflicts with BUY (-3)'

    if sweep == 'SWEEP_HIGH':
        below_vwap = vwap_pos in ('BELOW_VWAP', 'AT_VWAP')
        if below_vwap:
            adj = -12
            reason = 'SWEEP_HIGH + below VWAP = strong rejection (-12)'
        else:
            adj = -8
            reason = 'SWEEP_HIGH fake breakout (-8)'
    elif sweep == 'SWEEP_LOW':
        above_vwap = vwap_pos in ('ABOVE_VWAP', 'AT_VWAP')
        if above_vwap:
            adj = -12
            reason = 'SWEEP_LOW + above VWAP = strong bounce (-12)'
        else:
            adj = -8
            reason = 'SWEEP_LOW fake breakdown (-8)'

    return adj, reason


# === MAIN SIMULATION ===
print("=" * 100)
print(f"{'Symbol':<16} {'Type':<4} {'Old':>5} {'Adj':>5} {'New':>5} {'Action':<12} {'PnL':>10} {'New PnL':>10}  {'Signal':<28} {'Note'}")
print("=" * 100)

BLOCK_THRESHOLD = 56
STANDARD_THRESHOLD = 65
original_total_pnl = 0
adjusted_total_pnl = 0
original_gross_loss = 0
adjusted_gross_loss = 0
trades_blocked = 0
trades_boosted = 0
trades_sized_down = 0
trades_unchanged = 0
blocked_pnl_saved = 0

for t in today:
    sym = t.get('underlying', t.get('symbol', '?'))
    sym_short = sym.replace('NSE:', '').replace('BSE:', '')
    opt_type = t.get('option_type', '?')
    direction = t.get('direction', 'BUY' if opt_type == 'CE' else 'SELL')
    meta = t.get('entry_metadata', {}) or {}
    old_score = meta.get('entry_score', t.get('entry_score', 0)) or 0
    pnl = t.get('pnl', 0) or 0
    vwap_pos = meta.get('vwap_position', 'AT_VWAP')
    result = t.get('result', '?')

    original_total_pnl += pnl
    if pnl < 0:
        original_gross_loss += pnl

    # Detect BOS/Sweep
    if kite:
        bos, sweep, sh, sl = detect_bos_sweep_kite(sym, t.get('timestamp', ''), kite)
        confidence = 'API'
    else:
        bos, sweep, confidence = estimate_bos_sweep_metadata(t)

    # Calculate score adjustment
    adj, reason = calc_score_adjustment(bos, sweep, direction, vwap_pos)
    new_score = old_score + adj

    # Determine action
    note = ''
    if adj == 0:
        action = '-'
        new_pnl = pnl
        trades_unchanged += 1
    elif sweep != 'NONE' and new_score < BLOCK_THRESHOLD and old_score >= BLOCK_THRESHOLD:
        action = 'BLOCKED'
        new_pnl = 0
        trades_blocked += 1
        blocked_pnl_saved += pnl  # If pnl was negative, this is savings
        note = reason
    elif sweep != 'NONE' and new_score < STANDARD_THRESHOLD and old_score >= STANDARD_THRESHOLD:
        # Dropped tier -> 0.75x sizing
        action = 'SIZE_DOWN'
        new_pnl = pnl * 0.75
        trades_sized_down += 1
        note = reason
    elif adj > 0:
        action = 'BOOSTED'
        new_pnl = pnl
        trades_boosted += 1
        note = reason
    elif adj < 0:
        action = 'PENALIZED'
        new_pnl = pnl
        trades_unchanged += 1  # Still taken, score lower but same PnL
        note = reason
    else:
        action = '-'
        new_pnl = pnl
        trades_unchanged += 1

    adjusted_total_pnl += new_pnl
    if new_pnl < 0:
        adjusted_gross_loss += new_pnl

    signal_str = ''
    if bos != 'NONE':
        signal_str = f"{bos} [{confidence}]"
    elif sweep != 'NONE':
        signal_str = f"{sweep} [{confidence}]"
    else:
        signal_str = f"NONE"

    marker = ''
    if action == 'BLOCKED':
        marker = ' <<<'
    elif action == 'SIZE_DOWN':
        marker = ' <<'

    pnl_display = f"{pnl:>10,.0f}"
    new_pnl_display = f"{new_pnl:>10,.0f}"
    if new_pnl != pnl:
        new_pnl_display = f"{new_pnl:>10,.0f}*"

    print(f"{sym_short:<16} {opt_type:<4} {old_score:>5.0f} {adj:>+5d} {new_score:>5.0f} {action:<12} {pnl_display} {new_pnl_display}  {signal_str:<28} {note}{marker}")

print("=" * 100)

# === SUMMARY ===
delta = adjusted_total_pnl - original_total_pnl
print()
print("=" * 60)
print("  BOS/SWEEP IMPACT SIMULATION — SUMMARY")
print("=" * 60)
print(f"  Original Gross PnL:    Rs {original_total_pnl:>12,.0f}")
print(f"  Adjusted Gross PnL:    Rs {adjusted_total_pnl:>12,.0f}")
print(f"  Delta (improvement):   Rs {delta:>+12,.0f}")
pct = (delta / abs(original_total_pnl) * 100) if original_total_pnl != 0 else 0
print(f"  Improvement:           {pct:>+.1f}%")
print()
print(f"  Original Gross Loss:   Rs {original_gross_loss:>12,.0f}")
print(f"  Adjusted Gross Loss:   Rs {adjusted_gross_loss:>12,.0f}")
loss_saved = adjusted_gross_loss - original_gross_loss
print(f"  Loss Reduction:        Rs {loss_saved:>+12,.0f}")
print()
print(f"  Trades BLOCKED:        {trades_blocked:>3}  (sweep detected, score dropped below 56)")
print(f"  Trades SIZED DOWN:     {trades_sized_down:>3}  (sweep detected, dropped below standard tier)")
print(f"  Trades BOOSTED:        {trades_boosted:>3}  (BOS confirmed, +5 conviction)")
print(f"  Trades UNCHANGED:      {trades_unchanged:>3}  (no BOS/sweep detected)")
print(f"  Total:                 {len(today):>3}")
print()
if not kite:
    print("  NOTE: Using metadata estimation (Kite API offline).")
    print("  Confidence levels: HIGH = strong pattern match,")
    print("  MEDIUM = heuristic match, LOW = no signal detected.")
    print("  Real-candle detection would be MORE accurate.")
print("=" * 60)
