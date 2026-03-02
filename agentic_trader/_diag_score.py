"""Diagnose why PREMIERENE scores 0 - simpler version"""
import sys
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from options_trader import get_intraday_scorer, IntradaySignal

# Simulate what a VOLUME_SURGE +0.5% stock looks like:
# - No ORB breakout (INSIDE_ORB)
# - Slight positive move
# - Volume is HIGH (triggered surge)
d = {
    'ltp': 100.5, 'open': 100.0, 'high': 101.0, 'low': 99.5,
    'vwap': 100.2, 'orb_signal': 'INSIDE_ORB', 'price_vs_vwap': 'ABOVE_VWAP',
    'vwap_slope': 'FLAT', 'ema_regime': 'NORMAL', 'volume_regime': 'HIGH',
    'rsi_14': 52.0, 'momentum_15m': 0.3, 'htf_alignment': 'NEUTRAL',
    'chop_zone': False, 'follow_through_candles': 0,
    'range_expansion_ratio': 0.0, 'adx': 18, 'change_pct': 0.5,
    'oi_signal': 'NEUTRAL', 'market_breadth': 'MIXED',
    'bos_signal': 'NONE', 'atr_14': 2.5,
    'orb_high': 101.0, 'orb_low': 99.5, 'orb_strength': 0.0,
    'orb_hold_candles': 0, 'orb_reentries': 0,
    'buy_pressure_pct': 50, 'ema_9': 100.1,
    'swing_high_level': 0, 'swing_low_level': 0,
}

sig = IntradaySignal(
    symbol='NSE:PREMIERENE',
    orb_signal='INSIDE_ORB',
    vwap_position='ABOVE_VWAP',
    vwap_trend='FLAT',
    ema_regime='NORMAL',
    volume_regime='HIGH',
    rsi=52.0,
    price_momentum=0.3,
    htf_alignment='NEUTRAL',
    chop_zone=False,
    follow_through_candles=0,
    range_expansion_ratio=0.0,
    vwap_slope_steepening=False,
    atr=2.5
)

scorer = get_intraday_scorer()
dec = scorer.score_intraday_signal(sig, market_data=d, caller_direction=None)

print(f"confidence_score: {dec.confidence_score}")
print(f"direction: {dec.recommended_direction}")
print(f"should_trade: {dec.should_trade}")
print(f"audit: {getattr(scorer, '_last_score_audit', 'N/A')}")
print()
for r in dec.reasons:
    print(f"  R: {r}")
for w in dec.warnings:
    print(f"  W: {w}")
