"""Diagnose why PREMIERENE scores 0"""
import sys, json
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from options_trader import get_intraday_scorer, IntradaySignal
from zerodha_tools import get_zerodha_tools

tools = get_zerodha_tools()
md = tools.get_market_data(['NSE:PREMIERENE'], force_fresh=True)
d = md.get('NSE:PREMIERENE', {})

# Show key fields
keys = ['ltp','open','high','low','vwap','orb_signal','price_vs_vwap','vwap_slope',
        'ema_regime','volume_regime','rsi_14','momentum_15m','htf_alignment',
        'chop_zone','follow_through_candles','range_expansion_ratio','adx',
        'change_pct','oi_signal','market_breadth','bos_signal','atr_14',
        'orb_high','orb_low','orb_strength','orb_hold_candles','orb_reentries',
        'buy_pressure_pct','ema_9','swing_high_level','swing_low_level']
print("=== MARKET DATA ===")
for k in keys:
    v = d.get(k, "MISSING")
    print(f"  {k}: {v}")
print()

# Score it
sig = IntradaySignal(
    symbol='NSE:PREMIERENE',
    orb_signal=d.get('orb_signal','INSIDE_ORB'),
    vwap_position=d.get('price_vs_vwap', d.get('vwap_position','AT_VWAP')),
    vwap_trend=d.get('vwap_slope', d.get('vwap_trend','FLAT')),
    ema_regime=d.get('ema_regime','NORMAL'),
    volume_regime=d.get('volume_regime','NORMAL'),
    rsi=d.get('rsi_14',50.0),
    price_momentum=d.get('momentum_15m',0.0),
    htf_alignment=d.get('htf_alignment','NEUTRAL'),
    chop_zone=d.get('chop_zone',False),
    follow_through_candles=d.get('follow_through_candles',0),
    range_expansion_ratio=d.get('range_expansion_ratio',0.0),
    vwap_slope_steepening=d.get('vwap_slope_steepening',False),
    atr=d.get('atr_14',0.0)
)

scorer = get_intraday_scorer()
dec = scorer.score_intraday_signal(sig, market_data=d, caller_direction=None)

print("=== SCORING RESULT ===")
print(f"confidence_score: {dec.confidence_score}")
print(f"direction: {dec.recommended_direction}")
print(f"should_trade: {dec.should_trade}")
print(f"audit: {getattr(scorer, '_last_score_audit', 'N/A')}")
print()
print("=== REASONS ===")
for r in dec.reasons:
    print(f"  {r}")
print()
print("=== WARNINGS ===")
for w in dec.warnings:
    print(f"  {w}")
