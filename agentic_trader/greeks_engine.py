"""
Greeks Engine — Dynamic SL/Target Based on Option Greeks

Problem: Fixed % stop losses (28% SL) are context-blind.
  - Deep ITM (delta 0.85): 28% SL = underlying moved only ~3.3%
  - Far OTM (delta 0.20): 28% SL = underlying moved ~14% against you  
  - Near expiry: gamma amplifies moves, fixed SL gets hit by noise

Solution: This engine computes real-time Greeks and adjusts exits dynamically.

Key Functions:
  1. compute_live_greeks()     — recalculate delta/gamma/theta/vega from current prices
  2. get_greeks_adjusted_sl()  — dynamic SL based on delta + gamma + DTE
  3. get_greeks_adjusted_target() — dynamic target based on delta + gamma
  4. should_exit_on_greeks()   — exit signals from Greek deterioration
  5. classify_moneyness()      — ITM/ATM/OTM classification

Design Principles:
  - High delta (0.7+): Wider SL in % terms (moves track underlying, less noise)
  - Low delta (< 0.25): Tighter SL (far OTM, premium can evaporate fast)
  - High gamma (near ATM, near expiry): Tighter SL (moves accelerate)
  - Delta collapse (< 0.10): Exit signal (option becoming worthless)
  - Theta bleed guard: Exit if theta is eating more than 3% of premium per hour

Author: Titan v5.2
"""

import math
from datetime import datetime, date
from typing import Dict, Optional, Tuple, NamedTuple
from enum import Enum


class Moneyness(Enum):
    DEEP_ITM = "DEEP_ITM"     # delta > 0.80
    ITM = "ITM"               # delta 0.55 - 0.80
    ATM = "ATM"               # delta 0.40 - 0.55
    OTM = "OTM"               # delta 0.15 - 0.40
    DEEP_OTM = "DEEP_OTM"    # delta < 0.15


class GreeksSnapshot(NamedTuple):
    """Current Greeks state of an option position."""
    delta: float          # Rate of change vs underlying
    gamma: float          # Rate of change of delta
    theta: float          # Time decay per day (negative for longs)
    vega: float           # IV sensitivity
    iv: float             # Current implied volatility
    moneyness: Moneyness  # ITM/ATM/OTM classification
    dte: int              # Days to expiry


# ─── CONFIGURATION ────────────────────────────────────────────────
# These define the SL/target multipliers for each delta band.
# Format: (min_delta, max_delta) → (sl_pct, target_pct)
# sl_pct = max premium loss before exit
# target_pct = premium gain target

GREEKS_SL_TARGET_TABLE = {
    # Deep ITM: behaves like stock. Wide SL, modest target.
    # Delta 0.80+: 1% underlying move ≈ 0.8% premium move
    Moneyness.DEEP_ITM: {
        'sl_pct': 35,         # 35% SL (wide — low noise risk)
        'target_pct': 50,     # 50% target (moderate — premium dense)
        'speed_gate_candles': 15,  # More patient — low gamma
    },
    # ITM: Good leverage. Standard SL.
    # Delta 0.55-0.80: decent directional exposure
    Moneyness.ITM: {
        'sl_pct': 28,         # 28% SL (standard)
        'target_pct': 65,     # 65% target (good leverage) 
        'speed_gate_candles': 12,
    },
    # ATM: Highest gamma. Moves fast both ways. Moderate SL.
    # Delta 0.40-0.55: maximum gamma zone
    Moneyness.ATM: {
        'sl_pct': 25,         # 25% SL (slightly tighter — gamma amplifies)
        'target_pct': 80,     # 80% target (ATM has best R:R)
        'speed_gate_candles': 10,
    },
    # OTM: Leveraged but fragile. Tight SL.
    # Delta 0.15-0.40: premium can evaporate
    Moneyness.OTM: {
        'sl_pct': 22,         # 22% SL (tight — premium erodes fast)
        'target_pct': 100,    # 100% target (OTM doubles are the win scenario)
        'speed_gate_candles': 8,
    },
    # Deep OTM: Lottery ticket. Very tight SL.
    # Delta < 0.15: almost all time value, decays rapidly
    Moneyness.DEEP_OTM: {
        'sl_pct': 18,         # 18% SL (very tight — likely to expire worthless)
        'target_pct': 150,    # 150% target (moonshot or nothing)
        'speed_gate_candles': 6,
    },
}

# ─── GREEK-BASED EXIT THRESHOLDS (wired to GREEKS_EXIT_CONFIG) ───
def _load_greeks_config():
    try:
        from config import GREEKS_EXIT_CONFIG as _cfg
    except ImportError:
        _cfg = {}
    return {
        'delta_collapse': _cfg.get('delta_collapse_threshold', 0.08),
        'delta_collapse_max_dte': _cfg.get('delta_collapse_max_dte', 2),
        'theta_bleed': _cfg.get('theta_bleed_max_pct_hour', 3.0),
        'theta_bleed_min_candles': _cfg.get('theta_bleed_min_candles', 4),
    }

_GC = _load_greeks_config()
DELTA_COLLAPSE_THRESHOLD = _GC['delta_collapse']
DELTA_COLLAPSE_MAX_DTE = _GC['delta_collapse_max_dte']
THETA_BLEED_MAX_PCT_HOUR = _GC['theta_bleed']
THETA_BLEED_MIN_CANDLES = _GC['theta_bleed_min_candles']
GAMMA_RISK_MULTIPLIER = 1.5          # When gamma > 0.05, tighten SL by this factor


def classify_moneyness(delta: float) -> Moneyness:
    """Classify option moneyness from absolute delta value."""
    abs_d = abs(delta) if delta else 0
    if abs_d >= 0.80:
        return Moneyness.DEEP_ITM
    elif abs_d >= 0.55:
        return Moneyness.ITM
    elif abs_d >= 0.40:
        return Moneyness.ATM
    elif abs_d >= 0.15:
        return Moneyness.OTM
    else:
        return Moneyness.DEEP_OTM


def compute_live_greeks(
    spot_price: float,
    strike: float,
    dte: int,
    risk_free_rate: float,
    option_ltp: float,
    option_type: str,  # 'CE' or 'PE'
) -> Optional[GreeksSnapshot]:
    """
    Recompute Greeks from current market prices using Black-Scholes.
    
    This is called on every monitoring tick to get fresh Greeks
    rather than relying on stale entry-time values.
    
    Args:
        spot_price: Current underlying price
        strike: Option strike price
        dte: Days to expiry (0 = expiry day)
        risk_free_rate: Risk-free rate (default 0.065 = 6.5% India)
        option_ltp: Current option LTP
        option_type: 'CE' or 'PE'
        
    Returns:
        GreeksSnapshot or None if computation fails
    """
    try:
        from options_trader import BlackScholes, OptionType
    except ImportError:
        return None
    
    if spot_price <= 0 or strike <= 0 or option_ltp <= 0:
        return None
    
    # Time to expiry in years. For 0DTE, use fraction of day remaining.
    if dte <= 0:
        now = datetime.now()
        market_close = now.replace(hour=15, minute=30)
        remaining_hours = max(0.1, (market_close - now).total_seconds() / 3600)
        T = remaining_hours / (365 * 24)  # Fraction of year
    else:
        T = max(dte / 365, 1 / (365 * 24))  # At least 1 hour
    
    r = risk_free_rate
    opt_type = OptionType.CE if option_type.upper() == 'CE' else OptionType.PE
    
    # Compute IV from current market price
    iv = BlackScholes.implied_volatility(option_ltp, spot_price, strike, T, r, opt_type)
    if iv <= 0.01:
        iv = 0.30  # Fallback
    
    delta = BlackScholes.delta(spot_price, strike, T, r, iv, opt_type)
    gamma = BlackScholes.gamma(spot_price, strike, T, r, iv)
    theta = BlackScholes.theta(spot_price, strike, T, r, iv, opt_type)
    vega = BlackScholes.vega(spot_price, strike, T, r, iv)
    
    moneyness = classify_moneyness(delta)
    
    return GreeksSnapshot(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        iv=round(iv, 4),
        moneyness=moneyness,
        dte=max(0, dte),
    )


def get_greeks_adjusted_sl(
    entry_price: float,
    greeks: GreeksSnapshot,
    score_tier: str = "standard",
) -> float:
    """
    Compute dynamic stop loss based on current Greeks.
    
    Logic:
      1. Look up base SL% from moneyness table
      2. Apply gamma modifier (high gamma → tighter SL)
      3. Apply DTE modifier (0DTE → tighter SL via expiry_shield)
      
    Args:
        entry_price: Option entry premium
        greeks: Current GreeksSnapshot
        score_tier: 'premium', 'standard', or 'base'
        
    Returns:
        Stop loss price (for BUY positions, this is below entry)
    """
    table = GREEKS_SL_TARGET_TABLE.get(greeks.moneyness, GREEKS_SL_TARGET_TABLE[Moneyness.ATM])
    base_sl_pct = table['sl_pct']
    
    # Score tier adjustment: premium tier gets slightly wider SL (more conviction)
    if score_tier == "premium":
        base_sl_pct += 3  # +3% wider for premium setups
    elif score_tier == "base":
        base_sl_pct -= 2  # -2% tighter for base setups
    
    # Gamma modifier: high gamma = SL can be hit faster by noise
    # BUT also means the option can recover faster → net: tighten slightly
    if greeks.gamma > 0.05:
        gamma_factor = max(0.80, 1.0 - (greeks.gamma - 0.05) * 2)  # Up to 20% tighter
        base_sl_pct *= gamma_factor
    
    # Compute SL price
    sl_price = entry_price * (1 - base_sl_pct / 100)
    return round(max(0.05, sl_price), 2)


def get_greeks_adjusted_target(
    entry_price: float,
    greeks: GreeksSnapshot,
    score_tier: str = "standard",
) -> float:
    """
    Compute dynamic target based on current Greeks.
    
    Logic:
      1. Look up base target% from moneyness table
      2. Premium setups get higher targets
      
    Returns:
        Target price (for BUY positions, this is above entry)
    """
    table = GREEKS_SL_TARGET_TABLE.get(greeks.moneyness, GREEKS_SL_TARGET_TABLE[Moneyness.ATM])
    base_target_pct = table['target_pct']
    
    if score_tier == "premium":
        base_target_pct += 20  # Premium setups: let winners run more
    elif score_tier == "base":
        base_target_pct -= 10
    
    target_price = entry_price * (1 + base_target_pct / 100)
    return round(target_price, 2)


def should_exit_on_greeks(
    greeks: GreeksSnapshot,
    current_ltp: float,
    entry_price: float,
    candles_since_entry: int = 0,
) -> Tuple[bool, str]:
    """
    Check if Greeks-based exit signals fire.
    
    Exits:
      1. DELTA_COLLAPSE: delta < 0.08 → option is nearly worthless, no recovery possible
      2. THETA_BLEED: theta eating > 3% of LTP per hour and trade is flat/losing
      3. SPEED_GATE: moneyness-adjusted speed gate (OTM gets checked faster)
      
    Returns:
        (should_exit: bool, reason: str)
    """
    abs_delta = abs(greeks.delta)
    
    # 1. Delta collapse — option is dying
    if abs_delta < DELTA_COLLAPSE_THRESHOLD and greeks.dte <= DELTA_COLLAPSE_MAX_DTE:
        premium_pct = (current_ltp - entry_price) / entry_price * 100 if entry_price > 0 else 0
        # Only if losing money or flat (don't exit a winner with low delta)
        if premium_pct < 10:
            return True, (
                f"GREEKS EXIT — Delta collapse: δ={greeks.delta:.3f} < {DELTA_COLLAPSE_THRESHOLD} "
                f"({greeks.moneyness.value}, {greeks.dte}DTE). Option nearly worthless."
            )
    
    # 2. Theta bleed guard — time decay eating premium and trade isn't working
    if greeks.theta != 0 and current_ltp > 0:
        # Theta is per day. Convert to per hour (market hours = 6.25h)
        theta_per_hour = abs(greeks.theta) / 6.25
        theta_pct_per_hour = (theta_per_hour / current_ltp) * 100
        
        premium_pct = (current_ltp - entry_price) / entry_price * 100 if entry_price > 0 else 0
        
        # Exit if: theta > 3%/hour AND trade is losing AND been in > 4 candles (20 min)
        if (theta_pct_per_hour > THETA_BLEED_MAX_PCT_HOUR 
            and premium_pct < 0 
            and candles_since_entry > THETA_BLEED_MIN_CANDLES):
            return True, (
                f"GREEKS EXIT — Theta bleed: {theta_pct_per_hour:.1f}%/hour "
                f"(θ=₹{greeks.theta:.2f}/day) eating premium at {premium_pct:+.1f}% P&L. "
                f"Trade flatlined, theta dominating."
            )
    
    # 3. Moneyness-based speed gate
    table = GREEKS_SL_TARGET_TABLE.get(greeks.moneyness, GREEKS_SL_TARGET_TABLE[Moneyness.ATM])
    speed_candles = table['speed_gate_candles']
    if candles_since_entry >= speed_candles:
        premium_pct = (current_ltp - entry_price) / entry_price * 100 if entry_price > 0 else 0
        if premium_pct < 2.0:  # Less than 2% gain after speed gate candles
            return True, (
                f"GREEKS EXIT — Speed gate ({greeks.moneyness.value}): "
                f"{candles_since_entry} candles, only {premium_pct:+.1f}% gain "
                f"(need > 2% by candle {speed_candles})"
            )
    
    return False, ""


def get_greeks_speed_gate_params(greeks: GreeksSnapshot) -> Tuple[int, float]:
    """
    Get moneyness-adjusted speed gate parameters.
    
    Deep ITM = patient (15 candles, low % needed)
    Deep OTM = impatient (6 candles, high % needed)
    
    Returns:
        (candles, min_pct_gain)
    """
    table = GREEKS_SL_TARGET_TABLE.get(greeks.moneyness, GREEKS_SL_TARGET_TABLE[Moneyness.ATM])
    return table['speed_gate_candles'], 2.0  # Minimum 2% gain by speed gate time


def print_greeks_context(greeks: GreeksSnapshot, entry_price: float, current_ltp: float):
    """Print Greeks context for dashboard/logs."""
    pnl_pct = (current_ltp - entry_price) / entry_price * 100 if entry_price > 0 else 0
    table = GREEKS_SL_TARGET_TABLE.get(greeks.moneyness, {})
    
    print(f"   📐 Greeks: δ={greeks.delta:+.3f} γ={greeks.gamma:.5f} "
          f"θ=₹{greeks.theta:.2f}/day ν={greeks.vega:.3f} IV={greeks.iv:.1%}")
    print(f"   📊 Moneyness: {greeks.moneyness.value} | {greeks.dte}DTE | "
          f"P&L: {pnl_pct:+.1f}%")
    if table:
        print(f"   🎯 Greeks SL: {table['sl_pct']}% | Target: {table['target_pct']}% | "
              f"SpeedGate: {table['speed_gate_candles']} candles")
