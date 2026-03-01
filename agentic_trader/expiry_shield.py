"""
Expiry Day Shield — Gamma Risk Armor for Titan

On expiry days (weekly Thu for NIFTY, monthly last Thu for stocks),
option premiums can swing 200-500% in minutes due to extreme gamma.

This module provides:
  1. is_expiry_day()    — checks if today is expiry day for a given underlying
  2. get_dte()          — days to expiry from trade dict or symbol
  3. expiry_entry_gate()— blocks/restricts new entries on expiry day
  4. expiry_exit_rules()— tightens SL and forces early exits on expiry day

Rules enforced:
  - Block ALL new option entries after 1:30 PM on expiry day
  - Block new NAKED option entries after 12:00 PM (spreads ok till 1:30)
  - Tighten SL by 40% on expiry day (28% SL → ~17% effective)
  - Force-exit all same-week expiry positions by 2:45 PM
  - Target is widened 0% (keep same) — let winners run, just protect downside
  - Tag all expiry-day entries with 'EXPIRY_DAY_TRADE' for post-analysis

Author: Titan v5.2
"""

from datetime import datetime, date, time
from typing import Dict, Optional, Tuple
import re


# ─── EXPIRY SCHEDULE ──────────────────────────────────────────────
# Post SEBI Nov 2024: Only NIFTY has weekly expiry on NSE (Tuesday)
# All stock options: Monthly expiry (last Thursday of month)
# BANKNIFTY: Monthly expiry (last Tuesday of month)

# We don't hardcode dates — we read 'expiry' from the trade position dict
# which ZerodhaTools stores as ISO string from Zerodha's instrument data.

# ─── TIME GATES (wired to EXPIRY_SHIELD_CONFIG in config.py) ─────
def _load_expiry_config():
    """Load config values with safe defaults."""
    try:
        from config import EXPIRY_SHIELD_CONFIG as _cfg
    except ImportError:
        _cfg = {}
    def _parse_time(s, default):
        try:
            parts = s.split(':')
            return time(int(parts[0]), int(parts[1]))
        except Exception:
            return default
    return {
        'no_new_naked_after': _parse_time(_cfg.get('no_new_naked_after', '12:00'), time(12, 0)),
        'no_new_any_after': _parse_time(_cfg.get('no_new_any_after', '13:30'), time(13, 30)),
        'force_exit_by': _parse_time(_cfg.get('force_exit_0dte_by', '14:45'), time(14, 45)),
        'sl_tighten_0dte': _cfg.get('sl_tighten_factor_0dte', 0.60),
        'sl_tighten_1dte': _cfg.get('sl_tighten_factor_1dte', 0.80),
        'speed_candles': _cfg.get('speed_gate_candles_0dte', 6),
        'speed_pct': _cfg.get('speed_gate_pct_0dte', 5.0),
    }

_EC = _load_expiry_config()
EXPIRY_NO_NEW_NAKED_AFTER = _EC['no_new_naked_after']
EXPIRY_NO_NEW_ANY_AFTER = _EC['no_new_any_after']
EXPIRY_FORCE_EXIT_BY = _EC['force_exit_by']
EXPIRY_SL_TIGHTEN_FACTOR = _EC['sl_tighten_0dte']
EXPIRY_SL_TIGHTEN_1DTE = _EC['sl_tighten_1dte']
EXPIRY_SPEED_GATE_CANDLES = _EC['speed_candles']
EXPIRY_SPEED_GATE_PCT = _EC['speed_pct']


def get_dte(trade: Dict) -> int:
    """
    Get Days-To-Expiry from a trade position dict.
    
    Returns:
        int: Days to expiry. 0 = expiry day. -1 if expiry unknown.
    """
    expiry_str = trade.get('expiry', '')
    if not expiry_str:
        return -1
    try:
        if isinstance(expiry_str, str):
            # Handle both date and datetime ISO formats
            expiry_date = datetime.fromisoformat(expiry_str).date()
        elif isinstance(expiry_str, date):
            expiry_date = expiry_str
        else:
            return -1
        return (expiry_date - date.today()).days
    except (ValueError, TypeError):
        return -1


def is_expiry_day(trade: Dict) -> bool:
    """Check if today is the expiry day for this trade's option contract."""
    return get_dte(trade) == 0


def is_near_expiry(trade: Dict, max_dte: int = 1) -> bool:
    """Check if this trade expires within max_dte days (0 or 1)."""
    dte = get_dte(trade)
    return 0 <= dte <= max_dte


def expiry_entry_gate(trade_type: str, is_spread: bool, expiry_str: str = '') -> Tuple[bool, str]:
    """
    Gate that blocks/restricts new option entries on expiry day.
    
    Args:
        trade_type: 'naked_option', 'debit_spread', 'credit_spread', 'iron_condor'
        is_spread: True if any multi-leg strategy
        expiry_str: ISO date string of the option's expiry
        
    Returns:
        (allowed: bool, reason: str)
    """
    # If we can't determine expiry, allow (fail-open)
    if not expiry_str:
        return True, ""
    
    try:
        expiry_date = datetime.fromisoformat(expiry_str).date()
    except (ValueError, TypeError):
        return True, ""
    
    dte = (expiry_date - date.today()).days
    
    # Only apply restrictions for 0DTE (expiry day itself)
    if dte != 0:
        return True, ""
    
    now = datetime.now().time()
    
    # Rule 1: No trades at all after 1:30 PM on expiry day
    if now >= EXPIRY_NO_NEW_ANY_AFTER:
        return False, (
            f"EXPIRY SHIELD: Blocked — no new entries after {EXPIRY_NO_NEW_ANY_AFTER.strftime('%H:%M')} "
            f"on expiry day (extreme gamma risk)"
        )
    
    # Rule 2: No new NAKED options after 12:00 PM (spreads still ok)
    if not is_spread and now >= EXPIRY_NO_NEW_NAKED_AFTER:
        return False, (
            f"EXPIRY SHIELD: Blocked naked option after {EXPIRY_NO_NEW_NAKED_AFTER.strftime('%H:%M')} "
            f"on expiry day (only spreads allowed). trade_type={trade_type}"
        )
    
    # Allowed but tagged
    return True, "EXPIRY_DAY_TRADE"


def get_expiry_adjusted_sl(
    entry_price: float, 
    normal_sl: float, 
    dte: int,
    delta: float = 0.0,
) -> float:
    """
    Tighten stop loss on expiry day to account for gamma risk.
    
    On 0DTE:
      - SL distance is reduced by 40% (EXPIRY_SL_TIGHTEN_FACTOR = 0.60)
      - This means: normal 28% SL → ~17% SL
    On 1DTE:
      - SL distance is reduced by 20% 
    Otherwise: no change.
    
    Args:
        entry_price: Option entry premium
        normal_sl: Normal stop loss price
        dte: Days to expiry
        delta: Option delta (absolute value) — high delta = less tightening needed
        
    Returns:
        Adjusted stop loss price
    """
    if dte < 0 or dte > 1:
        return normal_sl
    
    # Calculate SL distance from entry
    sl_distance = abs(entry_price - normal_sl)
    
    if dte == 0:
        factor = EXPIRY_SL_TIGHTEN_FACTOR  # 0.60 — tighten by 40%
    else:  # dte == 1
        factor = EXPIRY_SL_TIGHTEN_1DTE  # 0.80 — tighten by 20% on day before expiry
    
    # High delta options (deep ITM, delta > 0.7) need less tightening
    # because they behave more like the underlying (lower gamma)
    abs_delta = abs(delta) if delta else 0
    if abs_delta > 0.7:
        factor = min(1.0, factor + 0.15)  # Relax tightening for deep ITM
    
    adjusted_distance = sl_distance * factor
    
    # For BUY positions: SL is below entry
    if normal_sl < entry_price:
        return entry_price - adjusted_distance
    else:
        # For SELL positions: SL is above entry
        return entry_price + adjusted_distance


def should_force_exit_expiry(trade: Dict) -> Tuple[bool, str]:
    """
    Check if a 0DTE position should be force-exited.
    
    Force exit at 2:45 PM on expiry day.
    
    Returns:
        (should_exit: bool, reason: str)
    """
    if not is_expiry_day(trade):
        return False, ""
    
    now = datetime.now().time()
    if now >= EXPIRY_FORCE_EXIT_BY:
        return True, (
            f"EXPIRY SHIELD: Force exit — 0DTE position at "
            f"{now.strftime('%H:%M')} past {EXPIRY_FORCE_EXIT_BY.strftime('%H:%M')} cutoff"
        )
    
    return False, ""


def get_expiry_speed_gate(dte: int) -> Tuple[int, float]:
    """
    On expiry day, use aggressive speed gate parameters.
    
    Normal: 12 candles, +3% gain needed
    Expiry: 6 candles, +5% gain needed (faster decision on 0DTE)
    
    Returns:
        (candles, min_pct_gain)
    """
    if dte == 0:
        return EXPIRY_SPEED_GATE_CANDLES, EXPIRY_SPEED_GATE_PCT
    elif dte == 1:
        return 9, 4.0  # Slightly more aggressive on day before expiry
    return 12, 3.0  # Normal


def format_expiry_tag(dte: int) -> str:
    """Return a human-readable tag for position logs."""
    if dte == 0:
        return "⚡0DTE"
    elif dte == 1:
        return "⏳1DTE"
    elif dte <= 3:
        return f"📅{dte}DTE"
    return ""


# ─── SUMMARY PRINT ───────────────────────────────────────────────
def print_expiry_shield_status():
    """Print current expiry shield status for dashboard."""
    now = datetime.now().time()
    print(f"\n🛡️ EXPIRY SHIELD STATUS ({datetime.now().strftime('%H:%M')})")
    print(f"   Naked block after: {EXPIRY_NO_NEW_NAKED_AFTER.strftime('%H:%M')} {'🔴 ACTIVE' if now >= EXPIRY_NO_NEW_NAKED_AFTER else '🟢 Open'}")
    print(f"   All-entry block:   {EXPIRY_NO_NEW_ANY_AFTER.strftime('%H:%M')} {'🔴 ACTIVE' if now >= EXPIRY_NO_NEW_ANY_AFTER else '🟢 Open'}")
    print(f"   Force-exit 0DTE:   {EXPIRY_FORCE_EXIT_BY.strftime('%H:%M')} {'🔴 ACTIVE' if now >= EXPIRY_FORCE_EXIT_BY else '⏳ Pending'}")
    print(f"   SL tighten factor: {EXPIRY_SL_TIGHTEN_FACTOR:.0%} of normal")
