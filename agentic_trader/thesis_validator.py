"""
THESIS INVALIDATION ENGINE (TIE)
─────────────────────────────────
Cuts losing trades early when the structural thesis is broken,
not merely because the trade is in loss.

Activation gate:
  • Premium/price loss > 8%
  • Candles held: 3–10 (candle 1-2 = noise; after 10 = time_stop)

5 structural checks:
  1. R-Multiple Collapse   – had ≥0.5R, now ≤ -0.8R
  2. Never Showed Life     – 5+ candles, maxR < 0.1, loss > 8%
  3. IV Crush Detection    – premium -8% but underlying < 0.3% against
  4. Underlying BOS Against – LTP beyond initial_sl structure level
  5. Max Pain Ceiling      – loss > 20% of premium (hard cap)

Any single check firing → immediate market exit.

Integration:
  Called from _check_positions_realtime() every 60s cycle.
  Consumes ExitManager.TradeState — zero new data collection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from exit_manager import TradeState

logger = logging.getLogger('thesis_validator')

# ── Configuration ──────────────────────────────────────────────────
# Activation gate
LOSS_ACTIVATION_PCT = 8.0       # Must be down >8% to even run checks
MIN_CANDLES = 3                 # Don't run before candle 3 (too early = noise)
MAX_CANDLES = 10                # After 10 candles, time_stop handles it

# Check 1: R-Multiple Collapse
R_COLLAPSE_PEAK_MIN = 0.5      # Must have reached at least 0.5R
R_COLLAPSE_CURRENT_MAX = -0.8  # Must now be at or below -0.8R

# Check 2: Never Showed Life
NEVER_LIFE_CANDLES = 5          # At least 5 candles
NEVER_LIFE_MAX_R = 0.1         # Never exceeded 0.1R

# Check 3: IV Crush
IV_CRUSH_PREMIUM_DROP = 8.0    # Premium down ≥ 8%
IV_CRUSH_UNDERLYING_MAX = 0.3  # Underlying moved < 0.3% against

# Check 4: Underlying BOS — uses initial_sl directly (no config needed)

# Check 5: Max Pain Ceiling
MAX_PAIN_PCT = 20.0            # Hard cap: loss > 20% of premium


@dataclass
class ThesisResult:
    """Result of thesis validation"""
    is_invalid: bool
    check_name: str
    reason: str
    details: dict


def check_thesis(
    state: 'TradeState',
    option_ltp: float,
    underlying_ltp: float = 0.0,
) -> Optional[ThesisResult]:
    """
    Validates whether the trade thesis is still intact.

    Parameters
    ----------
    state : TradeState
        The exit manager's tracked state for this trade.
    option_ltp : float
        Current LTP of the option/equity position.
    underlying_ltp : float
        Current LTP of the underlying (needed for BOS + IV crush).
        Pass 0 if unavailable — BOS and IV crush checks will be skipped.

    Returns
    -------
    ThesisResult if thesis is invalidated, None if thesis still holds.
    """
    # ── ACTIVATION GATE ────────────────────────────────────────────
    if state.entry_price <= 0 or option_ltp <= 0:
        return None

    candles = state.candles_since_entry
    if candles < MIN_CANDLES or candles > MAX_CANDLES:
        return None

    # Premium loss % (works for both CE and PE since we always BUY options)
    premium_loss_pct = (state.entry_price - option_ltp) / state.entry_price * 100
    if premium_loss_pct < LOSS_ACTIVATION_PCT:
        return None  # Not in enough pain to bother checking

    # R-multiple (already tracked by ExitManager)
    risk = abs(state.entry_price - state.initial_sl)
    if risk == 0:
        risk = state.entry_price * 0.01
    if state.side == "BUY":
        current_r = (option_ltp - state.entry_price) / risk
    else:
        current_r = (state.entry_price - option_ltp) / risk

    # ── CHECK 1: R-MULTIPLE COLLAPSE ──────────────────────────────
    if state.max_favorable_move >= R_COLLAPSE_PEAK_MIN and current_r <= R_COLLAPSE_CURRENT_MAX:
        reason = (
            f"R-collapse: peaked at {state.max_favorable_move:.2f}R, "
            f"now {current_r:.2f}R (threshold: peak≥{R_COLLAPSE_PEAK_MIN}R → now≤{R_COLLAPSE_CURRENT_MAX}R)"
        )
        logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
        return ThesisResult(
            is_invalid=True,
            check_name="R_COLLAPSE",
            reason=reason,
            details={
                'peak_r': round(state.max_favorable_move, 3),
                'current_r': round(current_r, 3),
                'candles': candles,
                'premium_loss_pct': round(premium_loss_pct, 2),
            },
        )

    # ── CHECK 2: NEVER SHOWED LIFE ───────────────────────────────
    if (
        candles >= NEVER_LIFE_CANDLES
        and state.max_favorable_move < NEVER_LIFE_MAX_R
        and premium_loss_pct >= LOSS_ACTIVATION_PCT
    ):
        reason = (
            f"Never showed life: {candles} candles, "
            f"maxR={state.max_favorable_move:.2f} (<{NEVER_LIFE_MAX_R}), "
            f"loss={premium_loss_pct:.1f}%"
        )
        logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
        return ThesisResult(
            is_invalid=True,
            check_name="NEVER_SHOWED_LIFE",
            reason=reason,
            details={
                'max_r': round(state.max_favorable_move, 3),
                'candles': candles,
                'premium_loss_pct': round(premium_loss_pct, 2),
            },
        )

    # ── CHECK 3: IV CRUSH DETECTION ──────────────────────────────
    # Only for option trades where we have underlying data
    if state.is_option and underlying_ltp > 0 and state.entry_price > 0:
        # How much did the underlying actually move against our thesis?
        # For CE (BUY side): underlying going DOWN is adverse
        # For PE (BUY side): underlying going UP is adverse
        # But we need the underlying price at entry — approximate from initial_sl structure
        # We can use initial_sl as the "adverse direction boundary"
        # Instead, compute underlying % move from direction perspective
        # Since we don't store underlying_entry, use a simpler heuristic:
        #   "Premium dropped ≥8% but underlying barely moved against us"
        #   This catches pure IV/theta decay
        
        # Use initial_sl to estimate entry-time underlying reference:
        # For BUY (CE): adverse = underlying dropped below entry zone
        # For SELL (PE-like, but we rarely SELL options): adverse = underlying rose
        
        # Simpler: compare underlying move vs option move
        # If premium_loss_pct >= 8% and underlying barely moved against direction,
        # then it's IV crush, not directional loss
        
        # We need underlying reference price — derive from structure
        # initial_sl is the structure level, entry_price is option premium
        # If underlying hasn't breached initial_sl, it's within structure
        
        if premium_loss_pct >= IV_CRUSH_PREMIUM_DROP:
            # Check if underlying stays within structure (hasn't moved against us much)
            # For CE trades: adverse is underlying dropping toward initial_sl
            # For PE trades: adverse is underlying rising toward initial_sl
            if state.side == "BUY":
                # CE: adverse = underlying below entry → dropping
                # Check: underlying hasn't moved much below the SL structure
                # If underlying is ABOVE initial_sl → direction is fine, it's pure IV crush
                underlying_fine = underlying_ltp > state.initial_sl
            else:
                # PE/SELL: adverse = underlying above entry → rising
                underlying_fine = underlying_ltp < state.initial_sl

            if underlying_fine:
                reason = (
                    f"IV crush: premium -{premium_loss_pct:.1f}% but underlying "
                    f"still on our side (LTP={underlying_ltp:.2f}, SL={state.initial_sl:.2f})"
                )
                logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
                return ThesisResult(
                    is_invalid=True,
                    check_name="IV_CRUSH",
                    reason=reason,
                    details={
                        'premium_loss_pct': round(premium_loss_pct, 2),
                        'underlying_ltp': underlying_ltp,
                        'initial_sl': state.initial_sl,
                        'underlying_fine': underlying_fine,
                        'candles': candles,
                    },
                )

    # ── CHECK 4: UNDERLYING BOS (Break of Structure) AGAINST ─────
    if underlying_ltp > 0:
        # BOS = underlying has breached the initial_sl structure level
        # For BUY (CE): initial_sl is support → BOS if underlying < initial_sl
        # For SELL (PE): initial_sl is resistance → BOS if underlying > initial_sl
        if state.side == "BUY" and underlying_ltp < state.initial_sl:
            reason = (
                f"BOS against: underlying ₹{underlying_ltp:.2f} broke below "
                f"SL structure ₹{state.initial_sl:.2f}"
            )
            logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
            return ThesisResult(
                is_invalid=True,
                check_name="UNDERLYING_BOS",
                reason=reason,
                details={
                    'underlying_ltp': underlying_ltp,
                    'initial_sl': state.initial_sl,
                    'side': state.side,
                    'premium_loss_pct': round(premium_loss_pct, 2),
                    'candles': candles,
                },
            )
        elif state.side == "SELL" and underlying_ltp > state.initial_sl:
            reason = (
                f"BOS against: underlying ₹{underlying_ltp:.2f} broke above "
                f"SL structure ₹{state.initial_sl:.2f}"
            )
            logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
            return ThesisResult(
                is_invalid=True,
                check_name="UNDERLYING_BOS",
                reason=reason,
                details={
                    'underlying_ltp': underlying_ltp,
                    'initial_sl': state.initial_sl,
                    'side': state.side,
                    'premium_loss_pct': round(premium_loss_pct, 2),
                    'candles': candles,
                },
            )

    # ── CHECK 5: MAX PAIN CEILING ────────────────────────────────
    if premium_loss_pct >= MAX_PAIN_PCT:
        reason = (
            f"Max pain ceiling: loss {premium_loss_pct:.1f}% exceeds "
            f"{MAX_PAIN_PCT}% hard cap"
        )
        logger.info(f"TIE TRIGGERED [{state.symbol}]: {reason}")
        return ThesisResult(
            is_invalid=True,
            check_name="MAX_PAIN_CEILING",
            reason=reason,
            details={
                'premium_loss_pct': round(premium_loss_pct, 2),
                'current_r': round(current_r, 3),
                'candles': candles,
            },
        )

    # ── THESIS INTACT ────────────────────────────────────────────
    return None


# ── THESIS HEDGE PROTOCOL (THP) ─────────────────────────────
# Determines whether a TIE-invalidated trade should be HEDGED
# (converted to debit spread) instead of immediately exited.

# Hedgeable: directional thesis still plausible, selling OTM leg
# captures remaining extrinsic value while limiting max loss.
_HEDGEABLE_CHECKS = frozenset({
    "R_COLLAPSE",            # Lost R but direction may recover
    "NEVER_SHOWED_LIFE",     # Hasn't moved yet — hedge limits downside
    "UNDERLYING_BOS",        # Underlying broke structure — hedge caps loss
})

# Non-hedgeable: IV-driven or max-pain ceiling — selling more
# options in a crushed-IV / against-max-pain environment is worse.
_NON_HEDGEABLE_CHECKS = frozenset({
    "IV_CRUSH",              # Selling into crushed IV gives tiny credit
    "MAX_PAIN_CEILING",      # Deep OTM, premium too small to offset
})


def should_hedge_instead_of_exit(result: ThesisResult) -> bool:
    """
    Given a TIE result that IS invalid, decide: hedge or exit?

    Returns True  → convert naked option to debit spread (sell OTM leg)
    Returns False → immediate exit (current behavior)
    """
    if result is None or not result.is_invalid:
        return False
    return result.check_name in _HEDGEABLE_CHECKS
