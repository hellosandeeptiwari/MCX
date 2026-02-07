"""
TREND FOLLOWING (MOMENTUM) STRATEGY MODULE FOR TITAN
=====================================================

"Price escapes balance and does not come back."

This is TITAN's REAL EDGE - catching strong directional moves and riding them.

CORE PHILOSOPHY:
- Don't predict, FOLLOW
- Trend is your friend until it bends
- Add to winners, cut losers fast
- Let profits run with trailing stops

TECHNICAL CHARACTERISTICS FOR TREND ENTRY:
1. VWAP slope STRONG (price pulling away from VWAP)
2. EMAs EXPANDING (9 EMA > 21 EMA widening for bullish)
3. Volume HIGH/EXPLOSIVE (institutional participation)
4. ORB break HOLDS (not a false breakout)
5. Pullbacks are SHALLOW and FAST (strong buyers/sellers)

ENTRY TYPES:
- INITIAL_BREAK: Fresh breakout with conviction
- PULLBACK_ENTRY: Buy/short the first shallow pullback
- ADD_TO_WINNER: Scale in on continued strength
- CONTINUATION: Re-entry after brief consolidation

EXIT TYPES:
- TRAILING_STOP: Dynamic stop that locks in profits
- TREND_REVERSAL: Opposite signal detected
- MOMENTUM_FADE: Volume/velocity declining
- TIME_STOP: Trend didn't develop as expected
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TrendState(Enum):
    """Current trend state"""
    STRONG_BULLISH = "STRONG_BULLISH"   # Score 80+
    BULLISH = "BULLISH"                  # Score 60-79
    NEUTRAL = "NEUTRAL"                  # Score 40-59
    BEARISH = "BEARISH"                  # Score 20-39
    STRONG_BEARISH = "STRONG_BEARISH"   # Score 0-19


class TrendEntryType(Enum):
    """Type of trend entry"""
    INITIAL_BREAK = "INITIAL_BREAK"     # Fresh breakout
    PULLBACK_ENTRY = "PULLBACK_ENTRY"   # Pullback to EMA
    ADD_TO_WINNER = "ADD_TO_WINNER"     # Scale in
    CONTINUATION = "CONTINUATION"        # Re-entry after pause


class TrendExitType(Enum):
    """Type of trend exit"""
    TRAILING_STOP = "TRAILING_STOP"
    TREND_REVERSAL = "TREND_REVERSAL"
    MOMENTUM_FADE = "MOMENTUM_FADE"
    TIME_STOP = "TIME_STOP"
    TARGET_HIT = "TARGET_HIT"


@dataclass
class TrendSignal:
    """Raw trend signal data from market"""
    symbol: str
    
    # Price data
    ltp: float
    open: float
    high: float
    low: float
    
    # VWAP data (CRITICAL for trend following)
    vwap: float
    vwap_slope: float           # % change in VWAP over last 15 candles
    vwap_distance_pct: float    # How far price is from VWAP (%)
    
    # EMA data (9 and 21 are standard for intraday)
    ema_9: float
    ema_21: float
    ema_spread_pct: float       # (EMA9 - EMA21) / EMA21 * 100
    ema_expanding: bool         # Is spread increasing?
    
    # Volume data (CONFIRMS trend)
    volume_regime: str          # EXPLOSIVE, HIGH, NORMAL, LOW
    volume_ratio: float         # Current vs 20-period average
    
    # ORB data (VALIDATES trend)
    orb_high: float
    orb_low: float
    orb_broken: str             # UP, DOWN, INSIDE
    orb_hold_candles: int       # How many candles held after break
    
    # Pullback data (ENTRY timing)
    pullback_depth_pct: float   # Current pullback from recent high/low
    pullback_candles: int       # How many candles in pullback
    
    # Momentum indicators
    rsi_14: float = 50.0
    adx: float = 20.0           # Trend strength (>25 = strong trend)
    
    # Higher timeframe alignment
    htf_trend: str = "NEUTRAL"  # 15min/hourly trend direction


@dataclass
class TrendDecision:
    """Output from trend analysis"""
    trend_state: TrendState
    trend_score: float              # 0-100 score
    
    should_enter: bool
    entry_type: Optional[TrendEntryType]
    entry_direction: str            # BUY, SELL, HOLD
    
    confidence: str                 # HIGH, MEDIUM, LOW
    position_size_multiplier: float # 0.5 to 2.0
    
    # Key levels
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    trailing_stop_pct: float
    
    # Scoring breakdown
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TrendFollowingEngine:
    """
    TREND FOLLOWING ENGINE
    
    Scores trend strength and determines entry/exit signals.
    
    SCORING WEIGHTS (100 points total):
    ┌─────────────────────┬────────┬────────────────────────────────┐
    │ Factor              │ Points │ Why It Matters                 │
    ├─────────────────────┼────────┼────────────────────────────────┤
    │ VWAP Slope          │   25   │ Shows institutional direction  │
    │ EMA Expansion       │   20   │ Confirms trend acceleration    │
    │ Volume Regime       │   20   │ Validates participation        │
    │ ORB Break Hold      │   15   │ Confirms momentum, not fake    │
    │ Pullback Quality    │   10   │ Shows strength of continuation │
    │ ADX Strength        │   10   │ Measures trend conviction      │
    └─────────────────────┴────────┴────────────────────────────────┘
    
    THRESHOLDS:
    - 80+ = STRONG trend, aggressive entry
    - 60+ = BULLISH/BEARISH, standard entry
    - 40-59 = NEUTRAL, wait for better setup
    - <40 = Against trend or no trend
    """
    
    # Scoring weights
    WEIGHTS = {
        'vwap_slope': 25,        # HIGHEST - shows where big money is going
        'ema_expansion': 20,     # Shows trend is accelerating
        'volume': 20,            # Confirms real participation
        'orb_hold': 15,          # Validates breakout quality
        'pullback': 10,          # Shows continuation quality
        'adx': 10,               # Overall trend strength
    }
    
    # Thresholds for trend state (entry)
    STRONG_THRESHOLD = 80
    TREND_THRESHOLD = 60
    NEUTRAL_THRESHOLD = 40
    
    # === ASYMMETRIC HYSTERESIS THRESHOLDS ===
    # Upgrades are HARDER than downgrades (STRONG is a "rare badge")
    UPGRADE_TO_STRONG = 82          # Need 82+ to upgrade to STRONG (not 80)
    STAY_STRONG_THRESHOLD = 70      # Stay STRONG if >= 70
    DOWNGRADE_FROM_STRONG = 70      # Drop to BULLISH if < 70
    STAY_TREND_THRESHOLD = 50       # Stay BULLISH/BEARISH if >= 50
    DOWNGRADE_FROM_TREND = 50       # Drop to NEUTRAL if < 50
    
    # === TIME-IN-STATE CONFIRMATION (micro noise killer) ===
    # Upgrades require consecutive candles at threshold
    UPGRADE_CONFIRMATION_CANDLES = 2    # Need 2 candles >= 82 to upgrade to STRONG
    DOWNGRADE_CONFIRMATION_CANDLES = 1  # Downgrades are immediate (1 candle)
    
    # === SHOCK OVERRIDE THRESHOLDS (tail-risk protection) ===
    SHOCK_ATR_MULTIPLIER = 2.0      # Force downgrade if move > 2x ATR against position
    SHOCK_VWAP_CROSS_OVERRIDE = True # Force downgrade if VWAP crossed with high volume
    
    # Pullback parameters
    MAX_PULLBACK_PCT = 0.5      # Max 0.5% pullback for "shallow"
    MAX_PULLBACK_CANDLES = 3    # Max 3 candles for "fast" pullback
    
    def __init__(self):
        self.active_trends: Dict[str, TrendDecision] = {}
        self.trend_history: Dict[str, List[TrendDecision]] = {}
        # Track previous states for hysteresis
        self.previous_states: Dict[str, TrendState] = {}
        # Track consecutive scores for time-in-state confirmation
        self.consecutive_scores: Dict[str, List[float]] = {}
        # Track entry prices for shock detection
        self.entry_prices: Dict[str, float] = {}
        self.entry_directions: Dict[str, str] = {}
        # Track last trading day for context-aware reset
        self.last_trading_day: Dict[str, str] = {}
        print("📈 Trend Following Engine: INITIALIZED (with enhanced hysteresis)")
    
    def _check_shock_override(self, symbol: str, signal: 'TrendSignal', 
                               prev_state: TrendState) -> Tuple[bool, str]:
        """
        Check for violent reversal that should force immediate downgrade.
        
        Shock conditions:
        1. Price crosses VWAP against position + HIGH/EXPLOSIVE volume
        2. Move exceeds 2x ATR against position
        
        Returns: (should_override, reason)
        """
        if prev_state == TrendState.NEUTRAL:
            return False, ""
        
        entry_price = self.entry_prices.get(symbol, signal.ltp)
        entry_dir = self.entry_directions.get(symbol, "LONG")
        is_long = prev_state in [TrendState.STRONG_BULLISH, TrendState.BULLISH]
        
        # === SHOCK 1: VWAP Cross Against Position + High Volume ===
        if self.SHOCK_VWAP_CROSS_OVERRIDE:
            vwap_crossed_against = False
            if is_long and signal.ltp < signal.vwap:
                # Long position but price fell below VWAP
                vwap_crossed_against = True
            elif not is_long and signal.ltp > signal.vwap:
                # Short position but price rose above VWAP
                vwap_crossed_against = True
            
            if vwap_crossed_against and signal.volume_regime in ["HIGH", "EXPLOSIVE"]:
                return True, f"⚡ SHOCK: VWAP crossed against position with {signal.volume_regime} volume"
        
        # === SHOCK 2: Large ATR Move Against Position ===
        # Estimate ATR from high-low range (simplified)
        estimated_atr = (signal.high - signal.low)
        if estimated_atr > 0:
            move_against = 0
            if is_long:
                move_against = entry_price - signal.ltp  # Negative for profit
            else:
                move_against = signal.ltp - entry_price  # Negative for profit
            
            if move_against > 0:  # Only if losing
                atr_multiple = move_against / estimated_atr
                if atr_multiple >= self.SHOCK_ATR_MULTIPLIER:
                    return True, f"⚡ SHOCK: Price moved {atr_multiple:.1f}x ATR against position"
        
        return False, ""
    
    def _apply_hysteresis(self, symbol: str, raw_score: float, 
                           bullish_points: float, bearish_points: float,
                           signal: 'TrendSignal' = None) -> TrendState:
        """
        Apply enhanced hysteresis with:
        1. Asymmetric thresholds (upgrades harder than downgrades)
        2. Time-in-state confirmation (2 candles for upgrades)
        3. Shock override (force downgrade on violent reversals)
        
        UPGRADE requires: score >= 82 for 2 consecutive candles
        STAY STRONG requires: score >= 70
        DOWNGRADE is immediate if: score < 70
        """
        prev_state = self.previous_states.get(symbol, TrendState.NEUTRAL)
        is_bullish_bias = bullish_points > bearish_points
        
        # === CONTEXT-AWARE RESET: Check for new trading day ===
        today = datetime.now().strftime("%Y-%m-%d")
        if symbol in self.last_trading_day and self.last_trading_day[symbol] != today:
            # New day - reset state
            self._context_reset(symbol, "NEW_DAY")
            prev_state = TrendState.NEUTRAL
        self.last_trading_day[symbol] = today
        
        # === SHOCK OVERRIDE CHECK ===
        if signal and prev_state != TrendState.NEUTRAL:
            shock_triggered, shock_reason = self._check_shock_override(symbol, signal, prev_state)
            if shock_triggered:
                print(f"   {shock_reason}")
                # Force immediate downgrade
                if prev_state in [TrendState.STRONG_BULLISH, TrendState.STRONG_BEARISH]:
                    new_state = TrendState.BULLISH if is_bullish_bias else TrendState.BEARISH
                else:
                    new_state = TrendState.NEUTRAL
                self.previous_states[symbol] = new_state
                self.consecutive_scores[symbol] = []  # Reset consecutive tracking
                return new_state
        
        # === TRACK CONSECUTIVE SCORES FOR TIME-IN-STATE ===
        if symbol not in self.consecutive_scores:
            self.consecutive_scores[symbol] = []
        self.consecutive_scores[symbol].append(raw_score)
        # Keep only last N scores
        max_history = max(self.UPGRADE_CONFIRMATION_CANDLES, self.DOWNGRADE_CONFIRMATION_CANDLES) + 1
        self.consecutive_scores[symbol] = self.consecutive_scores[symbol][-max_history:]
        
        # === DETERMINE RAW STATE ===
        if raw_score >= self.STRONG_THRESHOLD:
            raw_state = TrendState.STRONG_BULLISH if is_bullish_bias else TrendState.STRONG_BEARISH
        elif raw_score >= self.TREND_THRESHOLD:
            raw_state = TrendState.BULLISH if is_bullish_bias else TrendState.BEARISH
        else:
            raw_state = TrendState.NEUTRAL
        
        # === APPLY ASYMMETRIC HYSTERESIS ===
        final_state = raw_state
        
        # Check if upgrade to STRONG is requested
        wants_upgrade_to_strong = (
            prev_state in [TrendState.BULLISH, TrendState.BEARISH, TrendState.NEUTRAL] and
            raw_score >= self.UPGRADE_TO_STRONG
        )
        
        if wants_upgrade_to_strong:
            # Need N consecutive candles at upgrade threshold
            recent_scores = self.consecutive_scores[symbol][-self.UPGRADE_CONFIRMATION_CANDLES:]
            all_above_upgrade = all(s >= self.UPGRADE_TO_STRONG for s in recent_scores)
            
            if len(recent_scores) >= self.UPGRADE_CONFIRMATION_CANDLES and all_above_upgrade:
                # Confirmed upgrade!
                final_state = TrendState.STRONG_BULLISH if is_bullish_bias else TrendState.STRONG_BEARISH
                print(f"   🔥 UPGRADE CONFIRMED: {self.UPGRADE_CONFIRMATION_CANDLES} candles >= {self.UPGRADE_TO_STRONG}")
            else:
                # Not enough confirmation - stay at current level
                final_state = prev_state if prev_state != TrendState.NEUTRAL else raw_state
                candles_so_far = sum(1 for s in recent_scores if s >= self.UPGRADE_TO_STRONG)
                print(f"   ⏳ Upgrade pending: {candles_so_far}/{self.UPGRADE_CONFIRMATION_CANDLES} candles")
        
        # If previously STRONG, apply stay/downgrade logic
        elif prev_state in [TrendState.STRONG_BULLISH, TrendState.STRONG_BEARISH]:
            if raw_score >= self.STAY_STRONG_THRESHOLD:
                # Keep STRONG state
                final_state = prev_state
            elif raw_score >= self.STAY_TREND_THRESHOLD:
                # Downgrade to CONFIRMED
                final_state = TrendState.BULLISH if prev_state == TrendState.STRONG_BULLISH else TrendState.BEARISH
                print(f"   📉 Downgrade: STRONG → BULLISH (score {raw_score:.0f} < {self.STAY_STRONG_THRESHOLD})")
            else:
                # Downgrade to NEUTRAL
                final_state = TrendState.NEUTRAL
                print(f"   📉 Downgrade: STRONG → NEUTRAL (score {raw_score:.0f} < {self.STAY_TREND_THRESHOLD})")
        
        # If previously CONFIRMED, apply stay/downgrade logic
        elif prev_state in [TrendState.BULLISH, TrendState.BEARISH]:
            if raw_score >= self.STAY_TREND_THRESHOLD:
                # Keep CONFIRMED or check for upgrade
                if raw_score >= self.UPGRADE_TO_STRONG:
                    # Wants to upgrade but needs confirmation (handled above)
                    final_state = prev_state
                else:
                    final_state = prev_state
            else:
                # Downgrade to NEUTRAL
                final_state = TrendState.NEUTRAL
                print(f"   📉 Downgrade: BULLISH → NEUTRAL (score {raw_score:.0f} < {self.STAY_TREND_THRESHOLD})")
        
        # === UPDATE ENTRY TRACKING ===
        if final_state in [TrendState.STRONG_BULLISH, TrendState.BULLISH] and prev_state == TrendState.NEUTRAL:
            self.entry_prices[symbol] = signal.ltp if signal else 0
            self.entry_directions[symbol] = "LONG"
        elif final_state in [TrendState.STRONG_BEARISH, TrendState.BEARISH] and prev_state == TrendState.NEUTRAL:
            self.entry_prices[symbol] = signal.ltp if signal else 0
            self.entry_directions[symbol] = "SHORT"
        
        # Store for next iteration
        self.previous_states[symbol] = final_state
        
        return final_state
    
    def _context_reset(self, symbol: str, reason: str):
        """Internal context-aware reset"""
        if symbol in self.previous_states:
            del self.previous_states[symbol]
        if symbol in self.consecutive_scores:
            del self.consecutive_scores[symbol]
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        if symbol in self.entry_directions:
            del self.entry_directions[symbol]
        print(f"   🔄 Hysteresis reset ({reason}): {symbol}")
    
    def _get_raw_state(self, score: float, bullish_points: float, 
                       bearish_points: float) -> TrendState:
        """Get raw trend state without hysteresis (for logging)"""
        is_bullish = bullish_points > bearish_points
        if score >= self.STRONG_THRESHOLD:
            return TrendState.STRONG_BULLISH if is_bullish else TrendState.STRONG_BEARISH
        elif score >= self.TREND_THRESHOLD:
            return TrendState.BULLISH if is_bullish else TrendState.BEARISH
        return TrendState.NEUTRAL
    
    def reset_hysteresis(self, symbol: str = None, reason: str = "MANUAL"):
        """
        Context-aware hysteresis reset.
        
        Call this on:
        1. Position closed (POSITION_CLOSED)
        2. New trading day (NEW_DAY) - handled internally
        3. Symbol halted / Data Health Gate (DATA_HALT)
        4. Reconciliation mismatch (RECONCILIATION)
        5. Recovery mode triggered (RECOVERY_MODE)
        
        Args:
            symbol: Symbol to reset, or None for all
            reason: Reset reason for logging
        """
        if symbol:
            self._context_reset(symbol, reason)
        else:
            # Reset all symbols
            symbols = list(self.previous_states.keys())
            for sym in symbols:
                self._context_reset(sym, reason)
            print(f"🔄 Hysteresis reset for ALL symbols ({reason})")
    
    def analyze_trend(self, signal: TrendSignal) -> TrendDecision:
        """
        Analyze market data and produce trend following decision
        """
        score = 0
        reasons = []
        warnings = []
        breakdown = {}
        
        # Direction tracking
        bullish_points = 0
        bearish_points = 0
        
        # === 1. VWAP SLOPE (25 points) ===
        # Strong VWAP slope shows institutional direction
        vwap_score = self._score_vwap_slope(signal)
        score += vwap_score['score']
        breakdown['vwap_slope'] = vwap_score['score']
        if vwap_score['direction'] == 'BULLISH':
            bullish_points += vwap_score['score']
            reasons.append(f"📈 VWAP slope STRONG bullish: +{signal.vwap_slope:.2f}% (+{vwap_score['score']:.0f})")
        elif vwap_score['direction'] == 'BEARISH':
            bearish_points += vwap_score['score']
            reasons.append(f"📉 VWAP slope STRONG bearish: {signal.vwap_slope:.2f}% (+{vwap_score['score']:.0f})")
        else:
            warnings.append(f"⚠️ VWAP slope weak ({signal.vwap_slope:.2f}%)")
        
        # === 2. EMA EXPANSION (20 points) ===
        # Expanding EMAs confirm trend acceleration
        ema_score = self._score_ema_expansion(signal)
        score += ema_score['score']
        breakdown['ema_expansion'] = ema_score['score']
        if signal.ema_expanding:
            if signal.ema_spread_pct > 0:
                bullish_points += ema_score['score']
                reasons.append(f"📊 EMAs EXPANDING bullish: spread +{signal.ema_spread_pct:.2f}% (+{ema_score['score']:.0f})")
            else:
                bearish_points += ema_score['score']
                reasons.append(f"📊 EMAs EXPANDING bearish: spread {signal.ema_spread_pct:.2f}% (+{ema_score['score']:.0f})")
        else:
            if abs(signal.ema_spread_pct) < 0.1:
                warnings.append("⚠️ EMAs compressed - breakout imminent?")
            else:
                warnings.append("⚠️ EMAs contracting - trend weakening")
        
        # === 3. VOLUME REGIME (20 points) ===
        # Volume confirms institutional participation
        volume_score = self._score_volume(signal)
        score += volume_score
        breakdown['volume'] = volume_score
        if signal.volume_regime == "EXPLOSIVE":
            reasons.append(f"💥 EXPLOSIVE volume ({signal.volume_ratio:.1f}x avg) (+{volume_score:.0f})")
        elif signal.volume_regime == "HIGH":
            reasons.append(f"📊 HIGH volume ({signal.volume_ratio:.1f}x avg) (+{volume_score:.0f})")
        elif signal.volume_regime == "LOW":
            warnings.append(f"⚠️ LOW volume - trend may not hold (+{volume_score:.0f})")
        
        # === 4. ORB BREAK HOLD (15 points) ===
        # ORB break that HOLDS confirms trend validity
        orb_score = self._score_orb_hold(signal)
        score += orb_score['score']
        breakdown['orb_hold'] = orb_score['score']
        if signal.orb_broken == "UP" and signal.orb_hold_candles >= 2:
            bullish_points += orb_score['score']
            reasons.append(f"🚀 ORB UP held {signal.orb_hold_candles} candles (+{orb_score['score']:.0f})")
        elif signal.orb_broken == "DOWN" and signal.orb_hold_candles >= 2:
            bearish_points += orb_score['score']
            reasons.append(f"📉 ORB DOWN held {signal.orb_hold_candles} candles (+{orb_score['score']:.0f})")
        elif signal.orb_broken == "INSIDE":
            warnings.append("⏳ Still inside ORB - no trend confirmed")
        else:
            warnings.append(f"⚠️ ORB break weak (held only {signal.orb_hold_candles} candles)")
        
        # === 5. PULLBACK QUALITY (10 points) ===
        # Shallow, fast pullbacks = strong trend
        pullback_score = self._score_pullback(signal)
        score += pullback_score['score']
        breakdown['pullback'] = pullback_score['score']
        if pullback_score['quality'] == 'EXCELLENT':
            reasons.append(f"✨ SHALLOW pullback ({signal.pullback_depth_pct:.2f}%, {signal.pullback_candles} candles) (+{pullback_score['score']:.0f})")
        elif pullback_score['quality'] == 'GOOD':
            reasons.append(f"👍 Good pullback entry ({signal.pullback_depth_pct:.2f}%) (+{pullback_score['score']:.0f})")
        elif pullback_score['quality'] == 'DEEP':
            warnings.append(f"⚠️ Pullback too deep ({signal.pullback_depth_pct:.2f}%) - trend may reverse")
        
        # === 6. ADX STRENGTH (10 points) ===
        # ADX confirms trend conviction
        adx_score = self._score_adx(signal)
        score += adx_score
        breakdown['adx'] = adx_score
        if signal.adx > 40:
            reasons.append(f"💪 ADX STRONG ({signal.adx:.0f}) - powerful trend (+{adx_score:.0f})")
        elif signal.adx > 25:
            reasons.append(f"📈 ADX solid ({signal.adx:.0f}) - confirmed trend (+{adx_score:.0f})")
        else:
            warnings.append(f"⚠️ ADX weak ({signal.adx:.0f}) - no strong trend")
        
        # === DETERMINE TREND STATE (WITH HYSTERESIS) ===
        # Uses hysteresis to prevent state flickering on pullbacks
        trend_state = self._apply_hysteresis(
            symbol=signal.symbol,
            raw_score=score,
            bullish_points=bullish_points,
            bearish_points=bearish_points,
            signal=signal  # Pass signal for shock override detection
        )
        
        # Log hysteresis effect if state differs from raw
        prev_state = self.previous_states.get(signal.symbol)
        raw_state = self._get_raw_state(score, bullish_points, bearish_points)
        if prev_state and trend_state != raw_state:
            reasons.append(f"🔒 HYSTERESIS: Holding {trend_state.value} (raw: {raw_state.value})")
        
        # === DETERMINE ENTRY ===
        should_enter, entry_type, entry_direction = self._determine_entry(
            signal, score, bullish_points, bearish_points, trend_state
        )
        
        # === CALCULATE LEVELS ===
        entry_price, stop_loss, target_1, target_2, trailing_pct = self._calculate_levels(
            signal, entry_direction, score
        )
        
        # === POSITION SIZE MULTIPLIER ===
        if score >= 85:
            size_mult = 1.5
            reasons.append("💪 HIGH conviction - 1.5x position size")
        elif score >= 75:
            size_mult = 1.25
        elif score >= 60:
            size_mult = 1.0
        elif score >= 50:
            size_mult = 0.75
            warnings.append("⚠️ Medium conviction - 0.75x size")
        else:
            size_mult = 0.5
            warnings.append("⚠️ Low conviction - 0.5x size")
        
        # === CONFIDENCE ===
        if score >= 80:
            confidence = "HIGH"
        elif score >= 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        decision = TrendDecision(
            trend_state=trend_state,
            trend_score=min(100, max(0, score)),
            should_enter=should_enter,
            entry_type=entry_type,
            entry_direction=entry_direction,
            confidence=confidence,
            position_size_multiplier=size_mult,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            trailing_stop_pct=trailing_pct,
            score_breakdown=breakdown,
            reasons=reasons,
            warnings=warnings
        )
        
        # Store for tracking
        self.active_trends[signal.symbol] = decision
        
        return decision
    
    def _score_vwap_slope(self, signal: TrendSignal) -> Dict:
        """Score VWAP slope - shows institutional direction"""
        slope = abs(signal.vwap_slope)
        max_points = self.WEIGHTS['vwap_slope']
        
        # Strong slope thresholds
        if slope >= 0.3:  # Very strong trend
            score = max_points
        elif slope >= 0.2:  # Strong
            score = max_points * 0.8
        elif slope >= 0.1:  # Moderate
            score = max_points * 0.6
        elif slope >= 0.05:  # Weak
            score = max_points * 0.3
        else:
            score = 0
        
        direction = "BULLISH" if signal.vwap_slope > 0.05 else "BEARISH" if signal.vwap_slope < -0.05 else "NEUTRAL"
        
        return {"score": score, "direction": direction}
    
    def _score_ema_expansion(self, signal: TrendSignal) -> Dict:
        """Score EMA expansion - confirms trend acceleration"""
        max_points = self.WEIGHTS['ema_expansion']
        
        if not signal.ema_expanding:
            return {"score": max_points * 0.2, "expanding": False}
        
        spread = abs(signal.ema_spread_pct)
        
        if spread >= 0.5:  # Very wide spread
            score = max_points
        elif spread >= 0.3:  # Wide
            score = max_points * 0.8
        elif spread >= 0.15:  # Moderate
            score = max_points * 0.6
        else:
            score = max_points * 0.4
        
        return {"score": score, "expanding": True}
    
    def _score_volume(self, signal: TrendSignal) -> float:
        """Score volume regime"""
        max_points = self.WEIGHTS['volume']
        
        if signal.volume_regime == "EXPLOSIVE":
            return max_points
        elif signal.volume_regime == "HIGH":
            return max_points * 0.75
        elif signal.volume_regime == "NORMAL":
            return max_points * 0.4
        else:  # LOW
            return max_points * 0.1
    
    def _score_orb_hold(self, signal: TrendSignal) -> Dict:
        """Score ORB break and hold quality"""
        max_points = self.WEIGHTS['orb_hold']
        
        if signal.orb_broken == "INSIDE":
            return {"score": 0, "held": False}
        
        # More candles held = stronger signal
        hold_candles = signal.orb_hold_candles
        
        if hold_candles >= 5:
            score = max_points
        elif hold_candles >= 3:
            score = max_points * 0.8
        elif hold_candles >= 2:
            score = max_points * 0.6
        elif hold_candles >= 1:
            score = max_points * 0.3
        else:
            score = 0
        
        return {"score": score, "held": hold_candles >= 2}
    
    def _score_pullback(self, signal: TrendSignal) -> Dict:
        """Score pullback quality - shallow and fast = strong trend"""
        max_points = self.WEIGHTS['pullback']
        depth = signal.pullback_depth_pct
        candles = signal.pullback_candles
        
        # Ideal: < 0.3% depth, < 2 candles
        # Good: < 0.5% depth, < 3 candles
        # Deep: > 0.5% depth
        
        if depth <= 0.2 and candles <= 2:
            return {"score": max_points, "quality": "EXCELLENT"}
        elif depth <= 0.4 and candles <= 3:
            return {"score": max_points * 0.8, "quality": "GOOD"}
        elif depth <= 0.6 and candles <= 4:
            return {"score": max_points * 0.5, "quality": "ACCEPTABLE"}
        else:
            return {"score": max_points * 0.2, "quality": "DEEP"}
    
    def _score_adx(self, signal: TrendSignal) -> float:
        """Score ADX trend strength"""
        max_points = self.WEIGHTS['adx']
        
        if signal.adx >= 40:
            return max_points
        elif signal.adx >= 30:
            return max_points * 0.8
        elif signal.adx >= 25:
            return max_points * 0.6
        elif signal.adx >= 20:
            return max_points * 0.3
        else:
            return 0
    
    def _determine_entry(self, signal: TrendSignal, score: float, 
                         bullish: float, bearish: float,
                         trend_state: TrendState) -> Tuple[bool, Optional[TrendEntryType], str]:
        """Determine if we should enter and how"""
        
        # Need minimum score to enter
        if score < self.TREND_THRESHOLD:
            return False, None, "HOLD"
        
        # Determine direction
        if bullish > bearish:
            direction = "BUY"
        elif bearish > bullish:
            direction = "SELL"
        else:
            return False, None, "HOLD"
        
        # Determine entry type
        # INITIAL_BREAK: Fresh ORB breakout, no existing position
        if signal.orb_broken != "INSIDE" and signal.orb_hold_candles <= 3:
            if signal.volume_regime in ["EXPLOSIVE", "HIGH"]:
                return True, TrendEntryType.INITIAL_BREAK, direction
        
        # PULLBACK_ENTRY: Price pulled back to EMA in trend
        if signal.pullback_depth_pct > 0.1 and signal.pullback_depth_pct <= 0.5:
            if signal.pullback_candles <= 3:
                # Check price near EMA
                ema_dist = abs((signal.ltp - signal.ema_9) / signal.ema_9 * 100)
                if ema_dist <= 0.3:  # Within 0.3% of 9 EMA
                    return True, TrendEntryType.PULLBACK_ENTRY, direction
        
        # CONTINUATION: Strong trend, not in pullback
        if score >= 80 and signal.ema_expanding and signal.volume_regime in ["EXPLOSIVE", "HIGH"]:
            return True, TrendEntryType.CONTINUATION, direction
        
        # Default: Don't enter if conditions aren't met
        if score >= self.TREND_THRESHOLD:
            return True, TrendEntryType.CONTINUATION, direction
        
        return False, None, "HOLD"
    
    def _calculate_levels(self, signal: TrendSignal, direction: str, score: float) -> Tuple[float, float, float, float, float]:
        """Calculate entry, stop, and target levels"""
        
        if direction == "BUY":
            entry = signal.ltp
            
            # Stop loss: Below recent low or EMA, whichever is tighter
            sl_ema = signal.ema_21 * 0.998  # Slightly below 21 EMA
            sl_low = signal.low * 0.998
            stop_loss = max(sl_ema, sl_low)  # Tighter stop
            
            # Ensure minimum 0.5% stop
            min_sl = entry * 0.995
            stop_loss = min(stop_loss, min_sl)
            
            # Targets based on score
            if score >= 80:
                target_1 = entry * 1.015  # 1.5% for strong trends
                target_2 = entry * 1.025  # 2.5% extended
                trailing = 0.8           # Tight trailing in strong trend
            else:
                target_1 = entry * 1.01   # 1%
                target_2 = entry * 1.02   # 2%
                trailing = 1.0
            
        else:  # SELL
            entry = signal.ltp
            
            sl_ema = signal.ema_21 * 1.002
            sl_high = signal.high * 1.002
            stop_loss = min(sl_ema, sl_high)
            
            max_sl = entry * 1.005
            stop_loss = max(stop_loss, max_sl)
            
            if score >= 80:
                target_1 = entry * 0.985
                target_2 = entry * 0.975
                trailing = 0.8
            else:
                target_1 = entry * 0.99
                target_2 = entry * 0.98
                trailing = 1.0
        
        return entry, stop_loss, target_1, target_2, trailing
    
    def check_trend_exit(self, symbol: str, current_price: float, 
                         current_signal: TrendSignal) -> Optional[Dict]:
        """Check if trend trade should exit"""
        
        if symbol not in self.active_trends:
            return None
        
        decision = self.active_trends[symbol]
        
        exit_reasons = []
        
        # 1. TRAILING STOP HIT
        if decision.entry_direction == "BUY":
            trailing_price = current_price * (1 - decision.trailing_stop_pct / 100)
            if current_price < trailing_price:
                exit_reasons.append(("TRAILING_STOP", f"Price {current_price:.2f} hit trailing"))
        else:
            trailing_price = current_price * (1 + decision.trailing_stop_pct / 100)
            if current_price > trailing_price:
                exit_reasons.append(("TRAILING_STOP", f"Price {current_price:.2f} hit trailing"))
        
        # 2. TREND REVERSAL - Score dropped significantly
        new_decision = self.analyze_trend(current_signal)
        if decision.entry_direction == "BUY" and new_decision.trend_state in [TrendState.BEARISH, TrendState.STRONG_BEARISH]:
            exit_reasons.append(("TREND_REVERSAL", f"Trend flipped to {new_decision.trend_state.value}"))
        elif decision.entry_direction == "SELL" and new_decision.trend_state in [TrendState.BULLISH, TrendState.STRONG_BULLISH]:
            exit_reasons.append(("TREND_REVERSAL", f"Trend flipped to {new_decision.trend_state.value}"))
        
        # 3. MOMENTUM FADE - Volume dying
        if current_signal.volume_regime == "LOW" and not current_signal.ema_expanding:
            exit_reasons.append(("MOMENTUM_FADE", "Volume low, EMAs not expanding"))
        
        # 4. VWAP CROSS - Significant level break
        if decision.entry_direction == "BUY" and current_price < current_signal.vwap:
            exit_reasons.append(("VWAP_BREAK", f"Price {current_price:.2f} fell below VWAP {current_signal.vwap:.2f}"))
        elif decision.entry_direction == "SELL" and current_price > current_signal.vwap:
            exit_reasons.append(("VWAP_BREAK", f"Price {current_price:.2f} rose above VWAP {current_signal.vwap:.2f}"))
        
        if exit_reasons:
            return {
                "should_exit": True,
                "exit_type": exit_reasons[0][0],
                "reason": exit_reasons[0][1],
                "all_reasons": exit_reasons,
                "current_price": current_price
            }
        
        return None
    
    def get_trend_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary of current trend for symbol"""
        if symbol not in self.active_trends:
            return None
        
        decision = self.active_trends[symbol]
        return {
            "symbol": symbol,
            "trend_state": decision.trend_state.value,
            "score": decision.trend_score,
            "direction": decision.entry_direction,
            "entry_type": decision.entry_type.value if decision.entry_type else None,
            "confidence": decision.confidence,
            "size_mult": decision.position_size_multiplier,
            "stop_loss": decision.stop_loss,
            "target_1": decision.target_1,
            "target_2": decision.target_2,
            "trailing_pct": decision.trailing_stop_pct,
            "reasons": decision.reasons,
            "warnings": decision.warnings
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_trend_signal_from_market_data(symbol: str, market_data: Dict) -> TrendSignal:
    """Build TrendSignal from zerodha_tools market data"""
    
    ltp = market_data.get('ltp', 0)
    
    # Handle vwap_slope - can be numeric OR string ('RISING', 'FALLING', 'FLAT')
    vwap_slope_raw = market_data.get('vwap_slope', 0)
    if isinstance(vwap_slope_raw, str):
        # Convert string to approximate numeric value
        if vwap_slope_raw.upper() == 'RISING':
            vwap_slope = 0.2
        elif vwap_slope_raw.upper() == 'FALLING':
            vwap_slope = -0.2
        else:
            vwap_slope = 0.0
    else:
        vwap_slope = float(vwap_slope_raw)
    
    # Handle ema_expanding - can be bool OR string
    ema_regime = market_data.get('ema_regime', 'NORMAL')
    if isinstance(ema_regime, str):
        ema_expanding = 'EXPANDING' in ema_regime.upper()
    else:
        ema_expanding = bool(ema_regime)
    
    # Handle ema_spread_pct - fallback to calculating from ema_9 and ema_21
    ema_spread_pct = market_data.get('ema_spread_pct', 0)
    if ema_spread_pct == 0:
        ema_9 = market_data.get('ema_9', ltp)
        ema_21 = market_data.get('ema_21', ltp)
        if ema_21 > 0:
            ema_spread_pct = ((ema_9 - ema_21) / ema_21) * 100
    
    return TrendSignal(
        symbol=symbol,
        ltp=ltp,
        open=market_data.get('open', ltp),
        high=market_data.get('high', ltp),
        low=market_data.get('low', ltp),
        vwap=market_data.get('vwap', ltp),
        vwap_slope=vwap_slope,
        vwap_distance_pct=market_data.get('vwap_distance_pct', 0),
        ema_9=market_data.get('ema_9', ltp),
        ema_21=market_data.get('ema_21', ltp),
        ema_spread_pct=ema_spread_pct,
        ema_expanding=ema_expanding,
        volume_regime=market_data.get('volume_regime', 'NORMAL'),
        volume_ratio=market_data.get('volume_ratio', 1.0),
        orb_high=market_data.get('orb_high', ltp * 1.01),
        orb_low=market_data.get('orb_low', ltp * 0.99),
        orb_broken=market_data.get('orb_signal', 'INSIDE').replace('BREAKOUT_', '').replace('INSIDE_ORB', 'INSIDE'),
        orb_hold_candles=market_data.get('orb_hold_candles', 0),
        pullback_depth_pct=market_data.get('pullback_depth_pct', 0),
        pullback_candles=market_data.get('pullback_candles', 0),
        rsi_14=market_data.get('rsi_14', 50),
        adx=market_data.get('adx', 20),
        htf_trend=market_data.get('htf_alignment', 'NEUTRAL')
    )


# Singleton instance
_trend_engine: Optional[TrendFollowingEngine] = None

def get_trend_engine() -> TrendFollowingEngine:
    """Get or create singleton trend following engine"""
    global _trend_engine
    if _trend_engine is None:
        _trend_engine = TrendFollowingEngine()
    return _trend_engine


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING TREND FOLLOWING ENGINE")
    print("=" * 70)
    
    engine = get_trend_engine()
    
    # Test 1: Strong Bullish Trend
    print("\n[1] STRONG BULLISH TREND - All signals aligned:")
    signal1 = TrendSignal(
        symbol="NSE:RELIANCE",
        ltp=2550,
        open=2520,
        high=2560,
        low=2510,
        vwap=2530,
        vwap_slope=0.35,           # Strong rising
        vwap_distance_pct=0.8,     # Above VWAP
        ema_9=2545,
        ema_21=2525,
        ema_spread_pct=0.8,        # Good expansion
        ema_expanding=True,
        volume_regime="EXPLOSIVE",
        volume_ratio=2.5,
        orb_high=2535,
        orb_low=2510,
        orb_broken="UP",
        orb_hold_candles=5,
        pullback_depth_pct=0.2,
        pullback_candles=1,
        rsi_14=65,
        adx=35
    )
    
    decision1 = engine.analyze_trend(signal1)
    print(f"    Trend State: {decision1.trend_state.value}")
    print(f"    Score: {decision1.trend_score:.0f}/100")
    print(f"    Should Enter: {decision1.should_enter}")
    print(f"    Entry Type: {decision1.entry_type.value if decision1.entry_type else 'N/A'}")
    print(f"    Direction: {decision1.entry_direction}")
    print(f"    Size Mult: {decision1.position_size_multiplier}x")
    print(f"    Stop Loss: ₹{decision1.stop_loss:.2f}")
    print(f"    Target 1: ₹{decision1.target_1:.2f}")
    print("    Reasons:")
    for r in decision1.reasons[:4]:
        print(f"      {r}")
    
    # Test 2: Weak/No Trend
    print("\n[2] WEAK/NO TREND - Inside ORB, low volume:")
    signal2 = TrendSignal(
        symbol="NSE:TCS",
        ltp=3650,
        open=3645,
        high=3660,
        low=3640,
        vwap=3650,
        vwap_slope=0.02,           # Flat
        vwap_distance_pct=0.0,
        ema_9=3648,
        ema_21=3650,
        ema_spread_pct=-0.05,
        ema_expanding=False,
        volume_regime="LOW",
        volume_ratio=0.6,
        orb_high=3665,
        orb_low=3635,
        orb_broken="INSIDE",
        orb_hold_candles=0,
        pullback_depth_pct=0.0,
        pullback_candles=0,
        rsi_14=50,
        adx=15
    )
    
    decision2 = engine.analyze_trend(signal2)
    print(f"    Trend State: {decision2.trend_state.value}")
    print(f"    Score: {decision2.trend_score:.0f}/100")
    print(f"    Should Enter: {decision2.should_enter}")
    print(f"    Direction: {decision2.entry_direction}")
    
    # Test 3: Pullback Entry
    print("\n[3] PULLBACK ENTRY - Price at 9 EMA in uptrend:")
    signal3 = TrendSignal(
        symbol="NSE:INFY",
        ltp=1520,
        open=1510,
        high=1535,
        low=1515,
        vwap=1512,
        vwap_slope=0.25,
        vwap_distance_pct=0.5,
        ema_9=1518,                # Price near 9 EMA
        ema_21=1505,
        ema_spread_pct=0.85,
        ema_expanding=True,
        volume_regime="HIGH",
        volume_ratio=1.8,
        orb_high=1515,
        orb_low=1500,
        orb_broken="UP",
        orb_hold_candles=8,
        pullback_depth_pct=0.3,    # Shallow pullback
        pullback_candles=2,        # Fast
        rsi_14=55,
        adx=30
    )
    
    decision3 = engine.analyze_trend(signal3)
    print(f"    Trend State: {decision3.trend_state.value}")
    print(f"    Score: {decision3.trend_score:.0f}/100")
    print(f"    Should Enter: {decision3.should_enter}")
    print(f"    Entry Type: {decision3.entry_type.value if decision3.entry_type else 'N/A'}")
    print(f"    Direction: {decision3.entry_direction}")
    
    print("\n" + "=" * 70)
    print("✅ TREND FOLLOWING ENGINE TESTS COMPLETE")
    print("=" * 70)
    print("""
TREND FOLLOWING SCORING:
------------------------
• VWAP Slope:     25 pts (institutional direction)
• EMA Expansion:  20 pts (trend acceleration)  
• Volume:         20 pts (participation)
• ORB Hold:       15 pts (breakout quality)
• Pullback:       10 pts (continuation quality)
• ADX:            10 pts (trend strength)
------------------------
TOTAL:           100 pts

THRESHOLDS:
• 80+ = STRONG trend, aggressive entry
• 60+ = Confirmed trend, standard entry  
• 40-59 = Neutral, wait
• <40 = No trend
    """)
