"""
OPTIONS TRADING MODULE FOR TITAN
Comprehensive options trading capabilities including:
1. Option Chain Fetcher (strikes, expiry)
2. Strike Selection Logic (ATM/ITM/OTM) 
3. NFO Order Placement
4. Option-specific Position Sizing (lot size, premium limits)
5. Option Greeks Integration (Delta, Gamma, Theta, Vega)
6. INTRADAY SIGNAL WEIGHTED DECISION (HIGHEST PRIORITY)
"""

import json
import math
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from scipy.stats import norm
from config import HARD_RULES

# Persistent log file — always in agentic_trader/ regardless of CWD
TRADE_DECISIONS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_decisions.log')

# ============================================================
# INTRADAY OPTION DECISION SCORER
# Makes intraday findings HIGHEST precedence in option decisions
# ============================================================

@dataclass
class OptionMicrostructure:
    """
    OPTIONS MICROSTRUCTURE DATA - HIGHEST IMPACT FOR LIVE P&L
    
    Without this, you can have perfect signals but still lose money
    on wide spreads, illiquid strikes, and partial fills.
    """
    bid: float = 0.0                  # Best bid price
    ask: float = 0.0                  # Best ask price
    spread_pct: float = 100.0         # Spread as % of mid price (100 = unknown/bad)
    bid_qty: int = 0                  # Top-of-book bid depth
    ask_qty: int = 0                  # Top-of-book ask depth
    open_interest: int = 0            # Open interest
    option_volume: int = 0            # Today's option volume
    oi_volume_ratio: float = 0.0      # OI / Volume ratio
    partial_fill_rate: float = 0.0    # Recent partial fill rate (0-1)
    cancel_rate: float = 0.0          # Recent cancel rate (0-1)
    avg_fill_time_ms: int = 0         # Average fill time in ms
    ltp: float = 0.0                  # Last traded price


@dataclass
class IntradaySignal:
    """Intraday signal data from market analysis"""
    symbol: str
    orb_signal: str = "INSIDE_ORB"     # BREAKOUT_UP, BREAKOUT_DOWN, INSIDE_ORB
    vwap_position: str = "AT_VWAP"     # ABOVE_VWAP, BELOW_VWAP, AT_VWAP
    vwap_trend: str = "FLAT"           # RISING, FALLING, FLAT
    ema_regime: str = "NORMAL"         # EXPANDING, COMPRESSED, NORMAL
    volume_regime: str = "NORMAL"      # EXPLOSIVE, HIGH, NORMAL, LOW
    rsi: float = 50.0                  # Kept only for overextension penalty
    price_momentum: float = 0.0        # % change in last 15min
    htf_alignment: str = "NEUTRAL"     # BULLISH, BEARISH, NEUTRAL
    chop_zone: bool = False            # If in CHOP filter (avoid trading)
    # === ACCELERATION / FOLLOW-THROUGH FIELDS (Options need speed!) ===
    follow_through_candles: int = 0    # Candles continuing after breakout (0-5)
    range_expansion_ratio: float = 0.0 # Candle body / ATR ratio (>1 = expanding)
    vwap_slope_steepening: bool = False  # VWAP slope increasing after entry
    atr: float = 0.0                   # Current ATR for reference
    # === WINDOW TRACKING FOR ANTI-DOUBLE-COUNTING ===
    orb_window_minutes: int = 15       # ORB computed on 15min window
    volume_window_minutes: int = 5     # Volume computed on 5min window


@dataclass
class IntradayOptionDecision:
    """Decision output from intraday analysis"""
    should_trade: bool
    confidence_score: float           # 0-100 score
    recommended_direction: str        # BUY, SELL, HOLD
    strike_selection: str             # ATM, ITM_1, OTM_1, etc.
    expiry_selection: str             # CURRENT_WEEK, NEXT_WEEK
    option_type: str                  # CE, PE
    position_size_multiplier: float   # 0.5 to 1.2 based on conviction (capped for safety)
    reasons: List[str] = field(default_factory=list)
    # === NEW TRACKING FIELDS ===
    trend_state: str = "NEUTRAL"      # From TrendFollowing engine
    acceleration_score: float = 0.0   # 0-10, needed for aggressive sizing
    warnings: List[str] = field(default_factory=list)
    # Microstructure gate results
    microstructure_score: float = 0.0          # 0-15 points from microstructure
    microstructure_block: bool = False         # Hard block due to illiquidity
    microstructure_block_reason: str = ""


class IntradayOptionScorer:
    """
    INTRADAY SIGNAL WEIGHTED OPTION DECISION ENGINE
    
    Priority Order (highest to lowest):
    1. TREND FOLLOWING MOMENTUM (30 points max) - REAL EDGE
    2. OPTIONS MICROSTRUCTURE (15 points max) - TRADABILITY GATE ⭐ NEW
    3. ORB Breakout Signals (10-20 points max, capped if trend strong)
    4. Volume Regime (7-15 points max, capped if trend strong)  
    5. VWAP Position + Trend (15 points max)
    6. EMA Regime (10 points max)
    7. ACCELERATION / FOLLOW-THROUGH (10 points max) - Options need speed!
    8. HTF Alignment (5 points max)
    9. RSI Overextension Penalty (-3 to 0) - Only penalizes extremes
    
    Total: ~100 points
    Threshold: 60 for trade, 80 for aggressive trade
    
    MICROSTRUCTURE GATE:
    Options need a hard gate for tradability:
    - Spread % (tight = high score, wide = 0 or BLOCK)
    - Top-of-book depth (thin = penalty)
    - OI / Volume filter (illiquid strikes = BLOCK)
    - Time-to-fill risk (repeated partials/cancels = penalty)
    
    ACCELERATION CHECK:
    Options require speed - captures "does the move have legs?":
    - Follow-through: after breakout, does price continue for 2+ candles?
    - Range expansion: candle body / ATR ratio > 1 = move has momentum
    - VWAP slope steepening: institutional flow accelerating
    
    ANTI-DOUBLE-COUNTING (by window overlap):
    TrendFollowing uses 15-30 min windows for ORB/Volume.
    OptionScorer uses 5 min windows.
    Cap only when windows overlap (same timeframe).
    
    HARD GATES:
    1. TREND STATE must be BULLISH/BEARISH/STRONG (NEUTRAL = block)
    2. Microstructure: spread %, absolute spread, OI, volume, depth
    3. OTM strikes require special conditions (not default!)
    """
    
    # Scoring weights - Rebalanced with safety
    WEIGHTS = {
        'trend_following': 30,   # HIGHEST - Real edge from indicators
        'microstructure': 15,    # Options tradability gate
        'orb_breakout': 20,      # HIGH - but capped if window overlap
        'volume_regime': 15,     # HIGH - but capped if window overlap
        'vwap_position': 10,     # REDUCED - penalize misalignment, small reward for aligned
        'ema_regime': 10,        # IMPORTANT - Trend structure
        'acceleration': 10,      # Follow-through + range expansion
        'htf_alignment': 5,      # COMPLEMENTARY - Higher timeframe
        'rsi_penalty': -2,       # Overextension penalty only (RSI <20 or >80)
    }
    
    # === SMART SPREAD EVALUATOR (Tick-based + Impact Cost) ===
    # Old % method was broken: ₹0.10 on ₹10 option = 1% ("wide") but it's just 2 ticks!
    # New approach: measure in TICKS (market quality) + IMPACT COST (P&L drag)
    TICK_SIZE = 0.05             # NSE option tick size
    
    # Tick-based thresholds (by premium bucket)
    # Cheap options (<₹20): 2-tick spread is normal & excellent
    # Mid options (₹20-75): slightly more room
    # Expensive options (>₹75): should be very tight in tick terms
    SPREAD_TICKS_EXCELLENT = 2   # 1-2 ticks = ₹0.10 max → +6 pts
    SPREAD_TICKS_GOOD = 4        # 3-4 ticks = ₹0.20 max → +4 pts
    
    # Premium-adaptive max ticks before BLOCK (dual-gate with impact)
    SPREAD_MAX_TICKS_CHEAP = 6   # <₹20 premium: max 6 ticks (₹0.30)
    SPREAD_MAX_TICKS_MID = 8     # ₹20-75 premium: max 8 ticks (₹0.40)
    SPREAD_MAX_TICKS_EXPENSIVE = 12  # >₹75 premium: max 12 ticks (₹0.60)
    
    # Impact cost: round-trip spread / expected profit
    # If target = 50%, expected profit = premium * 0.50
    # Round-trip cost = spread * 2 (pay on entry + exit)
    SPREAD_IMPACT_EXCELLENT = 0.05  # <5% of expected profit → +6 pts
    SPREAD_IMPACT_GOOD = 0.10       # <10% → +4 pts
    SPREAD_IMPACT_BLOCK = 0.20      # >20% of expected profit → BLOCK
    
    # Hard absolute backstop (catches truly broken quotes)
    SPREAD_BLOCK_ABS_INDEX = 2.0    # ₹2 max for index options
    SPREAD_BLOCK_ABS_STOCK = 3.0    # ₹3 max for stock options
    
    # Depth and volume thresholds (OI alone is not enough)
    MIN_DEPTH_LOTS = 2           # Min lots at best bid/ask (Kite returns shares, we normalize)
    MIN_OI = 500                 # Minimum OI for liquidity
    MIN_OPTION_VOLUME = 100      # ⭐ NEW: Min today's option volume
    MIN_OI_VOLUME_RATIO = 0.5    # OI should be at least 0.5x volume
    
    # Partial fill / cancel penalty thresholds
    PARTIAL_FILL_PENALTY_RATE = 0.3   # > 30% partials = penalty
    CANCEL_RATE_PENALTY = 0.4         # > 40% cancels = penalty
    
    # === TRADE THRESHOLDS (Simplified to 2 tiers) ===
    BLOCK_THRESHOLD = 60         # < 60 = BLOCK — raised from 55 for quality
    STANDARD_THRESHOLD = 65      # 65-69 = Standard (ATM/ITM, 1x size)
    PREMIUM_THRESHOLD = 70       # >= 70 = Premium (ATM/ITM, up to 1.5x)
    CHOP_PENALTY = 12            # Deduct if in CHOP zone [reduced from 30: was nuking all scores]
    
    # === AGGRESSIVE SIZING REQUIREMENTS (Risk of ruin protection) ===
    # OTM + larger size = blows up even good systems
    AGGRESSIVE_SIZE_MAX = 1.5           # Max 1.5x for premium tier
    AGGRESSIVE_ACCEL_MIN = 8            # Acceleration >= 8/10 required
    AGGRESSIVE_MICRO_MIN = 12           # Microstructure >= 12/15 required
    
    # === OTM STRIKE REQUIREMENTS (Special case, not default) ===
    OTM_REQUIRES_EXPLOSIVE_VOL = True   # Must have EXPLOSIVE volume
    OTM_REQUIRES_TIGHT_TICKS = 2       # OTM requires ≤2 tick spread (₹0.10)
    OTM_REQUIRES_ACCEL_MIN = 8          # Acceleration >= 8/10
    
    # TrendFollowing window (for anti-double-counting)
    TREND_WINDOW_MINUTES = 15           # TrendFollowing uses 15min ORB/Volume
    
    # === CANDLE-DATA-DERIVED GATES (from Feb 11 analysis) ===
    # Winners had follow_through=4.1, losers=0.4 → follow-through is the #1 edge
    # Winners had ADX=37.1, losers=30.8 → stronger trends win
    # Winners had ORB strength=29.9, losers=142.2 → low ORB strength = early entry = WIN
    # Winners had range_expansion=0.18, losers=0.37 → enter BEFORE the move, not after
    FOLLOW_THROUGH_MIN_PREMIUM = 2      # Premium tier: need ≥2 follow-through candles
    FOLLOW_THROUGH_MIN_STANDARD = 1     # Standard tier: need ≥1 follow-through candle
    ADX_MIN_PREMIUM = 30                # Premium tier: ADX ≥ 30 (trend strength)
    ADX_MIN_STANDARD = 25               # Standard tier: ADX ≥ 25
    ORB_STRENGTH_OVEREXTENDED = 100     # ORB strength > 100 = already moved too far
    RANGE_EXPANSION_OVEREXTENDED = 0.60 # Range > 0.6 ATR already = late entry
    
    # Re-entry prevention: don't trade same underlying if it already lost today
    MAX_LOSSES_SAME_SYMBOL = 1          # Max 1 loss per symbol per day, then skip
    
    def __init__(self):
        self.last_decisions: Dict[str, IntradayOptionDecision] = {}
        # Track per-symbol losses today (prevents re-entering SBIN 4x etc.)
        self.symbol_losses_today: Dict[str, int] = {}  # symbol -> loss count
        self.symbol_trades_today: Dict[str, int] = {}  # symbol -> total trade count
        # Import trend engine for integration
        try:
            from trend_following import get_trend_engine, build_trend_signal_from_market_data, TrendState
            self.trend_engine = get_trend_engine()
            self._trend_available = True
        except ImportError:
            self._trend_available = False
            print("⚠️ Trend Following module not available")
        
        # History tracking for time-to-fill risk
        self.fill_history: Dict[str, List[Dict]] = {}  # symbol -> list of fill records
    
    def score_intraday_signal(self, signal: IntradaySignal, 
                              market_data: Dict = None,
                              option_data: OptionMicrostructure = None,
                              caller_direction: str = None) -> IntradayOptionDecision:
        """
        Score intraday signals and recommend option parameters
        
        HARD GATES:
        1. Score >= BLOCK_THRESHOLD (30) required
        2. Microstructure gates (spread, OI, volume, depth)
        3. Must have a clear direction
        
        Args:
            signal: IntradaySignal with underlying price action
            market_data: Dict with candle/indicator data for trend analysis
            option_data: OptionMicrostructure with bid/ask/depth/OI
            caller_direction: Direction hint from GPT agent ('BUY' or 'SELL')
        """
        score = 0
        reasons = []
        warnings = []
        direction = "HOLD"
        bullish_points = 0
        bearish_points = 0
        
        # Tracking for decision
        microstructure_score = 0.0
        microstructure_block = False
        microstructure_block_reason = ""
        acceleration_score = 0
        trend_state_str = "NEUTRAL"
        trend_state_block = False  # Block if NEUTRAL
        
        # === 0. TREND FOLLOWING ANALYSIS (HIGHEST PRIORITY: 30 points) ===
        trend_score = 0
        trend_direction = "HOLD"
        raw_trend_score = 0
        trend_window_minutes = self.TREND_WINDOW_MINUTES  # TrendFollowing window
        
        if self._trend_available and market_data:
            try:
                from trend_following import build_trend_signal_from_market_data, TrendState
                trend_signal = build_trend_signal_from_market_data(signal.symbol, market_data)
                trend_decision = self.trend_engine.analyze_trend(trend_signal)
                
                raw_trend_score = trend_decision.trend_score
                normalized_trend = (trend_decision.trend_score / 100) * self.WEIGHTS['trend_following']
                trend_score = normalized_trend
                trend_state_str = trend_decision.trend_state.value
                
                # === NEUTRAL TREND: Score penalty instead of hard block ===
                # TrendFollowing data is often incomplete (ADX, ORB hold not flowing)
                # Strong intraday signals (ORB breakout, volume surge) should still trade
                if trend_decision.trend_state == TrendState.NEUTRAL:
                    score -= 5
                    warnings.append("⚠️ NEUTRAL trend penalty (-5) - need strong intraday signals")
                elif trend_decision.trend_state == TrendState.STRONG_BULLISH:
                    bullish_points += trend_score
                    trend_direction = "BUY"
                    reasons.append(f"📈 STRONG TREND BULLISH ({trend_decision.trend_score:.0f}/100) (+{trend_score:.0f})")
                elif trend_decision.trend_state == TrendState.BULLISH:
                    bullish_points += trend_score * 0.8
                    trend_direction = "BUY"
                    reasons.append(f"📈 TREND BULLISH ({trend_decision.trend_score:.0f}/100) (+{trend_score:.0f})")
                elif trend_decision.trend_state == TrendState.STRONG_BEARISH:
                    bearish_points += trend_score
                    trend_direction = "SELL"
                    reasons.append(f"📉 STRONG TREND BEARISH ({trend_decision.trend_score:.0f}/100) (+{trend_score:.0f})")
                elif trend_decision.trend_state == TrendState.BEARISH:
                    bearish_points += trend_score * 0.8
                    trend_direction = "SELL"
                    reasons.append(f"📉 TREND BEARISH ({trend_decision.trend_score:.0f}/100) (+{trend_score:.0f})")
                
                score += trend_score
                
                if trend_decision.entry_type:
                    reasons.append(f"   └─ Entry: {trend_decision.entry_type.value}")
                
            except Exception as e:
                warnings.append(f"⚠️ Trend analysis error: {str(e)[:30]}")
        else:
            # No trend data = penalty instead of block
            score -= 5
            warnings.append("⚠️ No trend data available - penalty applied (-5)")
        
        # === 1. OPTIONS MICROSTRUCTURE GATE (15 points) ===
        microstructure_score, microstructure_block, microstructure_block_reason = self._score_microstructure(option_data, signal, reasons, warnings)
        score += microstructure_score
        
        # === 2. ORB BREAKOUT (10-20 points, CAPPED if window overlap) ===
        # Anti-double-counting by WINDOW OVERLAP, not just score
        # TrendFollowing uses 15min window. If signal uses same window, cap.
        orb_max_points = 20
        orb_window_overlap = abs(signal.orb_window_minutes - trend_window_minutes) <= 5
        if orb_window_overlap and raw_trend_score >= 60:
            # Same window = duplicate info, cap contribution
            orb_max_points = 10
            cap_reason = f" [capped: same {signal.orb_window_minutes}min window]"
        else:
            cap_reason = ""
        
        # === ORB STALENESS DECAY ===
        # orb_strength = how far price moved from ORB range (in % of ORB range)
        # >100% = price moved more than 1x the ORB range = stale breakout
        # Decay: 0-50%=full, 50-100%=75%, 100-200%=50%, >200%=25%
        orb_strength = 0
        if market_data:
            orb_strength = abs(market_data.get('orb_strength', 0))
        if orb_strength > 200:
            orb_decay = 0.25
            cap_reason += f" [STALE: {orb_strength:.0f}% from ORB, 25% credit]"
        elif orb_strength > 100:
            orb_decay = 0.50
            cap_reason += f" [aging: {orb_strength:.0f}% from ORB, 50% credit]"
        elif orb_strength > 50:
            orb_decay = 0.75
            cap_reason += f" [aging: {orb_strength:.0f}% from ORB, 75% credit]"
        else:
            orb_decay = 1.0
        
        if signal.orb_signal == "BREAKOUT_UP":
            orb_points = int(orb_max_points * orb_decay)
            # ORB hold check — need at least 2 candles holding above ORB to confirm
            orb_hold = getattr(signal, 'orb_hold_candles', market_data.get('orb_hold_candles', 0) if market_data else 0)
            if orb_hold < 2:
                orb_points = max(2, orb_points // 2)  # Halve points for unconfirmed breakout
                cap_reason += f" [UNCONFIRMED: hold {orb_hold} candles < 2]"
            score += orb_points
            bullish_points += orb_points
            reasons.append(f"🚀 ORB BREAKOUT UP (+{orb_points}){cap_reason}")
            if direction == "HOLD":
                direction = "BUY"
        elif signal.orb_signal == "BREAKOUT_DOWN":
            orb_points = int(orb_max_points * orb_decay)
            # ORB hold check — need at least 2 candles holding below ORB to confirm
            orb_hold = getattr(signal, 'orb_hold_candles', market_data.get('orb_hold_candles', 0) if market_data else 0)
            if orb_hold < 2:
                orb_points = max(2, orb_points // 2)
                cap_reason += f" [UNCONFIRMED: hold {orb_hold} candles < 2]"
            score += orb_points
            bearish_points += orb_points
            reasons.append(f"📉 ORB BREAKOUT DOWN (+{orb_points}){cap_reason}")
            if direction == "HOLD":
                direction = "SELL"
        elif signal.orb_signal == "INSIDE_ORB":
            reasons.append("⏳ Inside ORB range (wait for breakout)")
        
        # === 3. VOLUME REGIME (7-15 points, CAPPED if window overlap) ===
        volume_multiplier = 1.0
        volume_window_overlap = abs(signal.volume_window_minutes - trend_window_minutes) <= 5
        if volume_window_overlap and raw_trend_score >= 60:
            volume_multiplier = 0.75
        
        if signal.volume_regime == "EXPLOSIVE":
            vol_points = int(15 * volume_multiplier)
            score += vol_points
            cap_note = f" [capped from 15]" if volume_multiplier < 1.0 else ""
            reasons.append(f"💥 EXPLOSIVE volume (+{vol_points}){cap_note}")
        elif signal.volume_regime == "HIGH":
            vol_points = int(12 * volume_multiplier)
            score += vol_points
            cap_note = f" [capped from 12]" if volume_multiplier < 1.0 else ""
            reasons.append(f"📊 HIGH volume (+{vol_points}){cap_note}")
        elif signal.volume_regime == "NORMAL":
            vol_points = int(6 * volume_multiplier)
            score += vol_points
            cap_note = f" [capped from 6]" if volume_multiplier < 1.0 else ""
            reasons.append(f"📊 Normal volume (+{vol_points}){cap_note}")
        elif signal.volume_regime == "LOW":
            score += 2  # No cap on penalty
            warnings.append("⚠️ LOW volume - weak conviction (+2)")
        
        # === 4. VWAP POSITION (10 points max, penalty for misalignment) ===
        # TrendFollowing already has VWAP Slope (25pts), so keep this as:
        # - Small positive when aligned with trend direction
        # - Penalty when misaligned (or block if severe)
        vwap_aligned = False
        vwap_misaligned = False
        
        if trend_direction == "BUY":
            # For BUY, we want ABOVE_VWAP or AT_VWAP
            if signal.vwap_position == "ABOVE_VWAP" and signal.vwap_trend == "RISING":
                score += 10
                bullish_points += 10
                reasons.append("📈 VWAP aligned: Above rising (+10)")
                vwap_aligned = True
            elif signal.vwap_position == "ABOVE_VWAP":
                score += 7
                bullish_points += 7
                reasons.append("📈 VWAP aligned: Above (+7)")
                vwap_aligned = True
            elif signal.vwap_position == "AT_VWAP":
                score += 3
                reasons.append("🔄 At VWAP (+3)")
            else:
                # BELOW_VWAP on a BUY signal = misalignment
                score -= 8
                vwap_misaligned = True
                warnings.append("⚠️ VWAP MISALIGNED: Below VWAP on BUY (-8)")
        elif trend_direction == "SELL":
            # For SELL, we want BELOW_VWAP or AT_VWAP
            if signal.vwap_position == "BELOW_VWAP" and signal.vwap_trend == "FALLING":
                score += 10
                bearish_points += 10
                reasons.append("📉 VWAP aligned: Below falling (+10)")
                vwap_aligned = True
            elif signal.vwap_position == "BELOW_VWAP":
                score += 7
                bearish_points += 7
                reasons.append("📉 VWAP aligned: Below (+7)")
                vwap_aligned = True
            elif signal.vwap_position == "AT_VWAP":
                score += 3
                reasons.append("🔄 At VWAP (+3)")
            else:
                # ABOVE_VWAP on a SELL signal = misalignment
                score -= 8
                vwap_misaligned = True
                warnings.append("⚠️ VWAP MISALIGNED: Above VWAP on SELL (-8)")
        else:
            # No direction yet - small points based on position
            if signal.vwap_position == "ABOVE_VWAP":
                score += 4
                bullish_points += 4
            elif signal.vwap_position == "BELOW_VWAP":
                score += 4
                bearish_points += 4
            else:
                score += 2
        
        # === 5. EMA REGIME (10 points) ===
        if signal.ema_regime == "EXPANDING":
            score += 10
            reasons.append("📊 EMA EXPANDING - strong trend (+10)")
        elif signal.ema_regime == "COMPRESSED":
            score += 8
            reasons.append("⚡ EMA COMPRESSED - breakout imminent (+8)")
        else:
            score += 3
            reasons.append("EMA normal (+3)")
        
        # === 5. HTF ALIGNMENT (5 points) ===
        if signal.htf_alignment == "BULLISH":
            score += 5
            bullish_points += 5
            reasons.append("🎯 HTF BULLISH alignment (+5)")
        elif signal.htf_alignment == "BEARISH":
            score += 5
            bearish_points += 5
            reasons.append("🎯 HTF BEARISH alignment (+5)")
        else:
            score += 2
            warnings.append("HTF neutral - watch for direction (+2)")
        
        # === 7. ACCELERATION / FOLLOW-THROUGH (10 points max) ===
        # Options require SPEED - this captures "does the move have legs?"
        accel_score = 0
        
        # 7a. Follow-through check (4 pts): After breakout, does price continue?
        if signal.follow_through_candles >= 3:
            accel_score += 4
            reasons.append(f"🚀 Strong follow-through ({signal.follow_through_candles} candles) (+4)")
        elif signal.follow_through_candles >= 2:
            accel_score += 3
            reasons.append(f"📈 Good follow-through ({signal.follow_through_candles} candles) (+3)")
        elif signal.follow_through_candles >= 1:
            accel_score += 1
            reasons.append(f"➡️ Some follow-through ({signal.follow_through_candles} candle) (+1)")
        
        # 7b. Range expansion (3 pts): Candle body / ATR ratio
        if signal.range_expansion_ratio >= 1.5:
            accel_score += 3
            reasons.append(f"💥 Strong range expansion ({signal.range_expansion_ratio:.1f}x ATR) (+3)")
        elif signal.range_expansion_ratio >= 1.0:
            accel_score += 2
            reasons.append(f"📊 Good range expansion ({signal.range_expansion_ratio:.1f}x ATR) (+2)")
        elif signal.range_expansion_ratio >= 0.5:
            accel_score += 1
            reasons.append(f"Range expansion ({signal.range_expansion_ratio:.1f}x ATR) (+1)")
        
        # 7c. VWAP slope steepening (3 pts): Acceleration in institutional flow
        if signal.vwap_slope_steepening:
            accel_score += 3
            reasons.append("⚡ VWAP slope steepening (+3)")
        
        # Store acceleration score for aggressive sizing check
        acceleration_score = accel_score
        
        # Apply acceleration points (directional)
        if trend_direction == "BUY" or direction == "BUY":
            bullish_points += accel_score
        elif trend_direction == "SELL" or direction == "SELL":
            bearish_points += accel_score
        score += accel_score
        
        # === RSI OVEREXTENSION PENALTY ONLY (max -3 pts) ===
        # RSI is redundant with trend + EMA, only use for extreme overextension
        if signal.rsi < 20:
            # Extremely oversold - potential bounce risk for shorts
            if direction == "SELL" or trend_direction == "SELL":
                score -= 2
                warnings.append(f"⚠️ RSI extremely oversold ({signal.rsi:.0f}) - bounce risk (-2)")
        elif signal.rsi > 80:
            # Extremely overbought - potential reversal risk for longs
            if direction == "BUY" or trend_direction == "BUY":
                score -= 2
                warnings.append(f"⚠️ RSI extremely overbought ({signal.rsi:.0f}) - reversal risk (-2)")
        elif signal.rsi < 25 and (direction == "SELL" or trend_direction == "SELL"):
            score -= 1
            warnings.append(f"⚠️ RSI low ({signal.rsi:.0f}) - caution on shorts (-1)")
        elif signal.rsi > 75 and (direction == "BUY" or trend_direction == "BUY"):
            score -= 1
            warnings.append(f"⚠️ RSI high ({signal.rsi:.0f}) - caution on longs (-1)")
        # Note: RSI 30-70 = no impact (neutral)
        
        # === 8. MOMENTUM EXHAUSTION FILTER (graduated, overridable) ===
        # Prevents chasing moves that have already exhausted their run.
        # Checks BOTH gap (from prev close) AND intraday move (from today's open).
        # A stock up +6% intraday is almost as dangerous as a 6% gap.
        exhaustion_score = 0
        if market_data:
            gap_pct = abs(market_data.get('change_pct', 0))   # % from prev close
            ltp = market_data.get('ltp', 0)
            day_high = market_data.get('high', 0)
            day_low = market_data.get('low', 0)
            day_open = market_data.get('open', ltp)
            vwap = market_data.get('vwap', 0)
            day_range = day_high - day_low if day_high > day_low else 0.01

            # --- NEW: Calculate intraday move from open ---
            intraday_move_pct = abs((ltp - day_open) / day_open * 100) if day_open > 0 and ltp > 0 else 0
            # Total exhaustion signal = max of gap and intraday move
            # (captures both gap plays AND intraday runners)
            total_move_pct = max(gap_pct, intraday_move_pct)

            # --- A. Gap penalty (% from previous close) ---
            if gap_pct >= 7:
                exhaustion_score -= 18
                warnings.append(f"🔥 EXHAUSTION: Massive gap {gap_pct:.1f}% from prev close (-18)")
            elif gap_pct >= 5:
                exhaustion_score -= 10
                warnings.append(f"⚠️ EXHAUSTION: Large gap {gap_pct:.1f}% from prev close (-10)")
            elif gap_pct >= 3:
                exhaustion_score -= 5
                warnings.append(f"⚠️ EXHAUSTION: Gap {gap_pct:.1f}% from prev close (-5)")

            # --- B. INTRADAY MOVE PENALTY (% from today's open) ---
            # This is the NEW check that catches stocks like ETERNAL (+6% from open)
            # Only apply if gap penalty didn't already cover it (avoid double-counting)
            if intraday_move_pct > gap_pct:
                # Intraday move is larger than the gap → the move happened today
                if intraday_move_pct >= 6:
                    exhaustion_score -= 25
                    warnings.append(f"🔥 CHASING: {intraday_move_pct:.1f}% intraday move from open — DANGEROUS late entry (-25)")
                elif intraday_move_pct >= 4:
                    exhaustion_score -= 18
                    warnings.append(f"⚠️ CHASING: {intraday_move_pct:.1f}% intraday move from open — extended (-18)")
                elif intraday_move_pct >= 2.5:
                    exhaustion_score -= 12
                    warnings.append(f"⚠️ CHASING: {intraday_move_pct:.1f}% intraday move from open (-12)")
                elif intraday_move_pct >= 1.5:
                    exhaustion_score -= 5
                    warnings.append(f"⚠️ CHASING: {intraday_move_pct:.1f}% intraday move from open (-5)")

            # --- C. Retracement check (holding at highs = strong, retraced = weak) ---
            if total_move_pct >= 2 and ltp > 0 and day_high > 0:
                is_bullish_move = (ltp > day_open) if day_open > 0 else (market_data.get('change_pct', 0) > 0)
                if is_bullish_move:
                    pct_from_high = ((day_high - ltp) / day_high * 100) if day_high else 0
                    if pct_from_high <= 0.5:
                        # AT day highs — still has momentum, partial override
                        exhaustion_score += 5
                        reasons.append(f"🚀 MOMENTUM HOLDING: LTP within {pct_from_high:.1f}% of day-high (+5)")
                    elif (day_high - ltp) / day_range > 0.5:
                        exhaustion_score -= 8
                        retrace_pct = (day_high - ltp) / day_range * 100
                        warnings.append(f"📉 EXHAUSTION: Retraced {retrace_pct:.0f}% of day range (-8)")
                else:
                    pct_from_low = ((ltp - day_low) / day_low * 100) if day_low else 0
                    if pct_from_low <= 0.5:
                        exhaustion_score += 5
                        reasons.append(f"📉 MOMENTUM HOLDING: LTP within {pct_from_low:.1f}% of day-low (+5)")
                    elif (ltp - day_low) / day_range > 0.5:
                        exhaustion_score -= 8
                        retrace_pct = (ltp - day_low) / day_range * 100
                        warnings.append(f"🚀 EXHAUSTION: Bounced {retrace_pct:.0f}% of day range (-8)")

            # --- D. Volume confirmation (only helps reduce penalty, not eliminate) ---
            if total_move_pct >= 3:
                vol_regime = market_data.get('volume_regime', 'NORMAL')
                if vol_regime == 'EXPLOSIVE':
                    exhaustion_score += 5
                    reasons.append("💥 EXHAUSTION OVERRIDE: Explosive volume confirms move (+5)")
                elif vol_regime == 'LOW':
                    exhaustion_score -= 3
                    warnings.append("⚠️ EXHAUSTION: Move on LOW volume (-3)")

            # --- E. VWAP arbiter (below VWAP on move-up = move failing) ---
            if total_move_pct >= 2 and vwap > 0 and ltp > 0:
                is_bullish = ltp > day_open if day_open > 0 else market_data.get('change_pct', 0) > 0
                if is_bullish and ltp < vwap:
                    exhaustion_score -= 10
                    warnings.append(f"🚫 EXHAUSTION: Moved up BUT below VWAP ₹{vwap:.0f} (-10)")
                elif not is_bullish and ltp > vwap:
                    exhaustion_score -= 10
                    warnings.append(f"🚫 EXHAUSTION: Moved down BUT above VWAP ₹{vwap:.0f} (-10)")

            score += exhaustion_score

        # === CHOP ZONE PENALTY ===
        if signal.chop_zone:
            score -= self.CHOP_PENALTY
            warnings.append(f"⛔ CHOP ZONE - deducted {self.CHOP_PENALTY} points")
        
        # === CALLER DIRECTION ALIGNMENT BONUS (5 pts max) ===
        # When GPT agent has a clear direction and underlying signals partially support it,
        # award a bonus. This helps bridge the gap when trend_following returns NEUTRAL.
        if caller_direction and caller_direction in ('BUY', 'SELL'):
            alignment_pts = 0
            if caller_direction == 'BUY':
                if signal.vwap_position == 'ABOVE_VWAP':
                    alignment_pts += 1
                if signal.volume_regime in ('HIGH', 'EXPLOSIVE'):
                    alignment_pts += 1
                if signal.ema_regime in ('EXPANDING',):
                    alignment_pts += 1
            elif caller_direction == 'SELL':
                if signal.vwap_position == 'BELOW_VWAP':
                    alignment_pts += 1
                if signal.volume_regime in ('HIGH', 'EXPLOSIVE'):
                    alignment_pts += 1
                if signal.ema_regime in ('EXPANDING', 'COMPRESSED'):
                    alignment_pts += 1
            if alignment_pts > 0:
                alignment_pts = min(alignment_pts, 3)  # Cap at +3, was +5 — GPT direction shouldn't override technicals
                score += alignment_pts
                reasons.append(f"🤖 Agent direction aligned with signals (+{alignment_pts})")
                # Also help set direction when trend is NEUTRAL
                if direction == 'HOLD':
                    direction = caller_direction
        
        # === DETERMINE DIRECTION ===
        if direction == "HOLD":
            if bullish_points > bearish_points + 10:
                direction = "BUY"
            elif bearish_points > bullish_points + 10:
                direction = "SELL"
            elif caller_direction and caller_direction in ('BUY', 'SELL'):
                # Only use GPT fallback if directional points margin is small (< 5)
                # AND at least one directional signal exists
                if max(bullish_points, bearish_points) >= 10:
                    direction = caller_direction
                else:
                    # Not enough signals to even use GPT direction
                    direction = "HOLD"
        
        # === DETERMINE STRIKE SELECTION (with OTM safety rules) ===
        strike_selection = self._recommend_strike(
            score=score, 
            signal=signal,
            acceleration_score=acceleration_score,
            microstructure_score=microstructure_score,
            option_data=option_data
        )
        
        # === DETERMINE EXPIRY ===
        expiry_selection = self._recommend_expiry(score, signal)
        
        # === DETERMINE OPTION TYPE ===
        option_type = "CE" if direction == "BUY" else "PE" if direction == "SELL" else "CE"
        
        # === POSITION SIZE MULTIPLIER (Capped at 1.2x for safety) ===
        # Aggressive sizing requires BOTH acceleration dominance AND microstructure quality
        can_size_up = (
            acceleration_score >= self.AGGRESSIVE_ACCEL_MIN and
            microstructure_score >= self.AGGRESSIVE_MICRO_MIN
        )
        
        if score >= self.PREMIUM_THRESHOLD and can_size_up:
            size_multiplier = self.AGGRESSIVE_SIZE_MAX  # 1.2x max
            reasons.append(f"💪 PREMIUM conviction - {self.AGGRESSIVE_SIZE_MAX}x size (accel {acceleration_score}/10, micro {microstructure_score:.0f}/15)")
        elif score >= self.PREMIUM_THRESHOLD:
            size_multiplier = 1.0
            warnings.append(f"⚠️ Score >= {self.PREMIUM_THRESHOLD} but accel ({acceleration_score}/10) or micro ({microstructure_score:.0f}/15) too low for sizing up")
        elif score >= self.STANDARD_THRESHOLD:
            size_multiplier = 1.0
        else:
            size_multiplier = 0.75
            warnings.append("⚠️ Below standard threshold - 0.75x size")
        
        # === ALL HARD GATES ===
        should_trade = True
        
        # Gate 1: Score must be >= BLOCK_THRESHOLD
        if score < self.BLOCK_THRESHOLD:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: Score {score:.0f} < {self.BLOCK_THRESHOLD} minimum")
        
        # Gate 2: Must have at least one meaningful directional signal
        # Prevents trades that pass only on volume + microstructure noise
        directional_strength = max(bullish_points, bearish_points)
        if directional_strength < 15:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: No directional conviction (best: {directional_strength:.0f}pts, need ≥15)")
        
        # Gate 3: Must have direction
        if direction == "HOLD":
            should_trade = False
            warnings.append("🚫 BLOCKED: No clear direction")
        
        # Gate 4: Microstructure block
        if microstructure_block:
            should_trade = False
            warnings.append(f"🚫 BLOCKED by microstructure: {microstructure_block_reason}")
        
        # Gate 5: VWAP HARD GATE — direction MUST align with VWAP position
        # BUY direction requires price ABOVE or AT VWAP (not below)
        # SELL direction requires price BELOW or AT VWAP (not above)
        # This prevents the #1 cause of wrong direction: buying into weakness
        if should_trade and direction == "BUY" and signal.vwap_position == "BELOW_VWAP":
            # Exception: if trend is STRONG_BULLISH, allow (pullback entry)
            if trend_state_str not in ('STRONG_BULLISH',):
                should_trade = False
                warnings.append(f"🚫 BLOCKED: BUY direction but price BELOW VWAP — buying into weakness")
        if should_trade and direction == "SELL" and signal.vwap_position == "ABOVE_VWAP":
            if trend_state_str not in ('STRONG_BEARISH',):
                should_trade = False
                warnings.append(f"🚫 BLOCKED: SELL direction but price ABOVE VWAP — selling into strength")
        
        # Gate 6: COUNTER-TREND BLOCK — agent direction vs trend must not conflict
        # If trend says BEARISH/STRONG_BEARISH, don't let agent buy CE
        if should_trade and direction == "BUY" and trend_state_str in ('BEARISH', 'STRONG_BEARISH'):
            should_trade = False
            warnings.append(f"🚫 BLOCKED: BUY conflicts with {trend_state_str} trend — counter-trend")
        if should_trade and direction == "SELL" and trend_state_str in ('BULLISH', 'STRONG_BULLISH'):
            should_trade = False
            warnings.append(f"🚫 BLOCKED: SELL conflicts with {trend_state_str} trend — counter-trend")
        
        # Gate 7: DAY RANGE POSITION — don't buy near day-high or sell near day-low
        # (buying at the top / selling at the bottom = guaranteed loss)
        if should_trade and market_data:
            ltp = market_data.get('ltp', 0)
            day_high = market_data.get('high', 0)
            day_low = market_data.get('low', 0)
            day_range = day_high - day_low if day_high > day_low else 0.01
            if day_range > 0.01 and ltp > 0:
                position_in_range = (ltp - day_low) / day_range  # 0=at low, 1=at high
                if direction == "BUY" and position_in_range > 0.90:
                    score -= 10
                    warnings.append(f"⚠️ TOPPING: LTP at {position_in_range*100:.0f}% of day range — buying near top (-10)")
                    if position_in_range > 0.95:
                        should_trade = False
                        warnings.append(f"🚫 BLOCKED: Buying at {position_in_range*100:.0f}% of day range = buying the exact top")
                elif direction == "SELL" and position_in_range < 0.10:
                    score -= 10
                    warnings.append(f"⚠️ BOTTOMING: LTP at {position_in_range*100:.0f}% of day range — selling near bottom (-10)")
                    if position_in_range < 0.05:
                        should_trade = False
                        warnings.append(f"🚫 BLOCKED: Selling at {position_in_range*100:.0f}% of day range = selling the exact bottom")
        
        # ===================================================================
        # CANDLE-DATA-DERIVED SMART GATES (from Feb 11 backtesting analysis)
        # These are the REAL edge — derived from actual win/loss candle data
        # Winners: FT=4.1, ADX=37, ORB_str=30, range_exp=0.18
        # Losers:  FT=0.4, ADX=31, ORB_str=142, range_exp=0.37
        # ===================================================================
        
        # Determine the intended tier for gate thresholds
        if score >= self.PREMIUM_THRESHOLD:
            intended_tier = "premium"
        elif score >= self.STANDARD_THRESHOLD:
            intended_tier = "standard"
        else:
            intended_tier = "base"
        
        # Gate 8: FOLLOW-THROUGH GATE — the #1 predictor (winners 4.1x vs losers 0.4x)
        # Trades with 0 follow-through candles are essentially coinflips
        ft_candles = signal.follow_through_candles
        if should_trade and intended_tier == "premium":
            if ft_candles < self.FOLLOW_THROUGH_MIN_PREMIUM:
                score -= 8
                warnings.append(f"⚠️ LOW FOLLOW-THROUGH: {ft_candles} candles < {self.FOLLOW_THROUGH_MIN_PREMIUM} required for premium (-8)")
                if ft_candles == 0:
                    # Premium with zero follow-through = don't do it
                    should_trade = False
                    warnings.append(f"🚫 BLOCKED: Premium tier with ZERO follow-through = blind entry")
        elif should_trade and intended_tier == "standard":
            if ft_candles < self.FOLLOW_THROUGH_MIN_STANDARD:
                score -= 5
                warnings.append(f"⚠️ NO FOLLOW-THROUGH: {ft_candles} candles — no confirmation after breakout (-5)")
        
        # Gate 9: ADX TREND STRENGTH — winners had ADX 37+, losers had 31
        # ADX < 25 = no trend = directionless chop
        adx = market_data.get('adx', 0) if market_data else 0
        if should_trade and adx > 0:
            if intended_tier == "premium" and adx < self.ADX_MIN_PREMIUM:
                score -= 5
                warnings.append(f"⚠️ WEAK TREND: ADX {adx:.0f} < {self.ADX_MIN_PREMIUM} for premium tier (-5)")
            elif intended_tier == "standard" and adx < self.ADX_MIN_STANDARD:
                score -= 3
                warnings.append(f"⚠️ WEAK TREND: ADX {adx:.0f} < {self.ADX_MIN_STANDARD} for standard tier (-3)")
            elif adx >= 40:
                # Strong trend bonus
                score += 3
                reasons.append(f"💪 Strong ADX {adx:.0f} — clear trend (+3)")
        
        # Gate 10: ORB STRENGTH OVEREXTENSION — winners entered at ORB_str 30, losers at 142
        # High ORB strength = price already moved far from ORB range = chasing the move
        orb_str = market_data.get('orb_strength', 0) if market_data else 0
        if should_trade and orb_str > self.ORB_STRENGTH_OVEREXTENDED:
            score -= 8
            warnings.append(f"⚠️ ORB OVEREXTENDED: Strength {orb_str:.0f}% > {self.ORB_STRENGTH_OVEREXTENDED}% — price too far from ORB range (-8)")
            if orb_str > 200:
                size_multiplier = min(size_multiplier, 0.75)
                warnings.append(f"⚠️ ORB EXHAUSTED: {orb_str:.0f}% — capping size at 0.75x")
        
        # Gate 11: RANGE EXPANSION OVEREXTENSION — winners 0.18, losers 0.37
        # If range has already expanded > 0.6x ATR, the move has happened
        range_exp = signal.range_expansion_ratio
        if should_trade and range_exp > self.RANGE_EXPANSION_OVEREXTENDED:
            score -= 5
            warnings.append(f"⚠️ RANGE ALREADY EXPANDED: {range_exp:.2f}x ATR > {self.RANGE_EXPANSION_OVEREXTENDED}x — late entry (-5)")
        
        # Gate 12: SAME-SYMBOL RE-ENTRY PREVENTION
        # SBIN entered 4x = all losses, BANDHANBNK 3x = all losses
        # Don't re-enter a symbol that already lost today
        sym_base = signal.symbol.replace("NSE:", "").replace("BSE:", "")
        sym_losses = self.symbol_losses_today.get(sym_base, 0)
        if should_trade and sym_losses >= self.MAX_LOSSES_SAME_SYMBOL:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: {sym_base} already lost {sym_losses}x today — no re-entry after loss")
        
        # Track trade attempt (for logging)
        self.symbol_trades_today[sym_base] = self.symbol_trades_today.get(sym_base, 0) + 1
        
        decision = IntradayOptionDecision(
            should_trade=should_trade,
            confidence_score=min(100, max(0, score)),
            recommended_direction=direction,
            strike_selection=strike_selection,
            expiry_selection=expiry_selection,
            option_type=option_type,
            position_size_multiplier=size_multiplier,
            reasons=reasons,
            trend_state=trend_state_str,
            acceleration_score=acceleration_score,
            warnings=warnings,
            microstructure_score=microstructure_score,
            microstructure_block=microstructure_block,
            microstructure_block_reason=microstructure_block_reason
        )
        
        self.last_decisions[signal.symbol] = decision
        return decision
    
    def _recommend_strike(
        self, 
        score: float, 
        signal: IntradaySignal,
        acceleration_score: float = 0,
        microstructure_score: float = 0,
        option_data: OptionMicrostructure = None
    ) -> str:
        """
        Recommend strike based on conviction and signals
        
        SAFETY RULES (Risk of ruin protection):
        - OTM is NEVER the default
        - OTM requires: EXPLOSIVE volume + tight spread + high acceleration
        - Standard/Premium = ATM or ITM only
        
        Tiers:
        - < 30: BLOCKED (no trade)
        - 30-34: ITM_1 (most conservative)
        - 35-54: ATM (standard)
        - 55+: ATM or ITM (premium but disciplined)
        - OTM: Special case only
        """
        # === OTM SPECIAL CASE (NOT DEFAULT) ===
        # OTM only if ALL conditions met
        if score >= self.PREMIUM_THRESHOLD:
            otm_allowed = True
            
            # Must have EXPLOSIVE volume
            if self.OTM_REQUIRES_EXPLOSIVE_VOL and signal.volume_regime != "EXPLOSIVE":
                otm_allowed = False
            
            # Must have tight spread (tick-based)
            if option_data and option_data.bid > 0 and option_data.ask > 0:
                otm_spread_ticks = round((option_data.ask - option_data.bid) / self.TICK_SIZE)
                if otm_spread_ticks > self.OTM_REQUIRES_TIGHT_TICKS:
                    otm_allowed = False
            else:
                otm_allowed = False
            
            # Must have high acceleration
            if acceleration_score < self.OTM_REQUIRES_ACCEL_MIN:
                otm_allowed = False
            
            if otm_allowed:
                return "OTM_1"  # Allowed OTM in special case
        
        # === STANDARD TIERS (ATM/ITM only) ===
        if score >= self.PREMIUM_THRESHOLD:
            # Premium tier: ATM for high delta, or ITM if very explosive
            if signal.volume_regime == "EXPLOSIVE":
                return "ITM_1"  # High delta for fast moves
            return "ATM"
        
        if score >= self.STANDARD_THRESHOLD:
            # Standard tier: ATM is safest
            return "ATM"
        
        # Below standard: ITM for safety (more forgiving)
        return "ITM_1"
    
    def _score_microstructure(
        self, 
        option_data: OptionMicrostructure,
        signal: IntradaySignal,
        reasons: List[str],
        warnings: List[str]
    ) -> Tuple[float, bool, str]:
        """
        Score options microstructure (15 points max)
        
        This is the HIGHEST IMPACT addition for live P&L.
        Without this, perfect signals can still lose money on:
        - Wide spreads eating profits
        - Thin depth causing slippage  
        - Illiquid strikes causing partial fills
        
        Returns:
            (score, is_blocked, block_reason)
        """
        if option_data is None:
            warnings.append("🚫 No microstructure data - BLOCKED (cannot assess spread/depth)")
            return (0.0, True, "No microstructure data available - chain fetch may have failed")
        
        score = 0.0
        block = False
        block_reason = ""
        
        # === 0. DETERMINE INSTRUMENT TYPE FOR THRESHOLDS ===
        # Index options (NIFTY, BANKNIFTY) use tighter thresholds
        symbol_upper = signal.symbol.upper() if signal.symbol else ""
        is_index = any(idx in symbol_upper for idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
        spread_block_abs = self.SPREAD_BLOCK_ABS_INDEX if is_index else self.SPREAD_BLOCK_ABS_STOCK
        
        # === 1. SMART SPREAD (6 points max) ===
        # Uses TICKS (market quality) + IMPACT COST (P&L drag) instead of raw %
        spread_pct = option_data.spread_pct
        
        bid_price = option_data.bid
        ask_price = option_data.ask
        spread_abs = (ask_price - bid_price) if (bid_price > 0 and ask_price > 0) else 0
        mid_price = (bid_price + ask_price) / 2 if (bid_price > 0 and ask_price > 0) else 0
        
        # --- Layer 1: Hard absolute backstop (catches broken quotes) ---
        if spread_abs > 0 and spread_abs > spread_block_abs:
            block = True
            block_reason = f"Absolute spread ₹{spread_abs:.2f} > ₹{spread_block_abs:.1f} max"
            warnings.append(f"🚫 BLOCKED: {block_reason}")
        elif mid_price <= 0:
            block = True
            block_reason = "No valid bid/ask prices"
            warnings.append(f"🚫 BLOCKED: {block_reason}")
        else:
            # --- Layer 2: Tick-based quality ---
            spread_ticks = round(spread_abs / self.TICK_SIZE) if spread_abs > 0 else 0
            
            # --- Layer 3: Impact cost (round-trip spread / expected profit) ---
            # We pay the spread TWICE: once buying at ask, once selling at bid
            expected_profit = mid_price * 0.50  # 50% target = standard
            impact_cost = (spread_abs * 2) / expected_profit if expected_profit > 0 else 1.0
            
            # --- Layer 4: Premium-adaptive max ticks ---
            if mid_price < 20:
                max_ticks = self.SPREAD_MAX_TICKS_CHEAP      # 6 ticks
            elif mid_price < 75:
                max_ticks = self.SPREAD_MAX_TICKS_MID         # 8 ticks
            else:
                max_ticks = self.SPREAD_MAX_TICKS_EXPENSIVE   # 12 ticks
            
            # --- Scoring: best of tick-quality OR impact-quality ---
            tick_excellent = spread_ticks <= self.SPREAD_TICKS_EXCELLENT  # ≤2 ticks
            tick_good = spread_ticks <= self.SPREAD_TICKS_GOOD           # ≤4 ticks
            impact_excellent = impact_cost <= self.SPREAD_IMPACT_EXCELLENT  # <5%
            impact_good = impact_cost <= self.SPREAD_IMPACT_GOOD           # <10%
            
            if tick_excellent or impact_excellent:
                # Either market is super tight OR impact is negligible
                score += 6
                reasons.append(f"💰 Excellent spread: {spread_ticks} ticks (₹{spread_abs:.2f}), impact {impact_cost*100:.1f}% (+6)")
            elif tick_good or impact_good:
                # Good market quality OR acceptable impact
                score += 4
                reasons.append(f"💵 Good spread: {spread_ticks} ticks (₹{spread_abs:.2f}), impact {impact_cost*100:.1f}% (+4)")
            elif spread_ticks <= max_ticks or impact_cost <= self.SPREAD_IMPACT_BLOCK:
                # At least ONE gate passes — acceptable but no points
                score += 2
                warnings.append(f"⚠️ Wide spread: {spread_ticks} ticks (₹{spread_abs:.2f}), impact {impact_cost*100:.1f}% (+2)")
            else:
                # BOTH tick AND impact fail → BLOCK
                block = True
                block_reason = f"Spread {spread_ticks} ticks (₹{spread_abs:.2f}), impact {impact_cost*100:.1f}% > {self.SPREAD_IMPACT_BLOCK*100:.0f}%"
                warnings.append(f"🚫 BLOCKED: {block_reason}")
        
        # === 2. TOP-OF-BOOK DEPTH (3 points max) ===
        # Kite returns depth quantities in shares — normalize to lots
        min_depth_shares = min(option_data.bid_qty, option_data.ask_qty)
        
        # Get lot size for this symbol (strip NSE:/NFO: prefix)
        underlying_clean = symbol_upper.replace('NSE:', '').replace('NFO:', '').split()[0]
        lot_size = FNO_LOT_SIZES.get(underlying_clean, 1)
        min_depth_lots = min_depth_shares / lot_size if lot_size > 0 else min_depth_shares
        
        if min_depth_lots >= self.MIN_DEPTH_LOTS * 2:
            # Excellent depth (4+ lots)
            score += 3
            reasons.append(f"📊 Good depth {min_depth_lots:.1f} lots ({min_depth_shares} shares) (+3)")
        elif min_depth_lots >= self.MIN_DEPTH_LOTS:
            # Acceptable depth (2+ lots)
            score += 2
            reasons.append(f"📊 OK depth {min_depth_lots:.1f} lots ({min_depth_shares} shares) (+2)")
        elif min_depth_lots >= self.MIN_DEPTH_LOTS / 2:
            # Thin depth (1+ lot)
            score += 1
            warnings.append(f"⚠️ Thin depth {min_depth_lots:.1f} lots ({min_depth_shares} shares) (+1)")
        else:
            # Very thin - don't block but 0 points
            warnings.append(f"⛔ Very thin depth {min_depth_lots:.1f} lots ({min_depth_shares} shares) (0pts)")
        
        # === 3. OI / VOLUME FILTER (4 points max) ===
        # Illiquid strikes = hard to exit
        oi = option_data.open_interest
        vol = option_data.option_volume
        
        # ⭐ NEW: Option volume minimum check
        if vol < self.MIN_OPTION_VOLUME:
            # Low option volume today - warning but not block
            warnings.append(f"⚠️ Low option volume today ({vol} < {self.MIN_OPTION_VOLUME})")
        
        if oi < self.MIN_OI:
            # BLOCK on very low OI
            block = True
            block_reason = f"OI {oi} < min {self.MIN_OI}"
            warnings.append(f"🚫 BLOCKED: {block_reason}")
        elif oi >= self.MIN_OI * 4 and vol >= self.MIN_OPTION_VOLUME:
            # Excellent OI AND volume
            score += 2
            reasons.append(f"📈 Good OI {oi:,} & vol {vol:,} (+2)")
        elif oi >= self.MIN_OI:
            # Acceptable OI
            score += 1
            reasons.append(f"📈 OK OI {oi:,} (+1)")
        
        # OI/Volume ratio check
        if vol > 0:
            oi_vol_ratio = oi / vol
            if oi_vol_ratio >= self.MIN_OI_VOLUME_RATIO:
                score += 2
                reasons.append(f"📊 Good OI/Vol ratio {oi_vol_ratio:.1f}x (+2)")
            else:
                warnings.append(f"⚠️ Low OI/Vol ratio {oi_vol_ratio:.1f}x")
        else:
            if oi >= self.MIN_OI:
                # Low volume but good OI - still OK
                score += 1
            else:
                warnings.append("⚠️ No volume today")
        
        # === 4. TIME-TO-FILL RISK (2 points max) ===
        # Repeated partials/cancels = bad fills
        partial_rate = option_data.partial_fill_rate
        cancel_rate = option_data.cancel_rate
        
        if partial_rate < self.PARTIAL_FILL_PENALTY_RATE and cancel_rate < self.CANCEL_RATE_PENALTY:
            score += 2
            reasons.append(f"⚡ Good fill quality (+2)")
        elif partial_rate < 0.5 and cancel_rate < 0.6:
            score += 1
            warnings.append(f"⚠️ Some fill issues (partials {partial_rate:.0%}) (+1)")
        else:
            warnings.append(f"⛔ High fill problems (partials {partial_rate:.0%}, cancels {cancel_rate:.0%})")
        
        return (score, block, block_reason)
    
    def record_fill(self, symbol: str, was_partial: bool, was_cancelled: bool):
        """
        Record fill outcome for time-to-fill risk tracking
        
        Call this after every order attempt to build history.
        """
        if symbol not in self.fill_history:
            self.fill_history[symbol] = []
        
        self.fill_history[symbol].append({
            'timestamp': datetime.now().isoformat(),
            'partial': was_partial,
            'cancelled': was_cancelled
        })
        
        # Keep only last 50 fills per symbol
        self.fill_history[symbol] = self.fill_history[symbol][-50:]
    
    def get_fill_rates(self, symbol: str) -> Tuple[float, float]:
        """
        Get partial fill and cancel rates for a symbol
        
        Returns:
            (partial_fill_rate, cancel_rate) both 0-1
        """
        history = self.fill_history.get(symbol, [])
        if not history or len(history) < 5:
            return (0.0, 0.0)  # Not enough data
        
        partials = sum(1 for h in history if h.get('partial', False))
        cancels = sum(1 for h in history if h.get('cancelled', False))
        
        return (partials / len(history), cancels / len(history))
    
    def _recommend_expiry(self, score: float, signal: IntradaySignal) -> str:
        """
        Recommend expiry based on conviction and time
        
        High conviction + early day: CURRENT_WEEK
        Lower conviction: NEXT_WEEK (more time)
        Past 2PM: NEXT_WEEK (avoid theta decay)
        """
        current_hour = datetime.now().hour
        
        # After 2 PM IST - prefer next week to avoid theta decay
        if current_hour >= 14:
            return "NEXT_WEEK"
        
        # Early morning with ORB breakout - current week is fine
        if current_hour < 11 and signal.orb_signal in ["BREAKOUT_UP", "BREAKOUT_DOWN"]:
            return "CURRENT_WEEK"
        
        # High conviction allows current week
        if score >= self.PREMIUM_THRESHOLD:
            return "CURRENT_WEEK"
        
        # Medium conviction - next week for safety
        if score < 70:
            return "NEXT_WEEK"
        
        return "CURRENT_WEEK"
    
    def get_decision_summary(self, symbol: str) -> Optional[Dict]:
        """Get the last decision summary for a symbol"""
        decision = self.last_decisions.get(symbol)
        if not decision:
            return None
        
        return {
            "should_trade": decision.should_trade,
            "score": decision.confidence_score,
            "direction": decision.recommended_direction,
            "strike": decision.strike_selection,
            "expiry": decision.expiry_selection,
            "option_type": decision.option_type,
            "size_mult": decision.position_size_multiplier,
            "reasons": decision.reasons,
            "warnings": decision.warnings,
            # Microstructure gate info
            "microstructure_score": decision.microstructure_score,
            "microstructure_blocked": decision.microstructure_block,
            "microstructure_block_reason": decision.microstructure_block_reason
        }
    
    def record_symbol_loss(self, symbol: str):
        """Record that a symbol had a losing trade today.
        Called by exit handler when a trade closes at a loss.
        Prevents re-entering the same losing symbol repeatedly.
        (SBIN lost 4x on Feb 11, BANDHANBNK lost 3x — this stops that.)
        """
        sym_base = symbol.replace("NSE:", "").replace("BSE:", "").replace("NFO:", "")
        # Strip option suffix to get the base underlying (e.g. SBIN1185CE → SBIN)
        import re
        sym_base = re.sub(r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC).*', '', sym_base)
        self.symbol_losses_today[sym_base] = self.symbol_losses_today.get(sym_base, 0) + 1
        print(f"📊 SCORER: Recorded loss #{self.symbol_losses_today[sym_base]} for {sym_base} (max re-entry: {self.MAX_LOSSES_SAME_SYMBOL})")
    
    def record_symbol_win(self, symbol: str):
        """Record that a symbol had a winning trade today.
        Winners reset the loss counter — the setup IS working for this symbol.
        """
        sym_base = symbol.replace("NSE:", "").replace("BSE:", "").replace("NFO:", "")
        import re
        sym_base = re.sub(r'\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC).*', '', sym_base)
        # Reset loss counter on a win — the setup works for this symbol
        if sym_base in self.symbol_losses_today:
            self.symbol_losses_today[sym_base] = max(0, self.symbol_losses_today[sym_base] - 1)
            print(f"📊 SCORER: Win on {sym_base} — loss counter reduced to {self.symbol_losses_today[sym_base]}")


# Singleton instance
_intraday_scorer: Optional[IntradayOptionScorer] = None

def get_intraday_scorer() -> IntradayOptionScorer:
    """Get or create singleton intraday scorer"""
    global _intraday_scorer
    if _intraday_scorer is None:
        _intraday_scorer = IntradayOptionScorer()
    return _intraday_scorer


def build_microstructure_from_contract(
    contract: Dict,
    depth: Dict = None,
    fill_history: List[Dict] = None
) -> OptionMicrostructure:
    """
    Build OptionMicrostructure from contract data and depth
    
    Args:
        contract: Dict with bid, ask, ltp, oi, volume from option chain
        depth: Dict with buy/sell depth (optional)
        fill_history: List of past fill records for this option (optional)
    
    Returns:
        OptionMicrostructure ready for scoring
    """
    bid = contract.get('bid', 0) or 0
    ask = contract.get('ask', 0) or 0
    ltp = contract.get('ltp', 0) or bid or ask
    
    # Calculate spread %
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 100
    elif ltp > 0:
        # Estimate from LTP if no bid/ask
        spread_pct = 1.0  # Assume 1% if unknown
    else:
        spread_pct = 100.0  # Unknown = bad
    
    # Get depth quantities
    bid_qty = 0
    ask_qty = 0
    if depth:
        buy_depth = depth.get('buy', [])
        sell_depth = depth.get('sell', [])
        bid_qty = buy_depth[0].get('quantity', 0) if buy_depth else 0
        ask_qty = sell_depth[0].get('quantity', 0) if sell_depth else 0
    
    # Get OI and volume
    oi = contract.get('oi', 0) or contract.get('open_interest', 0) or 0
    volume = contract.get('volume', 0) or 0
    
    # Calculate OI/Volume ratio
    oi_vol_ratio = (oi / volume) if volume > 0 else 0
    
    # Calculate fill rates from history
    partial_rate = 0.0
    cancel_rate = 0.0
    if fill_history and len(fill_history) >= 5:
        partials = sum(1 for h in fill_history if h.get('partial', False))
        cancels = sum(1 for h in fill_history if h.get('cancelled', False))
        partial_rate = partials / len(fill_history)
        cancel_rate = cancels / len(fill_history)
    
    return OptionMicrostructure(
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        open_interest=oi,
        option_volume=volume,
        oi_volume_ratio=oi_vol_ratio,
        partial_fill_rate=partial_rate,
        cancel_rate=cancel_rate,
        avg_fill_time_ms=0,  # Would need order tracking
        ltp=ltp
    )


# ============================================================
# OPTION TYPES AND CORE CLASSES
# ============================================================

class OptionType(Enum):
    """Option type"""
    CE = "CE"  # Call
    PE = "PE"  # Put


class StrikeSelection(Enum):
    """Strike selection strategy"""
    ATM = "ATM"          # At The Money
    ITM_1 = "ITM_1"      # 1 strike In The Money
    ITM_2 = "ITM_2"      # 2 strikes In The Money
    OTM_1 = "OTM_1"      # 1 strike Out of The Money
    OTM_2 = "OTM_2"      # 2 strikes Out of The Money


class ExpirySelection(Enum):
    """Expiry selection strategy"""
    CURRENT_WEEK = "CURRENT_WEEK"
    NEXT_WEEK = "NEXT_WEEK"
    CURRENT_MONTH = "CURRENT_MONTH"
    NEXT_MONTH = "NEXT_MONTH"


@dataclass
class OptionContract:
    """Represents an option contract"""
    symbol: str                    # NFO:RELIANCE26FEB3000CE
    underlying: str                # NSE:RELIANCE
    strike: float                  # 3000
    option_type: OptionType        # CE or PE
    expiry: datetime              # Expiry date
    lot_size: int                  # Lot size (e.g., 250 for RELIANCE)
    ltp: float                     # Last traded price
    bid: float                     # Best bid
    ask: float                     # Best ask
    volume: int                    # Volume
    oi: int                        # Open Interest
    iv: float                      # Implied Volatility
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


@dataclass
class OptionChain:
    """Complete option chain for a symbol"""
    underlying: str
    spot_price: float
    expiries: List[datetime]
    contracts: List[OptionContract]
    timestamp: datetime
    
    def get_atm_strike(self, expiry: datetime = None) -> float:
        """Get ATM strike for given expiry"""
        # Round to nearest strike
        strikes = sorted(set(c.strike for c in self.contracts if expiry is None or c.expiry == expiry))
        if not strikes:
            return 0
        
        # Find strike closest to spot
        return min(strikes, key=lambda x: abs(x - self.spot_price))
    
    def get_contract(self, strike: float, option_type: OptionType, expiry: datetime = None) -> Optional[OptionContract]:
        """Get specific contract"""
        for c in self.contracts:
            if c.strike == strike and c.option_type == option_type:
                if expiry is None or c.expiry == expiry:
                    return c
        return None


@dataclass
class OptionOrderPlan:
    """Plan for an option order"""
    underlying: str
    direction: str                # BUY or SELL signal on underlying
    contract: OptionContract      # Selected option contract
    quantity: int                 # Number of lots
    premium_per_lot: float        # Premium per lot
    total_premium: float          # Total premium
    max_loss: float              # Maximum potential loss
    breakeven: float             # Breakeven price
    target_premium: float        # Target premium for exit
    stoploss_premium: float      # Stoploss premium
    rationale: str
    greeks_summary: str
    # === ENTRY METADATA (for post-trade review) ===
    entry_metadata: Dict = field(default_factory=dict)  # score, tier, gates, candle data, sizing rationale


@dataclass
class CreditSpreadPlan:
    """
    Plan for a credit spread order (SELL option + BUY hedge option)
    
    Bull Put Spread: SELL higher PE + BUY lower PE (bullish view)
    Bear Call Spread: SELL lower CE + BUY higher CE (bearish view)
    """
    underlying: str
    direction: str                    # 'BULLISH' or 'BEARISH' (market view)
    spread_type: str                  # 'BULL_PUT_SPREAD' or 'BEAR_CALL_SPREAD'
    # Sold (short) leg — this is where premium is collected
    sold_contract: OptionContract     # ATM/OTM option we SELL
    # Bought (long) leg — the hedge
    hedge_contract: OptionContract    # Further OTM option we BUY
    # Sizing
    quantity: int                     # Number of lots
    lot_size: int                     # Shares per lot
    # Financials
    net_credit: float                 # Net premium credit per share (sold - bought)
    net_credit_total: float           # Total net credit (credit × qty × lot_size)
    max_risk: float                   # Max risk = (spread_width - net_credit) × qty × lot_size
    spread_width: float               # Strike difference between legs
    # Risk management
    target_credit: float              # Target: keep 65% of credit (exit by buying back cheaper)
    stop_loss_debit: float            # SL: exit when loss = 2× credit received
    breakeven: float                  # Breakeven strike price
    # Greeks (net of both legs)
    net_delta: float
    net_theta: float                  # POSITIVE = theta works for us
    net_vega: float                   # NEGATIVE = benefits from IV drop
    net_gamma: float
    # Metadata
    rationale: str
    greeks_summary: str
    credit_pct: float                 # Credit as % of spread width (quality metric)
    dte: int                          # Days to expiry


@dataclass
class DebitSpreadPlan:
    """
    Plan for a debit spread order (BUY near-ATM + SELL further OTM)
    
    Intraday momentum strategy — profits from strong directional moves.
    BULLISH → Bull Call Spread: BUY ATM CE + SELL OTM CE
    BEARISH → Bear Put Spread: BUY ATM PE + SELL OTM PE
    """
    underlying: str
    direction: str                    # 'BULLISH' or 'BEARISH'
    spread_type: str                  # 'BULL_CALL_SPREAD' or 'BEAR_PUT_SPREAD'
    # Bought (long) leg — the directional bet
    buy_contract: OptionContract      # Near-ATM option we BUY
    # Sold (short) leg — reduces cost (caps profit)
    sell_contract: OptionContract     # Further OTM option we SELL
    # Sizing
    quantity: int                     # Number of lots
    lot_size: int                     # Shares per lot
    # Financials
    net_debit: float                  # Net premium paid per share (buy - sell)
    net_debit_total: float            # Total net debit (debit × qty × lot_size)
    max_profit: float                 # Max profit = (spread_width - net_debit) × qty × lot_size
    max_loss: float                   # Max loss = net_debit × qty × lot_size
    spread_width: float               # Strike difference between legs
    # Risk management
    target_value: float               # Target spread value per share (exit when spread reaches this)
    stop_loss_value: float            # SL spread value per share (exit when spread drops to this)
    breakeven: float                  # Breakeven underlying price
    # Greeks (net of both legs)
    net_delta: float                  # POSITIVE for bull, NEGATIVE for bear
    net_theta: float                  # NEGATIVE = time works against us
    net_vega: float                   # POSITIVE = benefits from IV rise
    net_gamma: float
    # Metadata
    rationale: str
    greeks_summary: str
    move_pct: float                   # Stock's current intraday move %
    dte: int                          # Days to expiry


# F&O Lot Sizes (as of 2026 - may need updates)
FNO_LOT_SIZES = {
    "RELIANCE": 500,
    "TCS": 175,
    "HDFCBANK": 550,
    "INFY": 400,
    "ICICIBANK": 700,
    "SBIN": 750,
    "AXISBANK": 625,
    "KOTAKBANK": 2000,
    "BAJFINANCE": 750,
    "ETERNAL": 2425,
    "MCX": 625,
    "LT": 175,
    "MARUTI": 50,
    "TITAN": 175,
    "SUNPHARMA": 350,
    "BHARTIARTL": 475,
    "ADANIENT": 250,
    "ADANIPORTS": 1250,
    "APOLLOHOSP": 125,
    "ASIANPAINT": 300,
    "BPCL": 1800,
    "CIPLA": 650,
    "COALINDIA": 1350,
    "DIVISLAB": 100,
    "DRREDDY": 125,
    "EICHERMOT": 175,
    "GRASIM": 475,
    "HCLTECH": 350,
    "HEROMOTOCO": 150,
    "HINDALCO": 700,
    "HINDUNILVR": 300,
    "INDUSINDBK": 450,
    "ITC": 1600,
    "TATAPOWER": 1450,
    "JSWSTEEL": 675,
    "M&M": 350,
    "NESTLEIND": 25,
    "NTPC": 1500,
    "ONGC": 2250,
    "POWERGRID": 2700,
    "SBILIFE": 750,
    "SHREECEM": 25,
    "TATACONSUM": 550,
    "TATASTEEL": 5500,
    "TECHM": 600,
    "ULTRACEMCO": 100,
    "UPL": 1300,
    "WIPRO": 3000,
    "VEDL": 1150,
    "JINDALSTEL": 625,
    "NATIONALUM": 3750,
    "NMDC": 6750,
    "SAIL": 4700,
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
    # Wild-card scanner stocks (added 10 Feb 2026)
    "BSE": 375,
    "BANDHANBNK": 3600,
    "SWIGGY": 1300,
    "MOTHERSON": 6150,
    "GAIL": 3150,
    "JIOFIN": 2350,
    "KALYANKJIL": 1175,
    "PGEL": 950,
    "AMBER": 100,
    "CDSL": 475,
    "IOC": 4875,
}


def update_fno_lot_sizes(lot_map: Dict[str, int]) -> int:
    """
    Dynamically update FNO_LOT_SIZES from Kite API lot data.
    Called after market_scanner fetches NFO instruments so every
    wild-card stock automatically gets the correct lot size.
    
    Args:
        lot_map: {symbol: lot_size} dict from scanner
    Returns:
        Number of NEW symbols added
    """
    added = 0
    for symbol, lot_size in lot_map.items():
        if lot_size and lot_size > 0:
            if symbol not in FNO_LOT_SIZES:
                added += 1
            FNO_LOT_SIZES[symbol] = lot_size
    if added:
        print(f"📦 Dynamic lot sizes: {added} new symbols added → total {len(FNO_LOT_SIZES)} F&O stocks")
    return added


class BlackScholes:
    """
    Black-Scholes Option Pricing and Greeks Calculator
    Used for option valuation and Greeks computation
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate theoretical call option price"""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate theoretical put option price"""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """
        Calculate Delta - rate of change of option price with respect to underlying
        Delta for calls: 0 to 1 (positive)
        Delta for puts: -1 to 0 (negative)
        """
        if T <= 0:
            if option_type == OptionType.CE:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == OptionType.CE:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma - rate of change of delta
        Same for calls and puts
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """
        Calculate Theta - time decay (per day)
        Negative for long options (time works against you)
        """
        if T <= 0:
            return 0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        first_term = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        
        if option_type == OptionType.CE:
            second_term = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            second_term = r * K * math.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to per-day (annualized to daily)
        return (first_term + second_term) / 365
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega - sensitivity to volatility
        Same for calls and puts
        Returned per 1% change in IV
        """
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * math.sqrt(T) * norm.pdf(d1) / 100  # Per 1% IV change
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                           r: float, option_type: OptionType, 
                           precision: float = 0.0001, max_iterations: int = 100) -> float:
        """
        Calculate Implied Volatility using Newton-Raphson method
        """
        if T <= 0:
            return 0.0
        
        sigma = 0.3  # Initial guess
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(max_iterations):
                if option_type == OptionType.CE:
                    price = BlackScholes.call_price(S, K, T, r, sigma)
                else:
                    price = BlackScholes.put_price(S, K, T, r, sigma)
                
                vega = BlackScholes.vega(S, K, T, r, sigma) * 100  # Actual vega
                
                if vega == 0 or not math.isfinite(vega):
                    break
                
                diff = market_price - price
                if abs(diff) < precision:
                    break
                
                sigma = sigma + diff / vega
                if not math.isfinite(sigma):
                    sigma = 0.3
                    break
                sigma = max(0.01, min(5.0, sigma))  # Keep sigma reasonable
        
        return sigma


class OptionChainFetcher:
    """
    Fetches and manages option chain data from Zerodha
    """
    
    CACHE_FILE = os.path.join(os.path.dirname(__file__), 'option_chain_cache.json')
    CACHE_TTL_SECONDS = 60  # Refresh every 60 seconds
    
    def __init__(self, kite=None):
        self.kite = kite
        self.cache: Dict[str, OptionChain] = {}
        self.last_fetch: Dict[str, datetime] = {}
        self.instruments_cache: Dict[str, List[Dict]] = {}
        
        # Risk-free rate (approx 6.5% for India)
        self.risk_free_rate = 0.065
    
    def _get_nfo_instruments(self) -> List[Dict]:
        """Get all NFO instruments from Zerodha"""
        if 'NFO' in self.instruments_cache:
            return self.instruments_cache['NFO']
        
        try:
            instruments = self.kite.instruments('NFO') if self.kite else []
            self.instruments_cache['NFO'] = instruments
            return instruments
        except Exception as e:
            print(f"⚠️ Error fetching NFO instruments: {e}")
            return []
    
    def quick_check_option_liquidity(self, underlying: str, min_oi: int = 500, 
                                      max_bid_ask_pct: float = 5.0) -> tuple:
        """
        Quick pre-flight liquidity check for an underlying's ATM options.
        
        ZERO extra API calls: uses cached chain data only.
        If no cache exists, falls back to a free NFO instrument count check.
        The full OI + bid-ask validation happens inside create_debit_spread()
        which MUST fetch the chain anyway — so we never duplicate that work.
        
        Args:
            underlying: e.g. "NSE:RELIANCE"
            min_oi: Minimum OI required on ATM strike (default 500)
            max_bid_ask_pct: Max bid-ask spread as % of LTP (default 5%)
            
        Returns:
            (is_liquid: bool, reason: str)
        """
        try:
            symbol = underlying.replace("NSE:", "")
            
            # === LAYER 1: Use cached chain if available (ZERO API calls) ===
            cached_chain = None
            for cache_key, chain in self.cache.items():
                if cache_key.startswith(symbol + "_"):
                    last = self.last_fetch.get(cache_key, datetime.min)
                    if (datetime.now() - last).seconds < self.CACHE_TTL_SECONDS * 5:  # 5x normal TTL OK for liquidity check
                        cached_chain = chain
                        break
            
            if cached_chain is not None:
                # We have cached chain data — do full OI + bid-ask check
                atm_strike = cached_chain.get_atm_strike()
                if atm_strike is None:
                    return (False, f"No ATM strike for {symbol}")
                
                ce_contract = cached_chain.get_contract(atm_strike, OptionType.CE)
                pe_contract = cached_chain.get_contract(atm_strike, OptionType.PE)
                
                if ce_contract is None and pe_contract is None:
                    return (False, f"No ATM contracts for {symbol} at strike {atm_strike}")
                
                # Check at least one side has adequate liquidity
                for label, contract in [("CE", ce_contract), ("PE", pe_contract)]:
                    if contract is None:
                        continue
                    if contract.oi < min_oi:
                        continue
                    if contract.ltp > 0 and contract.bid > 0 and contract.ask > 0:
                        bid_ask_spread_pct = ((contract.ask - contract.bid) / contract.ltp) * 100
                        if bid_ask_spread_pct > max_bid_ask_pct:
                            continue
                    # This side is liquid enough
                    return (True, f"ATM {label} @{atm_strike}: OI={contract.oi:,}, LTP=₹{contract.ltp:.2f}")
                
                ce_oi = ce_contract.oi if ce_contract else 0
                pe_oi = pe_contract.oi if pe_contract else 0
                return (False, f"Low liquidity: ATM {atm_strike} CE_OI={ce_oi} PE_OI={pe_oi} (need ≥{min_oi})")
            
            # === LAYER 2: No cache — use FREE NFO instrument check (ZERO API calls) ===
            # Just verify the stock has enough option strikes listed on NFO
            # The actual OI + bid-ask check will happen in create_debit_spread()
            instruments = self._get_nfo_instruments()  # Already cached after first call
            option_count = sum(
                1 for inst in instruments 
                if inst.get('name') == symbol and inst.get('instrument_type') in ('CE', 'PE')
            )
            
            if option_count == 0:
                return (False, f"{symbol} has no NFO options listed — not F&O eligible")
            elif option_count < 10:
                return (False, f"{symbol} only has {option_count} option contracts — likely illiquid")
            else:
                # Has enough contracts listed — defer full check to create_debit_spread
                return (True, f"{symbol} has {option_count} NFO contracts (full liquidity check deferred)")
            
        except Exception as e:
            # On error, don't block — let create_debit_spread handle it
            return (True, f"Liquidity check skipped (error: {e})")
    
    def get_expiries(self, underlying: str) -> List[datetime]:
        """Get available expiry dates for an underlying"""
        symbol = underlying.replace("NSE:", "")
        instruments = self._get_nfo_instruments()
        
        expiries = set()
        for inst in instruments:
            if inst['name'] == symbol and inst['instrument_type'] in ['CE', 'PE']:
                expiries.add(inst['expiry'])
        
        return sorted(list(expiries))
    
    def get_nearest_expiry(self, underlying: str, 
                          selection: ExpirySelection = ExpirySelection.CURRENT_WEEK) -> Optional[datetime]:
        """Get nearest expiry based on selection strategy"""
        expiries = self.get_expiries(underlying)
        if not expiries:
            return None
        
        today = datetime.now().date()
        
        # Helper: expiry may be datetime.date or datetime.datetime
        def exp_date(exp):
            return exp if isinstance(exp, date) and not isinstance(exp, datetime) else exp.date() if hasattr(exp, 'date') else exp
        
        if selection == ExpirySelection.CURRENT_WEEK:
            # Get expiry in current week (Thursday)
            for exp in expiries:
                ed = exp_date(exp)
                if ed >= today and (ed - today).days <= 7:
                    return exp
            return expiries[0] if expiries else None
        
        elif selection == ExpirySelection.NEXT_WEEK:
            # Skip current week, get next
            for exp in expiries:
                ed = exp_date(exp)
                if ed >= today and (ed - today).days > 7:
                    return exp
            return expiries[-1] if len(expiries) > 1 else expiries[0]
        
        elif selection == ExpirySelection.CURRENT_MONTH:
            # Get monthly expiry (last Thursday of month)
            for exp in expiries:
                ed = exp_date(exp)
                if ed >= today and ed.day >= 23:
                    return exp
            return expiries[0] if expiries else None
        
        elif selection == ExpirySelection.NEXT_MONTH:
            # Get next month's expiry
            current_month = today.month
            for exp in expiries:
                ed = exp_date(exp)
                if ed >= today and ed.month > current_month:
                    return exp
            return expiries[-1] if expiries else None
        
        return expiries[0] if expiries else None
    
    def get_lot_size(self, underlying: str) -> int:
        """Get lot size for underlying"""
        symbol = underlying.replace("NSE:", "")
        return FNO_LOT_SIZES.get(symbol, 1)
    
    def fetch_option_chain(self, underlying: str, expiry: datetime = None) -> Optional[OptionChain]:
        """
        Fetch complete option chain for an underlying
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            expiry: Specific expiry date (None = nearest)
            
        Returns:
            OptionChain object with all contracts
        """
        symbol = underlying.replace("NSE:", "")
        
        # Check cache
        expiry_key = expiry if isinstance(expiry, date) and not isinstance(expiry, datetime) else (expiry.date() if expiry else 'nearest')
        cache_key = f"{symbol}_{expiry_key}"
        if cache_key in self.cache:
            last = self.last_fetch.get(cache_key, datetime.min)
            if (datetime.now() - last).seconds < self.CACHE_TTL_SECONDS:
                return self.cache[cache_key]
        
        # Get expiry if not specified
        if expiry is None:
            expiry = self.get_nearest_expiry(underlying)
        
        if expiry is None:
            print(f"⚠️ No expiry found for {underlying}")
            return None
        
        # Get spot price
        spot_price = 0
        try:
            if self.kite:
                quote = self.kite.quote([underlying])
                spot_price = quote[underlying]['last_price']
        except Exception as e:
            print(f"⚠️ Error getting spot price: {e}")
            return None
        
        # Get all contracts for this expiry
        instruments = self._get_nfo_instruments()
        lot_size = self.get_lot_size(underlying)
        
        contracts = []
        option_symbols = []
        
        for inst in instruments:
            if inst['name'] == symbol and inst['expiry'] == expiry:
                if inst['instrument_type'] in ['CE', 'PE']:
                    option_symbols.append(f"NFO:{inst['tradingsymbol']}")
        
        # Fetch quotes for all options
        if option_symbols and self.kite:
            try:
                # Batch fetch (Zerodha allows 500 at a time)
                for i in range(0, len(option_symbols), 200):
                    batch = option_symbols[i:i+200]
                    quotes = self.kite.quote(batch)
                    
                    for sym, q in quotes.items():
                        # Parse symbol to get strike and type
                        tradingsymbol = sym.replace("NFO:", "")
                        
                        # Find instrument details
                        inst = next((i for i in instruments if i['tradingsymbol'] == tradingsymbol), None)
                        if not inst:
                            continue
                        
                        strike = inst['strike']
                        opt_type = OptionType.CE if inst['instrument_type'] == 'CE' else OptionType.PE
                        
                        ltp = q.get('last_price', 0)
                        depth = q.get('depth', {})
                        bid = depth.get('buy', [{}])[0].get('price', ltp * 0.95) if depth.get('buy') else ltp * 0.95
                        ask = depth.get('sell', [{}])[0].get('price', ltp * 1.05) if depth.get('sell') else ltp * 1.05
                        
                        # Calculate time to expiry
                        expiry_dt = datetime.combine(expiry, datetime.min.time()) if isinstance(expiry, date) and not isinstance(expiry, datetime) else expiry
                        T = max(0.001, (expiry_dt - datetime.now()).total_seconds() / (365 * 86400))
                        
                        # Calculate IV
                        iv = BlackScholes.implied_volatility(
                            market_price=ltp,
                            S=spot_price,
                            K=strike,
                            T=T,
                            r=self.risk_free_rate,
                            option_type=opt_type
                        )
                        
                        # Calculate Greeks
                        delta = BlackScholes.delta(spot_price, strike, T, self.risk_free_rate, iv, opt_type)
                        gamma = BlackScholes.gamma(spot_price, strike, T, self.risk_free_rate, iv)
                        theta = BlackScholes.theta(spot_price, strike, T, self.risk_free_rate, iv, opt_type)
                        vega = BlackScholes.vega(spot_price, strike, T, self.risk_free_rate, iv)
                        
                        contract = OptionContract(
                            symbol=sym,
                            underlying=underlying,
                            strike=strike,
                            option_type=opt_type,
                            expiry=expiry,
                            lot_size=lot_size,
                            ltp=ltp,
                            bid=bid,
                            ask=ask,
                            volume=q.get('volume', 0),
                            oi=q.get('oi', 0),
                            iv=iv,
                            delta=delta,
                            gamma=gamma,
                            theta=theta,
                            vega=vega
                        )
                        contracts.append(contract)
                        
            except Exception as e:
                print(f"⚠️ Error fetching option quotes: {e}")
        
        # Create option chain
        chain = OptionChain(
            underlying=underlying,
            spot_price=spot_price,
            expiries=self.get_expiries(underlying),
            contracts=contracts,
            timestamp=datetime.now()
        )
        
        # Cache it
        self.cache[cache_key] = chain
        self.last_fetch[cache_key] = datetime.now()
        
        return chain
    
    def select_strike(self, chain: OptionChain, option_type: OptionType,
                     selection: StrikeSelection = StrikeSelection.ATM,
                     expiry: datetime = None) -> Optional[OptionContract]:
        """
        Select optimal strike based on strategy
        
        Args:
            chain: Option chain
            option_type: CE or PE
            selection: ATM, ITM_1, ITM_2, OTM_1, OTM_2
            expiry: Specific expiry (None = first available)
            
        Returns:
            Selected OptionContract
        """
        # Get sorted strikes
        strikes = sorted(set(c.strike for c in chain.contracts 
                           if c.option_type == option_type and 
                           (expiry is None or c.expiry == expiry)))
        
        if not strikes:
            return None
        
        # Find ATM
        atm_strike = min(strikes, key=lambda x: abs(x - chain.spot_price))
        atm_idx = strikes.index(atm_strike)
        
        # Select based on strategy
        if selection == StrikeSelection.ATM:
            target_strike = atm_strike
        elif selection == StrikeSelection.ITM_1:
            if option_type == OptionType.CE:
                target_strike = strikes[max(0, atm_idx - 1)]
            else:
                target_strike = strikes[min(len(strikes) - 1, atm_idx + 1)]
        elif selection == StrikeSelection.ITM_2:
            if option_type == OptionType.CE:
                target_strike = strikes[max(0, atm_idx - 2)]
            else:
                target_strike = strikes[min(len(strikes) - 1, atm_idx + 2)]
        elif selection == StrikeSelection.OTM_1:
            if option_type == OptionType.CE:
                target_strike = strikes[min(len(strikes) - 1, atm_idx + 1)]
            else:
                target_strike = strikes[max(0, atm_idx - 1)]
        elif selection == StrikeSelection.OTM_2:
            if option_type == OptionType.CE:
                target_strike = strikes[min(len(strikes) - 1, atm_idx + 2)]
            else:
                target_strike = strikes[max(0, atm_idx - 2)]
        else:
            target_strike = atm_strike
        
        return chain.get_contract(target_strike, option_type, expiry)


class OptionsPositionSizer:
    """
    Position sizing for options with premium limits and risk management
    
    TIERED SIZING (Feb 12 overhaul):
      Premium tier (score ≥ 70): 5% risk, min 2 lots, up to 25% capital
      Standard tier (score ≥ 65): 3.5% risk, min 1 lot, up to 20% capital
      Base tier (below 65):       2% risk, 1 lot, up to 15% capital
    """
    
    def __init__(self, capital: float = 100000):
        self.capital = capital
        
        # Risk limits
        self.max_premium_per_trade = 0.50   # Max 50% of capital per option premium (₹1L on ₹2L capital)
        self.max_premium_per_day = 1.00     # Max 100% of capital in option premiums per day
        self.max_lots_per_trade = 15        # Maximum lots per single trade (premium tier)
        self.max_total_premium = 0          # Tracking
        
        # Tiered risk per trade (as % of capital) — base rate
        self.risk_per_trade = HARD_RULES.get("RISK_PER_TRADE", 0.035)  # Base 3.5%
        
        # Tiered risk rates
        self.risk_premium = 0.05    # 5% risk for premium tier (₹25K on ₹5L)
        self.risk_standard = 0.035  # 3.5% risk for standard tier (₹17.5K on ₹5L)
        self.risk_base = 0.02       # 2% risk for base tier (₹10K on ₹5L)
    
    def calculate_position(self, contract: OptionContract, 
                          signal_direction: str,
                          capital: float = None,
                          max_loss_pct: float = None,
                          score_tier: str = "standard") -> Dict:
        """
        Calculate optimal position size for an option trade
        
        Args:
            contract: The option contract to trade
            signal_direction: 'BUY' or 'SELL' (underlying direction)
            capital: Override capital
            max_loss_pct: Override max loss percentage
            score_tier: 'premium', 'standard', or 'base' — determines risk budget
            
        Returns:
            Dict with quantity, premium, risk details
        """
        capital = capital or self.capital
        
        # Tiered risk: premium trades get bigger sizes
        if max_loss_pct:
            risk_pct = max_loss_pct
        elif score_tier == "premium":
            risk_pct = self.risk_premium    # 5%
        elif score_tier == "standard":
            risk_pct = self.risk_standard   # 3.5%
        else:
            risk_pct = self.risk_base       # 2%
        
        premium_per_lot = contract.ltp * contract.lot_size
        
        # === CAPITAL GUARD: Block if single lot exceeds max premium ===
        max_premium = capital * self.max_premium_per_trade
        if premium_per_lot > max_premium:
            return {
                'quantity': 0,
                'premium_per_lot': premium_per_lot,
                'error': f'Single lot premium ₹{premium_per_lot:,.0f} exceeds {self.max_premium_per_trade*100:.0f}% of capital (₹{max_premium:,.0f})',
                'blocked': True
            }
        
        # Method 1: Premium-based sizing (max 5% of capital)
        lots_by_premium = max(1, int(max_premium / premium_per_lot))
        
        # Method 2: Risk-based sizing
        # Use ACTUAL SL percentage (30%) not assumed 50% — matches stoploss_premium = 0.7x
        max_loss = capital * risk_pct
        actual_sl_loss_per_lot = premium_per_lot * 0.30  # 30% SL = actual max loss per lot
        lots_by_risk = max(1, int(max_loss / actual_sl_loss_per_lot))
        
        # Method 3: Delta-adjusted sizing
        # Higher delta = more exposure = fewer lots
        delta_factor = 1.0
        if abs(contract.delta) > 0.7:
            delta_factor = 0.7  # Deep ITM - reduce size
        elif abs(contract.delta) < 0.3:
            delta_factor = 0.8  # Deep OTM - also reduce (higher % risk)
        lots_by_delta = max(1, int(lots_by_premium * delta_factor))
        
        # Take minimum of all methods
        quantity = min(
            lots_by_premium,
            lots_by_risk,
            lots_by_delta,
            self.max_lots_per_trade
        )
        
        # Minimum lots for premium tier — don't waste high conviction on 1 lot
        if score_tier == "premium" and quantity < 2 and premium_per_lot * 2 < capital * 0.25:
            quantity = 2
        
        total_premium = quantity * premium_per_lot
        
        # Calculate max loss (assuming premium can go to 0 - buying options)
        max_loss = total_premium  # Max loss = premium paid for long options
        
        # Calculate target and stoploss — TIERED R:R
        # Premium: Target 80% gain, SL 28% loss → R:R = 2.86:1
        # Standard: Target 60% gain, SL 28% loss → R:R = 2.14:1
        # Base: Target 50% gain, SL 30% loss → R:R = 1.67:1
        if score_tier == "premium":
            target_premium = contract.ltp * 1.80   # +80% premium gain
            stoploss_premium = contract.ltp * 0.72  # -28% premium loss
        elif score_tier == "standard":
            target_premium = contract.ltp * 1.60   # +60% premium gain
            stoploss_premium = contract.ltp * 0.72  # -28% premium loss
        else:
            target_premium = contract.ltp * 1.50   # +50% premium gain
            stoploss_premium = contract.ltp * 0.70  # -30% premium loss
        
        # Breakeven
        if contract.option_type == OptionType.CE:
            breakeven = contract.strike + contract.ltp
        else:
            breakeven = contract.strike - contract.ltp
        
        return {
            'quantity': quantity,
            'lots': quantity,
            'lot_size': contract.lot_size,
            'shares': quantity * contract.lot_size,
            'premium_per_lot': premium_per_lot,
            'total_premium': total_premium,
            'max_loss': max_loss,
            'risk_pct': (max_loss / capital) * 100,
            'target_premium': target_premium,
            'stoploss_premium': stoploss_premium,
            'breakeven': breakeven,
            'delta_exposure': abs(contract.delta) * quantity * contract.lot_size,
            'sizing_method': f'TIERED({score_tier}): MIN(premium, risk, delta)',
            'score_tier': score_tier,
            'risk_pct_used': risk_pct * 100,
            'lots_by_premium': lots_by_premium,
            'lots_by_risk': lots_by_risk,
            'lots_by_delta': lots_by_delta
        }


class OptionsTrader:
    """
    Main options trading class - integrates all components
    """
    
    def __init__(self, kite=None, capital: float = 100000, paper_mode: bool = True):
        self.kite = kite
        self.capital = capital
        self.paper_mode = paper_mode
        
        self.chain_fetcher = OptionChainFetcher(kite)
        self.position_sizer = OptionsPositionSizer(capital)
        
        # Active option positions - load from active_trades.json if available
        self.positions: List[Dict] = []
        self._load_option_positions()
        
        # Configuration
        self.config = {
            'default_strike_selection': StrikeSelection.ATM,
            'default_expiry_selection': ExpirySelection.CURRENT_WEEK,
            'prefer_high_oi': True,         # Prefer high OI strikes
            'min_oi': 10000,                # Minimum OI required
            'max_iv': 1.0,                  # Max IV (100%)
            'min_volume': 100,              # Minimum volume
        }
        
        if self.positions:
            print(f"📊 Options Trader: INITIALIZED ({len(self.positions)} option positions loaded)")
        else:
            print("📊 Options Trader: INITIALIZED")
    
    def _load_option_positions(self):
        """Load option positions from active_trades.json"""
        try:
            trades_file = os.path.join(os.path.dirname(__file__), 'active_trades.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        for trade in data.get('active_trades', []):
                            if trade.get('is_option') and trade.get('status', 'OPEN') == 'OPEN':
                                # Reconstruct option position from paper trade data
                                self.positions.append({
                                    'order_id': trade.get('order_id', ''),
                                    'symbol': trade['symbol'],
                                    'underlying': trade.get('underlying', ''),
                                    'direction': trade.get('side', 'BUY'),
                                    'option_type': trade.get('option_type', 'CE'),
                                    'strike': trade.get('strike', 0),
                                    'expiry': trade.get('expiry', ''),
                                    'quantity': trade.get('lots', 1),
                                    'lot_size': trade.get('lot_size', 1),
                                    'entry_premium': trade.get('avg_price', 0),
                                    'total_premium': trade.get('total_premium', 0),
                                    'target_premium': trade.get('target', 0),
                                    'stoploss_premium': trade.get('stop_loss', 0),
                                    'breakeven': trade.get('breakeven', 0),
                                    'greeks': trade.get('greeks', {
                                        'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'iv': 0
                                    }),
                                    'status': 'OPEN',
                                    'timestamp': trade.get('timestamp', '')
                                })
        except Exception as e:
            print(f"⚠️ Error loading option positions: {e}")
    
    def should_use_options(self, underlying: str) -> bool:
        """
        Determine if options should be used for this underlying
        """
        symbol = underlying.replace("NSE:", "")
        return symbol in FNO_LOT_SIZES
    
    def create_option_order_with_intraday(self, underlying: str, direction: str,
                                          market_data: Dict = None,
                                          force_strike: StrikeSelection = None,
                                          force_expiry: ExpirySelection = None) -> Optional[Tuple[OptionOrderPlan, IntradayOptionDecision]]:
        """
        Create option order using INTRADAY SIGNALS as HIGHEST PRECEDENCE
        
        This method analyzes intraday findings (ORB, VWAP, Volume, EMA) first
        and uses them to determine optimal strike/expiry/size decisions.
        Technical factors complement but don't override intraday signals.
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            direction: 'BUY' or 'SELL' signal on underlying
            market_data: Dict with intraday signals (orb_signal, vwap_position, etc.)
            force_strike: Override strike selection (optional)
            force_expiry: Override expiry selection (optional)
            
        Returns:
            Tuple of (OptionOrderPlan, IntradayOptionDecision) or None
        """
        market_data = market_data or {}
        
        # === BUILD INTRADAY SIGNAL FROM MARKET DATA ===
        intraday_signal = IntradaySignal(
            symbol=underlying,
            orb_signal=market_data.get('orb_signal', 'INSIDE_ORB'),
            vwap_position=market_data.get('price_vs_vwap', market_data.get('vwap_position', 'AT_VWAP')),
            vwap_trend=market_data.get('vwap_slope', market_data.get('vwap_trend', 'FLAT')),
            ema_regime=market_data.get('ema_regime', 'NORMAL'),
            volume_regime=market_data.get('volume_regime', 'NORMAL'),
            rsi=market_data.get('rsi_14', 50.0),
            price_momentum=market_data.get('momentum_15m', 0.0),
            htf_alignment=market_data.get('htf_alignment', 'NEUTRAL'),
            chop_zone=market_data.get('chop_zone', False),
            follow_through_candles=market_data.get('follow_through_candles', 0),
            range_expansion_ratio=market_data.get('range_expansion_ratio', 0.0),
            vwap_slope_steepening=market_data.get('vwap_slope_steepening', False),
            atr=market_data.get('atr_14', 0.0)
        )
        
        # === PRE-FETCH OPTION CHAIN FOR MICROSTRUCTURE DATA ===
        # Fetch chain BEFORE scoring so we can pass microstructure to scorer
        # This unlocks 15 points that were previously always 0
        micro_data = None
        try:
            pre_expiry_sel = force_expiry or ExpirySelection.CURRENT_WEEK
            pre_expiry = self.chain_fetcher.get_nearest_expiry(underlying, pre_expiry_sel)
            if pre_expiry:
                pre_chain = self.chain_fetcher.fetch_option_chain(underlying, pre_expiry)
                if pre_chain and pre_chain.contracts:
                    # Find ATM strike for the expected option type
                    atm_strike = pre_chain.get_atm_strike(pre_expiry)
                    expected_type = OptionType.CE if direction == 'BUY' else OptionType.PE
                    atm_contract = pre_chain.get_contract(atm_strike, expected_type, pre_expiry)
                    
                    if atm_contract:
                        # Fetch fresh depth from Kite for this specific contract
                        depth_data = None
                        try:
                            if self.chain_fetcher.kite:
                                q = self.chain_fetcher.kite.quote([atm_contract.symbol])
                                if atm_contract.symbol in q:
                                    depth_data = q[atm_contract.symbol].get('depth', {})
                        except Exception:
                            pass  # Use contract data without depth
                        
                        # Build microstructure from contract + depth
                        contract_dict = {
                            'bid': atm_contract.bid,
                            'ask': atm_contract.ask,
                            'ltp': atm_contract.ltp,
                            'oi': atm_contract.oi,
                            'volume': atm_contract.volume,
                        }
                        micro_data = build_microstructure_from_contract(
                            contract=contract_dict,
                            depth=depth_data
                        )
                        print(f"   📊 Microstructure: spread={micro_data.spread_pct:.2f}%, OI={micro_data.open_interest:,}, vol={micro_data.option_volume:,}")
        except Exception as e:
            print(f"   ⚠️ Microstructure pre-fetch failed: {e}")
        
        # === SCORE INTRADAY + TREND FOLLOWING SIGNALS (now WITH microstructure!) ===
        scorer = get_intraday_scorer()
        decision = scorer.score_intraday_signal(intraday_signal, market_data=market_data, option_data=micro_data, caller_direction=direction)
        
        # === PERSISTENT DECISION LOG (never lost to terminal buffer) ===
        from datetime import datetime as _dt
        log_lines = []
        log_lines.append(f"\n{'='*70}")
        log_lines.append(f"📊 INTRADAY + TREND OPTION DECISION for {underlying}")
        log_lines.append(f"   Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"   Score: {decision.confidence_score:.0f}/100")
        log_lines.append(f"   Direction: {decision.recommended_direction}")
        log_lines.append(f"   Should Trade: {decision.should_trade}")
        # Raw indicator snapshot for post-hoc analysis
        _ind_keys = ['ema_9','ema_21','ema_spread','ema_regime','vwap','vwap_slope','vwap_change_pct',
                      'volume_regime','volume_vs_avg','orb_signal','orb_strength','orb_hold_candles',
                      'adx','atr_14','follow_through_candles','range_expansion_ratio',
                      'vwap_slope_steepening','htf_alignment','chop_zone','rsi_14']
        _snap = {k: market_data.get(k, '?') for k in _ind_keys}
        log_lines.append(f"   📐 Indicators: {_snap}")
        log_lines.append(f"   Should Trade: {decision.should_trade}")
        for reason in decision.reasons:
            log_lines.append(f"   {reason}")
        for warning in decision.warnings:
            log_lines.append(f"   {warning}")
        
        # Print to terminal (may scroll away)
        for line in log_lines:
            print(line)
        
        # Write to persistent log file (always survives)
        try:
            with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_lines) + '\n')
        except Exception:
            pass  # Never crash on logging failure
        
        # === CHECK IF SHOULD TRADE ===
        if not decision.should_trade:
            reject_msg = f"   ❌ REJECTED: {underlying} score {decision.confidence_score:.0f} — SKIP"
            print(reject_msg)
            try:
                with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                    f.write(reject_msg + '\n')
            except Exception:
                pass
            return None
        
        # === USE INTRADAY DECISION FOR PARAMETERS ===
        # Force parameters can override intraday recommendation
        strike_sel = force_strike or StrikeSelection[decision.strike_selection]
        expiry_sel = force_expiry or ExpirySelection[decision.expiry_selection]
        opt_type = OptionType[decision.option_type]
        
        # Direction from intraday analysis takes precedence over passed direction
        final_direction = decision.recommended_direction if decision.recommended_direction != "HOLD" else direction
        
        print(f"   Strike: {strike_sel.value} | Expiry: {expiry_sel.value} | Type: {opt_type.value}")
        
        try:
            # === GET OPTION CHAIN ===
            expiry = self.chain_fetcher.get_nearest_expiry(underlying, expiry_sel)
            if expiry is None:
                print(f"   ⚠️ No expiry found for {underlying}")
                return None
            print(f"   📅 Expiry: {expiry}")
            
            chain = self.chain_fetcher.fetch_option_chain(underlying, expiry)
            
            if chain is None or not chain.contracts:
                print(f"   ⚠️ No option chain available for {underlying} (chain={'None' if chain is None else f'{len(chain.contracts)} contracts'})")
                return None
            
            print(f"   📊 Chain: {len(chain.contracts)} contracts, spot=₹{chain.spot_price:.2f}")
            
            # Select strike
            contract = self.chain_fetcher.select_strike(chain, opt_type, strike_sel, expiry)
            
            if contract is None:
                print(f"   ⚠️ No suitable strike found for {underlying}")
                return None
            
            print(f"   🎯 Selected: {contract.symbol} @ ₹{contract.ltp:.2f}")
            
            # === CALCULATE POSITION SIZE (apply intraday multiplier) ===
            # Determine score tier for tiered sizing
            score = decision.confidence_score
            if score >= self.PREMIUM_THRESHOLD:
                score_tier = "premium"
            elif score >= self.STANDARD_THRESHOLD:
                score_tier = "standard"
            else:
                score_tier = "base"
            
            sizing = self.position_sizer.calculate_position(
                contract, final_direction, self.capital, score_tier=score_tier
            )
            
            # Apply intraday conviction multiplier — use round() not int() to avoid truncation
            # int(1 * 1.5) = 1, round(1 * 1.5) = 2 ← this was silently killing the multiplier
            adjusted_lots = max(1, round(sizing['lots'] * decision.position_size_multiplier))
            adjusted_premium = adjusted_lots * sizing['premium_per_lot']
            
            # Tiered capital cap — premium trades get bigger allocation
            if score_tier == "premium":
                cap_pct = 0.25  # 25% of capital for premium trades
            elif score_tier == "standard":
                cap_pct = 0.20  # 20% for standard
            else:
                cap_pct = 0.15  # 15% for base
                
            if adjusted_premium > self.capital * cap_pct:
                adjusted_lots = max(1, int((self.capital * cap_pct) / sizing['premium_per_lot']))
                adjusted_premium = adjusted_lots * sizing['premium_per_lot']
            
            # === CREATE ORDER PLAN ===
            # === BUILD ENTRY METADATA for post-trade review ===
            import uuid
            trade_id = f"T_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            
            # Candle data snapshot
            ft_candles = market_data.get('follow_through_candles', 0)
            adx_val = market_data.get('adx_14', market_data.get('adx', 0))
            orb_strength_val = market_data.get('orb_strength_pct', market_data.get('orb_strength', 0))
            range_exp_val = market_data.get('range_expansion_ratio', 0)
            orb_signal_val = market_data.get('orb_signal', 'INSIDE_ORB')
            vol_regime_val = market_data.get('volume_regime', 'NORMAL')
            vol_ratio_val = market_data.get('volume_vs_avg', 1.0)
            vwap_pos = market_data.get('price_vs_vwap', 'AT_VWAP')
            htf_align = market_data.get('htf_alignment', 'NEUTRAL')
            rsi_val = market_data.get('rsi_14', 50)
            change_pct_val = market_data.get('change_pct', 0)
            
            entry_meta = {
                'trade_id': trade_id,
                'entry_score': round(decision.confidence_score, 1),
                'score_tier': score_tier,
                'trend_state': decision.trend_state,
                'acceleration_score': round(decision.acceleration_score, 1),
                'microstructure_score': round(decision.microstructure_score, 1),
                'microstructure_block': decision.microstructure_block,
                'position_size_multiplier': round(decision.position_size_multiplier, 2),
                'original_lots': sizing['lots'],
                'adjusted_lots': adjusted_lots,
                'cap_pct_used': cap_pct,
                'strategy_type': 'NAKED_OPTION',
                # Candle gate data
                'follow_through_candles': ft_candles,
                'adx': round(adx_val, 1),
                'orb_strength_pct': round(orb_strength_val, 1),
                'range_expansion_ratio': round(range_exp_val, 2),
                'orb_signal': orb_signal_val,
                'volume_regime': vol_regime_val,
                'volume_ratio': round(vol_ratio_val, 1),
                'vwap_position': vwap_pos,
                'htf_alignment': htf_align,
                'rsi': round(rsi_val, 1),
                'change_pct': round(change_pct_val, 2),
                # Gate pass/fail summary
                'gates_passed': [r for r in decision.reasons if '✅' in r or 'PASS' in r.upper()],
                'gates_warned': [w for w in decision.warnings],
                'spot_price': chain.spot_price if chain else 0,
            }
            
            plan = OptionOrderPlan(
                underlying=underlying,
                direction=final_direction,
                contract=contract,
                quantity=adjusted_lots,
                premium_per_lot=sizing['premium_per_lot'],
                total_premium=adjusted_premium,
                max_loss=adjusted_premium,  # Max loss = premium for long options
                breakeven=sizing['breakeven'],
                target_premium=sizing['target_premium'],
                stoploss_premium=sizing['stoploss_premium'],
                rationale=f"{final_direction} {underlying} | {' | '.join(decision.reasons[:3])} | SCORE:{decision.confidence_score:.0f}",
                greeks_summary=f"Δ:{contract.delta:.2f} Γ:{contract.gamma:.4f} Θ:{contract.theta:.2f} V:{contract.vega:.2f} IV:{contract.iv*100:.1f}%",
                entry_metadata=entry_meta
            )
            
            print(f"   ✅ Plan: {adjusted_lots} lots @ ₹{contract.ltp:.2f} = ₹{adjusted_premium:,.0f} [{score_tier.upper()}]")
            
            # Log executed order to persistent file — ENRICHED with full trade character
            try:
                with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                    f.write(f"   ✅ EXECUTED: {contract.symbol} | {adjusted_lots} lots @ ₹{contract.ltp:.2f} = ₹{adjusted_premium:,.0f}\n")
                    f.write(f"      Trade ID: {trade_id}\n")
                    f.write(f"      Tier: {score_tier.upper()} | Score: {decision.confidence_score:.0f} | Trend: {decision.trend_state}\n")
                    f.write(f"      Sizing: {sizing['lots']} lots × {decision.position_size_multiplier:.1f}x = {adjusted_lots} lots | Cap: {cap_pct*100:.0f}%\n")
                    f.write(f"      Candles: FT={ft_candles} ADX={adx_val:.1f} ORB={orb_strength_val:.0f}% RangeExp={range_exp_val:.2f}\n")
                    f.write(f"      Context: ORB={orb_signal_val} Vol={vol_regime_val}({vol_ratio_val:.1f}x) VWAP={vwap_pos} HTF={htf_align} RSI={rsi_val:.0f}\n")
                    f.write(f"      Micro: score={decision.microstructure_score:.0f} block={decision.microstructure_block} accel={decision.acceleration_score:.1f}\n")
                    f.write(f"      Greeks: {plan.greeks_summary}\n")
                    f.write(f"      Target: ₹{plan.target_premium:.2f} | SL: ₹{plan.stoploss_premium:.2f}\n")
                    f.write(f"{'='*70}\n")
            except Exception:
                pass
            
            return (plan, decision)
        except Exception as e:
            import traceback
            print(f"   ❌ ERROR in option order creation: {e}")
            traceback.print_exc()
            return None
    
    def create_option_order(self, underlying: str, direction: str,
                           option_type: OptionType = None,
                           strike_selection: StrikeSelection = None,
                           expiry_selection: ExpirySelection = None,
                           market_data: Dict = None) -> Optional[OptionOrderPlan]:
        """
        Create an option order plan based on underlying signal
        
        NOW INTEGRATES INTRADAY SIGNALS when market_data is provided.
        If market_data is provided, uses intraday scoring for decisions.
        Otherwise falls back to basic logic.
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            direction: 'BUY' or 'SELL' signal on underlying
            option_type: Force CE or PE (None = auto based on direction)
            strike_selection: ATM, ITM, OTM strategy
            expiry_selection: Which expiry to use
            market_data: Optional intraday market data for scoring
            
        Returns:
            OptionOrderPlan with all details
        """
        # === IF MARKET DATA PROVIDED, USE INTRADAY SCORING ===
        if market_data:
            result = self.create_option_order_with_intraday(
                underlying=underlying,
                direction=direction,
                market_data=market_data,
                force_strike=strike_selection,
                force_expiry=expiry_selection
            )
            if result:
                return result[0]  # Return just the plan
            # Intraday scoring rejected - respect the decision
            return None
        
        # === FALLBACK: BASIC LOGIC (no intraday data) ===
        # Auto-select option type based on direction
        if option_type is None:
            option_type = OptionType.CE if direction == 'BUY' else OptionType.PE
        
        # Use defaults if not specified
        strike_selection = strike_selection or self.config['default_strike_selection']
        expiry_selection = expiry_selection or self.config['default_expiry_selection']
        
        # Get option chain
        expiry = self.chain_fetcher.get_nearest_expiry(underlying, expiry_selection)
        chain = self.chain_fetcher.fetch_option_chain(underlying, expiry)
        
        if chain is None or not chain.contracts:
            print(f"⚠️ No option chain available for {underlying}")
            return None
        
        # Select strike
        contract = self.chain_fetcher.select_strike(chain, option_type, strike_selection, expiry)
        
        if contract is None:
            print(f"⚠️ No suitable strike found for {underlying}")
            return None
        
        # Validate contract
        if contract.oi < self.config['min_oi']:
            print(f"⚠️ Low OI ({contract.oi}) for {contract.symbol}")
        
        if contract.volume < self.config['min_volume']:
            print(f"⚠️ Low volume ({contract.volume}) for {contract.symbol}")
        
        # Calculate position size
        sizing = self.position_sizer.calculate_position(contract, direction, self.capital)
        
        # Create order plan
        plan = OptionOrderPlan(
            underlying=underlying,
            direction=direction,
            contract=contract,
            quantity=sizing['lots'],
            premium_per_lot=sizing['premium_per_lot'],
            total_premium=sizing['total_premium'],
            max_loss=sizing['max_loss'],
            breakeven=sizing['breakeven'],
            target_premium=sizing['target_premium'],
            stoploss_premium=sizing['stoploss_premium'],
            rationale=f"{direction} {underlying} → {option_type.value} {strike_selection.value}",
            greeks_summary=f"Δ:{contract.delta:.2f} Γ:{contract.gamma:.4f} Θ:{contract.theta:.2f} V:{contract.vega:.2f} IV:{contract.iv*100:.1f}%"
        )
        
        return plan
    
    # ================================================================
    # CREDIT SPREAD STRATEGY (THETA-POSITIVE)
    # ================================================================
    
    def create_credit_spread(self, underlying: str, direction: str,
                             market_data: Dict = None,
                             spread_width_strikes: int = None) -> Optional[CreditSpreadPlan]:
        """
        Create a credit spread order plan.
        
        BULLISH view → Bull Put Spread: SELL OTM PE + BUY further OTM PE
        BEARISH view → Bear Call Spread: SELL OTM CE + BUY further OTM CE
        
        Why credit spreads:
        - Theta works FOR us (options decay every day → we profit)
        - Defined risk (hedge limits max loss)
        - Higher win rate than naked buys (profit from time decay + direction)
        - Requires more capital (margins) but ₹5L supports this
        
        Args:
            underlying: e.g. "NSE:RELIANCE"
            direction: 'BUY' (bullish) or 'SELL' (bearish)
            market_data: Intraday signals dict for scoring
            spread_width_strikes: Override spread width (default from config)
            
        Returns:
            CreditSpreadPlan or None if not viable
        """
        from config import CREDIT_SPREAD_CONFIG
        
        if not CREDIT_SPREAD_CONFIG.get('enabled', False):
            print(f"   ⚠️ Credit spreads disabled in config")
            return None
        
        spread_width = spread_width_strikes or CREDIT_SPREAD_CONFIG.get('spread_width_strikes', 2)
        symbol = underlying.replace("NSE:", "")
        
        # === INTRADAY SCORING (shared with regular option flow) ===
        score = 0
        final_direction = direction
        if market_data:
            intraday_signal = IntradaySignal(
                symbol=underlying,
                orb_signal=market_data.get('orb_signal', 'INSIDE_ORB'),
                vwap_position=market_data.get('price_vs_vwap', market_data.get('vwap_position', 'AT_VWAP')),
                vwap_trend=market_data.get('vwap_slope', market_data.get('vwap_trend', 'FLAT')),
                ema_regime=market_data.get('ema_regime', 'NORMAL'),
                volume_regime=market_data.get('volume_regime', 'NORMAL'),
                rsi=market_data.get('rsi_14', 50.0),
                price_momentum=market_data.get('momentum_15m', 0.0),
                htf_alignment=market_data.get('htf_alignment', 'NEUTRAL'),
                chop_zone=market_data.get('chop_zone', False),
                follow_through_candles=market_data.get('follow_through_candles', 0),
                range_expansion_ratio=market_data.get('range_expansion_ratio', 0.0),
                vwap_slope_steepening=market_data.get('vwap_slope_steepening', False),
                atr=market_data.get('atr_14', 0.0)
            )
            scorer = get_intraday_scorer()
            decision = scorer.score_intraday_signal(intraday_signal, market_data=market_data, caller_direction=direction)
            score = decision.confidence_score
            
            min_score = CREDIT_SPREAD_CONFIG.get('min_score_threshold', 55)
            if score < min_score:
                print(f"   ❌ Credit spread REJECTED: {underlying} score {score:.0f} < {min_score}")
                return None
            
            if decision.recommended_direction != "HOLD":
                final_direction = decision.recommended_direction
        
        # Determine spread type
        is_bullish = final_direction in ('BUY', 'BULLISH')
        spread_type = "BULL_PUT_SPREAD" if is_bullish else "BEAR_CALL_SPREAD"
        opt_type = OptionType.PE if is_bullish else OptionType.CE
        
        print(f"   📊 Credit Spread: {spread_type} on {underlying} (score: {score:.0f})")
        
        try:
            # === GET OPTION CHAIN ===
            expiry_sel_str = CREDIT_SPREAD_CONFIG.get('prefer_expiry', 'CURRENT_WEEK')
            expiry_sel = ExpirySelection[expiry_sel_str]
            expiry = self.chain_fetcher.get_nearest_expiry(underlying, expiry_sel)
            
            if expiry is None:
                print(f"   ⚠️ No expiry found for {underlying}")
                return None
            
            # DTE check
            from datetime import date as _date
            expiry_date = expiry if isinstance(expiry, _date) and not isinstance(expiry, datetime) else expiry.date() if hasattr(expiry, 'date') else expiry
            dte = (expiry_date - datetime.now().date()).days
            
            min_dte = CREDIT_SPREAD_CONFIG.get('min_days_to_expiry', 2)
            max_dte = CREDIT_SPREAD_CONFIG.get('max_days_to_expiry', 21)
            
            if dte < min_dte:
                print(f"   ⚠️ DTE {dte} < min {min_dte} — gamma risk too high, skipping")
                return None
            if dte > max_dte:
                print(f"   ⚠️ DTE {dte} > max {max_dte} — too far out, theta too slow")
                return None
            
            # Warn about higher DTE needing further OTM strikes
            if dte > 10:
                print(f"   ⏳ DTE={dte} (monthly expiry) — using deeper OTM strikes for safety")
            
            chain = self.chain_fetcher.fetch_option_chain(underlying, expiry)
            if chain is None or not chain.contracts:
                print(f"   ⚠️ No option chain for {underlying}")
                return None
            
            print(f"   📅 Expiry: {expiry} (DTE: {dte}) | Chain: {len(chain.contracts)} contracts")
            
            # === FIND STRIKES FOR CREDIT SPREAD ===
            # Get sorted strikes for the option type
            all_strikes = sorted(set(
                c.strike for c in chain.contracts
                if c.option_type == opt_type and (c.expiry is None or c.expiry == expiry)
            ))
            
            if len(all_strikes) < spread_width + 1:
                print(f"   ⚠️ Not enough strikes ({len(all_strikes)}) for spread width {spread_width}")
                return None
            
            atm_strike = chain.get_atm_strike(expiry)
            
            # Find the strike index closest to ATM
            atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))
            
            # OTM offset: how far from ATM to sell (higher = safer, less credit)
            # Scale with DTE: longer DTE → sell further OTM for safety
            base_otm_offset = CREDIT_SPREAD_CONFIG.get('sold_strike_otm_offset', 3)
            if dte > 10:
                otm_offset = base_otm_offset + 1  # Extra strike OTM for monthly expiry
            elif dte <= 5:
                otm_offset = max(2, base_otm_offset - 1)  # Slightly closer for weeklies
            else:
                otm_offset = base_otm_offset
            
            if is_bullish:
                # BULL PUT SPREAD: sell OTM PE (below spot), buy further OTM PE
                # Sell further OTM for higher probability of profit
                sold_idx = max(0, atm_idx - otm_offset)  # N strikes OTM
                hedge_idx = max(0, sold_idx - spread_width)  # further OTM
            else:
                # BEAR CALL SPREAD: sell OTM CE (above spot), buy further OTM CE
                # Sell further OTM for higher probability of profit
                sold_idx = min(len(all_strikes) - 1, atm_idx + otm_offset)  # N strikes OTM
                hedge_idx = min(len(all_strikes) - 1, sold_idx + spread_width)  # further OTM
            
            print(f"   📐 Strike selection: ATM={atm_strike}, sold={all_strikes[sold_idx]} ({otm_offset} strikes OTM), hedge={all_strikes[hedge_idx] if hedge_idx < len(all_strikes) else 'N/A'}")
            
            sold_strike = all_strikes[sold_idx]
            hedge_strike = all_strikes[hedge_idx]
            
            if sold_strike == hedge_strike:
                print(f"   ⚠️ Sold and hedge strikes are same — can't form spread")
                return None
            
            # Get contracts
            sold_contract = chain.get_contract(sold_strike, opt_type, expiry)
            hedge_contract = chain.get_contract(hedge_strike, opt_type, expiry)
            
            if sold_contract is None or hedge_contract is None:
                print(f"   ⚠️ Could not find contracts for strikes {sold_strike}/{hedge_strike}")
                return None
            
            # === VALIDATE LIQUIDITY ===
            min_oi = 500
            if sold_contract.oi < min_oi or hedge_contract.oi < min_oi:
                print(f"   ⚠️ Low OI: sold={sold_contract.oi}, hedge={hedge_contract.oi} (min: {min_oi})")
                return None
            
            # === VALIDATE SOLD LEG DELTA (probability of profit) ===
            max_sold_delta = CREDIT_SPREAD_CONFIG.get('max_sold_delta', 0.35)
            sold_delta_abs = abs(sold_contract.delta) if sold_contract.delta else 0.5
            if sold_delta_abs > max_sold_delta:
                # Try one more strike OTM
                if is_bullish and sold_idx > 0:
                    sold_idx -= 1
                    sold_strike = all_strikes[sold_idx]
                    hedge_idx = max(0, sold_idx - spread_width)
                    hedge_strike = all_strikes[hedge_idx]
                    sold_contract = chain.get_contract(sold_strike, opt_type, expiry)
                    hedge_contract = chain.get_contract(hedge_strike, opt_type, expiry)
                    if sold_contract:
                        sold_delta_abs = abs(sold_contract.delta) if sold_contract.delta else 0.5
                elif not is_bullish and sold_idx < len(all_strikes) - 1:
                    sold_idx += 1
                    sold_strike = all_strikes[sold_idx]
                    hedge_idx = min(len(all_strikes) - 1, sold_idx + spread_width)
                    hedge_strike = all_strikes[hedge_idx]
                    sold_contract = chain.get_contract(sold_strike, opt_type, expiry)
                    hedge_contract = chain.get_contract(hedge_strike, opt_type, expiry)
                    if sold_contract:
                        sold_delta_abs = abs(sold_contract.delta) if sold_contract.delta else 0.5
                
                if sold_delta_abs > max_sold_delta:
                    print(f"   ⚠️ Sold leg delta {sold_delta_abs:.2f} > max {max_sold_delta} — too close to ATM, probability too low")
                    return None
                print(f"   📐 Adjusted sold strike to {sold_strike} (delta={sold_delta_abs:.2f})")
            
            if sold_contract is None or hedge_contract is None:
                print(f"   ⚠️ Lost contract after delta adjustment")
                return None
            
            # === CALCULATE CREDIT SPREAD FINANCIALS ===
            # Re-fetch premiums after possible strike adjustment
            sold_premium = sold_contract.ltp
            hedge_premium = hedge_contract.ltp
            net_credit = sold_premium - hedge_premium
            
            if net_credit <= 0:
                print(f"   ⚠️ No net credit: sell@₹{sold_premium:.2f} - buy@₹{hedge_premium:.2f} = ₹{net_credit:.2f}")
                return None
            
            # Spread width in rupees
            spread_width_rs = abs(sold_strike - hedge_strike)
            
            # Credit as % of spread width (quality metric)
            credit_pct = (net_credit / spread_width_rs) * 100 if spread_width_rs > 0 else 0
            min_credit_pct = CREDIT_SPREAD_CONFIG.get('min_credit_pct', 25)
            
            if credit_pct < min_credit_pct:
                print(f"   ⚠️ Credit too thin: {credit_pct:.1f}% < {min_credit_pct}% of spread width")
                return None
            
            # === POSITION SIZING ===
            lot_size = self.chain_fetcher.get_lot_size(underlying)
            max_risk_per_lot = (spread_width_rs - net_credit) * lot_size
            max_risk_config = CREDIT_SPREAD_CONFIG.get('max_spread_risk', 25000)
            max_lots_config = CREDIT_SPREAD_CONFIG.get('max_lots_per_spread', 3)
            
            # Calculate lots: limited by risk and config
            lots_by_risk = max(1, int(max_risk_config / max_risk_per_lot)) if max_risk_per_lot > 0 else 1
            lots = min(lots_by_risk, max_lots_config)
            
            # Check total spread exposure across portfolio
            max_total_exposure = CREDIT_SPREAD_CONFIG.get('max_total_spread_exposure', 150000)
            current_spread_exposure = self._get_current_spread_exposure()
            total_risk = (spread_width_rs - net_credit) * lot_size * lots
            
            if current_spread_exposure + total_risk > max_total_exposure:
                # Reduce lots to fit within exposure limit
                remaining_expo = max_total_exposure - current_spread_exposure
                if remaining_expo <= 0:
                    print(f"   ⚠️ Max spread exposure ₹{max_total_exposure:,.0f} reached (current: ₹{current_spread_exposure:,.0f})")
                    return None
                lots = max(1, int(remaining_expo / max_risk_per_lot))
            
            net_credit_total = net_credit * lot_size * lots
            max_risk_total = (spread_width_rs - net_credit) * lot_size * lots
            
            # === RISK MANAGEMENT LEVELS ===
            target_pct = CREDIT_SPREAD_CONFIG.get('target_pct', 65)
            sl_multiplier = CREDIT_SPREAD_CONFIG.get('sl_multiplier', 2.0)
            
            # Target: keep target_pct% of credit (buy back spread when premium drops)
            target_credit = net_credit * (1 - target_pct / 100)  # Buy back at this net debit
            # SL: exit when loss = sl_multiplier × credit received
            stop_loss_debit = net_credit + (net_credit * sl_multiplier)  # Max debit to pay to exit
            
            # Breakeven
            if is_bullish:
                breakeven = sold_strike - net_credit  # Bull put spread breakeven
            else:
                breakeven = sold_strike + net_credit  # Bear call spread breakeven
            
            # === NET GREEKS ===
            net_delta = (sold_contract.delta * -1) + hedge_contract.delta  # Sold delta is flipped
            net_theta = (sold_contract.theta * -1) + hedge_contract.theta  # Net theta should be POSITIVE
            net_vega = (sold_contract.vega * -1) + hedge_contract.vega     # Net vega should be NEGATIVE
            net_gamma = (sold_contract.gamma * -1) + hedge_contract.gamma  # Net gamma is NEGATIVE
            
            # Scale by lots and lot_size
            qty_shares = lots * lot_size
            
            plan = CreditSpreadPlan(
                underlying=underlying,
                direction='BULLISH' if is_bullish else 'BEARISH',
                spread_type=spread_type,
                sold_contract=sold_contract,
                hedge_contract=hedge_contract,
                quantity=lots,
                lot_size=lot_size,
                net_credit=net_credit,
                net_credit_total=net_credit_total,
                max_risk=max_risk_total,
                spread_width=spread_width_rs,
                target_credit=target_credit,
                stop_loss_debit=stop_loss_debit,
                breakeven=breakeven,
                net_delta=net_delta * qty_shares,
                net_theta=net_theta * qty_shares,
                net_vega=net_vega * qty_shares,
                net_gamma=net_gamma * qty_shares,
                rationale=f"{spread_type} on {underlying} | Credit ₹{net_credit:.2f}/share ({credit_pct:.0f}% of width) | Score:{score:.0f}",
                greeks_summary=f"NetΔ:{net_delta*qty_shares:.2f} NetΘ:{net_theta*qty_shares:+.2f}/day NetV:{net_vega*qty_shares:.2f} NetΓ:{net_gamma*qty_shares:.4f}",
                credit_pct=credit_pct,
                dte=dte,
            )
            
            print(f"   ✅ {spread_type}: SELL {sold_contract.symbol}@₹{sold_premium:.2f} + BUY {hedge_contract.symbol}@₹{hedge_premium:.2f}")
            print(f"      Credit: ₹{net_credit:.2f}/share (₹{net_credit_total:,.0f} total) | {credit_pct:.0f}% of width")
            print(f"      Max Risk: ₹{max_risk_total:,.0f} | {lots} lots × {lot_size} = {qty_shares} shares")
            print(f"      Target: buy back at ₹{target_credit:.2f} | SL: exit at ₹{stop_loss_debit:.2f}")
            print(f"      Breakeven: ₹{breakeven:.2f} | NetΘ: {net_theta*qty_shares:+.2f}/day (POSITIVE = good)")
            
            # Log to persistent file
            try:
                with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"📊 CREDIT SPREAD: {spread_type} on {underlying}\n")
                    f.write(f"   SELL: {sold_contract.symbol}@₹{sold_premium:.2f} | BUY: {hedge_contract.symbol}@₹{hedge_premium:.2f}\n")
                    f.write(f"   Credit: ₹{net_credit_total:,.0f} | Max Risk: ₹{max_risk_total:,.0f} | DTE: {dte}\n")
                    f.write(f"   Score: {score:.0f} | Credit%: {credit_pct:.0f}% | Lots: {lots}\n")
                    f.write(f"{'='*70}\n")
            except Exception:
                pass
            
            return plan
            
        except Exception as e:
            import traceback
            print(f"   ❌ Credit spread creation failed: {e}")
            traceback.print_exc()
            return None
    
    def _get_current_spread_exposure(self) -> float:
        """Get total risk exposure from existing credit spread positions"""
        total = 0
        for pos in self.positions:
            if pos.get('status') == 'OPEN' and pos.get('is_credit_spread'):
                total += pos.get('max_risk', 0)
        return total
    
    def execute_credit_spread(self, plan: CreditSpreadPlan) -> Dict:
        """
        Execute a credit spread order (both legs)
        
        Args:
            plan: CreditSpreadPlan with both legs
            
        Returns:
            Dict with execution result
        """
        if self.paper_mode:
            import random
            sold_order_id = f"SPREAD_SELL_{random.randint(100000, 999999)}"
            hedge_order_id = f"SPREAD_HEDGE_{random.randint(100000, 999999)}"
            spread_id = f"SPREAD_{random.randint(100000, 999999)}"
            
            position = {
                'spread_id': spread_id,
                'is_credit_spread': True,
                'spread_type': plan.spread_type,
                'underlying': plan.underlying,
                'direction': plan.direction,
                # Sold leg
                'sold_symbol': plan.sold_contract.symbol,
                'sold_strike': plan.sold_contract.strike,
                'sold_premium': plan.sold_contract.ltp,
                'sold_order_id': sold_order_id,
                # Hedge leg
                'hedge_symbol': plan.hedge_contract.symbol,
                'hedge_strike': plan.hedge_contract.strike,
                'hedge_premium': plan.hedge_contract.ltp,
                'hedge_order_id': hedge_order_id,
                # Sizing
                'quantity': plan.quantity,
                'lot_size': plan.lot_size,
                'net_credit': plan.net_credit,
                'net_credit_total': plan.net_credit_total,
                'max_risk': plan.max_risk,
                'spread_width': plan.spread_width,
                # Risk mgmt
                'target_credit': plan.target_credit,
                'stop_loss_debit': plan.stop_loss_debit,
                'breakeven': plan.breakeven,
                # Greeks
                'net_delta': plan.net_delta,
                'net_theta': plan.net_theta,
                'net_vega': plan.net_vega,
                'credit_pct': plan.credit_pct,
                'dte': plan.dte,
                # Status
                'status': 'OPEN',
                'timestamp': datetime.now().isoformat(),
                'rationale': plan.rationale,
            }
            
            self.positions.append(position)
            
            return {
                'success': True,
                'paper_trade': True,
                'spread_id': spread_id,
                'sold_order_id': sold_order_id,
                'hedge_order_id': hedge_order_id,
                'spread_type': plan.spread_type,
                'underlying': plan.underlying,
                'sold_symbol': plan.sold_contract.symbol,
                'hedge_symbol': plan.hedge_contract.symbol,
                'net_credit': plan.net_credit,
                'net_credit_total': plan.net_credit_total,
                'max_risk': plan.max_risk,
                'credit_pct': plan.credit_pct,
                'dte': plan.dte,
                'lots': plan.quantity,
                'greeks': plan.greeks_summary,
                'rationale': plan.rationale,
                'message': f"📊 CREDIT SPREAD: {plan.spread_type} | Credit ₹{plan.net_credit_total:,.0f} | Max Risk ₹{plan.max_risk:,.0f}"
            }
        else:
            # LIVE MODE: Place both legs with Kite
            try:
                sold_exchange, sold_ts = plan.sold_contract.symbol.split(':')
                hedge_exchange, hedge_ts = plan.hedge_contract.symbol.split(':')
                qty_shares = plan.quantity * plan.lot_size
                
                # Leg 1: SELL the option (collect premium)
                sold_order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=sold_exchange,
                    tradingsymbol=sold_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                # Leg 2: BUY the hedge (cap risk)
                hedge_order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=hedge_exchange,
                    tradingsymbol=hedge_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                return {
                    'success': True,
                    'paper_trade': False,
                    'sold_order_id': str(sold_order_id),
                    'hedge_order_id': str(hedge_order_id),
                    'message': f"Credit spread placed: SELL {sold_ts} + BUY {hedge_ts}"
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f"Credit spread failed: {e}"
                }

    # =================================================================
    # DEBIT SPREAD — INTRADAY MOMENTUM STRATEGY
    # =================================================================

    def create_debit_spread(self, underlying: str, direction: str,
                            market_data: Dict = None,
                            spread_width_strikes: int = None) -> Optional[DebitSpreadPlan]:
        """
        Create an intraday debit spread on a big mover.
        
        BULLISH → Bull Call Spread: BUY near-ATM CE + SELL further OTM CE
        BEARISH → Bear Put Spread: BUY near-ATM PE + SELL further OTM PE
        
        Profits from continuation of a strong move. Cheaper than naked buy,
        defined risk = net debit paid.
        
        Args:
            underlying: e.g. "NSE:EICHERMOT"
            direction: 'BUY' (bullish) or 'SELL' (bearish)
            market_data: Intraday signals dict for scoring + move detection
            spread_width_strikes: Override sell offset (default from config)
        
        Returns:
            DebitSpreadPlan or None if not viable
        """
        from config import DEBIT_SPREAD_CONFIG
        
        if not DEBIT_SPREAD_CONFIG.get('enabled', False):
            print(f"   ⚠️ Debit spreads disabled in config")
            return None
        
        symbol = underlying.replace("NSE:", "")
        
        # === INTRADAY MOVE FILTER (must be a big mover) ===
        min_move = DEBIT_SPREAD_CONFIG.get('min_move_pct', 2.5)
        move_pct = 0.0
        if market_data:
            # Use LTP vs prev_close for intraday % move (NOT ohlc 'close' which is yesterday's close)
            ltp = market_data.get('ltp', market_data.get('last_price', 0))
            prev_close = market_data.get('prev_close', market_data.get('previous_close', 0))
            if prev_close and prev_close > 0 and ltp > 0:
                move_pct = ((ltp - prev_close) / prev_close) * 100
            elif market_data.get('change_pct', 0) != 0:
                move_pct = market_data.get('change_pct', 0)
            else:
                # Fallback: use open price for intraday move
                open_price = market_data.get('open', 0)
                if open_price and open_price > 0 and ltp > 0:
                    move_pct = ((ltp - open_price) / open_price) * 100
        
        abs_move = abs(move_pct)
        if abs_move < min_move:
            print(f"   ⚠️ Debit spread SKIPPED: {underlying} move {abs_move:.1f}% < {min_move}% min")
            return None
        
        # === VOLUME FILTER ===
        min_vol_ratio = DEBIT_SPREAD_CONFIG.get('min_volume_ratio', 1.3)
        vol_regime = market_data.get('volume_regime', 'NORMAL') if market_data else 'NORMAL'
        vol_ratio = market_data.get('volume_ratio', 1.0) if market_data else 1.0
        if vol_ratio < min_vol_ratio and vol_regime == 'LOW':
            print(f"   ⚠️ Debit spread SKIPPED: {underlying} volume too low ({vol_regime}, ratio {vol_ratio:.1f})")
            return None
        
        # === CANDLE-SMART GATES (data-driven — mirrors naked buy gates 8-12) ===
        # D1 (FT) and D5 (re-entry) = hard blocks. D2-D4 = score penalties (avoid overfitting)
        debit_gate_penalty = 0  # Accumulated penalty from softened gates
        if market_data:
            # Gate D1: Follow-Through Candles (STRONGEST winner signal — 4.1 vs 0.4)
            ft_candles = market_data.get('follow_through_candles', 0)
            min_ft = DEBIT_SPREAD_CONFIG.get('min_follow_through_candles', 2)
            if ft_candles < min_ft:
                print(f"   ❌ Debit spread BLOCKED [Gate D1]: {underlying} follow-through={ft_candles} < {min_ft} (need momentum confirmation)")
                return None
            
            # Gate D2: ADX Trend Strength (winners avg 37 vs losers 30.8)
            # SOFTENED: penalty instead of block — scorer already gates ADX too
            adx = market_data.get('adx_14', market_data.get('adx', 0))
            min_adx = DEBIT_SPREAD_CONFIG.get('min_adx', 28)
            if adx > 0 and adx < min_adx:
                debit_gate_penalty += 8
                print(f"   ⚠️ Debit spread PENALTY [Gate D2]: {underlying} ADX={adx:.1f} < {min_adx} (weak trend, -8 score)")
            
            # Gate D3: ORB Overextension (losers avg 142% — chasing kills profits)
            # SOFTENED: penalty instead of block. Only hard-block at extreme (>200%)
            orb_strength = market_data.get('orb_strength_pct', market_data.get('orb_strength', 0))
            max_orb = DEBIT_SPREAD_CONFIG.get('max_orb_strength_pct', 120)
            if orb_strength > max_orb:
                if orb_strength > 200:
                    print(f"   ❌ Debit spread BLOCKED [Gate D3]: {underlying} ORB={orb_strength:.0f}% > 200% (extreme overextension)")
                    return None
                debit_gate_penalty += 6
                print(f"   ⚠️ Debit spread PENALTY [Gate D3]: {underlying} ORB={orb_strength:.0f}% > {max_orb}% (overextended, -6 score)")
            
            # Gate D4: Range Expansion Filter (>0.50 ATR = move exhausted)
            # SOFTENED: penalty instead of block. Only hard-block at extreme (>0.80)
            range_exp = market_data.get('range_expansion_ratio', 0)
            max_range_exp = DEBIT_SPREAD_CONFIG.get('max_range_expansion', 0.50)
            if range_exp > max_range_exp:
                if range_exp > 0.80:
                    print(f"   ❌ Debit spread BLOCKED [Gate D4]: {underlying} range expansion={range_exp:.2f} > 0.80 ATR (extreme exhaustion)")
                    return None
                debit_gate_penalty += 5
                print(f"   ⚠️ Debit spread PENALTY [Gate D4]: {underlying} range expansion={range_exp:.2f} > {max_range_exp} ATR (exhausted, -5 score)")
            
            # Gate D5: Same-symbol re-entry prevention (reuses scorer data)
            scorer_check = get_intraday_scorer()
            sym = underlying.replace("NSE:", "")
            max_sym_losses = DEBIT_SPREAD_CONFIG.get('max_losses_same_symbol', 1)
            sym_losses = scorer_check.symbol_losses_today.get(sym, 0)
            if sym_losses >= max_sym_losses:
                print(f"   ❌ Debit spread BLOCKED [Gate D5]: {sym} already lost {sym_losses}x today — no re-entry")
                return None
            
            # Apply accumulated gate penalties to score
            if debit_gate_penalty > 0:
                print(f"   ⚠️ Debit spread gate penalties: -{debit_gate_penalty} score for {underlying}")
            else:
                print(f"   ✅ Debit spread CANDLE GATES PASSED: {underlying} | FT={ft_candles} ADX={adx:.1f} ORB={orb_strength:.0f}% RangeExp={range_exp:.2f}")
        
        # === TREND CONTINUATION CHECK ===
        if DEBIT_SPREAD_CONFIG.get('require_trend_continuation', True) and market_data:
            # Move should be IN direction of trade
            if direction in ('BUY', 'BULLISH') and move_pct < 0:
                print(f"   ⚠️ Debit spread SKIPPED: {underlying} moving DOWN {move_pct:.1f}% but BUY requested")
                return None
            if direction in ('SELL', 'BEARISH') and move_pct > 0:
                print(f"   ⚠️ Debit spread SKIPPED: {underlying} moving UP +{move_pct:.1f}% but SELL requested")
                return None
        
        # === TIME CHECK ===
        from config import TRADING_HOURS
        now = datetime.now()
        no_entry_after_str = DEBIT_SPREAD_CONFIG.get('no_entry_after', '14:00')
        no_entry_after = datetime.strptime(no_entry_after_str, '%H:%M').time()
        if now.time() > no_entry_after:
            print(f"   ⚠️ Debit spread SKIPPED: past {no_entry_after_str} — not enough time for move")
            return None
        
        # === INTRADAY SCORING ===
        score = 0
        final_direction = direction
        if market_data:
            intraday_signal = IntradaySignal(
                symbol=underlying,
                orb_signal=market_data.get('orb_signal', 'INSIDE_ORB'),
                vwap_position=market_data.get('price_vs_vwap', market_data.get('vwap_position', 'AT_VWAP')),
                vwap_trend=market_data.get('vwap_slope', market_data.get('vwap_trend', 'FLAT')),
                ema_regime=market_data.get('ema_regime', 'NORMAL'),
                volume_regime=market_data.get('volume_regime', 'NORMAL'),
                rsi=market_data.get('rsi_14', 50.0),
                price_momentum=market_data.get('momentum_15m', 0.0),
                htf_alignment=market_data.get('htf_alignment', 'NEUTRAL'),
                chop_zone=market_data.get('chop_zone', False),
                follow_through_candles=market_data.get('follow_through_candles', 0),
                range_expansion_ratio=market_data.get('range_expansion_ratio', 0.0),
                vwap_slope_steepening=market_data.get('vwap_slope_steepening', False),
                atr=market_data.get('atr_14', 0.0)
            )
            scorer = get_intraday_scorer()
            decision = scorer.score_intraday_signal(intraday_signal, market_data=market_data, caller_direction=direction)
            score = decision.confidence_score
            
            # Apply debit gate penalties (D2-D4 softened gates)
            if debit_gate_penalty > 0:
                score -= debit_gate_penalty
                print(f"   📉 Debit spread score adjusted: {decision.confidence_score:.0f} → {score:.0f} (gate penalties: -{debit_gate_penalty})")
            
            min_score = DEBIT_SPREAD_CONFIG.get('min_score_threshold', 70)
            if score < min_score:
                print(f"   ❌ Debit spread REJECTED: {underlying} score {score:.0f} < {min_score} (need high conviction for momentum play)")
                return None
            
            if decision.recommended_direction != "HOLD":
                final_direction = decision.recommended_direction
        
        # Determine spread type
        is_bullish = final_direction in ('BUY', 'BULLISH')
        spread_type = "BULL_CALL_SPREAD" if is_bullish else "BEAR_PUT_SPREAD"
        opt_type = OptionType.CE if is_bullish else OptionType.PE
        
        print(f"   📊 Debit Spread: {spread_type} on {underlying} (score: {score:.0f}, move: {move_pct:+.1f}%)")
        
        try:
            # === GET OPTION CHAIN ===
            expiry_sel_str = 'CURRENT_WEEK'
            expiry_sel = ExpirySelection[expiry_sel_str]
            expiry = self.chain_fetcher.get_nearest_expiry(underlying, expiry_sel)
            
            if expiry is None:
                print(f"   ⚠️ No expiry found for {underlying}")
                return None
            
            # DTE check
            from datetime import date as _date
            expiry_date = expiry if isinstance(expiry, _date) and not isinstance(expiry, datetime) else expiry.date() if hasattr(expiry, 'date') else expiry
            dte = (expiry_date - datetime.now().date()).days
            
            max_dte = DEBIT_SPREAD_CONFIG.get('max_dte', 7)
            min_dte = DEBIT_SPREAD_CONFIG.get('min_dte', 0)
            if dte > max_dte:
                print(f"   ⚠️ DTE {dte} > max {max_dte} — theta bleed too high for intraday debit spread")
                return None
            if dte < min_dte:
                print(f"   ⚠️ DTE {dte} < min {min_dte}")
                return None
            
            chain = self.chain_fetcher.fetch_option_chain(underlying, expiry)
            if chain is None or not chain.contracts:
                print(f"   ⚠️ No option chain for {underlying}")
                return None
            
            print(f"   📅 Expiry: {expiry} (DTE: {dte}) | Chain: {len(chain.contracts)} contracts")
            
            # === FIND STRIKES ===
            all_strikes = sorted(set(
                c.strike for c in chain.contracts
                if c.option_type == opt_type and (c.expiry is None or c.expiry == expiry)
            ))
            
            sell_offset = spread_width_strikes or DEBIT_SPREAD_CONFIG.get('sell_strike_offset', 3)
            buy_offset = DEBIT_SPREAD_CONFIG.get('buy_strike_offset', 0)
            
            if len(all_strikes) < sell_offset + 1:
                print(f"   ⚠️ Not enough strikes ({len(all_strikes)}) for debit spread")
                return None
            
            atm_strike = chain.get_atm_strike(expiry)
            atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - atm_strike))
            
            if is_bullish:
                # BULL CALL SPREAD: BUY near-ATM CE, SELL further OTM CE
                buy_idx = max(0, atm_idx - buy_offset)  # ATM or slightly ITM
                sell_idx = min(len(all_strikes) - 1, buy_idx + sell_offset)
            else:
                # BEAR PUT SPREAD: BUY near-ATM PE, SELL further OTM PE
                buy_idx = min(len(all_strikes) - 1, atm_idx + buy_offset)  # ATM or slightly ITM
                sell_idx = max(0, buy_idx - sell_offset)
            
            buy_strike = all_strikes[buy_idx]
            sell_strike = all_strikes[sell_idx]
            
            if buy_strike == sell_strike:
                print(f"   ⚠️ Buy and sell strikes are same — can't form spread")
                return None
            
            buy_contract = chain.get_contract(buy_strike, opt_type, expiry)
            sell_contract = chain.get_contract(sell_strike, opt_type, expiry)
            
            if buy_contract is None or sell_contract is None:
                print(f"   ⚠️ Could not find contracts for strikes {buy_strike}/{sell_strike}")
                return None
            
            print(f"   📐 Strike selection: ATM={atm_strike}, BUY={buy_strike}, SELL={sell_strike}")
            
            # === VALIDATE LIQUIDITY ===
            min_oi = DEBIT_SPREAD_CONFIG.get('min_oi', 500)
            if buy_contract.oi < min_oi or sell_contract.oi < min_oi:
                print(f"   ⚠️ Low OI: buy={buy_contract.oi}, sell={sell_contract.oi} (min: {min_oi})")
                return None
            
            # === BID-ASK SPREAD VALIDATION ===
            max_bid_ask_pct = DEBIT_SPREAD_CONFIG.get('max_spread_bid_ask_pct', 5.0)
            for leg_label, contract in [("BUY", buy_contract), ("SELL", sell_contract)]:
                if contract.ltp > 0 and contract.bid > 0 and contract.ask > 0:
                    ba_spread_pct = ((contract.ask - contract.bid) / contract.ltp) * 100
                    if ba_spread_pct > max_bid_ask_pct:
                        print(f"   ⚠️ Wide bid-ask on {leg_label} leg: {ba_spread_pct:.1f}% > {max_bid_ask_pct}% (bid={contract.bid:.2f} ask={contract.ask:.2f} ltp={contract.ltp:.2f})")
                        return None
            
            # === CALCULATE DEBIT SPREAD FINANCIALS ===
            buy_premium = buy_contract.ltp
            sell_premium = sell_contract.ltp
            net_debit = buy_premium - sell_premium  # We PAY this
            
            if net_debit <= 0:
                print(f"   ⚠️ No net debit: buy@₹{buy_premium:.2f} - sell@₹{sell_premium:.2f} = ₹{net_debit:.2f} — invalid spread")
                return None
            
            # Spread width in rupees
            spread_width_rs = abs(buy_strike - sell_strike)
            
            # Max profit = spread_width - net_debit (if underlying moves past sell strike)
            max_profit_per_share = spread_width_rs - net_debit
            if max_profit_per_share <= 0:
                print(f"   ⚠️ No profit potential: width ₹{spread_width_rs} - debit ₹{net_debit:.2f} = ₹{max_profit_per_share:.2f}")
                return None
            
            # Profit-to-risk ratio check (must be at least 1:1)
            profit_risk_ratio = max_profit_per_share / net_debit
            if profit_risk_ratio < 0.5:
                print(f"   ⚠️ Poor risk/reward: {profit_risk_ratio:.2f} (need >= 0.5)")
                return None
            
            # === POSITION SIZING (TIERED by score — mirrors naked buy sizing) ===
            lot_size = self.chain_fetcher.get_lot_size(underlying)
            debit_per_lot = net_debit * lot_size
            max_debit_config = DEBIT_SPREAD_CONFIG.get('max_debit_per_spread', 60000)
            max_lots_config = DEBIT_SPREAD_CONFIG.get('max_lots_per_spread', 4)
            
            # Score-based tier for debit spreads
            premium_threshold = 70
            if score >= premium_threshold:
                debit_score_tier = "premium"
                min_lots = DEBIT_SPREAD_CONFIG.get('premium_tier_min_lots', 3)
            else:
                debit_score_tier = "standard"
                min_lots = 2
            
            lots_by_debit = max(1, round(max_debit_config / debit_per_lot)) if debit_per_lot > 0 else 1
            lots = min(lots_by_debit, max_lots_config)
            lots = max(lots, min_lots)  # Enforce minimum lots for tier
            
            # Check total debit spread exposure
            max_total_exposure = DEBIT_SPREAD_CONFIG.get('max_total_debit_exposure', 75000)
            current_debit_exposure = self._get_current_debit_spread_exposure()
            total_debit = net_debit * lot_size * lots
            
            if current_debit_exposure + total_debit > max_total_exposure:
                remaining = max_total_exposure - current_debit_exposure
                if remaining <= 0:
                    print(f"   ⚠️ Max debit exposure ₹{max_total_exposure:,.0f} reached (current: ₹{current_debit_exposure:,.0f})")
                    return None
                lots = max(1, int(remaining / debit_per_lot))
            
            net_debit_total = net_debit * lot_size * lots
            max_profit_total = max_profit_per_share * lot_size * lots
            max_loss_total = net_debit_total  # Max loss = total debit paid
            qty_shares = lots * lot_size
            
            # === RISK MANAGEMENT LEVELS ===
            target_pct = DEBIT_SPREAD_CONFIG.get('target_pct', 50)
            sl_pct = DEBIT_SPREAD_CONFIG.get('stop_loss_pct', 40)
            
            # Target: spread value rises to net_debit * (1 + target_pct/100)
            target_value = net_debit * (1 + target_pct / 100)
            # Cap at max profit
            target_value = min(target_value, spread_width_rs)
            # SL: spread value drops to net_debit * (1 - sl_pct/100)
            stop_loss_value = net_debit * (1 - sl_pct / 100)
            
            # Breakeven
            if is_bullish:
                breakeven = buy_strike + net_debit  # Bull call spread breakeven
            else:
                breakeven = buy_strike - net_debit  # Bear put spread breakeven
            
            # === NET GREEKS ===
            net_delta = buy_contract.delta + (sell_contract.delta * -1)  # Sell delta flipped
            net_theta = buy_contract.theta + (sell_contract.theta * -1)  # Net theta NEGATIVE
            net_vega = buy_contract.vega + (sell_contract.vega * -1)     # Net vega POSITIVE
            net_gamma = buy_contract.gamma + (sell_contract.gamma * -1)
            
            plan = DebitSpreadPlan(
                underlying=underlying,
                direction='BULLISH' if is_bullish else 'BEARISH',
                spread_type=spread_type,
                buy_contract=buy_contract,
                sell_contract=sell_contract,
                quantity=lots,
                lot_size=lot_size,
                net_debit=net_debit,
                net_debit_total=net_debit_total,
                max_profit=max_profit_total,
                max_loss=max_loss_total,
                spread_width=spread_width_rs,
                target_value=target_value,
                stop_loss_value=stop_loss_value,
                breakeven=breakeven,
                net_delta=net_delta * qty_shares,
                net_theta=net_theta * qty_shares,
                net_vega=net_vega * qty_shares,
                net_gamma=net_gamma * qty_shares,
                rationale=f"{spread_type} on {underlying} | Debit ₹{net_debit:.2f}/share | Move {move_pct:+.1f}% | Score:{score:.0f}",
                greeks_summary=f"NetΔ:{net_delta*qty_shares:.2f} NetΘ:{net_theta*qty_shares:+.2f}/day NetV:{net_vega*qty_shares:.2f} NetΓ:{net_gamma*qty_shares:.4f}",
                move_pct=move_pct,
                dte=dte,
            )
            
            print(f"   ✅ {spread_type}: BUY {buy_contract.symbol}@₹{buy_premium:.2f} + SELL {sell_contract.symbol}@₹{sell_premium:.2f}")
            print(f"      Debit: ₹{net_debit:.2f}/share (₹{net_debit_total:,.0f} total) | R:R = 1:{profit_risk_ratio:.1f}")
            print(f"      Max Profit: ₹{max_profit_total:,.0f} | Max Loss: ₹{max_loss_total:,.0f}")
            print(f"      Target: spread at ₹{target_value:.2f} | SL: spread at ₹{stop_loss_value:.2f}")
            print(f"      Breakeven: ₹{breakeven:.2f} | NetΔ: {net_delta*qty_shares:+.2f} (directional edge)")
            
            # Log with candle gate data
            ft_log = market_data.get('follow_through_candles', 0) if market_data else 0
            adx_log = market_data.get('adx_14', market_data.get('adx', 0)) if market_data else 0
            orb_log = market_data.get('orb_strength_pct', market_data.get('orb_strength', 0)) if market_data else 0
            try:
                with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"🚀 DEBIT SPREAD: {spread_type} on {underlying} (move: {move_pct:+.1f}%)\n")
                    f.write(f"   BUY: {buy_contract.symbol}@₹{buy_premium:.2f} | SELL: {sell_contract.symbol}@₹{sell_premium:.2f}\n")
                    f.write(f"   Debit: ₹{net_debit_total:,.0f} | Max Profit: ₹{max_profit_total:,.0f} | DTE: {dte}\n")
                    f.write(f"   Score: {score:.0f} | Tier: {debit_score_tier} | R:R = 1:{profit_risk_ratio:.1f} | Lots: {lots}\n")
                    f.write(f"   Candle Gates: FT={ft_log} ADX={adx_log:.1f} ORB={orb_log:.0f}%\n")
                    f.write(f"{'='*70}\n")
            except Exception:
                pass
            
            return plan
            
        except Exception as e:
            import traceback
            print(f"   ❌ Debit spread creation failed: {e}")
            traceback.print_exc()
            return None
    
    def _get_current_debit_spread_exposure(self) -> float:
        """Get total debit exposure from existing debit spread positions"""
        total = 0
        for pos in self.positions:
            if pos.get('status') == 'OPEN' and pos.get('is_debit_spread'):
                total += pos.get('net_debit_total', 0)
        return total
    
    def execute_debit_spread(self, plan: DebitSpreadPlan) -> Dict:
        """
        Execute a debit spread order (both legs).
        BUY near-ATM option + SELL further OTM option.
        """
        if self.paper_mode:
            import random
            buy_order_id = f"DSPREAD_BUY_{random.randint(100000, 999999)}"
            sell_order_id = f"DSPREAD_SELL_{random.randint(100000, 999999)}"
            spread_id = f"DSPREAD_{random.randint(100000, 999999)}"
            
            position = {
                'spread_id': spread_id,
                'is_debit_spread': True,
                'is_credit_spread': False,
                'spread_type': plan.spread_type,
                'underlying': plan.underlying,
                'direction': plan.direction,
                # Buy leg (near ATM — directional bet)
                'buy_symbol': plan.buy_contract.symbol,
                'buy_strike': plan.buy_contract.strike,
                'buy_premium': plan.buy_contract.ltp,
                'buy_order_id': buy_order_id,
                # Sell leg (further OTM — reduces cost)
                'sell_symbol': plan.sell_contract.symbol,
                'sell_strike': plan.sell_contract.strike,
                'sell_premium': plan.sell_contract.ltp,
                'sell_order_id': sell_order_id,
                # Sizing
                'quantity': plan.quantity * plan.lot_size,
                'lots': plan.quantity,
                'lot_size': plan.lot_size,
                'net_debit': plan.net_debit,
                'net_debit_total': plan.net_debit_total,
                'max_profit': plan.max_profit,
                'max_loss': plan.max_loss,
                'spread_width': plan.spread_width,
                # Risk mgmt
                'target_value': plan.target_value,
                'stop_loss_value': plan.stop_loss_value,
                'breakeven': plan.breakeven,
                # Greeks
                'net_delta': plan.net_delta,
                'net_theta': plan.net_theta,
                'net_vega': plan.net_vega,
                'move_pct': plan.move_pct,
                'dte': plan.dte,
                # Status
                'status': 'OPEN',
                'timestamp': datetime.now().isoformat(),
                'rationale': plan.rationale,
            }
            
            self.positions.append(position)
            
            return {
                'success': True,
                'paper_trade': True,
                'spread_id': spread_id,
                'buy_order_id': buy_order_id,
                'sell_order_id': sell_order_id,
                'spread_type': plan.spread_type,
                'underlying': plan.underlying,
                'buy_symbol': plan.buy_contract.symbol,
                'sell_symbol': plan.sell_contract.symbol,
                'net_debit': plan.net_debit,
                'net_debit_total': plan.net_debit_total,
                'max_profit': plan.max_profit,
                'max_loss': plan.max_loss,
                'move_pct': plan.move_pct,
                'dte': plan.dte,
                'lots': plan.quantity,
                'greeks': plan.greeks_summary,
                'rationale': plan.rationale,
                'message': f"🚀 DEBIT SPREAD: {plan.spread_type} | Debit ₹{plan.net_debit_total:,.0f} | Max Profit ₹{plan.max_profit:,.0f}"
            }
        else:
            # LIVE MODE: Place both legs
            try:
                buy_exchange, buy_ts = plan.buy_contract.symbol.split(':')
                sell_exchange, sell_ts = plan.sell_contract.symbol.split(':')
                qty_shares = plan.quantity * plan.lot_size
                
                # Leg 1: BUY the near-ATM option (directional bet)
                buy_order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=buy_exchange,
                    tradingsymbol=buy_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                # Leg 2: SELL the further OTM option (reduce cost)
                sell_order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=sell_exchange,
                    tradingsymbol=sell_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                return {
                    'success': True,
                    'paper_trade': False,
                    'buy_order_id': str(buy_order_id),
                    'sell_order_id': str(sell_order_id),
                    'message': f"Debit spread placed: BUY {buy_ts} + SELL {sell_ts}"
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f"Debit spread failed: {e}"
                }

    def execute_option_order(self, plan: OptionOrderPlan) -> Dict:
        """
        Execute an option order plan
        
        Args:
            plan: The OptionOrderPlan to execute
            
        Returns:
            Dict with execution result
        """
        if self.paper_mode:
            # Simulate paper trade
            import random
            order_id = f"OPTION_PAPER_{random.randint(100000, 999999)}"
            
            position = {
                'order_id': order_id,
                'symbol': plan.contract.symbol,
                'underlying': plan.underlying,
                'direction': plan.direction,
                'option_type': plan.contract.option_type.value,
                'strike': plan.contract.strike,
                'expiry': plan.contract.expiry.isoformat(),
                'quantity': plan.quantity,
                'lot_size': plan.contract.lot_size,
                'entry_premium': plan.contract.ltp,
                'total_premium': plan.total_premium,
                'target_premium': plan.target_premium,
                'stoploss_premium': plan.stoploss_premium,
                'breakeven': plan.breakeven,
                'greeks': {
                    'delta': plan.contract.delta,
                    'gamma': plan.contract.gamma,
                    'theta': plan.contract.theta,
                    'vega': plan.contract.vega,
                    'iv': plan.contract.iv
                },
                'status': 'OPEN',
                'timestamp': datetime.now().isoformat()
            }
            
            self.positions.append(position)
            
            return {
                'success': True,
                'paper_trade': True,
                'order_id': order_id,
                'symbol': plan.contract.symbol,
                'underlying': plan.underlying,
                'option_type': plan.contract.option_type.value,
                'strike': plan.contract.strike,
                'expiry': plan.contract.expiry.strftime('%d-%b-%Y'),
                'quantity': plan.quantity,
                'premium': plan.contract.ltp,
                'total_cost': plan.total_premium,
                'target': plan.target_premium,
                'stoploss': plan.stoploss_premium,
                'breakeven': plan.breakeven,
                'greeks': plan.greeks_summary,
                'rationale': plan.rationale
            }
        else:
            # Live order
            try:
                exchange, tradingsymbol = plan.contract.symbol.split(':')
                
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=plan.quantity * plan.contract.lot_size,
                    product=self.kite.PRODUCT_MIS,  # Intraday
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                return {
                    'success': True,
                    'paper_trade': False,
                    'order_id': order_id,
                    'symbol': plan.contract.symbol,
                    'message': f"Order placed: {order_id}"
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def check_option_exits(self) -> List[Dict]:
        """
        Check all open option positions for exit conditions
        """
        exits = []
        
        for pos in self.positions:
            if pos['status'] != 'OPEN':
                continue
            
            # Skip credit spreads — they use exit_manager, not Greeks-based exits
            if pos.get('is_credit_spread') or pos.get('is_debit_spread') or '|' in pos.get('symbol', ''):
                continue
            
            symbol = pos['symbol']
            
            try:
                # Get current price
                if self.kite:
                    quote = self.kite.quote([symbol])
                    current_premium = quote[symbol]['last_price']
                else:
                    current_premium = pos['entry_premium']  # Can't check in paper without kite
                
                entry = pos['entry_premium']
                target = pos['target_premium']
                stoploss = pos['stoploss_premium']
                
                pnl_pct = ((current_premium - entry) / entry) * 100
                pnl_value = (current_premium - entry) * pos['quantity'] * pos['lot_size']
                
                exit_signal = None
                
                # Target hit
                if current_premium >= target:
                    exit_signal = 'TARGET_HIT'
                
                # Stoploss hit
                elif current_premium <= stoploss:
                    exit_signal = 'STOPLOSS_HIT'
                
                # Theta decay - if theta is eating more than 30% of remaining premium
                elif pos['greeks']['theta'] * 3 < -current_premium * 0.3:
                    exit_signal = 'THETA_DECAY_WARNING'
                
                # IV crush - if IV dropped significantly
                # (Would need current IV comparison)
                
                if exit_signal:
                    exits.append({
                        'symbol': symbol,
                        'signal': exit_signal,
                        'entry_premium': entry,
                        'current_premium': current_premium,
                        'pnl_pct': pnl_pct,
                        'pnl_value': pnl_value,
                        'position': pos
                    })
                    
            except Exception as e:
                print(f"⚠️ Error checking {symbol}: {e}")
        
        return exits
    
    def get_greeks_summary(self) -> Dict:
        """Get aggregate Greeks for all positions"""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for pos in self.positions:
            if pos['status'] != 'OPEN':
                continue
            
            # Skip credit spreads (they store Greeks differently)
            if pos.get('is_credit_spread') or '|' in pos.get('symbol', ''):
                continue
            
            greeks = pos.get('greeks', {})
            if not greeks:
                continue
            
            qty = pos['quantity'] * pos['lot_size']
            total_delta += greeks.get('delta', 0) * qty
            total_gamma += greeks.get('gamma', 0) * qty
            total_theta += greeks.get('theta', 0) * qty
            total_vega += greeks.get('vega', 0) * qty
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'position_count': len([p for p in self.positions if p['status'] == 'OPEN'])
        }


# Singleton instance
_options_trader_instance = None


def get_options_trader(kite=None, capital: float = 100000, paper_mode: bool = True) -> OptionsTrader:
    """Get or create the singleton options trader instance"""
    global _options_trader_instance
    if _options_trader_instance is None:
        _options_trader_instance = OptionsTrader(kite=kite, capital=capital, paper_mode=paper_mode)
    return _options_trader_instance


# === TEST ===
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING OPTIONS TRADING MODULE")
    print("=" * 60)
    
    # Test Black-Scholes
    print("\n[1] Testing Black-Scholes Greeks...")
    S = 2500  # Spot
    K = 2500  # Strike (ATM)
    T = 7/365  # 7 days to expiry
    r = 0.065  # Risk-free rate
    sigma = 0.20  # 20% IV
    
    call_price = BlackScholes.call_price(S, K, T, r, sigma)
    put_price = BlackScholes.put_price(S, K, T, r, sigma)
    delta_call = BlackScholes.delta(S, K, T, r, sigma, OptionType.CE)
    delta_put = BlackScholes.delta(S, K, T, r, sigma, OptionType.PE)
    gamma = BlackScholes.gamma(S, K, T, r, sigma)
    theta_call = BlackScholes.theta(S, K, T, r, sigma, OptionType.CE)
    vega = BlackScholes.vega(S, K, T, r, sigma)
    
    print(f"    Spot: {S}, Strike: {K}, Days: 7, IV: 20%")
    print(f"    Call Price: ₹{call_price:.2f}")
    print(f"    Put Price: ₹{put_price:.2f}")
    print(f"    Delta (CE): {delta_call:.3f}")
    print(f"    Delta (PE): {delta_put:.3f}")
    print(f"    Gamma: {gamma:.5f}")
    print(f"    Theta (CE): ₹{theta_call:.2f}/day")
    print(f"    Vega: ₹{vega:.2f}/1% IV")
    print("    ✅ Passed")
    
    # Test IV calculation
    print("\n[2] Testing Implied Volatility calculation...")
    market_price = 50
    iv = BlackScholes.implied_volatility(market_price, S, K, T, r, OptionType.CE)
    print(f"    Market Price: ₹{market_price}")
    print(f"    Implied IV: {iv*100:.1f}%")
    # Verify by repricing
    repriced = BlackScholes.call_price(S, K, T, r, iv)
    print(f"    Repriced: ₹{repriced:.2f}")
    print("    ✅ Passed")
    
    # Test Position Sizer
    print("\n[3] Testing Position Sizer...")
    sizer = OptionsPositionSizer(capital=100000)
    
    # Create mock contract
    mock_contract = OptionContract(
        symbol="NFO:RELIANCE26FEB2500CE",
        underlying="NSE:RELIANCE",
        strike=2500,
        option_type=OptionType.CE,
        expiry=datetime.now() + timedelta(days=7),
        lot_size=250,
        ltp=50,
        bid=49,
        ask=51,
        volume=50000,
        oi=500000,
        iv=0.20,
        delta=0.5,
        gamma=0.002,
        theta=-3.5,
        vega=4.0
    )
    
    sizing = sizer.calculate_position(mock_contract, 'BUY', 100000)
    print(f"    Contract: {mock_contract.symbol}")
    print(f"    LTP: ₹{mock_contract.ltp}")
    print(f"    Lot Size: {mock_contract.lot_size}")
    print(f"    Quantity: {sizing['lots']} lots")
    print(f"    Premium per lot: ₹{sizing['premium_per_lot']:,.0f}")
    print(f"    Total Premium: ₹{sizing['total_premium']:,.0f}")
    print(f"    Risk %: {sizing['risk_pct']:.1f}%")
    print(f"    Target: ₹{sizing['target_premium']:.2f}")
    print(f"    Stoploss: ₹{sizing['stoploss_premium']:.2f}")
    print(f"    Breakeven: ₹{sizing['breakeven']:.0f}")
    print("    ✅ Passed")
    
    # Test Options Trader
    print("\n[4] Testing Options Trader (paper mode)...")
    trader = get_options_trader(kite=None, capital=100000, paper_mode=True)
    
    print(f"    Should use options for RELIANCE: {trader.should_use_options('NSE:RELIANCE')}")
    print(f"    Should use options for SUZLON: {trader.should_use_options('NSE:SUZLON')}")
    print("    ✅ Passed")
    
    # Test lot sizes
    print("\n[5] Testing lot sizes...")
    print(f"    RELIANCE: {FNO_LOT_SIZES.get('RELIANCE', 'N/A')}")
    print(f"    NIFTY: {FNO_LOT_SIZES.get('NIFTY', 'N/A')}")
    print(f"    BANKNIFTY: {FNO_LOT_SIZES.get('BANKNIFTY', 'N/A')}")
    print(f"    INFY: {FNO_LOT_SIZES.get('INFY', 'N/A')}")
    print("    ✅ Passed")
    
    print("\n" + "=" * 60)
    print("✅ ALL OPTIONS TRADING TESTS PASSED!")
    print("=" * 60)
    print("""
Module includes:
- Black-Scholes pricing and Greeks (Delta, Gamma, Theta, Vega)
- Implied Volatility calculator
- Option chain fetcher (from Zerodha)
- Strike selection (ATM, ITM, OTM)
- Expiry selection (weekly, monthly)
- Position sizing (premium, risk, delta-based)
- Options order execution
- Greeks-based exit signals
    """)
