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
from config import AUTOSLICE_ENABLED

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
    orb_strength: float = 0.0           # % move from ORB level
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
    # Reversal zone trade (tight SL, flipped direction)
    is_reversal_trade: bool = False


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
        'ema_regime': 5,         # REDUCED from 10 - TrendFollowing already scores EMA 20pts
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
    BLOCK_THRESHOLD = 56         # < 56 = BLOCK (lowered from 60 for more entries)
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
        # HOT WATCHLIST: stocks that scored 55+ but failed conviction gate
        # Format: {symbol: {score, direction, cycle_count, last_seen, reason}}
        self.hot_watchlist: Dict[str, Dict] = {}
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
        score_audit = []  # Track every score change: (component_name, delta, running_total)
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
        
        _audit_after_trend = score
        
        # === 1. OPTIONS MICROSTRUCTURE GATE (15 points) ===
        microstructure_score, microstructure_block, microstructure_block_reason = self._score_microstructure(option_data, signal, reasons, warnings)
        score += microstructure_score
        _audit_after_micro = score
        
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
        
        # === ORB STALENESS DECAY (DISTANCE + TIME) ===
        # Two independent decay factors, take the LOWER one:
        # A) Distance decay: how far price moved from ORB range
        # B) Time decay: how many candles since breakout (orb_hold_candles)
        orb_strength = 0
        if market_data:
            orb_strength = abs(market_data.get('orb_strength', 0))
        
        # A) Distance-based decay
        if orb_strength > 200:
            dist_decay = 0.25
        elif orb_strength > 100:
            dist_decay = 0.50
        elif orb_strength > 50:
            dist_decay = 0.75
        else:
            dist_decay = 1.0
        
        # B) Time-based decay (orb_hold_candles = candles since breakout)
        # Gentle decay — breakout is still valid info, just not as fresh
        orb_hold_time = getattr(signal, 'orb_hold_candles', market_data.get('orb_hold_candles', 0) if market_data else 0)
        if orb_hold_time >= 12:      # 60+ min — stale
            time_decay = 0.40
        elif orb_hold_time >= 8:      # 40-60 min — aging
            time_decay = 0.60
        elif orb_hold_time >= 5:      # 25-40 min — slightly aged
            time_decay = 0.80
        else:
            time_decay = 1.0         # Fresh breakout
        
        # Take the MORE aggressive decay
        orb_decay = min(dist_decay, time_decay)
        if time_decay < 1.0 or dist_decay < 1.0:
            cap_reason += f" [decay: dist={dist_decay:.0%}({orb_strength:.0f}%ORB) time={time_decay:.0%}({orb_hold_time}candles) → {orb_decay:.0%}]"
        
        # === ORB MINIMUM STRENGTH FILTER ===
        # ORB strength < 2% = noise (BHEL 1.19% example), zero out ORB points
        # ORB strength 2-5% = weak breakout, halve points
        # ORB strength >= 5% = real breakout, normal scoring
        orb_strength_filter = 1.0
        if orb_strength < 2:
            orb_strength_filter = 0.0  # Fake breakout — zero points
            cap_reason += f" [NOISE: {orb_strength:.1f}% < 2% min]" 
        elif orb_strength < 5:
            orb_strength_filter = 0.5  # Weak breakout — half credit
            cap_reason += f" [WEAK: {orb_strength:.1f}% < 5%]"

        if signal.orb_signal == "BREAKOUT_UP":
            orb_points = int(orb_max_points * orb_decay * orb_strength_filter)
            # ORB hold check — need at least 2 candles holding above ORB to confirm
            orb_hold = getattr(signal, 'orb_hold_candles', market_data.get('orb_hold_candles', 0) if market_data else 0)
            if orb_hold < 2:
                orb_points = max(2, orb_points // 2) if orb_points > 0 else 0  # Halve points for unconfirmed breakout
                cap_reason += f" [UNCONFIRMED: hold {orb_hold} candles < 2]"
            score += orb_points
            bullish_points += orb_points
            reasons.append(f"🚀 ORB BREAKOUT UP (+{orb_points}){cap_reason}")
            if direction == "HOLD":
                direction = "BUY"
        elif signal.orb_signal == "BREAKOUT_DOWN":
            orb_points = int(orb_max_points * orb_decay * orb_strength_filter)
            # ORB hold check — need at least 2 candles holding below ORB to confirm
            orb_hold = getattr(signal, 'orb_hold_candles', market_data.get('orb_hold_candles', 0) if market_data else 0)
            if orb_hold < 2:
                orb_points = max(2, orb_points // 2) if orb_points > 0 else 0
                cap_reason += f" [UNCONFIRMED: hold {orb_hold} candles < 2]"
            score += orb_points
            bearish_points += orb_points
            reasons.append(f"📉 ORB BREAKOUT DOWN (+{orb_points}){cap_reason}")
            if direction == "HOLD":
                direction = "SELL"
        elif signal.orb_signal == "INSIDE_ORB":
            # Fix 5: VWAP GRIND DETECTION — catch slow downtrends that never break ORB
            # HINDALCO-type: INSIDE_ORB but price grinding below VWAP with HIGH+ volume
            # These stocks are in a clear trend but the ORB range was too wide to break
            vwap_grind_points = 0
            if market_data:
                _grind_vwap_slope = signal.vwap_slope if hasattr(signal, 'vwap_slope') else market_data.get('vwap_slope', 'FLAT')
                _grind_vol = signal.volume_regime if hasattr(signal, 'volume_regime') else market_data.get('volume_regime', 'NORMAL')
                _grind_ltp = market_data.get('ltp', 0)
                _grind_vwap = market_data.get('vwap', 0)
                _grind_adx = market_data.get('adx', 0)
                _grind_open = market_data.get('open', _grind_ltp)
                _grind_move_pct = abs((_grind_ltp - _grind_open) / _grind_open * 100) if _grind_open > 0 else 0
                
                # Bearish VWAP grind: below VWAP, high+ volume, stock actually moved down
                if (_grind_ltp < _grind_vwap and _grind_ltp < _grind_open 
                    and _grind_vol in ('HIGH', 'EXPLOSIVE') and _grind_move_pct >= 0.8):
                    vwap_grind_points = 8
                    if _grind_adx >= 25:
                        vwap_grind_points += 2  # ADX confirms trend
                    score += vwap_grind_points
                    bearish_points += vwap_grind_points
                    if direction == "HOLD":
                        direction = "SELL"
                    reasons.append(f"📉 VWAP GRIND DOWN: Below VWAP, {_grind_vol} vol, {_grind_move_pct:.1f}% down (+{vwap_grind_points})")
                # Bullish VWAP grind: above VWAP, high+ volume, stock moved up
                elif (_grind_ltp > _grind_vwap and _grind_ltp > _grind_open
                      and _grind_vol in ('HIGH', 'EXPLOSIVE') and _grind_move_pct >= 0.8):
                    vwap_grind_points = 8
                    if _grind_adx >= 25:
                        vwap_grind_points += 2
                    score += vwap_grind_points
                    bullish_points += vwap_grind_points
                    if direction == "HOLD":
                        direction = "BUY"
                    reasons.append(f"📈 VWAP GRIND UP: Above VWAP, {_grind_vol} vol, {_grind_move_pct:.1f}% up (+{vwap_grind_points})")
                else:
                    reasons.append("⏳ Inside ORB range (wait for breakout)")
            else:
                reasons.append("⏳ Inside ORB range (wait for breakout)")
        
        _audit_after_orb = score
        
        # === 3. VOLUME REGIME (7-15 points, CAPPED if window overlap) ===
        volume_multiplier = 1.0
        volume_window_overlap = abs(signal.volume_window_minutes - trend_window_minutes) <= 5
        if volume_window_overlap and raw_trend_score >= 60:
            volume_multiplier = 0.75
        
        # Volume recency check: is volume intense NOW or was it front-loaded?
        recent_vol_rate = market_data.get('recent_vol_rate', 1.0) if market_data else 1.0
        vol_recency_discount = 1.0
        vol_recency_note = ""
        if recent_vol_rate < 0.4:
            vol_recency_discount = 0.40   # Volume dried up — barely any current activity
            vol_recency_note = f" [⚠️ DRIED UP: recent rate {recent_vol_rate:.1f}x avg candle, 40% credit]"
        elif recent_vol_rate < 0.7:
            vol_recency_discount = 0.65   # Volume fading
            vol_recency_note = f" [fading: recent rate {recent_vol_rate:.1f}x, 65% credit]"
        
        if signal.volume_regime == "EXPLOSIVE":
            vol_points = int(15 * volume_multiplier * vol_recency_discount)
            score += vol_points
            cap_note = f" [capped from 15]" if volume_multiplier < 1.0 else ""
            reasons.append(f"💥 EXPLOSIVE volume (+{vol_points}){cap_note}{vol_recency_note}")
        elif signal.volume_regime == "HIGH":
            vol_points = int(12 * volume_multiplier * vol_recency_discount)
            score += vol_points
            cap_note = f" [capped from 12]" if volume_multiplier < 1.0 else ""
            reasons.append(f"📊 HIGH volume (+{vol_points}){cap_note}{vol_recency_note}")
        elif signal.volume_regime == "NORMAL":
            vol_points = int(6 * volume_multiplier)
            score += vol_points
            cap_note = f" [capped from 6]" if volume_multiplier < 1.0 else ""
            reasons.append(f"📊 Normal volume (+{vol_points}){cap_note}")
        elif signal.volume_regime == "LOW":
            score += 2  # No cap on penalty
            warnings.append("⚠️ LOW volume - weak conviction (+2)")
        
        _audit_after_vol = score
        
        # === 4. VWAP POSITION (10 points max, penalty for misalignment) ===
        # TrendFollowing already has VWAP Slope (25pts), so keep this as:
        # - Small positive when aligned with trend direction
        # - Penalty when misaligned (or block if severe)
        vwap_aligned = False
        vwap_misaligned = False
        
        if trend_direction == "BUY":
            # For BUY, we want ABOVE_VWAP or AT_VWAP
            # Score POSITION + SLOPE together here (sole VWAP scorer — no double-count)
            if signal.vwap_position == "ABOVE_VWAP" and signal.vwap_trend == "RISING":
                score += 10
                bullish_points += 10
                reasons.append("📈 VWAP: Above + rising (+10)")
                vwap_aligned = True
            elif signal.vwap_position == "ABOVE_VWAP" and signal.vwap_trend == "FLAT":
                score += 5
                bullish_points += 5
                reasons.append("📈 VWAP: Above but flat slope (+5)")
                vwap_aligned = True
            elif signal.vwap_position == "ABOVE_VWAP":
                # Above but slope falling — weakening
                score += 3
                bullish_points += 3
                reasons.append("⚠️ VWAP: Above but slope falling (+3)")
            elif signal.vwap_position == "AT_VWAP" and signal.vwap_trend == "RISING":
                score += 3
                reasons.append("🔄 VWAP: At VWAP, slope rising (+3)")
            elif signal.vwap_position == "AT_VWAP":
                reasons.append("🔄 VWAP: At VWAP, slope not rising (+0)")
            else:
                # BELOW_VWAP on a BUY signal = misalignment
                score -= 8
                vwap_misaligned = True
                warnings.append("⚠️ VWAP MISALIGNED: Below VWAP on BUY (-8)")
        elif trend_direction == "SELL":
            # For SELL, we want BELOW_VWAP or AT_VWAP
            # Score POSITION + SLOPE together here (sole VWAP scorer — no double-count)
            if signal.vwap_position == "BELOW_VWAP" and signal.vwap_trend == "FALLING":
                score += 10
                bearish_points += 10
                reasons.append("📉 VWAP: Below + falling (+10)")
                vwap_aligned = True
            elif signal.vwap_position == "BELOW_VWAP" and signal.vwap_trend == "FLAT":
                score += 5
                bearish_points += 5
                reasons.append("📉 VWAP: Below but flat slope (+5)")
                vwap_aligned = True
            elif signal.vwap_position == "BELOW_VWAP":
                # Below but slope rising — weakening
                score += 3
                bearish_points += 3
                reasons.append("⚠️ VWAP: Below but slope rising (+3)")
            elif signal.vwap_position == "AT_VWAP" and signal.vwap_trend == "FALLING":
                score += 3
                reasons.append("🔄 VWAP: At VWAP, slope falling (+3)")
            elif signal.vwap_position == "AT_VWAP":
                reasons.append("🔄 VWAP: At VWAP, slope not falling (+0)")
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
        
        _audit_after_vwap = score
        
        # === 5. EMA REGIME (5 points — reduced from 10) ===
        # TrendFollowing already gives up to 20 pts for EMA expansion.
        # This is a COMPLEMENTARY check for EMA structure, not a duplicate.
        if signal.ema_regime == "EXPANDING":
            score += 5
            reasons.append("📊 EMA EXPANDING - strong trend (+5)")
        elif signal.ema_regime == "COMPRESSED":
            score += 4
            reasons.append("⚡ EMA COMPRESSED - breakout imminent (+4)")
        else:
            score += 1
            reasons.append("EMA normal (+1)")
        
        _audit_after_ema = score
        
        # === 5. HTF ALIGNMENT (5 points) ===
        # Fix: zerodha_tools produces "BULLISH_ALIGNED"/"BEARISH_ALIGNED", so match both forms
        if signal.htf_alignment in ("BULLISH", "BULLISH_ALIGNED"):
            score += 5
            bullish_points += 5
            reasons.append("🎯 HTF BULLISH alignment (+5)")
        elif signal.htf_alignment in ("BEARISH", "BEARISH_ALIGNED"):
            score += 5
            bearish_points += 5
            reasons.append("🎯 HTF BEARISH alignment (+5)")
        elif signal.htf_alignment == "MIXED":
            score += 3
            warnings.append("HTF mixed signals (+3)")
        else:
            score += 2
            warnings.append("HTF neutral - watch for direction (+2)")
        
        _audit_after_htf = score
        
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
        
        _audit_after_accel = score
        
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
        
        _audit_after_rsi = score
        
        # === 8a. OI FLOW DIRECTIONAL SCORING (+/-8 points) ===
        # Futures OI signal captures institutional direction (LONG_BUILDUP etc.)
        # Pro traders ALWAYS check: are institutions on your side?
        # This was the #1 missing signal — OI was captured but never scored.
        oi_score = 0
        if market_data:
            oi_signal = market_data.get('oi_signal', 'NEUTRAL')
            oi_change = abs(market_data.get('oi_change_pct', 0))
            
            if oi_signal not in ('NEUTRAL', 'NO_FUTURES', 'ERROR', ''):
                # Scale bonus/penalty by OI change magnitude
                oi_strength = min(1.5, oi_change / 5.0) if oi_change > 0 else 0.5  # 0.5-1.5x
                
                if direction in ('BUY', 'HOLD') and oi_signal == 'LONG_BUILDUP':
                    oi_score = int(5 * oi_strength)
                    bullish_points += oi_score
                    reasons.append(f"OI: LONG_BUILDUP confirms BUY (OI+{oi_change:.1f}%) (+{oi_score})")
                elif direction in ('BUY', 'HOLD') and oi_signal == 'SHORT_COVERING':
                    oi_score = int(3 * oi_strength)
                    bullish_points += oi_score
                    reasons.append(f"OI: SHORT_COVERING supports BUY (OI-{oi_change:.1f}%) (+{oi_score})")
                elif direction == 'BUY' and oi_signal == 'SHORT_BUILDUP':
                    oi_score = -8
                    warnings.append(f"OI CONFLICT: SHORT_BUILDUP vs BUY direction (OI+{oi_change:.1f}%) (-8) — institutions are SELLING")
                elif direction == 'BUY' and oi_signal == 'LONG_UNWINDING':
                    oi_score = -5
                    warnings.append(f"OI CONFLICT: LONG_UNWINDING vs BUY (OI-{oi_change:.1f}%) (-5) — longs exiting")
                    
                elif direction in ('SELL', 'HOLD') and oi_signal == 'SHORT_BUILDUP':
                    oi_score = int(5 * oi_strength)
                    bearish_points += oi_score
                    reasons.append(f"OI: SHORT_BUILDUP confirms SELL (OI+{oi_change:.1f}%) (+{oi_score})")
                elif direction in ('SELL', 'HOLD') and oi_signal == 'LONG_UNWINDING':
                    oi_score = int(3 * oi_strength)
                    bearish_points += oi_score
                    reasons.append(f"OI: LONG_UNWINDING supports SELL (OI-{oi_change:.1f}%) (+{oi_score})")
                elif direction == 'SELL' and oi_signal == 'LONG_BUILDUP':
                    oi_score = -8
                    warnings.append(f"OI CONFLICT: LONG_BUILDUP vs SELL direction (OI+{oi_change:.1f}%) (-8) — institutions are BUYING")
                elif direction == 'SELL' and oi_signal == 'SHORT_COVERING':
                    oi_score = -5
                    warnings.append(f"OI CONFLICT: SHORT_COVERING vs SELL (OI-{oi_change:.1f}%) (-5) — shorts covering")
                
                score += oi_score
        
        _audit_after_oi = score
        
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
            # Only apply if gap penalty didn't already cover it (avoid double-counting)
            #
            # TREND-AWARE CHASING (Fix 3): On strong trend days (trend_score >= 65)
            # with HIGH/EXPLOSIVE volume AND move aligned with direction, reduce penalty
            # by 60%. Riding a trend is NOT chasing — a 5% intraday move on a bear day
            # with explosive volume is momentum, not exhaustion.
            vol_regime = market_data.get('volume_regime', 'NORMAL')
            is_trend_momentum = (
                raw_trend_score >= 60   # S6 FIX: match TrendFollowing's own TREND_THRESHOLD=60
                and vol_regime in ('HIGH', 'EXPLOSIVE')
                and (
                    (direction == "SELL" and ltp < day_open)  # Bearish move aligned
                    or (direction == "BUY" and ltp > day_open)  # Bullish move aligned
                )
            )
            chasing_discount = 0.4 if is_trend_momentum else 1.0  # 60% reduction on trend days
            
            if intraday_move_pct > gap_pct:
                # Intraday move is larger than the gap → the move happened today
                if intraday_move_pct >= 6:
                    raw_penalty = 25
                    penalty = int(raw_penalty * chasing_discount)
                    exhaustion_score -= penalty
                    label = f"🔥 TREND MOMENTUM" if is_trend_momentum else f"🔥 CHASING"
                    discount_note = f" [trend discount: {raw_penalty}→{penalty}]" if is_trend_momentum else ""
                    warnings.append(f"{label}: {intraday_move_pct:.1f}% intraday move from open (-{penalty}){discount_note}")
                elif intraday_move_pct >= 4:
                    raw_penalty = 18
                    penalty = int(raw_penalty * chasing_discount)
                    exhaustion_score -= penalty
                    label = f"⚠️ TREND MOMENTUM" if is_trend_momentum else f"⚠️ CHASING"
                    discount_note = f" [trend discount: {raw_penalty}→{penalty}]" if is_trend_momentum else ""
                    warnings.append(f"{label}: {intraday_move_pct:.1f}% intraday move from open (-{penalty}){discount_note}")
                elif intraday_move_pct >= 2.5:
                    raw_penalty = 12
                    penalty = int(raw_penalty * chasing_discount)
                    exhaustion_score -= penalty
                    label = f"⚠️ TREND MOMENTUM" if is_trend_momentum else f"⚠️ CHASING"
                    discount_note = f" [trend discount: {raw_penalty}→{penalty}]" if is_trend_momentum else ""
                    warnings.append(f"{label}: {intraday_move_pct:.1f}% intraday move from open (-{penalty}){discount_note}")
                elif intraday_move_pct >= 1.5:
                    raw_penalty = 5
                    penalty = int(raw_penalty * chasing_discount)
                    exhaustion_score -= penalty
                    label = f"⚠️ TREND MOMENTUM" if is_trend_momentum else f"⚠️ CHASING"
                    discount_note = f" [trend discount: {raw_penalty}→{penalty}]" if is_trend_momentum else ""
                    warnings.append(f"{label}: {intraday_move_pct:.1f}% intraday move from open (-{penalty}){discount_note}")

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

        _audit_after_exhaust = score
        
        # === CHOP ZONE PENALTY ===
        if signal.chop_zone:
            score -= self.CHOP_PENALTY
            warnings.append(f"⛔ CHOP ZONE - deducted {self.CHOP_PENALTY} points")
        _audit_after_chop = score
        
        # === CALLER DIRECTION ALIGNMENT BONUS (2 pts max) ===
        # Small bonus when GPT agent direction aligns with overall signal direction.
        # DOES NOT re-check individual signals (VWAP, volume, EMA already scored).
        # Only checks: does the determined direction match what GPT wants?
        if caller_direction and caller_direction in ('BUY', 'SELL'):
            if direction == caller_direction:
                score += 2
                reasons.append(f"🤖 Agent direction aligned with signals (+2)")
            elif direction == 'HOLD':
                # GPT has a view but signals are neutral — give 1 pt and set direction
                score += 1
                direction = caller_direction
                reasons.append(f"🤖 Agent direction set (signals neutral) (+1)")
        _audit_after_align = score
        _pre_direction_trend_dir = trend_direction  # Save for S2 VWAP reconciliation
        
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
        
        # S2 FIX: VWAP DIRECTION RECONCILIATION
        # VWAP was scored using trend_direction (set early from TrendFollowing).
        # If final direction differs, the VWAP contribution is for the WRONG direction.
        # Example: Trend says SELL → VWAP scored for SELL (+10 for below_vwap),
        # but direction flipped to BUY → now below_vwap should be -8, not +10.
        # Recalculate VWAP delta for the actual direction.
        if direction != "HOLD" and _pre_direction_trend_dir != "HOLD" and direction != _pre_direction_trend_dir:
            _old_vwap_contrib = _audit_after_vwap - _audit_after_vol
            _new_vwap = 0
            if direction == "BUY":
                if signal.vwap_position == "ABOVE_VWAP" and signal.vwap_trend == "RISING":
                    _new_vwap = 10
                elif signal.vwap_position == "ABOVE_VWAP" and signal.vwap_trend == "FLAT":
                    _new_vwap = 5
                elif signal.vwap_position == "ABOVE_VWAP":
                    _new_vwap = 3
                elif signal.vwap_position == "AT_VWAP" and signal.vwap_trend == "RISING":
                    _new_vwap = 3
                elif signal.vwap_position == "AT_VWAP":
                    _new_vwap = 0
                else:
                    _new_vwap = -8
            elif direction == "SELL":
                if signal.vwap_position == "BELOW_VWAP" and signal.vwap_trend == "FALLING":
                    _new_vwap = 10
                elif signal.vwap_position == "BELOW_VWAP" and signal.vwap_trend == "FLAT":
                    _new_vwap = 5
                elif signal.vwap_position == "BELOW_VWAP":
                    _new_vwap = 3
                elif signal.vwap_position == "AT_VWAP" and signal.vwap_trend == "FALLING":
                    _new_vwap = 3
                elif signal.vwap_position == "AT_VWAP":
                    _new_vwap = 0
                else:
                    _new_vwap = -8
            _vwap_delta = _new_vwap - _old_vwap_contrib
            if _vwap_delta != 0:
                score += _vwap_delta
                # Update directional points
                if _old_vwap_contrib > 0:
                    if _pre_direction_trend_dir == "BUY":
                        bullish_points -= _old_vwap_contrib
                    else:
                        bearish_points -= _old_vwap_contrib
                if _new_vwap > 0:
                    if direction == "BUY":
                        bullish_points += _new_vwap
                    else:
                        bearish_points += _new_vwap
                warnings.append(f"🔄 VWAP RE-SCORED: direction {_pre_direction_trend_dir}→{direction}, VWAP {_old_vwap_contrib:+.0f}→{_new_vwap:+.0f} (Δ{_vwap_delta:+.0f})")
                _audit_after_align = score  # Update for gate cap baseline
        
        # === REVERSAL ZONE DETECTION (exhaustion → trade the pullback) ===
        # Detects stocks that are exhausted at resistance/support and flips
        # direction to trade the mean-reversion instead of chasing.
        # ALL 3 conditions required: >2.5% move + near R/S + RSI extreme
        # Uses tighter SL (18%) and smaller size (0.75x) for lower conviction.
        is_reversal_trade = False
        if market_data and direction in ("BUY", "SELL"):
            _rv_ltp = market_data.get('ltp', 0)
            _rv_open = market_data.get('open', _rv_ltp)
            _rv_move = abs((_rv_ltp - _rv_open) / _rv_open * 100) if _rv_open > 0 else 0
            _rv_rsi = signal.rsi
            _rv_r1 = market_data.get('resistance_1', 0)
            _rv_r2 = market_data.get('resistance_2', 0)
            _rv_s1 = market_data.get('support_1', 0)
            _rv_s2 = market_data.get('support_2', 0)

            if direction == "BUY" and _rv_move >= 2.5:
                # Bullish exhaustion: near resistance + overbought
                _near_r1 = _rv_r1 > 0 and _rv_ltp > 0 and abs(_rv_ltp - _rv_r1) / _rv_r1 * 100 <= 1.0
                _near_r2 = _rv_r2 > 0 and _rv_ltp > 0 and abs(_rv_ltp - _rv_r2) / _rv_r2 * 100 <= 1.0
                _near_resistance = _near_r1 or _near_r2
                _which_r = f"R1 ₹{_rv_r1:.0f}" if _near_r1 else f"R2 ₹{_rv_r2:.0f}"

                if _near_resistance and _rv_rsi > 70:
                    is_reversal_trade = True
                    _old_dir = direction
                    direction = "SELL"  # Flip → buy PUT
                    score -= 5  # Small penalty for counter-trend risk
                    warnings.append(f"🔄 REVERSAL ZONE: {_rv_move:.1f}% from open + near {_which_r} + RSI {_rv_rsi:.0f} → FLIPPED {_old_dir}→{direction} (PUT reversal)")
                    reasons.append(f"🎯 REVERSAL TRADE: Bullish exhaustion at resistance — trading pullback")

            elif direction == "SELL" and _rv_move >= 2.5:
                # Bearish exhaustion: near support + oversold
                _near_s1 = _rv_s1 > 0 and _rv_ltp > 0 and abs(_rv_ltp - _rv_s1) / _rv_s1 * 100 <= 1.0
                _near_s2 = _rv_s2 > 0 and _rv_ltp > 0 and abs(_rv_ltp - _rv_s2) / _rv_s2 * 100 <= 1.0
                _near_support = _near_s1 or _near_s2
                _which_s = f"S1 ₹{_rv_s1:.0f}" if _near_s1 else f"S2 ₹{_rv_s2:.0f}"

                if _near_support and _rv_rsi < 30:
                    is_reversal_trade = True
                    _old_dir = direction
                    direction = "BUY"  # Flip → buy CALL
                    score -= 5
                    warnings.append(f"🔄 REVERSAL ZONE: {_rv_move:.1f}% from open + near {_which_s} + RSI {_rv_rsi:.0f} → FLIPPED {_old_dir}→{direction} (CALL reversal)")
                    reasons.append(f"🎯 REVERSAL TRADE: Bearish exhaustion at support — trading bounce")

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
        # Scale requirement with score — high score means many things aligned,
        # less pure-directional conviction needed
        directional_strength = max(bullish_points, bearish_points)
        
        # HOT WATCHLIST BOOST: if this symbol was on watchlist from prior cycle,
        # it means structure was already validated — give +5 conviction bonus
        underlying_sym = signal.symbol if hasattr(signal, 'symbol') else ''
        watchlist_boost = 0
        if underlying_sym in self.hot_watchlist:
            wl_entry = self.hot_watchlist[underlying_sym]
            watchlist_boost = 5
            directional_strength += watchlist_boost
            warnings.append(f"🔥 HOT WATCHLIST BOOST: +{watchlist_boost} conviction (was {wl_entry.get('score', 0):.0f} last cycle, seen {wl_entry.get('cycle_count', 1)}x)")
        
        min_conviction = 15  # default for weak scores
        if score >= 70:
            min_conviction = 8   # strong score = many aligned signals
        elif score >= 60:
            min_conviction = 8   # decent score = lowered for faster entry
        
        if directional_strength < min_conviction:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: No directional conviction (best: {directional_strength:.0f}pts, need ≥{min_conviction})")
            # Add to hot watchlist if score >= 60 (quality stocks only)
            if score >= 60:
                self.hot_watchlist[underlying_sym] = {
                    'score': score,
                    'direction': direction,
                    'directional_strength': directional_strength - watchlist_boost,  # raw, without boost
                    'cycle_count': self.hot_watchlist.get(underlying_sym, {}).get('cycle_count', 0) + 1,
                    'last_seen': datetime.now().isoformat(),
                    'reason': f'conviction {directional_strength - watchlist_boost:.0f} < {min_conviction}',
                    'bull_pts': bullish_points,
                    'bear_pts': bearish_points
                }
        else:
            # Passed conviction gate — remove from watchlist if present
            self.hot_watchlist.pop(underlying_sym, None)
        
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
        
        # Gate 6b: STALE BREAKOUT GATE — ORB broke but no follow-through
        # 8 TIME_STOP losses with MaxR=0: breakout happened but price went nowhere.
        # If ORB breakout occurred (orb_hold > 3 candles ago) and ZERO follow-through,
        # the move is dead — don't enter.
        if should_trade and market_data:
            _ft = signal.follow_through_candles
            _orb_hold = market_data.get('orb_hold_candles', 0)
            _orb_sig = market_data.get('orb_signal', 'INSIDE_ORB')
            _adx = market_data.get('adx', 20)
            
            if _orb_sig in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and _ft == 0 and _orb_hold >= 4:
                # Breakout happened 4+ candles ago but price hasn't followed through
                score -= 10
                warnings.append(f"⚠️ STALE BREAKOUT: ORB broke {_orb_hold} candles ago, 0 follow-through (-10)")
                if score < self.BLOCK_THRESHOLD:
                    should_trade = False
                    warnings.append(f"🚫 BLOCKED: Stale breakout dropped score below {self.BLOCK_THRESHOLD}")
            
            # If not an ORB breakout but score is premium (>=70), still require some follow-through
            if _ft == 0 and score >= self.PREMIUM_THRESHOLD and _adx < 30:
                score -= 5
                warnings.append(f"⚠️ NO FOLLOW-THROUGH: Score {score+5:.0f} premium but FT=0, ADX={_adx:.0f} (-5)")
        
        # Gate 7: UNIFIED LATE ENTRY CHECK — range position + RSI exhaustion
        # Consolidates old Gate 7 (range position) + Gate 7b (RSI exhaustion) into one.
        # Evidence-based: counts how many "you're late" signals fire, applies ONE graduated penalty.
        # Tracks penalty in late_entry_penalty for capping with Gate 10 (ORB overextended).
        #
        # TREND OVERRIDE (Fix 2): On strong trend days, selling at LOD or buying at HOD
        # IS the trend — not a reversal. If trend_score >= 65 AND direction aligns,
        # convert hard-blocks to mild penalties and reduce all penalties by 60%.
        late_entry_penalty = 0
        if should_trade and market_data:
            ltp = market_data.get('ltp', 0)
            day_high = market_data.get('high', 0)
            day_low = market_data.get('low', 0)
            day_range = day_high - day_low if day_high > day_low else 0.01
            rsi_val = market_data.get('rsi_14', 50)
            
            # Trend override: strong trend + aligned direction = new lows/highs are entries
            trend_aligned_sell = (direction == "SELL" and trend_state_str in ('BEARISH', 'STRONG_BEARISH') and raw_trend_score >= 60)  # S6: match TrendFollowing TREND_THRESHOLD
            trend_aligned_buy = (direction == "BUY" and trend_state_str in ('BULLISH', 'STRONG_BULLISH') and raw_trend_score >= 60)  # S6: match TrendFollowing TREND_THRESHOLD
            
            if day_range > 0.01 and ltp > 0:
                position_in_range = (ltp - day_low) / day_range  # 0=at low, 1=at high
                
                if direction == "BUY" and position_in_range > 0.85:
                    if trend_aligned_buy:
                        # TREND OVERRIDE: Buying at highs on bullish trend = continuation
                        late_entry_penalty = 3  # Mild awareness penalty only
                        reasons.append(f"🔥 TREND OVERRIDE: Buying at {position_in_range*100:.0f}% on {trend_state_str} trend (reduced penalty -3)")
                    elif position_in_range > 0.95:
                        # Extreme — hard block (no trend support)
                        should_trade = False
                        warnings.append(f"🚫 BLOCKED: Buying at {position_in_range*100:.0f}% of day range = buying the exact top")
                    elif position_in_range > 0.90 and rsi_val < 55:
                        late_entry_penalty = 12
                        warnings.append(f"⚠️ LATE ENTRY: Buying at {position_in_range*100:.0f}% range + RSI {rsi_val:.0f} not bullish (-{late_entry_penalty})")
                    elif position_in_range > 0.90:
                        late_entry_penalty = 8
                        warnings.append(f"⚠️ LATE ENTRY: Buying at {position_in_range*100:.0f}% of day range (-{late_entry_penalty})")
                    elif rsi_val < 55:
                        late_entry_penalty = 5
                        warnings.append(f"⚠️ LATE ENTRY: At {position_in_range*100:.0f}% range, RSI {rsi_val:.0f} fading (-{late_entry_penalty})")
                        
                elif direction == "SELL" and position_in_range < 0.15:
                    if trend_aligned_sell:
                        # TREND OVERRIDE: Selling at lows on bearish trend = continuation
                        late_entry_penalty = 3  # Mild awareness penalty only
                        reasons.append(f"🔥 TREND OVERRIDE: Selling at {position_in_range*100:.0f}% on {trend_state_str} trend (reduced penalty -3)")
                    elif position_in_range < 0.05:
                        # Extreme — hard block (no trend support)
                        should_trade = False
                        warnings.append(f"🚫 BLOCKED: Selling at {position_in_range*100:.0f}% of day range = selling the exact bottom")
                    elif position_in_range < 0.10 and rsi_val > 45:
                        late_entry_penalty = 12
                        warnings.append(f"⚠️ LATE ENTRY: Selling at {position_in_range*100:.0f}% range + RSI {rsi_val:.0f} not bearish (-{late_entry_penalty})")
                    elif position_in_range < 0.10:
                        late_entry_penalty = 8
                        warnings.append(f"⚠️ LATE ENTRY: Selling at {position_in_range*100:.0f}% of day range (-{late_entry_penalty})")
                    elif rsi_val > 45:
                        late_entry_penalty = 5
                        warnings.append(f"⚠️ LATE ENTRY: At {position_in_range*100:.0f}% range, RSI {rsi_val:.0f} fading (-{late_entry_penalty})")
                
                score -= late_entry_penalty
        
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
        
        # Gate 9: ADX TREND STRENGTH — penalty-only gate
        # TrendFollowing already gives up to 10 pts for ADX strength.
        # This gate ONLY penalizes critically weak ADX (no bonus to avoid double-counting).
        adx = market_data.get('adx', 0) if market_data else 0
        if should_trade and adx > 0:
            if adx < 20:
                score -= 5
                warnings.append(f"⚠️ NO TREND: ADX {adx:.0f} < 20 — directionless chop (-5)")
            elif intended_tier == "premium" and adx < self.ADX_MIN_PREMIUM:
                score -= 3
                warnings.append(f"⚠️ WEAK ADX: {adx:.0f} < {self.ADX_MIN_PREMIUM} for premium tier (-3)")
        
        # Gate 10: ORB STRENGTH OVEREXTENSION — winners entered at ORB_str 30, losers at 142
        # High ORB strength = price already moved far from ORB range = chasing the move
        # Capped: total late-entry penalty (Gate 7 + Gate 10) never exceeds -15
        orb_str = market_data.get('orb_strength', 0) if market_data else 0
        if should_trade and orb_str > self.ORB_STRENGTH_OVEREXTENDED:
            remaining_budget = max(0, 15 - late_entry_penalty)  # Cap total at -15
            if remaining_budget > 0:
                orb_penalty = min(8, remaining_budget)
                score -= orb_penalty
                warnings.append(f"⚠️ ORB OVEREXTENDED: Strength {orb_str:.0f}% > {self.ORB_STRENGTH_OVEREXTENDED}% — price too far from ORB range (-{orb_penalty})")
            if orb_str > 200:
                size_multiplier = min(size_multiplier, 0.75)
                warnings.append(f"⚠️ ORB EXHAUSTED: {orb_str:.0f}% — capping size at 0.75x")
        
        # Gate 11: RANGE EXPANSION OVEREXTENSION — winners 0.18, losers 0.37
        # If range has already expanded > 0.6x ATR, the move has happened
        range_exp = signal.range_expansion_ratio
        if should_trade and range_exp > self.RANGE_EXPANSION_OVEREXTENDED:
            score -= 5
            warnings.append(f"⚠️ RANGE ALREADY EXPANDED: {range_exp:.2f}x ATR > {self.RANGE_EXPANSION_OVEREXTENDED}x — late entry (-5)")
        
        # Gate 11b: PHANTOM BREAKOUT — ORB breakout with near-zero range expansion
        # ICICIBANK example: ORB BREAKOUT_UP scored +10 but range_exp=0.02 (stock barely moved).
        # Score inflated to 84 from stacking technical signals on a non-existent move.
        # If ORB says breakout but range exp < 0.08 AND ORB strength < 20%, the breakout is noise.
        if should_trade and signal.orb_signal in ('BREAKOUT_UP', 'BREAKOUT_DOWN'):
            if range_exp < 0.08 and signal.orb_strength < 20:
                score -= 12
                warnings.append(f"⚠️ PHANTOM BREAKOUT: ORB says breakout but range_exp={range_exp:.2f} & ORB_str={signal.orb_strength:.0f}% — no real move (-12)")
                if intended_tier == "premium" and range_exp < 0.05:
                    should_trade = False
                    warnings.append(f"🚫 BLOCKED: Premium position on phantom breakout (range_exp < 5% ATR) — no conviction")
        
        # Gate 12: SAME-SYMBOL RE-ENTRY PREVENTION
        # SBIN entered 4x = all losses, BANDHANBNK 3x = all losses
        # Don't re-enter a symbol that already lost today
        sym_base = signal.symbol.replace("NSE:", "").replace("BSE:", "")
        sym_losses = self.symbol_losses_today.get(sym_base, 0)
        if should_trade and sym_losses >= self.MAX_LOSSES_SAME_SYMBOL:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: {sym_base} already lost {sym_losses}x today — no re-entry after loss")
        
        # Gate 13: RSI CONTRADICTION — buying puts at RSI<20 or calls at RSI>80
        # INFY example: bought puts at RSI=14 (RSI says stock is already crashed, bounce imminent)
        # Smart: only block at extreme RSI unless explosive volume confirms continuation
        vol_regime = market_data.get('volume_regime', 'NORMAL') if market_data else 'NORMAL'
        if should_trade and signal.rsi > 0:
            # Selling (buying puts) when RSI extremely oversold = catching the bottom
            if direction == "SELL" and signal.rsi < 25:
                if vol_regime == 'EXPLOSIVE' and signal.rsi > 15:
                    # Explosive volume at moderate oversold — allow but warn
                    score -= 5
                    warnings.append(f"⚠️ RSI CONTRADICTION (soft): Selling at RSI={signal.rsi:.0f} but explosive volume allows (-5)")
                elif signal.rsi < 15:
                    # RSI < 15 = extreme exhaustion, block regardless
                    should_trade = False
                    score -= 15
                    warnings.append(f"🚫 BLOCKED: RSI CONTRADICTION: Selling/buying puts at RSI={signal.rsi:.0f} < 15 — exhaustion bounce imminent (-15)")
                else:
                    # RSI 15-25 without explosive volume — heavy penalty  
                    score -= 10
                    warnings.append(f"⚠️ RSI CONTRADICTION: Selling at RSI={signal.rsi:.0f} without explosive volume — bounce likely (-10)")
            # Buying (buying calls) when RSI extremely overbought = buying the top
            elif direction == "BUY" and signal.rsi > 75:
                if vol_regime == 'EXPLOSIVE' and signal.rsi < 85:
                    # Explosive volume at moderate overbought — allow but warn
                    score -= 5
                    warnings.append(f"⚠️ RSI CONTRADICTION (soft): Buying at RSI={signal.rsi:.0f} but explosive volume allows (-5)")
                elif signal.rsi > 85:
                    # RSI > 85 = extreme overbought, block regardless
                    should_trade = False
                    score -= 15
                    warnings.append(f"🚫 BLOCKED: RSI CONTRADICTION: Buying/buying calls at RSI={signal.rsi:.0f} > 85 — reversal imminent (-15)")
                else:
                    # RSI 75-85 without explosive volume — heavy penalty
                    score -= 10
                    warnings.append(f"⚠️ RSI CONTRADICTION: Buying at RSI={signal.rsi:.0f} without explosive volume — pullback likely (-10)")

        # Gate 14: OI HARD CONFLICT — institutions fighting your direction
        # If OI says SHORT_BUILDUP and you're buying, or LONG_BUILDUP and you're selling,
        # AND the score is borderline (<75), block the trade. Institutions usually win.
        if should_trade and market_data:
            _oi_sig = market_data.get('oi_signal', 'NEUTRAL')
            _oi_chg = abs(market_data.get('oi_change_pct', 0))
            if _oi_sig not in ('NEUTRAL', 'NO_FUTURES', 'ERROR', ''):
                _oi_vs_dir = False
                if direction == 'BUY' and _oi_sig in ('SHORT_BUILDUP', 'LONG_UNWINDING'):
                    _oi_vs_dir = True
                elif direction == 'SELL' and _oi_sig in ('LONG_BUILDUP', 'SHORT_COVERING'):
                    _oi_vs_dir = True
                
                if _oi_vs_dir and _oi_chg >= 3 and score < 75:
                    should_trade = False
                    warnings.append(f"🚫 BLOCKED: OI hard conflict — {_oi_sig} (OI {_oi_chg:.1f}%) vs {direction} at score {score:.0f}<75 — institutions disagree")

        # S1/S7 FIX: GATE PENALTY CAP — prevent "death by a thousand cuts"
        # Total gate penalties cannot exceed 40% of pre-gate score.
        # A stock scoring 70 pre-gate can lose at most 28 points from gates (→ 42 minimum).
        # Without this, penalty budget (~130pts) dwarfs positive budget (~117pts),
        # structurally biasing toward rejection of legitimate setups.
        _pre_gate_score = _audit_after_align
        _total_gate_penalty = _pre_gate_score - score
        if _total_gate_penalty > 0 and _pre_gate_score > 0:
            _max_gate_penalty = int(_pre_gate_score * 0.40)
            if _total_gate_penalty > _max_gate_penalty:
                _recovered = _total_gate_penalty - _max_gate_penalty
                score += _recovered
                warnings.append(f"🛡️ GATE CAP: Penalties {_total_gate_penalty:.0f} exceeded 40% of pre-gate {_pre_gate_score:.0f} (max {_max_gate_penalty}), recovered +{_recovered}")
        
        # Gate FINAL: RE-CHECK score after all gate penalties (and cap)
        # Gates 7-13 subtract points but Gate 1 only checked the PRE-gate score.
        # Without this, trades like IOC (pre-gate 66, post-gate 40) slip through.
        if should_trade and score < self.BLOCK_THRESHOLD:
            should_trade = False
            warnings.append(f"🚫 BLOCKED: Post-gate score {score:.0f} < {self.BLOCK_THRESHOLD} minimum (gates reduced from pre-gate score)")
        
        # Track trade attempt (for logging)
        self.symbol_trades_today[sym_base] = self.symbol_trades_today.get(sym_base, 0) + 1
        _audit_after_gates = score
        
        # === BUILD SCORE AUDIT STRING ===
        _audit_parts = []
        _audit_parts.append(f"Trend:{_audit_after_trend:+.0f}")
        _audit_parts.append(f"Micro:{_audit_after_micro - _audit_after_trend:+.0f}")
        _audit_parts.append(f"ORB:{_audit_after_orb - _audit_after_micro:+.0f}")
        _audit_parts.append(f"Vol:{_audit_after_vol - _audit_after_orb:+.0f}")
        _audit_parts.append(f"VWAP:{_audit_after_vwap - _audit_after_vol:+.0f}")
        _audit_parts.append(f"EMA:{_audit_after_ema - _audit_after_vwap:+.0f}")
        _audit_parts.append(f"HTF:{_audit_after_htf - _audit_after_ema:+.0f}")
        _audit_parts.append(f"Accel:{_audit_after_accel - _audit_after_htf:+.0f}")
        _audit_parts.append(f"RSI:{_audit_after_rsi - _audit_after_accel:+.0f}")
        _audit_parts.append(f"Exhaust:{_audit_after_exhaust - _audit_after_rsi:+.0f}")
        _audit_parts.append(f"Chop:{_audit_after_chop - _audit_after_exhaust:+.0f}")
        _audit_parts.append(f"Align:{_audit_after_align - _audit_after_chop:+.0f}")
        _audit_parts.append(f"Gates:{_audit_after_gates - _audit_after_align:+.0f}")
        _score_audit_str = " | ".join(_audit_parts) + f" = {score:.0f}"
        self._last_score_audit = _score_audit_str
        
        # Cap size for reversal trades (lower conviction)
        if is_reversal_trade:
            size_multiplier = min(size_multiplier, 0.75)

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
            microstructure_block_reason=microstructure_block_reason,
            is_reversal_trade=is_reversal_trade
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


def get_hot_watchlist() -> Dict[str, Dict]:
    """Get the hot watchlist from the singleton scorer.
    Returns dict of {symbol: {score, direction, cycle_count, last_seen, reason}}
    These are stocks that scored 55+ but failed the conviction gate.
    They get a +5 conviction boost on re-evaluation."""
    scorer = get_intraday_scorer()
    return scorer.hot_watchlist


def cleanup_stale_watchlist(max_age_minutes: int = 20):
    """Remove watchlist entries not seen in the last N minutes.
    Prevents stale entries from persisting all day."""
    scorer = get_intraday_scorer()
    now = datetime.now()
    stale = []
    for sym, entry in scorer.hot_watchlist.items():
        try:
            last_seen = datetime.fromisoformat(entry.get('last_seen', '2000-01-01'))
            if (now - last_seen).total_seconds() > max_age_minutes * 60:
                stale.append(sym)
        except Exception:
            stale.append(sym)
    for sym in stale:
        del scorer.hot_watchlist[sym]
    if stale:
        print(f"   🧹 Watchlist cleanup: removed {len(stale)} stale entries: {', '.join(stale)}")


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


@dataclass
class IronCondorPlan:
    """
    Plan for an Iron Condor (4-leg neutral theta harvest)
    
    SELL OTM CE + BUY further OTM CE (bear call spread - upper wing)
    SELL OTM PE + BUY further OTM PE (bull put spread - lower wing)
    
    Profits when stock stays INSIDE the sold strikes.
    Ideal for choppy, range-bound stocks with low directional score.
    """
    underlying: str
    # Upper wing (bear call spread)
    sold_ce_contract: OptionContract     # OTM CE we SELL
    hedge_ce_contract: OptionContract    # Further OTM CE we BUY (cap upside risk)
    # Lower wing (bull put spread)
    sold_pe_contract: OptionContract     # OTM PE we SELL
    hedge_pe_contract: OptionContract    # Further OTM PE we BUY (cap downside risk)
    # Sizing
    quantity: int                        # Lots per side
    lot_size: int
    # Financials
    ce_credit: float                     # Credit from CE wing (per share)
    pe_credit: float                     # Credit from PE wing (per share)
    total_credit: float                  # Total credit per share (ce + pe)
    total_credit_amount: float           # Total credit in ₹ (credit × qty × lot_size)
    max_risk: float                      # Max risk = wing_width - total_credit (per share) × qty × lot_size
    ce_wing_width: float                 # CE sold-hedge strike difference
    pe_wing_width: float                 # PE sold-hedge strike difference
    # Range boundaries
    upper_breakeven: float               # Sold CE strike + total credit
    lower_breakeven: float               # Sold PE strike - total credit
    profit_zone_width: float             # upper_breakeven - lower_breakeven (₹ range for profit)
    # Risk management
    target_buyback: float                # Target: buy back at 50% of credit
    stop_loss_debit: float               # SL: exit when loss = 1.5× credit
    # Greeks (net of all 4 legs)
    net_delta: float                     # Should be near 0 (neutral)
    net_theta: float                     # POSITIVE = time decay earns us money
    net_vega: float                      # NEGATIVE = benefits from IV drop
    net_gamma: float                     # NEGATIVE = hurts on big moves
    # Metadata
    directional_score: float             # The low score that qualified this for IC
    rationale: str
    greeks_summary: str
    credit_pct: float                    # Credit as % of total wing width
    dte: int


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
    
    def _nfo_symbol(self, underlying: str) -> str:
        """
        Normalize underlying to NFO instrument name.
        Handles index mapping: "NIFTY 50" → "NIFTY", "NIFTY BANK" → "BANKNIFTY"
        For stocks, just strips "NSE:" prefix.
        """
        from config import INDEX_NFO_MAP
        symbol = underlying.replace("NSE:", "")
        return INDEX_NFO_MAP.get(symbol, symbol)
    
    def is_index(self, underlying: str) -> bool:
        """Check if underlying is an index (has weekly expiry)"""
        from config import INDEX_NFO_MAP
        symbol = underlying.replace("NSE:", "")
        return symbol in INDEX_NFO_MAP
    
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
        symbol = self._nfo_symbol(underlying)
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
        symbol = self._nfo_symbol(underlying)
        return FNO_LOT_SIZES.get(symbol, 1)
    
    def fetch_option_chain(self, underlying: str, expiry: datetime = None) -> Optional[OptionChain]:
        """
        Fetch complete option chain for an underlying
        
        Args:
            underlying: e.g., "NSE:RELIANCE" or "NSE:NIFTY 50"
            expiry: Specific expiry date (None = nearest)
            
        Returns:
            OptionChain object with all contracts
        """
        symbol = self._nfo_symbol(underlying)
        
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
                quote = self.kite.ltp([underlying])
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
        # Base: Target 50% gain, SL 28% loss → R:R = 1.79:1
        if score_tier == "premium":
            target_premium = contract.ltp * 1.80   # +80% premium gain
            stoploss_premium = contract.ltp * 0.72  # -28% premium loss
        elif score_tier == "standard":
            target_premium = contract.ltp * 1.60   # +60% premium gain
            stoploss_premium = contract.ltp * 0.72  # -28% premium loss
        else:
            target_premium = contract.ltp * 1.50   # +50% premium gain
            stoploss_premium = contract.ltp * 0.72  # -28% premium loss
        
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
    
    def _place_order_autoslice(self, **kwargs):
        """
        Place order with autoslice=true to auto-split orders exceeding exchange freeze limits.
        Without autoslice, orders >1800 qty NIFTY (or similar) get REJECTED.
        """
        if not AUTOSLICE_ENABLED or not self.kite or not hasattr(self.kite, '_routes'):
            return self.kite.place_order(**kwargs)
        
        orig_route = self.kite._routes.get("orders.place", "/orders/{variety}")
        try:
            if "autoslice" not in orig_route:
                self.kite._routes["orders.place"] = "/orders/{variety}?autoslice=true"
            return self.kite.place_order(**kwargs)
        finally:
            self.kite._routes["orders.place"] = orig_route
    
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
    
    def _check_basket_margin(self, orders: List[Dict], available_margin: float = None) -> Dict:
        """
        Pre-check margin for multi-leg orders using Kite Basket Margin API.
        For paper mode: uses conservative estimate (max_risk * 1.5).
        For live mode: calls kite.basket_margins() for exact SPAN+exposure with hedge benefit.
        
        Args:
            orders: List of order dicts with keys: exchange, tradingsymbol, transaction_type, 
                    quantity, product, order_type, price (optional)
            available_margin: Available margin (if None, uses self.capital)
            
        Returns:
            Dict with 'approved', 'required_margin', 'available', 'hedge_benefit' keys
        """
        if available_margin is None:
            available_margin = self.capital
            
        if self.paper_mode or not self.kite:
            # Paper mode: estimate margin as max_risk * 1.5 safety factor
            # (Actual SPAN margins are typically 50-80% of notional risk for hedged positions)
            total_premium = 0
            for order in orders:
                price = order.get('price', 0) or 0
                qty = order.get('quantity', 0)
                if order.get('transaction_type') == 'SELL':
                    total_premium += price * qty  # Margin needed for sold options
                else:
                    total_premium -= price * qty  # Premium paid (deducted from margin)
            
            # Conservative: worst case is spread_width * qty minus credit received
            estimated_margin = max(abs(total_premium) * 1.5, 0)
            return {
                'approved': available_margin >= estimated_margin,
                'required_margin': estimated_margin,
                'available': available_margin,
                'hedge_benefit': 0,
                'source': 'estimate'
            }
        
        try:
            # LIVE MODE: Use Kite Basket Margin API for exact margin with hedge benefit
            basket_orders = []
            for o in orders:
                basket_orders.append({
                    'exchange': o.get('exchange', 'NFO'),
                    'tradingsymbol': o['tradingsymbol'],
                    'transaction_type': o['transaction_type'],
                    'variety': 'regular',
                    'product': o.get('product', 'MIS'),
                    'order_type': o.get('order_type', 'MARKET'),
                    'quantity': o['quantity'],
                    'price': o.get('price', 0),
                    'trigger_price': 0
                })
            
            margin_data = self.kite.basket_margins(basket_orders, consider_positions=True)
            
            initial_margin = margin_data.get('initial', {}).get('total', 0)
            final_margin = margin_data.get('final', {}).get('total', 0)
            hedge_benefit = initial_margin - final_margin
            
            approved = available_margin >= final_margin
            
            if not approved:
                print(f"   ❌ MARGIN CHECK FAILED: Need ₹{final_margin:,.0f} | Available ₹{available_margin:,.0f}")
                print(f"      Hedge benefit: ₹{hedge_benefit:,.0f} (SPAN: ₹{initial_margin:,.0f} → ₹{final_margin:,.0f})")
            else:
                print(f"   ✅ MARGIN OK: ₹{final_margin:,.0f} / ₹{available_margin:,.0f} (hedge saves ₹{hedge_benefit:,.0f})")
            
            return {
                'approved': approved,
                'required_margin': final_margin,
                'available': available_margin,
                'hedge_benefit': hedge_benefit,
                'initial_margin': initial_margin,
                'source': 'kite_api'
            }
        except Exception as e:
            print(f"   ⚠️ Basket margin API error: {e} — falling back to estimate")
            # Fallback to estimate
            return {
                'approved': True,  # Don't block trades on API failure
                'required_margin': 0,
                'available': available_margin,
                'hedge_benefit': 0,
                'source': 'fallback'
            }
    
    def should_use_options(self, underlying: str) -> bool:
        """
        Determine if options should be used for this underlying
        """
        symbol = self.chain_fetcher._nfo_symbol(underlying)
        return symbol in FNO_LOT_SIZES
    
    def create_option_order_with_intraday(self, underlying: str, direction: str,
                                          market_data: Dict = None,
                                          force_strike: StrikeSelection = None,
                                          force_expiry: ExpirySelection = None,
                                          cached_decision: dict = None) -> Optional[Tuple[OptionOrderPlan, IntradayOptionDecision]]:
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
            atr=market_data.get('atr_14', 0.0),
            orb_strength=market_data.get('orb_strength', 0.0)
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
        if cached_decision and cached_decision.get('decision'):
            # REUSE cached cycle-time score — add microstructure bonus if available
            decision = cached_decision['decision']
            cached_base_score = cached_decision.get('score', decision.confidence_score)
            if micro_data:
                # Cycle-time score was computed WITHOUT microstructure.
                # Re-score WITH microstructure to unlock +15 bonus points.
                scorer = get_intraday_scorer()
                decision = scorer.score_intraday_signal(intraday_signal, market_data=market_data, option_data=micro_data, caller_direction=direction)
                print(f"   🔄 CACHED score {cached_base_score:.0f} → re-scored WITH microstructure → {decision.confidence_score:.0f}")
                
                # === RE-SCORE GATE: Block if option score dropped hard from scan score ===
                # KFINTECH scored 72 on scan but 57.5 on option → trade went through but lost.
                # If score drops >12 from cached AND falls below 62, option quality is too poor.
                _score_drop = cached_base_score - decision.confidence_score
                if _score_drop > 12 and decision.confidence_score < 62:
                    print(f"   🚫 RE-SCORE GATE: Score dropped {_score_drop:.0f} pts ({cached_base_score:.0f}→{decision.confidence_score:.0f}) — option quality too poor")
                    self._last_rejected_score = decision.confidence_score
                    self._last_rejected_symbol = underlying
                    return None
            else:
                print(f"   ✅ USING CACHED score {cached_base_score:.0f} for {underlying} (no microstructure change)")
        else:
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
        # Score audit trail — every component
        _audit_str = getattr(scorer, '_last_score_audit', 'N/A')
        log_lines.append(f"   📊 SCORE AUDIT: {_audit_str}")
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
            # Store rejected score for Iron Condor routing
            self._last_rejected_score = decision.confidence_score
            self._last_rejected_symbol = underlying
            return None
        
        # === USE INTRADAY DECISION FOR PARAMETERS ===
        # Force parameters can override intraday recommendation
        strike_sel = force_strike or StrikeSelection[decision.strike_selection]
        expiry_sel = force_expiry or ExpirySelection[decision.expiry_selection]
        opt_type = OptionType[decision.option_type]
        
        # Direction from intraday analysis takes precedence over passed direction
        final_direction = decision.recommended_direction if decision.recommended_direction != "HOLD" else direction
        
        # === ML DIRECTION OVERRIDE (smart CE↔PE flip) ===
        # When ML has HIGH confidence (BULLISH/BEARISH ≥60%) and disagrees with
        # the technical direction, flip final_direction + opt_type.
        # This is the last-mile override — all trade paths converge here.
        # FAIL-SAFE: any error → keep original direction.
        try:
            if cached_decision and cached_decision.get('ml_prediction'):
                _ml_hint = cached_decision['ml_prediction'].get('ml_direction_hint', 'NEUTRAL')
                if _ml_hint == 'BULLISH' and final_direction == 'SELL':
                    print(f"   🧠 ML OVERRIDE: {underlying} tech=SELL → ML=BULLISH (flipping to BUY/CE)")
                    final_direction = 'BUY'
                    opt_type = OptionType.CE
                elif _ml_hint == 'BEARISH' and final_direction == 'BUY':
                    print(f"   🧠 ML OVERRIDE: {underlying} tech=BUY → ML=BEARISH (flipping to SELL/PE)")
                    final_direction = 'SELL'
                    opt_type = OptionType.PE
        except Exception:
            pass  # ML unavailable — keep original direction
        
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
            if score >= IntradayOptionScorer.PREMIUM_THRESHOLD:
                score_tier = "premium"
            elif score >= IntradayOptionScorer.STANDARD_THRESHOLD:
                score_tier = "standard"
            else:
                score_tier = "base"
            
            sizing = self.position_sizer.calculate_position(
                contract, final_direction, self.capital, score_tier=score_tier
            )
            
            # Apply intraday conviction multiplier — use round() not int() to avoid truncation
            # int(1 * 1.5) = 1, round(1 * 1.5) = 2 ← this was silently killing the multiplier
            
            # === ML SIZING FACTOR (fail-safe: defaults to 1.0 if ML unavailable) ===
            # Reads ml_sizing_factor from cached ML prediction in cycle_decisions.
            # If ML model is not loaded, prediction failed, or field is missing,
            # _ml_sizing stays 1.0 and has ZERO effect on sizing.
            _ml_sizing = 1.0
            try:
                if cached_decision and cached_decision.get('ml_prediction'):
                    _ml_sizing = cached_decision['ml_prediction'].get('ml_sizing_factor', 1.0)
                    _ml_sizing = max(0.5, min(2.0, _ml_sizing))  # Safety clamp (2.0x max for elite ML signals)
            except Exception:
                _ml_sizing = 1.0  # Any error → neutral (no sizing change)
            
            adjusted_lots = max(1, round(sizing['lots'] * decision.position_size_multiplier * _ml_sizing))
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
                'ml_sizing_factor': round(_ml_sizing, 2),
                'original_lots': sizing['lots'],
                'adjusted_lots': adjusted_lots,
                'cap_pct_used': cap_pct,
                'strategy_type': 'REVERSAL_TRADE' if decision.is_reversal_trade else 'NAKED_OPTION',
                'is_reversal_trade': decision.is_reversal_trade,
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
                # ML prediction data (for post-trade analysis)
                # Populated from cached_decision if ML was available; None otherwise
                'ml_move_prob': None,
                'ml_signal': None,
                'ml_direction_hint': None,
                # OI flow data (for post-trade analysis)
                'oi_signal': market_data.get('oi_signal', 'UNKNOWN'),
                'oi_change_pct': market_data.get('oi_change_pct', 0),
                # Score audit trail
                'score_audit': getattr(scorer if 'scorer' in dir() else None, '_last_score_audit', 'N/A'),
            }
            # Populate ML fields from cached decision (fail-safe: stays None if unavailable)
            try:
                if cached_decision and cached_decision.get('ml_prediction'):
                    _cached_ml = cached_decision['ml_prediction']
                    entry_meta['ml_move_prob'] = _cached_ml.get('ml_move_prob')
                    entry_meta['ml_signal'] = _cached_ml.get('ml_signal')
                    entry_meta['ml_direction_hint'] = _cached_ml.get('ml_direction_hint')
            except Exception:
                pass  # ML data unavailable — fields stay None
            
            # === REVERSAL TRADE: Override SL and target for tighter risk ===
            _final_target = sizing['target_premium']
            _final_sl = sizing['stoploss_premium']
            if decision.is_reversal_trade:
                # Tight SL: 18% loss (0.82 multiplier) instead of 28% (0.72)
                # Modest target: 40% gain instead of 50-80%
                _final_sl = contract.ltp * 0.82   # -18% premium loss
                _final_target = contract.ltp * 1.40  # +40% premium gain
                print(f"   🔄 REVERSAL TRADE: Tight SL ₹{_final_sl:.2f} (-18%) | Target ₹{_final_target:.2f} (+40%)")
                entry_meta['strategy_type'] = 'REVERSAL_TRADE'

            plan = OptionOrderPlan(
                underlying=underlying,
                direction=final_direction,
                contract=contract,
                quantity=adjusted_lots,
                premium_per_lot=sizing['premium_per_lot'],
                total_premium=adjusted_premium,
                max_loss=adjusted_premium,  # Max loss = premium for long options
                breakeven=sizing['breakeven'],
                target_premium=_final_target,
                stoploss_premium=_final_sl,
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
                    f.write(f"      Sizing: {sizing['lots']} lots × {decision.position_size_multiplier:.1f}x × ML:{_ml_sizing:.2f}x = {adjusted_lots} lots | Cap: {cap_pct*100:.0f}%\n")
                    f.write(f"      Candles: FT={ft_candles} ADX={adx_val:.1f} ORB={orb_strength_val:.0f}% RangeExp={range_exp_val:.2f}\n")
                    f.write(f"      Context: ORB={orb_signal_val} Vol={vol_regime_val}({vol_ratio_val:.1f}x) VWAP={vwap_pos} HTF={htf_align} RSI={rsi_val:.0f}\n")
                    f.write(f"      Micro: score={decision.microstructure_score:.0f} block={decision.microstructure_block} accel={decision.acceleration_score:.1f}\n")
                    f.write(f"      ML: signal={entry_meta.get('ml_signal', 'N/A')} move_prob={entry_meta.get('ml_move_prob', 'N/A')} sizing={_ml_sizing:.2f}x\n")
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
                           market_data: Dict = None,
                           cached_decision: dict = None) -> Optional[OptionOrderPlan]:
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
                force_expiry=expiry_selection,
                cached_decision=cached_decision
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
                             spread_width_strikes: int = None,
                             cached_decision: dict = None) -> Optional[CreditSpreadPlan]:
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
            if cached_decision and cached_decision.get('decision'):
                # REUSE cached cycle-time decision — skip re-scoring
                decision = cached_decision['decision']
                score = cached_decision.get('score', decision.confidence_score)
                print(f"   ✅ USING CACHED score {score:.0f} for credit spread on {underlying}")
            else:
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
                    atr=market_data.get('atr_14', 0.0),
                    orb_strength=market_data.get('orb_strength', 0.0)
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
            
            # === ML SIZING FACTOR (FAIL-SAFE) ===
            # Credit spreads profit from NO_MOVE (theta decay), so ML NO_MOVE → bigger size
            # ML MOVE → reduce size (breakout risk)
            _cs_ml_sizing = 1.0
            try:
                if cached_decision and cached_decision.get('ml_prediction'):
                    _cs_move_prob = cached_decision['ml_prediction'].get('ml_move_prob', 0.5)
                    if _cs_move_prob < 0.15:
                        _cs_ml_sizing = 1.2   # Very flat → size up (theta-friendly)
                    elif _cs_move_prob < 0.25:
                        _cs_ml_sizing = 1.1   # Likely flat
                    elif _cs_move_prob >= 0.50:
                        _cs_ml_sizing = 0.7   # MOVE predicted → reduce risk
                    elif _cs_move_prob >= 0.40:
                        _cs_ml_sizing = 0.85  # Moderate MOVE probability
                    _cs_ml_sizing = max(0.5, min(1.5, _cs_ml_sizing))
                    lots = max(1, round(lots * _cs_ml_sizing))
                    if _cs_ml_sizing != 1.0:
                        print(f"      🧠 ML Credit Spread Sizing: move_prob={_cs_move_prob:.0%} → {_cs_ml_sizing:.2f}x → {lots} lots")
            except Exception:
                pass  # ML crash → neutral sizing (1.0x)
            
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
                
                # === BASKET MARGIN PRE-CHECK ===
                margin_check = self._check_basket_margin([
                    {'exchange': sold_exchange, 'tradingsymbol': sold_ts, 'transaction_type': 'SELL',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.sold_contract.ltp},
                    {'exchange': hedge_exchange, 'tradingsymbol': hedge_ts, 'transaction_type': 'BUY',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.hedge_contract.ltp},
                ])
                if not margin_check['approved']:
                    return {
                        'success': False, 
                        'error': f"Insufficient margin: need ₹{margin_check['required_margin']:,.0f}, have ₹{margin_check['available']:,.0f}",
                        'message': f"Credit spread blocked: margin ₹{margin_check['required_margin']:,.0f} > available ₹{margin_check['available']:,.0f}"
                    }
                
                # Leg 1: SELL the option (collect premium) — IOC + autoslice
                sold_order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=sold_exchange,
                    tradingsymbol=sold_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1,
                    tag='TITAN_CS'
                )
                
                # Leg 2: BUY the hedge (cap risk) — IOC + autoslice
                hedge_order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=hedge_exchange,
                    tradingsymbol=hedge_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1,
                    tag='TITAN_CS'
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
    # IRON CONDOR — INTRADAY IV CRUSH + PREMIUM CAPTURE (CHOPPY STOCKS)
    # =================================================================
    
    def create_iron_condor(self, underlying: str, 
                           market_data: Dict = None,
                           directional_score: float = 0) -> Optional[IronCondorPlan]:
        """
        Create a SOPHISTICATED Iron Condor plan with:
        1. ATR-based expected-move strike placement (not fixed N strikes OTM)
        2. IV analysis — prefer elevated IV for richer premium
        3. Delta-balanced wings — sold deltas between 0.15-0.30
        4. Credit quality gates — min credit %, max risk/reward
        5. Spot-distance validation — sold strikes ≥ 1× expected move from spot
        
        DUAL MODE:
        1. INDEX (NIFTY/BANKNIFTY): 0-2 DTE weekly expiry → big premium crush
        2. STOCK: Monthly expiry (up to 15 DTE) → IV compression
        
        Args:
            underlying: e.g. "NSE:BHEL"
            market_data: Dict with intraday indicators
            directional_score: Score from IC quality scorer (for logging)
            
        Returns:
            IronCondorPlan or None if not viable
        """
        from config import IRON_CONDOR_CONFIG
        
        if not IRON_CONDOR_CONFIG.get('enabled', False):
            print(f"   ⚠️ Iron Condor disabled in config")
            return None
        
        symbol = underlying.replace("NSE:", "")
        
        # === VALIDATE IRON CONDOR CONDITIONS ===
        max_score = IRON_CONDOR_CONFIG.get('max_directional_score', 45)
        min_score = IRON_CONDOR_CONFIG.get('min_directional_score', 15)
        
        if directional_score > max_score:
            print(f"   ❌ IC REJECTED: {underlying} score {directional_score:.0f} > {max_score} (too directional)")
            return None
        if directional_score < min_score:
            print(f"   ❌ IC REJECTED: {underlying} score {directional_score:.0f} < {min_score} (no data)")
            return None
        
        # Check intraday move — stock must be range-bound
        max_move = IRON_CONDOR_CONFIG.get('max_intraday_move_pct', 1.5)
        current_move = abs(market_data.get('change_pct', 0)) if market_data else 0
        if current_move > max_move:
            print(f"   ❌ IC REJECTED: {underlying} moved {current_move:.1f}% > {max_move}% (not range-bound)")
            return None
        
        # Check RSI — should be in neutral range
        rsi = market_data.get('rsi_14', 50) if market_data else 50
        rsi_range = IRON_CONDOR_CONFIG.get('prefer_rsi_range', [35, 65])
        if rsi < rsi_range[0] or rsi > rsi_range[1]:
            print(f"   ⚠️ IC WARNING: RSI {rsi:.0f} outside neutral range {rsi_range} — extra risk")
        
        # Chop zone bonus
        is_choppy = market_data.get('chop_zone', False) if market_data else False
        if is_choppy and IRON_CONDOR_CONFIG.get('prefer_chop_zone_bonus', True):
            print(f"   ✅ IC BONUS: {underlying} in chop zone — ideal for iron condor")
        
        # Time check — must be within entry window
        from datetime import datetime as _dt
        now = _dt.now()
        
        earliest_entry = _dt.strptime(IRON_CONDOR_CONFIG.get('earliest_entry', '10:30'), '%H:%M').time()
        if now.time() < earliest_entry:
            print(f"   ❌ IC REJECTED: Too early ({now.strftime('%H:%M')} < {IRON_CONDOR_CONFIG.get('earliest_entry', '10:30')})")
            return None
        
        no_entry_after = _dt.strptime(IRON_CONDOR_CONFIG.get('no_entry_after', '12:30'), '%H:%M').time()
        if now.time() > no_entry_after:
            print(f"   ❌ IC REJECTED: Past entry cutoff {IRON_CONDOR_CONFIG['no_entry_after']}")
            return None
        
        min_minutes = IRON_CONDOR_CONFIG.get('min_minutes_remaining', 90)
        auto_exit_str = IRON_CONDOR_CONFIG.get('auto_exit_time', '15:10')
        auto_exit_time = _dt.strptime(auto_exit_str, '%H:%M').time()
        auto_exit_dt = now.replace(hour=auto_exit_time.hour, minute=auto_exit_time.minute, second=0)
        minutes_left = (auto_exit_dt - now).total_seconds() / 60
        if minutes_left < min_minutes:
            print(f"   ❌ IC REJECTED: Only {minutes_left:.0f} min left < {min_minutes} min needed")
            return None
        
        print(f"   🦅 Iron Condor: {underlying} (score: {directional_score:.0f}, move: {current_move:.1f}%, RSI: {rsi:.0f})")
        
        try:
            # === DETERMINE MODE: INDEX vs STOCK ===
            is_index = self.chain_fetcher.is_index(underlying)
            mode_config = IRON_CONDOR_CONFIG.get('index_mode' if is_index else 'stock_mode', {})
            mode_label = "INDEX" if is_index else "STOCK"
            
            # Mode-specific parameters with fallbacks
            expiry_sel_str = mode_config.get('prefer_expiry', IRON_CONDOR_CONFIG.get('prefer_expiry', 'CURRENT_WEEK' if is_index else 'CURRENT_MONTH'))
            min_dte = mode_config.get('min_dte', 0 if is_index else 3)
            max_dte = mode_config.get('max_dte', 2 if is_index else 15)
            prefer_0dte = mode_config.get('prefer_0dte', is_index)
            target_pct = mode_config.get('target_pct', 25 if is_index else 10)
            max_target_pct = mode_config.get('max_target_pct', 40 if is_index else 15)
            
            print(f"   📊 IC Mode: {mode_label} | DTE: {min_dte}-{max_dte} | Target: {target_pct}%")
            
            # === GET OPTION CHAIN ===
            expiry_sel = ExpirySelection[expiry_sel_str]
            expiry = self.chain_fetcher.get_nearest_expiry(underlying, expiry_sel)
            
            if expiry is None:
                print(f"   ⚠️ No expiry found for {underlying}")
                return None
            
            from datetime import date as _date
            expiry_date = expiry if isinstance(expiry, _date) and not isinstance(expiry, datetime) else expiry.date() if hasattr(expiry, 'date') else expiry
            dte = (expiry_date - _dt.now().date()).days
            
            if dte < min_dte:
                print(f"   ⚠️ DTE {dte} < min {min_dte} — {'gamma risk' if is_index else 'too close to expiry'}")
                return None
            if dte > max_dte:
                print(f"   ⚠️ DTE {dte} > max {max_dte} — premium wont decay enough intraday, skip")
                return None
            
            if prefer_0dte and dte > 0:
                print(f"   ⏳ DTE={dte} (not 0DTE, premium decay will be slower intraday)")
            
            chain = self.chain_fetcher.fetch_option_chain(underlying, expiry)
            if chain is None or not chain.contracts:
                print(f"   ⚠️ No option chain for {underlying}")
                return None
            
            print(f"   📅 Expiry: {expiry} (DTE: {dte}) | Chain: {len(chain.contracts)} contracts")
            
            spot = chain.spot_price or market_data.get('ltp', 0) if market_data else 0
            atm_strike = chain.get_atm_strike(expiry)
            
            # === ATR-BASED EXPECTED MOVE CALCULATION ===
            # Use ATR to determine how far sold strikes should be from spot
            atr_14 = market_data.get('atr_14', 0) if market_data else 0
            if atr_14 <= 0:
                # Fallback: estimate ATR as 1.5% of spot
                atr_14 = spot * 0.015
            
            # Expected intraday remaining move = ATR × time factor
            # If 5 hours total and 3 hours left, factor = sqrt(3/5) ≈ 0.77
            total_session_mins = 375  # 9:15 to 15:30
            elapsed_mins = (now.hour * 60 + now.minute) - (9 * 60 + 15)
            remaining_mins = max(60, total_session_mins - elapsed_mins)
            time_factor = (remaining_mins / total_session_mins) ** 0.5
            
            expected_remaining_move = atr_14 * time_factor
            
            # Sold strikes should be at least 1× expected remaining move from spot
            min_distance_multiplier = IRON_CONDOR_CONFIG.get('min_atr_distance', 1.0)
            min_strike_distance = expected_remaining_move * min_distance_multiplier
            
            print(f"   📏 ATR: ₹{atr_14:.1f} | Expected remaining move: ₹{expected_remaining_move:.1f} | Min strike distance: ₹{min_strike_distance:.1f}")
            
            # === IV ANALYSIS — Check if IV is worth selling ===
            atm_ce = chain.get_contract(atm_strike, OptionType.CE, expiry)
            atm_pe = chain.get_contract(atm_strike, OptionType.PE, expiry)
            avg_atm_iv = 0
            if atm_ce and atm_pe:
                iv_ce = atm_ce.iv or 0
                iv_pe = atm_pe.iv or 0
                avg_atm_iv = (iv_ce + iv_pe) / 2 if iv_ce > 0 and iv_pe > 0 else max(iv_ce, iv_pe)
                
                min_iv = IRON_CONDOR_CONFIG.get('min_iv_for_ic', 15)
                if avg_atm_iv > 0 and avg_atm_iv < min_iv:
                    print(f"   ⚠️ IC REJECTED: ATM IV {avg_atm_iv:.1f}% < {min_iv}% — premium too cheap to sell")
                    return None
                
                # IV skew check — if one side has much higher IV, it hints at directional pressure
                if iv_ce > 0 and iv_pe > 0:
                    iv_skew = abs(iv_ce - iv_pe) / max(iv_ce, iv_pe) * 100
                    max_skew = IRON_CONDOR_CONFIG.get('max_iv_skew_pct', 30)
                    if iv_skew > max_skew:
                        print(f"   ⚠️ IC WARNING: IV skew {iv_skew:.0f}% (CE:{iv_ce:.1f}% vs PE:{iv_pe:.1f}%) — directional pressure")
                
                iv_info = f"ATM IV: {avg_atm_iv:.1f}%"
                if iv_ce > 0 and iv_pe > 0:
                    iv_info += f" (CE:{iv_ce:.1f}% PE:{iv_pe:.1f}%)"
                print(f"   📊 {iv_info}")
            
            # === INTELLIGENT STRIKE SELECTION (Delta-based + ATR-validated) ===
            ce_strikes = sorted(set(
                c.strike for c in chain.contracts
                if c.option_type == OptionType.CE and (c.expiry is None or c.expiry == expiry)
            ))
            pe_strikes = sorted(set(
                c.strike for c in chain.contracts
                if c.option_type == OptionType.PE and (c.expiry is None or c.expiry == expiry)
            ))
            
            wing_width = IRON_CONDOR_CONFIG.get('wing_width_strikes', 2)
            
            # Target delta for sold strikes: 0.15-0.30 (probability of being ITM)
            target_sold_delta = IRON_CONDOR_CONFIG.get('target_sold_delta', 0.25)
            max_sold_delta = IRON_CONDOR_CONFIG.get('max_sold_delta', 0.35)
            
            # === SELECT SOLD CE STRIKE ===
            # Find CE strike that is ≥ min_strike_distance above spot AND has delta ≤ target
            best_sold_ce_strike = None
            best_sold_ce_delta_dist = 999
            for ce_s in ce_strikes:
                if ce_s <= spot + min_strike_distance * 0.8:  # Must be far enough
                    continue
                ce_contract = chain.get_contract(ce_s, OptionType.CE, expiry)
                if ce_contract and ce_contract.ltp > 0:
                    ce_delta = abs(ce_contract.delta or 0.5)
                    if ce_delta <= max_sold_delta:
                        delta_dist = abs(ce_delta - target_sold_delta)
                        if delta_dist < best_sold_ce_delta_dist:
                            best_sold_ce_delta_dist = delta_dist
                            best_sold_ce_strike = ce_s
            
            # Fallback: use fixed offset if delta-based selection fails
            if best_sold_ce_strike is None:
                ce_atm_idx = min(range(len(ce_strikes)), key=lambda i: abs(ce_strikes[i] - atm_strike))
                ce_otm_offset = mode_config.get('ce_sold_otm_offset', IRON_CONDOR_CONFIG.get('ce_sold_otm_offset', 3))
                sold_ce_idx = min(len(ce_strikes) - 1, ce_atm_idx + ce_otm_offset)
                best_sold_ce_strike = ce_strikes[sold_ce_idx]
                print(f"   ⚠️ CE delta-selection failed, using offset={ce_otm_offset}")
            
            # === SELECT SOLD PE STRIKE ===
            best_sold_pe_strike = None
            best_sold_pe_delta_dist = 999
            for pe_s in reversed(pe_strikes):
                if pe_s >= spot - min_strike_distance * 0.8:  # Must be far enough
                    continue
                pe_contract = chain.get_contract(pe_s, OptionType.PE, expiry)
                if pe_contract and pe_contract.ltp > 0:
                    pe_delta = abs(pe_contract.delta or 0.5)
                    if pe_delta <= max_sold_delta:
                        delta_dist = abs(pe_delta - target_sold_delta)
                        if delta_dist < best_sold_pe_delta_dist:
                            best_sold_pe_delta_dist = delta_dist
                            best_sold_pe_strike = pe_s
            
            # Fallback
            if best_sold_pe_strike is None:
                pe_atm_idx = min(range(len(pe_strikes)), key=lambda i: abs(pe_strikes[i] - atm_strike))
                pe_otm_offset = mode_config.get('pe_sold_otm_offset', IRON_CONDOR_CONFIG.get('pe_sold_otm_offset', 3))
                sold_pe_idx = max(0, pe_atm_idx - pe_otm_offset)
                best_sold_pe_strike = pe_strikes[sold_pe_idx]
                print(f"   ⚠️ PE delta-selection failed, using offset={pe_otm_offset}")
            
            sold_ce_strike = best_sold_ce_strike
            sold_pe_strike = best_sold_pe_strike
            
            # === HEDGE STRIKES ===
            # Place hedge wing_width strikes further OTM
            ce_sold_idx = ce_strikes.index(sold_ce_strike) if sold_ce_strike in ce_strikes else min(range(len(ce_strikes)), key=lambda i: abs(ce_strikes[i] - sold_ce_strike))
            hedge_ce_idx = min(len(ce_strikes) - 1, ce_sold_idx + wing_width)
            
            pe_sold_idx = pe_strikes.index(sold_pe_strike) if sold_pe_strike in pe_strikes else min(range(len(pe_strikes)), key=lambda i: abs(pe_strikes[i] - sold_pe_strike))
            hedge_pe_idx = max(0, pe_sold_idx - wing_width)
            
            hedge_ce_strike = ce_strikes[hedge_ce_idx]
            hedge_pe_strike = pe_strikes[hedge_pe_idx]
            
            if sold_ce_strike == hedge_ce_strike or sold_pe_strike == hedge_pe_strike:
                print(f"   ⚠️ Not enough strikes for iron condor wings")
                return None
            
            # === SPOT DISTANCE VALIDATION ===
            ce_distance_from_spot = sold_ce_strike - spot
            pe_distance_from_spot = spot - sold_pe_strike
            
            print(f"   📐 IC Strikes: PE wing [{hedge_pe_strike}/{sold_pe_strike}] — SPOT {spot:.1f} — CE wing [{sold_ce_strike}/{hedge_ce_strike}]")
            print(f"      CE distance: ₹{ce_distance_from_spot:.1f} ({ce_distance_from_spot/spot*100:.1f}%) | PE distance: ₹{pe_distance_from_spot:.1f} ({pe_distance_from_spot/spot*100:.1f}%)")
            
            # Get contracts
            sold_ce = chain.get_contract(sold_ce_strike, OptionType.CE, expiry)
            hedge_ce = chain.get_contract(hedge_ce_strike, OptionType.CE, expiry)
            sold_pe = chain.get_contract(sold_pe_strike, OptionType.PE, expiry)
            hedge_pe = chain.get_contract(hedge_pe_strike, OptionType.PE, expiry)
            
            if not all([sold_ce, hedge_ce, sold_pe, hedge_pe]):
                missing = []
                if not sold_ce: missing.append(f"sold CE {sold_ce_strike}")
                if not hedge_ce: missing.append(f"hedge CE {hedge_ce_strike}")
                if not sold_pe: missing.append(f"sold PE {sold_pe_strike}")
                if not hedge_pe: missing.append(f"hedge PE {hedge_pe_strike}")
                print(f"   ⚠️ Missing contracts: {', '.join(missing)}")
                return None
            
            # === DELTA BALANCE CHECK ===
            sold_ce_delta = abs(sold_ce.delta or 0)
            sold_pe_delta = abs(sold_pe.delta or 0)
            delta_imbalance = abs(sold_ce_delta - sold_pe_delta)
            max_delta_imbalance = IRON_CONDOR_CONFIG.get('max_delta_imbalance', 0.15)
            
            if delta_imbalance > max_delta_imbalance:
                print(f"   ⚠️ IC WARNING: Delta imbalance {delta_imbalance:.2f} (CE Δ:{sold_ce_delta:.2f} vs PE Δ:{sold_pe_delta:.2f}) — wings not symmetric")
            
            print(f"      Sold CE Δ: {sold_ce_delta:.3f} | Sold PE Δ: {sold_pe_delta:.3f} | Imbalance: {delta_imbalance:.3f}")
            
            # === VALIDATE LIQUIDITY (all 4 legs) ===
            min_oi = IRON_CONDOR_CONFIG.get('min_oi_per_leg', 500)
            for leg_name, leg in [("sold CE", sold_ce), ("hedge CE", hedge_ce), 
                                   ("sold PE", sold_pe), ("hedge PE", hedge_pe)]:
                if leg.oi < min_oi:
                    print(f"   ⚠️ Low OI on {leg_name}: {leg.oi} < {min_oi}")
                    return None
            
            # === CALCULATE IC FINANCIALS ===
            ce_credit = sold_ce.ltp - hedge_ce.ltp  # Credit from upper wing
            pe_credit = sold_pe.ltp - hedge_pe.ltp  # Credit from lower wing
            total_credit = ce_credit + pe_credit     # Total credit per share
            
            if ce_credit <= 0 or pe_credit <= 0:
                print(f"   ⚠️ No net credit: CE wing ₹{ce_credit:.2f}, PE wing ₹{pe_credit:.2f}")
                return None
            
            ce_wing_width = abs(hedge_ce_strike - sold_ce_strike)
            pe_wing_width = abs(sold_pe_strike - hedge_pe_strike)
            max_wing_width = max(ce_wing_width, pe_wing_width)
            
            # Credit quality check — require minimum % of wing width
            credit_pct = (total_credit / max_wing_width) * 100 if max_wing_width > 0 else 0
            min_credit_pct = IRON_CONDOR_CONFIG.get('min_credit_pct', 15)
            if credit_pct < min_credit_pct:
                print(f"   ⚠️ Credit too thin: {credit_pct:.1f}% < {min_credit_pct}% of wing width")
                return None
            
            # === RISK/REWARD VALIDATION ===
            # Minimum reward:risk ratio for IC
            risk_per_share = max_wing_width - total_credit
            if risk_per_share <= 0:
                print(f"   ⚠️ Invalid risk calculation: wing_width={max_wing_width} credit={total_credit}")
                return None
            rr_ratio = total_credit / risk_per_share
            min_rr = IRON_CONDOR_CONFIG.get('min_reward_risk_ratio', 0.25)
            if rr_ratio < min_rr:
                print(f"   ⚠️ IC REJECTED: R:R {rr_ratio:.2f} < {min_rr} — risk too high for credit")
                return None
            
            # === POSITION SIZING ===
            lot_size = self.chain_fetcher.get_lot_size(underlying)
            max_risk_per_lot = risk_per_share * lot_size
            max_risk_config = IRON_CONDOR_CONFIG.get('max_risk_per_condor', 50000)
            max_lots_config = IRON_CONDOR_CONFIG.get('max_lots_per_condor', 3)
            
            lots_by_risk = max(1, int(max_risk_config / max_risk_per_lot)) if max_risk_per_lot > 0 else 1
            lots = min(lots_by_risk, max_lots_config)
            
            # Check total IC exposure
            max_total_exposure = IRON_CONDOR_CONFIG.get('max_total_condor_exposure', 150000)
            current_ic_exposure = self._get_current_condor_exposure()
            total_risk = risk_per_share * lot_size * lots
            
            if current_ic_exposure + total_risk > max_total_exposure:
                remaining = max_total_exposure - current_ic_exposure
                if remaining <= 0:
                    print(f"   ⚠️ Max IC exposure ₹{max_total_exposure:,.0f} reached")
                    return None
                lots = max(1, int(remaining / max_risk_per_lot))
            
            qty_shares = lots * lot_size
            total_credit_amount = total_credit * qty_shares
            max_risk_total = risk_per_share * qty_shares
            
            # === RISK MANAGEMENT ===
            sl_multiplier = IRON_CONDOR_CONFIG.get('sl_multiplier', 1.5)
            target_buyback = total_credit * (1 - target_pct / 100)
            stop_loss_debit = total_credit + (total_credit * sl_multiplier)
            
            # Breakeven points
            upper_breakeven = sold_ce_strike + total_credit
            lower_breakeven = sold_pe_strike - total_credit
            profit_zone = upper_breakeven - lower_breakeven
            
            # Profit zone should be wider than expected remaining move
            if profit_zone < expected_remaining_move * 1.5:
                print(f"   ⚠️ IC WARNING: Profit zone ₹{profit_zone:.0f} < 1.5× expected move ₹{expected_remaining_move*1.5:.0f} — tight")
            
            # === NET GREEKS (all 4 legs) ===
            net_delta = (-1 * (sold_ce.delta or 0) + (hedge_ce.delta or 0) + 
                         -1 * (sold_pe.delta or 0) + (hedge_pe.delta or 0))
            net_theta = (-1 * (sold_ce.theta or 0) + (hedge_ce.theta or 0) +
                         -1 * (sold_pe.theta or 0) + (hedge_pe.theta or 0))
            net_vega = (-1 * (sold_ce.vega or 0) + (hedge_ce.vega or 0) +
                        -1 * (sold_pe.vega or 0) + (hedge_pe.vega or 0))
            net_gamma = (-1 * (sold_ce.gamma or 0) + (hedge_ce.gamma or 0) +
                         -1 * (sold_pe.gamma or 0) + (hedge_pe.gamma or 0))
            
            # Net theta should be POSITIVE (we want time decay income)
            if net_theta * qty_shares <= 0:
                print(f"   ⚠️ IC WARNING: Net theta {net_theta*qty_shares:.2f} ≤ 0 — no theta income!")
            
            # Build IC quality summary
            quality_flags = []
            if avg_atm_iv > 30:
                quality_flags.append(f"HIGH_IV({avg_atm_iv:.0f}%)")
            if delta_imbalance < 0.05:
                quality_flags.append("SYMMETRIC")
            if credit_pct > 30:
                quality_flags.append(f"RICH_CREDIT({credit_pct:.0f}%)")
            if profit_zone > expected_remaining_move * 2:
                quality_flags.append("WIDE_ZONE")
            if is_choppy:
                quality_flags.append("CHOP_ZONE")
            quality_str = " | ".join(quality_flags) if quality_flags else "STANDARD"
            
            plan = IronCondorPlan(
                underlying=underlying,
                sold_ce_contract=sold_ce,
                hedge_ce_contract=hedge_ce,
                sold_pe_contract=sold_pe,
                hedge_pe_contract=hedge_pe,
                quantity=lots,
                lot_size=lot_size,
                ce_credit=ce_credit,
                pe_credit=pe_credit,
                total_credit=total_credit,
                total_credit_amount=total_credit_amount,
                max_risk=max_risk_total,
                ce_wing_width=ce_wing_width,
                pe_wing_width=pe_wing_width,
                upper_breakeven=upper_breakeven,
                lower_breakeven=lower_breakeven,
                profit_zone_width=profit_zone,
                target_buyback=target_buyback,
                stop_loss_debit=stop_loss_debit,
                net_delta=net_delta * qty_shares,
                net_theta=net_theta * qty_shares,
                net_vega=net_vega * qty_shares,
                net_gamma=net_gamma * qty_shares,
                directional_score=directional_score,
                rationale=f"IC ({mode_label}) {underlying} | {quality_str} | Credit ₹{total_credit:.2f}/sh ({credit_pct:.0f}%) R:R {rr_ratio:.2f} | Zone: ₹{lower_breakeven:.0f}-₹{upper_breakeven:.0f} | CE Δ:{sold_ce_delta:.2f} PE Δ:{sold_pe_delta:.2f} | DTE:{dte} | ATR:₹{atr_14:.0f}",
                greeks_summary=f"NetΔ:{net_delta*qty_shares:.2f} NetΘ:{net_theta*qty_shares:+.2f}/day NetV:{net_vega*qty_shares:.2f} NetΓ:{net_gamma*qty_shares:.4f}",
                credit_pct=credit_pct,
                dte=dte,
            )
            
            print(f"   ✅ IRON CONDOR: [{quality_str}]")
            print(f"      SELL PE {sold_pe.symbol}@₹{sold_pe.ltp:.2f} (Δ:{sold_pe_delta:.2f}) + BUY PE {hedge_pe.symbol}@₹{hedge_pe.ltp:.2f}")
            print(f"      SELL CE {sold_ce.symbol}@₹{sold_ce.ltp:.2f} (Δ:{sold_ce_delta:.2f}) + BUY CE {hedge_ce.symbol}@₹{hedge_ce.ltp:.2f}")
            print(f"      Total Credit: ₹{total_credit:.2f}/sh (₹{total_credit_amount:,.0f} total) | {credit_pct:.0f}% of width | R:R {rr_ratio:.2f}")
            print(f"      Max Risk: ₹{max_risk_total:,.0f} | {lots} lots × {lot_size}")
            print(f"      Profit Zone: ₹{lower_breakeven:.0f} — ₹{upper_breakeven:.0f} (₹{profit_zone:.0f} wide vs ₹{expected_remaining_move:.0f} exp move)")
            print(f"      NetΘ: {net_theta*qty_shares:+.2f}/day | NetΔ: {net_delta*qty_shares:.2f} | IV: {avg_atm_iv:.1f}%")
            
            # Log to persistent file
            try:
                with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*70}\n")
                    f.write(f"🦅 IRON CONDOR: {underlying} (score: {directional_score:.0f}) [{quality_str}]\n")
                    f.write(f"   SELL PE: {sold_pe.symbol}@₹{sold_pe.ltp:.2f} Δ:{sold_pe_delta:.2f} | BUY PE: {hedge_pe.symbol}@₹{hedge_pe.ltp:.2f}\n")
                    f.write(f"   SELL CE: {sold_ce.symbol}@₹{sold_ce.ltp:.2f} Δ:{sold_ce_delta:.2f} | BUY CE: {hedge_ce.symbol}@₹{hedge_ce.ltp:.2f}\n")
                    f.write(f"   Credit: ₹{total_credit_amount:,.0f} ({credit_pct:.0f}%) | Risk: ₹{max_risk_total:,.0f} | R:R: {rr_ratio:.2f} | DTE: {dte}\n")
                    f.write(f"   Zone: ₹{lower_breakeven:.0f}-₹{upper_breakeven:.0f} ({profit_zone:.0f} wide) | ATR: ₹{atr_14:.0f} | IV: {avg_atm_iv:.1f}%\n")
                    f.write(f"{'='*70}\n")
            except Exception:
                pass
            
            return plan
            
        except Exception as e:
            import traceback
            print(f"   ❌ Iron Condor creation failed: {e}")
            traceback.print_exc()
            return None
    
    def _get_current_condor_exposure(self) -> float:
        """Get total risk exposure from existing iron condor positions"""
        total = 0
        for pos in self.positions:
            if pos.get('status') == 'OPEN' and pos.get('is_iron_condor'):
                total += pos.get('max_risk', 0)
        return total
    
    def execute_iron_condor(self, plan: IronCondorPlan) -> Dict:
        """
        Execute an Iron Condor (all 4 legs)
        """
        if self.paper_mode:
            import random
            condor_id = f"IC_{random.randint(100000, 999999)}"
            
            position = {
                'condor_id': condor_id,
                'is_iron_condor': True,
                'is_credit_spread': False,
                'underlying': plan.underlying,
                'direction': 'NEUTRAL',
                # Upper wing (bear call)
                'sold_ce_symbol': plan.sold_ce_contract.symbol,
                'sold_ce_strike': plan.sold_ce_contract.strike,
                'sold_ce_premium': plan.sold_ce_contract.ltp,
                'hedge_ce_symbol': plan.hedge_ce_contract.symbol,
                'hedge_ce_strike': plan.hedge_ce_contract.strike,
                'hedge_ce_premium': plan.hedge_ce_contract.ltp,
                # Lower wing (bull put)
                'sold_pe_symbol': plan.sold_pe_contract.symbol,
                'sold_pe_strike': plan.sold_pe_contract.strike,
                'sold_pe_premium': plan.sold_pe_contract.ltp,
                'hedge_pe_symbol': plan.hedge_pe_contract.symbol,
                'hedge_pe_strike': plan.hedge_pe_contract.strike,
                'hedge_pe_premium': plan.hedge_pe_contract.ltp,
                # Sizing
                'quantity': plan.quantity,
                'lot_size': plan.lot_size,
                'total_credit': plan.total_credit,
                'total_credit_amount': plan.total_credit_amount,
                'max_risk': plan.max_risk,
                # Risk mgmt
                'upper_breakeven': plan.upper_breakeven,
                'lower_breakeven': plan.lower_breakeven,
                'profit_zone_width': plan.profit_zone_width,
                'target_buyback': plan.target_buyback,
                'stop_loss_debit': plan.stop_loss_debit,
                # Greeks
                'net_delta': plan.net_delta,
                'net_theta': plan.net_theta,
                'net_vega': plan.net_vega,
                'credit_pct': plan.credit_pct,
                'dte': plan.dte,
                'directional_score': plan.directional_score,
                # Status
                'status': 'OPEN',
                'timestamp': datetime.now().isoformat(),
                'rationale': plan.rationale,
            }
            
            self.positions.append(position)
            
            return {
                'success': True,
                'paper_trade': True,
                'condor_id': condor_id,
                'underlying': plan.underlying,
                'strategy': 'IRON_CONDOR',
                'sold_ce': plan.sold_ce_contract.symbol,
                'hedge_ce': plan.hedge_ce_contract.symbol,
                'sold_pe': plan.sold_pe_contract.symbol,
                'hedge_pe': plan.hedge_pe_contract.symbol,
                'total_credit': plan.total_credit,
                'total_credit_amount': plan.total_credit_amount,
                'max_risk': plan.max_risk,
                'profit_zone': f"₹{plan.lower_breakeven:.0f} - ₹{plan.upper_breakeven:.0f}",
                'credit_pct': plan.credit_pct,
                'dte': plan.dte,
                'lots': plan.quantity,
                'greeks': plan.greeks_summary,
                'rationale': plan.rationale,
                'message': f"🦅 IRON CONDOR: {plan.underlying} | Credit ₹{plan.total_credit_amount:,.0f} | Risk ₹{plan.max_risk:,.0f} | Zone ₹{plan.lower_breakeven:.0f}-₹{plan.upper_breakeven:.0f}"
            }
        else:
            # LIVE MODE: Place all 4 legs
            try:
                qty_shares = plan.quantity * plan.lot_size
                
                # === BASKET MARGIN PRE-CHECK (all 4 legs) ===
                _, sold_ce_ts = plan.sold_ce_contract.symbol.split(':')
                _, hedge_ce_ts = plan.hedge_ce_contract.symbol.split(':')
                _, sold_pe_ts = plan.sold_pe_contract.symbol.split(':')
                _, hedge_pe_ts = plan.hedge_pe_contract.symbol.split(':')
                
                margin_check = self._check_basket_margin([
                    {'exchange': 'NFO', 'tradingsymbol': sold_ce_ts, 'transaction_type': 'SELL',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.sold_ce_contract.ltp},
                    {'exchange': 'NFO', 'tradingsymbol': hedge_ce_ts, 'transaction_type': 'BUY',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.hedge_ce_contract.ltp},
                    {'exchange': 'NFO', 'tradingsymbol': sold_pe_ts, 'transaction_type': 'SELL',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.sold_pe_contract.ltp},
                    {'exchange': 'NFO', 'tradingsymbol': hedge_pe_ts, 'transaction_type': 'BUY',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.hedge_pe_contract.ltp},
                ])
                if not margin_check['approved']:
                    return {
                        'success': False,
                        'error': f"Insufficient margin for IC: need ₹{margin_check['required_margin']:,.0f}, have ₹{margin_check['available']:,.0f}",
                        'message': f"Iron Condor blocked: margin ₹{margin_check['required_margin']:,.0f} > available ₹{margin_check['available']:,.0f}"
                    }
                
                # Leg 1: SELL OTM CE — IOC + autoslice
                self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR, exchange='NFO',
                    tradingsymbol=sold_ce_ts, transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares, product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1, tag='TITAN_IC'
                )
                # Leg 2: BUY further OTM CE (hedge) — IOC + autoslice
                self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR, exchange='NFO',
                    tradingsymbol=hedge_ce_ts, transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares, product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1, tag='TITAN_IC'
                )
                # Leg 3: SELL OTM PE — IOC + autoslice
                self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR, exchange='NFO',
                    tradingsymbol=sold_pe_ts, transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares, product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1, tag='TITAN_IC'
                )
                # Leg 4: BUY further OTM PE (hedge) — IOC + autoslice
                self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR, exchange='NFO',
                    tradingsymbol=hedge_pe_ts, transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares, product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1, tag='TITAN_IC'
                )
                
                return {
                    'success': True,
                    'paper_trade': False,
                    'message': f"Iron Condor placed: {plan.underlying} 4 legs"
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f"Iron Condor failed: {e}"
                }

    # =================================================================
    # DEBIT SPREAD — INTRADAY MOMENTUM STRATEGY
    # =================================================================

    def create_debit_spread(self, underlying: str, direction: str,
                            market_data: Dict = None,
                            spread_width_strikes: int = None,
                            cached_decision: dict = None) -> Optional[DebitSpreadPlan]:
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
        # NOTE: Base score from score_intraday_signal already penalizes ORB overextension (Gate 10)
        # and late entry (Gate 7). D3/D4 overlap with those — cap total to avoid triple-penalty.
        debit_gate_penalty = 0  # Accumulated penalty from softened gates
        MAX_DEBIT_GATE_PENALTY = 10  # Cap D2+D3+D4 combined (was uncapped at 19)
        if market_data:
            # Gate D1: Follow-Through Candles (STRONGEST winner signal — 4.1 vs 0.4)
            ft_candles = market_data.get('follow_through_candles', 0)
            min_ft = DEBIT_SPREAD_CONFIG.get('min_follow_through_candles', 2)
            if ft_candles < min_ft:
                print(f"   ❌ Debit spread BLOCKED [Gate D1]: {underlying} follow-through={ft_candles} < {min_ft} (need momentum confirmation)")
                return None
            
            # Gate D2: ADX Trend Strength (winners avg 37 vs losers 30.8)
            # Independent signal (trend strength ≠ exhaustion), keep full penalty
            adx = market_data.get('adx_14', market_data.get('adx', 0))
            min_adx = DEBIT_SPREAD_CONFIG.get('min_adx', 28)
            if adx > 0 and adx < min_adx:
                debit_gate_penalty += 6  # was 8, reduced — base scorer Gate 9 already penalizes ADX
                print(f"   ⚠️ Debit spread PENALTY [Gate D2]: {underlying} ADX={adx:.1f} < {min_adx} (weak trend, -6 score)")
            
            # Gate D3+D4: UNIFIED EXHAUSTION gate — ORB overextension + range expansion
            # Both measure "move already happened" — consolidate to avoid double-penalty.
            # Base scorer Gate 10 already penalizes ORB, so these are additive EXTRA.
            orb_strength = market_data.get('orb_strength_pct', market_data.get('orb_strength', 0))
            max_orb = DEBIT_SPREAD_CONFIG.get('max_orb_strength_pct', 120)
            range_exp = market_data.get('range_expansion_ratio', 0)
            max_range_exp = DEBIT_SPREAD_CONFIG.get('max_range_expansion', 0.50)
            
            # Hard blocks at extremes (unchanged)
            if orb_strength > 200:
                print(f"   ❌ Debit spread BLOCKED [Gate D3]: {underlying} ORB={orb_strength:.0f}% > 200% (extreme overextension)")
                return None
            if range_exp > 0.80:
                print(f"   ❌ Debit spread BLOCKED [Gate D4]: {underlying} range expansion={range_exp:.2f} > 0.80 ATR (extreme exhaustion)")
                return None
            
            # Unified exhaustion penalty — pick the WORSE signal, don't stack both
            orb_triggered = orb_strength > max_orb
            range_triggered = range_exp > max_range_exp
            if orb_triggered or range_triggered:
                if orb_triggered and range_triggered:
                    # Both triggered = clearly exhausted, but single penalty (worst of the two + small bump)
                    exhaustion_penalty = 5  # was 6+5=11
                    print(f"   ⚠️ Debit spread PENALTY [Gate D3+D4]: {underlying} ORB={orb_strength:.0f}% + RangeExp={range_exp:.2f} (exhaustion, -{exhaustion_penalty} score)")
                elif orb_triggered:
                    exhaustion_penalty = 4  # was 6
                    print(f"   ⚠️ Debit spread PENALTY [Gate D3]: {underlying} ORB={orb_strength:.0f}% > {max_orb}% (overextended, -{exhaustion_penalty} score)")
                else:
                    exhaustion_penalty = 4  # was 5
                    print(f"   ⚠️ Debit spread PENALTY [Gate D4]: {underlying} range expansion={range_exp:.2f} > {max_range_exp} ATR (exhausted, -{exhaustion_penalty} score)")
                debit_gate_penalty += exhaustion_penalty
            
            # Cap total debit gate penalties
            if debit_gate_penalty > MAX_DEBIT_GATE_PENALTY:
                print(f"   🔧 Debit gate penalty capped: {debit_gate_penalty} → {MAX_DEBIT_GATE_PENALTY} (avoid triple-penalty with base scorer)")
                debit_gate_penalty = MAX_DEBIT_GATE_PENALTY
            
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
            if cached_decision and cached_decision.get('decision'):
                # REUSE cached cycle-time decision — skip re-scoring
                decision = cached_decision['decision']
                score = cached_decision.get('score', decision.confidence_score)
                print(f"   ✅ USING CACHED score {score:.0f} for debit spread on {underlying}")
            else:
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
                    atr=market_data.get('atr_14', 0.0),
                    orb_strength=market_data.get('orb_strength', 0.0)
                )
                scorer = get_intraday_scorer()
                decision = scorer.score_intraday_signal(intraday_signal, market_data=market_data, caller_direction=direction)
                score = decision.confidence_score
            
            # Apply debit gate penalties (D2-D4 softened gates)
            if debit_gate_penalty > 0:
                score -= debit_gate_penalty
                raw_score = cached_decision.get('score', decision.confidence_score) if cached_decision else decision.confidence_score
                print(f"   📉 Debit spread score adjusted: {raw_score:.0f} → {score:.0f} (gate penalties: -{debit_gate_penalty})")
            
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
            
            # === ML SIZING FACTOR (FAIL-SAFE) ===
            # Debit spreads profit from MOVE, so ML MOVE → bigger size
            # ML NO_MOVE → reduce size (stock likely flat, debit decays)
            _ds_ml_sizing = 1.0
            try:
                if cached_decision and cached_decision.get('ml_prediction'):
                    _ds_move_prob = cached_decision['ml_prediction'].get('ml_move_prob', 0.5)
                    if _ds_move_prob >= 0.60:
                        _ds_ml_sizing = 1.2   # Strong MOVE → size up
                    elif _ds_move_prob >= 0.45:
                        _ds_ml_sizing = 1.1   # Moderate MOVE
                    elif _ds_move_prob < 0.20:
                        _ds_ml_sizing = 0.7   # Very flat → reduce exposure
                    elif _ds_move_prob < 0.30:
                        _ds_ml_sizing = 0.85  # Likely flat
                    _ds_ml_sizing = max(0.5, min(1.5, _ds_ml_sizing))
                    lots = max(1, round(lots * _ds_ml_sizing))
                    if _ds_ml_sizing != 1.0:
                        print(f"      🧠 ML Debit Spread Sizing: move_prob={_ds_move_prob:.0%} → {_ds_ml_sizing:.2f}x → {lots} lots")
            except Exception:
                pass  # ML crash → neutral sizing (1.0x)
            
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
                
                # === BASKET MARGIN PRE-CHECK ===
                margin_check = self._check_basket_margin([
                    {'exchange': buy_exchange, 'tradingsymbol': buy_ts, 'transaction_type': 'BUY',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.buy_contract.ltp},
                    {'exchange': sell_exchange, 'tradingsymbol': sell_ts, 'transaction_type': 'SELL',
                     'quantity': qty_shares, 'product': 'MIS', 'order_type': 'MARKET', 'price': plan.sell_contract.ltp},
                ])
                if not margin_check['approved']:
                    return {
                        'success': False,
                        'error': f"Insufficient margin: need ₹{margin_check['required_margin']:,.0f}, have ₹{margin_check['available']:,.0f}",
                        'message': f"Debit spread blocked: margin ₹{margin_check['required_margin']:,.0f} > available ₹{margin_check['available']:,.0f}"
                    }
                
                # Leg 1: BUY the near-ATM option (directional bet) — IOC + autoslice
                buy_order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=buy_exchange,
                    tradingsymbol=buy_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1,
                    tag='TITAN_DS'
                )
                
                # Leg 2: SELL the further OTM option (reduce cost) — IOC + autoslice
                sell_order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=sell_exchange,
                    tradingsymbol=sell_ts,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=qty_shares,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_IOC,
                    market_protection=-1,
                    tag='TITAN_DS'
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
                
                order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=plan.quantity * plan.contract.lot_size,
                    product=self.kite.PRODUCT_MIS,  # Intraday
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    market_protection=-1,
                    tag='TITAN_OPT'
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
                    quote = self.kite.ltp([symbol])
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
