"""
REGIME SCORE MODULE
Confidence gate for trade quality - reduces "almost good" trades

Scoring (0-100):
- VWAP alignment: +25
- EMA expansion: +20
- ORB breakout + high/exp vol: +30
- HTF alignment: +15
- Chop filter clean: +10
- Spread/slippage within best band: +10
- Penalties for conflicts

Trade Thresholds:
- ORB breakout: score >= 70
- VWAP trend trades: score >= 60
- Mean reversion: score >= 65
"""

from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TradeType(Enum):
    """Types of trades with different thresholds"""
    ORB_BREAKOUT = "ORB_BREAKOUT"
    VWAP_TREND = "VWAP_TREND"
    EMA_SQUEEZE = "EMA_SQUEEZE"
    MEAN_REVERSION = "MEAN_REVERSION"  # RSI oversold/overbought
    EOD_PLAY = "EOD_PLAY"
    MOMENTUM = "MOMENTUM"


@dataclass
class ScoreComponent:
    """Individual score component"""
    name: str
    points: int
    max_points: int
    reason: str
    is_penalty: bool = False


@dataclass
class RegimeScore:
    """Complete regime score with breakdown"""
    symbol: str
    direction: str  # BUY or SELL
    trade_type: str
    total_score: int
    threshold: int
    passes_threshold: bool
    components: List[ScoreComponent]
    final_verdict: str  # TRADE, SKIP, BORDERLINE
    confidence: str  # HIGH, MEDIUM, LOW
    summary: str


class RegimeScorer:
    """
    Computes regime score for trade quality assessment
    
    Higher score = higher confidence in trade
    Must exceed threshold to execute
    
    NOW INTEGRATES TREND FOLLOWING with enhanced hysteresis!
    """
    
    def __init__(self):
        # Score thresholds by trade type
        self.thresholds = {
            TradeType.ORB_BREAKOUT.value: 70,
            TradeType.VWAP_TREND.value: 60,
            TradeType.EMA_SQUEEZE.value: 65,
            TradeType.MEAN_REVERSION.value: 65,
            TradeType.EOD_PLAY.value: 55,
            TradeType.MOMENTUM.value: 60,
        }
        
        # Default threshold for unknown trade types
        self.default_threshold = 60
        
        # === TREND FOLLOWING INTEGRATION ===
        # Uses hysteresis-aware TrendFollowingEngine for intraday equity
        try:
            from trend_following import get_trend_engine, build_trend_signal_from_market_data, TrendState
            self.trend_engine = get_trend_engine()
            self._trend_available = True
            print("📈 RegimeScorer: TrendFollowing with hysteresis ENABLED")
        except ImportError:
            self._trend_available = False
            self.trend_engine = None
            print("⚠️ RegimeScorer: TrendFollowing not available")
    
    def get_threshold(self, trade_type: str) -> int:
        """Get score threshold for trade type"""
        return self.thresholds.get(trade_type, self.default_threshold)
    
    def calculate_score(
        self,
        symbol: str,
        direction: str,
        trade_type: str,
        market_data: Dict
    ) -> RegimeScore:
        """
        Calculate regime score for a potential trade
        
        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            trade_type: Type of trade (ORB_BREAKOUT, VWAP_TREND, etc.)
            market_data: Market data dict with indicators
        
        Returns:
            RegimeScore with full breakdown
        """
        components = []
        
        # === 0. TREND FOLLOWING (NEW - +20 bonus with hysteresis) ===
        # Trend Following is the "real edge" - uses enhanced hysteresis
        trend_score = self._score_trend_following(symbol, direction, market_data)
        if trend_score:
            components.append(trend_score)
        
        # === 1. VWAP ALIGNMENT (+25 max) ===
        vwap_score = self._score_vwap_alignment(direction, market_data)
        components.append(vwap_score)
        
        # === 2. EMA EXPANSION (+20 max) ===
        ema_score = self._score_ema_expansion(direction, market_data)
        components.append(ema_score)
        
        # === 3. ORB BREAKOUT + VOLUME (+30 max) ===
        orb_score = self._score_orb_breakout(direction, market_data)
        components.append(orb_score)
        
        # === 4. HTF ALIGNMENT (+15 max) ===
        htf_score = self._score_htf_alignment(direction, market_data)
        components.append(htf_score)
        
        # === 5. CHOP FILTER CLEAN (+10 max) ===
        chop_score = self._score_chop_filter(market_data)
        components.append(chop_score)
        
        # === 6. SPREAD/SLIPPAGE QUALITY (+10 max) ===
        spread_score = self._score_spread_quality(market_data)
        components.append(spread_score)
        
        # === 7. CONFLICT PENALTIES ===
        penalty_components = self._calculate_penalties(direction, market_data)
        components.extend(penalty_components)
        
        # Calculate total
        total_score = sum(c.points for c in components)
        total_score = max(0, min(100, total_score))  # Clamp to 0-100
        
        # Get threshold
        threshold = self.get_threshold(trade_type)
        passes_threshold = total_score >= threshold
        
        # Determine verdict and confidence
        if total_score >= threshold + 15:
            final_verdict = "TRADE"
            confidence = "HIGH"
        elif total_score >= threshold:
            final_verdict = "TRADE"
            confidence = "MEDIUM"
        elif total_score >= threshold - 10:
            final_verdict = "BORDERLINE"
            confidence = "LOW"
        else:
            final_verdict = "SKIP"
            confidence = "LOW"
        
        # Generate summary
        positives = [c for c in components if c.points > 0 and not c.is_penalty]
        negatives = [c for c in components if c.points < 0 or (c.points == 0 and c.max_points > 0)]
        
        summary_parts = []
        if positives:
            top_positives = sorted(positives, key=lambda x: x.points, reverse=True)[:3]
            summary_parts.append(f"Strengths: {', '.join(c.name for c in top_positives)}")
        if negatives:
            summary_parts.append(f"Concerns: {', '.join(c.name for c in negatives if c.points <= 0)}")
        
        summary = " | ".join(summary_parts) if summary_parts else "No significant factors"
        
        return RegimeScore(
            symbol=symbol,
            direction=direction,
            trade_type=trade_type,
            total_score=total_score,
            threshold=threshold,
            passes_threshold=passes_threshold,
            components=components,
            final_verdict=final_verdict,
            confidence=confidence,
            summary=summary
        )
    
    def _score_vwap_alignment(self, direction: str, data: Dict) -> ScoreComponent:
        """Score VWAP alignment (max +25)"""
        max_points = 25
        points = 0
        reason = ""
        
        price_vs_vwap = data.get('price_vs_vwap', 'AT_VWAP')
        vwap_slope = data.get('vwap_slope', 'FLAT')
        
        if direction == "BUY":
            if price_vs_vwap == "ABOVE_VWAP" and vwap_slope == "RISING":
                points = 25
                reason = "Perfect: Above rising VWAP"
            elif price_vs_vwap == "ABOVE_VWAP":
                points = 15
                reason = "Good: Above VWAP"
            elif price_vs_vwap == "AT_VWAP" and vwap_slope == "RISING":
                points = 10
                reason = "OK: At rising VWAP"
            elif price_vs_vwap == "BELOW_VWAP":
                points = 0
                reason = "Weak: Below VWAP for BUY"
        else:  # SELL
            if price_vs_vwap == "BELOW_VWAP" and vwap_slope == "FALLING":
                points = 25
                reason = "Perfect: Below falling VWAP"
            elif price_vs_vwap == "BELOW_VWAP":
                points = 15
                reason = "Good: Below VWAP"
            elif price_vs_vwap == "AT_VWAP" and vwap_slope == "FALLING":
                points = 10
                reason = "OK: At falling VWAP"
            elif price_vs_vwap == "ABOVE_VWAP":
                points = 0
                reason = "Weak: Above VWAP for SELL"
        
        return ScoreComponent(
            name="VWAP",
            points=points,
            max_points=max_points,
            reason=reason
        )
    
    def _score_ema_expansion(self, direction: str, data: Dict) -> ScoreComponent:
        """Score EMA expansion (max +20)"""
        max_points = 20
        points = 0
        reason = ""
        
        ema_regime = data.get('ema_regime', 'NEUTRAL')
        ema_spread = data.get('ema_spread', 0)
        
        if direction == "BUY":
            if ema_regime == "EXPANDING_BULL":
                points = 20
                reason = f"Bullish expansion ({ema_spread:.2f}%)"
            elif ema_regime == "COMPRESSED":
                points = 10  # Pending breakout
                reason = "EMA compressed - breakout potential"
            elif ema_regime == "EXPANDING_BEAR":
                points = 0
                reason = "Against trend: EMAs expanding bearish"
        else:  # SELL
            if ema_regime == "EXPANDING_BEAR":
                points = 20
                reason = f"Bearish expansion ({ema_spread:.2f}%)"
            elif ema_regime == "COMPRESSED":
                points = 10
                reason = "EMA compressed - breakout potential"
            elif ema_regime == "EXPANDING_BULL":
                points = 0
                reason = "Against trend: EMAs expanding bullish"
        
        return ScoreComponent(
            name="EMA",
            points=points,
            max_points=max_points,
            reason=reason
        )
    
    def _score_orb_breakout(self, direction: str, data: Dict) -> ScoreComponent:
        """Score ORB breakout with volume (max +30)"""
        max_points = 30
        points = 0
        reason = ""
        
        orb_signal = data.get('orb_signal', 'INSIDE_ORB')
        orb_strength = data.get('orb_strength', 0)
        volume_regime = data.get('volume_regime', 'NORMAL')
        
        # Volume multiplier
        vol_mult = {
            "EXPLOSIVE": 1.0,
            "HIGH": 0.8,
            "NORMAL": 0.5,
            "LOW": 0.2
        }.get(volume_regime, 0.5)
        
        if direction == "BUY":
            if orb_signal == "BREAKOUT_UP":
                base_points = min(30, 15 + orb_strength)  # More strength = more points
                points = int(base_points * vol_mult)
                reason = f"ORB↑ {orb_strength:.1f}% + {volume_regime} vol"
            elif orb_signal == "INSIDE_ORB":
                points = 5
                reason = "Inside ORB - waiting for breakout"
            else:  # BREAKOUT_DOWN
                points = 0
                reason = "ORB breaking down - wrong direction"
        else:  # SELL
            if orb_signal == "BREAKOUT_DOWN":
                base_points = min(30, 15 + orb_strength)
                points = int(base_points * vol_mult)
                reason = f"ORB↓ {orb_strength:.1f}% + {volume_regime} vol"
            elif orb_signal == "INSIDE_ORB":
                points = 5
                reason = "Inside ORB - waiting for breakout"
            else:  # BREAKOUT_UP
                points = 0
                reason = "ORB breaking up - wrong direction"
        
        return ScoreComponent(
            name="ORB+VOL",
            points=points,
            max_points=max_points,
            reason=reason
        )
    
    def _score_htf_alignment(self, direction: str, data: Dict) -> ScoreComponent:
        """Score HTF alignment (max +15)"""
        max_points = 15
        points = 0
        reason = ""
        
        htf_trend = data.get('htf_trend', 'NEUTRAL')
        htf_ema_slope = data.get('htf_ema_slope', 'FLAT')
        htf_alignment = data.get('htf_alignment', 'NEUTRAL')
        
        if direction == "BUY":
            if htf_trend == "BULLISH" and htf_ema_slope == "RISING":
                points = 15
                reason = "HTF fully aligned bullish"
            elif htf_trend == "BULLISH":
                points = 10
                reason = "HTF bullish"
            elif htf_trend == "NEUTRAL":
                points = 5
                reason = "HTF neutral"
            elif htf_trend == "BEARISH":
                points = 0
                reason = "HTF against trade (bearish)"
        else:  # SELL
            if htf_trend == "BEARISH" and htf_ema_slope == "FALLING":
                points = 15
                reason = "HTF fully aligned bearish"
            elif htf_trend == "BEARISH":
                points = 10
                reason = "HTF bearish"
            elif htf_trend == "NEUTRAL":
                points = 5
                reason = "HTF neutral"
            elif htf_trend == "BULLISH":
                points = 0
                reason = "HTF against trade (bullish)"
        
        return ScoreComponent(
            name="HTF",
            points=points,
            max_points=max_points,
            reason=reason
        )
    
    def _score_trend_following(self, symbol: str, direction: str, data: Dict) -> Optional[ScoreComponent]:
        """
        Score using TrendFollowingEngine with enhanced hysteresis.
        
        This is the "real edge" - price escapes balance and doesn't come back.
        Uses:
        - Asymmetric hysteresis (82 for upgrade, 70 to stay)
        - Time-in-state confirmation (2 candles)
        - Shock override (VWAP cross + volume)
        - Context-aware reset (new day, etc.)
        
        Max bonus: +20 points (separate from base 100)
        """
        if not self._trend_available or not self.trend_engine:
            return None
        
        max_points = 20  # Bonus points
        
        try:
            from trend_following import build_trend_signal_from_market_data, TrendState
            
            # Build trend signal from market data
            trend_signal = build_trend_signal_from_market_data(symbol, data)
            trend_decision = self.trend_engine.analyze_trend(trend_signal)
            
            # Get hysteresis-aware state (this is the key!)
            trend_state = trend_decision.trend_state
            trend_score = trend_decision.trend_score
            
            points = 0
            reason = ""
            
            # === STRONG MATCH = FULL BONUS ===
            if direction == "BUY" and trend_state == TrendState.STRONG_BULLISH:
                points = 20
                reason = f"🔥 STRONG TREND BULLISH ({trend_score:.0f}/100) - hysteresis confirmed"
            elif direction == "SELL" and trend_state == TrendState.STRONG_BEARISH:
                points = 20
                reason = f"🔥 STRONG TREND BEARISH ({trend_score:.0f}/100) - hysteresis confirmed"
            
            # === CONFIRMED MATCH = GOOD BONUS ===
            elif direction == "BUY" and trend_state == TrendState.BULLISH:
                points = 12
                reason = f"📈 TREND BULLISH ({trend_score:.0f}/100) - confirmed"
            elif direction == "SELL" and trend_state == TrendState.BEARISH:
                points = 12
                reason = f"📉 TREND BEARISH ({trend_score:.0f}/100) - confirmed"
            
            # === NEUTRAL = NO BONUS ===
            elif trend_state == TrendState.NEUTRAL:
                points = 0
                reason = f"⚠️ No trend ({trend_score:.0f}/100) - hysteresis filtering"
            
            # === OPPOSITE DIRECTION = PENALTY ===
            elif direction == "BUY" and trend_state in [TrendState.BEARISH, TrendState.STRONG_BEARISH]:
                points = -10
                reason = f"❌ TREND AGAINST (bearish {trend_score:.0f}/100)"
            elif direction == "SELL" and trend_state in [TrendState.BULLISH, TrendState.STRONG_BULLISH]:
                points = -10
                reason = f"❌ TREND AGAINST (bullish {trend_score:.0f}/100)"
            
            # Add entry type info
            if trend_decision.entry_type and points > 0:
                reason += f" | Entry: {trend_decision.entry_type.value}"
            
            return ScoreComponent(
                name="TREND_FOLLOWING",
                points=points,
                max_points=max_points,
                reason=reason
            )
            
        except Exception as e:
            return ScoreComponent(
                name="TREND_FOLLOWING",
                points=0,
                max_points=max_points,
                reason=f"Error: {str(e)[:30]}"
            )
    
    def _score_chop_filter(self, data: Dict) -> ScoreComponent:
        """Score chop filter (max +10)"""
        max_points = 10
        
        chop_zone = data.get('chop_zone', False)
        chop_reason = data.get('chop_reason', '')
        orb_reentries = data.get('orb_reentries', 0)
        
        if chop_zone:
            return ScoreComponent(
                name="CHOP",
                points=0,
                max_points=max_points,
                reason=f"In chop zone: {chop_reason}"
            )
        elif orb_reentries >= 2:
            return ScoreComponent(
                name="CHOP",
                points=3,
                max_points=max_points,
                reason=f"Some whipsaw ({orb_reentries} re-entries)"
            )
        else:
            return ScoreComponent(
                name="CHOP",
                points=10,
                max_points=max_points,
                reason="Clean trend, no chop"
            )
    
    def _score_spread_quality(self, data: Dict) -> ScoreComponent:
        """Score spread/slippage quality (max +10)"""
        max_points = 10
        
        # Use volume as proxy for liquidity
        volume_regime = data.get('volume_regime', 'NORMAL')
        
        if volume_regime == "EXPLOSIVE":
            points = 10
            reason = "Excellent liquidity (EXPLOSIVE)"
        elif volume_regime == "HIGH":
            points = 8
            reason = "Good liquidity (HIGH)"
        elif volume_regime == "NORMAL":
            points = 5
            reason = "Normal liquidity"
        else:  # LOW
            points = 0
            reason = "Poor liquidity (LOW volume)"
        
        return ScoreComponent(
            name="LIQUIDITY",
            points=points,
            max_points=max_points,
            reason=reason
        )
    
    def _calculate_penalties(self, direction: str, data: Dict) -> List[ScoreComponent]:
        """Calculate penalty components for conflicts"""
        penalties = []
        
        # Penalty 1: RSI extreme against direction
        rsi = data.get('rsi_14', 50)
        if direction == "BUY" and rsi > 75:
            penalties.append(ScoreComponent(
                name="RSI_OVERBOUGHT",
                points=-15,
                max_points=0,
                reason=f"RSI overbought ({rsi:.0f}) - risky BUY",
                is_penalty=True
            ))
        elif direction == "SELL" and rsi < 25:
            penalties.append(ScoreComponent(
                name="RSI_OVERSOLD",
                points=-15,
                max_points=0,
                reason=f"RSI oversold ({rsi:.0f}) - risky SELL",
                is_penalty=True
            ))
        
        # Penalty 2: VWAP slope against direction
        vwap_slope = data.get('vwap_slope', 'FLAT')
        price_vs_vwap = data.get('price_vs_vwap', 'AT_VWAP')
        
        if direction == "BUY" and vwap_slope == "FALLING" and price_vs_vwap == "BELOW_VWAP":
            penalties.append(ScoreComponent(
                name="VWAP_CONFLICT",
                points=-10,
                max_points=0,
                reason="VWAP falling + price below - weak BUY",
                is_penalty=True
            ))
        elif direction == "SELL" and vwap_slope == "RISING" and price_vs_vwap == "ABOVE_VWAP":
            penalties.append(ScoreComponent(
                name="VWAP_CONFLICT",
                points=-10,
                max_points=0,
                reason="VWAP rising + price above - weak SELL",
                is_penalty=True
            ))
        
        # Penalty 3: Low volume on breakout
        orb_signal = data.get('orb_signal', 'INSIDE_ORB')
        volume_regime = data.get('volume_regime', 'NORMAL')
        
        if orb_signal in ["BREAKOUT_UP", "BREAKOUT_DOWN"] and volume_regime == "LOW":
            penalties.append(ScoreComponent(
                name="LOW_VOL_BREAKOUT",
                points=-20,
                max_points=0,
                reason="Breakout on LOW volume - likely false",
                is_penalty=True
            ))
        
        return penalties
    
    def reset_hysteresis(self, symbol: str = None, reason: str = "POSITION_CLOSED"):
        """
        Reset TrendFollowing hysteresis for a symbol.
        
        Call this when:
        - Position is closed (POSITION_CLOSED)
        - Data health issue (DATA_HALT)
        - Reconciliation mismatch (RECONCILIATION)
        - New trading day is handled automatically
        
        Args:
            symbol: Symbol to reset, or None for all
            reason: Reason for logging
        """
        if self._trend_available and self.trend_engine:
            self.trend_engine.reset_hysteresis(symbol, reason)
    
    def format_score_report(self, score: RegimeScore) -> str:
        """Format score as readable report"""
        lines = [
            f"═══ REGIME SCORE: {score.symbol} {score.direction} ═══",
            f"Score: {score.total_score}/100 (Threshold: {score.threshold})",
            f"Verdict: {score.final_verdict} | Confidence: {score.confidence}",
            "",
            "BREAKDOWN:"
        ]
        
        # Sort by points (positives first)
        sorted_components = sorted(score.components, key=lambda x: x.points, reverse=True)
        
        for c in sorted_components:
            if c.is_penalty:
                lines.append(f"  ❌ {c.name}: {c.points:+d} - {c.reason}")
            elif c.points > 0:
                lines.append(f"  ✅ {c.name}: +{c.points}/{c.max_points} - {c.reason}")
            else:
                lines.append(f"  ⚠️ {c.name}: {c.points}/{c.max_points} - {c.reason}")
        
        lines.append("")
        lines.append(f"Summary: {score.summary}")
        
        return "\n".join(lines)


# Singleton instance
_regime_scorer: Optional[RegimeScorer] = None


def get_regime_scorer() -> RegimeScorer:
    """Get singleton instance of RegimeScorer"""
    global _regime_scorer
    if _regime_scorer is None:
        _regime_scorer = RegimeScorer()
    return _regime_scorer
