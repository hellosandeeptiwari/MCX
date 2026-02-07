"""
BACKTEST: Trend Following with Enhanced Hysteresis

Tests on historical data:
- Time-in-state confirmation (2 candles for upgrade)
- Asymmetric thresholds (82 for upgrade, 70 to stay)
- Shock override (VWAP cross + volume)
- Context-aware reset

Usage: python backtest_trend_following.py
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trend_following import TrendFollowingEngine, TrendSignal, TrendState, TrendEntryType


@dataclass
class BacktestTrade:
    """Single backtest trade"""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    entry_reason: str
    entry_score: float
    entry_state: str
    
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    pnl: float = 0
    pnl_pct: float = 0
    r_multiple: float = 0
    
    stop_loss: float = 0
    target: float = 0
    holding_candles: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    symbol: str
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_r_multiple: float
    
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Enhanced hysteresis stats
    hysteresis_prevented_entries: int = 0
    strong_signals_count: int = 0
    shock_overrides: int = 0
    upgrade_confirmations: int = 0


class TrendFollowingBacktest:
    """
    Backtest trend following strategy on historical data.
    
    Uses 5-minute candles and simulates:
    - Entry on STRONG_BULLISH/STRONG_BEARISH with hysteresis confirmation
    - Exit on SL hit, target hit, or state downgrade
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.engine = TrendFollowingEngine()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Risk parameters
        self.risk_per_trade = 0.01  # 1% risk
        self.target_r = 2.0  # 2R target
        self.max_positions = 2
        
        # Tracking
        self.trades: List[BacktestTrade] = []
        self.open_positions: Dict[str, BacktestTrade] = {}
        self.equity_curve: List[float] = [initial_capital]
        
        # Stats
        self.hysteresis_prevented = 0
        self.shock_overrides = 0
        self.upgrade_confirmations = 0
    
    def _calculate_indicators(self, candles: List[Dict], idx: int, day_start_idx: int = None) -> Dict:
        """Calculate indicators for a given candle index"""
        if idx < 20:  # Need at least 20 candles for indicators
            return {}
        
        # Get relevant candles
        lookback = candles[max(0, idx-50):idx+1]
        current = candles[idx]
        
        # Find day start for ORB
        if day_start_idx is None:
            current_date = current['date'].date()
            for i in range(idx, max(0, idx-80), -1):
                if candles[i]['date'].date() != current_date:
                    day_start_idx = i + 1
                    break
            else:
                day_start_idx = max(0, idx - 75)  # Fallback
        
        # Calculate VWAP (simplified - cumulative from day start)
        day_candles = candles[day_start_idx:idx+1]
        total_pv = sum(c['close'] * c['volume'] for c in day_candles)
        total_vol = sum(c['volume'] for c in day_candles)
        vwap = total_pv / total_vol if total_vol > 0 else current['close']
        
        # VWAP slope (10-candle rolling comparison for stronger slope)
        if len(lookback) >= 15:
            # Use price trend as proxy for VWAP slope
            price_10_ago = lookback[-11]['close']
            price_now = current['close']
            vwap_slope = ((price_now - price_10_ago) / price_10_ago * 100) if price_10_ago > 0 else 0
        else:
            vwap_slope = 0
        
        # EMA 9 and 21
        def calc_ema(data, period):
            if len(data) < period:
                return data[-1]['close']
            multiplier = 2 / (period + 1)
            ema = data[0]['close']
            for c in data[1:]:
                ema = (c['close'] - ema) * multiplier + ema
            return ema
        
        ema_9 = calc_ema(lookback, 9)
        ema_21 = calc_ema(lookback, 21)
        ema_spread = ((ema_9 - ema_21) / ema_21 * 100) if ema_21 > 0 else 0
        
        # Check if EMAs are expanding (spread increasing over last 5 candles)
        if len(lookback) >= 7:
            prev_ema_9 = calc_ema(lookback[:-5], 9)
            prev_ema_21 = calc_ema(lookback[:-5], 21)
            prev_spread = ((prev_ema_9 - prev_ema_21) / prev_ema_21 * 100) if prev_ema_21 > 0 else 0
            ema_expanding = abs(ema_spread) > abs(prev_spread) * 1.1  # 10% more expansion
        else:
            ema_expanding = False
        
        # Volume regime - compare to 20-candle average
        avg_vol = sum(c['volume'] for c in lookback[-20:]) / 20 if len(lookback) >= 20 else current['volume']
        vol_ratio = current['volume'] / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio > 2.0:
            volume_regime = "EXPLOSIVE"
        elif vol_ratio > 1.3:
            volume_regime = "HIGH"
        elif vol_ratio > 0.7:
            volume_regime = "NORMAL"
        else:
            volume_regime = "LOW"
        
        # ORB (first 3 candles of the day = 15 minutes)
        orb_candles = candles[day_start_idx:min(day_start_idx+3, idx+1)]
        if orb_candles:
            orb_high = max(c['high'] for c in orb_candles)
            orb_low = min(c['low'] for c in orb_candles)
        else:
            orb_high = current['high']
            orb_low = current['low']
        
        # Track ORB break and hold candles
        orb_broken = "NONE"
        orb_hold_candles = 0
        for i in range(day_start_idx + 3, idx + 1):
            if i >= len(candles):
                break
            if candles[i]['close'] > orb_high:
                if orb_broken != "UP":
                    orb_broken = "UP"
                    orb_hold_candles = 0
                orb_hold_candles += 1
            elif candles[i]['close'] < orb_low:
                if orb_broken != "DOWN":
                    orb_broken = "DOWN"
                    orb_hold_candles = 0
                orb_hold_candles += 1
            else:
                if orb_broken != "NONE":
                    orb_hold_candles += 1  # Still holding even inside ORB
        
        # Pullback detection (last 5 candles)
        if len(lookback) >= 6:
            recent = lookback[-6:]
            highest = max(c['high'] for c in recent)
            lowest = min(c['low'] for c in recent)
            current_close = current['close']
            
            # If trending up, pullback is distance from high
            if ema_spread > 0:  # Bullish trend
                pullback_pct = (highest - current_close) / highest * 100 if highest > 0 else 0
                pullback_candles = sum(1 for c in recent if c['close'] < c['open'])  # Red candles
            else:  # Bearish trend
                pullback_pct = (current_close - lowest) / lowest * 100 if lowest > 0 else 0
                pullback_candles = sum(1 for c in recent if c['close'] > c['open'])  # Green candles
            
            pullback_depth_pct = pullback_pct
        else:
            pullback_depth_pct = 0
            pullback_candles = 0
        
        # RSI calculation
        gains = []
        losses = []
        for i in range(1, min(15, len(lookback))):
            change = lookback[-i]['close'] - lookback[-i-1]['close']
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        # ADX (simplified - trend strength)
        true_ranges = []
        plus_dm = []
        minus_dm = []
        for i in range(1, min(15, len(lookback))):
            high = lookback[-i]['high']
            low = lookback[-i]['low']
            prev_high = lookback[-i-1]['high']
            prev_low = lookback[-i-1]['low']
            prev_close = lookback[-i-1]['close']
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            
            up_move = high - prev_high
            down_move = prev_low - low
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 1
        avg_plus_dm = sum(plus_dm) / len(plus_dm) if plus_dm else 0
        avg_minus_dm = sum(minus_dm) / len(minus_dm) if minus_dm else 0
        
        plus_di = (avg_plus_dm / atr * 100) if atr > 0 else 0
        minus_di = (avg_minus_dm / atr * 100) if atr > 0 else 0
        
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        adx = (di_diff / di_sum * 100) if di_sum > 0 else 20
        
        # HTF trend (using 50-period EMA as proxy)
        ema_50 = calc_ema(lookback, min(50, len(lookback)))
        if current['close'] > ema_50 * 1.005:
            htf_trend = "BULLISH"
        elif current['close'] < ema_50 * 0.995:
            htf_trend = "BEARISH"
        else:
            htf_trend = "NEUTRAL"
        
        return {
            'ltp': current['close'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'vwap': vwap,
            'vwap_slope': vwap_slope,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_spread_pct': ema_spread,
            'ema_expanding': ema_expanding,
            'volume_regime': volume_regime,
            'volume_ratio': vol_ratio,
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_broken': orb_broken,
            'orb_hold_candles': orb_hold_candles,
            'rsi_14': rsi,
            'adx': adx,
            'htf_trend': htf_trend,
            'pullback_depth_pct': pullback_depth_pct,
            'pullback_candles': pullback_candles
        }
    
    def _build_trend_signal(self, symbol: str, indicators: Dict) -> TrendSignal:
        """Build TrendSignal from indicators"""
        return TrendSignal(
            symbol=symbol,
            ltp=indicators.get('ltp', 0),
            open=indicators.get('open', 0),
            high=indicators.get('high', 0),
            low=indicators.get('low', 0),
            vwap=indicators.get('vwap', 0),
            vwap_slope=indicators.get('vwap_slope', 0),
            vwap_distance_pct=((indicators.get('ltp', 0) - indicators.get('vwap', 0)) / indicators.get('vwap', 1)) * 100,
            ema_9=indicators.get('ema_9', 0),
            ema_21=indicators.get('ema_21', 0),
            ema_spread_pct=indicators.get('ema_spread_pct', 0),
            ema_expanding=indicators.get('ema_expanding', False),
            volume_regime=indicators.get('volume_regime', 'NORMAL'),
            volume_ratio=indicators.get('volume_ratio', 1.0),
            orb_high=indicators.get('orb_high', 0),
            orb_low=indicators.get('orb_low', 0),
            orb_broken=indicators.get('orb_broken', 'NONE'),
            orb_hold_candles=indicators.get('orb_hold_candles', 0),
            pullback_depth_pct=indicators.get('pullback_depth_pct', 0),
            pullback_candles=indicators.get('pullback_candles', 0),
            rsi_14=indicators.get('rsi_14', 50),
            adx=indicators.get('adx', 20),
            htf_trend=indicators.get('htf_trend', 'NEUTRAL')
        )
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        quantity = int(risk_amount / risk_per_share)
        return max(1, quantity)
    
    def run_backtest(self, symbol: str, candles: List[Dict]) -> BacktestResult:
        """
        Run backtest on historical candles.
        
        Candles format: [{'date': datetime, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}]
        """
        print(f"\n{'='*70}")
        print(f"BACKTEST: Trend Following with Enhanced Hysteresis")
        print(f"Symbol: {symbol}")
        print(f"Period: {candles[0]['date']} to {candles[-1]['date']}")
        print(f"Candles: {len(candles)}")
        print(f"{'='*70}\n")
        
        # Reset state
        self.trades = []
        self.open_positions = {}
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.engine = TrendFollowingEngine()  # Fresh engine
        
        self.hysteresis_prevented = 0
        self.shock_overrides = 0
        self.upgrade_confirmations = 0
        strong_signals = 0
        
        # Process each candle
        for idx, candle in enumerate(candles):
            if idx < 20:  # Need warmup
                continue
            
            # Calculate indicators
            indicators = self._calculate_indicators(candles, idx)
            if not indicators:
                continue
            
            current_price = candle['close']
            candle_time = candle['date']
            
            # Update open positions
            for pos_symbol, trade in list(self.open_positions.items()):
                trade.holding_candles += 1
                
                # Check stop loss
                if trade.direction == "LONG" and candle['low'] <= trade.stop_loss:
                    self._close_position(trade, trade.stop_loss, candle_time, "STOP_LOSS")
                elif trade.direction == "SHORT" and candle['high'] >= trade.stop_loss:
                    self._close_position(trade, trade.stop_loss, candle_time, "STOP_LOSS")
                # Check target
                elif trade.direction == "LONG" and candle['high'] >= trade.target:
                    self._close_position(trade, trade.target, candle_time, "TARGET_HIT")
                elif trade.direction == "SHORT" and candle['low'] <= trade.target:
                    self._close_position(trade, trade.target, candle_time, "TARGET_HIT")
            
            # Get trend decision
            trend_signal = self._build_trend_signal(symbol, indicators)
            decision = self.engine.analyze_trend(trend_signal)
            
            # Track stats
            if decision.trend_state in [TrendState.STRONG_BULLISH, TrendState.STRONG_BEARISH]:
                strong_signals += 1
            
            # Check for upgrade confirmation message in decision
            for reason in decision.reasons:
                if "UPGRADE CONFIRMED" in reason:
                    self.upgrade_confirmations += 1
                if "SHOCK" in reason:
                    self.shock_overrides += 1
                if "Upgrade pending" in reason:
                    self.hysteresis_prevented += 1
            
            # Entry logic - only on STRONG signals with should_enter
            if decision.should_enter and len(self.open_positions) < self.max_positions:
                if decision.trend_state == TrendState.STRONG_BULLISH:
                    direction = "LONG"
                    entry_price = current_price
                    stop_loss = indicators['orb_low'] if indicators['orb_broken'] == 'UP' else entry_price * 0.99
                    target = entry_price + (entry_price - stop_loss) * self.target_r
                    
                    quantity = self._calculate_position_size(entry_price, stop_loss)
                    
                    trade = BacktestTrade(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        entry_time=candle_time,
                        entry_reason=decision.entry_type.value if decision.entry_type else "TREND",
                        entry_score=decision.trend_score,
                        entry_state=decision.trend_state.value,
                        stop_loss=stop_loss,
                        target=target
                    )
                    
                    self.open_positions[symbol] = trade
                    print(f"📈 LONG ENTRY @ {entry_price:.2f} | SL: {stop_loss:.2f} | Target: {target:.2f}")
                    print(f"   Score: {decision.trend_score:.0f} | State: {decision.trend_state.value}")
                
                elif decision.trend_state == TrendState.STRONG_BEARISH:
                    direction = "SHORT"
                    entry_price = current_price
                    stop_loss = indicators['orb_high'] if indicators['orb_broken'] == 'DOWN' else entry_price * 1.01
                    target = entry_price - (stop_loss - entry_price) * self.target_r
                    
                    quantity = self._calculate_position_size(entry_price, stop_loss)
                    
                    trade = BacktestTrade(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        entry_time=candle_time,
                        entry_reason=decision.entry_type.value if decision.entry_type else "TREND",
                        entry_score=decision.trend_score,
                        entry_state=decision.trend_state.value,
                        stop_loss=stop_loss,
                        target=target
                    )
                    
                    self.open_positions[symbol] = trade
                    print(f"📉 SHORT ENTRY @ {entry_price:.2f} | SL: {stop_loss:.2f} | Target: {target:.2f}")
                    print(f"   Score: {decision.trend_score:.0f} | State: {decision.trend_state.value}")
            
            # Exit on trend state downgrade
            if symbol in self.open_positions:
                trade = self.open_positions[symbol]
                if trade.direction == "LONG" and decision.trend_state in [TrendState.NEUTRAL, TrendState.BEARISH, TrendState.STRONG_BEARISH]:
                    self._close_position(trade, current_price, candle_time, "STATE_DOWNGRADE")
                elif trade.direction == "SHORT" and decision.trend_state in [TrendState.NEUTRAL, TrendState.BULLISH, TrendState.STRONG_BULLISH]:
                    self._close_position(trade, current_price, candle_time, "STATE_DOWNGRADE")
            
            # Update equity curve
            unrealized_pnl = sum(
                (current_price - t.entry_price) if t.direction == "LONG" else (t.entry_price - current_price)
                for t in self.open_positions.values()
            )
            self.equity_curve.append(self.capital + unrealized_pnl)
        
        # Close any remaining positions
        final_price = candles[-1]['close']
        final_time = candles[-1]['date']
        for trade in list(self.open_positions.values()):
            self._close_position(trade, final_price, final_time, "END_OF_DATA")
        
        # Calculate results
        return self._calculate_results(symbol, candles, strong_signals)
    
    def _close_position(self, trade: BacktestTrade, exit_price: float, exit_time: datetime, reason: str):
        """Close a position and record results"""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        
        if trade.direction == "LONG":
            trade.pnl = exit_price - trade.entry_price
            trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
        else:
            trade.pnl = trade.entry_price - exit_price
            trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
        
        risk = abs(trade.entry_price - trade.stop_loss)
        trade.r_multiple = trade.pnl / risk if risk > 0 else 0
        
        # Update capital
        position_size = self.capital * self.risk_per_trade / (abs(trade.entry_price - trade.stop_loss) / trade.entry_price)
        self.capital += trade.pnl * (position_size / trade.entry_price)
        
        self.trades.append(trade)
        
        if trade.symbol in self.open_positions:
            del self.open_positions[trade.symbol]
            self.engine.reset_hysteresis(trade.symbol, "POSITION_CLOSED")
        
        result_emoji = "✅" if trade.pnl > 0 else "❌"
        print(f"{result_emoji} EXIT {trade.direction} @ {exit_price:.2f} | PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%) | R: {trade.r_multiple:.2f} | Reason: {reason}")
    
    def _calculate_results(self, symbol: str, candles: List[Dict], strong_signals: int) -> BacktestResult:
        """Calculate final backtest results"""
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = abs(sum(t.pnl for t in losing) / len(losing)) if losing else 0
        profit_factor = (sum(t.pnl for t in winning) / abs(sum(t.pnl for t in losing))) if losing and sum(t.pnl for t in losing) != 0 else 0
        
        avg_r = sum(t.r_multiple for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        period = f"{candles[0]['date'].strftime('%Y-%m-%d')} to {candles[-1]['date'].strftime('%Y-%m-%d')}"
        
        return BacktestResult(
            symbol=symbol,
            period=period,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_dd,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_r_multiple=avg_r,
            trades=self.trades,
            hysteresis_prevented_entries=self.hysteresis_prevented,
            strong_signals_count=strong_signals,
            shock_overrides=self.shock_overrides,
            upgrade_confirmations=self.upgrade_confirmations
        )
    
    def print_results(self, result: BacktestResult):
        """Print formatted backtest results"""
        print(f"\n{'='*70}")
        print("BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Symbol: {result.symbol}")
        print(f"Period: {result.period}")
        print()
        
        print("═══ PERFORMANCE ═══")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Winners:         {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"Losers:          {result.losing_trades} ({100-result.win_rate:.1f}%)")
        print()
        print(f"Total P&L:       ₹{result.total_pnl:.2f} ({result.total_pnl_pct:.2f}%)")
        print(f"Max Drawdown:    {result.max_drawdown:.2f}%")
        print()
        print(f"Avg Win:         ₹{result.avg_win:.2f}")
        print(f"Avg Loss:        ₹{result.avg_loss:.2f}")
        print(f"Profit Factor:   {result.profit_factor:.2f}")
        print(f"Avg R-Multiple:  {result.avg_r_multiple:.2f}R")
        print()
        
        print("═══ HYSTERESIS STATS ═══")
        print(f"STRONG Signals:           {result.strong_signals_count}")
        print(f"Upgrade Confirmations:    {result.upgrade_confirmations}")
        print(f"Hysteresis Prevented:     {result.hysteresis_prevented_entries}")
        print(f"Shock Overrides:          {result.shock_overrides}")
        print()
        
        # Trade breakdown
        print("═══ TRADE LOG ═══")
        for i, trade in enumerate(result.trades[:20], 1):  # Show first 20
            emoji = "✅" if trade.pnl > 0 else "❌"
            print(f"{i:2d}. {emoji} {trade.direction:5s} @ {trade.entry_price:.2f} → {trade.exit_price:.2f} | "
                  f"P&L: {trade.pnl:+.2f} ({trade.r_multiple:+.2f}R) | {trade.exit_reason}")
        
        if len(result.trades) > 20:
            print(f"... and {len(result.trades) - 20} more trades")


def generate_synthetic_data(symbol: str, days: int = 30) -> List[Dict]:
    """
    Generate synthetic 5-minute candle data for testing.
    Simulates realistic market movement with STRONG trends and consolidations.
    Designed to generate high-quality trend signals for testing hysteresis.
    """
    import random
    
    candles = []
    base_price = 2500  # Starting price
    current_price = base_price
    
    # Simulate trading days (9:15 AM to 3:30 PM = 75 five-minute candles per day)
    candles_per_day = 75
    start_date = datetime.now() - timedelta(days=days)
    
    # Trend phases - create stronger, longer trends
    trend_direction = 1  # 1 = bullish, -1 = bearish
    trend_strength = random.uniform(0.5, 0.9)  # Stronger trends
    trend_duration = random.randint(40, 100)   # Longer trends
    trend_counter = 0
    
    # Volume cycle
    volume_base = 50000
    
    for day in range(days):
        day_date = start_date + timedelta(days=day)
        
        # Skip weekends
        if day_date.weekday() >= 5:
            continue
        
        day_start = day_date.replace(hour=9, minute=15, second=0, microsecond=0)
        day_open = current_price
        day_high = current_price
        day_low = current_price
        
        for candle_idx in range(candles_per_day):
            candle_time = day_start + timedelta(minutes=candle_idx * 5)
            
            # Update trend - longer duration for stronger signals
            trend_counter += 1
            if trend_counter >= trend_duration:
                # 30% chance to reverse, 70% to continue
                if random.random() > 0.7:
                    trend_direction *= -1
                trend_strength = random.uniform(0.4, 0.8)
                trend_duration = random.randint(30, 80)
                trend_counter = 0
            
            # Generate OHLC with strong directional bias
            volatility = base_price * 0.0015  # 0.15% base volatility
            trend_move = trend_direction * trend_strength * volatility
            
            # Add momentum - trends accelerate
            momentum = 1 + (trend_counter / trend_duration) * 0.5
            trend_move *= momentum
            
            noise = random.gauss(0, volatility * 0.3)  # Less noise relative to trend
            
            open_price = current_price
            close_price = open_price + trend_move + noise
            
            # High/low based on trend direction
            if trend_direction > 0:
                high_price = max(open_price, close_price) + random.uniform(0, volatility * 0.5)
                low_price = min(open_price, close_price) - random.uniform(0, volatility * 0.2)
            else:
                high_price = max(open_price, close_price) + random.uniform(0, volatility * 0.2)
                low_price = min(open_price, close_price) - random.uniform(0, volatility * 0.5)
            
            # Update day range
            day_high = max(day_high, high_price)
            day_low = min(day_low, low_price)
            
            # Volume - higher on stronger trend moves, spikes on breakouts
            trend_progress = trend_counter / trend_duration
            volume_multiplier = 1 + trend_strength * 2 + trend_progress
            
            # First candle of day or breakout candles get extra volume
            if candle_idx < 3:
                volume_multiplier *= 1.5
            if abs(close_price - day_open) / day_open > 0.005:  # 0.5% breakout
                volume_multiplier *= 2
            
            volume = int(volume_base * volume_multiplier * random.uniform(0.8, 1.2))
            
            candles.append({
                'date': candle_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
            
            current_price = close_price
    
    return candles


def fetch_kite_historical(symbol: str, days: int = 30) -> Optional[List[Dict]]:
    """
    Fetch real historical data from Zerodha Kite API.
    Falls back to synthetic data if API unavailable.
    """
    try:
        from zerodha_tools import get_tools
        tools = get_tools(paper_mode=True)
        
        if not tools.kite:
            print("⚠️ Kite API not available, using synthetic data")
            return None
        
        # Parse symbol to get exchange and tradingsymbol
        if ":" in symbol:
            exchange, tradingsymbol = symbol.split(":")
        else:
            exchange = "NSE"
            tradingsymbol = symbol
        
        # Get instrument token
        print(f"📡 Fetching instruments for {exchange}...")
        instruments = tools.kite.instruments(exchange)
        token = None
        for inst in instruments:
            if inst['tradingsymbol'] == tradingsymbol:
                token = inst['instrument_token']
                print(f"   Found token: {token} for {tradingsymbol}")
                break
        
        if not token:
            print(f"⚠️ No instrument token found for {tradingsymbol}")
            return None
        
        # Fetch historical data (5-minute candles)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        print(f"📊 Fetching {days} days of 5-minute data...")
        data = tools.kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="5minute"
        )
        
        if not data:
            print("⚠️ No historical data returned")
            return None
        
        # Convert to our format
        candles = []
        for d in data:
            candles.append({
                'date': d['date'] if isinstance(d['date'], datetime) else datetime.fromisoformat(str(d['date'])),
                'open': float(d['open']),
                'high': float(d['high']),
                'low': float(d['low']),
                'close': float(d['close']),
                'volume': int(d['volume'])
            })
        
        print(f"✅ Fetched {len(candles)} real candles from Kite")
        return candles
        
    except Exception as e:
        print(f"⚠️ Could not fetch historical data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("TREND FOLLOWING BACKTEST WITH ENHANCED HYSTERESIS")
    print("=" * 70)
    print()
    print("Features being tested:")
    print("  1. Time-in-state confirmation (2 candles for STRONG upgrade)")
    print("  2. Asymmetric thresholds (82 to upgrade, 70 to stay)")
    print("  3. Shock override (VWAP cross + volume)")
    print("  4. Context-aware reset (position closed, etc.)")
    print()
    
    # Configuration
    symbol = "NSE:RELIANCE"
    days = 30
    
    # Try to fetch REAL data from Kite first
    print(f"Attempting to fetch REAL historical data from Kite API...")
    candles = fetch_kite_historical(symbol, days)
    
    if candles is None or len(candles) < 100:
        print(f"\n⚠️ Real data unavailable, falling back to synthetic data...")
        print(f"Generating {days} days of synthetic 5-min data for {symbol}...")
        candles = generate_synthetic_data(symbol, days)
        print(f"Generated {len(candles)} synthetic candles")
    else:
        print(f"\n✅ Using REAL historical data: {len(candles)} candles")
    
    print()
    
    # Run backtest
    backtest = TrendFollowingBacktest(initial_capital=100000)
    result = backtest.run_backtest(symbol, candles)
    
    # Print results
    backtest.print_results(result)
    
    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
