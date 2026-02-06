"""
SIGNAL ENGINE MODULE
Implements trading strategies and generates BUY/SELL signals

Strategies:
1. Moving Average Crossover (Trend Following)
2. RSI Mean Reversion
3. Breakout (Previous High/Low)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

from config import (
    MA_FAST, MA_SLOW, MA_TREND,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BREAKOUT_LOOKBACK, VOLUME_MULTIPLIER,
    DEFAULT_SL_PERCENT, DEFAULT_TARGET_PERCENT
)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyType(Enum):
    MA_CROSSOVER = "MA_CROSSOVER"
    RSI_REVERSION = "RSI_REVERSION"
    BREAKOUT = "BREAKOUT"


@dataclass
class Signal:
    """Trading signal with all relevant info"""
    symbol: str
    signal_type: SignalType
    strategy: StrategyType
    entry_price: float
    stop_loss: float
    target: float
    strength: float  # 0-100 signal strength
    reason: str
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strategy': self.strategy.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'strength': self.strength,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


class TechnicalIndicators:
    """Calculate technical indicators on OHLC data"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range (for volatility)"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()


class SignalEngine:
    """
    Main signal generation engine
    Combines multiple strategies and filters
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.active_strategies = [
            StrategyType.MA_CROSSOVER,
            StrategyType.RSI_REVERSION,
            StrategyType.BREAKOUT
        ]
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze a symbol and return signal if any
        
        Args:
            symbol: Trading symbol (e.g., NSE:RELIANCE)
            df: OHLC DataFrame with columns: open, high, low, close, volume
        
        Returns:
            Signal object or None
        """
        if df.empty or len(df) < 50:
            return None
        
        signals = []
        
        # Run all active strategies
        for strategy in self.active_strategies:
            signal = self._run_strategy(symbol, df, strategy)
            if signal and signal.signal_type != SignalType.HOLD:
                signals.append(signal)
        
        # Return strongest signal
        if signals:
            return max(signals, key=lambda s: s.strength)
        
        return None
    
    def _run_strategy(self, symbol: str, df: pd.DataFrame, strategy: StrategyType) -> Optional[Signal]:
        """Run a specific strategy"""
        if strategy == StrategyType.MA_CROSSOVER:
            return self._ma_crossover_strategy(symbol, df)
        elif strategy == StrategyType.RSI_REVERSION:
            return self._rsi_reversion_strategy(symbol, df)
        elif strategy == StrategyType.BREAKOUT:
            return self._breakout_strategy(symbol, df)
        return None
    
    def _ma_crossover_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Moving Average Crossover Strategy
        
        BUY: Fast MA crosses above Slow MA, price above Trend MA
        SELL: Fast MA crosses below Slow MA, price below Trend MA
        """
        close = df['close']
        
        # Calculate MAs
        fast_ma = self.indicators.ema(close, MA_FAST)
        slow_ma = self.indicators.ema(close, MA_SLOW)
        trend_ma = self.indicators.sma(close, MA_TREND)
        
        # Current values
        current_price = close.iloc[-1]
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        current_trend = trend_ma.iloc[-1]
        
        # Previous values (for crossover detection)
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Crossover detection
        bullish_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
        bearish_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)
        
        # Trend filter
        uptrend = current_price > current_trend
        downtrend = current_price < current_trend
        
        signal_type = SignalType.HOLD
        reason = ""
        strength = 0
        
        if bullish_cross and uptrend:
            signal_type = SignalType.BUY
            reason = f"Fast EMA({MA_FAST}) crossed above Slow EMA({MA_SLOW}), price above SMA({MA_TREND})"
            # Strength based on MA separation
            separation = (current_fast - current_slow) / current_slow * 100
            strength = min(80, 50 + separation * 10)
            
        elif bearish_cross and downtrend:
            signal_type = SignalType.SELL
            reason = f"Fast EMA({MA_FAST}) crossed below Slow EMA({MA_SLOW}), price below SMA({MA_TREND})"
            separation = (current_slow - current_fast) / current_slow * 100
            strength = min(80, 50 + separation * 10)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate SL and target
        atr = self.indicators.atr(df).iloc[-1]
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (2 * atr)
            target = current_price + (3 * atr)
        else:
            stop_loss = current_price + (2 * atr)
            target = current_price - (3 * atr)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy=StrategyType.MA_CROSSOVER,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            strength=strength,
            reason=reason,
            timestamp=datetime.now()
        )
    
    def _rsi_reversion_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        RSI Mean Reversion Strategy
        
        BUY: RSI < 30 (oversold) and turning up
        SELL: RSI > 70 (overbought) and turning down
        """
        close = df['close']
        rsi = self.indicators.rsi(close, RSI_PERIOD)
        
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        signal_type = SignalType.HOLD
        reason = ""
        strength = 0
        
        # Oversold bounce
        if current_rsi < RSI_OVERSOLD and current_rsi > prev_rsi:
            signal_type = SignalType.BUY
            reason = f"RSI({current_rsi:.1f}) oversold and turning up"
            strength = min(90, 60 + (RSI_OVERSOLD - current_rsi) * 2)
            
        # Overbought reversal
        elif current_rsi > RSI_OVERBOUGHT and current_rsi < prev_rsi:
            signal_type = SignalType.SELL
            reason = f"RSI({current_rsi:.1f}) overbought and turning down"
            strength = min(90, 60 + (current_rsi - RSI_OVERBOUGHT) * 2)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # SL/Target based on percentage
        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - DEFAULT_SL_PERCENT / 100)
            target = current_price * (1 + DEFAULT_TARGET_PERCENT / 100)
        else:
            stop_loss = current_price * (1 + DEFAULT_SL_PERCENT / 100)
            target = current_price * (1 - DEFAULT_TARGET_PERCENT / 100)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy=StrategyType.RSI_REVERSION,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            strength=strength,
            reason=reason,
            timestamp=datetime.now()
        )
    
    def _breakout_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
        Breakout Strategy
        
        BUY: Price breaks above N-day high with volume confirmation
        SELL: Price breaks below N-day low with volume confirmation
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # N-day high/low (excluding current candle)
        lookback = min(BREAKOUT_LOOKBACK, len(df) - 1)
        n_day_high = high.iloc[-(lookback+1):-1].max()
        n_day_low = low.iloc[-(lookback+1):-1].min()
        
        current_price = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = self.indicators.volume_sma(volume, 20).iloc[-1]
        
        # Volume confirmation
        volume_confirmed = current_volume > (avg_volume * VOLUME_MULTIPLIER)
        
        signal_type = SignalType.HOLD
        reason = ""
        strength = 0
        
        # Breakout above resistance
        if current_high > n_day_high and volume_confirmed:
            signal_type = SignalType.BUY
            breakout_pct = (current_price - n_day_high) / n_day_high * 100
            volume_ratio = current_volume / avg_volume
            reason = f"Breakout above {BREAKOUT_LOOKBACK}-day high (₹{n_day_high:.2f}), volume {volume_ratio:.1f}x avg"
            strength = min(85, 55 + breakout_pct * 5 + (volume_ratio - 1) * 10)
            
        # Breakdown below support
        elif current_low < n_day_low and volume_confirmed:
            signal_type = SignalType.SELL
            breakdown_pct = (n_day_low - current_price) / n_day_low * 100
            volume_ratio = current_volume / avg_volume
            reason = f"Breakdown below {BREAKOUT_LOOKBACK}-day low (₹{n_day_low:.2f}), volume {volume_ratio:.1f}x avg"
            strength = min(85, 55 + breakdown_pct * 5 + (volume_ratio - 1) * 10)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # SL at breakout level, target at 2x risk
        if signal_type == SignalType.BUY:
            stop_loss = n_day_high * 0.995  # Just below breakout level
            risk = current_price - stop_loss
            target = current_price + (risk * 2)
        else:
            stop_loss = n_day_low * 1.005  # Just above breakdown level
            risk = stop_loss - current_price
            target = current_price - (risk * 2)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy=StrategyType.BREAKOUT,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            strength=strength,
            reason=reason,
            timestamp=datetime.now()
        )
    
    def scan_universe(self, ohlc_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Scan all symbols in universe for signals
        
        Args:
            ohlc_data: Dict of symbol -> OHLC DataFrame
        
        Returns:
            List of signals sorted by strength
        """
        signals = []
        
        for symbol, df in ohlc_data.items():
            signal = self.analyze(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by strength (highest first)
        signals.sort(key=lambda s: s.strength, reverse=True)
        
        return signals
    
    def get_strategy_summary(self, symbol: str, df: pd.DataFrame) -> dict:
        """Get indicator values and analysis summary for a symbol"""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        
        # Calculate all indicators
        fast_ma = self.indicators.ema(close, MA_FAST).iloc[-1]
        slow_ma = self.indicators.ema(close, MA_SLOW).iloc[-1]
        trend_ma = self.indicators.sma(close, MA_TREND).iloc[-1]
        rsi = self.indicators.rsi(close, RSI_PERIOD).iloc[-1]
        atr = self.indicators.atr(df).iloc[-1]
        
        current_price = close.iloc[-1]
        
        return {
            'symbol': symbol,
            'price': current_price,
            'fast_ma': round(fast_ma, 2),
            'slow_ma': round(slow_ma, 2),
            'trend_ma': round(trend_ma, 2),
            'rsi': round(rsi, 2),
            'atr': round(atr, 2),
            'trend': 'BULLISH' if current_price > trend_ma else 'BEARISH',
            'ma_trend': 'BULLISH' if fast_ma > slow_ma else 'BEARISH',
            'rsi_zone': 'OVERSOLD' if rsi < 30 else ('OVERBOUGHT' if rsi > 70 else 'NEUTRAL')
        }


# Singleton instance
_signal_engine = None

def get_signal_engine() -> SignalEngine:
    """Get singleton SignalEngine instance"""
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = SignalEngine()
    return _signal_engine


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, '..')
    
    from data_manager import get_data_manager
    
    dm = get_data_manager()
    se = get_signal_engine()
    
    if dm.is_authenticated():
        print("📊 Testing Signal Engine...")
        
        # Get data for a symbol
        df = dm.get_historical_data("NSE:RELIANCE", "5minute", days=10)
        
        if not df.empty:
            print(f"\n📈 Data: {len(df)} candles")
            
            # Get summary
            summary = se.get_strategy_summary("NSE:RELIANCE", df)
            print("\n📋 Indicator Summary:")
            for k, v in summary.items():
                print(f"   {k}: {v}")
            
            # Get signal
            signal = se.analyze("NSE:RELIANCE", df)
            if signal:
                print(f"\n🎯 Signal: {signal.signal_type.value}")
                print(f"   Strategy: {signal.strategy.value}")
                print(f"   Entry: ₹{signal.entry_price:,.2f}")
                print(f"   SL: ₹{signal.stop_loss:,.2f}")
                print(f"   Target: ₹{signal.target:,.2f}")
                print(f"   Strength: {signal.strength:.0f}/100")
                print(f"   Reason: {signal.reason}")
            else:
                print("\n⏸️ No signal - HOLD")
    else:
        print("❌ Not authenticated")
