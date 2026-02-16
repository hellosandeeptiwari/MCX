"""
DAILY DIRECTION FEATURE ENGINEERING — Features for next-day UP vs DOWN prediction

Architecture:
  DAILY timeframe (not 5-min — intraday direction is proven unpredictable)
  
  35 features in 7 groups:
    1. Multi-timeframe momentum (6 features: 1/2/3/5/10/20 day returns)
    2. Trend indicators (5: RSI-14, RSI-7, dist_SMA10, dist_SMA20, dist_SMA50)
    3. Volatility (4: ATR-14, range_ratio, BB_position, realized_vol_5d)
    4. Volume (3: vol_ratio_5, vol_ratio_20, volume_trend)
    5. Candle structure (4: body_pct, upper_wick, lower_wick, close_position) 
    6. Streak/Pattern (5: consec_days, up_days_5, mean_reversion, trend_strength, acceleration)
    7. Calendar/Context (3: day_of_week_sin, day_of_week_cos, week_of_month)
    8. High-confidence signals (5: proximity to 20d high/low, gap, vol_on_up, vol_on_down)
    
  Total: 35 pure technical features from daily OHLCV data
"""

import numpy as np
import pandas as pd
from typing import List


def get_direction_feature_names() -> List[str]:
    """Return all 35 feature names for daily direction model."""
    return [
        # Momentum (6)
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        # Trend (5)
        'rsi_14', 'rsi_7', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
        # Volatility (4)
        'atr_14_pct', 'range_ratio', 'bb_position', 'realized_vol_5d',
        # Volume (3)
        'vol_ratio_5', 'vol_ratio_20', 'volume_trend',
        # Candle structure (4)
        'body_pct', 'upper_wick_pct', 'lower_wick_pct', 'close_position',
        # Streak/Pattern (5)
        'consec_days', 'up_days_5', 'mean_reversion', 'trend_strength', 'acceleration',
        # Calendar (3)
        'day_of_week_sin', 'day_of_week_cos', 'week_of_month',
        # High-confidence signals (5)
        'high_20d_proximity', 'low_20d_proximity', 'gap_pct', 'vol_on_up_ratio', 'vol_on_down_ratio',
    ]


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI for an array of prices."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    
    if n < period + 1:
        return rsi
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Seed with SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - 100 / (1 + rs)
    else:
        rsi[period] = 100.0
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - 100 / (1 + rs)
        else:
            rsi[i + 1] = 100.0
    
    return rsi


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    result = np.full(len(arr), np.nan)
    if len(arr) >= window:
        kernel = np.ones(window) / window
        conv = np.convolve(arr, kernel, mode='full')[:len(arr)]
        result[window - 1:] = conv[window - 1:]
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def compute_direction_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Compute 35 daily technical features for direction prediction.
    
    Args:
        df: Daily OHLCV DataFrame with columns: date, open, high, low, close, volume
        symbol: Stock symbol (for info only)
    
    Returns:
        DataFrame with 35 feature columns added
    """
    if len(df) < 55:  # Need at least 50 days for SMA_50 + some buffer
        return pd.DataFrame()
    
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    n = len(c)
    
    # ========== 1. MOMENTUM (6 features) ==========
    
    for period, name in [(1, 'ret_1d'), (2, 'ret_2d'), (3, 'ret_3d'), 
                          (5, 'ret_5d'), (10, 'ret_10d'), (20, 'ret_20d')]:
        ret = np.full(n, 0.0)
        ret[period:] = (c[period:] - c[:-period]) / c[:-period] * 100
        df[name] = ret
    
    # ========== 2. TREND INDICATORS (5 features) ==========
    
    df['rsi_14'] = _rsi(c, 14)
    df['rsi_7'] = _rsi(c, 7)
    
    sma10 = _sma(c, 10)
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    
    df['dist_sma_10'] = np.where(sma10 > 0, (c - sma10) / sma10 * 100, 0)
    df['dist_sma_20'] = np.where(sma20 > 0, (c - sma20) / sma20 * 100, 0)
    df['dist_sma_50'] = np.where(sma50 > 0, (c - sma50) / sma50 * 100, 0)
    
    # ========== 3. VOLATILITY (4 features) ==========
    
    atr14 = _atr(h, l, c, 14)
    df['atr_14_pct'] = np.where(c > 0, atr14 / c * 100, 0)
    
    # Range ratio: today's range / ATR(14)
    daily_range = h - l
    df['range_ratio'] = np.where(atr14 > 0, daily_range / atr14, 1.0)
    
    # Bollinger Band position: (close - BB_lower) / (BB_upper - BB_lower)
    bb_mid = sma20
    bb_std = np.full(n, 0.0)
    for i in range(19, n):
        bb_std[i] = np.std(c[i-19:i+1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    df['bb_position'] = np.where(bb_width > 0, (c - bb_lower) / bb_width, 0.5)
    
    # Realized volatility (5-day)
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(c[1:] / c[:-1])
    rv5 = np.full(n, 0.0)
    for i in range(5, n):
        rv5[i] = np.std(log_ret[i-4:i+1]) * np.sqrt(252) * 100
    df['realized_vol_5d'] = rv5
    
    # ========== 4. VOLUME (3 features) ==========
    
    vol_sma5 = _sma(v, 5)
    vol_sma20 = _sma(v, 20)
    
    df['vol_ratio_5'] = np.where(vol_sma5 > 0, v / vol_sma5, 1.0)
    df['vol_ratio_20'] = np.where(vol_sma20 > 0, v / vol_sma20, 1.0)
    
    # Volume trend: is volume increasing? (5-day slope normalized)
    vol_trend = np.zeros(n)
    for i in range(5, n):
        vw = v[i-4:i+1]
        if np.std(vw) > 0:
            x = np.arange(5, dtype=float)
            vol_trend[i] = np.corrcoef(x, vw)[0, 1]
    df['volume_trend'] = vol_trend
    
    # ========== 5. CANDLE STRUCTURE (4 features) ==========
    
    total_range = h - l
    total_range_safe = np.where(total_range > 0, total_range, 1e-10)
    
    df['body_pct'] = np.abs(c - o) / total_range_safe  # Body as fraction of range
    df['upper_wick_pct'] = (h - np.maximum(c, o)) / total_range_safe
    df['lower_wick_pct'] = (np.minimum(c, o) - l) / total_range_safe
    df['close_position'] = (c - l) / total_range_safe  # Where in range did we close (0=low, 1=high)
    
    # ========== 6. STREAK / PATTERN (5 features) ==========
    
    # Consecutive up/down days (positive = up streak, negative = down streak)
    consec = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i-1]:
            consec[i] = max(consec[i-1], 0) + 1
        elif c[i] < c[i-1]:
            consec[i] = min(consec[i-1], 0) - 1
        else:
            consec[i] = 0
    df['consec_days'] = consec
    
    # Up days out of last 5
    up_days_5 = np.zeros(n)
    for i in range(5, n):
        up_days_5[i] = sum(1 for j in range(i-4, i+1) if c[j] > c[j-1])
    df['up_days_5'] = up_days_5
    
    # Mean reversion signal: z-score of current price vs 20-day mean
    mr_signal = np.zeros(n)
    for i in range(20, n):
        window = c[i-19:i+1]
        std = np.std(window)
        if std > 0:
            mr_signal[i] = (c[i] - np.mean(window)) / std
    df['mean_reversion'] = mr_signal
    
    # Trend strength: (SMA10 - SMA50) / SMA50 (positive = uptrend)
    df['trend_strength'] = np.where(sma50 > 0, (sma10 - sma50) / sma50 * 100, 0)
    
    # Acceleration: ret_5d - ret_10d (momentum speeding up or slowing down?)
    df['acceleration'] = df['ret_5d'] - df['ret_10d'] * 0.5  # Normalized
    
    # ========== 7. CALENDAR (3 features) ==========
    
    # Encode day of week as sin/cos (no ordinal artifact)
    if hasattr(df['date'].dt, 'dayofweek'):
        dow = df['date'].dt.dayofweek.values.astype(float)
    else:
        dow = np.zeros(n)
    df['day_of_week_sin'] = np.sin(2 * np.pi * dow / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * dow / 5)
    
    # Week of month (1-5)
    if hasattr(df['date'].dt, 'day'):
        df['week_of_month'] = ((df['date'].dt.day - 1) // 7 + 1).astype(float)
    else:
        df['week_of_month'] = 1.0
    
    # ========== 8. HIGH-CONFIDENCE SIGNALS (5 features) ==========
    
    # Proximity to 20-day high (0 = at 20d low, 1 = at 20d high)
    high_20d = np.zeros(n)
    low_20d = np.zeros(n)
    for i in range(20, n):
        h20 = np.max(h[i-19:i+1])
        l20 = np.min(l[i-19:i+1])
        rng = h20 - l20
        if rng > 0:
            high_20d[i] = (c[i] - l20) / rng
            low_20d[i] = (h20 - c[i]) / rng
    df['high_20d_proximity'] = high_20d
    df['low_20d_proximity'] = low_20d
    
    # Gap: open vs previous close
    gap = np.zeros(n)
    gap[1:] = (o[1:] - c[:-1]) / c[:-1] * 100
    df['gap_pct'] = gap
    
    # Volume on up days for last 10 days ratio, and down days
    vol_up_ratio = np.zeros(n)
    vol_down_ratio = np.zeros(n)
    for i in range(10, n):
        total_vol = np.sum(v[i-9:i+1])
        if total_vol > 0:
            up_vol = sum(v[j] for j in range(i-9, i+1) if c[j] > c[j-1])
            dn_vol = sum(v[j] for j in range(i-9, i+1) if c[j] < c[j-1])
            vol_up_ratio[i] = up_vol / total_vol
            vol_down_ratio[i] = dn_vol / total_vol
    df['vol_on_up_ratio'] = vol_up_ratio
    df['vol_on_down_ratio'] = vol_down_ratio
    
    # ========== CLEANUP ==========
    
    feature_names = get_direction_feature_names()
    for col in feature_names:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Drop first 50 rows (warmup for SMA50)
    df = df.iloc[50:].reset_index(drop=True)
    
    return df


if __name__ == '__main__':
    from ml_models.data_fetcher import load_daily_candles
    
    df = load_daily_candles("SBIN")
    if len(df) > 0:
        result = compute_direction_features(df, symbol="SBIN")
        print(f"Input: {len(df)} daily candles")
        print(f"Output: {len(result)} rows (after warmup), {len(result.columns)} columns")
        
        feature_names = get_direction_feature_names()
        print(f"\nDaily Direction Features ({len(feature_names)}):")
        for f in feature_names:
            if f in result.columns:
                vals = result[f]
                print(f"  {f:<25s} min={vals.min():>8.3f}  max={vals.max():>8.3f}  mean={vals.mean():>8.3f}")
            else:
                print(f"  {f:<25s} MISSING")
    else:
        print("No SBIN daily data. Run: python -m ml_models.data_fetcher --daily")
