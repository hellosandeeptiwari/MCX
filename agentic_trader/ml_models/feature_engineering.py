"""
FEATURE ENGINEERING — Convert raw OHLCV candles into ML features

Computes ~35 features per candle that capture:
- Momentum (RSI, rate of change, EMA cross)
- Trend (EMA alignment, slope, ADX proxy)
- Volume (ratio vs avg, accumulation/distribution)
- Volatility (ATR %, range expansion, Bollinger width)
- Structure (VWAP distance, ORB position, time-of-day)
- Daily Context (vol regime, trend regime, proximity to highs, move frequency)

All features are computed without lookahead bias.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

# Suppress noisy warnings during feature computation (safe to ignore):
# - RuntimeWarning: divide-by-zero and invalid-value in ATR/range calcs (handled by np.where guards)
# - PerformanceWarning: DataFrame fragmentation from many column inserts (unavoidable in feature pipeline)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def compute_features(df: pd.DataFrame, symbol: str = "", daily_df: pd.DataFrame = None,
                      oi_df: pd.DataFrame = None,
                      futures_oi_df: pd.DataFrame = None,
                      intraday_oi_df: pd.DataFrame = None,
                      options_oi_iv_df: pd.DataFrame = None,
                      nifty_5min_df: pd.DataFrame = None,
                      nifty_daily_df: pd.DataFrame = None,
                      sector_5min_df: pd.DataFrame = None,
                      sector_daily_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute all ML features from raw OHLCV candles.
    
    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        symbol: Optional symbol name for debugging
        daily_df: Optional daily OHLCV candles (341+ days) for context features
        oi_df: Optional OI snapshot DataFrame with columns:
               timestamp, pcr_oi, pcr_oi_change, buildup_strength,
               spot_vs_max_pain, iv_skew, atm_iv, atm_delta, call_resistance_dist
        futures_oi_df: Optional futures OI daily features DataFrame with columns:
               date, fut_oi_change_pct, fut_oi_buildup, fut_basis_pct,
               fut_oi_5d_trend, fut_vol_ratio
        nifty_5min_df: Optional NIFTY50 5-min candles for market context.
        nifty_daily_df: Optional NIFTY50 daily candles for market context.
        sector_5min_df: Optional sector index 5-min candles for sector context.
        sector_daily_df: Optional sector index daily candles for sector context.
        
    Returns:
        DataFrame with original columns + feature columns.
        Rows with insufficient history for indicators are dropped.
    """
    if len(df) < 50:
        return pd.DataFrame()
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    o = df['open'].values
    v = df['volume'].values.astype(float)
    
    # === MOMENTUM ===
    
    # RSI (14-period)
    df['rsi_14'] = _rsi(c, 14)
    
    # RSI (7-period) — faster for intraday
    df['rsi_7'] = _rsi(c, 7)
    
    # Rate of change (% change over N candles)
    df['roc_6'] = pd.Series(c).pct_change(6).values * 100    # 30-min momentum
    df['roc_12'] = pd.Series(c).pct_change(12).values * 100   # 60-min momentum
    
    # Price momentum (close vs close 6 candles ago, normalized by ATR)
    atr14 = _atr(h, l, c, 14)
    df['momentum_atr_norm'] = np.where(
        atr14 > 0, (c - np.roll(c, 6)) / atr14, 0
    )
    
    # === TREND ===
    
    # EMA 9 and 21
    ema9 = _ema(c, 9)
    ema21 = _ema(c, 21)
    df['ema_9'] = ema9
    df['ema_21'] = ema21
    
    # EMA spread (normalized)
    df['ema_spread'] = np.where(ema21 > 0, (ema9 - ema21) / ema21 * 100, 0)
    
    # EMA 9 slope (rate of change over 3 candles)
    df['ema9_slope'] = pd.Series(ema9).pct_change(3).values * 100
    
    # Price vs EMA9 (normalized distance)
    df['price_vs_ema9'] = np.where(ema9 > 0, (c - ema9) / ema9 * 100, 0)
    
    # SMA 20 (daily trend proxy)
    sma20 = _sma(c, 20)
    df['price_vs_sma20'] = np.where(sma20 > 0, (c - sma20) / sma20 * 100, 0)
    
    # ADX proxy — average directional movement (simplified)
    df['adx_proxy'] = _adx_proxy(h, l, c, 14)
    
    # === VOLUME ===
    
    # Volume ratio vs 20-period average
    vol_ma20 = _sma(v, 20)
    df['volume_ratio'] = np.where(vol_ma20 > 0, v / vol_ma20, 1.0)
    
    # Volume trend (rising or falling over last 5 candles)
    vol_sma5 = _sma(v, 5)
    vol_sma20 = _sma(v, 20)
    df['volume_trend'] = np.where(vol_sma20 > 0, vol_sma5 / vol_sma20, 1.0)
    
    # On-Balance Volume change rate
    obv = _obv(c, v)
    df['obv_slope'] = pd.Series(obv).pct_change(6).values * 100
    
    # === VOLATILITY ===
    
    # ATR as % of price
    df['atr_pct'] = np.where(c > 0, atr14 / c * 100, 0)
    
    # Range expansion ratio (current candle range vs ATR)
    candle_range = h - l
    df['range_expansion'] = np.where(atr14 > 0, candle_range / atr14, 0)
    
    # Bollinger Band width (normalized)
    sma20_v = _sma(c, 20)
    std20 = _rolling_std(c, 20)
    df['bb_width'] = np.where(sma20_v > 0, (2 * std20) / sma20_v * 100, 0)
    
    # Bollinger Band position (-1 to +1, where 0 = at SMA)
    upper_bb = sma20_v + 2 * std20
    lower_bb = sma20_v - 2 * std20
    bb_range = upper_bb - lower_bb
    df['bb_position'] = np.where(bb_range > 0, (c - lower_bb) / bb_range * 2 - 1, 0)
    
    # === STRUCTURE (INTRADAY) ===
    
    # VWAP (intraday — resets each day)
    df['vwap'] = _intraday_vwap(df)
    vwap = df['vwap'].values
    df['price_vs_vwap'] = np.where(vwap > 0, (c - vwap) / vwap * 100, 0)
    
    # Time of day (minutes since market open, normalized 0-1)
    df['time_of_day'] = _time_of_day_feature(df)
    
    # Candle body ratio (bullish/bearish strength)
    candle_body = c - o
    df['body_ratio'] = np.where(candle_range > 0, candle_body / candle_range, 0)
    
    # Upper/lower wick ratios
    df['upper_wick'] = np.where(candle_range > 0, 
                                (h - np.maximum(c, o)) / candle_range, 0)
    df['lower_wick'] = np.where(candle_range > 0,
                                (np.minimum(c, o) - l) / candle_range, 0)
    
    # Consecutive up/down candles
    df['consec_up'] = _consecutive_direction(c, o, direction='up')
    df['consec_down'] = _consecutive_direction(c, o, direction='down')
    
    # Day's cumulative return (from open)
    df['day_return_pct'] = _intraday_return(df)
    
    # === MACD (key missing momentum indicator) ===
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    # Normalize MACD histogram by price (makes it comparable across stocks)
    df['macd_hist_pct'] = np.where(c > 0, macd_hist / c * 100, 0)
    # MACD histogram slope (acceleration of momentum)
    df['macd_hist_slope'] = pd.Series(macd_hist).diff(3).values
    df['macd_hist_slope'] = np.where(c > 0, df['macd_hist_slope'].values / c * 100, 0)
    
    # === GAP OPEN (overnight gap — avg 0.58%, max 8.36%) ===
    df['gap_open_pct'] = _gap_open_feature(df)
    
    # === PRICE POSITION IN DAY'S RANGE ===
    # (close - day_low) / (day_high - day_low) → 0 to 1
    # Near 1 = at highs (momentum or reversal pending)
    # Near 0 = at lows (weakness or bounce pending)
    df['day_position'] = _intraday_position(df)
    
    # === RETURN DISTRIBUTION SHAPE (what ATR/BB miss entirely) ===
    # Rolling skewness: positive = upside tail fatter → more likely to spike up
    # Rolling kurtosis: high = fat tails → more likely to have extreme moves
    returns = pd.Series(c).pct_change().values * 100
    df['return_skew_20'] = pd.Series(returns).rolling(20, min_periods=10).skew().fillna(0).values
    df['return_kurtosis_20'] = pd.Series(returns).rolling(20, min_periods=10).kurt().fillna(0).values
    # Clip extreme kurtosis (can be 50+ with few candles)
    df['return_kurtosis_20'] = np.clip(df['return_kurtosis_20'].values, -10, 10)
    
    # === VOLUME-MOMENTUM CONFLUENCE (conviction signal) ===
    # When momentum and volume agree = institutional flow
    # momentum_atr_norm * volume_ratio: big value → high-conviction directional move
    mom_atr = df['momentum_atr_norm'].values
    vol_rat = df['volume_ratio'].values
    df['volume_momentum'] = mom_atr * np.clip(vol_rat, 0, 5)
    
    # Volume-price divergence: price moving but volume declining → weak move
    price_chg_6 = pd.Series(c).pct_change(6).values * 100
    vol_chg_6 = pd.Series(v).pct_change(6).values
    df['vol_price_divergence'] = np.where(
        np.abs(price_chg_6) > 0.01,
        np.sign(price_chg_6) * np.where(vol_chg_6 < -0.1, -1, np.where(vol_chg_6 > 0.1, 1, 0)),
        0
    )
    
    # === PRICE ACCELERATION (2nd derivative — catches momentum inflections) ===
    # 1st derivative = roc_6 (already have). 2nd derivative = change of roc
    roc_6 = df['roc_6'].values
    df['price_acceleration'] = pd.Series(roc_6).diff(3).fillna(0).values
    
    # === VOLATILITY-OF-VOLATILITY (vol regime changes) ===
    # Rising ATR% → trending regime forming. Falling → compression → breakout pending
    atr_series = pd.Series(atr14)
    df['atr_change_pct'] = atr_series.pct_change(6).fillna(0).values * 100
    df['atr_change_pct'] = np.clip(df['atr_change_pct'].values, -100, 100)
    
    # === TEMPORAL MICROSTRUCTURE (market session effects) ===
    # Day of week: encoding as sin/cos for cyclical nature
    # Monday (0) = gap plays, Friday (4) = position squaring
    dow = df['date'].dt.dayofweek.values.astype(float)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 5)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 5)
    
    # Session bucket: Indian market has distinct session profiles
    # 0=pre-market rush (9:15-9:45), 1=mid-morning, 2=lunch lull, 3=afternoon, 4=closing rush
    df['session_bucket'] = _session_bucket(df)
    
    # === DAILY CONTEXT FEATURES (from 341-day daily candles) ===
    if daily_df is not None and len(daily_df) >= 50:
        df = _add_daily_context(df, daily_df)
    else:
        # Fill with neutral defaults so model still works without daily data
        for col in _get_daily_feature_names():
            df[col] = 0.0
    
    # === OPTIONS OI CONTEXT FEATURES (from NSE F&O Bhav Copy daily data) ===
    # Historical daily option OI features: PCR, IV skew, max pain, OI buildup.
    # Downloaded via download_oi_history.py from NSE archives.
    # Each candle gets PREVIOUS day's EOD option OI (no lookahead).
    if oi_df is not None and len(oi_df) > 0:
        df = _add_options_oi_context(df, oi_df)
    else:
        for col in _get_oi_feature_names():
            df[col] = 0.0
    
    # === FUTURES OI FEATURES (from DhanHQ historical futures data) ===
    if futures_oi_df is not None and len(futures_oi_df) > 0:
        df = _add_futures_oi_context(df, futures_oi_df)
    else:
        for col in _get_futures_oi_feature_names():
            df[col] = 0.0
    
    # === INTRADAY 5-MIN FUTURES OI FEATURES (from DhanHQ intraday backfill) ===
    # True 5-min OI: each candle gets a unique OI value reflecting real-time
    # institutional position changes. Much more discriminative than daily OI
    # which was constant per day and degraded both models.
    if intraday_oi_df is not None and len(intraday_oi_df) > 0:
        df = _add_intraday_oi_context(df, intraday_oi_df)
    else:
        for col in _get_intraday_oi_feature_names():
            df[col] = 0.0
    
    # === IV REGIME FEATURES (from NSE Bhav Copy daily option chain data) ===
    # Daily IV is appropriate as a regime signal (unlike OI which is a flow signal).
    # IV percentile & IV rank are naturally daily features — they classify the
    # volatility regime (high IV = expect big moves, low IV = range-bound).
    if options_oi_iv_df is not None and len(options_oi_iv_df) > 0:
        df = _add_iv_regime_context(df, options_oi_iv_df)
    else:
        for col in _get_iv_regime_feature_names():
            df[col] = 0.0
    
    # === DIRECTION-DISCRIMINATIVE FEATURES ===
    # Features that specifically help distinguish UP from DOWN (not just MOVE vs FLAT)
    
    # 1. Chaikin Money Flow (CMF-20) — institutional accumulation vs distribution
    #    Positive = buying pressure (bullish), Negative = selling pressure (bearish)
    mfv = np.where(candle_range > 0,
                   ((c - l) - (h - c)) / candle_range * v,
                   0.0)
    cmf_sum_vol = pd.Series(v).rolling(20, min_periods=10).sum().values
    cmf_sum_mfv = pd.Series(mfv).rolling(20, min_periods=10).sum().values
    df['cmf_20'] = np.where(cmf_sum_vol > 0, cmf_sum_mfv / cmf_sum_vol, 0.0)
    
    # 2. Net buying pressure — per-candle buy/sell volume estimate
    #    (close-low)/(high-low) = fraction of range that is "buying"
    buy_pct = np.where(candle_range > 0, (c - l) / candle_range, 0.5)
    df['net_buying_pressure'] = pd.Series(buy_pct * 2 - 1).rolling(12, min_periods=6).mean().fillna(0).values  # -1 to +1
    
    # 3. Momentum alignment — do multiple timeframes agree on direction?
    #    Count how many of: roc_6, macd_hist, ema_spread, day_return_pct agree
    #    Range: -1 (all bearish) to +1 (all bullish)
    sig_roc = np.sign(df['roc_6'].values)
    sig_macd = np.sign(df['macd_hist_pct'].values)
    sig_ema = np.sign(df['ema_spread'].values)
    sig_day = np.sign(df['day_return_pct'].values)
    df['momentum_alignment'] = (sig_roc + sig_macd + sig_ema + sig_day) / 4.0
    
    # 4. Opening Range Breakout (ORB) direction
    #    After first 6 candles (30 min), is price above or below the opening range?
    df['orb_signal'] = _orb_direction(df)
    
    # 5. Previous close strength — where did yesterday's close land in yesterday's range?
    #    Near 1.0 = strong close (bullish follow-through), near 0.0 = weak close (bearish)
    df['prev_close_strength'] = _prev_close_strength(df)
    
    # 6. First-hour momentum persistence
    #    The return during first 60 min (12 candles) — if strong, rest of day follows
    df['first_hour_return'] = _first_hour_return(df)
    
    # 7. Volume-weighted direction — OBV rate sign over 12 candles
    #    Positive OBV slope = buying, negative = selling
    obv_dir = np.sign(df['obv_slope'].values)
    vol_agree = obv_dir * np.abs(df['volume_ratio'].values - 1.0)
    df['volume_direction'] = np.clip(vol_agree, -3, 3)
    
    # 8. Intraday trend consistency — fraction of last 12 candles that closed up
    #    > 0.5 = bullish, < 0.5 = bearish
    close_up = (c > o).astype(float)
    df['trend_consistency'] = pd.Series(close_up).rolling(12, min_periods=6).mean().fillna(0.5).values * 2 - 1  # -1 to +1
    
    # 9. Relative strength vs NIFTY (short-term: 3 candles = 15 min)
    #    Faster version of relative_strength for immediate momentum
    roc_3 = pd.Series(c).pct_change(3).values * 100
    df['roc_3'] = np.nan_to_num(roc_3, 0.0)
    
    # 10. Futures OI × price direction interaction
    #     When OI builds AND price moves same direction → strong conviction signal
    if 'fut_oi_buildup' in df.columns:
        df['oi_price_confirm'] = df['fut_oi_buildup'].values * np.sign(df['day_return_pct'].values)
    else:
        df['oi_price_confirm'] = 0.0
    
    # 11. Previous day's return (%) — momentum carry-over signal
    #     Yesterday's direction tends to persist (especially in first hour)
    df['prev_day_return'] = _prev_day_return(df)
    
    # 12. VWAP slope — rate of change of VWAP over 6 candles
    #     Rising VWAP = institutional buying (money-weighted price increasing)
    #     Falling VWAP = institutional selling
    vwap_vals = df['vwap'].values
    vwap_roc = np.zeros(len(vwap_vals))
    _vwap_prev = vwap_vals[:-6]
    vwap_roc[6:] = np.where(_vwap_prev > 0,
                            (vwap_vals[6:] - _vwap_prev) / _vwap_prev * 100, 0)
    df['vwap_slope'] = vwap_roc
    
    # 13. Close vs previous day's high-low range — breakout/breakdown detection
    #     > +1 = above yesterday's high (bullish breakout)
    #     < -1 = below yesterday's low (bearish breakdown)
    #     -1 to +1 = inside yesterday's range (0 = midpoint)
    df['close_vs_prev_range'] = _close_vs_prev_range(df)
    
    # 14. 2-hour momentum (fills gap between roc_12 and day_return_pct)
    df['roc_24'] = pd.Series(c).pct_change(24).values * 100
    
    # 15. Buying/selling climax — volume spike at price extremes (Wyckoff signal)
    #     Distribution top: big volume + up candle near highs → exhaustion
    #     Capitulation bottom: big volume + down candle near lows → reversal
    vol_surge = np.where(vol_ma20 > 0, v / vol_ma20, 1.0)
    candle_dir = np.sign(c - o)
    day_pos = df['day_position'].values
    df['buying_climax'] = np.where(vol_surge > 2.0,
                                    candle_dir * (day_pos - 0.5) * np.minimum(vol_surge, 5.0),
                                    0.0)
    
    # 16. ATR-normalized day return — how significant is today's move?
    #     A 1% return in a 0.5% ATR stock = 2× normal (very significant)
    #     A 1% return in a 2% ATR stock = 0.5× normal (just noise)
    day_ret = df['day_return_pct'].values
    df['atr_norm_day_return'] = np.where(df['atr_pct'].values > 0,
                                          day_ret / df['atr_pct'].values,
                                          0.0)
    
    # === CROSS-FEATURE INTERACTION FEATURES ===
    # Only keep interactions that survived feature importance pruning (>0.3%)
    
    # Net buying pressure × ATR-normalized momentum — buying conviction + price move
    df['pressure_x_mom'] = df['net_buying_pressure'].values * df['momentum_atr_norm'].values
    
    # ORB signal × volume ratio — breakout with volume confirmation
    df['orb_x_volume'] = df['orb_signal'].values * np.clip(df['volume_ratio'].values, 0, 5)
    
    # === NIFTY50 MARKET CONTEXT FEATURES ===
    if nifty_5min_df is not None and len(nifty_5min_df) > 50:
        df = _add_nifty_context(df, nifty_5min_df, nifty_daily_df)
    else:
        for col in _get_nifty_feature_names():
            df[col] = 0.0
    
    # === SECTOR INDEX CONTEXT FEATURES ===
    # Sector-relative features: how is the stock doing vs its own sector?
    # This is the #1 missing signal for directional prediction — a stock rarely
    # sustains a rally when its entire sector is falling (e.g., JINDALSTEL CE on
    # a NIFTY METAL -2% day).
    if sector_5min_df is not None and len(sector_5min_df) > 50:
        df = _add_sector_context(df, sector_5min_df, sector_daily_df)
    else:
        for col in _get_sector_feature_names():
            df[col] = 0.0
    
    # === CLEAN UP ===
    
    # Drop rows where indicators aren't ready (need ~25 candle warmup)
    feature_cols = [col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # Replace inf/nan
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Drop warmup rows (first 25 candles of each day are unreliable for some indicators)
    # But we keep them tagged — the training pipeline will handle warmup filtering
    df['_warmup'] = False
    
    return df


# === HELPER FUNCTIONS ===

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return result
    
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
    
    return result


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return result
    
    cumsum = np.cumsum(values)
    result[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index"""
    result = np.full_like(close, 50.0, dtype=float)
    if len(close) < period + 1:
        return result
    
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range"""
    result = np.full_like(close, 0.0, dtype=float)
    if len(close) < period + 1:
        return result
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    
    # Wilder smoothing
    result[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, len(close)):
        result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    
    return result


def _adx_proxy(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Simplified ADX - measures trend strength without full DI+/DI- computation."""
    atr = _atr(high, low, close, period)
    
    # Use price range / ATR ratio as trend strength proxy
    lookback = period * 2
    result = np.full_like(close, 20.0, dtype=float)
    
    for i in range(lookback, len(close)):
        window_range = np.max(close[i-lookback:i+1]) - np.min(close[i-lookback:i+1])
        if atr[i] > 0:
            # Normalize to 0-100 range
            result[i] = min(100, (window_range / atr[i]) * 3.5)
    
    return result


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation"""
    result = np.full_like(values, 0.0, dtype=float)
    for i in range(period - 1, len(values)):
        result[i] = np.std(values[i-period+1:i+1])
    return result


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume"""
    result = np.zeros_like(close, dtype=float)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            result[i] = result[i-1] + volume[i]
        elif close[i] < close[i-1]:
            result[i] = result[i-1] - volume[i]
        else:
            result[i] = result[i-1]
    return result


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP (resets each trading day)"""
    df = df.copy()
    df['_date'] = df['date'].dt.date
    df['_typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['_tp_vol'] = df['_typical'] * df['volume']
    
    vwap = pd.Series(0.0, index=df.index)
    
    for day, group in df.groupby('_date'):
        idx = group.index
        cum_tp_vol = group['_tp_vol'].cumsum()
        cum_vol = group['volume'].cumsum()
        day_vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, group['close'])
        vwap.iloc[idx] = day_vwap
    
    return vwap


def _time_of_day_feature(df: pd.DataFrame) -> np.ndarray:
    """Minutes since market open (9:15), normalized to 0-1 range.
    0 = market open, 1 = market close (15:30).
    """
    market_minutes = 375  # 9:15 to 15:30
    result = np.zeros(len(df))
    
    for i, dt in enumerate(df['date']):
        if hasattr(dt, 'hour'):
            mins = (dt.hour * 60 + dt.minute) - (9 * 60 + 15)
            result[i] = max(0, min(1, mins / market_minutes))
    
    return result


def _consecutive_direction(close: np.ndarray, open_: np.ndarray, direction: str = 'up') -> np.ndarray:
    """Count consecutive bullish or bearish candles."""
    result = np.zeros(len(close), dtype=float)
    
    for i in range(1, len(close)):
        if direction == 'up' and close[i] > open_[i]:
            result[i] = result[i-1] + 1
        elif direction == 'down' and close[i] < open_[i]:
            result[i] = result[i-1] + 1
        else:
            result[i] = 0
    
    return result


def _gap_open_feature(df: pd.DataFrame) -> np.ndarray:
    """Overnight gap: (today's open - yesterday's close) / yesterday's close * 100.
    
    For first candle of each day: gap from previous day's last close.
    For subsequent intraday candles: 0 (gap only applies at open).
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    prev_close = None
    for day, group in df_copy.groupby('_date', sort=True):
        idx = group.index
        day_open = group['open'].iloc[0]
        
        if prev_close is not None and prev_close > 0:
            gap_pct = (day_open - prev_close) / prev_close * 100
            # Apply gap to ALL candles of the day (it's a daily-level feature)
            result[idx] = gap_pct
        
        prev_close = group['close'].iloc[-1]
    
    return result


def _session_bucket(df: pd.DataFrame) -> np.ndarray:
    """Indian market session bucket (normalized 0-1).
    
    0.0 = Opening rush (9:15-9:45)  — gap plays, high volatility
    0.25 = Mid-morning (9:45-11:30) — trend establishment
    0.50 = Lunch lull (11:30-13:30) — low volume, choppy
    0.75 = Afternoon (13:30-14:45)  — institutional flow
    1.0 = Closing rush (14:45-15:30) — position squaring, high volume
    """
    result = np.full(len(df), 0.5)  # default mid-day
    for i, dt in enumerate(df['date']):
        if not hasattr(dt, 'hour'):
            continue
        mins = dt.hour * 60 + dt.minute
        if mins < 9 * 60 + 45:      # 9:15-9:45
            result[i] = 0.0
        elif mins < 11 * 60 + 30:    # 9:45-11:30
            result[i] = 0.25
        elif mins < 13 * 60 + 30:    # 11:30-13:30
            result[i] = 0.50
        elif mins < 14 * 60 + 45:    # 13:30-14:45
            result[i] = 0.75
        else:                         # 14:45-15:30
            result[i] = 1.0
    return result


def _intraday_position(df: pd.DataFrame) -> np.ndarray:
    """Price position within the day's range so far: (close - day_low) / (day_high - day_low).
    
    Returns 0-1 where 1 = at day's high, 0 = at day's low.
    Uses rolling day high/low up to current candle (no lookahead).
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    for day, group in df_copy.groupby('_date'):
        idx = group.index
        highs = group['high'].values
        lows = group['low'].values
        closes = group['close'].values
        
        running_high = np.maximum.accumulate(highs)
        running_low = np.minimum.accumulate(lows)
        day_range = running_high - running_low
        
        pos = np.where(day_range > 0, (closes - running_low) / day_range, 0.5)
        result[idx] = pos
    
    return result


def _intraday_return(df: pd.DataFrame) -> np.ndarray:
    """Cumulative return from day's open price."""
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    for day, group in df_copy.groupby('_date'):
        idx = group.index
        day_open = group['open'].iloc[0]
        if day_open > 0:
            result[idx] = (group['close'].values - day_open) / day_open * 100
    
    return result


def _get_daily_feature_names() -> list:
    """Feature names from daily context."""
    return [
        'daily_vol_regime',       # ATR% percentile vs last 100 days (0-1)
        'daily_trend_strength',   # 20d EMA slope normalized
        'daily_trend_direction',  # Price vs 50d EMA (%)
        'daily_dist_from_high',   # Distance from 100d high (%)
        'daily_dist_from_low',    # Distance from 100d low (%)
        'daily_range_pct_20d',    # 20d price range as % of price
        'daily_move_freq_10d',    # Fraction of last 10 days with >1% move
        'daily_move_freq_20d',    # Fraction of last 20 days with >1% move
        'daily_volume_regime',    # Today volume vs 20d avg
        'daily_rsi_14',           # Daily RSI(14)
    ]


def _get_oi_feature_names() -> list:
    """Feature names from options OI context (NSE F&O Bhav Copy daily data).
    
    These map to per-stock daily features computed from NSE option chain data
    via download_oi_history.py. Each 5-min candle gets the PREVIOUS day's
    EOD option OI snapshot (no lookahead).
    """
    return [
        'oi_pcr',                 # Put-Call ratio by OI (>1 bearish, <1 bullish)
        'oi_pcr_change',          # PCR computed from OI change direction
        'oi_buildup_strength',    # -1 (short buildup) to +1 (long buildup)
        'oi_spot_vs_max_pain',    # (spot - max_pain) / spot * 100
        'oi_iv_skew',             # ATM put IV - call IV (>0 bearish fear)
        'oi_atm_iv',              # ATM average IV (high = expected big move)
        'oi_call_resistance_dist', # Distance from top-call-OI strike (% of spot)
        'oi_put_support_dist',    # Distance from top-put-OI strike (% of spot)
    ]


def _get_futures_oi_feature_names() -> list:
    """Feature names from DhanHQ historical futures OI data."""
    return [
        'fut_oi_change_pct',   # Daily % change in futures OI
        'fut_oi_buildup',      # +1=long buildup, -1=short buildup, +0.5=short cover, -0.5=long unwind
        'fut_basis_pct',       # Futures premium/discount vs spot (%)
        'fut_oi_5d_trend',     # 5-day OI trend (%)
        'fut_vol_ratio',       # Futures volume / Equity volume
    ]


def _get_intraday_oi_feature_names() -> list:
    """Feature names from 5-min intraday futures OI (DhanHQ backfill).
    
    Unlike daily OI (same value for all 75 candles/day = no signal),
    these have a UNIQUE OI value per 5-min candle reflecting real-time
    institutional position changes.
    """
    return [
        'oi_change_5m',           # OI % change per candle (building/unwinding)
        'oi_velocity_6',          # Rate of OI change over last 6 candles (30 min)
        'oi_session_change',      # Cumulative OI % change since day open
        'oi_vs_volume',           # OI change / volume (conviction measure, higher = stronger)
        'oi_acceleration',        # 2nd derivative: is OI buildup accelerating or decelerating?
        'oi_regime',              # OI relative to 20-candle MA: >1 = above avg, <1 = below
    ]


def _get_iv_regime_feature_names() -> list:
    """Feature names from daily IV regime context (NSE Bhav Copy).
    
    Daily IV is naturally a slow-moving regime signal - it classifies the
    volatility environment, not per-candle flow. High IV rank means options
    are expensive (expect big moves), low IV rank = range-bound.
    """
    return [
        'iv_rank_30d',           # IV Rank: (current IV - 30d min) / (30d max - 30d min)
        'iv_percentile_30d',     # IV Percentile: % of last 30 days with IV below current
        'iv_skew_zscore',        # IV skew z-score: how extreme is today's put-call IV gap?
        'iv_regime',             # Categorical: 0=low (<25pct), 1=mid, 2=high (>75pct)
        'pcr_oi_regime',         # PCR regime: normalized PCR vs 30d history
    ]


def _get_nifty_feature_names() -> list:
    """Feature names derived from NIFTY50 index (market context)."""
    return [
        'nifty_roc_6',            # NIFTY momentum (same timeframe as stock roc_6)
        'nifty_rsi_14',           # NIFTY RSI-14
        'nifty_bb_position',      # NIFTY Bollinger Band position (0-1)
        'nifty_ema9_slope',       # NIFTY EMA-9 slope (trend direction)
        'nifty_atr_pct',          # NIFTY ATR% (market volatility)
        'relative_strength',      # Stock roc_6 minus NIFTY roc_6 (ALPHA signal)
        'nifty_daily_trend',      # NIFTY daily: price vs EMA-50 (%)
        'nifty_daily_rsi',        # NIFTY daily RSI-14
    ]


# ══════════════════════════════════════════════════════════════
#  STOCK-TO-SECTOR MAPPING
# ══════════════════════════════════════════════════════════════
# Maps each F&O stock to its NIFTY sector index.
# The parquet filename convention is SECTOR_{NAME}.parquet
# (e.g., SECTOR_METAL.parquet for NIFTY METAL index candles).
# Stocks not in this map get zeros for all sector features.

STOCK_SECTOR_MAP = {
    # METALS
    'TATASTEEL': 'METAL', 'JSWSTEEL': 'METAL', 'JINDALSTEL': 'METAL',
    'HINDALCO': 'METAL', 'VEDL': 'METAL', 'NMDC': 'METAL',
    'NATIONALUM': 'METAL', 'HINDZINC': 'METAL', 'SAIL': 'METAL',
    'HINDCOPPER': 'METAL', 'COALINDIA': 'METAL',
    # IT
    'INFY': 'IT', 'TCS': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT',
    'TECHM': 'IT', 'LTIM': 'IT', 'KPITTECH': 'IT', 'COFORGE': 'IT',
    'MPHASIS': 'IT', 'PERSISTENT': 'IT',
    # BANKS
    'SBIN': 'BANK', 'HDFCBANK': 'BANK', 'ICICIBANK': 'BANK',
    'AXISBANK': 'BANK', 'KOTAKBANK': 'BANK', 'BANKBARODA': 'BANK',
    'PNB': 'BANK', 'IDFCFIRSTB': 'BANK', 'INDUSINDBK': 'BANK',
    'FEDERALBNK': 'BANK', 'RBLBANK': 'BANK', 'AUBANK': 'BANK',
    'CANBK': 'BANK', 'UNIONBANK': 'BANK',
    # AUTO
    'MARUTI': 'AUTO', 'TATAMOTORS': 'AUTO', 'M&M': 'AUTO',
    'BAJAJ-AUTO': 'AUTO', 'HEROMOTOCO': 'AUTO', 'EICHERMOT': 'AUTO',
    'ASHOKLEY': 'AUTO', 'BHARATFORG': 'AUTO', 'MOTHERSON': 'AUTO',
    'BALKRISIND': 'AUTO',
    # PHARMA
    'SUNPHARMA': 'PHARMA', 'CIPLA': 'PHARMA', 'DRREDDY': 'PHARMA',
    'DIVISLAB': 'PHARMA', 'AUROPHARMA': 'PHARMA', 'BIOCON': 'PHARMA',
    'LUPIN': 'PHARMA', 'APOLLOHOSP': 'PHARMA', 'MAXHEALTH': 'PHARMA',
    'LALPATHLAB': 'PHARMA',
    # ENERGY
    'RELIANCE': 'ENERGY', 'ONGC': 'ENERGY', 'NTPC': 'ENERGY',
    'POWERGRID': 'ENERGY', 'TATAPOWER': 'ENERGY', 'ADANIENT': 'ENERGY',
    'ADANIGREEN': 'ENERGY', 'BPCL': 'ENERGY', 'IOC': 'ENERGY',
    'GAIL': 'ENERGY',
    # FMCG
    'ITC': 'FMCG', 'HINDUNILVR': 'FMCG', 'NESTLEIND': 'FMCG',
    'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'GODREJCP': 'FMCG',
    'MARICO': 'FMCG', 'COLPAL': 'FMCG', 'TATACONSUM': 'FMCG',
    'VBL': 'FMCG',
    # REALTY
    'DLF': 'REALTY', 'GODREJPROP': 'REALTY', 'OBEROIRLTY': 'REALTY',
    'PRESTIGE': 'REALTY', 'PHOENIXLTD': 'REALTY', 'BRIGADE': 'REALTY',
    'LODHA': 'REALTY', 'SOBHA': 'REALTY',
    # INFRA
    'LT': 'INFRA', 'ADANIPORTS': 'INFRA', 'ULTRACEMCO': 'INFRA',
    'GRASIM': 'INFRA', 'SHREECEM': 'INFRA', 'AMBUJACEM': 'INFRA',
    'ACC': 'INFRA', 'SIEMENS': 'INFRA', 'ABB': 'INFRA',
    'BEL': 'INFRA', 'HAL': 'INFRA', 'BHEL': 'INFRA',
    # CONGLOMERATES (use ENERGY as proxy — Reliance/Adani)
    'BAJFINANCE': 'BANK', 'TITAN': 'FMCG',
    'IDEA': 'IT',  # Telecom — closest to IT
}


def get_sector_for_symbol(symbol: str) -> str:
    """Get the sector name for a stock symbol. Returns empty string if unknown."""
    return STOCK_SECTOR_MAP.get(symbol, '')


def _get_sector_feature_names() -> list:
    """Feature names derived from sector index context."""
    return [
        'sector_roc_6',           # Sector index 30-min momentum (%)
        'sector_rsi_14',          # Sector index RSI-14
        'sector_bb_position',     # Sector index Bollinger Band position (0-1)
        'sector_ema9_slope',      # Sector index EMA-9 slope (trend direction)
        'sector_atr_pct',         # Sector index ATR% (sector volatility)
        'sector_relative',        # Stock roc_6 minus sector roc_6 (sector alpha)
        'sector_daily_trend',     # Sector daily: price vs EMA-50 (%)
        'sector_daily_rsi',       # Sector daily RSI-14
    ]


def _get_direction_feature_names() -> list:
    """Feature names that specifically discriminate UP from DOWN."""
    return [
        'cmf_20',                  # Chaikin Money Flow (20-period)
        'net_buying_pressure',     # Buy/sell pressure from candle structure
        'momentum_alignment',      # Multi-timeframe direction agreement (-1 to +1)
        'orb_signal',              # Opening Range Breakout direction
        'prev_close_strength',     # Previous day close position in range
        'first_hour_return',       # First 60 min cumulative return
        'volume_direction',        # OBV direction × volume ratio
        'trend_consistency',       # Fraction of recent candles closing up (-1 to +1)
        'roc_3',                   # Short-term 15 min momentum
        'oi_price_confirm',        # Futures OI buildup × price direction
        'prev_day_return',         # Previous day's full return (momentum carry)
        'vwap_slope',              # VWAP rate of change (institutional direction)
        'close_vs_prev_range',     # Close position vs prior day's range
        'roc_24',                  # 2-hour momentum
        'buying_climax',           # Volume spike at price extremes (Wyckoff)
        'atr_norm_day_return',     # Day return normalized by ATR
    ]


def _get_structure_feature_names() -> list:
    """Structural features (BOS/Sweep removed — 0% importance in both models)."""
    return []


def _get_interaction_feature_names() -> list:
    """Cross-feature interaction names — only those with >0.3% importance."""
    return [
        'pressure_x_mom',          # Net buying pressure × ATR-norm momentum
        'orb_x_volume',            # ORB signal × volume ratio
    ]


def get_feature_names() -> list:
    """Return the list of feature column names (for model training)."""
    return [
        'rsi_14', 'rsi_7',
        'roc_6', 'roc_12', 'momentum_atr_norm',
        'ema_spread', 'ema9_slope', 'price_vs_ema9', 'price_vs_sma20', 'adx_proxy',
        'volume_ratio', 'volume_trend', 'obv_slope',
        'atr_pct', 'range_expansion', 'bb_width', 'bb_position',
        'price_vs_vwap', 'time_of_day',
        'body_ratio', 'upper_wick', 'lower_wick',
        'consec_up', 'consec_down',
        'day_return_pct',
        'macd_hist_pct', 'macd_hist_slope',
        'gap_open_pct',
        'day_position',
        # Distribution shape (what ATR/BB miss)
        'return_skew_20', 'return_kurtosis_20',
        # Volume-momentum interaction
        'volume_momentum', 'vol_price_divergence',
        # 2nd derivative
        'price_acceleration',
        # Volatility regime change
        'atr_change_pct',
        # Temporal microstructure
        'dow_sin', 'dow_cos', 'session_bucket',
    ] + _get_direction_feature_names() + _get_interaction_feature_names() + _get_structure_feature_names() + _get_daily_feature_names() + _get_futures_oi_feature_names() + _get_nifty_feature_names()
    # NOTE: Daily OI features (_get_oi_feature_names) degraded BOTH models:
    #   GMM v5.1: UP AUROC 0.6445→0.5608, DOWN AUROC 0.6446→0.6229
    #   XGBoost v5.1: Model collapsed to all-FLAT (UP F1=0, DOWN F1=0, macro_F1=0.294)
    # Cause: daily OI was forward-filled to 75 identical candles/day = zero intraday signal.
    #
    # Intraday OI + IV Regime features (_get_intraday_oi_feature_names + _get_iv_regime_feature_names)
    # also collapsed XGBoost to all-FLAT (macro_F1=0.294, UP/DOWN F1=0.000).
    # Root cause: DhanHQ intraday OI only covers ~90 days (Dec 2025-Feb 2026) while
    # training data spans ~250 days (Sep 2025-Feb 2026). 64% of training samples had
    # all-zero intraday OI features, creating a dominant "zero → FLAT" pattern that
    # overwhelmed the directional signal. iv_regime did reach top 10 features (0.026 importance)
    # but wasn't sufficient to overcome the noise from sparse intraday OI coverage.
    #
    # Sector features (_get_sector_feature_names) also excluded — no signal in either model.


def _orb_direction(df: pd.DataFrame) -> np.ndarray:
    """Opening Range Breakout direction signal.
    
    After the first 6 candles (30 min) of each day, measures whether the current
    price is above or below the Opening Range (first 30 min high/low).
    
    Returns:
        Values from -1 (below ORB low, bearish) to +1 (above ORB high, bullish).
        During the first 30 min, returns 0 (ORB not established yet).
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    for day, group in df_copy.groupby('_date'):
        idx = group.index
        if len(group) < 7:
            continue
        
        # Opening range: high/low of first 6 candles (30 min)
        orb_high = group['high'].iloc[:6].max()
        orb_low = group['low'].iloc[:6].min()
        orb_range = orb_high - orb_low
        
        if orb_range <= 0:
            continue
        
        # After first 6 candles: position relative to ORB
        for i in range(6, len(group)):
            pos = idx[i]
            close_val = group['close'].iloc[i]
            if close_val > orb_high:
                # Above ORB high — bullish breakout
                result[pos] = min(1.0, (close_val - orb_high) / orb_range)
            elif close_val < orb_low:
                # Below ORB low — bearish breakout
                result[pos] = max(-1.0, (close_val - orb_low) / orb_range)
            else:
                # Inside ORB range
                result[pos] = (close_val - (orb_high + orb_low) / 2) / (orb_range / 2)
    
    return result


def _prev_close_strength(df: pd.DataFrame) -> np.ndarray:
    """Previous day's close position within the day's range.
    
    Near 1.0 = closed at the high (strong, bullish follow-through likely).
    Near 0.0 = closed at the low (weak, bearish follow-through likely).
    Uses previous day to avoid lookahead.
    """
    result = np.full(len(df), 0.5)  # Default neutral
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    prev_strength = 0.5
    prev_date = None
    
    for day, group in sorted(df_copy.groupby('_date'), key=lambda x: x[0]):
        idx = group.index
        
        # Apply previous day's strength to all candles of current day
        if prev_date is not None:
            result[idx] = prev_strength
        
        # Compute this day's close strength
        day_high = group['high'].max()
        day_low = group['low'].min()
        day_close = group['close'].iloc[-1]
        day_range = day_high - day_low
        
        if day_range > 0:
            prev_strength = (day_close - day_low) / day_range
        else:
            prev_strength = 0.5
        
        prev_date = day
    
    return result


def _first_hour_return(df: pd.DataFrame) -> np.ndarray:
    """Cumulative return during the first 60 minutes (12 candles) of each day.
    
    Applied to all subsequent candles of the day (after the first hour).
    During the first hour itself, shows the running return.
    Strong first-hour moves tend to persist.
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    for day, group in df_copy.groupby('_date'):
        idx = group.index
        day_open = group['open'].iloc[0]
        
        if day_open <= 0 or len(group) < 2:
            continue
        
        # First hour return = close at candle 12 vs open
        first_hour_candles = min(12, len(group))
        close_at_1hr = group['close'].iloc[first_hour_candles - 1]
        fh_return = (close_at_1hr - day_open) / day_open * 100
        
        # During first hour: running return
        for i in range(first_hour_candles):
            pos = idx[i]
            result[pos] = (group['close'].iloc[i] - day_open) / day_open * 100
        
        # After first hour: fixed first-hour return value
        for i in range(first_hour_candles, len(group)):
            pos = idx[i]
            result[pos] = fh_return
    
    return result


def _prev_day_return(df: pd.DataFrame) -> np.ndarray:
    """Previous day's open-to-close return (%).
    
    For each candle today, returns yesterday's full-day return.
    Captures momentum carry-over: strong up/down days tend to follow through.
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    prev_return = 0.0
    prev_date = None
    
    for day, group in sorted(df_copy.groupby('_date'), key=lambda x: x[0]):
        idx = group.index
        if prev_date is not None:
            result[idx] = prev_return
        
        day_open = group['open'].iloc[0]
        day_close = group['close'].iloc[-1]
        if day_open > 0:
            prev_return = (day_close - day_open) / day_open * 100
        else:
            prev_return = 0.0
        prev_date = day
    
    return result


def _close_vs_prev_range(df: pd.DataFrame) -> np.ndarray:
    """Current close position relative to previous day's high-low range.
    
    Returns:
        > +1: above yesterday's high (bullish breakout)
        < -1: below yesterday's low (bearish breakdown)
        Between -1 and +1: inside yesterday's range (0 = midpoint)
    Capped at ±3 to avoid extreme outliers.
    """
    result = np.zeros(len(df))
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    
    prev_high = None
    prev_low = None
    prev_date = None
    
    for day, group in sorted(df_copy.groupby('_date'), key=lambda x: x[0]):
        idx = group.index
        if prev_date is not None and prev_high is not None and prev_low is not None:
            prev_range = prev_high - prev_low
            if prev_range > 0:
                prev_mid = (prev_high + prev_low) / 2.0
                closes = group['close'].values
                above = closes > prev_high
                below = closes < prev_low
                inside = ~above & ~below
                vals = np.zeros(len(closes))
                vals[above] = np.minimum(3.0, (closes[above] - prev_high) / prev_range + 1.0)
                vals[below] = np.maximum(-3.0, (closes[below] - prev_low) / prev_range - 1.0)
                vals[inside] = (closes[inside] - prev_mid) / (prev_range / 2.0)
                result[idx] = vals
        
        prev_high = group['high'].max()
        prev_low = group['low'].min()
        prev_date = day
    
    return result


def _add_daily_context(intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily-derived context features onto each 5-min candle.
    
    For each trading day in intraday_df, computes features from daily_df
    using ONLY data available up to the PREVIOUS day (no lookahead).
    
    Args:
        intraday_df: 5-min candle DataFrame (already has date column)
        daily_df: Daily OHLCV DataFrame (341+ days)
    
    Returns:
        intraday_df with 10 new daily context columns added
    """
    daily = daily_df.copy().sort_values('date').reset_index(drop=True)
    daily['_date'] = daily['date'].dt.date if hasattr(daily['date'].dt, 'date') else daily['date'].apply(lambda x: x.date() if hasattr(x, 'date') else x)
    
    # Pre-compute daily indicators
    dc = daily['close'].values
    dh = daily['high'].values
    dl = daily['low'].values
    dv = daily['volume'].values.astype(float)
    
    daily_atr = _atr(dh, dl, dc, 14)
    daily_atr_pct = np.where(dc > 0, daily_atr / dc * 100, 0)
    daily_ema20 = _ema(dc, 20)
    daily_ema50 = _ema(dc, 50)
    daily_rsi = _rsi(dc, 14)
    daily_vol_ma20 = _sma(dv, 20)
    
    # Daily returns for move frequency
    daily_returns = np.zeros(len(dc))
    daily_returns[1:] = np.abs(np.diff(dc) / dc[:-1] * 100)
    
    # Build lookup: date → context features
    context_by_date = {}
    
    for i in range(50, len(daily)):  # Need 50 days warmup
        d = daily['_date'].iloc[i]
        
        # Use data up to PREVIOUS day (index i-1) to avoid lookahead
        prev_idx = i - 1
        
        # 1. Volatility regime: ATR% percentile over last 100 days
        lookback = min(100, prev_idx)
        atr_window = daily_atr_pct[max(0, prev_idx - lookback):prev_idx + 1]
        vol_regime = np.searchsorted(np.sort(atr_window), daily_atr_pct[prev_idx]) / max(len(atr_window), 1)
        
        # 2. Trend strength: EMA20 slope (% change over 5 days)
        if prev_idx >= 5 and daily_ema20[prev_idx] > 0 and daily_ema20[prev_idx - 5] > 0:
            trend_strength = (daily_ema20[prev_idx] / daily_ema20[prev_idx - 5] - 1) * 100
        else:
            trend_strength = 0.0
        
        # 3. Trend direction: price vs 50d EMA
        if daily_ema50[prev_idx] > 0:
            trend_dir = (dc[prev_idx] - daily_ema50[prev_idx]) / daily_ema50[prev_idx] * 100
        else:
            trend_dir = 0.0
        
        # 4-5. Distance from 100d high/low
        high_100d = np.max(dh[max(0, prev_idx - 99):prev_idx + 1])
        low_100d = np.min(dl[max(0, prev_idx - 99):prev_idx + 1])
        dist_from_high = (dc[prev_idx] - high_100d) / high_100d * 100 if high_100d > 0 else 0
        dist_from_low = (dc[prev_idx] - low_100d) / low_100d * 100 if low_100d > 0 else 0
        
        # 6. 20d price range as % of price
        high_20d = np.max(dh[max(0, prev_idx - 19):prev_idx + 1])
        low_20d = np.min(dl[max(0, prev_idx - 19):prev_idx + 1])
        range_20d = (high_20d - low_20d) / dc[prev_idx] * 100 if dc[prev_idx] > 0 else 0
        
        # 7-8. Move frequency (fraction of days with >1% absolute move)
        returns_10d = daily_returns[max(0, prev_idx - 9):prev_idx + 1]
        returns_20d = daily_returns[max(0, prev_idx - 19):prev_idx + 1]
        move_freq_10d = np.mean(returns_10d > 1.0) if len(returns_10d) > 0 else 0
        move_freq_20d = np.mean(returns_20d > 1.0) if len(returns_20d) > 0 else 0
        
        # 9. Volume regime
        vol_regime_v = dv[prev_idx] / daily_vol_ma20[prev_idx] if daily_vol_ma20[prev_idx] > 0 else 1.0
        
        # 10. Daily RSI
        d_rsi = daily_rsi[prev_idx]
        
        context_by_date[d] = {
            'daily_vol_regime': float(vol_regime),
            'daily_trend_strength': float(trend_strength),
            'daily_trend_direction': float(trend_dir),
            'daily_dist_from_high': float(dist_from_high),
            'daily_dist_from_low': float(dist_from_low),
            'daily_range_pct_20d': float(range_20d),
            'daily_move_freq_10d': float(move_freq_10d),
            'daily_move_freq_20d': float(move_freq_20d),
            'daily_volume_regime': float(vol_regime_v),
            'daily_rsi_14': float(d_rsi),
        }
    
    # Forward-fill: for intraday dates AFTER the last daily date, use the
    # latest available daily context.  Daily candles update EOD so today's
    # live candles won't have today's daily yet — they should use yesterday's.
    _daily_dates_sorted = sorted(context_by_date.keys()) if context_by_date else []
    _last_daily_ctx = context_by_date[_daily_dates_sorted[-1]] if _daily_dates_sorted else None
    
    def _daily_lookup(d, c):
        ctx = context_by_date.get(d)
        if ctx:
            return ctx.get(c, 0.0)
        # Forward-fill: date beyond daily coverage → use latest
        if _last_daily_ctx is not None and _daily_dates_sorted and d > _daily_dates_sorted[-1]:
            return _last_daily_ctx.get(c, 0.0)
        return 0.0
    
    # Map daily context onto each 5-min candle
    intraday_df = intraday_df.copy()
    intraday_df['_date'] = intraday_df['date'].dt.date
    
    for col in _get_daily_feature_names():
        intraday_df[col] = intraday_df['_date'].map(
            lambda d, c=col: _daily_lookup(d, c)
        )
    
    intraday_df.drop(columns=['_date'], inplace=True, errors='ignore')
    
    # Replace inf/nan
    for col in _get_daily_feature_names():
        intraday_df[col] = intraday_df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return intraday_df


def _add_oi_context(intraday_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OI-derived context features onto each 5-min candle.
    
    OI snapshots are taken periodically (every 3-5 min) during market hours.
    Each candle gets the most recent OI snapshot that is <= candle timestamp
    (no lookahead).
    
    Args:
        intraday_df: 5-min candle DataFrame (with date column)
        oi_df: OI snapshot DataFrame with columns:
            timestamp (datetime), pcr_oi, pcr_oi_change, buildup_strength,
            spot_vs_max_pain, iv_skew, atm_iv, atm_delta, call_resistance_dist
    
    Returns:
        intraday_df with 8 new OI context columns added
    """
    result = intraday_df.copy()
    oi = oi_df.copy()
    
    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(oi['timestamp']):
        oi['timestamp'] = pd.to_datetime(oi['timestamp'])
    
    oi = oi.sort_values('timestamp').reset_index(drop=True)
    
    oi_feature_cols = _get_oi_feature_names()
    
    # Map OI columns to expected feature names
    col_map = {
        'pcr_oi': 'oi_pcr',
        'pcr_oi_change': 'oi_pcr_change',
        'buildup_strength': 'oi_buildup_strength',
        'spot_vs_max_pain': 'oi_spot_vs_max_pain',
        'iv_skew': 'oi_iv_skew',
        'atm_iv': 'oi_atm_iv',
        'atm_delta': 'oi_atm_delta',
        'call_resistance_dist': 'oi_call_resistance_dist',
    }
    
    # Reverse map: feature_name -> oi_df column name
    rev_map = {v: k for k, v in col_map.items()}
    
    # Use merge_asof for efficient time-based join (no lookahead)
    result = result.sort_values('date').reset_index(drop=True)
    
    # Prepare OI data for merge_asof
    oi_merge = oi[['timestamp'] + [c for c in rev_map.values() if c in oi.columns]].copy()
    oi_merge = oi_merge.rename(columns=col_map)
    oi_merge = oi_merge.rename(columns={'timestamp': 'date'})
    oi_merge = oi_merge.sort_values('date')
    
    # Merge: each candle gets the latest OI snapshot at or before its timestamp
    available_oi_cols = [c for c in oi_feature_cols if c in oi_merge.columns]
    
    if available_oi_cols:
        # Normalize timezones to avoid merge_asof tz mismatch
        _left = result[['date']].copy()
        _right = oi_merge[['date'] + available_oi_cols].copy()
        if _left['date'].dt.tz is not None:
            _left['date'] = _left['date'].dt.tz_localize(None)
        if _right['date'].dt.tz is not None:
            _right['date'] = _right['date'].dt.tz_localize(None)
        merged = pd.merge_asof(
            _left,
            _right,
            on='date',
            direction='backward',
        )
        for col in available_oi_cols:
            result[col] = merged[col].values
    
    # Fill any missing OI features with 0
    for col in oi_feature_cols:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


def _add_options_oi_context(intraday_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily options OI features onto each 5-min candle.
    
    For each trading day in intraday_df, uses the PREVIOUS day's option OI
    features to avoid lookahead bias (same logic as _add_futures_oi_context).
    
    Data comes from NSE F&O Bhav Copy daily data downloaded via
    download_oi_history.py. Contains per-stock: PCR, IV skew, max pain,
    OI buildup strength, call resistance, put support.
    
    Args:
        intraday_df: 5-min candle DataFrame (already has 'date' column)
        oi_df: Daily options OI DataFrame with columns:
            trade_date, symbol, pcr_oi, pcr_oi_change, oi_buildup_strength,
            spot_vs_max_pain, iv_skew, atm_iv, call_resistance_dist, put_support_dist
    
    Returns:
        intraday_df with 8 new option OI context columns added
    """
    oi_features = _get_oi_feature_names()
    
    if oi_df is None or len(oi_df) == 0:
        for col in oi_features:
            intraday_df[col] = 0.0
        return intraday_df
    
    oi = oi_df.copy()
    
    # Normalize the date column
    if 'trade_date' in oi.columns:
        oi['_date'] = pd.to_datetime(oi['trade_date']).dt.date
    elif 'date' in oi.columns:
        oi['_date'] = pd.to_datetime(oi['date']).dt.date
    else:
        for col in oi_features:
            intraday_df[col] = 0.0
        return intraday_df
    
    oi = oi.sort_values('_date').reset_index(drop=True)
    
    # Map downloaded column names to feature names
    col_map = {
        'pcr_oi': 'oi_pcr',
        'pcr_oi_change': 'oi_pcr_change',
        'oi_buildup_strength': 'oi_buildup_strength',
        'spot_vs_max_pain': 'oi_spot_vs_max_pain',
        'iv_skew': 'oi_iv_skew',
        'atm_iv': 'oi_atm_iv',
        'call_resistance_dist': 'oi_call_resistance_dist',
        'put_support_dist': 'oi_put_support_dist',
    }
    
    # Build lookup: for each date, store the PREVIOUS day's features (no lookahead)
    context_by_date = {}
    sorted_oi_dates = oi['_date'].unique()
    
    for i in range(1, len(oi)):
        current_date = oi['_date'].iloc[i]
        prev_row = oi.iloc[i - 1]
        
        feat_dict = {}
        for src_col, feat_name in col_map.items():
            if src_col in oi.columns and pd.notna(prev_row.get(src_col)):
                feat_dict[feat_name] = float(prev_row[src_col])
            else:
                feat_dict[feat_name] = 0.0
        context_by_date[current_date] = feat_dict
    
    # Forward-fill: for dates after last OI date, use latest row
    _last_oi_context = None
    if len(oi) > 0:
        _last_row = oi.iloc[-1]
        _last_oi_context = {
            feat_name: float(_last_row.get(src_col, 0)) if pd.notna(_last_row.get(src_col)) else 0.0
            for src_col, feat_name in col_map.items()
        }
    
    # Map onto each 5-min candle
    result = intraday_df.copy()
    result['_date'] = result['date'].dt.date
    
    def _lookup(d, c):
        val = context_by_date.get(d, {}).get(c)
        if val is not None:
            return val
        # Forward-fill for dates beyond OI coverage
        if _last_oi_context is not None and len(sorted_oi_dates) > 0 and d > sorted_oi_dates[-1]:
            return _last_oi_context.get(c, 0.0)
        return 0.0
    
    for col in oi_features:
        result[col] = result['_date'].map(lambda d, c=col: _lookup(d, c))
    
    result.drop(columns=['_date'], inplace=True, errors='ignore')
    
    # Replace inf/nan
    for col in oi_features:
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


def _add_nifty_context(intraday_df: pd.DataFrame, nifty_5min_df: pd.DataFrame,
                       nifty_daily_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge NIFTY50 market context features onto each 5-min candle.
    
    Uses time-aligned NIFTY candles (merge_asof) to add market direction,
    momentum, and relative strength features. These are the #1 missing signal
    for directional prediction — individual stocks are 60-80% correlated with
    the index.
    
    The killer feature: relative_strength = stock_roc - nifty_roc
    This isolates genuine stock-specific alpha from market beta.
    
    Args:
        intraday_df: 5-min stock candles (with 'date', 'close', and 'roc_6' columns)
        nifty_5min_df: NIFTY50 5-min OHLCV candles
        nifty_daily_df: Optional NIFTY50 daily candles for daily context
    
    Returns:
        intraday_df with 8 new NIFTY market context columns
    """
    nifty_feat_names = _get_nifty_feature_names()
    result = intraday_df.copy()
    
    # ── Compute NIFTY 5-min features ──
    nf = nifty_5min_df.copy().sort_values('date').reset_index(drop=True)
    nc = nf['close'].values
    nh = nf['high'].values
    nl = nf['low'].values
    
    # NIFTY momentum (same as stock roc_6: % change over 6 candles = 30 min)
    nifty_roc_6 = np.zeros(len(nc))
    nifty_roc_6[6:] = (nc[6:] - nc[:-6]) / nc[:-6] * 100
    nf['nifty_roc_6'] = nifty_roc_6
    
    # NIFTY RSI-14
    nf['nifty_rsi_14'] = _rsi(nc, 14)
    
    # NIFTY Bollinger position
    sma20 = _sma(nc, 20)
    std20 = pd.Series(nc).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = np.maximum(bb_upper - bb_lower, 1e-8)
    nf['nifty_bb_position'] = np.clip((nc - bb_lower) / bb_range, 0, 1)
    
    # NIFTY EMA-9 slope (% change over 3 candles)
    nifty_ema9 = _ema(nc, 9)
    nifty_slope = np.zeros(len(nc))
    nifty_slope[3:] = np.where(
        nifty_ema9[:-3] > 0,
        (nifty_ema9[3:] - nifty_ema9[:-3]) / nifty_ema9[:-3] * 100,
        0
    )
    nf['nifty_ema9_slope'] = nifty_slope
    
    # NIFTY ATR%
    nifty_atr14 = _atr(nh, nl, nc, 14)
    nf['nifty_atr_pct'] = np.where(nc > 0, nifty_atr14 / nc * 100, 0)
    
    # Prepare for merge_asof
    nifty_merge_cols = ['date', 'nifty_roc_6', 'nifty_rsi_14', 'nifty_bb_position',
                        'nifty_ema9_slope', 'nifty_atr_pct']
    nf_merge = nf[nifty_merge_cols].copy()
    nf_merge = nf_merge.sort_values('date')
    
    result = result.sort_values('date').reset_index(drop=True)
    
    # Time-aligned merge: each stock candle gets the NIFTY values at same timestamp
    # Normalize timezones to avoid merge_asof tz mismatch
    _left = result[['date']].copy()
    _right = nf_merge.copy()
    if _left['date'].dt.tz is not None:
        _left['date'] = _left['date'].dt.tz_localize(None)
    if _right['date'].dt.tz is not None:
        _right['date'] = _right['date'].dt.tz_localize(None)
    merged = pd.merge_asof(
        _left,
        _right,
        on='date',
        direction='backward',  # Use most recent NIFTY data at or before stock candle
        tolerance=pd.Timedelta('10min'),  # Don't use stale data
    )
    
    _nifty_5m_cols = ['nifty_roc_6', 'nifty_rsi_14', 'nifty_bb_position', 'nifty_ema9_slope', 'nifty_atr_pct']
    for col in _nifty_5m_cols:
        result[col] = merged[col].values
    
    # ROBUSTNESS: If merge_asof left NaN tails (stock data extends beyond NIFTY data),
    # forward-fill from previous valid values. This prevents 0.0 defaults that the
    # direction model interprets as "extremely oversold index" → false UP bias.
    for col in _nifty_5m_cols:
        if result[col].isna().any():
            result[col] = result[col].ffill()
    
    # relative_strength = stock momentum - NIFTY momentum (THE alpha signal)
    # If stock goes +0.5% while NIFTY goes +0.3%, relative_strength = +0.2%
    # Positive = stock outperforming market = genuine bullish signal
    if 'roc_6' in result.columns:
        result['relative_strength'] = result['roc_6'] - result['nifty_roc_6'].fillna(0)
    else:
        result['relative_strength'] = 0.0
    
    # ── NIFTY Daily context (previous day's values) ──
    if nifty_daily_df is not None and len(nifty_daily_df) >= 60:
        nd = nifty_daily_df.copy().sort_values('date').reset_index(drop=True)
        ndc = nd['close'].values
        nd_ema50 = _ema(ndc, 50)
        nd_rsi = _rsi(ndc, 14)
        
        nd['_date'] = nd['date'].dt.date if hasattr(nd['date'].dt, 'date') else nd['date'].apply(
            lambda x: x.date() if hasattr(x, 'date') else x
        )
        
        # Build lookup: date D → previous day's NIFTY daily features
        daily_ctx = {}
        for i in range(51, len(nd)):
            d = nd['_date'].iloc[i]
            prev = i - 1
            trend = (ndc[prev] - nd_ema50[prev]) / nd_ema50[prev] * 100 if nd_ema50[prev] > 0 else 0.0
            daily_ctx[d] = {
                'nifty_daily_trend': float(trend),
                'nifty_daily_rsi': float(nd_rsi[prev]),
            }
        
        result['_date'] = result['date'].dt.date
        # ROBUSTNESS: If stock candles include dates NOT in NIFTY daily
        # (e.g., today's data before NIFTY daily is updated), use the most
        # recent available NIFTY daily values instead of defaulting to 0.
        # This prevents the direction model seeing 0=oversold → false UP bias.
        _all_dates_sorted = sorted(daily_ctx.keys())
        def _get_daily_ctx(d, key):
            ctx = daily_ctx.get(d)
            if ctx:
                return ctx.get(key, 0.0)
            # Date not in NIFTY daily → use most recent prior date
            import bisect
            idx = bisect.bisect_right(_all_dates_sorted, d) - 1
            if idx >= 0:
                return daily_ctx[_all_dates_sorted[idx]].get(key, 0.0)
            return 0.0
        
        result['nifty_daily_trend'] = result['_date'].map(
            lambda d: _get_daily_ctx(d, 'nifty_daily_trend')
        )
        result['nifty_daily_rsi'] = result['_date'].map(
            lambda d: _get_daily_ctx(d, 'nifty_daily_rsi')
        )
        result.drop(columns=['_date'], inplace=True, errors='ignore')
    else:
        result['nifty_daily_trend'] = 0.0
        result['nifty_daily_rsi'] = 0.0
    
    # Clean up
    for col in nifty_feat_names:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


def _add_futures_oi_context(intraday_df: pd.DataFrame, futures_oi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily futures OI features onto each 5-min candle.
    
    For each trading day in intraday_df, uses the PREVIOUS day's futures OI
    features to avoid lookahead bias (same logic as _add_daily_context).
    
    Args:
        intraday_df: 5-min candle DataFrame (already has 'date' column)
        futures_oi_df: Daily futures OI DataFrame with columns:
            date, symbol, fut_oi_change_pct, fut_oi_buildup,
            fut_basis_pct, fut_oi_5d_trend, fut_vol_ratio
    
    Returns:
        intraday_df with 5 new futures OI context columns added
    """
    fut_features = _get_futures_oi_feature_names()
    
    if futures_oi_df is None or len(futures_oi_df) == 0:
        for col in fut_features:
            intraday_df[col] = 0.0
        return intraday_df
    
    foi = futures_oi_df.copy().sort_values('date').reset_index(drop=True)
    
    # Normalize dates to date-only for lookup
    foi['_date'] = foi['date'].dt.date if hasattr(foi['date'].dt, 'date') else foi['date'].apply(
        lambda x: x.date() if hasattr(x, 'date') else x
    )
    
    # Build lookup: for each date, store the PREVIOUS day's features
    # This means for date D, we use features from the row at index i-1
    context_by_date = {}
    sorted_dates = foi['_date'].unique()
    
    for i in range(1, len(foi)):
        current_date = foi['_date'].iloc[i]
        prev_row = foi.iloc[i - 1]
        
        context_by_date[current_date] = {
            col: float(prev_row[col]) if col in foi.columns and pd.notna(prev_row[col])
            else 0.0
            for col in fut_features
        }
    
    # Forward-fill: for intraday dates AFTER the last OI date, use the
    # latest available OI row.  OI is updated EOD so today's live candles
    # won't have today's OI yet — they should use yesterday's.
    if len(foi) > 0:
        _last_oi_row = foi.iloc[-1]
        _last_oi_context = {
            col: float(_last_oi_row[col]) if col in foi.columns and pd.notna(_last_oi_row[col])
            else 0.0
            for col in fut_features
        }
    else:
        _last_oi_context = None
    
    # Map onto each 5-min candle
    result = intraday_df.copy()
    result['_date'] = result['date'].dt.date
    _ffill_logged = False
    
    def _lookup(d, c):
        nonlocal _ffill_logged
        val = context_by_date.get(d, {}).get(c)
        if val is not None:
            return val
        # Forward-fill: date beyond OI coverage → use latest OI
        if _last_oi_context is not None and len(sorted_dates) > 0 and d > sorted_dates[-1]:
            if not _ffill_logged:
                import logging
                logging.getLogger('feature_engineering').info(
                    f'OI forward-fill: candle date {d} > last OI date {sorted_dates[-1]}, '
                    f'using latest OI (fut_oi_buildup={_last_oi_context.get("fut_oi_buildup", 0):.3f})'
                )
                _ffill_logged = True
            return _last_oi_context.get(c, 0.0)
        return 0.0
    
    for col in fut_features:
        result[col] = result['_date'].map(lambda d, c=col: _lookup(d, c))
    
    result.drop(columns=['_date'], inplace=True, errors='ignore')
    
    # Replace inf/nan
    for col in fut_features:
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


# ══════════════════════════════════════════════════════════════
#  INTRADAY 5-MIN FUTURES OI CONTEXT
# ══════════════════════════════════════════════════════════════

def _add_intraday_oi_context(intraday_df: pd.DataFrame, oi_intra_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 5-min intraday futures OI features onto each price candle.
    
    Uses merge_asof to align OI candles with price candles by timestamp.
    Then computes derived features capturing OI flow dynamics:
      - Per-candle OI change (building vs unwinding)
      - OI velocity (rate over 30 min)
      - Session-level cumulative OI change
      - OI/volume conviction ratio
      - OI acceleration (2nd derivative)
      - OI regime (vs 20-candle MA)
    
    Args:
        intraday_df: 5-min price candles (with 'date' column)
        oi_intra_df: 5-min futures OI candles with columns:
            date, open, high, low, close, volume, fut_oi, symbol
    
    Returns:
        intraday_df with 6 new intraday OI columns
    """
    feat_names = _get_intraday_oi_feature_names()
    result = intraday_df.copy()
    
    if oi_intra_df is None or len(oi_intra_df) == 0:
        for col in feat_names:
            result[col] = 0.0
        return result
    
    oi = oi_intra_df.copy()
    oi['date'] = pd.to_datetime(oi['date'])
    oi = oi.sort_values('date').reset_index(drop=True)
    
    # Filter out zero-OI rows (contract not yet active)
    oi = oi[oi['fut_oi'] > 0].reset_index(drop=True)
    
    if len(oi) < 10:
        for col in feat_names:
            result[col] = 0.0
        return result
    
    # --- Compute OI-derived features on the OI dataframe first ---
    oi_vals = oi['fut_oi'].values.astype(float)
    oi_vol = oi['volume'].values.astype(float)
    
    # 1. OI % change per candle
    oi_pct = np.zeros(len(oi_vals))
    oi_pct[1:] = np.where(oi_vals[:-1] > 0,
                           (oi_vals[1:] - oi_vals[:-1]) / oi_vals[:-1] * 100,
                           0.0)
    oi['oi_change_5m'] = np.clip(oi_pct, -10, 10)
    
    # 2. OI velocity over 6 candles (30 min)
    oi_vel = np.zeros(len(oi_vals))
    oi_vel[6:] = np.where(oi_vals[:-6] > 0,
                           (oi_vals[6:] - oi_vals[:-6]) / oi_vals[:-6] * 100,
                           0.0)
    oi['oi_velocity_6'] = np.clip(oi_vel, -25, 25)
    
    # 3. Session cumulative OI change (% from day's first OI reading)
    oi['_date'] = oi['date'].dt.date
    session_change = np.zeros(len(oi))
    for d, grp in oi.groupby('_date'):
        idx = grp.index
        day_start_oi = grp['fut_oi'].iloc[0]
        if day_start_oi > 0:
            session_change[idx] = (grp['fut_oi'].values - day_start_oi) / day_start_oi * 100
    oi['oi_session_change'] = np.clip(session_change, -20, 20)
    
    # 4. OI change / volume (conviction: higher = position change per unit vol)
    oi_abs_change = np.abs(np.diff(oi_vals, prepend=oi_vals[0]))
    oi['oi_vs_volume'] = np.where(oi_vol > 0,
                                   oi_abs_change / oi_vol,
                                   0.0)
    oi['oi_vs_volume'] = np.clip(oi['oi_vs_volume'].values, 0, 10)
    
    # 5. OI acceleration (2nd derivative: d(oi_change)/dt)
    oi_accel = np.zeros(len(oi_vals))
    oi_accel[2:] = oi_pct[2:] - oi_pct[1:-1]
    oi['oi_acceleration'] = np.clip(oi_accel, -5, 5)
    
    # 6. OI regime (current OI / 20-candle MA)
    oi_ma20 = pd.Series(oi_vals).rolling(20, min_periods=5).mean().values
    oi['oi_regime'] = np.where(oi_ma20 > 0, oi_vals / oi_ma20, 1.0)
    oi['oi_regime'] = np.clip(oi['oi_regime'].values, 0.5, 2.0)
    
    # --- merge_asof: align OI features onto price candles ---
    merge_cols = ['date'] + feat_names
    oi_merge = oi[merge_cols].copy()
    oi_merge = oi_merge.sort_values('date')
    
    result = result.sort_values('date').reset_index(drop=True)
    
    # Normalize timezones
    _left = result[['date']].copy()
    _right = oi_merge.copy()
    if _left['date'].dt.tz is not None:
        _left['date'] = _left['date'].dt.tz_localize(None)
    if _right['date'].dt.tz is not None:
        _right['date'] = _right['date'].dt.tz_localize(None)
    
    merged = pd.merge_asof(
        _left,
        _right,
        on='date',
        direction='backward',
        tolerance=pd.Timedelta('10min'),
    )
    
    for col in feat_names:
        if col in merged.columns:
            result[col] = merged[col].values
        else:
            result[col] = 0.0
    
    # Clean up
    for col in feat_names:
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    if '_date' in result.columns:
        result.drop(columns=['_date'], inplace=True, errors='ignore')
    
    return result


# ══════════════════════════════════════════════════════════════
#  IV REGIME CONTEXT (from daily option chain data)
# ══════════════════════════════════════════════════════════════

def _add_iv_regime_context(intraday_df: pd.DataFrame, iv_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily IV regime features from NSE Bhav Copy option chain data.
    
    IV is a naturally slow-moving signal — it classifies the volatility
    environment, not per-candle flow. This is exactly how it should be used:
      - IV Rank: where is current IV vs recent range? (0-1)
      - IV Percentile: % of recent days below current IV (0-1)
      - IV Skew Z-score: how extreme is today's put-call IV gap?
      - IV Regime: categorical (low=0, mid=1, high=2)
      - PCR OI Regime: normalized PCR vs history
    
    Uses PREVIOUS day's values (no lookahead).
    
    Args:
        intraday_df: 5-min price candles (with 'date' column)
        iv_df: Daily options OI DataFrame with columns:
            trade_date, atm_iv, iv_skew, pcr_oi
    
    Returns:
        intraday_df with 5 new IV regime columns
    """
    feat_names = _get_iv_regime_feature_names()
    result = intraday_df.copy()
    
    if iv_df is None or len(iv_df) < 5:
        for col in feat_names:
            result[col] = 0.0
        return result
    
    iv = iv_df.copy()
    
    # Normalize date
    if 'trade_date' in iv.columns:
        iv['_date'] = pd.to_datetime(iv['trade_date']).dt.date
    elif 'date' in iv.columns:
        iv['_date'] = pd.to_datetime(iv['date']).dt.date
    else:
        for col in feat_names:
            result[col] = 0.0
        return result
    
    iv = iv.sort_values('_date').reset_index(drop=True)
    
    # Ensure required columns exist
    for c in ['atm_iv', 'iv_skew', 'pcr_oi']:
        if c not in iv.columns:
            iv[c] = 0.0
    
    # --- Compute rolling IV features on the daily data ---
    atm_iv = iv['atm_iv'].values.astype(float)
    iv_skew_vals = iv['iv_skew'].values.astype(float)
    pcr_vals = iv['pcr_oi'].values.astype(float)
    
    window = 30  # 30 trading days lookback
    
    # IV Rank: (current - min) / (max - min) over window
    iv_min = pd.Series(atm_iv).rolling(window, min_periods=5).min().values
    iv_max = pd.Series(atm_iv).rolling(window, min_periods=5).max().values
    iv_range = np.maximum(iv_max - iv_min, 0.01)
    iv['iv_rank_30d'] = np.clip((atm_iv - iv_min) / iv_range, 0, 1)
    
    # IV Percentile: fraction of past 30d with IV below current
    iv_pctile = np.zeros(len(atm_iv))
    for i in range(window, len(atm_iv)):
        lookback = atm_iv[max(0, i - window):i]
        iv_pctile[i] = np.mean(lookback < atm_iv[i])
    # For early rows with less history, use what's available
    for i in range(5, min(window, len(atm_iv))):
        lookback = atm_iv[:i]
        iv_pctile[i] = np.mean(lookback < atm_iv[i])
    iv['iv_percentile_30d'] = np.clip(iv_pctile, 0, 1)
    
    # IV Skew Z-score: (current skew - 30d mean) / 30d std
    skew_mean = pd.Series(iv_skew_vals).rolling(window, min_periods=5).mean().values
    skew_std = pd.Series(iv_skew_vals).rolling(window, min_periods=5).std().values
    skew_std = np.maximum(skew_std, 0.01)
    iv['iv_skew_zscore'] = np.clip((iv_skew_vals - skew_mean) / skew_std, -3, 3)
    
    # IV Regime: categorical bucket
    iv['iv_regime'] = np.where(iv['iv_percentile_30d'] < 0.25, 0,
                      np.where(iv['iv_percentile_30d'] > 0.75, 2, 1)).astype(float)
    
    # PCR OI Regime: z-score of PCR vs 30d rolling
    pcr_mean = pd.Series(pcr_vals).rolling(window, min_periods=5).mean().values
    pcr_std = pd.Series(pcr_vals).rolling(window, min_periods=5).std().values
    pcr_std = np.maximum(pcr_std, 0.01)
    iv['pcr_oi_regime'] = np.clip((pcr_vals - pcr_mean) / pcr_std, -3, 3)
    
    # --- Build per-date lookup: use PREVIOUS day's values (no lookahead) ---
    context_by_date = {}
    for i in range(1, len(iv)):
        curr_date = iv['_date'].iloc[i]
        prev_row = iv.iloc[i - 1]
        context_by_date[curr_date] = {
            col: float(prev_row[col]) if col in iv.columns and pd.notna(prev_row.get(col)) else 0.0
            for col in feat_names
        }
    
    # Forward-fill for dates beyond IV coverage
    _last_ctx = None
    if len(iv) > 0:
        _last = iv.iloc[-1]
        _last_ctx = {
            col: float(_last[col]) if col in iv.columns and pd.notna(_last.get(col)) else 0.0
            for col in feat_names
        }
    
    sorted_iv_dates = sorted(iv['_date'].unique())
    
    # Map onto each 5-min candle
    result['_date'] = result['date'].dt.date
    
    def _lookup(d, c):
        val = context_by_date.get(d, {}).get(c)
        if val is not None:
            return val
        if _last_ctx is not None and len(sorted_iv_dates) > 0 and d > sorted_iv_dates[-1]:
            return _last_ctx.get(c, 0.0)
        return 0.0
    
    for col in feat_names:
        result[col] = result['_date'].map(lambda d, c=col: _lookup(d, c))
    
    result.drop(columns=['_date'], inplace=True, errors='ignore')
    
    # Clean up
    for col in feat_names:
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


# ══════════════════════════════════════════════════════════════
#  SECTOR INDEX CONTEXT
# ══════════════════════════════════════════════════════════════

def _add_sector_context(intraday_df: pd.DataFrame, sector_5min_df: pd.DataFrame,
                        sector_daily_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge sector index context features onto each 5-min candle.
    
    Exactly parallels _add_nifty_context() but uses the stock's specific sector
    index instead of NIFTY50. This isolates sector-specific momentum vs broad
    market moves.
    
    The killer feature: sector_relative = stock_roc_6 - sector_roc_6
    If a metal stock rallies +1% but NIFTY METAL is -2%, sector_relative = +3%
    which is extremely unusual and likely unsustainable → model learns to flag it.
    
    Args:
        intraday_df: 5-min stock candles (with 'date', 'close', 'roc_6' columns)
        sector_5min_df: Sector index 5-min OHLCV candles
        sector_daily_df: Optional sector index daily candles for daily context
    
    Returns:
        intraday_df with 8 new sector context columns
    """
    sector_feat_names = _get_sector_feature_names()
    result = intraday_df.copy()
    
    # ── Compute sector 5-min features ──
    sf = sector_5min_df.copy().sort_values('date').reset_index(drop=True)
    sc = sf['close'].values
    sh = sf['high'].values
    sl = sf['low'].values
    
    # Sector momentum (% change over 6 candles = 30 min)
    sector_roc_6 = np.zeros(len(sc))
    sector_roc_6[6:] = (sc[6:] - sc[:-6]) / np.where(sc[:-6] > 0, sc[:-6], 1) * 100
    sf['sector_roc_6'] = sector_roc_6
    
    # Sector RSI-14
    sf['sector_rsi_14'] = _rsi(sc, 14)
    
    # Sector Bollinger position
    sma20 = _sma(sc, 20)
    std20 = pd.Series(sc).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = np.maximum(bb_upper - bb_lower, 1e-8)
    sf['sector_bb_position'] = np.clip((sc - bb_lower) / bb_range, 0, 1)
    
    # Sector EMA-9 slope
    sector_ema9 = _ema(sc, 9)
    sector_slope = np.zeros(len(sc))
    sector_slope[3:] = np.where(
        sector_ema9[:-3] > 0,
        (sector_ema9[3:] - sector_ema9[:-3]) / sector_ema9[:-3] * 100,
        0
    )
    sf['sector_ema9_slope'] = sector_slope
    
    # Sector ATR%
    sector_atr14 = _atr(sh, sl, sc, 14)
    sf['sector_atr_pct'] = np.where(sc > 0, sector_atr14 / sc * 100, 0)
    
    # Prepare for merge_asof
    sector_merge_cols = ['date', 'sector_roc_6', 'sector_rsi_14', 'sector_bb_position',
                         'sector_ema9_slope', 'sector_atr_pct']
    sf_merge = sf[sector_merge_cols].copy()
    sf_merge = sf_merge.sort_values('date')
    
    result = result.sort_values('date').reset_index(drop=True)
    
    # Time-aligned merge: each stock candle gets the sector values at same timestamp
    # Normalize timezones to avoid merge_asof tz mismatch
    _left = result[['date']].copy()
    _right = sf_merge.copy()
    if _left['date'].dt.tz is not None:
        _left['date'] = _left['date'].dt.tz_localize(None)
    if _right['date'].dt.tz is not None:
        _right['date'] = _right['date'].dt.tz_localize(None)
    merged = pd.merge_asof(
        _left,
        _right,
        on='date',
        direction='backward',
        tolerance=pd.Timedelta('10min'),
    )
    
    for col in ['sector_roc_6', 'sector_rsi_14', 'sector_bb_position', 'sector_ema9_slope', 'sector_atr_pct']:
        result[col] = merged[col].values
    
    # ROBUSTNESS: Forward-fill NaN tails (stock data extends beyond sector data)
    _sector_5m_cols = ['sector_roc_6', 'sector_rsi_14', 'sector_bb_position', 'sector_ema9_slope', 'sector_atr_pct']
    for col in _sector_5m_cols:
        if result[col].isna().any():
            result[col] = result[col].ffill()
    
    # sector_relative = stock momentum - sector momentum (SECTOR ALPHA signal)
    # If stock goes +0.5% while sector goes -1.0%, sector_relative = +1.5%
    # This is even more specific than relative_strength (vs NIFTY50)
    if 'roc_6' in result.columns:
        result['sector_relative'] = result['roc_6'] - result['sector_roc_6'].fillna(0)
    else:
        result['sector_relative'] = 0.0
    
    # ── Sector daily context (previous day's values) ──
    if sector_daily_df is not None and len(sector_daily_df) >= 60:
        sd = sector_daily_df.copy().sort_values('date').reset_index(drop=True)
        sdc = sd['close'].values
        sd_ema50 = _ema(sdc, 50)
        sd_rsi = _rsi(sdc, 14)
        
        sd['_date'] = sd['date'].dt.date if hasattr(sd['date'].dt, 'date') else sd['date'].apply(
            lambda x: x.date() if hasattr(x, 'date') else x
        )
        
        # Build lookup: date D → previous day's sector daily features
        daily_ctx = {}
        for i in range(51, len(sd)):
            d = sd['_date'].iloc[i]
            prev = i - 1
            trend = (sdc[prev] - sd_ema50[prev]) / sd_ema50[prev] * 100 if sd_ema50[prev] > 0 else 0.0
            daily_ctx[d] = {
                'sector_daily_trend': float(trend),
                'sector_daily_rsi': float(sd_rsi[prev]),
            }
        
        result['_date'] = result['date'].dt.date
        # ROBUSTNESS: Forward-fill for dates not in sector daily data
        _all_sec_dates = sorted(daily_ctx.keys())
        def _get_sec_daily_ctx(d, key):
            ctx = daily_ctx.get(d)
            if ctx:
                return ctx.get(key, 0.0)
            import bisect
            idx = bisect.bisect_right(_all_sec_dates, d) - 1
            if idx >= 0:
                return daily_ctx[_all_sec_dates[idx]].get(key, 0.0)
            return 0.0
        
        result['sector_daily_trend'] = result['_date'].map(
            lambda d: _get_sec_daily_ctx(d, 'sector_daily_trend')
        )
        result['sector_daily_rsi'] = result['_date'].map(
            lambda d: _get_sec_daily_ctx(d, 'sector_daily_rsi')
        )
        result.drop(columns=['_date'], inplace=True, errors='ignore')
    else:
        result['sector_daily_trend'] = 0.0
        result['sector_daily_rsi'] = 0.0
    
    # Clean up
    for col in sector_feat_names:
        if col not in result.columns:
            result[col] = 0.0
        result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    return result


if __name__ == '__main__':
    # Quick test on synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2026-01-02 09:15', periods=n, freq='5min')
    price = 100 + np.cumsum(np.random.randn(n) * 0.3)
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': price + np.random.randn(n) * 0.1,
        'high': price + abs(np.random.randn(n) * 0.5),
        'low': price - abs(np.random.randn(n) * 0.5),
        'close': price,
        'volume': np.random.randint(10000, 500000, n)
    })
    
    result = compute_features(test_df, symbol="TEST")
    print(f"Input: {len(test_df)} candles")
    print(f"Output: {len(result)} rows, {len(result.columns)} columns")
    print(f"\nFeature columns ({len(get_feature_names())}):")
    for f in get_feature_names():
        vals = result[f].dropna()
        print(f"  {f:<25s} min={vals.min():>8.2f}  max={vals.max():>8.2f}  mean={vals.mean():>8.2f}")
