"""
LABEL CREATOR — Generate training labels from raw candles

For each candle at time T, looks ahead N candles to determine:
  - Label +1 (BIG UP)   →  price rises ≥ threshold within lookahead
  - Label -1 (BIG DOWN) →  price drops ≥ threshold within lookahead
  - Label  0 (NO MOVE)  →  stays within ± threshold

Supported labeling methods:
  1. "first_to_break" (legacy):
     Scans forward candle-by-candle; first direction to exceed threshold wins.
     Pro: captures intrabar momentum. Con: noisy in choppy markets.

  2. "net_return" (recommended, default):
     Labels based on close[t+N] vs close[t]. If net return ≥ +threshold → UP,
     ≤ -threshold → DOWN, else FLAT. Simpler, less noisy, aligns with actual
     hold-period P&L.

Key design decisions:
  - Default lookahead: 8 candles (40 min on 5-min data)
  - Default threshold: ATR × 2.0 (only labels clear moves)
  - ATR-normalized threshold: each candle's threshold = atr_factor × ATR%,
    so every stock contributes balanced UP/DOWN/FLAT regardless of volatility
  - Labels are applied to the CURRENT candle row (not the future one)
  - Last N candles of each day get label=NaN (insufficient lookahead)
"""

import numpy as np
import pandas as pd
from typing import Optional


def _compute_atr(high, low, close, period=14):
    """Compute Average True Range (Wilder smoothing)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def create_labels(
    df: pd.DataFrame,
    lookahead_candles: int = 8,
    threshold_pct: float = 1.0,
    use_directional: bool = True,
    use_atr_threshold: bool = True,
    atr_factor: float = 2.0,
    label_method: str = 'net_return',
) -> pd.DataFrame:
    """Add target labels to feature DataFrame.
    
    Args:
        df: DataFrame with at minimum: date, high, low, close
        lookahead_candles: How many candles to look ahead (default 8 = 40 min)
        threshold_pct: Fixed minimum % move (used when use_atr_threshold=False)
        use_directional: If True, label is +1/-1/0. If False, label is 1/0
        use_atr_threshold: If True, threshold = atr_factor × ATR% per candle
            This normalizes across stocks so volatile and non-volatile stocks
            contribute balanced directional labels.
        atr_factor: Multiplier for ATR-based threshold (default 2.0).
            Higher = harder to trigger = more FLAT labels but cleaner signals.
        label_method: 'net_return' (recommended) or 'first_to_break' (legacy).
            net_return: label = sign(close[t+N] - close[t]) if |ret| >= threshold
            first_to_break: first direction to breach threshold within N candles
            
    Returns:
        DataFrame with added columns:
          - label: Target label (+1, -1, or 0)
          - max_up_pct: Max upside % within lookahead
          - max_down_pct: Max downside % within lookahead
          - net_return_pct: Net close-to-close return over lookahead (net_return mode)
          - label_quality: 'clean' or 'boundary'
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    max_up = np.full(n, np.nan)
    max_down = np.full(n, np.nan)
    
    # Compute ATR for dynamic threshold
    atr = _compute_atr(high, low, close, 14)
    atr_pct = np.where(close > 0, atr / close * 100, 0)
    
    # Compute max up/down for reporting (still useful for analysis)
    for i in range(n - lookahead_candles):
        future_highs = high[i + 1 : i + 1 + lookahead_candles]
        future_lows  = low[i + 1 : i + 1 + lookahead_candles]
        max_up[i] = (np.max(future_highs) - close[i]) / close[i] * 100
        max_down[i] = (np.min(future_lows) - close[i]) / close[i] * 100
    
    df['max_up_pct'] = max_up
    df['max_down_pct'] = max_down
    
    # === LABELING ===
    label = np.full(n, np.nan)
    net_ret = np.full(n, np.nan)
    
    if use_directional:
        if label_method == 'net_return':
            # ── NET RETURN METHOD (recommended) ──
            # Label based on where price IS after N candles (close-to-close)
            # Simpler, less noisy, aligns with actual hold-period P&L
            for i in range(n - lookahead_candles):
                if np.isnan(atr_pct[i]):
                    continue
                
                # Determine threshold for this candle
                if use_atr_threshold and not np.isnan(atr[i]) and atr_pct[i] > 0:
                    thresh = max(0.10, atr_factor * atr_pct[i])
                else:
                    thresh = threshold_pct
                
                # Net return = close[t+N] vs close[t]
                future_close = close[i + lookahead_candles]
                ret_pct = (future_close - close[i]) / close[i] * 100
                net_ret[i] = ret_pct
                
                if ret_pct >= thresh:
                    label[i] = 1    # UP
                elif ret_pct <= -thresh:
                    label[i] = -1   # DOWN
                else:
                    label[i] = 0    # FLAT
            
            df['net_return_pct'] = net_ret
        
        elif label_method == 'first_to_break':
            # ── FIRST-TO-BREAK METHOD (legacy) ──
            # Scan forward candle by candle; first direction to exceed threshold wins
            for i in range(n - lookahead_candles):
                if np.isnan(atr_pct[i]):
                    continue
                
                if use_atr_threshold and not np.isnan(atr[i]) and atr_pct[i] > 0:
                    thresh = max(0.10, atr_factor * atr_pct[i])
                else:
                    thresh = threshold_pct
                
                up_level = close[i] * (1 + thresh / 100)
                down_level = close[i] * (1 - thresh / 100)
                
                found = False
                for j in range(1, lookahead_candles + 1):
                    idx = i + j
                    hit_up = high[idx] >= up_level
                    hit_down = low[idx] <= down_level
                    
                    if hit_up and hit_down:
                        up_mag = high[idx] - close[i]
                        down_mag = close[i] - low[idx]
                        label[i] = 1 if up_mag >= down_mag else -1
                        found = True
                        break
                    elif hit_up:
                        label[i] = 1
                        found = True
                        break
                    elif hit_down:
                        label[i] = -1
                        found = True
                        break
                
                if not found:
                    label[i] = 0
        
        else:
            raise ValueError(f"Unknown label_method: {label_method}. Use 'net_return' or 'first_to_break'")
        
        df['label'] = label
    else:
        # Binary: 1 = big move (either direction), 0 = no move
        for i in range(n - lookahead_candles):
            if np.isnan(max_up[i]):
                continue
            
            if use_atr_threshold and not np.isnan(atr[i]) and atr_pct[i] > 0:
                thresh = max(0.10, atr_factor * atr_pct[i])
            else:
                thresh = threshold_pct
            
            if max_up[i] >= thresh or abs(max_down[i]) >= thresh:
                label[i] = 1
            else:
                label[i] = 0
        
        df['label'] = label
    
    # Quality flag — last candles of each day may have cross-day lookahead issues
    df['_date'] = df['date'].dt.date
    df['label_quality'] = 'clean'
    
    for day, group in df.groupby('_date'):
        idx = group.index
        if len(idx) < lookahead_candles:
            df.loc[idx, 'label_quality'] = 'boundary'
        else:
            boundary_idx = idx[-lookahead_candles:]
            df.loc[boundary_idx, 'label_quality'] = 'boundary'
    
    df.drop(columns=['_date'], inplace=True)
    
    return df


def label_distribution(df: pd.DataFrame) -> dict:
    """Print and return label distribution stats."""
    clean = df[df['label_quality'] == 'clean']
    labeled = clean.dropna(subset=['label'])
    
    total = len(labeled)
    if total == 0:
        return {'total': 0}
    
    up_count = int((labeled['label'] == 1).sum())
    down_count = int((labeled['label'] == -1).sum())
    no_move = int((labeled['label'] == 0).sum())
    
    stats = {
        'total_labeled': total,
        'big_up': up_count,
        'big_down': down_count,
        'no_move': no_move,
        'up_pct': round(up_count / total * 100, 1),
        'down_pct': round(down_count / total * 100, 1),
        'no_move_pct': round(no_move / total * 100, 1),
        'boundary_excluded': int((df['label_quality'] == 'boundary').sum()),
        'avg_max_up': round(labeled['max_up_pct'].mean(), 3),
        'avg_max_down': round(labeled['max_down_pct'].mean(), 3),
    }
    
    # Add net return stats if available
    if 'net_return_pct' in labeled.columns:
        net_valid = labeled['net_return_pct'].dropna()
        if len(net_valid) > 0:
            stats['avg_net_return'] = round(net_valid.mean(), 4)
            stats['std_net_return'] = round(net_valid.std(), 4)
    
    print(f"\n{'='*50}")
    print(f"LABEL DISTRIBUTION")
    print(f"{'='*50}")
    print(f"  Total labeled candles: {stats['total_labeled']:,}")
    print(f"  Big UP   (+1): {stats['big_up']:>6,}  ({stats['up_pct']}%)")
    print(f"  Big DOWN (-1): {stats['big_down']:>6,}  ({stats['down_pct']}%)")
    print(f"  No move  ( 0): {stats['no_move']:>6,}  ({stats['no_move_pct']}%)")
    print(f"  Boundary excluded: {stats['boundary_excluded']:,}")
    print(f"  Avg max up:   {stats['avg_max_up']:+.3f}%")
    print(f"  Avg max down: {stats['avg_max_down']:+.3f}%")
    if 'avg_net_return' in stats:
        print(f"  Avg net return: {stats['avg_net_return']:+.4f}%")
        print(f"  Std net return: {stats['std_net_return']:.4f}%")
    print(f"{'='*50}\n")
    
    return stats


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
    
    result = create_labels(test_df, lookahead_candles=8, threshold_pct=1.0,
                           label_method='net_return', atr_factor=2.0)
    label_distribution(result)
