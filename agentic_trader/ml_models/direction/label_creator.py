"""
DIRECTION LABEL CREATOR — Generate UP/DOWN labels on DAILY timeframe

Architecture change: 5-minute direction is unpredictable (proven by 2 failed
training attempts at ~50% accuracy). DAILY direction has real signal (53.3%
accuracy, 59% precision at high confidence, 127 trees before early stopping).

Label logic:
  - Input: Daily OHLCV candles 
  - Label = 1 (UP):   next day close > today's close by >= threshold
  - Label = 0 (DOWN): next day close < today's close by >= threshold
  - Small moves (|return| < threshold): NaN (excluded from training)
"""

import numpy as np
import pandas as pd
from typing import Tuple


LABEL_MAP = {0: 0, 1: 1}  # 0=DOWN, 1=UP
LABEL_NAMES = {0: 'DOWN', 1: 'UP'}


def create_direction_labels(
    df: pd.DataFrame,
    threshold_pct: float = 0.3,
) -> pd.DataFrame:
    """Add UP/DOWN direction labels based on next-day close-to-close return.
    
    Args:
        df: Daily OHLCV DataFrame with columns: date, open, high, low, close, volume
        threshold_pct: Minimum absolute % return to assign label (default 0.3%).
                       Moves smaller than this get label=NaN (skipped).
        
    Returns:
        DataFrame with added columns:
          - dir_label: 1 (UP) or 0 (DOWN), NaN for small-movers
          - next_return_pct: Next day's close-to-close return %
          - next_day_range_pct: Next day's (high-low)/close range %
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    next_return = np.full(n, np.nan)
    next_range = np.full(n, np.nan)
    
    for i in range(n - 1):
        next_return[i] = (close[i + 1] - close[i]) / close[i] * 100
        next_range[i] = (high[i + 1] - low[i + 1]) / close[i + 1] * 100
    
    df['next_return_pct'] = next_return
    df['next_day_range_pct'] = next_range
    
    # Direction labels — only for significant moves
    dir_label = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(next_return[i]):
            continue
        if abs(next_return[i]) >= threshold_pct:
            dir_label[i] = 1 if next_return[i] > 0 else 0
    
    df['dir_label'] = dir_label
    
    return df


def direction_label_distribution(df: pd.DataFrame, threshold_pct: float = 0.3) -> dict:
    """Print and return direction label stats."""
    labeled = df.dropna(subset=['dir_label'])
    unlabeled = df[df['dir_label'].isna() & df['next_return_pct'].notna()]
    
    total = len(labeled) + len(unlabeled)
    total_labeled = len(labeled)
    
    if total_labeled == 0:
        print("No labeled samples found!")
        return {'total': 0}
    
    up_count = int((labeled['dir_label'] == 1).sum())
    down_count = int((labeled['dir_label'] == 0).sum())
    
    stats = {
        'total_days': total,
        'labeled': total_labeled,
        'skipped_small': len(unlabeled),
        'up': up_count,
        'down': down_count,
        'up_pct': round(up_count / total_labeled * 100, 1),
        'down_pct': round(down_count / total_labeled * 100, 1),
        'label_rate': round(total_labeled / max(total, 1) * 100, 1),
    }
    
    print(f"\n{'='*55}")
    print(f"  DAILY DIRECTION LABEL DISTRIBUTION")
    print(f"{'='*55}")
    print(f"  Total trading days:    {total:,}")
    print(f"  Labeled (|ret|>={threshold_pct}%): {total_labeled:,} ({stats['label_rate']}%)")
    print(f"  Skipped (small move):  {len(unlabeled):,}")
    print(f"  ───────────────────────────────")
    print(f"  UP   (1): {up_count:>6,}  ({stats['up_pct']}%)")
    print(f"  DOWN (0): {down_count:>6,}  ({stats['down_pct']}%)")
    print(f"{'='*55}\n")
    
    return stats


if __name__ == '__main__':
    # Quick test with daily data
    from ml_models.data_fetcher import load_daily_candles
    
    df = load_daily_candles("SBIN")
    if len(df) > 0:
        result = create_direction_labels(df, threshold_pct=0.3)
        direction_label_distribution(result)
        print(f"Sample labeled rows:")
        sample = result.dropna(subset=['dir_label']).tail(5)
        for _, row in sample.iterrows():
            label = "UP" if row['dir_label'] == 1 else "DOWN"
            print(f"  {row['date'].strftime('%Y-%m-%d')} close={row['close']:.2f} next_ret={row['next_return_pct']:+.2f}% -> {label}")
    else:
        print("No SBIN daily data. Run: python -m ml_models.data_fetcher --daily")
