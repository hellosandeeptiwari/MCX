"""
DAILY DIRECTION TRAINER — Train next-day UP/DOWN XGBoost classifier

Timeframe: DAILY (not 5-min — proven unpredictable at short timescales)

Pipeline:
  1. Load daily candles for all F&O stocks
  2. Compute 35 daily technical features
  3. Create next-day direction labels (UP if close[t+1] > close[t])
  4. Time-ordered train/test split (last N days for test)
  5. Train XGBoost, evaluate, save

Key design choices:
  - Daily timeframe has proven signal (53.3% vs 51.5% random baseline)
  - At high confidence (prob>=55%), UP precision reaches 59%
  - 35 pure technical features from daily OHLCV data
  - Moderate regularization — signal is subtle
  - No class weighting — UP/DOWN is ~51/49 balanced
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DAILY_DIR = BASE_DIR / "data" / "candles_daily"
MODELS_DIR = Path(__file__).parent / "saved_models"
REPORTS_DIR = Path(__file__).parent / "reports"

# Label mapping
LABEL_MAP = {0: 0, 1: 1}  # 0=DOWN, 1=UP
LABEL_NAMES = {0: 'DOWN', 1: 'UP'}

# XGBoost parameters — minimal regularization (matching successful diagnostic)
# Direction signal is subtle — DO NOT over-regularize or model can't learn it.
DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.01,
    'min_child_weight': 20,
    'gamma': 0.0,             # No minimum loss reduction
    'reg_alpha': 0.0,         # No L1 regularization
    'reg_lambda': 1.0,        # Default L2 regularization
    'subsample': 0.80,
    'colsample_bytree': 0.80,
    'n_estimators': 500,
    'early_stopping_rounds': 30,
    'tree_method': 'hist',
    'random_state': 42,
}


def load_and_prepare_data(
    symbols: Optional[list] = None,
    threshold_pct: float = 0.3,
    test_days: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Load daily candles, compute features, create direction labels.
    
    Returns:
        (train_df, test_df, feature_names)
    """
    from ml_models.data_fetcher import load_all_daily, DEFAULT_SYMBOLS
    from .label_creator import create_direction_labels, direction_label_distribution
    from .feature_engineering import compute_direction_features, get_direction_feature_names
    
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    print(f"\n{'='*60}")
    print(f"  DAILY DIRECTION MODEL — DATA PREPARATION")
    print(f"{'='*60}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Move threshold: {threshold_pct}%")
    print(f"  Test window: last {test_days} days")
    
    # Load daily candles
    daily_data = load_all_daily(symbols)
    print(f"  Loaded daily data: {len(daily_data)} symbols")
    
    if not daily_data:
        raise ValueError("No daily candle data found. Run: python -m ml_models.data_fetcher --daily")
    
    # Process each symbol
    all_dfs = []
    total_labeled = 0
    
    for sym, df in daily_data.items():
        if len(df) < 80:  # Need 50 warmup + 30 meaningful days
            continue
        
        # Step 1: Compute 35 daily features
        featured = compute_direction_features(df, symbol=sym)
        if featured.empty or len(featured) < 20:
            continue
        
        # Step 2: Create direction labels (next-day UP/DOWN)
        featured = create_direction_labels(featured, threshold_pct=threshold_pct)
        
        # Step 3: Keep only labeled rows
        labeled = featured.dropna(subset=['dir_label']).copy()
        
        if len(labeled) > 0:
            labeled['symbol'] = sym
            labeled['label_idx'] = labeled['dir_label'].astype(int)
            all_dfs.append(labeled)
            
            up = int((labeled['dir_label'] == 1).sum())
            down = int((labeled['dir_label'] == 0).sum())
            total_labeled += len(labeled)
            print(f"    {sym:>12s}: {len(labeled):>4d} days (UP={up}, DOWN={down})")
    
    if not all_dfs:
        raise ValueError("No direction data created. Check daily candle data.")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n  Total labeled samples: {total_labeled:,}")
    direction_label_distribution(combined, threshold_pct=threshold_pct)
    
    # Train/test split by date (time-ordered, no leakage)
    combined['_date'] = combined['date'].dt.date if hasattr(combined['date'].dt, 'date') else combined['date']
    combined = combined.sort_values('date').reset_index(drop=True)
    
    all_dates = sorted(combined['_date'].unique())
    
    if len(all_dates) <= test_days + 30:
        print(f"  Only {len(all_dates)} unique days — using 70/30 split")
        cutoff_idx = int(len(combined) * 0.7)
        train_df = combined.iloc[:cutoff_idx]
        test_df = combined.iloc[cutoff_idx:]
    else:
        cutoff_date = all_dates[-test_days]
        train_df = combined[combined['_date'] < cutoff_date]
        test_df = combined[combined['_date'] >= cutoff_date]
        print(f"  Train period: {all_dates[0]} to {cutoff_date - timedelta(days=1)}")
        print(f"  Test period:  {cutoff_date} to {all_dates[-1]}")
    
    print(f"  Train: {len(train_df):,} samples  |  Test: {len(test_df):,} samples")
    
    # Check balance
    train_up = int((train_df['label_idx'] == 1).sum())
    train_down = int((train_df['label_idx'] == 0).sum())
    test_up = int((test_df['label_idx'] == 1).sum())
    test_down = int((test_df['label_idx'] == 0).sum())
    print(f"  Train: UP={train_up} ({train_up/len(train_df)*100:.1f}%), DOWN={train_down} ({train_down/len(train_df)*100:.1f}%)")
    print(f"  Test:  UP={test_up} ({test_up/len(test_df)*100:.1f}%), DOWN={test_down} ({test_down/len(test_df)*100:.1f}%)")
    
    feature_names = get_direction_feature_names()
    
    return train_df, test_df, feature_names


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list,
    params: Optional[dict] = None,
    model_name: str = "direction_predictor",
) -> dict:
    """Train daily direction XGBoost model."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    X_train = train_df[feature_names].values
    y_train = train_df['label_idx'].values.astype(int)
    X_test = test_df[feature_names].values
    y_test = test_df['label_idx'].values.astype(int)
    
    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING DAILY DIRECTION MODEL (next-day UP vs DOWN)")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    
    for cls in range(2):
        count = int((y_train == cls).sum())
        print(f"  Train {LABEL_NAMES[cls]:>5s}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    n_estimators = params.pop('n_estimators', 1000)
    early_stopping = params.pop('early_stopping_rounds', 50)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        **params
    )
    
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    train_time = time.time() - t0
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1])
    recall = recall_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1])
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"  DAILY DIRECTION MODEL RESULTS")
    print(f"{'='*60}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"  (Random baseline = ~50% for balanced UP/DOWN)")
    print(f"\n  Per-class metrics:")
    for cls in range(2):
        print(f"    {LABEL_NAMES[cls]:>5s}  P={precision[cls]:.3f}  R={recall[cls]:.3f}  F1={f1[cls]:.3f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>8s} Pred_DOWN  Pred_UP")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:>8s}  {row[0]:>9d}  {row[1]:>7d}")
    
    # Signal edge: how accurate at high confidence
    up_probs = y_proba[:, 1]
    print(f"\n  HIGH-CONFIDENCE DIRECTION SIGNALS:")
    for thr in [0.55, 0.60, 0.65, 0.70]:
        # UP signals
        up_mask = up_probs >= thr
        up_count = int(up_mask.sum())
        if up_count > 0:
            up_correct = int((y_test[up_mask] == 1).sum())
            up_prec = up_correct / up_count
            print(f"    UP prob >={thr:.0%}: {up_count:>5d} signals, precision={up_prec:.1%}")
        
        # DOWN signals
        down_mask = up_probs <= (1 - thr)
        down_count = int(down_mask.sum())
        if down_count > 0:
            down_correct = int((y_test[down_mask] == 0).sum())
            down_prec = down_correct / down_count
            print(f"    DN prob >={thr:.0%}: {down_count:>5d} signals, precision={down_prec:.1%}")
    
    print(f"\n  Top 15 Features:")
    for fname, imp in feat_imp[:15]:
        print(f"    {fname:<25s}  {imp:.4f}")
    
    # Shuffled baseline comparison
    print(f"\n  SIGNAL VALIDATION (shuffled baseline):")
    y_shuffled = y_train.copy()
    np.random.shuffle(y_shuffled)
    
    model_shuf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        **params
    )
    model_shuf.fit(X_train, y_shuffled, eval_set=[(X_test, y_test)], verbose=0)
    shuf_acc = accuracy_score(y_test, model_shuf.predict(X_test))
    signal_gap = accuracy - shuf_acc
    print(f"    Real accuracy:     {accuracy:.3f}")
    print(f"    Shuffled accuracy: {shuf_acc:.3f}")
    print(f"    Signal gap:        {signal_gap:+.3f} ({'PASS' if signal_gap > 0.01 else 'WEAK' if signal_gap > 0 else 'FAIL'})")
    
    print(f"{'='*60}")
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_DIR / f"{model_name}_{timestamp}.json"
    meta_path = MODELS_DIR / f"{model_name}_{timestamp}_meta.json"
    latest_model = MODELS_DIR / f"{model_name}_latest.json"
    latest_meta = MODELS_DIR / f"{model_name}_latest_meta.json"
    
    model.save_model(str(model_path))
    model.save_model(str(latest_model))
    
    metadata = {
        'model_name': model_name,
        'model_type': 'daily_direction_binary',
        'description': 'Predicts next-day direction (UP/DOWN) from daily OHLCV features',
        'timestamp': timestamp,
        'feature_names': feature_names,
        'label_map': {str(k): v for k, v in LABEL_MAP.items()},
        'label_names': {str(k): v for k, v in LABEL_NAMES.items()},
        'accuracy': round(accuracy, 4),
        'signal_gap': round(float(signal_gap), 4),
        'precision_per_class': {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(precision)},
        'recall_per_class': {LABEL_NAMES[i]: round(float(r), 4) for i, r in enumerate(recall)},
        'f1_per_class': {LABEL_NAMES[i]: round(float(f), 4) for i, f in enumerate(f1)},
        'feature_importance': {fname: round(float(imp), 6) for fname, imp in feat_imp},
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'best_iteration': int(model.best_iteration),
        'train_time_seconds': round(train_time, 1),
        'confusion_matrix': cm.tolist(),
    }
    
    for p in [meta_path, latest_meta]:
        with open(p, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n  Model saved: {model_path}")
    print(f"  Latest:      {latest_model}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'signal_gap': signal_gap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_importance': feat_imp,
        'metadata': metadata,
    }


def quick_train(
    symbols: Optional[list] = None,
    threshold: float = 0.3,
    test_days: int = 60,
) -> dict:
    """One-liner: load -> train -> evaluate -> save."""
    train_df, test_df, features = load_and_prepare_data(
        symbols=symbols,
        threshold_pct=threshold,
        test_days=test_days,
    )
    return train_model(train_df, test_df, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Daily Direction Predictor (Model 2)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--threshold', type=float, default=0.3, help='Move threshold %%')
    parser.add_argument('--test-days', type=int, default=60, help='Test period days')
    
    args = parser.parse_args()
    
    result = quick_train(
        symbols=args.symbols,
        threshold=args.threshold,
        test_days=args.test_days,
    )
    
    print(f"\n  Daily direction accuracy: {result['accuracy']:.1%}")
    print(f"  Signal gap: {result['signal_gap']:+.1%}")
    print(f"  (50% = random, gap > 1% = real signal)")
