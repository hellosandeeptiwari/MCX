"""
XGBOOST TRAINER — Train, evaluate, and save Move Predictor models

Pipeline:
  1. Load parquet candles for all symbols
  2. Compute features (feature_engineering.py)
  3. Create labels (label_creator.py)
  4. Train/test split (last 20 days = test, rest = train)
  5. Train XGBoost classifier
  6. Evaluate metrics (accuracy, precision, recall, F1, confusion matrix)
  7. Save model + metadata

Handles class imbalance via scale_pos_weight and stratified splits.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from .feature_engineering import compute_features, get_feature_names
from .label_creator import create_labels, label_distribution


# === CONFIG ===

MODELS_DIR = Path(__file__).parent / "saved_models"
DATA_DIR = Path(__file__).parent / "data" / "candles_5min"
REPORTS_DIR = Path(__file__).parent / "reports"

DEFAULT_PARAMS = {
    'objective': 'multi:softprob',    # 3-class: UP, DOWN, FLAT
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 6,                  # Deeper for 3-class with 53 features
    'learning_rate': 0.03,           # Faster convergence (0.01 was too slow)
    'n_estimators': 2000,            # More room for complex patterns
    'min_child_weight': 10,          # Less aggressive → let model split more
    'subsample': 0.75,               # 75% row sampling per tree
    'colsample_bytree': 0.75,        # 75% feature sampling per tree
    'gamma': 1.0,                    # Reduced pruning (was 2.0)
    'reg_alpha': 0.5,                # Less L1
    'reg_lambda': 1.5,               # Less L2
    'random_state': 42,
    'early_stopping_rounds': 80,     # Faster convergence → less patience needed
    'verbosity': 1,
}

# 3-class label mapping: label_creator gives -1/0/+1, XGBoost needs 0/1/2
# DOWN(-1)->0, FLAT(0)->1, UP(+1)->2
LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # DOWN=0, FLAT=1, UP=2
LABEL_NAMES = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 3

# Default threshold for "big move" — 0.5% in 30 min
DEFAULT_THRESHOLD = 0.5


def load_and_prepare_data(
    symbols: Optional[list] = None,
    lookahead_candles: int = 6,
    threshold_pct: float = 0.5,
    test_days: int = 20,
    val_days: int = 10,
    use_atr_threshold: bool = True,
    atr_factor: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Load candles, compute features, create labels, split train/val/test.
    
    Returns:
        (train_df, val_df, test_df, feature_names)
    """
    if symbols is None:
        # Load all available parquet files
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"No data directory found: {DATA_DIR}")
        symbols = [f.stem for f in DATA_DIR.glob("*.parquet")]
    
    # Exclude index symbols from training (they contaminate stock models)
    _index_symbols = {'NIFTY50', 'NIFTY_50', 'NIFTY 50', 'BANKNIFTY', 'FINNIFTY'}
    symbols = [s for s in symbols if s not in _index_symbols]
    
    if not symbols:
        raise ValueError("No symbols found. Run data_fetcher first.")
    
    print(f"\n📊 Loading data for {len(symbols)} symbols...")
    
    # Try loading daily candles for context features
    daily_data = {}
    try:
        from .data_fetcher import load_all_daily
        daily_data = load_all_daily(symbols)
        if daily_data:
            print(f"   Daily context: {len(daily_data)} symbols with {next(iter(daily_data.values()), pd.DataFrame()).shape[0]} days")
    except Exception:
        pass
    
    # Try loading OI snapshots for OI context features
    oi_data = {}
    try:
        from .oi_collector import load_all_oi_snapshots
        oi_data = load_all_oi_snapshots(symbols)
        if oi_data:
            print(f"   OI context: {len(oi_data)} symbols with snapshots")
        else:
            print(f"   OI context: No snapshots yet (will fill with 0s)")
    except Exception:
        pass
    
    # Try loading futures OI features (from DhanHQ backfill)
    futures_oi_data = {}
    try:
        from dhan_futures_oi import load_all_futures_oi_daily
        futures_oi_data = load_all_futures_oi_daily(symbols)
        if futures_oi_data:
            sample_len = next(iter(futures_oi_data.values()), pd.DataFrame()).shape[0]
            print(f"   Futures OI context: {len(futures_oi_data)} symbols with ~{sample_len} days")
        else:
            print(f"   Futures OI context: No data yet (will fill with 0s)")
    except Exception as e:
        print(f"   Futures OI context: Load failed ({e}), filling with 0s")
    
    # Load NIFTY50 market context data (shared across all symbols)
    nifty_5min_df = None
    nifty_daily_df = None
    try:
        nifty_5min_path = DATA_DIR / "NIFTY50.parquet"
        nifty_daily_path = DATA_DIR.parent / "candles_daily" / "NIFTY50.parquet"
        if nifty_5min_path.exists():
            nifty_5min_df = pd.read_parquet(nifty_5min_path)
            nifty_5min_df['date'] = pd.to_datetime(nifty_5min_df['date'])
            print(f"   NIFTY50 5-min: {len(nifty_5min_df):,} candles")
        else:
            print(f"   NIFTY50 5-min: Not found (run fetch_nifty.py)")
        if nifty_daily_path.exists():
            nifty_daily_df = pd.read_parquet(nifty_daily_path)
            nifty_daily_df['date'] = pd.to_datetime(nifty_daily_df['date'])
            print(f"   NIFTY50 daily: {len(nifty_daily_df):,} candles")
        else:
            print(f"   NIFTY50 daily: Not found")
    except Exception as e:
        print(f"   NIFTY50 context: Load failed ({e})")
    
    all_dfs = []
    
    for sym in symbols:
        fpath = DATA_DIR / f"{sym}.parquet"
        if not fpath.exists():
            print(f"  ⚠ {sym}: no parquet file, skipping")
            continue
        
        df = pd.read_parquet(fpath)
        if len(df) < 100:
            print(f"  ⚠ {sym}: only {len(df)} candles, skipping")
            continue
        
        # Compute features (with daily + OI + futures OI + NIFTY context if available)
        daily_df = daily_data.get(sym)
        oi_df = oi_data.get(sym)
        futures_oi_df = futures_oi_data.get(sym)
        df = compute_features(df, symbol=sym, daily_df=daily_df, oi_df=oi_df, futures_oi_df=futures_oi_df,
                              nifty_5min_df=nifty_5min_df, nifty_daily_df=nifty_daily_df)
        if df.empty:
            continue
        
        # Create labels (3-class: UP=+1, DOWN=-1, FLAT=0)
        # ATR-normalized threshold: each stock's threshold adapts to its volatility
        df = create_labels(df, lookahead_candles=lookahead_candles, threshold_pct=threshold_pct,
                           use_directional=True,
                           use_atr_threshold=use_atr_threshold, atr_factor=atr_factor)
        
        # Add symbol column
        df['symbol'] = sym
        all_dfs.append(df)
        print(f"  ✓ {sym}: {len(df):,} candles")
    
    if not all_dfs:
        raise ValueError("No data loaded successfully.")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter: only 'clean' labels, drop NaN labels
    combined = combined[combined['label_quality'] == 'clean']
    combined = combined.dropna(subset=['label'])
    
    # Map labels to 0-indexed
    combined['label_idx'] = combined['label'].map(LABEL_MAP)
    
    print(f"\nTotal usable samples: {len(combined):,}")
    label_distribution(combined)
    
    # Train/Val/Test split by date:
    #   Train: oldest data → (end - test_days - val_days)
    #   Val:   (end - test_days - val_days) → (end - test_days)  [for early stopping]
    #   Test:  (end - test_days) → latest  [truly unseen evaluation]
    combined['_date'] = combined['date'].dt.date
    all_dates = sorted(combined['_date'].unique())
    total_holdout = test_days + val_days
    
    if len(all_dates) <= total_holdout:
        print(f"⚠ Only {len(all_dates)} trading days — using 70/15/15 split")
        combined = combined.sort_values('date').reset_index(drop=True)
        n = len(combined)
        train_df = combined.iloc[:int(n * 0.70)]
        val_df = combined.iloc[int(n * 0.70):int(n * 0.85)]
        test_df = combined.iloc[int(n * 0.85):]
    else:
        test_cutoff = all_dates[-test_days]
        val_cutoff = all_dates[-(test_days + val_days)]
        train_df = combined[combined['_date'] < val_cutoff]
        val_df = combined[(combined['_date'] >= val_cutoff) & (combined['_date'] < test_cutoff)]
        test_df = combined[combined['_date'] >= test_cutoff]
        print(f"Train period: {all_dates[0]} → {val_cutoff - timedelta(days=1)}")
        print(f"Val period:   {val_cutoff} → {test_cutoff - timedelta(days=1)}  (early stopping)")
        print(f"Test period:  {test_cutoff} → {all_dates[-1]}  (unseen evaluation)")
    
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    feature_names = get_feature_names()
    
    return train_df, val_df, test_df, feature_names


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list,
    params: Optional[dict] = None,
    model_name: str = "move_predictor",
) -> dict:
    """Train XGBoost model and evaluate.
    
    Val set is used for early stopping (prevents test-set leakage).
    Test set is truly unseen for final evaluation.
    
    Returns:
        dict with model, metrics, feature importances
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    X_train = train_df[feature_names].values
    y_train = train_df['label_idx'].values.astype(int)
    X_val = val_df[feature_names].values
    y_val = val_df['label_idx'].values.astype(int)
    X_test = test_df[feature_names].values
    y_test = test_df['label_idx'].values.astype(int)
    
    print(f"\n🔧 Training XGBoost model (3-class: DOWN/FLAT/UP)...")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples (early stopping)")
    print(f"  Test:  {X_test.shape[0]:,} samples (unseen evaluation)")
    
    # Compute class weights for imbalanced 3-class data
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    total = len(y_train)
    
    print(f"  Class distribution (train):")
    for cls in range(NUM_CLASSES):
        print(f"    {LABEL_NAMES[cls]}: {class_counts[cls]:,} ({class_counts[cls]/total*100:.1f}%)")
    
    # Compute sample weights to balance classes
    class_weights = total / (NUM_CLASSES * np.maximum(class_counts, 1))
    sample_weights = np.array([class_weights[y] for y in y_train])
    # Extract XGBoost-specific params
    n_estimators = params.pop('n_estimators', 500)
    early_stopping = params.pop('early_stopping_rounds', 30)
    
    # Train
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        **params
    )
    
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Val set for early stopping (NOT test)
        verbose=50,
        sample_weight=sample_weights,
    )
    train_time = time.time() - t0
    
    # ── Isotonic Calibration (fitted on VAL set, applied to TEST predictions) ──
    print(f"\n  Fitting isotonic calibration on val set ({len(X_val):,} samples)...")
    val_proba_raw = model.predict_proba(X_val)  # shape (n_val, 3)
    
    calibrators = {}
    for cls in range(NUM_CLASSES):
        ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        # Binary target: 1 if true class == cls, else 0
        binary_target = (y_val == cls).astype(float)
        ir.fit(val_proba_raw[:, cls], binary_target)
        calibrators[cls] = ir
        # Show calibration effect
        raw_mean = val_proba_raw[:, cls].mean()
        cal_mean = ir.predict(val_proba_raw[:, cls]).mean()
        print(f"    {LABEL_NAMES[cls]}: raw_mean={raw_mean:.3f} → cal_mean={cal_mean:.3f}")
    
    print(f"  ✓ Isotonic calibrators fitted for {NUM_CLASSES} classes")
    
    # Predict on TEST set with calibration
    y_proba_raw = model.predict_proba(X_test)
    
    # Apply isotonic calibration
    y_proba = np.zeros_like(y_proba_raw)
    for cls in range(NUM_CLASSES):
        y_proba[:, cls] = calibrators[cls].predict(y_proba_raw[:, cls])
    
    # Re-normalize so probabilities sum to 1
    row_sums = y_proba.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)  # avoid div by zero
    y_proba = y_proba / row_sums
    
    # Calibrated predictions
    y_pred = np.argmax(y_proba, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics (3-class)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    recall = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
    
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"\n  Per-class metrics:")
    for cls in range(NUM_CLASSES):
        print(f"    {LABEL_NAMES[cls]:>10s}  P={precision[cls]:.3f}  R={recall[cls]:.3f}  F1={f1[cls]:.3f}")
    
    # Key metrics: directional F1s
    down_f1 = f1[0] if len(f1) > 0 else 0
    flat_f1 = f1[1] if len(f1) > 1 else 0
    up_f1 = f1[2] if len(f1) > 2 else 0
    macro_f1 = np.mean(f1)
    print(f"\n  ** DOWN F1 = {down_f1:.3f} | FLAT F1 = {flat_f1:.3f} | UP F1 = {up_f1:.3f} **")
    print(f"  ** Macro F1 = {macro_f1:.3f} ** (overall metric)")
    
    print(f"\n  Confusion Matrix:")
    header = ''.join(f"  Pred_{LABEL_NAMES[i]:>5s}" for i in range(NUM_CLASSES))
    print(f"  {'':>12s}{header}")
    for i, row in enumerate(cm):
        vals = ''.join(f"  {row[j]:>10d}" for j in range(NUM_CLASSES))
        print(f"  {LABEL_NAMES[i]:>12s}{vals}")
    
    print(f"\n  Top 10 Features:")
    for fname, imp in feat_imp[:10]:
        print(f"    {fname:<25s}  {imp:.4f}")
    print(f"{'='*60}\n")
    
    # --- High-confidence analysis (3-class) ---
    print(f"\n\U0001f4c8 HIGH-CONFIDENCE SIGNAL ANALYSIS:")
    # predict_proba returns shape (n, 3): [prob_down, prob_flat, prob_up]
    up_probs = y_proba[:, 2]   # UP class
    down_probs = y_proba[:, 0]  # DOWN class
    flat_probs = y_proba[:, 1]  # FLAT class
    
    print(f"  === UP signals ===")
    for thr in [0.30, 0.35, 0.40, 0.50, 0.60]:
        mask = up_probs >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test[mask] == 2).sum()
            prec = correct / count
            total_up = (y_test == 2).sum()
            print(f"    UP prob>={thr:.0%}: {count:>6d} signals, precision={prec:.1%}, recall={correct}/{total_up} ({correct/max(total_up,1)*100:.1f}%)")
    
    print(f"  === DOWN signals ===")
    for thr in [0.30, 0.35, 0.40, 0.50, 0.60]:
        mask = down_probs >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test[mask] == 0).sum()
            prec = correct / count
            total_down = (y_test == 0).sum()
            print(f"    DOWN prob>={thr:.0%}: {count:>6d} signals, precision={prec:.1%}, recall={correct}/{total_down} ({correct/max(total_down,1)*100:.1f}%)")
    
    print(f"  === FLAT signals ===")
    for thr in [0.50, 0.60, 0.70, 0.80]:
        mask = flat_probs >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test[mask] == 1).sum()
            prec = correct / count
            total_flat = (y_test == 1).sum()
            print(f"    FLAT prob>={thr:.0%}: {count:>6d} signals, precision={prec:.1%}, recall={correct}/{total_flat} ({correct/max(total_flat,1)*100:.1f}%)")
    
    # Save model + metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_DIR / f"{model_name}_{timestamp}.json"
    meta_path = MODELS_DIR / f"{model_name}_{timestamp}_meta.json"
    
    model.save_model(str(model_path))
    
    # Also save a "latest" symlink equivalent
    latest_model_path = MODELS_DIR / f"{model_name}_latest.json"
    latest_meta_path = MODELS_DIR / f"{model_name}_latest_meta.json"
    model.save_model(str(latest_model_path))
    
    # Save isotonic calibrators
    cal_path = MODELS_DIR / f"{model_name}_{timestamp}_calibrators.pkl"
    latest_cal_path = MODELS_DIR / f"{model_name}_latest_calibrators.pkl"
    joblib.dump(calibrators, str(cal_path))
    joblib.dump(calibrators, str(latest_cal_path))
    print(f"✅ Calibrators saved: {cal_path}")
    
    metadata = {
        'model_name': model_name,
        'model_type': '3-class',
        'num_classes': NUM_CLASSES,
        'timestamp': timestamp,
        'feature_names': feature_names,
        'label_map': {str(k): v for k, v in LABEL_MAP.items()},
        'label_names': {str(k): v for k, v in LABEL_NAMES.items()},
        'accuracy': round(accuracy, 4),
        'macro_f1': round(float(macro_f1), 4),
        'up_f1': round(float(up_f1), 4),
        'down_f1': round(float(down_f1), 4),
        'flat_f1': round(float(flat_f1), 4),
        'precision_per_class': {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(precision)},
        'recall_per_class': {LABEL_NAMES[i]: round(r, 4) for i, r in enumerate(recall)},
        'f1_per_class': {LABEL_NAMES[i]: round(f, 4) for i, f in enumerate(f1)},
        'feature_importance': {fname: round(float(imp), 6) for fname, imp in feat_imp},
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'best_iteration': int(model.best_iteration),
        'train_time_seconds': round(train_time, 1),
        'confusion_matrix': cm.tolist(),
        'calibrated': True,
        'calibration_method': 'isotonic',
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(latest_meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {meta_path}")
    print(f"✅ Latest model: {latest_model_path}")
    
    # Save report
    report_path = REPORTS_DIR / f"training_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(f"MOVE PREDICTOR TRAINING REPORT (3-CLASS: DOWN/FLAT/UP)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Train samples: {len(train_df):,}\n")
        f.write(f"Test samples: {len(test_df):,}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"UP F1: {up_f1:.4f} | DOWN F1: {down_f1:.4f} | FLAT F1: {flat_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in range(NUM_CLASSES)]))
        f.write(f"\n\nFeature Importances:\n")
        for fname, imp in feat_imp:
            f.write(f"  {fname:<25s}  {imp:.6f}\n")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'feature_importance': feat_imp,
        'metadata': metadata,
        'model_path': str(latest_model_path),
    }


def quick_train(
    symbols: Optional[list] = None,
    lookahead: int = 6,
    threshold: float = 0.5,
    test_days: int = 20,
    val_days: int = 10,
    use_atr_threshold: bool = True,
    atr_factor: float = 1.0,
) -> dict:
    """One-liner to load data → train → evaluate → save."""
    train_df, val_df, test_df, features = load_and_prepare_data(
        symbols=symbols,
        lookahead_candles=lookahead,
        threshold_pct=threshold,
        test_days=test_days,
        val_days=val_days,
        use_atr_threshold=use_atr_threshold,
        atr_factor=atr_factor,
    )
    return train_model(train_df, val_df, test_df, features)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Move Predictor model')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train on')
    parser.add_argument('--lookahead', type=int, default=6, help='Lookahead candles (default: 6)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Move threshold %% (fallback when no ATR)')
    parser.add_argument('--test-days', type=int, default=20, help='Test period days (default: 20)')
    parser.add_argument('--val-days', type=int, default=10, help='Validation period days (default: 10)')
    parser.add_argument('--atr-factor', type=float, default=1.0, help='ATR multiplier for threshold (default: 1.0)')
    parser.add_argument('--fixed-threshold', action='store_true', help='Use fixed threshold instead of ATR-normalized')
    
    args = parser.parse_args()
    
    result = quick_train(
        symbols=args.symbols,
        lookahead=args.lookahead,
        threshold=args.threshold,
        test_days=args.test_days,
        val_days=args.val_days,
        use_atr_threshold=not args.fixed_threshold,
        atr_factor=args.atr_factor,
    )
    
    print(f"\n🎯 Final accuracy: {result['accuracy']:.1%}")
    print(f"   Macro F1: {result.get('macro_f1', 0):.3f}")
    up_f1 = result.get('f1', [0, 0, 0])
    print(f"   DOWN F1: {up_f1[0]:.3f} | FLAT F1: {up_f1[1]:.3f} | UP F1: {up_f1[2]:.3f}")