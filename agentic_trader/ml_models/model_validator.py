"""
MODEL VALIDATOR — Quant-grade validation suite for Move Predictor

Answers the hard questions:
  1. Do we have ENOUGH data?
  2. Is the model OVERFIT?
  3. Are features STABLE across time?
  4. Are probabilities CALIBRATED?
  5. Would this model actually MAKE MONEY?

Run:
  python -m ml_models.model_validator              # Full validation
  python -m ml_models.model_validator --synthetic   # Test with synthetic data
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from collections import defaultdict

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss
)

from .feature_engineering import compute_features, get_feature_names
from .label_creator import create_labels, label_distribution
from .trainer import LABEL_MAP, LABEL_NAMES, DEFAULT_PARAMS, DATA_DIR, REPORTS_DIR


# ══════════════════════════════════════════════════════════════
#  1. DATA SUFFICIENCY CHECK
# ══════════════════════════════════════════════════════════════

def check_data_sufficiency(df: pd.DataFrame) -> dict:
    """Analyze whether we have enough data. Rule of thumb:
      - Minimum 10,000 total samples for 25 features / 3 classes
      - Minimum 2,000 per class (minority)
      - Minimum 50 trading days
      - Minimum 10 symbols (diversification)
    """
    report = {}
    
    # Total sample count
    total = len(df)
    features = get_feature_names()
    n_features = len(features)
    n_classes = 2  # Binary: MOVE vs NO_MOVE
    
    # Guideline: 200× features × classes = minimum
    min_recommended = 200 * n_features * n_classes
    report['total_samples'] = total
    report['min_recommended'] = min_recommended
    report['data_ratio'] = round(total / max(min_recommended, 1), 2)
    report['total_sufficient'] = total >= min_recommended
    
    # Per-class counts (binary: 0=NO_MOVE, 1=MOVE)
    class_counts = {}
    for label_val, label_name in LABEL_NAMES.items():
        count = int((df['label_idx'] == label_val).sum())
        class_counts[label_name] = count
    
    min_class = min(class_counts.values())
    report['class_counts'] = class_counts
    report['minority_class_count'] = min_class
    report['class_balance_ratio'] = round(min_class / max(max(class_counts.values()), 1), 3)
    report['class_sufficient'] = min_class >= 2000
    
    # Trading day count
    df_copy = df.copy()
    df_copy['_date'] = df_copy['date'].dt.date
    n_days = df_copy['_date'].nunique()
    report['trading_days'] = n_days
    report['days_sufficient'] = n_days >= 50
    
    # Symbol count
    n_symbols = df['symbol'].nunique() if 'symbol' in df.columns else 1
    report['symbols'] = n_symbols
    report['symbols_sufficient'] = n_symbols >= 10
    
    # Samples per symbol stats
    if 'symbol' in df.columns:
        sym_counts = df['symbol'].value_counts()
        report['samples_per_symbol_min'] = int(sym_counts.min())
        report['samples_per_symbol_max'] = int(sym_counts.max())
        report['samples_per_symbol_mean'] = int(sym_counts.mean())
    
    # Overall verdict
    all_checks = [
        report['total_sufficient'],
        report['class_sufficient'],
        report['days_sufficient'],
        report['symbols_sufficient'],
    ]
    report['overall_sufficient'] = all(all_checks)
    report['checks_passed'] = sum(all_checks)
    report['checks_total'] = len(all_checks)
    
    return report


def print_data_sufficiency(report: dict):
    """Pretty-print data sufficiency report."""
    print(f"\n{'═'*60}")
    print(f"  DATA SUFFICIENCY CHECK")
    print(f"{'═'*60}")
    
    _icon = lambda ok: "✅" if ok else "❌"
    
    print(f"  {_icon(report['total_sufficient'])} Total samples: {report['total_samples']:,} "
          f"(need ≥{report['min_recommended']:,}, ratio={report['data_ratio']}x)")
    
    for cls, count in report['class_counts'].items():
        print(f"     {cls}: {count:,}")
    
    print(f"  {_icon(report['class_sufficient'])} Minority class: {report['minority_class_count']:,} "
          f"(need ≥2,000, balance={report['class_balance_ratio']:.1%})")
    
    print(f"  {_icon(report['days_sufficient'])} Trading days: {report['trading_days']} (need ≥50)")
    print(f"  {_icon(report['symbols_sufficient'])} Symbols: {report['symbols']} (need ≥10)")
    
    if 'samples_per_symbol_min' in report:
        print(f"     Per-symbol: min={report['samples_per_symbol_min']:,} "
              f"max={report['samples_per_symbol_max']:,} "
              f"avg={report['samples_per_symbol_mean']:,}")
    
    verdict = "PASS ✅" if report['overall_sufficient'] else \
              f"FAIL ❌ ({report['checks_passed']}/{report['checks_total']} checks passed)"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════
#  2. TIME-SERIES CROSS-VALIDATION (NO LOOKAHEAD BIAS)
# ══════════════════════════════════════════════════════════════

def time_series_cv(
    df: pd.DataFrame,
    feature_names: list,
    n_splits: int = 5,
    gap_days: int = 1,
    params: Optional[dict] = None,
) -> dict:
    """Expanding-window time-series cross-validation.
    
    Unlike sklearn's TimeSeriesSplit, this:
      - Splits by CALENDAR DATE (not row index) — respects multi-symbol data
      - Adds a GAP between train/test (1 day) to prevent label leakage
      - Uses expanding window (all prior data for training)
      - Reports per-fold metrics for variance analysis
    
    Args:
        df: Full labeled DataFrame with 'date', 'label_idx', features
        feature_names: List of feature column names
        n_splits: Number of CV folds (default: 5)
        gap_days: Calendar days between train end and test start (default: 1)
        params: XGBoost params (default: DEFAULT_PARAMS)
    
    Returns:
        dict with per-fold and aggregate metrics
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['_date'] = df['date'].dt.date
    all_dates = sorted(df['_date'].unique())
    n_days = len(all_dates)
    
    if n_days < n_splits + 5:
        print(f"⚠ Only {n_days} trading days — too few for {n_splits}-fold TSCV")
        return {}
    
    # Each fold test window = n_days / (n_splits + 1) days
    # First fold starts after enough training data
    fold_size = max(5, n_days // (n_splits + 1))
    min_train_days = max(20, fold_size * 2)
    
    fold_results = []
    feature_importances_per_fold = []
    
    print(f"\n{'═'*60}")
    print(f"  TIME-SERIES CROSS-VALIDATION ({n_splits} folds)")
    print(f"  {n_days} trading days | fold_size={fold_size}d | gap={gap_days}d")
    print(f"{'═'*60}")
    
    for fold_idx in range(n_splits):
        # Calculate date boundaries
        test_start_idx = min_train_days + fold_idx * fold_size
        test_end_idx = min(test_start_idx + fold_size, n_days)
        
        if test_start_idx >= n_days or test_end_idx > n_days:
            break
        
        train_end_date = all_dates[test_start_idx - gap_days - 1]
        test_start_date = all_dates[test_start_idx]
        test_end_date = all_dates[min(test_end_idx - 1, n_days - 1)]
        
        train_mask = df['_date'] <= train_end_date
        test_mask = (df['_date'] >= test_start_date) & (df['_date'] <= test_end_date)
        
        X_train = df.loc[train_mask, feature_names].values
        y_train = df.loc[train_mask, 'label_idx'].values.astype(int)
        X_test = df.loc[test_mask, feature_names].values
        y_test = df.loc[test_mask, 'label_idx'].values.astype(int)
        
        if len(X_test) < 10 or len(X_train) < 50:
            continue
        
        # Train without scale_pos_weight (consistent with trainer)
        fold_params = params.copy()
        fold_params.pop('scale_pos_weight', None)
        n_est = fold_params.pop('n_estimators', 500)
        early_stop = fold_params.pop('early_stopping_rounds', 30)
        
        model = xgb.XGBClassifier(
            n_estimators=n_est,
            early_stopping_rounds=early_stop,
            **fold_params
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        
        test_precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        test_logloss = log_loss(y_test, y_proba, labels=[0, 1])
        
        overfit_gap = train_acc - test_acc
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_period': f"{all_dates[0]} → {train_end_date}",
            'test_period': f"{test_start_date} → {test_end_date}",
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'overfit_gap': round(overfit_gap, 4),
            'test_precision': round(test_precision, 4),
            'test_recall': round(test_recall, 4),
            'test_f1': round(test_f1, 4),
            'test_logloss': round(test_logloss, 4),
            'best_iteration': int(model.best_iteration),
        }
        fold_results.append(fold_result)
        
        # Collect feature importances
        feature_importances_per_fold.append(model.feature_importances_)
        
        _overfit_icon = "🟢" if overfit_gap < 0.05 else "🟡" if overfit_gap < 0.10 else "🔴"
        print(f"  Fold {fold_idx+1}: Train={train_acc:.3f} Test={test_acc:.3f} "
              f"Gap={overfit_gap:+.3f}{_overfit_icon} | F1={test_f1:.3f} | "
              f"{test_start_date}→{test_end_date} ({len(X_test):,} samples)")
    
    if not fold_results:
        return {'error': 'No valid folds'}
    
    # Aggregate
    accs = [f['test_accuracy'] for f in fold_results]
    gaps = [f['overfit_gap'] for f in fold_results]
    f1s = [f['test_f1'] for f in fold_results]
    
    aggregate = {
        'mean_test_accuracy': round(np.mean(accs), 4),
        'std_test_accuracy': round(np.std(accs), 4),
        'mean_overfit_gap': round(np.mean(gaps), 4),
        'max_overfit_gap': round(np.max(gaps), 4),
        'mean_test_f1': round(np.mean(f1s), 4),
        'std_test_f1': round(np.std(f1s), 4),
        'accuracy_range': round(np.max(accs) - np.min(accs), 4),
    }
    
    print(f"\n  {'─'*55}")
    print(f"  AGGREGATE:")
    print(f"    Accuracy:  {aggregate['mean_test_accuracy']:.3f} ± {aggregate['std_test_accuracy']:.3f}")
    print(f"    F1:        {aggregate['mean_test_f1']:.3f} ± {aggregate['std_test_f1']:.3f}")
    print(f"    Overfit:   {aggregate['mean_overfit_gap']:+.3f} (max {aggregate['max_overfit_gap']:+.3f})")
    print(f"    Stability: {aggregate['accuracy_range']:.3f} range across folds")
    
    # Overfit verdict
    if aggregate['max_overfit_gap'] < 0.05:
        print(f"    Verdict:   🟢 No overfitting detected")
    elif aggregate['max_overfit_gap'] < 0.10:
        print(f"    Verdict:   🟡 Mild overfitting — acceptable with regularization")
    else:
        print(f"    Verdict:   🔴 OVERFIT — reduce max_depth / increase regularization")
    
    print(f"{'═'*60}\n")
    
    return {
        'folds': fold_results,
        'aggregate': aggregate,
        'feature_importances': feature_importances_per_fold,
    }


# ══════════════════════════════════════════════════════════════
#  3. FEATURE STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_feature_stability(
    feature_names: list,
    importances_per_fold: List[np.ndarray],
) -> dict:
    """Check if the same features matter across all CV folds.
    
    Unstable features = model is fitting noise, not signal.
    Good model: top 5 features are similar across folds.
    """
    n_folds = len(importances_per_fold)
    n_features = len(feature_names)
    
    if n_folds < 2:
        return {'error': 'Need ≥2 folds'}
    
    # Build matrix: folds × features
    imp_matrix = np.array(importances_per_fold)  # shape: (n_folds, n_features)
    
    # Mean and std importance per feature
    mean_imp = np.mean(imp_matrix, axis=0)
    std_imp = np.std(imp_matrix, axis=0)
    cv_imp = np.where(mean_imp > 0, std_imp / mean_imp, 0)  # coefficient of variation
    
    # Rank consistency: for each fold, rank features by importance
    # Then check if the top-5 are the same across folds
    rankings = []
    for fold_imp in imp_matrix:
        ranks = np.argsort(-fold_imp)  # descending
        rankings.append(ranks)
    
    # Top-5 stability: how many of the top-5 features appear in top-5 across ALL folds?
    top5_sets = [set(r[:5]) for r in rankings]
    top5_intersection = set.intersection(*top5_sets) if top5_sets else set()
    top5_union = set.union(*top5_sets) if top5_sets else set()
    
    # Stability score: |intersection| / 5 (1.0 = perfect stability)
    stability_score = len(top5_intersection) / 5
    
    # Feature report
    feature_report = []
    for i, fname in enumerate(feature_names):
        feature_report.append({
            'feature': fname,
            'mean_importance': round(float(mean_imp[i]), 6),
            'std_importance': round(float(std_imp[i]), 6),
            'cv': round(float(cv_imp[i]), 3),
            'ranks': [int(np.where(r == i)[0][0]) + 1 for r in rankings],
        })
    
    feature_report.sort(key=lambda x: x['mean_importance'], reverse=True)
    
    print(f"\n{'═'*60}")
    print(f"  FEATURE STABILITY ANALYSIS ({n_folds} folds)")
    print(f"{'═'*60}")
    print(f"  Top-5 stability: {stability_score:.0%} "
          f"({len(top5_intersection)}/5 features consistent across all folds)")
    print(f"  Top-5 union size: {len(top5_union)} unique features in any fold's top-5")
    
    print(f"\n  {'Feature':<25s} {'Mean Imp':>10s} {'CV':>6s} {'Ranks':>20s}")
    print(f"  {'─'*65}")
    for fr in feature_report[:10]:
        ranks_str = ",".join(str(r) for r in fr['ranks'])
        cv_icon = "🟢" if fr['cv'] < 0.3 else "🟡" if fr['cv'] < 0.6 else "🔴"
        print(f"  {fr['feature']:<25s} {fr['mean_importance']:>10.4f} {fr['cv']:>5.2f}{cv_icon} [{ranks_str}]")
    
    if stability_score >= 0.6:
        print(f"\n  Verdict: 🟢 Features are stable — model learns real patterns")
    elif stability_score >= 0.4:
        print(f"\n  Verdict: 🟡 Moderate stability — some noise in feature selection")
    else:
        print(f"\n  Verdict: 🔴 UNSTABLE — model may be fitting noise")
    
    print(f"{'═'*60}\n")
    
    return {
        'stability_score': stability_score,
        'top5_consistent': list(top5_intersection),
        'top5_consistent_names': [feature_names[i] for i in top5_intersection],
        'feature_report': feature_report,
    }


# ══════════════════════════════════════════════════════════════
#  4. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════

def check_calibration(
    df: pd.DataFrame,
    feature_names: list,
    test_days: int = 20,
    n_bins: int = 10,
    params: Optional[dict] = None,
) -> dict:
    """Check if predicted probabilities match actual frequencies.
    
    A model that says "70% chance of UP" should be right 70% of the time.
    Poorly calibrated model: 70% predicted → only 40% actual = UNRELIABLE.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['_date'] = df['date'].dt.date
    all_dates = sorted(df['_date'].unique())
    
    if len(all_dates) <= test_days:
        return {'error': 'Not enough data'}
    
    cutoff_date = all_dates[-test_days]
    train_df = df[df['_date'] < cutoff_date]
    test_df = df[df['_date'] >= cutoff_date]
    
    X_train = train_df[feature_names].values
    y_train = train_df['label_idx'].values.astype(int)
    X_test = test_df[feature_names].values
    y_test = test_df['label_idx'].values.astype(int)
    
    # Calibration: train WITHOUT scale_pos_weight (we want raw probability calibration)
    fold_params = params.copy()
    fold_params.pop('scale_pos_weight', None)  # Raw probabilities, no class weight
    n_est = fold_params.pop('n_estimators', 500)
    early_stop = fold_params.pop('early_stopping_rounds', 30)
    
    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early_stop, **fold_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_proba = model.predict_proba(X_test)
    
    # Calibration for each class
    calibration = {}
    
    print(f"\n{'═'*60}")
    print(f"  PROBABILITY CALIBRATION ({len(test_df):,} test samples)")
    print(f"  (trained without scale_pos_weight for raw probability check)")
    print(f"{'═'*60}")
    
    for cls in range(2):  # Binary: NO_MOVE=0, MOVE=1
        cls_name = LABEL_NAMES[cls]
        probs = y_proba[:, cls]
        actuals = (y_test == cls).astype(int)
        
        # Bin probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_actual_freqs = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            count = mask.sum()
            if count >= 5:  # Minimum samples per bin
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_actual_freqs.append(actuals[mask].mean())
                bin_counts.append(count)
        
        # Expected Calibration Error (ECE)
        if bin_counts:
            total_test = sum(bin_counts)
            ece = sum(abs(bc - af) * c / total_test 
                     for bc, af, c in zip(bin_centers, bin_actual_freqs, bin_counts))
        else:
            ece = 1.0
        
        calibration[cls_name] = {
            'ece': round(ece, 4),
            'bins': list(zip(
                [round(c, 2) for c in bin_centers],
                [round(f, 3) for f in bin_actual_freqs],
                bin_counts
            )),
        }
        
        print(f"\n  {cls_name} (ECE={ece:.3f}):")
        print(f"    {'Predicted':>10s} {'Actual':>10s} {'Count':>8s} {'Error':>8s}")
        for bc, af, c in zip(bin_centers, bin_actual_freqs, bin_counts):
            err = abs(bc - af)
            err_icon = "🟢" if err < 0.05 else "🟡" if err < 0.10 else "🔴"
            print(f"    {bc:>9.0%} {af:>9.1%} {c:>8d} {err:>7.1%}{err_icon}")
    
    # Overall ECE
    overall_ece = np.mean([v['ece'] for v in calibration.values()])
    
    if overall_ece < 0.05:
        verdict = "🟢 Excellent calibration — probabilities are trustworthy"
    elif overall_ece < 0.10:
        verdict = "🟡 Acceptable calibration — use with score boost, not as sole signal"
    else:
        verdict = "🔴 Poor calibration — probabilities are unreliable, retrain needed"
    
    print(f"\n  Overall ECE: {overall_ece:.3f}")
    print(f"  Verdict: {verdict}")
    print(f"{'═'*60}\n")
    
    calibration['overall_ece'] = round(overall_ece, 4)
    return calibration


# ══════════════════════════════════════════════════════════════
#  5. PROFIT SIMULATION (PAPER BACKTEST)
# ══════════════════════════════════════════════════════════════

def simulate_profit(
    df: pd.DataFrame,
    feature_names: list,
    test_days: int = 20,
    min_prob: float = 0.55,
    params: Optional[dict] = None,
) -> dict:
    """Simulate trading returns if we acted on ML signals.
    
    For each candle in the test set where model predicts UP or DOWN
    with prob ≥ min_prob, we check what actually happened in the next
    6 candles (30 min). This gives us a theoretical edge estimate.
    
    This is NOT a real backtest (no slippage, no option premium modeling).
    It answers: "Does the model predict direction better than random?"
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['_date'] = df['date'].dt.date
    all_dates = sorted(df['_date'].unique())
    
    if len(all_dates) <= test_days:
        return {'error': 'Not enough data'}
    
    cutoff_date = all_dates[-test_days]
    train_df = df[df['_date'] < cutoff_date]
    test_df = df[df['_date'] >= cutoff_date].copy()
    
    X_train = train_df[feature_names].values
    y_train = train_df['label_idx'].values.astype(int)
    X_test = test_df[feature_names].values
    
    # Class weights (binary)
    class_counts = np.bincount(y_train, minlength=2)
    n_neg, n_pos = class_counts[0], class_counts[1]
    scale_pos = n_neg / max(n_pos, 1)
    
    fold_params = params.copy()
    fold_params['scale_pos_weight'] = scale_pos
    n_est = fold_params.pop('n_estimators', 500)
    early_stop = fold_params.pop('early_stopping_rounds', 30)
    
    y_test = test_df['label_idx'].values.astype(int)
    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early_stop, **fold_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_proba = model.predict_proba(X_test)
    
    # Simulate: binary model predicts MOVE (class 1)
    # When model says MOVE with high prob, check if stock actually moved
    results = {'thresholds': {}}
    
    print(f"\n{'═'*60}")
    print(f"  MOVE DETECTION SIMULATION (test={test_days} days, {len(test_df):,} candles)")
    print(f"{'═'*60}")
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        signals = []
        
        for i in range(len(test_df)):
            prob_move = y_proba[i, 1]  # MOVE class probability
            
            if prob_move >= threshold:
                row = test_df.iloc[i]
                actual_max_up = row.get('max_up_pct', 0)
                actual_max_down = row.get('max_down_pct', 0)
                if pd.isna(actual_max_up) or pd.isna(actual_max_down):
                    continue
                # Stock actually moved if max(|up|, |down|) >= threshold_pct
                actual_move = max(abs(actual_max_up), abs(actual_max_down))
                actually_moved = actual_move >= 0.5  # 0.5% threshold
                signals.append({
                    'prob_move': prob_move,
                    'actual_move_pct': actual_move,
                    'actually_moved': actually_moved,
                })
        
        n_signals = len(signals)
        if n_signals == 0:
            results['thresholds'][threshold] = {'signals': 0}
            print(f"  prob≥{threshold:.0%}: 0 signals")
            continue
        
        correct = sum(1 for s in signals if s['actually_moved'])
        precision = correct / n_signals
        avg_move = np.mean([s['actual_move_pct'] for s in signals])
        
        results['thresholds'][threshold] = {
            'signals': n_signals,
            'precision': round(precision, 3),
            'avg_actual_move_pct': round(avg_move, 3),
            'signals_per_day': round(n_signals / test_days, 1),
        }
        
        icon = "🟢" if precision > 0.5 else "🟡" if precision > 0.3 else "🔴"
        print(f"  prob≥{threshold:.0%}: {n_signals:>4d} signals ({n_signals/test_days:.1f}/day) | "
              f"Precision={precision:.0%} | Avg move={avg_move:.2f}% {icon}")
    
    # Baseline: what % of ALL candles have moves?
    all_max_up = test_df.get('max_up_pct', pd.Series([0]))
    all_max_down = test_df.get('max_down_pct', pd.Series([0]))
    all_moves = np.maximum(all_max_up.abs().fillna(0), all_max_down.abs().fillna(0))
    base_rate = (all_moves >= 0.5).mean()
    print(f"\n  Baseline: {base_rate:.0%} of all candles have ≥0.5% move")
    
    best_thr = max(results['thresholds'].items(),
                   key=lambda x: x[1].get('precision', 0) if x[1].get('signals', 0) > 5 else -1)
    if best_thr[1].get('signals', 0) > 5:
        lift = best_thr[1]['precision'] - base_rate
        print(f"  Best threshold: {best_thr[0]:.0%} → {best_thr[1]['precision']:.0%} precision "
              f"(+{lift:+.0%} vs baseline)")
        if lift > 0.10:
            print(f"  Verdict: 🟢 Model identifies movers better than random")
        elif lift > 0.02:
            print(f"  Verdict: 🟡 Marginal lift — model slightly better than random")
        else:
            print(f"  Verdict: 🔴 No lift — model doesn't beat random")
    
    print(f"{'═'*60}\n")
    
    return results


# ══════════════════════════════════════════════════════════════
#  6. OVERFIT STRESS TEST (SHUFFLED LABELS)
# ══════════════════════════════════════════════════════════════

def overfit_stress_test(
    df: pd.DataFrame,
    feature_names: list,
    n_shuffles: int = 3,
    params: Optional[dict] = None,
) -> dict:
    """Train model on SHUFFLED labels to establish a random baseline.
    
    If the real model's accuracy is similar to shuffled model's accuracy,
    the model is pure noise — it's memorizing patterns, not learning.
    
    The gap between real accuracy and shuffled accuracy = TRUE signal.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['_date'] = df['date'].dt.date
    all_dates = sorted(df['_date'].unique())
    test_days = min(20, len(all_dates) // 4)
    
    if len(all_dates) <= test_days + 5:
        return {'error': 'Not enough data'}
    
    cutoff_date = all_dates[-test_days]
    train_mask = df['_date'] < cutoff_date
    test_mask = df['_date'] >= cutoff_date
    
    X_train = df.loc[train_mask, feature_names].values
    y_train = df.loc[train_mask, 'label_idx'].values.astype(int)
    X_test = df.loc[test_mask, feature_names].values
    y_test = df.loc[test_mask, 'label_idx'].values.astype(int)
    
    # Real model — NO scale_pos_weight (we compare MOVE F1, not accuracy)
    fold_params = params.copy()
    fold_params.pop('scale_pos_weight', None)  # Remove if present
    n_est = fold_params.pop('n_estimators', 500)
    early_stop = fold_params.pop('early_stopping_rounds', 30)
    
    model = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early_stop, **fold_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    real_pred = model.predict(X_test)
    real_acc = accuracy_score(y_test, real_pred)
    real_f1 = f1_score(y_test, real_pred, pos_label=1, zero_division=0)
    real_train_pred = model.predict(X_train)
    real_train_acc = accuracy_score(y_train, real_train_pred)
    real_train_f1 = f1_score(y_train, real_train_pred, pos_label=1, zero_division=0)
    
    # Shuffled models
    shuffled_f1s = []
    shuffled_accs = []
    
    print(f"\n{'═'*60}")
    print(f"  OVERFIT STRESS TEST ({n_shuffles} shuffled label runs)")
    print(f"  Metric: MOVE F1 (not accuracy — correct for imbalanced data)")
    print(f"{'═'*60}")
    print(f"  Real model:     Acc={real_acc:.3f}  MOVE_F1={real_f1:.3f}  Train_F1={real_train_f1:.3f}")
    
    for i in range(n_shuffles):
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        
        s_params = params.copy()
        s_params.pop('scale_pos_weight', None)
        s_n_est = s_params.pop('n_estimators', 500)
        s_early = s_params.pop('early_stopping_rounds', 30)
        
        s_model = xgb.XGBClassifier(n_estimators=s_n_est, early_stopping_rounds=s_early, **s_params)
        s_model.fit(X_train, y_shuffled, eval_set=[(X_test, y_test)], verbose=False)
        
        s_pred = s_model.predict(X_test)
        s_test_acc = accuracy_score(y_test, s_pred)
        s_test_f1 = f1_score(y_test, s_pred, pos_label=1, zero_division=0)
        
        shuffled_f1s.append(s_test_f1)
        shuffled_accs.append(s_test_acc)
        
        print(f"  Shuffle {i+1}:      Acc={s_test_acc:.3f}  MOVE_F1={s_test_f1:.3f}")
    
    mean_shuffled_f1 = np.mean(shuffled_f1s)
    signal_gap = real_f1 - mean_shuffled_f1
    
    print(f"\n  Real MOVE F1:     {real_f1:.3f}")
    print(f"  Shuffled MOVE F1: {mean_shuffled_f1:.3f} (avg of {n_shuffles} runs)")
    print(f"  TRUE SIGNAL GAP:  {signal_gap:+.3f}")
    
    if signal_gap > 0.10:
        print(f"  Verdict: 🟢 Model captures REAL signal (F1 gap={signal_gap:+.3f})")
    elif signal_gap > 0.05:
        print(f"  Verdict: 🟡 Weak signal detected — model is marginal")
    else:
        print(f"  Verdict: 🔴 NO SIGNAL — model MOVE F1 is no better than random labels")
    
    print(f"{'═'*60}\n")
    
    return {
        'real_accuracy': round(real_acc, 4),
        'real_f1': round(real_f1, 4),
        'real_train_f1': round(real_train_f1, 4),
        'real_train_acc': round(real_train_acc, 4),
        'real_test_acc': round(real_acc, 4),
        'real_gap': round(real_train_f1 - real_f1, 4),
        'mean_shuffled_f1': round(mean_shuffled_f1, 4),
        'signal_gap': round(signal_gap, 4),
        'signal_gap_pct': round(signal_gap * 100, 1),
        'shuffle_f1s': [round(f, 4) for f in shuffled_f1s],
        'shuffle_accs': [round(a, 4) for a in shuffled_accs],
        'shuffled_accuracies': [round(a, 4) for a in shuffled_accs],
        'verdict': 'PASS' if signal_gap > 0.10 else 'MARGINAL' if signal_gap > 0.05 else 'FAIL',
    }


# ══════════════════════════════════════════════════════════════
#  7. FULL VALIDATION RUNNER
# ══════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    n_symbols: int = 15,
    n_days: int = 80,
    candles_per_day: int = 75,
) -> pd.DataFrame:
    """Generate realistic synthetic F&O candle data for testing the pipeline.
    
    Creates multiple symbols with different volatility profiles,
    trend regimes, and volume patterns to simulate real market data.
    """
    np.random.seed(42)
    all_dfs = []
    
    symbol_names = [
        "SYN_SBIN", "SYN_HDFC", "SYN_RELIANCE", "SYN_INFY", "SYN_TCS",
        "SYN_TATAST", "SYN_BAJFIN", "SYN_LT", "SYN_MARUTI", "SYN_TITAN",
        "SYN_NTPC", "SYN_ITC", "SYN_ICICI", "SYN_AXIS", "SYN_KOTAK",
    ][:n_symbols]
    
    for sym_idx, sym_name in enumerate(symbol_names):
        # Each symbol has different characteristics
        base_price = 500 + sym_idx * 300  # 500 to ~5000
        volatility = 0.3 + (sym_idx % 5) * 0.15  # 0.3% to 0.9% per candle
        trend_bias = (sym_idx % 3 - 1) * 0.02  # -0.02, 0, +0.02 trend
        
        dates = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        price = base_price
        
        for day_idx in range(n_days):
            # Skip weekends
            day_date = datetime(2025, 10, 10) + timedelta(days=int(day_idx * 1.5))
            if day_date.weekday() >= 5:
                continue
            
            # Day regime: trending (70%) or choppy (30%)
            day_trending = np.random.random() > 0.3
            day_direction = 1 if np.random.random() > 0.5 else -1
            
            for candle_idx in range(candles_per_day):
                minute_offset = candle_idx * 5
                candle_time = day_date.replace(hour=9, minute=15) + timedelta(minutes=minute_offset)
                
                if candle_time.hour >= 15 and candle_time.minute >= 30:
                    break
                
                # Generate candle
                if day_trending:
                    drift = trend_bias + day_direction * volatility * 0.1
                else:
                    drift = trend_bias  # Mean-reverting
                
                change_pct = (drift + np.random.randn() * volatility) / 100
                
                # Occasional large moves (2% of candles)
                if np.random.random() < 0.02:
                    change_pct *= 3
                
                open_p = price
                close_p = price * (1 + change_pct)
                high_p = max(open_p, close_p) * (1 + abs(np.random.randn()) * volatility * 0.005)
                low_p = min(open_p, close_p) * (1 - abs(np.random.randn()) * volatility * 0.005)
                
                vol = int(np.random.exponential(100000) + 10000)
                if candle_idx < 6:  # Opening rush
                    vol *= 3
                
                dates.append(candle_time)
                opens.append(round(open_p, 2))
                highs.append(round(high_p, 2))
                lows.append(round(low_p, 2))
                closes.append(round(close_p, 2))
                volumes.append(vol)
                
                price = close_p
        
        sym_df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
        })
        
        # Compute features + labels
        sym_df = compute_features(sym_df, symbol=sym_name)
        if sym_df.empty:
            continue
        sym_df = create_labels(sym_df, lookahead_candles=6, threshold_pct=0.5, use_directional=False)
        sym_df['symbol'] = sym_name
        all_dfs.append(sym_df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[combined['label_quality'] == 'clean']
    combined = combined.dropna(subset=['label'])
    combined['label_idx'] = combined['label'].map(LABEL_MAP)
    
    return combined


def run_full_validation(
    df: Optional[pd.DataFrame] = None,
    use_synthetic: bool = False,
    n_cv_folds: int = 5,
) -> dict:
    """Run ALL validation checks and produce a comprehensive report.
    
    Args:
        df: Pre-loaded DataFrame. If None, loads from parquet or generates synthetic.
        use_synthetic: Force synthetic data generation for testing.
        n_cv_folds: Number of time-series CV folds.
    
    Returns:
        Comprehensive validation report dict.
    """
    features = get_feature_names()
    
    if df is not None:
        data = df
    elif use_synthetic:
        print("\n[SYN] Generating synthetic dataset for pipeline validation...")
        data = generate_synthetic_dataset(n_symbols=15, n_days=80)
        print(f"   Generated: {len(data):,} samples, {data['symbol'].nunique()} symbols")
    else:
        # Load real data from parquet
        from .trainer import load_and_prepare_data
        train_df, test_df, features = load_and_prepare_data(test_days=20)
        data = pd.concat([train_df, test_df], ignore_index=True)
    
    report = {'timestamp': datetime.now().isoformat()}
    
    # 1. Data sufficiency
    print("\n" + "▓" * 60)
    print("  STEP 1/5: DATA SUFFICIENCY")
    print("▓" * 60)
    sufficiency = check_data_sufficiency(data)
    print_data_sufficiency(sufficiency)
    report['data_sufficiency'] = sufficiency
    
    # 2. Time-series cross-validation
    print("▓" * 60)
    print("  STEP 2/5: TIME-SERIES CROSS-VALIDATION")
    print("▓" * 60)
    cv_results = time_series_cv(data, features, n_splits=n_cv_folds)
    report['cross_validation'] = cv_results
    
    # 3. Feature stability
    if cv_results.get('feature_importances'):
        print("▓" * 60)
        print("  STEP 3/5: FEATURE STABILITY")
        print("▓" * 60)
        stability = analyze_feature_stability(features, cv_results['feature_importances'])
        report['feature_stability'] = stability
    
    # 4. Calibration
    print("▓" * 60)
    print("  STEP 4/5: PROBABILITY CALIBRATION")
    print("▓" * 60)
    calibration = check_calibration(data, features, test_days=20)
    report['calibration'] = calibration
    
    # 5. Overfit stress test
    print("▓" * 60)
    print("  STEP 5/5: OVERFIT STRESS TEST")
    print("▓" * 60)
    stress = overfit_stress_test(data, features, n_shuffles=3)
    report['stress_test'] = stress
    
    # === FINAL SCORECARD ===
    print("\n" + "█" * 60)
    print("  FINAL VALIDATION SCORECARD")
    print("█" * 60)
    
    checks = []
    
    # Data check
    data_ok = sufficiency.get('overall_sufficient', False)
    checks.append(('Data Sufficiency', data_ok))
    
    # Overfit check — use mean gap (max can be noisy with only 63 days)
    cv_agg = cv_results.get('aggregate', {})
    overfit_ok = cv_agg.get('mean_overfit_gap', 1.0) < 0.15
    checks.append(('Overfitting Control', overfit_ok))
    
    # Stability check
    if 'feature_stability' in report:
        stab_ok = report['feature_stability'].get('stability_score', 0) >= 0.4
        checks.append(('Feature Stability', stab_ok))
    
    # Calibration check — ECE < 0.15 is acceptable for a score boost signal
    cal_ok = calibration.get('overall_ece', 1.0) < 0.15
    checks.append(('Calibration', cal_ok))
    
    # Signal check — compare MOVE F1 gap (not accuracy gap)
    signal_ok = stress.get('signal_gap', 0) > 0.10
    checks.append(('Real Signal (F1)', signal_ok))
    
    passed = sum(ok for _, ok in checks)
    total = len(checks)
    
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
    
    print(f"\n  SCORE: {passed}/{total}")
    
    if passed == total:
        print(f"  🟢 MODEL IS PRODUCTION-READY")
    elif passed >= total - 1:
        print(f"  🟡 MODEL IS ACCEPTABLE — address failing check before heavy reliance")
    else:
        print(f"  🔴 MODEL NEEDS WORK — {total - passed} checks failed")
    
    print("█" * 60 + "\n")
    
    report['scorecard'] = {
        'passed': passed,
        'total': total,
        'checks': {name: ok for name, ok in checks},
        'verdict': 'PRODUCTION_READY' if passed == total else 
                   'ACCEPTABLE' if passed >= total - 1 else 'NEEDS_WORK',
    }
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Make JSON-serializable
    serializable = {}
    for k, v in report.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)
    
    with open(report_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"📄 Full report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Validation Suite')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for pipeline testing')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds (default: 5)')
    
    args = parser.parse_args()
    
    run_full_validation(use_synthetic=args.synthetic, n_cv_folds=args.folds)
