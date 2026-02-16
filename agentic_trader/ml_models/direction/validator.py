"""
DAILY DIRECTION VALIDATOR — Validate the next-day UP/DOWN model

5-check validation framework adapted for daily direction prediction:
  1. Data Sufficiency — enough labeled daily samples?
  2. Time-Series CV — does direction accuracy hold across time periods?
  3. Feature Stability — are top features consistent across folds?
  4. Calibration — are UP/DOWN probabilities well-calibrated?
  5. Overfit Stress Test — does model beat shuffled labels?

Baseline is ~50% (balanced UP/DOWN), so accuracy > 52% is signal.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from .label_creator import LABEL_MAP, LABEL_NAMES
from .feature_engineering import get_direction_feature_names


REPORTS_DIR = Path(__file__).parent / "reports"


def check_data_sufficiency(df: pd.DataFrame) -> dict:
    """Check if we have enough directional samples."""
    movers = df.dropna(subset=['label_idx'])
    n_total = len(movers)
    n_up = (movers['label_idx'] == 1).sum()
    n_down = (movers['label_idx'] == 0).sum()
    n_symbols = movers['symbol'].nunique() if 'symbol' in movers.columns else 1
    n_days = movers['date'].dt.date.nunique() if 'date' in movers.columns else 0
    
    # Thresholds (lower than Model 1 since we only have movers)
    min_total = 3000
    min_minority = 1000
    min_days = 30
    min_symbols = 10
    
    checks = {
        'total': n_total >= min_total,
        'minority': min(n_up, n_down) >= min_minority,
        'days': n_days >= min_days,
        'symbols': n_symbols >= min_symbols,
    }
    
    verdict = all(checks.values())
    
    print(f"\n{'='*60}")
    print(f"  DATA SUFFICIENCY CHECK (Direction Model)")
    print(f"{'='*60}")
    v = lambda x: 'PASS' if x else 'FAIL'
    print(f"  {v(checks['total'])} Total movers: {n_total:,} (need >={min_total:,})")
    print(f"     UP: {n_up:,} ({n_up/max(n_total,1)*100:.1f}%)")
    print(f"     DOWN: {n_down:,} ({n_down/max(n_total,1)*100:.1f}%)")
    print(f"  {v(checks['minority'])} Minority class: {min(n_up, n_down):,} (need >={min_minority:,})")
    print(f"  {v(checks['days'])} Trading days: {n_days} (need >={min_days})")
    print(f"  {v(checks['symbols'])} Symbols: {n_symbols} (need >={min_symbols})")
    vstr = 'PASS' if verdict else 'FAIL'
    print(f"\n  VERDICT: {vstr}")
    print(f"{'='*60}")
    
    return {
        'verdict': verdict,
        'total': n_total,
        'up': int(n_up),
        'down': int(n_down),
        'days': n_days,
        'symbols': n_symbols,
    }


def time_series_cv(df: pd.DataFrame, features: list, n_folds: int = 5) -> dict:
    """Walk-forward cross validation for direction model."""
    from .trainer import DEFAULT_PARAMS
    
    movers = df.dropna(subset=['label_idx']).copy()
    movers = movers.sort_values('date').reset_index(drop=True)
    
    all_dates = sorted(movers['date'].dt.date.unique())
    n_days = len(all_dates)
    fold_size = max(5, n_days // (n_folds + 2))
    gap = 1
    
    print(f"\n{'='*60}")
    print(f"  TIME-SERIES CV (Direction, {n_folds} folds)")
    print(f"  {n_days} days | fold_size={fold_size}d | gap={gap}d")
    print(f"{'='*60}")
    
    fold_results = []
    
    for fold in range(n_folds):
        test_end_idx = n_days - 1 - fold * fold_size
        test_start_idx = test_end_idx - fold_size + 1
        train_end_idx = test_start_idx - gap - 1
        
        if train_end_idx < 20 or test_start_idx < 0:
            break
        
        train_dates = set(all_dates[:train_end_idx + 1])
        test_dates = set(all_dates[test_start_idx:test_end_idx + 1])
        
        train = movers[movers['date'].dt.date.isin(train_dates)]
        test = movers[movers['date'].dt.date.isin(test_dates)]
        
        if len(train) < 500 or len(test) < 100:
            continue
        
        X_train = np.nan_to_num(train[features].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = train['label_idx'].values.astype(int)
        X_test = np.nan_to_num(test[features].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = test['label_idx'].values.astype(int)
        
        params = DEFAULT_PARAMS.copy()
        params.pop('n_estimators', None)
        params.pop('early_stopping_rounds', None)
        
        model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_f1 = f1_score(y_test, model.predict(X_test), average='macro')
        gap_val = train_acc - test_acc
        
        fold_results.append({
            'train_acc': train_acc, 'test_acc': test_acc,
            'gap': gap_val, 'f1': test_f1, 'test_size': len(test),
        })
        
        emoji = 'OK' if gap_val < 0.05 else 'WARN' if gap_val < 0.10 else 'BAD'
        dates_str = f"{all_dates[test_start_idx]} to {all_dates[min(test_end_idx, n_days-1)]}"
        print(f"  Fold {fold+1}: Train={train_acc:.3f} Test={test_acc:.3f} "
              f"Gap={gap_val:+.3f}[{emoji}] | F1={test_f1:.3f} | {dates_str} ({len(test)} samples)")
    
    if not fold_results:
        return {'verdict': False, 'reason': 'No valid folds'}
    
    avg_acc = np.mean([r['test_acc'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_gap = np.mean([r['gap'] for r in fold_results])
    max_gap = max(r['gap'] for r in fold_results)
    stability = max(r['test_acc'] for r in fold_results) - min(r['test_acc'] for r in fold_results)
    
    # Direction model: 50% is random, so accuracy > 52% is already signal
    verdict = avg_gap < 0.15 and avg_acc > 0.52
    
    vstr = 'PASS' if verdict else 'OVERFIT' if avg_gap >= 0.15 else 'NO SIGNAL'
    print(f"\n  AGGREGATE:")
    print(f"    Accuracy:  {avg_acc:.3f} +/- {np.std([r['test_acc'] for r in fold_results]):.3f}")
    print(f"    F1:        {avg_f1:.3f}")
    print(f"    Overfit:   {avg_gap:+.3f} (max {max_gap:+.3f})")
    print(f"    Stability: {stability:.3f}")
    print(f"    Verdict:   {vstr}")
    print(f"{'='*60}")
    
    return {
        'verdict': verdict,
        'avg_accuracy': round(avg_acc, 4),
        'avg_f1': round(avg_f1, 4),
        'avg_gap': round(avg_gap, 4),
        'max_gap': round(max_gap, 4),
        'folds': fold_results,
    }


def check_feature_stability(df: pd.DataFrame, features: list, n_folds: int = 5) -> dict:
    """Check if top features are consistent across time folds."""
    from .trainer import DEFAULT_PARAMS
    
    movers = df.dropna(subset=['label_idx']).copy()
    movers = movers.sort_values('date').reset_index(drop=True)
    all_dates = sorted(movers['date'].dt.date.unique())
    n_days = len(all_dates)
    fold_size = max(5, n_days // (n_folds + 2))
    
    top5_per_fold = []
    importance_per_fold = []
    
    for fold in range(n_folds):
        test_end = n_days - 1 - fold * fold_size
        test_start = test_end - fold_size + 1
        train_end = test_start - 2
        
        if train_end < 20:
            break
        
        train_dates = set(all_dates[:train_end + 1])
        train = movers[movers['date'].dt.date.isin(train_dates)]
        
        if len(train) < 300:
            continue
        
        X = np.nan_to_num(train[features].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = train['label_idx'].values.astype(int)
        
        params = DEFAULT_PARAMS.copy()
        params.pop('n_estimators', None)
        params.pop('early_stopping_rounds', None)
        
        model = xgb.XGBClassifier(n_estimators=300, **params)
        model.fit(X, y, verbose=0)
        
        imp = dict(zip(features, model.feature_importances_))
        importance_per_fold.append(imp)
        top5 = sorted(imp, key=imp.get, reverse=True)[:5]
        top5_per_fold.append(top5)
    
    if len(top5_per_fold) < 2:
        return {'verdict': False, 'reason': 'Too few folds'}
    
    # How many of fold 0's top-5 appear in ALL other folds' top-5?
    baseline = set(top5_per_fold[0])
    consistent = sum(1 for f in baseline if all(f in fold_top for fold_top in top5_per_fold))
    stability = consistent / 5
    
    all_top5 = set()
    for t in top5_per_fold:
        all_top5.update(t)
    
    # Mean importance per feature
    mean_imp = {}
    for f in features:
        vals = [fold.get(f, 0) for fold in importance_per_fold]
        mean_imp[f] = np.mean(vals)
    
    top10 = sorted(mean_imp, key=mean_imp.get, reverse=True)[:10]
    
    print(f"\n{'='*60}")
    print(f"  FEATURE STABILITY (Direction, {len(top5_per_fold)} folds)")
    print(f"{'='*60}")
    print(f"  Top-5 stability: {stability*100:.0f}% ({consistent}/5)")
    print(f"  Top-5 union: {len(all_top5)} unique features")
    
    print(f"\n  {'Feature':<25s} {'Mean Imp':>10s}  Ranks")
    print(f"  {'─'*55}")
    for f in top10:
        ranks = []
        for fold_imp in importance_per_fold:
            sorted_feats = sorted(fold_imp, key=fold_imp.get, reverse=True)
            rank = sorted_feats.index(f) + 1 if f in sorted_feats else '?'
            ranks.append(rank)
        
        print(f"  {f:<25s} {mean_imp[f]:>10.4f}  {ranks}")
    
    verdict = stability >= 0.20  # At least 1/5 top features consistent
    vstr = 'PASS' if verdict else 'UNSTABLE'
    print(f"\n  Verdict: {vstr}")
    print(f"{'='*60}")
    
    return {
        'verdict': verdict,
        'stability': stability,
        'top5_union_size': len(all_top5),
    }


def overfit_stress_test(df: pd.DataFrame, features: list) -> dict:
    """Compare real model vs shuffled-label models using accuracy (valid for balanced data)."""
    from .trainer import DEFAULT_PARAMS
    
    movers = df.dropna(subset=['label_idx']).copy()
    movers = movers.sort_values('date').reset_index(drop=True)
    
    # 70/30 split
    split_idx = int(len(movers) * 0.7)
    train = movers.iloc[:split_idx]
    test = movers.iloc[split_idx:]
    
    X_train = np.nan_to_num(train[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train['label_idx'].values.astype(int)
    X_test = np.nan_to_num(test[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test['label_idx'].values.astype(int)
    
    params = DEFAULT_PARAMS.copy()
    params.pop('n_estimators', None)
    params.pop('early_stopping_rounds', None)
    
    # Real model
    real_model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **params)
    real_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    
    real_pred = real_model.predict(X_test)
    real_acc = accuracy_score(y_test, real_pred)
    real_f1 = f1_score(y_test, real_pred, average='macro')
    real_train_f1 = f1_score(y_train, real_model.predict(X_train), average='macro')
    
    # Shuffled models
    shuffle_accs = []
    shuffle_f1s = []
    for _ in range(3):
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        
        shuf_model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **params)
        shuf_model.fit(X_train, y_shuffled, eval_set=[(X_test, y_test)], verbose=0)
        
        shuf_pred = shuf_model.predict(X_test)
        shuffle_accs.append(accuracy_score(y_test, shuf_pred))
        shuffle_f1s.append(f1_score(y_test, shuf_pred, average='macro'))
    
    avg_shuffle_acc = np.mean(shuffle_accs)
    avg_shuffle_f1 = np.mean(shuffle_f1s)
    signal_gap = real_acc - avg_shuffle_acc
    f1_gap = real_f1 - avg_shuffle_f1
    
    print(f"\n{'='*60}")
    print(f"  OVERFIT STRESS TEST (Direction)")
    print(f"  Metric: Accuracy + Macro F1 (balanced data)")
    print(f"{'='*60}")
    print(f"  Real model:     Acc={real_acc:.3f}  F1={real_f1:.3f}  Train_F1={real_train_f1:.3f}")
    for i, (sa, sf) in enumerate(zip(shuffle_accs, shuffle_f1s)):
        print(f"  Shuffle {i+1}:      Acc={sa:.3f}  F1={sf:.3f}")
    print(f"\n  Real accuracy:    {real_acc:.3f}")
    print(f"  Shuffled avg:     {avg_shuffle_acc:.3f}")
    print(f"  SIGNAL GAP:       {signal_gap:+.3f}")
    print(f"  F1 GAP:           {f1_gap:+.3f}")
    
    # For direction: accuracy gap > 0.03 is meaningful (50% baseline)
    verdict = signal_gap > 0.03 and real_acc > 0.52
    vstr = 'PASS' if verdict else 'FAIL'
    print(f"  Verdict: {vstr} (need gap > 0.03 and acc > 52%)")
    print(f"{'='*60}")
    
    return {
        'verdict': verdict,
        'real_acc': round(real_acc, 4),
        'real_f1': round(real_f1, 4),
        'shuffled_acc': round(avg_shuffle_acc, 4),
        'shuffled_f1': round(avg_shuffle_f1, 4),
        'signal_gap': round(signal_gap, 4),
        'f1_gap': round(f1_gap, 4),
    }


def check_calibration(df: pd.DataFrame, features: list) -> dict:
    """Check if UP/DOWN probabilities are calibrated."""
    from .trainer import DEFAULT_PARAMS
    
    movers = df.dropna(subset=['label_idx']).copy()
    movers = movers.sort_values('date').reset_index(drop=True)
    split_idx = int(len(movers) * 0.7)
    
    train = movers.iloc[:split_idx]
    test = movers.iloc[split_idx:]
    
    X_train = np.nan_to_num(train[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train['label_idx'].values.astype(int)
    X_test = np.nan_to_num(test[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test['label_idx'].values.astype(int)
    
    params = DEFAULT_PARAMS.copy()
    params.pop('n_estimators', None)
    params.pop('early_stopping_rounds', None)
    
    model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    
    proba = model.predict_proba(X_test)
    up_probs = proba[:, 1]
    
    # Calibration by probability bins
    bins = np.arange(0.05, 1.0, 0.10)
    ece = 0.0
    n_total = len(y_test)
    
    print(f"\n{'='*60}")
    print(f"  CALIBRATION CHECK (Direction)")
    print(f"{'='*60}")
    print(f"  UP probability calibration:")
    print(f"     Predicted    Actual     Count    Error")
    
    for b in bins:
        mask = (up_probs >= b) & (up_probs < b + 0.10)
        count = mask.sum()
        if count < 5:
            continue
        actual = y_test[mask].mean()
        predicted = up_probs[mask].mean()
        error = abs(predicted - actual)
        ece += error * count / n_total
        
        emoji = 'OK' if error < 0.05 else 'WARN' if error < 0.10 else 'BAD'
        print(f"     {predicted:>6.0%}      {actual:>5.1%}     {count:>5d}    {error:.1%} [{emoji}]")
    
    verdict = ece < 0.10
    vstr = 'PASS' if verdict else 'FAIL'
    print(f"\n  Overall ECE: {ece:.3f}")
    print(f"  Verdict: {vstr}")
    print(f"{'='*60}")
    
    return {
        'verdict': verdict,
        'ece': round(ece, 4),
    }


def run_full_validation(n_cv_folds: int = 5) -> dict:
    """Run all 5 validation checks on the direction model."""
    from .trainer import load_and_prepare_data
    from .feature_engineering import get_direction_feature_names
    
    print(f"\n{'*'*60}")
    print(f"  DIRECTION MODEL VALIDATION")  
    print(f"  (Model 2: UP vs DOWN)")
    print(f"{'*'*60}")
    
    train_df, test_df, features = load_and_prepare_data(test_days=60)
    data = pd.concat([train_df, test_df], ignore_index=True)
    
    report = {'timestamp': datetime.now().isoformat()}
    
    # 1. Data Sufficiency
    print(f"\n{'#'*60}")
    print(f"  STEP 1/5: DATA SUFFICIENCY")
    print(f"{'#'*60}")
    report['data_sufficiency'] = check_data_sufficiency(data)
    
    # 2. Time-Series CV
    print(f"\n{'#'*60}")
    print(f"  STEP 2/5: TIME-SERIES CROSS-VALIDATION")
    print(f"{'#'*60}")
    report['time_series_cv'] = time_series_cv(data, features, n_cv_folds)
    
    # 3. Feature Stability
    print(f"\n{'#'*60}")
    print(f"  STEP 3/5: FEATURE STABILITY")
    print(f"{'#'*60}")
    report['feature_stability'] = check_feature_stability(data, features, n_cv_folds)
    
    # 4. Calibration
    print(f"\n{'#'*60}")
    print(f"  STEP 4/5: PROBABILITY CALIBRATION")
    print(f"{'#'*60}")
    report['calibration'] = check_calibration(data, features)
    
    # 5. Overfit Stress Test
    print(f"\n{'#'*60}")
    print(f"  STEP 5/5: OVERFIT STRESS TEST")
    print(f"{'#'*60}")
    report['stress_test'] = overfit_stress_test(data, features)
    
    # Scorecard
    checks = {
        'Data Sufficiency': report['data_sufficiency']['verdict'],
        'Overfitting Control': report['time_series_cv']['verdict'],
        'Feature Stability': report['feature_stability']['verdict'],
        'Calibration': report['calibration']['verdict'],
        'Real Signal': report['stress_test']['verdict'],
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"\n{'='*60}")
    print(f"  FINAL DIRECTION MODEL SCORECARD")
    print(f"{'='*60}")
    for name, result in checks.items():
        mark = 'PASS' if result else 'FAIL'
        print(f"  [{mark}] {name}")
    
    print(f"\n  SCORE: {passed}/{total}")
    if passed == total:
        print(f"  MODEL IS PRODUCTION-READY")
    elif passed >= 4:
        print(f"  MODEL IS ACCEPTABLE")
    elif passed >= 3:
        print(f"  MODEL NEEDS IMPROVEMENT")
    else:
        print(f"  MODEL IS NOT READY — do not deploy")
    print(f"{'='*60}")
    
    report['scorecard'] = checks
    report['score'] = f"{passed}/{total}"
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORTS_DIR / f"direction_validation_{ts}.json"
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=convert)
    
    print(f"\n  Report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    run_full_validation()
