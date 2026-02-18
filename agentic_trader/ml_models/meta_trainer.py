"""
META-LABELING TRAINER — Two-model architecture for better directional accuracy

Architecture:
  Model 1 (GATE):  Binary MOVE vs FLAT — "Will this stock move?"
  Model 2 (DIRECTION): Binary UP vs DOWN — trained ONLY on samples that actually moved

Why this works:
  - The Gate model handles the easier question (volatility prediction) with strong features
  - The Direction model focuses purely on directional signal, trained WITHOUT flat-noise dilution
  - Combined: P(UP) = P(MOVE) × P(UP|MOVE), P(DOWN) = P(MOVE) × P(DOWN|MOVE)

Usage:
  python -m ml_models.meta_trainer --atr-factor 1.5
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
from sklearn.isotonic import IsotonicRegression

from .trainer import load_and_prepare_data, LABEL_MAP, LABEL_NAMES, MODELS_DIR, REPORTS_DIR

# Gate model: FLAT=0, MOVE=1
GATE_LABELS = {0: 'FLAT', 1: 'MOVE'}
# Direction model: DOWN=0, UP=1
DIR_LABELS = {0: 'DOWN', 1: 'UP'}


# Gate model params — binary, easier problem
GATE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.03,           # Lower LR — needs more trees for 80/20 split
    'n_estimators': 3000,            # More epochs for convergence
    'min_child_weight': 15,
    'subsample': 0.75,
    'colsample_bytree': 0.8,
    'gamma': 1.5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': 42,
    'early_stopping_rounds': 120,    # More patience with lower LR
    'verbosity': 1,
    'scale_pos_weight': 1.0,  # will be computed dynamically
}

# Direction model params — binary, harder problem, needs different tuning
# v2: Deeper tree + lighter regularization to let the model learn sector/direction
# patterns. Previous version was over-regularized (gamma=2, lambda=3.5) which
# compressed predictions to ~0.49-0.51 range, making DOWN nearly undetectable.
# Key changes: depth 7 (was 6), lower gamma/lambda, higher colsample to let
# sector features participate more in each tree.
DIR_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 7,                  # Deeper — more capacity for sector interactions
    'learning_rate': 0.012,          # Slightly slower for better convergence
    'n_estimators': 4000,            # More trees with lower LR
    'min_child_weight': 18,          # Slightly less conservative (was 22)
    'subsample': 0.72,               # Moderate row sampling
    'colsample_bytree': 0.75,        # Higher — let sector features participate (was 0.65)
    'gamma': 1.0,                    # Less aggressive pruning (was 2.0)
    'reg_alpha': 0.5,                # Less L1 (was 1.0)
    'reg_lambda': 2.0,               # Less L2 (was 3.5) — let model spread predictions
    'random_state': 42,
    'early_stopping_rounds': 150,    # More patience with lower LR
    'verbosity': 1,
    'scale_pos_weight': 1.0,  # will be computed dynamically
}


def train_meta_models(
    atr_factor: float = 2.0,
    test_days: int = 20,
    val_days: int = 10,
    symbols: Optional[list] = None,
    label_method: str = 'net_return',
    hybrid: bool = False,
    gate_atr_factor: float = 1.5,
    gate_label_method: str = 'first_to_break',
    dir_atr_factor: float = 1.0,
    dir_label_method: str = 'net_return',
) -> dict:
    """Train both Gate and Direction models using meta-labeling.
    
    When hybrid=True, uses different labeling for gate vs direction:
      - Gate:      gate_label_method + gate_atr_factor (default: first_to_break×1.5)
      - Direction: dir_label_method + dir_atr_factor  (default: net_return×1.0)
    This combines the gate's better MOVE/FLAT separation with the direction
    model's more balanced UP/DOWN detection.
    
    Returns dict with both models, metrics, and paths.
    """
    
    # ── Step 1: Load data with 3-class labels ──
    if hybrid:
        print(f"\n🔀 HYBRID LABELING MODE")
        print(f"   Gate:      {gate_label_method} × ATR {gate_atr_factor}")
        print(f"   Direction: {dir_label_method} × ATR {dir_atr_factor}")
        
        # Load gate labels (first_to_break, ATR×1.5)
        print(f"\n── Loading data for GATE model ({gate_label_method}, ATR×{gate_atr_factor}) ──")
        train_df, val_df, test_df, feature_names = load_and_prepare_data(
            symbols=symbols,
            atr_factor=gate_atr_factor,
            test_days=test_days,
            val_days=val_days,
            label_method=gate_label_method,
        )
        
        # Load direction labels (net_return, ATR×1.0)
        print(f"\n── Loading data for DIRECTION model ({dir_label_method}, ATR×{dir_atr_factor}) ──")
        train_df_dir, val_df_dir, test_df_dir, feature_names_check = load_and_prepare_data(
            symbols=symbols,
            atr_factor=dir_atr_factor,
            test_days=test_days,
            val_days=val_days,
            label_method=dir_label_method,
        )
        assert feature_names == feature_names_check, "Feature mismatch between gate and direction data loads!"
        
        # Effective label_method/atr_factor for metadata
        label_method = f"hybrid({gate_label_method}+{dir_label_method})"
        atr_factor_display = f"gate={gate_atr_factor}/dir={dir_atr_factor}"
    else:
        train_df, val_df, test_df, feature_names = load_and_prepare_data(
            symbols=symbols,
            atr_factor=atr_factor,
            test_days=test_days,
            val_days=val_days,
            label_method=label_method,
        )
        train_df_dir = val_df_dir = test_df_dir = None  # not used
        atr_factor_display = str(atr_factor)
    
    # ══════════════════════════════════════════════════════════════
    #  GATE MODEL: MOVE vs FLAT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  GATE MODEL: MOVE (UP+DOWN) vs FLAT")
    print(f"{'='*70}")
    
    # Convert 3-class labels to binary: MOVE=1, FLAT=0
    # Original: DOWN=0, FLAT=1, UP=2
    train_gate_y = (train_df['label_idx'] != 1).astype(int).values  # NOT FLAT = MOVE
    val_gate_y = (val_df['label_idx'] != 1).astype(int).values
    test_gate_y = (test_df['label_idx'] != 1).astype(int).values
    
    X_train = train_df[feature_names].values
    X_val = val_df[feature_names].values
    X_test = test_df[feature_names].values
    
    n_flat = (train_gate_y == 0).sum()
    n_move = (train_gate_y == 1).sum()
    print(f"  Train: FLAT={n_flat:,} ({n_flat/len(train_gate_y)*100:.1f}%), MOVE={n_move:,} ({n_move/len(train_gate_y)*100:.1f}%)")
    
    gate_params = GATE_PARAMS.copy()
    gate_params['scale_pos_weight'] = n_flat / max(n_move, 1)  # Balance classes
    
    n_est_gate = gate_params.pop('n_estimators')
    es_gate = gate_params.pop('early_stopping_rounds')
    
    gate_model = xgb.XGBClassifier(
        n_estimators=n_est_gate,
        early_stopping_rounds=es_gate,
        **gate_params,
    )
    
    t0 = time.time()
    gate_model.fit(
        X_train, train_gate_y,
        eval_set=[(X_val, val_gate_y)],
        verbose=100,
    )
    gate_time = time.time() - t0
    
    # Gate calibrator
    gate_val_proba = gate_model.predict_proba(X_val)[:, 1]  # P(MOVE)
    gate_cal = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    gate_cal.fit(gate_val_proba, val_gate_y)
    
    # Gate test evaluation
    gate_test_raw = gate_model.predict_proba(X_test)[:, 1]
    gate_test_cal = gate_cal.predict(gate_test_raw)
    gate_pred = (gate_test_cal >= 0.5).astype(int)
    
    gate_acc = accuracy_score(test_gate_y, gate_pred)
    gate_prec = precision_score(test_gate_y, gate_pred, pos_label=1, zero_division=0)
    gate_rec = recall_score(test_gate_y, gate_pred, pos_label=1, zero_division=0)
    gate_f1 = f1_score(test_gate_y, gate_pred, pos_label=1, zero_division=0)
    
    print(f"\n  Gate Model Results:")
    print(f"    Best iteration: {gate_model.best_iteration}")
    print(f"    Training time: {gate_time:.1f}s")
    print(f"    Accuracy: {gate_acc:.1%}")
    print(f"    MOVE precision: {gate_prec:.1%} | recall: {gate_rec:.1%} | F1: {gate_f1:.3f}")
    
    # Gate high-confidence
    print(f"\n  Gate High-Confidence:")
    for thr in [0.40, 0.50, 0.60, 0.70, 0.80]:
        mask = gate_test_cal >= thr
        count = mask.sum()
        if count > 0:
            correct = test_gate_y[mask].sum()
            prec = correct / count
            total_move = test_gate_y.sum()
            print(f"    MOVE≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_move} ({correct/max(total_move,1)*100:.1f}%)")
    
    # Gate feature importance top 10
    gate_imp = sorted(zip(feature_names, gate_model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Gate Top 10 Features:")
    for fname, imp in gate_imp[:10]:
        print(f"    {fname:<25s}  {imp:.4f}")
    
    # ══════════════════════════════════════════════════════════════
    #  DIRECTION MODEL: UP vs DOWN (trained only on MOVE samples)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DIRECTION MODEL: UP vs DOWN (only MOVE samples)")
    print(f"{'='*70}")
    
    # Filter to only samples that actually moved (UP or DOWN)
    # In hybrid mode, use direction-specific labels for the direction model
    # Original 3-class: DOWN=0, FLAT=1, UP=2
    if hybrid and train_df_dir is not None:
        # Use net_return labels for direction training (more balanced UP/DOWN)
        dir_src_train = train_df_dir
        dir_src_val = val_df_dir
        dir_src_test = test_df_dir
        print(f"  (Using {dir_label_method} labels for direction training)")
    else:
        dir_src_train = train_df
        dir_src_val = val_df
        dir_src_test = test_df
    
    train_move_mask = dir_src_train['label_idx'] != 1  # Not FLAT
    val_move_mask = dir_src_val['label_idx'] != 1
    test_move_mask = dir_src_test['label_idx'] != 1
    
    X_train_dir = dir_src_train.loc[train_move_mask, feature_names].values
    X_val_dir = dir_src_val.loc[val_move_mask, feature_names].values
    X_test_dir = dir_src_test.loc[test_move_mask, feature_names].values
    
    # Direction labels: DOWN(0)→0, UP(2)→1
    train_dir_y = (dir_src_train.loc[train_move_mask, 'label_idx'] == 2).astype(int).values
    val_dir_y = (dir_src_val.loc[val_move_mask, 'label_idx'] == 2).astype(int).values
    test_dir_y = (dir_src_test.loc[test_move_mask, 'label_idx'] == 2).astype(int).values
    
    n_down = (train_dir_y == 0).sum()
    n_up = (train_dir_y == 1).sum()
    print(f"  Train (MOVE only): DOWN={n_down:,} ({n_down/len(train_dir_y)*100:.1f}%), UP={n_up:,} ({n_up/len(train_dir_y)*100:.1f}%)")
    print(f"  Total MOVE training samples: {len(train_dir_y):,} (excluded {(~train_move_mask).sum():,} FLAT)")
    
    dir_params = DIR_PARAMS.copy()
    dir_params['scale_pos_weight'] = n_down / max(n_up, 1)  # Balance UP vs DOWN
    
    n_est_dir = dir_params.pop('n_estimators')
    es_dir = dir_params.pop('early_stopping_rounds')
    
    dir_model = xgb.XGBClassifier(
        n_estimators=n_est_dir,
        early_stopping_rounds=es_dir,
        **dir_params,
    )
    
    t0 = time.time()
    dir_model.fit(
        X_train_dir, train_dir_y,
        eval_set=[(X_val_dir, val_dir_y)],
        verbose=100,
    )
    dir_time = time.time() - t0
    
    # ── Feature pruning: drop features with < 0.3% importance, retrain ──
    importance_threshold = 0.003
    importances = dir_model.feature_importances_
    keep_mask = importances >= importance_threshold
    n_pruned = (~keep_mask).sum()
    
    if n_pruned > 0:
        pruned_names = [f for f, keep in zip(feature_names, keep_mask) if not keep]
        kept_names = [f for f, keep in zip(feature_names, keep_mask) if keep]
        print(f"\n  Feature Pruning: dropping {n_pruned} features with <{importance_threshold:.1%} importance:")
        for pn in pruned_names:
            print(f"    - {pn}")
        print(f"  Retraining direction model with {len(kept_names)} features...")
        
        # Keep indices for subsetting
        keep_indices = np.where(keep_mask)[0]
        X_train_dir_pruned = X_train_dir[:, keep_indices]
        X_val_dir_pruned = X_val_dir[:, keep_indices]
        X_test_dir_pruned = X_test_dir[:, keep_indices]
        
        # Retrain with pruned feature set
        dir_params_2 = dir_params.copy()
        dir_model = xgb.XGBClassifier(
            n_estimators=n_est_dir,
            early_stopping_rounds=es_dir,
            **dir_params_2,
        )
        t0 = time.time()
        dir_model.fit(
            X_train_dir_pruned, train_dir_y,
            eval_set=[(X_val_dir_pruned, val_dir_y)],
            verbose=100,
        )
        dir_time += time.time() - t0
        
        # Update references for downstream code
        X_train_dir = X_train_dir_pruned
        X_val_dir = X_val_dir_pruned
        X_test_dir = X_test_dir_pruned
        feature_names_dir = kept_names
        
        # Also prune for all-sample direction prediction
        X_test_dir_all = test_df[feature_names].values[:, keep_indices]
    else:
        feature_names_dir = list(feature_names)
        X_test_dir_all = test_df[feature_names].values
        print(f"\n  Feature Pruning: all features above {importance_threshold:.1%} threshold, no pruning needed.")
    
    # Direction calibrator
    dir_val_proba = dir_model.predict_proba(X_val_dir)[:, 1]  # P(UP|MOVE)
    dir_cal = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    dir_cal.fit(dir_val_proba, val_dir_y)
    
    # Direction test evaluation (on MOVE samples only)
    dir_test_raw = dir_model.predict_proba(X_test_dir)[:, 1]
    dir_test_cal = dir_cal.predict(dir_test_raw)
    dir_pred = (dir_test_cal >= 0.5).astype(int)
    
    dir_acc = accuracy_score(test_dir_y, dir_pred)
    dir_prec_up = precision_score(test_dir_y, dir_pred, pos_label=1, zero_division=0)
    dir_rec_up = recall_score(test_dir_y, dir_pred, pos_label=1, zero_division=0)
    dir_f1_up = f1_score(test_dir_y, dir_pred, pos_label=1, zero_division=0)
    dir_prec_down = precision_score(test_dir_y, dir_pred, pos_label=0, zero_division=0)
    dir_rec_down = recall_score(test_dir_y, dir_pred, pos_label=0, zero_division=0)
    dir_f1_down = f1_score(test_dir_y, dir_pred, pos_label=0, zero_division=0)
    
    print(f"\n  Direction Model Results (MOVE samples only):")
    print(f"    Best iteration: {dir_model.best_iteration}")
    print(f"    Training time: {dir_time:.1f}s")
    print(f"    Accuracy: {dir_acc:.1%} (chance=50%, must beat this)")
    print(f"    UP  precision: {dir_prec_up:.1%} | recall: {dir_rec_up:.1%} | F1: {dir_f1_up:.3f}")
    print(f"    DOWN precision: {dir_prec_down:.1%} | recall: {dir_rec_down:.1%} | F1: {dir_f1_down:.3f}")
    
    # Direction high-confidence
    print(f"\n  Direction High-Confidence (on MOVE samples):")
    print(f"  === UP signals ===")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = dir_test_cal >= thr
        count = mask.sum()
        if count > 0:
            correct = test_dir_y[mask].sum()
            prec = correct / count
            total_up = test_dir_y.sum()
            print(f"    P(UP|MOVE)≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_up} ({correct/max(total_up,1)*100:.1f}%)")
    
    print(f"  === DOWN signals ===")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = dir_test_cal <= (1 - thr)  # P(DOWN|MOVE) >= thr means P(UP|MOVE) <= (1-thr)
        count = mask.sum()
        if count > 0:
            correct = (test_dir_y[mask] == 0).sum()
            prec = correct / count
            total_down = (test_dir_y == 0).sum()
            print(f"    P(DOWN|MOVE)≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_down} ({correct/max(total_down,1)*100:.1f}%)")
    
    # Direction feature importance
    dir_imp = sorted(zip(feature_names_dir, dir_model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Direction Top {min(15, len(feature_names_dir))} Features ({len(feature_names_dir)} total after pruning):")
    for fname, imp in dir_imp[:15]:
        print(f"    {fname:<25s}  {imp:.4f}")
    
    # ══════════════════════════════════════════════════════════════
    #  COMBINED SYSTEM EVALUATION (on ALL test samples)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  COMBINED META-LABELING SYSTEM EVALUATION")
    if hybrid:
        print(f"  (Ground truth: gate labels = {gate_label_method} × ATR {gate_atr_factor})")
    print(f"{'='*70}")
    
    # For all test samples, get P(MOVE) from gate (uses full feature set)
    X_test_gate = test_df[feature_names].values
    all_gate_cal = gate_cal.predict(gate_model.predict_proba(X_test_gate)[:, 1])
    
    # For all test samples, get P(UP|MOVE) from direction model (uses pruned features)
    all_dir_raw = dir_model.predict_proba(X_test_dir_all)[:, 1]
    all_dir_cal = dir_cal.predict(all_dir_raw)
    
    # Combined probabilities:
    # P(UP) = P(MOVE) × P(UP|MOVE)
    # P(DOWN) = P(MOVE) × P(DOWN|MOVE) = P(MOVE) × (1 - P(UP|MOVE))
    # P(FLAT) = 1 - P(MOVE)
    p_move = all_gate_cal
    p_up_given_move = all_dir_cal
    
    p_up = p_move * p_up_given_move
    p_down = p_move * (1 - p_up_given_move)
    p_flat = 1 - p_move
    
    # Original 3-class test labels
    y_test_3class = test_df['label_idx'].values.astype(int)
    
    # Combined prediction: argmax of [p_down, p_flat, p_up]
    combined_probs = np.stack([p_down, p_flat, p_up], axis=1)  # (N, 3)
    combined_pred = np.argmax(combined_probs, axis=1)  # 0=DOWN, 1=FLAT, 2=UP
    
    combined_acc = accuracy_score(y_test_3class, combined_pred)
    combined_f1 = f1_score(y_test_3class, combined_pred, average=None, zero_division=0, labels=[0, 1, 2])
    combined_prec = precision_score(y_test_3class, combined_pred, average=None, zero_division=0, labels=[0, 1, 2])
    combined_rec = recall_score(y_test_3class, combined_pred, average=None, zero_division=0, labels=[0, 1, 2])
    macro_f1 = np.mean(combined_f1)
    
    cm = confusion_matrix(y_test_3class, combined_pred, labels=[0, 1, 2])
    
    print(f"  Overall accuracy: {combined_acc:.1%}")
    print(f"  Macro F1: {macro_f1:.3f}")
    print(f"\n  Per-class metrics:")
    for cls in range(3):
        print(f"    {LABEL_NAMES[cls]:>10s}  P={combined_prec[cls]:.3f}  R={combined_rec[cls]:.3f}  F1={combined_f1[cls]:.3f}")
    
    print(f"\n  Confusion Matrix:")
    header = ''.join(f"  Pred_{LABEL_NAMES[i]:>5s}" for i in range(3))
    print(f"  {'':>12s}{header}")
    for i, row in enumerate(cm):
        vals = ''.join(f"  {row[j]:>10d}" for j in range(3))
        print(f"  {LABEL_NAMES[i]:>12s}{vals}")
    
    # High-confidence signal analysis (combined probabilities)
    print(f"\n📈 COMBINED HIGH-CONFIDENCE SIGNAL ANALYSIS:")
    print(f"  === UP signals (P(MOVE)×P(UP|MOVE)) ===")
    for thr in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        mask = p_up >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test_3class[mask] == 2).sum()
            prec = correct / count
            total_up = (y_test_3class == 2).sum()
            print(f"    UP prob≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_up} ({correct/max(total_up,1)*100:.1f}%)")
    
    print(f"  === DOWN signals (P(MOVE)×P(DOWN|MOVE)) ===")
    for thr in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        mask = p_down >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test_3class[mask] == 0).sum()
            prec = correct / count
            total_down = (y_test_3class == 0).sum()
            print(f"    DOWN prob≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_down} ({correct/max(total_down,1)*100:.1f}%)")
    
    print(f"  === FLAT signals (1 - P(MOVE)) ===")
    for thr in [0.50, 0.60, 0.70, 0.80]:
        mask = p_flat >= thr
        count = mask.sum()
        if count > 0:
            correct = (y_test_3class[mask] == 1).sum()
            prec = correct / count
            total_flat = (y_test_3class == 1).sum()
            print(f"    FLAT prob≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}, recall={correct}/{total_flat} ({correct/max(total_flat,1)*100:.1f}%)")
    
    # ── Compare with single 3-class model baseline ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Meta-labeling vs previous 3-class single model")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Meta-labeling':>15s}")
    print(f"  {'Accuracy':<25s} {combined_acc:>14.1%}")
    print(f"  {'Macro F1':<25s} {macro_f1:>14.3f}")
    print(f"  {'DOWN F1':<25s} {combined_f1[0]:>14.3f}")
    print(f"  {'FLAT F1':<25s} {combined_f1[1]:>14.3f}")
    print(f"  {'UP F1':<25s} {combined_f1[2]:>14.3f}")
    print(f"  {'Gate accuracy':<25s} {gate_acc:>14.1%}")
    print(f"  {'Direction accuracy':<25s} {dir_acc:>14.1%}")
    
    # ── Save models ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save gate model
    gate_path = MODELS_DIR / f"meta_gate_{timestamp}.json"
    gate_latest = MODELS_DIR / "meta_gate_latest.json"
    gate_model.save_model(str(gate_path))
    gate_model.save_model(str(gate_latest))
    
    # Save gate calibrator
    gate_cal_path = MODELS_DIR / f"meta_gate_{timestamp}_calibrator.pkl"
    gate_cal_latest = MODELS_DIR / "meta_gate_latest_calibrator.pkl"
    joblib.dump(gate_cal, str(gate_cal_path))
    joblib.dump(gate_cal, str(gate_cal_latest))
    
    # Save direction model
    dir_path = MODELS_DIR / f"meta_direction_{timestamp}.json"
    dir_latest = MODELS_DIR / "meta_direction_latest.json"
    dir_model.save_model(str(dir_path))
    dir_model.save_model(str(dir_latest))
    
    # Save direction calibrator
    dir_cal_path = MODELS_DIR / f"meta_direction_{timestamp}_calibrator.pkl"
    dir_cal_latest = MODELS_DIR / "meta_direction_latest_calibrator.pkl"
    joblib.dump(dir_cal, str(dir_cal_path))
    joblib.dump(dir_cal, str(dir_cal_latest))
    
    # Save combined metadata
    meta = {
        'model_type': 'meta_labeling',
        'timestamp': timestamp,
        'atr_factor': atr_factor_display if hybrid else atr_factor,
        'label_method': label_method,
        'hybrid': hybrid,
        'gate_config': {'label_method': gate_label_method, 'atr_factor': gate_atr_factor} if hybrid else None,
        'dir_config': {'label_method': dir_label_method, 'atr_factor': dir_atr_factor} if hybrid else None,
        'feature_names': feature_names,
        'direction_feature_names': feature_names_dir,
        'features_pruned': int(n_pruned),
        'gate_model': {
            'accuracy': round(gate_acc, 4),
            'move_precision': round(gate_prec, 4),
            'move_recall': round(gate_rec, 4),
            'move_f1': round(gate_f1, 4),
            'best_iteration': int(gate_model.best_iteration),
            'train_time': round(gate_time, 1),
        },
        'direction_model': {
            'accuracy': round(dir_acc, 4),
            'up_precision': round(float(dir_prec_up), 4),
            'up_recall': round(float(dir_rec_up), 4),
            'up_f1': round(float(dir_f1_up), 4),
            'down_precision': round(float(dir_prec_down), 4),
            'down_recall': round(float(dir_rec_down), 4),
            'down_f1': round(float(dir_f1_down), 4),
            'best_iteration': int(dir_model.best_iteration),
            'train_time': round(dir_time, 1),
        },
        'combined': {
            'accuracy': round(combined_acc, 4),
            'macro_f1': round(float(macro_f1), 4),
            'down_f1': round(float(combined_f1[0]), 4),
            'flat_f1': round(float(combined_f1[1]), 4),
            'up_f1': round(float(combined_f1[2]), 4),
        },
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'gate_feature_importance': {f: round(float(i), 6) for f, i in gate_imp},
        'dir_feature_importance': {f: round(float(i), 6) for f, i in dir_imp},
    }
    
    meta_path = MODELS_DIR / f"meta_labeling_{timestamp}_meta.json"
    meta_latest = MODELS_DIR / "meta_labeling_latest_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    with open(meta_latest, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Gate model saved: {gate_path}")
    print(f"✅ Direction model saved: {dir_path}")
    print(f"✅ Metadata saved: {meta_path}")
    print(f"✅ Latest links updated")
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"   Gate (MOVE vs FLAT): {gate_acc:.1%} accuracy, MOVE F1={gate_f1:.3f}")
    print(f"   Direction (UP vs DOWN): {dir_acc:.1%} accuracy")
    print(f"   Combined: {combined_acc:.1%} accuracy, Macro F1={macro_f1:.3f}")
    print(f"   DOWN F1: {combined_f1[0]:.3f} | FLAT F1: {combined_f1[1]:.3f} | UP F1: {combined_f1[2]:.3f}")
    
    return {
        'gate_model': gate_model,
        'dir_model': dir_model,
        'gate_cal': gate_cal,
        'dir_cal': dir_cal,
        'meta': meta,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Meta-Labeling models (Gate + Direction)')
    parser.add_argument('--atr-factor', type=float, default=2.0, help='ATR multiplier for threshold (default: 2.0)')
    parser.add_argument('--test-days', type=int, default=20, help='Test period days (default: 20)')
    parser.add_argument('--val-days', type=int, default=10, help='Validation period days (default: 10)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--label-method', type=str, default='net_return', choices=['net_return', 'first_to_break'],
                        help='Labeling method when not using hybrid (default: net_return)')
    parser.add_argument('--hybrid', action='store_true',
                        help='Use hybrid labeling: first_to_break for gate, net_return for direction')
    parser.add_argument('--gate-atr', type=float, default=1.5, help='Gate ATR factor in hybrid mode (default: 1.5)')
    parser.add_argument('--gate-label', type=str, default='first_to_break', choices=['net_return', 'first_to_break'],
                        help='Gate label method in hybrid mode (default: first_to_break)')
    parser.add_argument('--dir-atr', type=float, default=1.0, help='Direction ATR factor in hybrid mode (default: 1.0)')
    parser.add_argument('--dir-label', type=str, default='net_return', choices=['net_return', 'first_to_break'],
                        help='Direction label method in hybrid mode (default: net_return)')
    
    args = parser.parse_args()
    
    result = train_meta_models(
        atr_factor=args.atr_factor,
        test_days=args.test_days,
        val_days=args.val_days,
        symbols=args.symbols,
        label_method=args.label_method,
        hybrid=args.hybrid,
        gate_atr_factor=args.gate_atr,
        gate_label_method=args.gate_label,
        dir_atr_factor=args.dir_atr,
        dir_label_method=args.dir_label,
    )
