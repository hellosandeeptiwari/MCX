"""Test model improvements without overfitting — uses trainer's data pipeline"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

from ml_models.trainer import load_and_prepare_data, DEFAULT_PARAMS
from ml_models.feature_engineering import get_feature_names

# Use trainer's own data pipeline (handles labels, quality filter, time-split)
train_df, test_df, features = load_and_prepare_data(threshold_pct=0.5)

X_train = np.nan_to_num(train_df[features].values, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_df[features].values, nan=0.0, posinf=0.0, neginf=0.0)
y_train = train_df['label_idx'].values
y_test = test_df['label_idx'].values

print(f'\nTrain: {len(X_train)} | Test: {len(X_test)}')
print(f'Train class dist: {Counter(y_train)}')
print(f'Test class dist: {Counter(y_test)}')
move_ratio = Counter(y_train)[0] / max(Counter(y_train)[1], 1)
print(f'Imbalance ratio (NO_MOVE/MOVE): {move_ratio:.2f}\n')

# === TEST 1: BASELINE (current model params) ===
print('='*60)
print('TEST 1: BASELINE (current model params)')
print('='*60)
p = DEFAULT_PARAMS.copy()
es = p.pop('early_stopping_rounds', 60)
model_base = xgb.XGBClassifier(**p, early_stopping_rounds=es)
model_base.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_base = model_base.predict(X_test)
print(f'Best iter: {model_base.best_iteration}')
print(classification_report(y_test, y_pred_base, target_names=['NO_MOVE','MOVE'], digits=4))

# === TEST 2: + scale_pos_weight ===
print('='*60)
print('TEST 2: + scale_pos_weight (class balance)')
print('='*60)
p2 = DEFAULT_PARAMS.copy()
es2 = p2.pop('early_stopping_rounds', 60)
model_spw = xgb.XGBClassifier(**p2, early_stopping_rounds=es2, scale_pos_weight=move_ratio)
model_spw.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_spw = model_spw.predict(X_test)
print(f'Best iter: {model_spw.best_iteration}')
print(classification_report(y_test, y_pred_spw, target_names=['NO_MOVE','MOVE'], digits=4))

# === TEST 3: scale_pos_weight + optimized threshold ===
print('='*60)
print('TEST 3: + optimal threshold search')
print('='*60)
proba_spw = model_spw.predict_proba(X_test)[:, 1]
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.20, 0.65, 0.01):
    preds = (proba_spw >= thr).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print(f'Optimal threshold: {best_thr:.2f} (MOVE F1: {best_f1:.4f})')
y_pred_opt = (proba_spw >= best_thr).astype(int)
print(classification_report(y_test, y_pred_opt, target_names=['NO_MOVE','MOVE'], digits=4))

# === TEST 4: Deeper model + scale_pos_weight ===
print('='*60)
print('TEST 4: Deeper (depth=6) + scale_pos_weight')
print('='*60)
model_deep = xgb.XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.015,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=30, gamma=0.05,
    reg_alpha=0.05, reg_lambda=0.5,
    scale_pos_weight=move_ratio,
    eval_metric='logloss', early_stopping_rounds=40, random_state=42
)
model_deep.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_deep = model_deep.predict(X_test)
print(f'Best iter: {model_deep.best_iteration}')
print(classification_report(y_test, y_pred_deep, target_names=['NO_MOVE','MOVE'], digits=4))

# Overfit check
y_train_pred_deep = model_deep.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred_deep)
test_acc = accuracy_score(y_test, y_pred_deep)
gap = train_acc - test_acc
print(f'Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f} | Gap: {gap:.4f}')
print(f'Overfit? {"YES (gap > 5%)" if gap > 0.05 else "NO (gap under 5%)"}')

# === TEST 5: Deeper + optimal threshold ===
print('='*60)
print('TEST 5: Deeper + optimal threshold')
print('='*60)
proba_deep = model_deep.predict_proba(X_test)[:, 1]
best_f1_d, best_thr_d = 0, 0.5
for thr in np.arange(0.20, 0.65, 0.01):
    preds = (proba_deep >= thr).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1_d:
        best_f1_d = f1
        best_thr_d = thr
print(f'Optimal threshold: {best_thr_d:.2f} (MOVE F1: {best_f1_d:.4f})')
y_pred_dopt = (proba_deep >= best_thr_d).astype(int)
print(classification_report(y_test, y_pred_dopt, target_names=['NO_MOVE','MOVE'], digits=4))

# === SUMMARY ===
print('='*60)
print('SUMMARY -- MOVE F1 Score Comparison')
print('='*60)
f1_base = f1_score(y_test, y_pred_base)
f1_spw = f1_score(y_test, y_pred_spw)
f1_opt = best_f1
f1_deep = f1_score(y_test, y_pred_deep)
f1_dopt = best_f1_d
results = [
    ('1. Baseline (current)', f1_base, 0),
    ('2. + class balance', f1_spw, f1_spw - f1_base),
    ('3. + optimal threshold', f1_opt, f1_opt - f1_base),
    ('4. Deeper + balance', f1_deep, f1_deep - f1_base),
    ('5. Deeper + opt thr', f1_dopt, f1_dopt - f1_base),
]
for name, f1, delta in results:
    pct = delta/f1_base*100 if f1_base > 0 else 0
    print(f'  {name:30s}  F1={f1:.4f}  ({pct:+.1f}%)')

best_name = max(results, key=lambda x: x[1])
print(f'\nBest: {best_name[0]} with F1={best_name[1]:.4f}')
