"""
MODEL BACKTEST & VERIFICATION
=============================
Loads the production meta-labeling models, evaluates on the held-out test
period (last 20 trading days), and simulates directional option trades.

Sections:
  1. Model Accuracy Verification (reproduce training metrics)
  2. Per-Stock Direction Accuracy (which stocks does the model work best on?)
  3. Simulated Trading P&L (ML-guided CE/PE trades on 5-min candles)
  4. Walk-Forward Stability (5-day rolling accuracy)

Usage:
  cd agentic_trader
  python backtest_model.py [--atr-factor 1.5] [--test-days 20]
"""

import os
import sys
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.feature_engineering import compute_features, get_feature_names
from ml_models.label_creator import create_labels
from ml_models.trainer import load_and_prepare_data, LABEL_MAP, LABEL_NAMES, MODELS_DIR


# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SimTrade:
    """A simulated directional trade"""
    symbol: str
    date: str
    time: str
    direction: str      # UP or DOWN
    ml_confidence: float  # P(UP|MOVE) or P(DOWN|MOVE)
    gate_prob: float     # P(MOVE)
    entry_price: float
    
    # Outcome
    actual_label: int    # 0=DOWN, 1=FLAT, 2=UP  
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    correct: bool = False
    
    # Option P&L simulation
    option_pnl: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ═══════════════════════════════════════════════════════════════════

def load_production_models():
    """Load the production meta-labeling models."""
    gate_path = MODELS_DIR / "meta_gate_latest.json"
    dir_path = MODELS_DIR / "meta_direction_latest.json"
    gate_cal_path = MODELS_DIR / "meta_gate_latest_calibrator.pkl"
    dir_cal_path = MODELS_DIR / "meta_direction_latest_calibrator.pkl"
    meta_path = MODELS_DIR / "meta_labeling_latest_meta.json"
    
    for p in [gate_path, dir_path]:
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
    
    gate_model = xgb.XGBClassifier()
    gate_model.load_model(str(gate_path))
    
    dir_model = xgb.XGBClassifier()
    dir_model.load_model(str(dir_path))
    
    gate_cal = joblib.load(str(gate_cal_path)) if gate_cal_path.exists() else None
    dir_cal = joblib.load(str(dir_cal_path)) if dir_cal_path.exists() else None
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    feature_names = metadata.get('feature_names', get_feature_names())
    dir_feature_names = metadata.get('direction_feature_names', feature_names)
    
    print(f"✅ Models loaded: {metadata.get('timestamp', '?')}")
    print(f"   ATR factor: {metadata.get('atr_factor', '?')}")
    print(f"   Features: {len(feature_names)} (gate), {len(dir_feature_names)} (direction)")
    print(f"   Gate calibrator: {'✓' if gate_cal else '✗'}")
    print(f"   Direction calibrator: {'✓' if dir_cal else '✗'}")
    
    return {
        'gate_model': gate_model,
        'dir_model': dir_model,
        'gate_cal': gate_cal,
        'dir_cal': dir_cal,
        'metadata': metadata,
        'feature_names': feature_names,
        'dir_feature_names': dir_feature_names,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: MODEL ACCURACY VERIFICATION
# ═══════════════════════════════════════════════════════════════════

def verify_model_accuracy(models: dict, test_df: pd.DataFrame, feature_names: list):
    """Reproduce training metrics on test set to verify models are correct."""
    print(f"\n{'='*70}")
    print(f"  SECTION 1: MODEL ACCURACY VERIFICATION")
    print(f"{'='*70}")
    
    gate_model = models['gate_model']
    dir_model = models['dir_model']
    gate_cal = models['gate_cal']
    dir_cal = models['dir_cal']
    dir_feature_names = models['dir_feature_names']
    
    X_test = test_df[feature_names].values
    y_test = test_df['label_idx'].values.astype(int)
    
    # Binary gate labels
    test_gate_y = (y_test != 1).astype(int)  # MOVE=1, FLAT=0
    
    # Gate predictions
    gate_raw = gate_model.predict_proba(X_test)[:, 1]
    gate_cal_probs = gate_cal.predict(gate_raw) if gate_cal else gate_raw
    gate_pred = (gate_cal_probs >= 0.5).astype(int)
    
    gate_acc = accuracy_score(test_gate_y, gate_pred)
    gate_prec = precision_score(test_gate_y, gate_pred, pos_label=1, zero_division=0)
    gate_rec = recall_score(test_gate_y, gate_pred, pos_label=1, zero_division=0)
    
    print(f"\n  ┌─ GATE MODEL (MOVE vs FLAT) ─────────────────┐")
    print(f"  │  Accuracy:  {gate_acc:.1%}                        │")
    print(f"  │  MOVE prec: {gate_prec:.1%} | recall: {gate_rec:.1%}       │")
    print(f"  └────────────────────────────────────────────────┘")
    
    # Gate high-confidence
    print(f"\n  Gate High-Confidence Tiers:")
    for thr in [0.50, 0.60, 0.70, 0.80]:
        mask = gate_cal_probs >= thr
        count = mask.sum()
        if count > 0:
            correct = test_gate_y[mask].sum()
            prec = correct / count
            print(f"    MOVE≥{thr:.0%}: {count:>6,} signals, precision={prec:.1%}")
    
    # Direction model on MOVE samples
    move_mask = y_test != 1
    X_test_move = test_df.loc[test_df.index[move_mask], feature_names].values
    
    # Pruned features for direction
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
        X_dir = X_test_move[:, dir_indices]
    else:
        X_dir = X_test_move
    
    test_dir_y = (y_test[move_mask] == 2).astype(int)  # 1=UP, 0=DOWN
    
    dir_raw = dir_model.predict_proba(X_dir)[:, 1]
    dir_cal_probs = dir_cal.predict(dir_raw) if dir_cal else dir_raw
    dir_pred = (dir_cal_probs >= 0.5).astype(int)
    
    dir_acc = accuracy_score(test_dir_y, dir_pred)
    dir_prec_up = precision_score(test_dir_y, dir_pred, pos_label=1, zero_division=0)
    dir_prec_down = precision_score(test_dir_y, dir_pred, pos_label=0, zero_division=0)
    
    print(f"\n  ┌─ DIRECTION MODEL (UP vs DOWN, MOVE only) ────┐")
    print(f"  │  Accuracy:    {dir_acc:.1%}                       │")
    print(f"  │  UP precision:   {dir_prec_up:.1%}                    │")
    print(f"  │  DOWN precision: {dir_prec_down:.1%}                    │")
    print(f"  └────────────────────────────────────────────────┘")
    
    # Direction high-confidence
    print(f"\n  Direction High-Confidence (MOVE samples only):")
    print(f"  {'Threshold':<12s} {'UP signals':>12s} {'UP prec':>10s} │ {'DOWN signals':>13s} {'DOWN prec':>10s}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} │ {'─'*13} {'─'*10}")
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # UP
        up_mask = dir_cal_probs >= thr
        up_count = up_mask.sum()
        up_prec = test_dir_y[up_mask].sum() / max(up_count, 1)
        # DOWN
        down_mask = dir_cal_probs <= (1 - thr)
        down_count = down_mask.sum()
        down_prec = (test_dir_y[down_mask] == 0).sum() / max(down_count, 1)
        
        print(f"  ≥{thr:.0%}         {up_count:>8,}     {up_prec:>8.1%}  │  {down_count:>9,}     {down_prec:>8.1%}")
    
    # Combined system
    print(f"\n  ┌─ COMBINED SYSTEM (all test samples) ──────────┐")
    
    # Get predictions for ALL test samples
    all_gate = gate_cal.predict(gate_model.predict_proba(X_test)[:, 1]) if gate_cal else gate_model.predict_proba(X_test)[:, 1]
    
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
        X_all_dir = X_test[:, dir_indices]
    else:
        X_all_dir = X_test
    
    all_dir_raw = dir_model.predict_proba(X_all_dir)[:, 1]
    all_dir = dir_cal.predict(all_dir_raw) if dir_cal else all_dir_raw
    
    p_up = all_gate * all_dir
    p_down = all_gate * (1 - all_dir)
    p_flat = 1 - all_gate
    
    combined_probs = np.stack([p_down, p_flat, p_up], axis=1)
    combined_pred = np.argmax(combined_probs, axis=1)
    
    combined_acc = accuracy_score(y_test, combined_pred)
    macro_f1 = f1_score(y_test, combined_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, combined_pred, labels=[0, 1, 2])
    
    print(f"  │  3-class accuracy: {combined_acc:.1%}                   │")
    print(f"  │  Macro F1:         {macro_f1:.3f}                   │")
    print(f"  └────────────────────────────────────────────────┘")
    
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = ''.join(f"  {'P_'+LABEL_NAMES[i]:>8s}" for i in range(3))
    print(f"  {'':>10s}{header}")
    for i, row in enumerate(cm):
        vals = ''.join(f"  {row[j]:>8d}" for j in range(3))
        print(f"  {LABEL_NAMES[i]:>10s}{vals}")
    
    return {
        'gate_acc': gate_acc, 'dir_acc': dir_acc, 'combined_acc': combined_acc,
        'macro_f1': macro_f1, 'gate_cal_probs': gate_cal_probs,
        'all_gate': all_gate, 'all_dir': all_dir,
        'p_up': p_up, 'p_down': p_down, 'p_flat': p_flat,
        'dir_cal_probs': dir_cal_probs, 'test_dir_y': test_dir_y,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: PER-STOCK ACCURACY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════

def per_stock_analysis(models: dict, test_df: pd.DataFrame, feature_names: list):
    """Break down direction accuracy per stock."""
    print(f"\n{'='*70}")
    print(f"  SECTION 2: PER-STOCK DIRECTION ACCURACY")
    print(f"{'='*70}")
    
    gate_model = models['gate_model']
    dir_model = models['dir_model']
    gate_cal = models['gate_cal']
    dir_cal = models['dir_cal']
    dir_feature_names = models['dir_feature_names']
    
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
    else:
        dir_indices = None
    
    results = []
    
    if 'symbol' not in test_df.columns:
        print("  ⚠ No 'symbol' column in test data — cannot do per-stock analysis")
        return results
    
    symbols = sorted(test_df['symbol'].unique())
    
    print(f"\n  {'Symbol':<16s} {'Samples':>8s} {'MOVE%':>7s} {'Gate':>7s} {'Dir':>7s} {'Dir≥60%':>8s} {'Dir≥70%':>8s} │ {'Best_tier':>10s}")
    print(f"  {'─'*16} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*8} │ {'─'*10}")
    
    for sym in symbols:
        sym_mask = test_df['symbol'] == sym
        sym_df = test_df[sym_mask]
        X_sym = sym_df[feature_names].values
        y_sym = sym_df['label_idx'].values.astype(int)
        
        n_total = len(y_sym)
        n_move = (y_sym != 1).sum()
        move_pct = n_move / max(n_total, 1) * 100
        
        # Gate accuracy for this stock
        gate_y = (y_sym != 1).astype(int)
        gate_raw = gate_model.predict_proba(X_sym)[:, 1]
        gate_probs = gate_cal.predict(gate_raw) if gate_cal else gate_raw
        gate_pred = (gate_probs >= 0.5).astype(int)
        gate_acc = accuracy_score(gate_y, gate_pred) if n_total > 0 else 0
        
        # Direction accuracy (MOVE samples only)
        move_mask = y_sym != 1
        if move_mask.sum() < 10:
            results.append({'symbol': sym, 'n': n_total, 'dir_acc': 0, 'skip': True})
            print(f"  {sym:<16s} {n_total:>8d} {move_pct:>6.1f}% {gate_acc:>6.1%}   ---     ---      ---    │     <10 MOVE")
            continue
        
        X_move = X_sym[move_mask]
        dir_y = (y_sym[move_mask] == 2).astype(int)
        
        if dir_indices is not None:
            X_dir = X_move[:, dir_indices]
        else:
            X_dir = X_move
        
        dir_raw = dir_model.predict_proba(X_dir)[:, 1]
        dir_probs = dir_cal.predict(dir_raw) if dir_cal else dir_raw
        dir_pred = (dir_probs >= 0.5).astype(int)
        dir_acc = accuracy_score(dir_y, dir_pred)
        
        # High-confidence direction accuracy
        dir_60_mask = (dir_probs >= 0.60) | (dir_probs <= 0.40)
        n_60 = dir_60_mask.sum()
        if n_60 > 0:
            dir_60_pred = (dir_probs[dir_60_mask] >= 0.5).astype(int)
            dir_60_acc = accuracy_score(dir_y[dir_60_mask], dir_60_pred)
        else:
            dir_60_acc = 0
        
        dir_70_mask = (dir_probs >= 0.70) | (dir_probs <= 0.30)
        n_70 = dir_70_mask.sum()
        if n_70 > 0:
            dir_70_pred = (dir_probs[dir_70_mask] >= 0.5).astype(int)
            dir_70_acc = accuracy_score(dir_y[dir_70_mask], dir_70_pred)
        else:
            dir_70_acc = 0
        
        # Determine best tier
        if n_70 > 5 and dir_70_acc >= 0.70:
            best_tier = f"★ {dir_70_acc:.0%}"
        elif n_60 > 5 and dir_60_acc >= 0.65:
            best_tier = f"● {dir_60_acc:.0%}"
        elif dir_acc >= 0.55:
            best_tier = f"○ {dir_acc:.0%}"
        else:
            best_tier = f"  {dir_acc:.0%}"
        
        results.append({
            'symbol': sym, 'n': n_total, 'n_move': n_move,
            'gate_acc': gate_acc, 'dir_acc': dir_acc,
            'dir_60_acc': dir_60_acc, 'n_60': n_60,
            'dir_70_acc': dir_70_acc, 'n_70': n_70,
            'skip': False,
        })
        
        dir_60_str = f"{dir_60_acc:.0%}({n_60})" if n_60 > 0 else "---"
        dir_70_str = f"{dir_70_acc:.0%}({n_70})" if n_70 > 0 else "---"
        
        print(f"  {sym:<16s} {n_total:>8d} {move_pct:>6.1f}% {gate_acc:>6.1%} {dir_acc:>6.1%} {dir_60_str:>8s} {dir_70_str:>8s} │ {best_tier:>10s}")
    
    # Summary stats
    valid = [r for r in results if not r.get('skip', False)]
    if valid:
        avg_dir = np.mean([r['dir_acc'] for r in valid])
        best = max(valid, key=lambda r: r['dir_acc'])
        worst = min(valid, key=lambda r: r['dir_acc'])
        
        gt55 = sum(1 for r in valid if r['dir_acc'] >= 0.55)
        gt60 = sum(1 for r in valid if r.get('dir_60_acc', 0) >= 0.65)
        
        print(f"\n  Summary: {len(valid)} stocks analyzed")
        print(f"  Avg direction accuracy: {avg_dir:.1%}")
        print(f"  Best:  {best['symbol']} ({best['dir_acc']:.1%})")
        print(f"  Worst: {worst['symbol']} ({worst['dir_acc']:.1%})")
        print(f"  Stocks with Dir≥55%: {gt55}/{len(valid)}")
        print(f"  Stocks with Dir≥60% (≥65% acc): {gt60}/{len(valid)}")
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: SIMULATED TRADING P&L
# ═══════════════════════════════════════════════════════════════════

def simulate_trading(models: dict, test_df: pd.DataFrame, feature_names: list, 
                     initial_capital: float = 500_000):
    """Simulate directional option trades using ML signals.
    
    Strategy:
    - If P(MOVE)≥0.55 & P(UP|MOVE)≥0.60 → Buy CE (long call)
    - If P(MOVE)≥0.55 & P(DOWN|MOVE)≥0.60 → Buy PE (long put)
    - Otherwise: no trade
    
    P&L approximation:
    - Option premium ≈ stock_price × 0.015 (1.5% of stock price)
    - Delta ≈ 0.50 for ATM options
    - Gamma effect: if stock moves 1%, option moves ~1.5% (delta + gamma)
    - Holding period: 6 candles (30 min) = the label lookahead
    - SL: -3% of premium
    - Target: +5% of premium
    """
    print(f"\n{'='*70}")
    print(f"  SECTION 3: SIMULATED TRADING P&L")
    print(f"{'='*70}")
    
    gate_model = models['gate_model']
    dir_model = models['dir_model']
    gate_cal = models['gate_cal']
    dir_cal = models['dir_cal']
    dir_feature_names = models['dir_feature_names']
    
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
    else:
        dir_indices = None
    
    X_test = test_df[feature_names].values
    y_test = test_df['label_idx'].values.astype(int)
    
    # Gate predictions
    gate_raw = gate_model.predict_proba(X_test)[:, 1]
    gate_probs = gate_cal.predict(gate_raw) if gate_cal else gate_raw
    
    # Direction predictions
    if dir_indices is not None:
        X_dir = X_test[:, dir_indices]
    else:
        X_dir = X_test
    
    dir_raw = dir_model.predict_proba(X_dir)[:, 1]
    dir_probs = dir_cal.predict(dir_raw) if dir_cal else dir_raw
    
    trades: List[SimTrade] = []
    capital = initial_capital
    
    # Score tiers for position sizing
    PREMIUM_THR = 0.70
    STANDARD_THR = 0.65
    BASE_THR = 0.60
    
    dates = test_df['date'].dt.date.values if 'date' in test_df.columns else [None]*len(test_df)
    times = test_df['date'].dt.time.values if 'date' in test_df.columns else [None]*len(test_df)
    symbols = test_df['symbol'].values if 'symbol' in test_df.columns else ['UNKNOWN']*len(test_df)
    close_prices = test_df['close'].values if 'close' in test_df.columns else np.ones(len(test_df))
    
    # Lookahead P&L columns (from label creator)
    has_pnl_cols = 'max_up_pct' in test_df.columns and 'max_down_pct' in test_df.columns
    if has_pnl_cols:
        max_up = test_df['max_up_pct'].values
        max_down = test_df['max_down_pct'].values
    
    n_signals = 0
    n_correct = 0
    total_pnl = 0.0
    daily_pnl = defaultdict(float)
    tier_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    for i in range(len(X_test)):
        p_move = gate_probs[i]
        p_up = dir_probs[i]
        p_down = 1 - p_up
        
        # Entry filter
        if p_move < 0.55:
            continue
        
        direction = None
        confidence = 0.0
        
        if p_up >= BASE_THR:
            direction = 'UP'
            confidence = p_up
        elif p_down >= BASE_THR:
            direction = 'DOWN'
            confidence = p_down
        else:
            continue
        
        # Determine tier
        if confidence >= PREMIUM_THR:
            tier = 'PREMIUM'
            size_pct = 0.05  # 5% of capital per trade
        elif confidence >= STANDARD_THR:
            tier = 'STANDARD'
            size_pct = 0.04
        else:
            tier = 'BASE'
            size_pct = 0.03
        
        actual = y_test[i]
        entry_price = close_prices[i]
        
        # Option P&L approximation
        # Premium ≈ 1.5% of stock price for ATM weekly
        premium = entry_price * 0.015
        
        # Position size
        n_lots = max(1, int(capital * size_pct / premium))
        
        # Compute P&L based on actual outcome
        if direction == 'UP':
            correct = (actual == 2)  # Actually went UP
            if has_pnl_cols:
                stock_move = max_up[i] if actual == 2 else -max_down[i]
            else:
                stock_move = 0.5 if actual == 2 else (-0.5 if actual == 0 else 0)
            
            # ATM CE: delta ~0.50, gamma adds ~0.15× for 1% move
            option_move_pct = stock_move * 0.65 * (1 + abs(stock_move) * 0.1)  # Rough option multiplier
            
        else:  # DOWN
            correct = (actual == 0)  # Actually went DOWN
            if has_pnl_cols:
                stock_move = max_down[i] if actual == 0 else -max_up[i]
            else:
                stock_move = 0.5 if actual == 0 else (-0.5 if actual == 2 else 0)
            
            option_move_pct = stock_move * 0.65 * (1 + abs(stock_move) * 0.1)
        
        # Apply basic SL/target logic
        option_pnl_pct = max(-3.0, min(5.0, option_move_pct))  # Cap at -3% SL, +5% target
        option_pnl = premium * n_lots * option_pnl_pct / 100
        
        trade = SimTrade(
            symbol=str(symbols[i]),
            date=str(dates[i]),
            time=str(times[i]),
            direction=direction,
            ml_confidence=confidence,
            gate_prob=p_move,
            entry_price=entry_price,
            actual_label=int(actual),
            correct=correct,
            pnl_pct=option_pnl_pct,
            option_pnl=option_pnl,
            max_favorable=float(max_up[i] if direction == 'UP' and has_pnl_cols else 0),
            max_adverse=float(max_down[i] if direction == 'UP' and has_pnl_cols else 0),
        )
        
        trades.append(trade)
        n_signals += 1
        if correct:
            n_correct += 1
        total_pnl += option_pnl
        daily_pnl[str(dates[i])] += option_pnl
        
        tier_stats[tier]['trades'] += 1
        if correct:
            tier_stats[tier]['wins'] += 1
        tier_stats[tier]['pnl'] += option_pnl
    
    # Print results
    if n_signals == 0:
        print("  No signals generated — check thresholds")
        return trades
    
    win_rate = n_correct / n_signals * 100
    winners = [t for t in trades if t.correct]
    losers = [t for t in trades if not t.correct]
    
    avg_win = np.mean([t.option_pnl for t in winners]) if winners else 0
    avg_loss = np.mean([abs(t.option_pnl) for t in losers]) if losers else 0
    profit_factor = sum(t.option_pnl for t in winners) / max(sum(abs(t.option_pnl) for t in losers), 1)
    
    print(f"\n  ┌─ TRADING PERFORMANCE ──────────────────────────┐")
    print(f"  │  Capital:       ₹{initial_capital:>12,.0f}                │")
    print(f"  │  Total signals: {n_signals:>8,}                        │")
    print(f"  │  Wins/Losses:   {n_correct:,}/{n_signals - n_correct:,}                        │")
    print(f"  │  Win rate:      {win_rate:>7.1f}%                       │")
    print(f"  │  Total P&L:     ₹{total_pnl:>+12,.0f}                │")
    print(f"  │  ROI:           {total_pnl/initial_capital*100:>+7.2f}%                       │")
    print(f"  │  Avg win:       ₹{avg_win:>+10,.0f}                  │")
    print(f"  │  Avg loss:      ₹{avg_loss:>10,.0f}                  │")
    print(f"  │  Profit factor: {profit_factor:>7.2f}                       │")
    print(f"  └─────────────────────────────────────────────────┘")
    
    # Tier breakdown
    print(f"\n  Tier Breakdown:")
    print(f"  {'Tier':<12s} {'Trades':>8s} {'Wins':>6s} {'WR%':>7s} {'P&L':>12s}")
    print(f"  {'─'*12} {'─'*8} {'─'*6} {'─'*7} {'─'*12}")
    for tier in ['PREMIUM', 'STANDARD', 'BASE']:
        s = tier_stats.get(tier, {'trades': 0, 'wins': 0, 'pnl': 0})
        wr = s['wins'] / max(s['trades'], 1) * 100
        print(f"  {tier:<12s} {s['trades']:>8,} {s['wins']:>6,} {wr:>6.1f}% ₹{s['pnl']:>+10,.0f}")
    
    # Direction breakdown
    up_trades = [t for t in trades if t.direction == 'UP']
    down_trades = [t for t in trades if t.direction == 'DOWN']
    up_wins = sum(1 for t in up_trades if t.correct)
    down_wins = sum(1 for t in down_trades if t.correct)
    
    print(f"\n  Direction Breakdown:")
    print(f"  {'Dir':<8s} {'Trades':>8s} {'Wins':>6s} {'WR%':>7s} {'P&L':>12s}")
    print(f"  {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*12}")
    if up_trades:
        up_pnl = sum(t.option_pnl for t in up_trades)
        print(f"  {'UP':<8s} {len(up_trades):>8,} {up_wins:>6,} {up_wins/len(up_trades)*100:>6.1f}% ₹{up_pnl:>+10,.0f}")
    if down_trades:
        down_pnl = sum(t.option_pnl for t in down_trades)
        print(f"  {'DOWN':<8s} {len(down_trades):>8,} {down_wins:>6,} {down_wins/len(down_trades)*100:>6.1f}% ₹{down_pnl:>+10,.0f}")
    
    # Daily P&L
    print(f"\n  Daily P&L (test period):")
    sorted_days = sorted(daily_pnl.keys())
    cumulative = 0.0
    for day in sorted_days:
        cumulative += daily_pnl[day]
        bar = '█' * max(0, int(daily_pnl[day] / max(abs(v) for v in daily_pnl.values()) * 20)) if daily_pnl[day] > 0 else '░' * max(0, int(-daily_pnl[day] / max(abs(v) for v in daily_pnl.values()) * 20))
        print(f"    {day}  ₹{daily_pnl[day]:>+10,.0f}  cum: ₹{cumulative:>+10,.0f}  {bar}")
    
    return trades


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: WALK-FORWARD STABILITY
# ═══════════════════════════════════════════════════════════════════

def walk_forward_stability(models: dict, test_df: pd.DataFrame, feature_names: list, window_days: int = 5):
    """Check if model accuracy is stable across 5-day windows."""
    print(f"\n{'='*70}")
    print(f"  SECTION 4: WALK-FORWARD STABILITY ({window_days}-day windows)")
    print(f"{'='*70}")
    
    gate_model = models['gate_model']
    dir_model = models['dir_model']
    gate_cal = models['gate_cal']
    dir_cal = models['dir_cal']
    dir_feature_names = models['dir_feature_names']
    
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
    else:
        dir_indices = None
    
    if 'date' not in test_df.columns:
        print("  ⚠ No date column — cannot do walk-forward analysis")
        return
    
    test_df = test_df.copy()
    test_df['_date'] = test_df['date'].dt.date
    all_dates = sorted(test_df['_date'].unique())
    
    print(f"\n  Test period: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} trading days)")
    print(f"\n  {'Window':<25s} {'Samples':>8s} {'Gate':>7s} {'Dir':>7s} {'Dir≥60%':>8s} {'Combined':>9s}")
    print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*9}")
    
    window_results = []
    
    for start_idx in range(0, len(all_dates), window_days):
        end_idx = min(start_idx + window_days, len(all_dates))
        window_dates = all_dates[start_idx:end_idx]
        
        if len(window_dates) < 2:
            continue
        
        mask = test_df['_date'].isin(window_dates)
        w_df = test_df[mask]
        
        X_w = w_df[feature_names].values
        y_w = w_df['label_idx'].values.astype(int)
        
        # Gate
        gate_y = (y_w != 1).astype(int)
        gate_raw = gate_model.predict_proba(X_w)[:, 1]
        gate_probs = gate_cal.predict(gate_raw) if gate_cal else gate_raw
        gate_pred = (gate_probs >= 0.5).astype(int)
        gate_acc = accuracy_score(gate_y, gate_pred)
        
        # Direction (MOVE only)
        move_mask = y_w != 1
        if move_mask.sum() < 5:
            continue
        
        X_move = X_w[move_mask]
        dir_y = (y_w[move_mask] == 2).astype(int)
        
        if dir_indices is not None:
            X_dir = X_move[:, dir_indices]
        else:
            X_dir = X_move
        
        dir_raw = dir_model.predict_proba(X_dir)[:, 1]
        dir_probs = dir_cal.predict(dir_raw) if dir_cal else dir_raw
        dir_pred = (dir_probs >= 0.5).astype(int)
        dir_acc = accuracy_score(dir_y, dir_pred)
        
        # High-confidence direction
        hc_mask = (dir_probs >= 0.60) | (dir_probs <= 0.40)
        n_hc = hc_mask.sum()
        if n_hc > 0:
            hc_pred = (dir_probs[hc_mask] >= 0.5).astype(int)
            hc_acc = accuracy_score(dir_y[hc_mask], hc_pred)
        else:
            hc_acc = 0
        
        # Combined
        gate_all = gate_cal.predict(gate_model.predict_proba(X_w)[:, 1]) if gate_cal else gate_model.predict_proba(X_w)[:, 1]
        if dir_indices is not None:
            X_all_dir = X_w[:, dir_indices]
        else:
            X_all_dir = X_w
        dir_all = dir_cal.predict(dir_model.predict_proba(X_all_dir)[:, 1]) if dir_cal else dir_model.predict_proba(X_all_dir)[:, 1]
        p_up = gate_all * dir_all
        p_down = gate_all * (1 - dir_all)
        p_flat = 1 - gate_all
        comb = np.argmax(np.stack([p_down, p_flat, p_up], axis=1), axis=1)
        comb_acc = accuracy_score(y_w, comb)
        
        window_label = f"{window_dates[0]} → {window_dates[-1]}"
        hc_str = f"{hc_acc:.0%}({n_hc})" if n_hc > 0 else "---"
        
        # Color indicator
        dir_indicator = "✓" if dir_acc >= 0.55 else "△" if dir_acc >= 0.50 else "✗"
        
        print(f"  {window_label:<25s} {len(y_w):>8d} {gate_acc:>6.1%} {dir_acc:>6.1%} {hc_str:>8s} {comb_acc:>8.1%} {dir_indicator}")
        
        window_results.append({
            'start': str(window_dates[0]), 'end': str(window_dates[-1]),
            'samples': len(y_w), 'gate_acc': gate_acc, 'dir_acc': dir_acc,
            'hc_acc': hc_acc, 'combined_acc': comb_acc,
        })
    
    if window_results:
        dir_accs = [w['dir_acc'] for w in window_results]
        print(f"\n  Direction accuracy range: {min(dir_accs):.1%} → {max(dir_accs):.1%}")
        print(f"  Direction accuracy std:   {np.std(dir_accs):.1%}")
        print(f"  Windows ≥55% dir:         {sum(1 for a in dir_accs if a >= 0.55)}/{len(dir_accs)}")
        
        if np.std(dir_accs) > 0.05:
            print(f"\n  ⚠ High variance across windows — model may be overfit or regime-dependent")
        else:
            print(f"\n  ✓ Low variance across windows — model is stable")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: CONFIDENCE CALIBRATION CHECK
# ═══════════════════════════════════════════════════════════════════

def calibration_check(verify_results: dict):
    """Check if predicted probabilities match actual frequencies."""
    print(f"\n{'='*70}")
    print(f"  SECTION 5: CONFIDENCE CALIBRATION CHECK")
    print(f"{'='*70}")
    
    dir_probs = verify_results['dir_cal_probs']
    dir_y = verify_results['test_dir_y']
    
    print(f"\n  Direction model calibration (MOVE samples):")
    print(f"  {'Predicted P(UP)':>20s} {'Actual UP freq':>16s} {'Samples':>10s} {'Calibration':>12s}")
    print(f"  {'─'*20} {'─'*16} {'─'*10} {'─'*12}")
    
    bins = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), 
            (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
    
    for lo, hi in bins:
        mask = (dir_probs >= lo) & (dir_probs < hi)
        count = mask.sum()
        if count < 5:
            continue
        actual_freq = dir_y[mask].mean()
        predicted_mid = (lo + hi) / 2
        diff = abs(actual_freq - predicted_mid)
        
        cal_status = "✓ Good" if diff < 0.05 else ("△ Fair" if diff < 0.10 else "✗ Poor")
        
        print(f"  {lo:.2f} - {hi:.2f}         {actual_freq:>12.1%}   {count:>8,}   {cal_status}")
    
    # Gate calibration
    gate_probs = verify_results['gate_cal_probs']
    gate_y = verify_results.get('gate_cal_probs', None)  # Recompute from actuals
    
    print(f"\n  (Direction model calibration shows how well predicted probabilities")
    print(f"   match actual outcomes. A well-calibrated model has predicted and")
    print(f"   actual frequencies aligned.)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Backtest & Verify ML Models')
    parser.add_argument('--atr-factor', type=float, default=1.5, help='ATR threshold factor (default: 1.5)')
    parser.add_argument('--test-days', type=int, default=20, help='Test period days (default: 20)')
    parser.add_argument('--val-days', type=int, default=10, help='Val period days (default: 10)')
    parser.add_argument('--capital', type=float, default=500_000, help='Simulated capital (default: 500000)')
    args = parser.parse_args()
    
    print(f"\n{'█'*70}")
    print(f"  ML MODEL BACKTEST & VERIFICATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  ATR factor: {args.atr_factor}")
    print(f"  Test days: {args.test_days}")
    print(f"{'█'*70}")
    
    # Load models
    t0 = time.time()
    models = load_production_models()
    feature_names = models['feature_names']
    
    # Load data (same pipeline as training)
    print(f"\n📊 Loading & preparing test data...")
    _, _, test_df, loaded_features = load_and_prepare_data(
        atr_factor=args.atr_factor,
        test_days=args.test_days,
        val_days=args.val_days,
    )
    
    # Verify feature alignment
    if loaded_features != feature_names:
        print(f"  ⚠ Feature mismatch: model expects {len(feature_names)}, data has {len(loaded_features)}")
        # Find mismatches
        in_model = set(feature_names) - set(loaded_features)
        in_data = set(loaded_features) - set(feature_names)
        if in_model:
            print(f"    In model but not data: {in_model}")
        if in_data:
            print(f"    In data but not model: {in_data}")
    else:
        print(f"  ✓ Feature alignment confirmed: {len(feature_names)} features")
    
    load_time = time.time() - t0
    print(f"  Data loaded in {load_time:.1f}s — {len(test_df):,} test samples")
    
    # Run all sections
    verify_results = verify_model_accuracy(models, test_df, feature_names)
    per_stock_analysis(models, test_df, feature_names)
    simulate_trading(models, test_df, feature_names, initial_capital=args.capital)
    walk_forward_stability(models, test_df, feature_names)
    calibration_check(verify_results)
    
    print(f"\n{'█'*70}")
    print(f"  BACKTEST COMPLETE — Total time: {time.time() - t0:.1f}s")
    print(f"{'█'*70}\n")


if __name__ == '__main__':
    main()
