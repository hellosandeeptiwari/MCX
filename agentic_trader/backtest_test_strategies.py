"""
BACKTEST: TEST_GMM (Regime Divergence) & TEST_XGB (Pure XGB Conviction)
=======================================================================
Loads production models, runs inference on the ENTIRE historical dataset,
and simulates entry signals for both TEST_GMM and TEST_XGB strategies.

Measures:
  - How often each strategy triggers (signal rate)
  - Forward returns (next 8 candles = 40 min) after signal
  - Win rate, avg win, avg loss, expectancy
  - Directional accuracy (did the predicted direction materialize?)

Usage:
  cd agentic_trader
  python backtest_test_strategies.py [--test-days 20]
"""

import os, sys, json, time, joblib, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Optional

import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_models.feature_engineering import compute_features, get_feature_names
from ml_models.label_creator import create_labels
from ml_models.trainer import load_and_prepare_data, LABEL_MAP, LABEL_NAMES, MODELS_DIR
from ml_models.down_risk_detector import DownRiskDetector, DetectorConfig, load_detector_dataset
from config import TEST_GMM, TEST_XGB


# ═══════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ═══════════════════════════════════════════════════════════════════

def load_xgb_models():
    """Load production XGB gate + direction models."""
    gate_path = MODELS_DIR / "meta_gate_latest.json"
    dir_path = MODELS_DIR / "meta_direction_latest.json"
    gate_cal_path = MODELS_DIR / "meta_gate_latest_calibrator.pkl"
    dir_cal_path = MODELS_DIR / "meta_direction_latest_calibrator.pkl"
    meta_path = MODELS_DIR / "meta_labeling_latest_meta.json"
    
    gate_model = xgb.XGBClassifier()
    gate_model.load_model(str(gate_path))
    
    dir_model = xgb.XGBClassifier()
    dir_model.load_model(str(dir_path))
    
    gate_cal = joblib.load(str(gate_cal_path)) if gate_cal_path.exists() else None
    dir_cal = joblib.load(str(dir_cal_path)) if dir_cal_path.exists() else None
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    return {
        'gate_model': gate_model,
        'dir_model': dir_model,
        'gate_cal': gate_cal,
        'dir_cal': dir_cal,
        'metadata': metadata,
        'feature_names': metadata.get('feature_names', get_feature_names()),
        'dir_feature_names': metadata.get('direction_feature_names', 
                                          metadata.get('feature_names', get_feature_names())),
    }


def load_dr_detector():
    """Load production Down-Risk detector."""
    config = DetectorConfig()
    detector = DownRiskDetector(config)
    if not detector.load():
        print("⚠️ Down-Risk detector not found — TEST_GMM backtest skipped")
        return None
    return detector


# ═══════════════════════════════════════════════════════════════════
#  COMPUTE FORWARD RETURNS
# ═══════════════════════════════════════════════════════════════════

def add_forward_returns(df: pd.DataFrame, lookahead: int = 8):
    """Add forward return (next N candles) per symbol."""
    fwd_returns = np.full(len(df), np.nan)
    fwd_max_up = np.full(len(df), np.nan)
    fwd_max_down = np.full(len(df), np.nan)
    
    for sym, grp in df.groupby('symbol'):
        idx = grp.index.values
        close = grp['close'].values
        
        for i_pos in range(len(idx)):
            i = idx[i_pos]
            end_pos = min(i_pos + lookahead, len(idx) - 1)
            if end_pos <= i_pos:
                continue
            
            entry = close[i_pos]
            future_slice = close[i_pos+1:end_pos+1]
            if len(future_slice) == 0:
                continue
            
            exit_price = future_slice[-1]
            fwd_returns[i] = (exit_price - entry) / entry
            fwd_max_up[i] = (future_slice.max() - entry) / entry
            fwd_max_down[i] = (future_slice.min() - entry) / entry
    
    df['fwd_return'] = fwd_returns
    df['fwd_max_up'] = fwd_max_up
    df['fwd_max_down'] = fwd_max_down
    return df


# ═══════════════════════════════════════════════════════════════════
#  XGB PREDICTIONS (batch)
# ═══════════════════════════════════════════════════════════════════

def compute_xgb_predictions(df: pd.DataFrame, xgb_models: dict):
    """Run XGB gate + direction on entire dataset."""
    feature_names = xgb_models['feature_names']
    dir_feature_names = xgb_models['dir_feature_names']
    
    X = df[feature_names].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gate: P(MOVE)
    gate_raw = xgb_models['gate_model'].predict_proba(X)[:, 1]
    if xgb_models['gate_cal'] is not None:
        move_prob = xgb_models['gate_cal'].predict(gate_raw)
    else:
        move_prob = gate_raw
    
    # Direction: P(UP|features)
    if dir_feature_names != feature_names:
        dir_indices = [feature_names.index(f) for f in dir_feature_names if f in feature_names]
        X_dir = X[:, dir_indices]
    else:
        X_dir = X
    
    dir_raw = xgb_models['dir_model'].predict_proba(X_dir)[:, 1]
    if xgb_models['dir_cal'] is not None:
        prob_up = xgb_models['dir_cal'].predict(dir_raw)
    else:
        prob_up = dir_raw
    
    prob_down = 1 - prob_up
    
    df['xgb_move_prob'] = move_prob
    df['xgb_prob_up'] = prob_up
    df['xgb_prob_down'] = prob_down
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  GMM DR PREDICTIONS (batch per regime)
# ═══════════════════════════════════════════════════════════════════

def compute_dr_predictions(df: pd.DataFrame, detector: DownRiskDetector, feature_names: list):
    """Run DR detector on entire dataset for both regimes."""
    
    # Use the DR detector's own feature names if available
    dr_feature_names = getattr(detector, '_feature_names', None) or feature_names
    
    # Get features that exist in both
    available = [f for f in dr_feature_names if f in df.columns]
    missing = [f for f in dr_feature_names if f not in df.columns]
    if missing:
        print(f"  ⚠️ {len(missing)} DR features missing, padding with 0")
    
    X = np.zeros((len(df), len(dr_feature_names)), dtype=np.float32)
    for i, f in enumerate(dr_feature_names):
        if f in df.columns:
            X[:, i] = df[f].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # UP regime predictions
    print("  Computing UP regime DR scores...")
    up_result = detector.predict_single(X, 'UP')
    df['dr_up_score'] = up_result['anomaly_score']
    df['dr_up_flag'] = up_result['anomaly_flag']
    
    # DOWN regime predictions
    print("  Computing DOWN regime DR scores...")
    down_result = detector.predict_single(X, 'DOWN')
    df['dr_down_score'] = down_result['anomaly_score']
    df['dr_down_flag'] = down_result['anomaly_flag']
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION: TEST_GMM (Regime Divergence)
# ═══════════════════════════════════════════════════════════════════

def generate_test_gmm_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TEST_GMM regime divergence logic to identify entries."""
    cfg = TEST_GMM
    
    call_min_dn = cfg.get('call_min_down_score', 0.20)
    call_max_up = cfg.get('call_max_up_score', 0.14)
    put_min_up = cfg.get('put_min_up_score', 0.18)
    put_max_dn = cfg.get('put_max_down_score', 0.14)
    min_gap = cfg.get('min_divergence_gap', 0.06)
    
    up_scores = df['dr_up_score'].values
    dn_scores = df['dr_down_score'].values
    
    # CALL: DOWN model high + UP model low
    call_ok = (dn_scores >= call_min_dn) & (up_scores <= call_max_up) & ((dn_scores - up_scores) >= min_gap)
    # PUT: UP model high + DOWN model low
    put_ok = (up_scores >= put_min_up) & (dn_scores <= put_max_dn) & ((up_scores - dn_scores) >= min_gap)
    
    # Direction and divergence
    direction = np.where(call_ok & ~put_ok, 'CALL',
               np.where(put_ok & ~call_ok, 'PUT',
               np.where(call_ok & put_ok, 
                        np.where((dn_scores - up_scores) >= (up_scores - dn_scores), 'CALL', 'PUT'),
                        'NONE')))
    
    divergence = np.where(direction == 'CALL', dn_scores - up_scores,
                np.where(direction == 'PUT', up_scores - dn_scores, 0.0))
    
    df['gmm_signal'] = direction
    df['gmm_divergence'] = divergence
    df['gmm_triggered'] = direction != 'NONE'
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION: TEST_XGB (Pure XGB Conviction)
# ═══════════════════════════════════════════════════════════════════

def generate_test_xgb_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TEST_XGB pure XGB logic to identify entries."""
    cfg = TEST_XGB
    
    min_move = cfg.get('min_move_prob', 0.58)
    min_dir_prob = cfg.get('min_directional_prob', 0.42)
    min_margin = cfg.get('min_directional_margin', 0.10)
    
    move_prob = df['xgb_move_prob'].values
    prob_up = df['xgb_prob_up'].values
    prob_down = df['xgb_prob_down'].values
    margin = np.abs(prob_up - prob_down)
    
    # Gate: P(MOVE) floor
    gate_ok = move_prob >= min_move
    
    # Direction: clear lean
    call_ok = gate_ok & (prob_up > prob_down) & (prob_up >= min_dir_prob) & (margin >= min_margin)
    put_ok = gate_ok & (prob_down > prob_up) & (prob_down >= min_dir_prob) & (margin >= min_margin)
    
    direction = np.where(call_ok & ~put_ok, 'CALL',
               np.where(put_ok & ~call_ok, 'PUT', 'NONE'))
    
    conviction = np.where(direction == 'CALL', move_prob * margin,
                np.where(direction == 'PUT', move_prob * margin, 0.0))
    
    df['xgb_signal'] = direction
    df['xgb_conviction'] = conviction
    df['xgb_triggered'] = direction != 'NONE'
    
    return df


# ═══════════════════════════════════════════════════════════════════
#  ANALYZE STRATEGY PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

def analyze_strategy(df: pd.DataFrame, signal_col: str, direction_col: str, 
                     name: str, metric_col: str = 'fwd_return'):
    """Analyze win rate, returns, expectancy for a strategy's signals."""
    
    signals = df[df[signal_col]].copy()
    total = len(df)
    n_signals = len(signals)
    
    if n_signals == 0:
        print(f"\n  === {name}: 0 signals out of {total:,} candles ===")
        print(f"  ⚠️ Strategy never triggers — thresholds too tight")
        return {}
    
    signal_rate = n_signals / total * 100
    print(f"\n{'='*70}")
    print(f"  {name}: {n_signals:,} signals / {total:,} candles ({signal_rate:.2f}%)")
    print(f"{'='*70}")
    
    # Filter valid forward returns
    valid = signals.dropna(subset=['fwd_return']).copy()
    if len(valid) == 0:
        print(f"  ⚠️ No valid forward returns (all boundary rows)")
        return {}
    
    # Directional return: positive if direction correct
    dir_vals = valid[direction_col].values
    fwd = valid['fwd_return'].values
    fwd_max_up = valid['fwd_max_up'].values
    fwd_max_down = valid['fwd_max_down'].values
    
    directed_return = np.where(dir_vals == 'CALL', fwd, -fwd)
    directed_max_fav = np.where(dir_vals == 'CALL', fwd_max_up, -fwd_max_down)
    directed_max_adv = np.where(dir_vals == 'CALL', fwd_max_down, -fwd_max_up)
    
    wins = directed_return > 0
    win_rate = wins.mean()
    avg_win = directed_return[wins].mean() if wins.any() else 0
    avg_loss = directed_return[~wins].mean() if (~wins).any() else 0
    avg_return = directed_return.mean()
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    
    # Max favorable excursion (reaching target)
    mfe_positive = (directed_max_fav > 0).mean()  # How often price moves in our favor at all
    
    print(f"\n  ┌─ PERFORMANCE ─────────────────────────────┐")
    print(f"  │  Valid signals:    {len(valid):>8,}                │")
    print(f"  │  Win rate:         {win_rate:>8.1%}                │")
    print(f"  │  Avg win:          {avg_win*100:>+7.3f}%               │")
    print(f"  │  Avg loss:         {avg_loss*100:>+7.3f}%               │")
    print(f"  │  Avg directed ret: {avg_return*100:>+7.3f}%               │")
    print(f"  │  Expectancy:       {expectancy*100:>+7.3f}%               │")
    print(f"  │  MFE positive:     {mfe_positive:>8.1%}                │")
    print(f"  └─────────────────────────────────────────────┘")
    
    # Split CALL vs PUT
    for side in ['CALL', 'PUT']:
        mask = dir_vals == side
        if mask.sum() == 0:
            continue
        side_ret = directed_return[mask]
        side_wins = side_ret > 0
        s_wr = side_wins.mean()
        s_avg = side_ret.mean()
        s_n = mask.sum()
        print(f"  {side:4s}: {s_n:>6,} signals | WR={s_wr:.1%} | avg={s_avg*100:+.3f}%")
    
    # Conviction tiers
    if direction_col == 'gmm_signal':
        conv_col = 'gmm_divergence'
        conv_name = 'Divergence'
    else:
        conv_col = 'xgb_conviction'
        conv_name = 'Conviction'
    
    if conv_col in valid.columns:
        print(f"\n  {conv_name} Tiers:")
        conv = valid[conv_col].values
        for q_label, lo, hi in [('Bottom 25%', 0, 25), ('25-50%', 25, 50), 
                                  ('50-75%', 50, 75), ('Top 25%', 75, 100)]:
            lo_v = np.percentile(conv, lo)
            hi_v = np.percentile(conv, hi) if hi < 100 else conv.max() + 1
            tier_mask = (conv >= lo_v) & (conv < hi_v)
            if tier_mask.sum() == 0:
                continue
            tier_ret = directed_return[tier_mask]
            tier_wr = (tier_ret > 0).mean()
            tier_avg = tier_ret.mean()
            print(f"    {q_label:12s}: {tier_mask.sum():>6,} | WR={tier_wr:.1%} | avg={tier_avg*100:+.3f}%")
    
    # Per-day aggregation
    valid['_date'] = valid['date'].dt.date
    daily = valid.groupby('_date').apply(
        lambda g: pd.Series({
            'n_signals': len(g),
            'avg_dir_return': np.where(g[direction_col] == 'CALL', 
                                       g['fwd_return'], -g['fwd_return']).mean(),
        })
    )
    
    print(f"\n  Daily Stats ({len(daily)} trading days):")
    print(f"    Avg signals/day: {daily['n_signals'].mean():.1f}")
    print(f"    Days with any signal: {(daily['n_signals'] > 0).sum()}")
    profitable_days = (daily['avg_dir_return'] > 0).sum()
    print(f"    Profitable days: {profitable_days}/{len(daily)} ({profitable_days/max(len(daily),1):.0%})")
    
    return {
        'n_signals': n_signals,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-days', type=int, default=20, help='Test period days')
    parser.add_argument('--lookahead', type=int, default=8, help='Forward candles for return (8=40min)')
    parser.add_argument('--test-only', action='store_true', help='Only use test period (not full dataset)')
    args = parser.parse_args()
    
    print(f"\n{'━'*70}")
    print(f"  BACKTEST: TEST_GMM (Regime Divergence) & TEST_XGB (Pure XGB)")
    print(f"  Test period: last {args.test_days} days | Lookahead: {args.lookahead} candles")
    print(f"{'━'*70}")
    
    # ── Load models ──
    print("\n📦 Loading XGB models...")
    xgb_models = load_xgb_models()
    
    print("📦 Loading DR detector...")
    detector = load_dr_detector()
    
    # ── Load data ──
    print("\n📊 Loading dataset (this takes ~60s)...")
    t0 = time.time()
    
    dr_config = DetectorConfig(test_days=args.test_days, val_days=args.test_days)
    combined, dr_feature_names = load_detector_dataset(dr_config)
    
    # Get test period
    combined['_date'] = combined['date'].dt.date
    all_dates = sorted(combined['_date'].unique())
    test_cutoff = all_dates[-args.test_days]
    
    if args.test_only:
        df = combined[combined['_date'] >= test_cutoff].copy()
        period_label = f"TEST ({args.test_days} days)"
    else:
        df = combined.copy()
        period_label = f"FULL ({len(all_dates)} days)"
    
    print(f"  Dataset: {len(df):,} rows, {len(all_dates)} unique dates")
    print(f"  Evaluation period: {period_label}")
    print(f"  Test cutoff: {test_cutoff}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    # ── Add forward returns ──
    print("\n📈 Computing forward returns...")
    t1 = time.time()
    df = add_forward_returns(df, lookahead=args.lookahead)
    valid_fwd = df['fwd_return'].notna().sum()
    print(f"  {valid_fwd:,}/{len(df):,} rows have valid forward returns ({time.time()-t1:.1f}s)")
    
    # ── XGB predictions ──
    print("\n🤖 Computing XGB predictions...")
    t2 = time.time()
    feature_names = xgb_models['feature_names']
    # Ensure all XGB features exist
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0.0
    df = compute_xgb_predictions(df, xgb_models)
    print(f"  XGB done ({time.time()-t2:.1f}s)")
    print(f"  P(MOVE) stats: mean={df['xgb_move_prob'].mean():.3f}, "
          f"median={df['xgb_move_prob'].median():.3f}, "
          f"P(MOVE)>0.58: {(df['xgb_move_prob']>=0.58).sum():,}")
    
    # ── DR predictions ──
    if detector:
        print("\n🔬 Computing DR predictions (both regimes)...")
        t3 = time.time()
        df = compute_dr_predictions(df, detector, dr_feature_names)
        print(f"  DR done ({time.time()-t3:.1f}s)")
        print(f"  UP score stats: mean={df['dr_up_score'].mean():.3f}, "
              f"median={df['dr_up_score'].median():.3f}")
        print(f"  DOWN score stats: mean={df['dr_down_score'].mean():.3f}, "
              f"median={df['dr_down_score'].median():.3f}")
    
    # ── Generate signals ──
    print("\n🎯 Generating strategy signals...")
    
    # TEST_XGB
    df = generate_test_xgb_signals(df)
    xgb_count = df['xgb_triggered'].sum()
    print(f"  TEST_XGB: {xgb_count:,} signals ({xgb_count/len(df)*100:.2f}%)")
    
    # TEST_GMM
    if detector:
        df = generate_test_gmm_signals(df)
        gmm_count = df['gmm_triggered'].sum()
        print(f"  TEST_GMM: {gmm_count:,} signals ({gmm_count/len(df)*100:.2f}%)")
    
    # ── Analyze ──
    xgb_results = analyze_strategy(df, 'xgb_triggered', 'xgb_signal', 'TEST_XGB')
    
    if detector:
        gmm_results = analyze_strategy(df, 'gmm_triggered', 'gmm_signal', 'TEST_GMM')
    
    # ── Overlap analysis ──
    if detector:
        both = df['xgb_triggered'] & df['gmm_triggered']
        n_both = both.sum()
        if n_both > 0:
            # Agreement: both say same direction
            agree = (df.loc[both, 'xgb_signal'] == df.loc[both, 'gmm_signal']).sum()
            print(f"\n{'='*70}")
            print(f"  OVERLAP: {n_both:,} candles where BOTH trigger")
            print(f"  Direction agreement: {agree}/{n_both} ({agree/n_both:.0%})")
            
            # Performance when both agree
            agree_mask = both & (df['xgb_signal'] == df['gmm_signal'])
            if agree_mask.sum() > 0:
                analyze_strategy(df, agree_mask.values if isinstance(agree_mask, pd.Series) else agree_mask, 
                               'xgb_signal', 'BOTH_AGREE')
    
    # ── Summary ──
    print(f"\n{'━'*70}")
    print(f"  VERDICT")
    print(f"{'━'*70}")
    
    for name, results in [('TEST_XGB', xgb_results)]:
        if results:
            wr = results['win_rate']
            exp = results['expectancy']
            verdict = '✅ EDGE' if wr > 0.52 and exp > 0 else '⚠️ MARGINAL' if wr > 0.50 else '❌ NO EDGE'
            print(f"  {name}: WR={wr:.1%}, Exp={exp*100:+.3f}% → {verdict}")
    
    if detector:
        results = gmm_results
        if results:
            wr = results['win_rate']
            exp = results['expectancy']
            verdict = '✅ EDGE' if wr > 0.52 and exp > 0 else '⚠️ MARGINAL' if wr > 0.50 else '❌ NO EDGE'
            print(f"  TEST_GMM: WR={wr:.1%}, Exp={exp*100:+.3f}% → {verdict}")
    
    print(f"\n  ⏱️ Total time: {time.time()-t0:.1f}s")
    print()


if __name__ == '__main__':
    main()
