"""
Simulate Score-Graduated Soft Gating on last 3 days (Feb 18-20, 2026).

Loads real 5-min candle data for Tier-1 + Tier-2 stocks, runs:
  1. Feature engineering  
  2. XGB Gate+Direction prediction → signal (UP/DOWN/FLAT)
  3. Down-Risk GMM scoring (UP or DOWN regime)
  4. Graduated soft scoring (NEW) vs old binary scoring

Outputs a side-by-side comparison showing how each stock's score changes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

# ── Config ──
DATA_DIR = Path(__file__).parent / "ml_models" / "data" / "candles_5min"
SIM_DATES = [date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20)]

# Tier-1 + Tier-2 symbols (without NSE: prefix)
SYMBOLS = [
    'SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK',
    'BAJFINANCE', 'RELIANCE', 'BHARTIARTL', 'INFY', 'TCS',
    'TATASTEEL', 'JSWSTEEL', 'JINDALSTEL', 'HINDALCO', 'LT',
    'MARUTI', 'TITAN', 'SUNPHARMA', 'ONGC', 'NTPC',
    'ITC', 'TATAMOTORS', 'CIPLA',
]

# Old config (binary)
OLD_SAFE_BONUS = 5
OLD_RISK_PENALTY = 5

# New config (graduated)
HIGH_THRESH = 0.7
HIGH_PENALTY = 15
MID_PENALTY = 8
CLEAN_THRESH = 0.3
CLEAN_BOOST = 8


def old_scoring(dr_flag, dr_score):
    """Binary: flagged → −5, clean → +5"""
    if dr_flag:
        return -OLD_RISK_PENALTY, "PENALTY"
    else:
        return +OLD_SAFE_BONUS, "BONUS"


def new_scoring(dr_flag, dr_score):
    """Graduated: HIGH → −15, CAUTION → −8, CLEAN → +8, NEUTRAL → 0"""
    if dr_score > HIGH_THRESH:
        return -HIGH_PENALTY, "HIGH_RISK"
    elif dr_flag:
        return -MID_PENALTY, "CAUTION"
    elif dr_score < CLEAN_THRESH:
        return +CLEAN_BOOST, "CLEAN"
    else:
        return 0, "NEUTRAL"


def main():
    print("=" * 80)
    print("  GRADUATED SCORING SIMULATION — Last 3 Trading Days")
    print("  Comparing OLD (binary ±5) vs NEW (graduated −15/−8/0/+8)")
    print("=" * 80)

    # Load ML components
    from agentic_trader.ml_models.feature_engineering import compute_features, get_feature_names
    from agentic_trader.ml_models.predictor import MovePredictor
    from agentic_trader.ml_models.down_risk_detector import DownRiskDetector

    feature_names = get_feature_names()

    # Load XGB predictor
    predictor = MovePredictor()
    if not predictor.ready:
        print("ERROR: XGB models not loaded. Exiting.")
        return

    # Load GMM detector
    detector = DownRiskDetector()
    if not detector.load():
        print("ERROR: GMM models not loaded. Exiting.")
        return

    print(f"\n  Models loaded. Features: {len(feature_names)}")
    print(f"  Symbols: {len(SYMBOLS)}")
    print(f"  Dates: {', '.join(str(d) for d in SIM_DATES)}")

    # ── Process each date ──
    all_results = []

    for sim_date in SIM_DATES:
        print(f"\n{'─' * 80}")
        print(f"  DATE: {sim_date}")
        print(f"{'─' * 80}")

        date_results = []

        for sym in SYMBOLS:
            parquet_path = DATA_DIR / f"{sym}.parquet"
            if not parquet_path.exists():
                continue

            try:
                df = pd.read_parquet(parquet_path)
                df['date'] = pd.to_datetime(df['date'])

                # Filter: keep data up to sim_date (simulate not seeing future)
                end_of_day = pd.Timestamp(sim_date) + pd.Timedelta(hours=23)
                df_available = df[df['date'] <= end_of_day].copy()

                if len(df_available) < 100:
                    continue

                # Get candles for this specific day (for forward-looking P&L check)
                day_candles = df_available[df_available['date'].dt.date == sim_date]
                if len(day_candles) < 10:
                    continue

                # Compute features
                featured = compute_features(df_available, symbol=sym)
                if featured.empty:
                    continue

                # Get candles for this day
                featured_day = featured[featured['date'].dt.date == sim_date]
                if len(featured_day) < 5:
                    continue

                # Sample 3 candles per day: 10:00, 12:00, 14:00 approx
                sample_indices = [
                    min(3, len(featured_day) - 1),   # ~10:30
                    min(12, len(featured_day) - 1),   # ~12:15
                    min(20, len(featured_day) - 1),   # ~14:00
                ]
                sample_indices = list(set(sample_indices))  # dedup

                for si in sample_indices:
                    row = featured_day.iloc[si]
                    candle_time = row['date']

                    # Extract features
                    missing = [f for f in feature_names if f not in featured_day.columns]
                    for mf in missing:
                        featured_day = featured_day.copy()
                        featured_day[mf] = 0.0

                    X = featured_day.iloc[si:si+1][feature_names].values
                    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

                    # XGB prediction
                    pred = predictor.predict(df_available[df_available['date'] <= candle_time].tail(200))
                    if not pred:
                        continue

                    prob_up = pred.get('ml_prob_up', 0.33)
                    prob_down = pred.get('ml_prob_down', 0.33)
                    signal = 'UP' if prob_up > prob_down and prob_up > 0.4 else ('DOWN' if prob_down > prob_up and prob_down > 0.4 else 'FLAT')

                    # GMM scoring
                    regime = 'DOWN' if signal == 'DOWN' else 'UP'
                    gmm_result = detector.predict_single(X, regime)

                    dr_flag = bool(gmm_result['down_risk_flag'][0])
                    dr_score = float(gmm_result['anomaly_score'][0])
                    dr_bucket = str(gmm_result['confidence_bucket'][0])

                    # Forward P&L: what actually happened in next 8 candles?
                    remaining = featured_day.iloc[si:]
                    if len(remaining) > 1:
                        close_now = row['close']
                        future_closes = remaining['close'].values[1:min(9, len(remaining))]
                        max_up = ((future_closes.max() - close_now) / close_now * 100) if len(future_closes) > 0 else 0
                        max_down = ((close_now - future_closes.min()) / close_now * 100) if len(future_closes) > 0 else 0
                        end_pnl = ((future_closes[-1] - close_now) / close_now * 100) if len(future_closes) > 0 else 0
                    else:
                        max_up = max_down = end_pnl = 0

                    # Compare old vs new scoring
                    old_adj, old_label = old_scoring(dr_flag, dr_score)
                    new_adj, new_label = new_scoring(dr_flag, dr_score)

                    result = {
                        'date': sim_date,
                        'time': candle_time.strftime('%H:%M'),
                        'symbol': sym,
                        'close': row['close'],
                        'signal': signal,
                        'regime': regime,
                        'dr_score': dr_score,
                        'dr_flag': dr_flag,
                        'dr_bucket': dr_bucket,
                        'old_adj': old_adj,
                        'old_label': old_label,
                        'new_adj': new_adj,
                        'new_label': new_label,
                        'diff': new_adj - old_adj,
                        'max_up_pct': max_up,
                        'max_down_pct': max_down,
                        'end_pnl_pct': end_pnl,
                    }
                    date_results.append(result)

            except Exception as e:
                pass  # Skip problematic symbols

        all_results.extend(date_results)

        # ── Print day summary ──
        if date_results:
            df_day = pd.DataFrame(date_results)

            # Group by new_label
            for label in ['HIGH_RISK', 'CAUTION', 'CLEAN', 'NEUTRAL']:
                subset = df_day[df_day['new_label'] == label]
                if len(subset) == 0:
                    continue
                icon = {'HIGH_RISK': '🔴', 'CAUTION': '🟡', 'CLEAN': '🟢', 'NEUTRAL': '⚪'}.get(label, '')
                avg_score = subset['dr_score'].mean()
                avg_up = subset['max_up_pct'].mean()
                avg_down = subset['max_down_pct'].mean()
                avg_end = subset['end_pnl_pct'].mean()
                print(f"\n  {icon} {label}: {len(subset)} candles (avg score: {avg_score:.3f})")
                print(f"     Old adj: {subset['old_adj'].iloc[0]:+d}  →  New adj: {subset['new_adj'].iloc[0]:+d}")
                print(f"     Actual: max_up {avg_up:.2f}% | max_down {avg_down:.2f}% | end {avg_end:+.2f}%")

                # Show top 3 examples
                for _, r in subset.head(3).iterrows():
                    direction = 'CE' if r['signal'] in ('UP', 'FLAT') else 'PE'
                    print(f"       {r['symbol']:12s} {r['time']} {direction} score={r['dr_score']:.3f} → end={r['end_pnl_pct']:+.2f}%")

    # ── Final summary across all days ──
    if all_results:
        df_all = pd.DataFrame(all_results)
        print(f"\n{'=' * 80}")
        print(f"  OVERALL SUMMARY (3 days, {len(df_all)} candle-signals)")
        print(f"{'=' * 80}")

        # Key question: Does graduated scoring correctly differentiate?
        print("\n  OLD vs NEW scoring — who wins?")
        print(f"  {'Category':<14s} {'Count':>5s} {'Old adj':>8s} {'New adj':>8s} {'Actual end%':>12s} {'Right?':>8s}")
        print(f"  {'─' * 55}")

        for label in ['HIGH_RISK', 'CAUTION', 'NEUTRAL', 'CLEAN']:
            subset = df_all[df_all['new_label'] == label]
            if len(subset) == 0:
                continue
            old_a = subset['old_adj'].mean()
            new_a = subset['new_adj'].mean()
            actual = subset['end_pnl_pct'].mean()

            # For CE (UP signal): positive end% = good (CE made money)
            # For PE (DOWN signal): negative end% = good (PE made money)
            # Simplified: is the scoring direction aligned with outcome?
            ce_subset = subset[subset['signal'].isin(['UP', 'FLAT'])]
            pe_subset = subset[subset['signal'] == 'DOWN']

            ce_correct = len(ce_subset[ce_subset['end_pnl_pct'] > 0]) if len(ce_subset) > 0 else 0
            pe_correct = len(pe_subset[pe_subset['end_pnl_pct'] < 0]) if len(pe_subset) > 0 else 0
            total_correct = ce_correct + pe_correct
            total = len(subset)
            win_rate = total_correct / total * 100 if total > 0 else 0

            print(f"  {label:<14s} {len(subset):>5d} {old_a:>+8.1f} {new_a:>+8.1f} {actual:>+12.3f} {win_rate:>7.0f}%")

        # Score distribution
        print(f"\n  GMM Score Distribution:")
        for bucket, lo, hi in [('Very Low', 0, 0.2), ('Low', 0.2, 0.3), ('Mid', 0.3, 0.5), 
                                ('High', 0.5, 0.7), ('Very High', 0.7, 1.01)]:
            subset = df_all[(df_all['dr_score'] >= lo) & (df_all['dr_score'] < hi)]
            if len(subset) == 0:
                continue
            avg_end = subset['end_pnl_pct'].mean()
            avg_down = subset['max_down_pct'].mean()
            print(f"    {bucket:<10s} [{lo:.1f}-{hi:.1f}): {len(subset):>4d} signals | end={avg_end:+.3f}% | max_down={avg_down:.3f}%")

        # CE vs PE breakdown
        print(f"\n  Directional Breakdown:")
        for direction, sig_list in [('CE (BUY)', ['UP', 'FLAT']), ('PE (SELL)', ['DOWN'])]:
            subset = df_all[df_all['signal'].isin(sig_list)]
            if len(subset) == 0:
                continue
            print(f"\n    {direction}: {len(subset)} signals")
            for label in ['HIGH_RISK', 'CAUTION', 'NEUTRAL', 'CLEAN']:
                sub2 = subset[subset['new_label'] == label]
                if len(sub2) == 0:
                    continue
                avg_end = sub2['end_pnl_pct'].mean()
                avg_down = sub2['max_down_pct'].mean()
                avg_up = sub2['max_up_pct'].mean()
                print(f"      {label:<12s}: {len(sub2):>3d} | end={avg_end:+.3f}% | maxup={avg_up:.3f}% | maxdn={avg_down:.3f}%")

    print(f"\n{'=' * 80}")
    print(f"  Simulation complete.")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
