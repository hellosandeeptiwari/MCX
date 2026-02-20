"""
Conditional Strategy Simulator — Regime-Switching
==================================================
Bypasses the 60.3% Direction model by using MARKET REGIME to decide CE vs PE.

Logic (at ~09:40):
  1. NIFTY first-30min return = market direction
  2. GMM UP_score distribution = anomaly quality indicator
  3. Decision matrix:
     - NIFTY < -0.20% AND GMM mean < 15% → PE MODE on GMM outliers
     - Otherwise → CE MODE on Gate=MOVE stocks
  4. Gate model still used as quality filter in BOTH modes
  5. Sector diversification: max 1 per sector

Key insight: The GMM distribution itself differentiates crash vs noisy days.
  - Feb 17 (UP day): mean=28.2% → "everything looks anomalous" = noise
  - Feb 19 (crash): mean=8.0% → "mostly calm, few genuine outliers" = real signal
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from ml_models.predictor import MovePredictor
from ml_models.down_risk_detector import DownRiskDetector
from ml_models.feature_engineering import compute_features, get_feature_names, get_sector_for_symbol
from ml_models.data_fetcher import get_kite_client, get_instrument_token

# Exit thresholds (same as exit_manager.py)
QUICK_PROFIT_PCT = 0.18
STOPLOSS_PCT = -0.28
TIME_STOP_CANDLES = 10
MAX_TRADES = 7

# Data directories
DATA_DIR = 'ml_models/data/candles_5min'
FOI_DIR = 'ml_models/data/futures_oi'


def load_gmm_cache(day_str):
    """Load pre-computed GMM scores from cache file."""
    cache_file = f"_gmm_cache_{day_str}.json"
    with open(cache_file, 'r') as f:
        return json.load(f)


def analyze_gmm_distribution(gmm_cache):
    """Analyze GMM UP_score distribution for regime detection."""
    scores = [v['up_score'] for v in gmm_cache.values()]
    s = np.array(scores)
    mean_s = float(np.mean(s))
    std_s = float(np.std(s))
    threshold_2sd = mean_s + 2 * std_s
    outlier_syms = {k: v['up_score'] for k, v in gmm_cache.items() if v['up_score'] > threshold_2sd}
    return {
        'mean': mean_s,
        'std': std_s,
        'threshold_2sd': threshold_2sd,
        'n_outliers': len(outlier_syms),
        'outlier_syms': outlier_syms,
    }


def get_nifty_pre_entry_return(nifty_df, sim_date):
    """Get NIFTY return in first 5 candles (09:15-09:40) from parquet data."""
    day_data = nifty_df[nifty_df['date'].dt.date == sim_date].reset_index(drop=True)
    if len(day_data) < 5:
        return 0.0
    open_price = day_data.iloc[0]['open']
    pre_entry_close = day_data.iloc[4]['close']
    return (pre_entry_close - open_price) / open_price * 100


def decide_regime(nifty_return_pct, gmm_stats):
    """Decide trading regime based on NIFTY + GMM distribution."""
    nifty_falling = nifty_return_pct < -0.20
    gmm_clean = gmm_stats['mean'] < 0.15
    has_outliers = gmm_stats['n_outliers'] >= 3

    if nifty_falling and gmm_clean and has_outliers:
        return 'PE', f"NIFTY {nifty_return_pct:+.2f}%, GMM mean={gmm_stats['mean']*100:.1f}%, {gmm_stats['n_outliers']} outliers"
    else:
        return 'CE', f"NIFTY {nifty_return_pct:+.2f}%, normal/noisy regime"


def find_atm_option(kite, nfo_df, symbol, spot_price, opt_type, sim_date):
    """Find ATM option and get its 5min candles."""
    STRIKE_INTERVALS = {
        'BAJFINANCE': 100, 'RELIANCE': 20, 'TCS': 50, 'HDFCBANK': 20,
        'INFY': 20, 'ICICIBANK': 20, 'SBIN': 10, 'TATAMOTORS': 10,
        'ITC': 5, 'HINDUNILVR': 25, 'LT': 25, 'MARUTI': 100,
        'AXISBANK': 25, 'KOTAKBANK': 25, 'TITAN': 25, 'BHARTIARTL': 20,
        'M&M': 25, 'WIPRO': 10, 'ADANIENT': 25, 'ADANIPORTS': 20,
        'NESTLEIND': 100, 'TATASTEEL': 5, 'JSWSTEEL': 10, 'POWERGRID': 5,
        'NTPC': 5, 'SUNPHARMA': 25, 'HCLTECH': 25, 'CIPLA': 25,
        'DRREDDY': 50, 'TECHM': 25, 'ONGC': 5, 'COALINDIA': 5,
        'GRASIM': 25, 'BPCL': 10, 'INDUSINDBK': 25, 'EICHERMOT': 50,
        'SHRIRAMFIN': 25, 'DIVISLAB': 25, 'APOLLOHOSP': 50,
        'TRENT': 25, 'HEROMOTOCO': 50, 'BEL': 5, 'IRCTC': 25,
        'M&MFIN': 5, 'BANKBARODA': 5, 'PNB': 2.5,
    }
    interval = STRIKE_INTERVALS.get(symbol, 50)
    atm_strike = round(spot_price / interval) * interval

    opts = nfo_df[(nfo_df['tradingsymbol'].str.startswith(symbol)) &
                  (nfo_df['instrument_type'] == opt_type) &
                  (nfo_df['strike'] == atm_strike) &
                  (pd.to_datetime(nfo_df['expiry']) >= pd.Timestamp(sim_date))]
    if opts.empty:
        return None, None

    opts = opts.sort_values('expiry')
    token = int(opts.iloc[0]['instrument_token'])
    ts = opts.iloc[0]['tradingsymbol']

    try:
        candles = kite.historical_data(token, sim_date, sim_date, '5minute')
        if candles:
            return ts, candles
    except:
        pass
    return ts, None


LOT_SIZES = {
    'RELIANCE': 250, 'TCS': 175, 'HDFCBANK': 550, 'INFY': 400,
    'ICICIBANK': 700, 'SBIN': 750, 'TATAMOTORS': 700, 'ITC': 1600,
    'HINDUNILVR': 300, 'LT': 150, 'MARUTI': 100, 'AXISBANK': 600,
    'KOTAKBANK': 400, 'TITAN': 375, 'BHARTIARTL': 475, 'M&M': 350,
    'WIPRO': 1500, 'ADANIENT': 250, 'ADANIPORTS': 500, 'NESTLEIND': 50,
    'TATASTEEL': 5600, 'JSWSTEEL': 900, 'POWERGRID': 2700, 'NTPC': 2100,
    'SUNPHARMA': 350, 'HCLTECH': 350, 'CIPLA': 650, 'DRREDDY': 125,
    'TECHM': 600, 'ONGC': 3250, 'COALINDIA': 2100, 'GRASIM': 250,
    'BPCL': 1800, 'INDUSINDBK': 500, 'EICHERMOT': 175, 'SHRIRAMFIN': 200,
    'DIVISLAB': 100, 'APOLLOHOSP': 125, 'TRENT': 125, 'HEROMOTOCO': 150,
    'BEL': 2700, 'IRCTC': 500, 'BANKBARODA': 2700, 'PNB': 4000,
    'BAJFINANCE': 125, 'BAJAJFINSV': 50, 'ASIANPAINT': 300, 'ULTRACEMCO': 100,
}


def simulate_option_trade(candles, entry_candle_idx=5):
    """Simulate a single option trade. Entry at candle 5 (09:40)."""
    if not candles or entry_candle_idx >= len(candles):
        return None

    entry_price = candles[entry_candle_idx]['close']
    if entry_price <= 0:
        return None

    target_price = entry_price * (1 + QUICK_PROFIT_PCT)
    sl_price = entry_price * (1 + STOPLOSS_PCT)

    for i in range(entry_candle_idx + 1, min(entry_candle_idx + TIME_STOP_CANDLES + 1, len(candles))):
        c = candles[i]
        if c['high'] >= target_price:
            return {'entry': entry_price, 'exit': target_price,
                    'return_pct': QUICK_PROFIT_PCT * 100, 'reason': '+18% TARGET',
                    'candles': i - entry_candle_idx}
        if c['low'] <= sl_price:
            return {'entry': entry_price, 'exit': sl_price,
                    'return_pct': STOPLOSS_PCT * 100, 'reason': '-28% STOPLOSS',
                    'candles': i - entry_candle_idx}

    last_idx = min(entry_candle_idx + TIME_STOP_CANDLES, len(candles) - 1)
    exit_price = candles[last_idx]['close']
    ret = (exit_price - entry_price) / entry_price * 100
    return {'entry': entry_price, 'exit': exit_price,
            'return_pct': ret, 'reason': 'TIMEOUT',
            'candles': last_idx - entry_candle_idx}


def run_day(kite, predictor, nfo_df, sim_date):
    """Run conditional strategy for one day."""
    day_str = str(sim_date)
    print(f"\n{'='*80}")
    print(f"  CONDITIONAL STRATEGY — {day_str}")
    print(f"{'='*80}")

    nifty_df = pd.read_parquet(os.path.join(DATA_DIR, 'NIFTY50.parquet'))

    # Step 1: NIFTY pre-entry return
    nifty_ret = get_nifty_pre_entry_return(nifty_df, sim_date)
    print(f"\n[1] NIFTY pre-entry return (5 candles): {nifty_ret:+.2f}%")

    # Step 2: GMM distribution
    gmm_cache = load_gmm_cache(day_str)
    gmm_stats = analyze_gmm_distribution(gmm_cache)
    print(f"[2] GMM distribution: mean={gmm_stats['mean']*100:.1f}%, "
          f"std={gmm_stats['std']*100:.1f}%, outliers(>2SD)={gmm_stats['n_outliers']}")

    # Step 3: Regime decision
    regime, reason = decide_regime(nifty_ret, gmm_stats)
    print(f"[3] REGIME: {regime} — {reason}")

    # Step 4: Get candidates
    parquets = [f for f in os.listdir(DATA_DIR)
                if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
    all_symbols = sorted([f.replace('.parquet', '') for f in parquets])

    candidates = []

    if regime == 'PE':
        # PE MODE: Only consider GMM outliers
        outlier_syms = sorted(gmm_stats['outlier_syms'].keys(),
                              key=lambda s: gmm_stats['outlier_syms'][s], reverse=True)
        print(f"\n[4] PE MODE — {len(outlier_syms)} GMM outliers to evaluate with Gate:")
        for sym in outlier_syms:
            print(f"    {sym:15s}  UP_score={gmm_stats['outlier_syms'][sym]*100:.1f}%")
        
        for sym in outlier_syms:
            try:
                # Load stock candles
                df = pd.read_parquet(os.path.join(DATA_DIR, sym + '.parquet'))
                df = df[df['date'].dt.date <= sim_date]

                # Load supplementary data
                foi_path = os.path.join(FOI_DIR, sym + '_futures_oi.parquet')
                foi_df = pd.read_parquet(foi_path) if os.path.exists(foi_path) else None
                sector = get_sector_for_symbol(sym)
                sector_key = 'SECTOR_' + sector.upper().replace(' ', '_') if sector and sector != 'Other' else None
                sector_df = None
                if sector_key:
                    sp = os.path.join(DATA_DIR, sector_key + '.parquet')
                    if os.path.exists(sp):
                        sector_df = pd.read_parquet(sp)

                # Get Gate prediction — we just need P(MOVE) to confirm stock will move
                pred = predictor.get_titan_signals(df, futures_oi_df=foi_df,
                                                   nifty_5min_df=nifty_df, sector_5min_df=sector_df)
                if not pred:
                    print(f"    {sym:15s}  — no prediction, skipping")
                    continue

                p_move = pred.get('ml_move_prob', 0)
                signal = pred.get('ml_signal', 'UNKNOWN')
                
                # Day's spot price (for ATM strike lookup)
                day_data = df[df['date'].dt.date == sim_date]
                if day_data.empty or len(day_data) < 6:
                    continue
                spot = day_data.iloc[5]['close']  # at entry candle

                gate_pass = p_move >= 0.50

                print(f"    {sym:15s}  P(MOVE)={p_move:.1%}  signal={signal:8s}  "
                      f"Gate={'PASS' if gate_pass else 'FAIL'}")

                if gate_pass:
                    candidates.append({
                        'symbol': sym,
                        'direction': 'PE',
                        'up_score': gmm_stats['outlier_syms'][sym],
                        'p_move': p_move,
                        'spot': spot,
                        'sector': sector or 'OTHER',
                        'sort_key': gmm_stats['outlier_syms'][sym],
                    })
            except Exception as e:
                continue

    else:
        # CE MODE: Gate=MOVE stocks among all 207, sorted by P(MOVE)
        print(f"\n[4] CE MODE — scanning {len(all_symbols)} stocks for Gate=MOVE...")
        for sym in all_symbols:
            try:
                df = pd.read_parquet(os.path.join(DATA_DIR, sym + '.parquet'))
                df = df[df['date'].dt.date <= sim_date]

                foi_path = os.path.join(FOI_DIR, sym + '_futures_oi.parquet')
                foi_df = pd.read_parquet(foi_path) if os.path.exists(foi_path) else None
                sector = get_sector_for_symbol(sym)
                sector_key = 'SECTOR_' + sector.upper().replace(' ', '_') if sector and sector != 'Other' else None
                sector_df = None
                if sector_key:
                    sp = os.path.join(DATA_DIR, sector_key + '.parquet')
                    if os.path.exists(sp):
                        sector_df = pd.read_parquet(sp)

                pred = predictor.get_titan_signals(df, futures_oi_df=foi_df,
                                                   nifty_5min_df=nifty_df, sector_5min_df=sector_df)
                if not pred:
                    continue

                p_move = pred.get('ml_move_prob', 0)
                if p_move < 0.50:
                    continue

                day_data = df[df['date'].dt.date == sim_date]
                if day_data.empty or len(day_data) < 6:
                    continue
                spot = day_data.iloc[5]['close']

                # In CE mode, prefer LOW up_score (less crash risk)
                up_score = gmm_cache.get(sym, {}).get('up_score', 0)

                candidates.append({
                    'symbol': sym,
                    'direction': 'CE',
                    'up_score': up_score,
                    'p_move': p_move,
                    'spot': spot,
                    'sector': sector or 'OTHER',
                    'sort_key': p_move,  # higher P(MOVE) first
                })
            except Exception:
                continue
        print(f"    -> {len(candidates)} stocks passed Gate")

    print(f"\n[5] Candidates: {len(candidates)}")

    if not candidates:
        print("  No candidates. Skipping day.")
        return {'regime': regime, 'trades': [], 'total_pnl': 0, 'wins': 0, 'losses': 0}

    # Step 5: Sector diversification + top N
    candidates.sort(key=lambda x: x['sort_key'], reverse=True)
    selected = []
    seen_sectors = set()
    for c in candidates:
        if c['sector'] not in seen_sectors:
            selected.append(c)
            seen_sectors.add(c['sector'])
            if len(selected) >= MAX_TRADES:
                break
    # Backfill if needed
    if len(selected) < MAX_TRADES:
        for c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= MAX_TRADES:
                    break

    print(f"\n[6] Selected {len(selected)} trades:")
    for s in selected:
        print(f"    {s['symbol']:15s} {s['direction']}  UP_score={s['up_score']*100:.1f}%  "
              f"P(MOVE)={s['p_move']:.1%}  sector={s['sector']}")

    # Step 6: Simulate option trades
    print(f"\n{'─'*80}")
    print(f"  TRADE RESULTS")
    print(f"{'─'*80}")

    total_pnl = 0
    wins = 0
    losses = 0
    trades = []

    for s in selected:
        ts, candles = find_atm_option(kite, nfo_df, s['symbol'], s['spot'], s['direction'], sim_date)
        if not candles:
            print(f"  {s['symbol']:15s} {s['direction']}  NO OPTION DATA")
            continue

        result = simulate_option_trade(candles, entry_candle_idx=5)
        if not result:
            print(f"  {s['symbol']:15s} {s['direction']}  SIMULATION FAILED")
            continue

        lot = LOT_SIZES.get(s['symbol'], 500)
        pnl = (result['exit'] - result['entry']) * lot
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        print(f"  {s['symbol']:15s} {s['direction']}  entry={result['entry']:.1f}  exit={result['exit']:.1f}  "
              f"ret={result['return_pct']:+.1f}%  pnl=Rs {pnl:+,.0f}  "
              f"({result['reason']}, {result['candles']}c)  "
              f"[{'WIN' if pnl > 0 else 'LOSS'}]")

        trades.append({'symbol': s['symbol'], 'direction': s['direction'],
                       'pnl': pnl, 'return_pct': result['return_pct'],
                       'reason': result['reason']})

    print(f"\n  SUMMARY: {wins}W/{losses}L  Total PnL = Rs {total_pnl:+,.0f}")
    return {'regime': regime, 'reason': reason, 'nifty_ret': nifty_ret,
            'gmm_mean': gmm_stats['mean'], 'trades': trades,
            'total_pnl': total_pnl, 'wins': wins, 'losses': losses}


def main():
    # Load models
    predictor = MovePredictor()
    if not predictor.ready:
        print('ERROR: MovePredictor not ready')
        sys.exit(1)
    print(f'Models loaded: {predictor.model_type}')

    kite = get_kite_client()
    nfo_df = pd.DataFrame(kite.instruments('NFO'))
    print(f'{len(nfo_df)} NFO instruments loaded')

    sim_dates = [date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19)]
    results = {}
    grand_pnl = 0
    grand_wins = 0
    grand_losses = 0

    for d in sim_dates:
        r = run_day(kite, predictor, nfo_df, d)
        results[str(d)] = r
        grand_pnl += r['total_pnl']
        grand_wins += r['wins']
        grand_losses += r['losses']

    print(f"\n{'='*80}")
    print(f"  GRAND TOTAL — CONDITIONAL (REGIME-SWITCHING) STRATEGY")
    print(f"{'='*80}")
    for d, r in results.items():
        print(f"  {d}: [{r['regime']:4s}] {r['wins']}W/{r['losses']}L  Rs {r['total_pnl']:+,.0f}")
    print(f"\n  TOTAL: {grand_wins}W/{grand_losses}L  Rs {grand_pnl:+,.0f}")
    print(f"  Capital: Rs 500,000  ROI: {grand_pnl/500000*100:+.2f}%")

    print(f"\n{'='*80}")
    print(f"  STRATEGY COMPARISON (all 3 days: Feb 17-19)")
    print(f"{'='*80}")
    print(f"  Current (CE bias, Feb 19 only):       Rs -24,524 (2W/5L)")
    print(f"  GMM forced PE (all days):             Rs -11,820 (13W/17L)")
    print(f"  GMM+Gate+Dir (v2, Feb 19 only):       Rs -14,968 (3W/4L)")
    print(f"  CONDITIONAL (regime-switch):           Rs {grand_pnl:+,.0f} ({grand_wins}W/{grand_losses}L)")


if __name__ == '__main__':
    main()
