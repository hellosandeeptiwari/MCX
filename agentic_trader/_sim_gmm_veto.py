"""
GMM Veto/Boost Simulation — Layered on top of Titan's existing pipeline
========================================================================

User's strategy:
  1. Titan scans stocks, scores them, identifies direction (CE or PE)
  2. For each stock, compute where its GMM dr_score sits in the DAY's distribution
  3. If stock is a GMM OUTLIER (>mean+2SD = "high std deviation"):
     - Titan says PE → BOOST lots (GMM confirms abnormal movement = crash risk)
     - Titan says CE → HARD BLOCK (don't buy — abnormal pattern, CE will die)
  4. If stock is in LOWER bracket (normal GMM, not outlier):
     - Titan says CE → ALLOW (normal movement, CE is fine)
     - Titan says PE → BLOCK (no abnormal signal to justify PE)

Result: GMM acts as a VETO/BOOST filter, not as a direction predictor.
Direction comes from Titan (XGB Gate+Direction), GMM just validates.

GMM HIGH + PE → BOOST (2x lots)
GMM HIGH + CE → HARD BLOCK
GMM LOW  + CE → ALLOW (normal lots)
GMM LOW  + PE → BLOCK
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date

from ml_models.predictor import MovePredictor
from ml_models.feature_engineering import get_sector_for_symbol
from ml_models.data_fetcher import get_kite_client, get_instrument_token

# Directories
DATA_DIR = 'ml_models/data/candles_5min'
FOI_DIR = 'ml_models/data/futures_oi'

# Exit rules (same as Titan)
QUICK_PROFIT_PCT = 0.18
STOPLOSS_PCT = -0.28
TIME_STOP_CANDLES = 10
MAX_TRADES = 7

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


def load_gmm_cache(day_str):
    with open(f"_gmm_cache_{day_str}.json") as f:
        return json.load(f)


def gmm_distribution(gmm_cache):
    """Compute mean/std of UP_scores across all stocks for today."""
    scores = np.array([v['up_score'] for v in gmm_cache.values()])
    return float(np.mean(scores)), float(np.std(scores))


def gmm_veto_boost(sym, direction, dr_score, gmm_mean, gmm_std):
    """
    Apply the veto/boost rule.
    
    Returns: (action, lot_multiplier, reason)
      action: 'ALLOW', 'BOOST', 'BLOCK'
      lot_multiplier: 1 for normal, 2 for boosted
    """
    threshold = gmm_mean + 2 * gmm_std
    is_outlier = dr_score > threshold
    
    if is_outlier:
        if direction == 'SELL':  # PE
            return 'BOOST', 2, f"GMM HIGH ({dr_score:.3f} > {threshold:.3f}) + PE = BOOST 2x"
        else:  # CE/BUY
            return 'BLOCK', 0, f"GMM HIGH ({dr_score:.3f} > {threshold:.3f}) + CE = HARD BLOCK"
    else:
        if direction == 'BUY':  # CE
            return 'ALLOW', 1, f"GMM LOW ({dr_score:.3f} < {threshold:.3f}) + CE = OK"
        else:  # PE/SELL
            return 'BLOCK', 0, f"GMM LOW ({dr_score:.3f} < {threshold:.3f}) + PE = NO SIGNAL"


def find_atm_option(kite, nfo_df, symbol, spot_price, opt_type, sim_date):
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
    try:
        candles = kite.historical_data(token, sim_date, sim_date, '5minute')
        if candles:
            return opts.iloc[0]['tradingsymbol'], candles
    except:
        pass
    return None, None


def simulate_trade(candles, entry_idx=5):
    if not candles or entry_idx >= len(candles):
        return None
    entry = candles[entry_idx]['close']
    if entry <= 0:
        return None
    target = entry * (1 + QUICK_PROFIT_PCT)
    sl = entry * (1 + STOPLOSS_PCT)
    for i in range(entry_idx + 1, min(entry_idx + TIME_STOP_CANDLES + 1, len(candles))):
        if candles[i]['high'] >= target:
            return {'entry': entry, 'exit': target, 'ret': QUICK_PROFIT_PCT * 100,
                    'reason': '+18% TARGET', 'candles': i - entry_idx}
        if candles[i]['low'] <= sl:
            return {'entry': entry, 'exit': sl, 'ret': STOPLOSS_PCT * 100,
                    'reason': '-28% STOPLOSS', 'candles': i - entry_idx}
    last = min(entry_idx + TIME_STOP_CANDLES, len(candles) - 1)
    exit_p = candles[last]['close']
    return {'entry': entry, 'exit': exit_p, 'ret': (exit_p - entry) / entry * 100,
            'reason': 'TIMEOUT', 'candles': last - entry_idx}


def run_day(kite, predictor, nfo_df, sim_date):
    day_str = str(sim_date)
    print(f"\n{'='*90}")
    print(f"  GMM VETO/BOOST — {day_str}")
    print(f"{'='*90}")

    # Load GMM cache + compute distribution
    gmm_cache = load_gmm_cache(day_str)
    gmm_mean, gmm_std = gmm_distribution(gmm_cache)
    threshold_2sd = gmm_mean + 2 * gmm_std
    outlier_count = sum(1 for v in gmm_cache.values() if v['up_score'] > threshold_2sd)
    print(f"\n  GMM distribution: mean={gmm_mean*100:.1f}%  std={gmm_std*100:.1f}%  "
          f"outlier threshold={threshold_2sd*100:.1f}%  outliers={outlier_count}")

    # Load NIFTY data
    nifty_df = pd.read_parquet(os.path.join(DATA_DIR, 'NIFTY50.parquet'))

    # Step 1: Run Titan's normal pipeline on ALL stocks → get predictions + scores
    parquets = [f for f in os.listdir(DATA_DIR)
                if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
    all_symbols = sorted([f.replace('.parquet', '') for f in parquets])

    print(f"  Scanning {len(all_symbols)} stocks through Titan pipeline...")

    titan_candidates = []
    blocked_by_gmm = []
    
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

            # Get full Titan prediction (Gate + Direction + Down-Risk)
            pred = predictor.get_titan_signals(df, futures_oi_df=foi_df,
                                               nifty_5min_df=nifty_df, sector_5min_df=sector_df)
            if not pred:
                continue

            ml_signal = pred.get('ml_signal', 'UNKNOWN')
            p_move = pred.get('ml_move_prob', 0)
            ml_conf = pred.get('ml_confidence', 0)
            prob_up = pred.get('ml_prob_up', 0)
            prob_down = pred.get('ml_prob_down', 0)

            # Gate filter: need MOVE signal
            if p_move < 0.50:
                continue

            # Direction from Titan's XGB
            if ml_signal == 'UP':
                direction = 'BUY'
                opt_type = 'CE'
                dir_prob = prob_up
            elif ml_signal == 'DOWN':
                direction = 'SELL'
                opt_type = 'PE'
                dir_prob = prob_down
            else:
                continue  # Skip FLAT/UNKNOWN

            # Get GMM score for this stock
            gmm_data = gmm_cache.get(sym, {})
            dr_score = gmm_data.get('up_score', 0)

            # ╔═══════════════════════════════════════════════╗
            # ║  GMM VETO/BOOST FILTER — the KEY innovation  ║
            # ╚═══════════════════════════════════════════════╝
            action, lot_mult, gmm_reason = gmm_veto_boost(sym, direction, dr_score, gmm_mean, gmm_std)

            # Day's spot price
            day_data = df[df['date'].dt.date == sim_date]
            if day_data.empty or len(day_data) < 6:
                continue
            spot = day_data.iloc[5]['close']

            if action == 'BLOCK':
                blocked_by_gmm.append({
                    'sym': sym, 'direction': direction, 'opt_type': opt_type,
                    'dr_score': dr_score, 'reason': gmm_reason,
                    'p_move': p_move, 'dir_prob': dir_prob,
                })
                continue

            titan_candidates.append({
                'sym': sym,
                'direction': direction,
                'opt_type': opt_type,
                'p_move': p_move,
                'dir_prob': dir_prob,
                'ml_conf': ml_conf,
                'dr_score': dr_score,
                'action': action,
                'lot_mult': lot_mult,
                'gmm_reason': gmm_reason,
                'spot': spot,
                'sector': sector or 'OTHER',
                # Smart score (same formula as Titan, direction-aware)
                'smart_score': ml_conf * dir_prob * 40 + (1.0 - dr_score if direction == 'BUY' else dr_score) * 20 + p_move * 5,
            })
        except Exception:
            continue

    # Print blocked trades
    if blocked_by_gmm:
        print(f"\n  GMM BLOCKED {len(blocked_by_gmm)} trades:")
        for b in sorted(blocked_by_gmm, key=lambda x: x['dr_score'], reverse=True):
            print(f"    BLOCKED: {b['sym']:15s} {b['opt_type']}  dr={b['dr_score']:.3f}  "
                  f"P(MOVE)={b['p_move']:.1%}  dir={b['dir_prob']:.2f}  — {b['reason']}")

    print(f"\n  After GMM filter: {len(titan_candidates)} candidates ({len(blocked_by_gmm)} blocked)")

    if not titan_candidates:
        print("  No candidates survived. Skipping day.")
        return {'trades': [], 'total_pnl': 0, 'wins': 0, 'losses': 0,
                'blocked': len(blocked_by_gmm), 'boosted': 0}

    # Sort by smart score + sector diversification
    titan_candidates.sort(key=lambda x: x['smart_score'], reverse=True)
    selected = []
    seen_sectors = set()
    for c in titan_candidates:
        if c['sector'] not in seen_sectors:
            selected.append(c)
            seen_sectors.add(c['sector'])
            if len(selected) >= MAX_TRADES:
                break
    if len(selected) < MAX_TRADES:
        for c in titan_candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= MAX_TRADES:
                    break

    # Display selected
    boosted_count = sum(1 for s in selected if s['action'] == 'BOOST')
    ce_count = sum(1 for s in selected if s['opt_type'] == 'CE')
    pe_count = sum(1 for s in selected if s['opt_type'] == 'PE')
    print(f"\n  Selected {len(selected)} trades ({ce_count} CE / {pe_count} PE, {boosted_count} BOOSTED):")
    for s in selected:
        boost_tag = ' ** 2x LOTS **' if s['action'] == 'BOOST' else ''
        print(f"    {s['sym']:15s} {s['opt_type']}  smart={s['smart_score']:.1f}  "
              f"dr={s['dr_score']:.3f}  P(MOVE)={s['p_move']:.1%}  "
              f"dir={s['dir_prob']:.2f}  [{s['sector']}]{boost_tag}")

    # Simulate trades
    print(f"\n{'─'*90}")
    print(f"  TRADE RESULTS")
    print(f"{'─'*90}")

    total_pnl = 0
    wins = 0
    losses = 0
    trades = []

    for s in selected:
        ts, candles = find_atm_option(kite, nfo_df, s['sym'], s['spot'], s['opt_type'], sim_date)
        if not candles:
            print(f"  {s['sym']:15s} {s['opt_type']}  NO OPTION DATA")
            continue

        result = simulate_trade(candles, entry_idx=5)
        if not result:
            print(f"  {s['sym']:15s} {s['opt_type']}  SIM FAILED")
            continue

        lot = LOT_SIZES.get(s['sym'], 500) * s['lot_mult']  # Apply boost multiplier!
        pnl = (result['exit'] - result['entry']) * lot
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        boost_tag = ' [2x BOOST]' if s['lot_mult'] > 1 else ''
        wl = 'WIN' if pnl > 0 else 'LOSS'
        print(f"  {s['sym']:15s} {s['opt_type']}  entry={result['entry']:.1f}  exit={result['exit']:.1f}  "
              f"ret={result['ret']:+.1f}%  lots={s['lot_mult']}x  pnl=Rs {pnl:+,.0f}  "
              f"({result['reason']}, {result['candles']}c)  [{wl}]{boost_tag}")

        trades.append({'sym': s['sym'], 'opt_type': s['opt_type'], 'pnl': pnl,
                       'ret': result['ret'], 'reason': result['reason'],
                       'lots': s['lot_mult'], 'action': s['action']})

    print(f"\n  SUMMARY: {wins}W/{losses}L  Total PnL = Rs {total_pnl:+,.0f}  "
          f"(blocked={len(blocked_by_gmm)}, boosted={boosted_count})")
    
    return {'trades': trades, 'total_pnl': total_pnl, 'wins': wins, 'losses': losses,
            'blocked': len(blocked_by_gmm), 'boosted': boosted_count}


def run_comparison_no_gmm(kite, predictor, nfo_df, sim_date, gmm_cache):
    """Run the SAME pipeline WITHOUT GMM veto/boost for comparison."""
    day_str = str(sim_date)
    nifty_df = pd.read_parquet(os.path.join(DATA_DIR, 'NIFTY50.parquet'))
    
    parquets = [f for f in os.listdir(DATA_DIR)
                if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
    all_symbols = sorted([f.replace('.parquet', '') for f in parquets])

    candidates = []
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
            ml_conf = pred.get('ml_confidence', 0)
            ml_signal = pred.get('ml_signal', 'UNKNOWN')
            prob_up = pred.get('ml_prob_up', 0)
            prob_down = pred.get('ml_prob_down', 0)

            if p_move < 0.50:
                continue
            if ml_signal == 'UP':
                direction = 'BUY'; opt_type = 'CE'; dir_prob = prob_up
            elif ml_signal == 'DOWN':
                direction = 'SELL'; opt_type = 'PE'; dir_prob = prob_down
            else:
                continue

            day_data = df[df['date'].dt.date == sim_date]
            if day_data.empty or len(day_data) < 6:
                continue
            spot = day_data.iloc[5]['close']
            dr_score = gmm_cache.get(sym, {}).get('up_score', 0)

            candidates.append({
                'sym': sym, 'direction': direction, 'opt_type': opt_type,
                'p_move': p_move, 'dir_prob': dir_prob, 'ml_conf': ml_conf,
                'dr_score': dr_score, 'spot': spot, 'sector': sector or 'OTHER',
                'smart_score': ml_conf * dir_prob * 40 + (1.0 - dr_score if direction == 'BUY' else dr_score) * 20 + p_move * 5,
            })
        except:
            continue

    candidates.sort(key=lambda x: x['smart_score'], reverse=True)
    selected = []
    seen_sectors = set()
    for c in candidates:
        if c['sector'] not in seen_sectors:
            selected.append(c)
            seen_sectors.add(c['sector'])
            if len(selected) >= MAX_TRADES:
                break
    if len(selected) < MAX_TRADES:
        for c in candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= MAX_TRADES:
                    break

    total_pnl = 0; wins = 0; losses = 0
    for s in selected:
        ts, candles = find_atm_option(kite, nfo_df, s['sym'], s['spot'], s['opt_type'], sim_date)
        if not candles:
            continue
        result = simulate_trade(candles, entry_idx=5)
        if not result:
            continue
        lot = LOT_SIZES.get(s['sym'], 500)
        pnl = (result['exit'] - result['entry']) * lot
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1

    return {'total_pnl': total_pnl, 'wins': wins, 'losses': losses,
            'ce_count': sum(1 for s in selected if s['opt_type'] == 'CE'),
            'pe_count': sum(1 for s in selected if s['opt_type'] == 'PE')}


def main():
    predictor = MovePredictor()
    if not predictor.ready:
        print('ERROR: MovePredictor not ready'); sys.exit(1)
    print(f'Models loaded: {predictor.model_type}')

    kite = get_kite_client()
    nfo_df = pd.DataFrame(kite.instruments('NFO'))
    print(f'{len(nfo_df)} NFO instruments loaded')

    sim_dates = [date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19)]
    
    grand_pnl = 0; grand_wins = 0; grand_losses = 0
    grand_blocked = 0; grand_boosted = 0
    nogmm_pnl = 0; nogmm_wins = 0; nogmm_losses = 0

    results_by_day = {}

    for d in sim_dates:
        r = run_day(kite, predictor, nfo_df, d)
        grand_pnl += r['total_pnl']
        grand_wins += r['wins']
        grand_losses += r['losses']
        grand_blocked += r['blocked']
        grand_boosted += r['boosted']
        results_by_day[str(d)] = r

        # Also run without GMM for comparison
        gmm_cache = load_gmm_cache(str(d))
        nr = run_comparison_no_gmm(kite, predictor, nfo_df, d, gmm_cache)
        nogmm_pnl += nr['total_pnl']
        nogmm_wins += nr['wins']
        nogmm_losses += nr['losses']
        results_by_day[str(d)]['no_gmm'] = nr

    print(f"\n{'='*90}")
    print(f"  GRAND TOTAL — GMM VETO/BOOST vs NO GMM")
    print(f"{'='*90}")
    print(f"\n  {'Day':<12s} {'GMM Veto/Boost':>25s}    {'No GMM (baseline)':>25s}    {'Delta':>10s}")
    print(f"  {'─'*12} {'─'*25}    {'─'*25}    {'─'*10}")
    for d in sim_dates:
        r = results_by_day[str(d)]
        nr = r['no_gmm']
        delta = r['total_pnl'] - nr['total_pnl']
        print(f"  {str(d):<12s} {r['wins']}W/{r['losses']}L Rs {r['total_pnl']:>+9,.0f}    "
              f"{nr['wins']}W/{nr['losses']}L Rs {nr['total_pnl']:>+9,.0f}    Rs {delta:>+9,.0f}")

    delta_total = grand_pnl - nogmm_pnl
    print(f"\n  {'TOTAL':<12s} {grand_wins}W/{grand_losses}L Rs {grand_pnl:>+9,.0f}    "
          f"{nogmm_wins}W/{nogmm_losses}L Rs {nogmm_pnl:>+9,.0f}    Rs {delta_total:>+9,.0f}")
    print(f"\n  GMM filter stats: {grand_blocked} trades blocked, {grand_boosted} trades boosted (2x lots)")
    print(f"  Capital: Rs 500,000  ROI with GMM: {grand_pnl/500000*100:+.2f}%  ROI without: {nogmm_pnl/500000*100:+.2f}%")
    print(f"  GMM improvement: Rs {delta_total:+,.0f}")


if __name__ == '__main__':
    main()
