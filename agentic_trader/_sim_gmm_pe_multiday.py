"""
Multi-day simulation: Top 10 GMM UP_score → ALL FORCED PE
Tests the hypothesis across Feb 17, 18, 19 (2026)

For each day:
  1. Compute GMM UP_score for all 207 stocks using data UP TO that day
  2. Take top 10 by UP_score
  3. Force all to PE
  4. Simulate with actual ATM PE option premiums
  5. Exit rules: +18% profit, -28% SL, 10-candle time stop, 12-candle speed gate
"""
import sys, os, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

# Exit thresholds
QUICK_PROFIT_PCT = 0.18
STOPLOSS_PCT = -0.28
TIME_STOP_CANDLES = 10
SPEED_GATE_CANDLES = 12
SPEED_GATE_MIN_PCT = 0.03
ENTRY_CANDLE = 5  # index 5 = ~09:40

DAYS_TO_SIM = [
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
]

from ml_models.predictor import MovePredictor
from ml_models.down_risk_detector import DownRiskDetector
from ml_models.feature_engineering import compute_features, get_feature_names, get_sector_for_symbol
from ml_models.data_fetcher import get_kite_client, get_instrument_token

# Load models
predictor = MovePredictor()
assert predictor.ready, 'MovePredictor not ready'
print('XGB loaded: ' + predictor.model_type)

gmm_detector = DownRiskDetector()
assert gmm_detector.load(), 'GMM not loaded'
print('GMM loaded')

kite = get_kite_client()
assert kite, 'No Kite client'
print('Kite connected')

nfo = kite.instruments('NFO')
print('%d NFO instruments' % len(nfo))

# Data dirs
data_dir = 'ml_models/data/candles_5min'
foi_dir = 'ml_models/data/futures_oi'
feature_names = get_feature_names()

parquets = [f for f in os.listdir(data_dir)
            if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
symbols = sorted([f.replace('.parquet', '') for f in parquets])
print('%d stocks available\n' % len(symbols))


def compute_gmm_scores(sim_date):
    """Compute GMM UP_score for all stocks using data up to sim_date."""
    cache_file = '_gmm_cache_%s.json' % str(sim_date)
    if os.path.exists(cache_file):
        print('  Loading cached GMM scores from %s...' % cache_file)
        with open(cache_file, 'r') as f:
            return json.load(f)

    print('  Computing GMM scores for %s (%d stocks)...' % (str(sim_date), len(symbols)))
    results = {}
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print('    [%d/%d]...' % (i + 1, len(symbols)))
        try:
            df = pd.read_parquet(os.path.join(data_dir, sym + '.parquet'))
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.date <= sim_date]
            if len(df) < 100:
                continue

            feat_df = compute_features(df, symbol=sym)
            if feat_df.empty:
                continue

            last_row = feat_df.iloc[-1:]
            X = np.nan_to_num(last_row[feature_names].values.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)

            result_up = gmm_detector.predict_single(X, 'UP')
            up_score = float(result_up['anomaly_score'][0])

            results[sym] = {
                'up_score': round(up_score, 4),
            }
        except Exception:
            pass

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    print('  Cached %d scores to %s' % (len(results), cache_file))
    return results


def find_atm_option(sym, option_type, spot_price, nfo_instruments, sim_date):
    """Find ATM option for nearest weekly expiry on sim_date."""
    opts = [i for i in nfo_instruments
            if i['name'] == sym
            and i['instrument_type'] == option_type
            and i['segment'] == 'NFO-OPT']
    if not opts:
        return None
    future_opts = [o for o in opts if o['expiry'] >= sim_date]
    if not future_opts:
        return None
    expiries = sorted(set(o['expiry'] for o in future_opts))
    nearest_expiry = None
    for exp in expiries:
        if (exp - sim_date).days <= 7:
            nearest_expiry = exp
            break
    if nearest_expiry is None:
        nearest_expiry = expiries[0]
    exp_opts = [o for o in future_opts if o['expiry'] == nearest_expiry]
    strikes = sorted(set(o['strike'] for o in exp_opts))
    if not strikes:
        return None
    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    match = [o for o in exp_opts if o['strike'] == atm_strike]
    if not match:
        return None
    inst = match[0]
    return {
        'token': inst['instrument_token'],
        'tradingsymbol': inst['tradingsymbol'],
        'strike': atm_strike,
        'expiry': str(nearest_expiry),
        'lot_size': inst['lot_size'],
    }


def simulate_option_trade(kite, inst, entry_candle_idx, sim_date):
    """Simulate PE option trade using actual 5min premium data."""
    token = inst['token']
    data = kite.historical_data(token, sim_date, sim_date, '5minute')
    if not data or len(data) <= entry_candle_idx:
        return {'error': 'No option data (%d candles)' % (len(data) if data else 0)}

    entry_price = data[entry_candle_idx]['close']
    if entry_price <= 0:
        return {'error': 'Zero entry price'}

    sl_price = entry_price * (1 + STOPLOSS_PCT)
    target_price = entry_price * (1 + QUICK_PROFIT_PCT)

    exit_price = None
    exit_reason = None
    exit_candle = None
    max_premium = entry_price

    for j in range(entry_candle_idx + 1, len(data)):
        candle = data[j]
        high = candle['high']
        low = candle['low']
        close = candle['close']
        candle_time = candle['date']
        candles_held = j - entry_candle_idx

        if high > max_premium:
            max_premium = high

        if hasattr(candle_time, 'hour'):
            if candle_time.hour >= 15 and candle_time.minute >= 15:
                exit_price = close
                exit_reason = 'SESSION_CUTOFF'
                exit_candle = j
                break

        if low <= sl_price:
            exit_price = sl_price
            exit_reason = 'STOP_LOSS (-28%%)'
            exit_candle = j
            break

        if high >= target_price:
            exit_price = target_price
            exit_reason = 'QUICK_PROFIT (+18%%)'
            exit_candle = j
            break

        if candles_held >= TIME_STOP_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < 0.09:
                exit_price = close
                exit_reason = 'TIME_STOP (%dc, max +%.1f%%)' % (candles_held, pct_gain * 100)
                exit_candle = j
                break

        if candles_held >= SPEED_GATE_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < SPEED_GATE_MIN_PCT:
                exit_price = close
                exit_reason = 'SPEED_GATE (%dc, max +%.1f%%)' % (candles_held, pct_gain * 100)
                exit_candle = j
                break

    if exit_price is None:
        exit_price = data[-1]['close']
        exit_reason = 'EOD_CLOSE'
        exit_candle = len(data) - 1

    pnl_pct = (exit_price - entry_price) / entry_price * 100
    candles_held = exit_candle - entry_candle_idx

    return {
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'pnl_pct': round(pnl_pct, 2),
        'pnl_abs': round(exit_price - entry_price, 2),
        'exit_reason': exit_reason,
        'candles_held': candles_held,
        'time_held_min': candles_held * 5,
        'max_premium': round(max_premium, 2),
        'max_pct': round((max_premium - entry_price) / entry_price * 100, 2),
    }


def run_day_simulation(sim_date):
    """Run full top-10 GMM PE simulation for one day."""
    print('\n' + '=' * 80)
    print('  DAY: %s  — Top 10 GMM UP_score → ALL FORCED PE' % str(sim_date))
    print('=' * 80)

    # Step 1: compute GMM scores
    gmm_scores = compute_gmm_scores(sim_date)
    if not gmm_scores:
        print('  ERROR: No GMM scores computed')
        return []

    # Rank by UP_score
    ranked = sorted(gmm_scores.items(), key=lambda x: x[1]['up_score'], reverse=True)
    vals = [v['up_score'] for _, v in ranked]
    mean_sc = np.mean(vals)
    std_sc = np.std(vals)

    top10 = ranked[:10]
    print('\n  Top 10 UP_score (mean=%.4f, std=%.4f):' % (mean_sc, std_sc))
    for i, (sym, info) in enumerate(top10, 1):
        z = (info['up_score'] - mean_sc) / std_sc if std_sc > 0 else 0
        print('    #%2d %-15s UP=%.2f%%  Z=%+.2f' % (i, sym, info['up_score'] * 100, z))

    # Step 2: simulate PE trades
    results = []
    for sym, info in top10:
        print('\n  %s PE (UP=%.2f%%)' % (sym, info['up_score'] * 100))

        spot_token = get_instrument_token(kite, sym)
        if spot_token == 0:
            print('    SKIP: No spot token')
            continue

        time.sleep(0.3)
        spot_data = kite.historical_data(spot_token, sim_date, sim_date, '5minute')
        if not spot_data or len(spot_data) <= ENTRY_CANDLE:
            print('    SKIP: No spot data (%d candles)' % (len(spot_data) if spot_data else 0))
            continue

        spot_at_entry = spot_data[ENTRY_CANDLE]['close']
        print('    Spot@09:40 = %.1f' % spot_at_entry)

        # Get NIFTY return for context
        nifty_data = kite.historical_data(
            get_instrument_token(kite, 'NIFTY 50', exchange='NSE'),
            sim_date, sim_date, '5minute'
        ) if False else None  # skip for speed

        inst = find_atm_option(sym, 'PE', spot_at_entry, nfo, sim_date)
        if not inst:
            print('    SKIP: No ATM PE option')
            continue

        print('    Option: %s (strike=%s, exp=%s, lot=%d)' % (
            inst['tradingsymbol'], inst['strike'], inst['expiry'], inst['lot_size']))

        time.sleep(0.3)
        result = simulate_option_trade(kite, inst, ENTRY_CANDLE, sim_date)
        if 'error' in result:
            print('    ERROR: %s' % result['error'])
            continue

        lot_size = inst['lot_size']
        premium_per_lot = result['entry_price'] * lot_size
        risk_per_trade = 500000 * 0.02
        max_lots = max(1, int(risk_per_trade / (premium_per_lot * 0.28)))
        lots = min(max_lots, 15)
        total_premium = result['entry_price'] * lot_size * lots
        pnl_rupees = result['pnl_abs'] * lot_size * lots
        tag = 'WIN' if result['pnl_pct'] > 0 else 'LOSS'

        print('    Entry: %.2f -> Exit: %.2f  %+.2f%% %s' % (
            result['entry_price'], result['exit_price'], result['pnl_pct'], tag))
        print('    Exit: %s | Held: %dc (%dmin) | Max: +%.2f%%' % (
            result['exit_reason'], result['candles_held'], result['time_held_min'], result['max_pct']))
        print('    Size: %d lots x %d = %d qty | P&L: Rs %+d' % (
            lots, lot_size, lots * lot_size, round(pnl_rupees)))

        results.append({
            'date': str(sim_date),
            'symbol': sym,
            'up_score': info['up_score'],
            'tradingsymbol': inst['tradingsymbol'],
            'entry': result['entry_price'],
            'exit': result['exit_price'],
            'pnl_pct': result['pnl_pct'],
            'pnl_rs': round(pnl_rupees),
            'exit_reason': result['exit_reason'],
            'candles_held': result['candles_held'],
            'max_pct': result['max_pct'],
            'lots': lots,
            'lot_size': lot_size,
            'invested': round(total_premium),
        })

    # Day summary
    if results:
        wins = sum(1 for r in results if r['pnl_pct'] > 0)
        losses = len(results) - wins
        total_rs = sum(r['pnl_rs'] for r in results)
        total_inv = sum(r['invested'] for r in results)
        avg_pnl = sum(r['pnl_pct'] for r in results) / len(results)

        print('\n  ' + '-' * 60)
        print('  %s SUMMARY' % str(sim_date))
        print('  ' + '-' * 60)
        for r in results:
            tag = 'WIN' if r['pnl_pct'] > 0 else 'LOSS'
            print('    %-15s %s  UP=%.1f%%  %+.2f%%  Rs %+6d  %s  %s' % (
                r['symbol'], r['tradingsymbol'], r['up_score'] * 100,
                r['pnl_pct'], r['pnl_rs'],
                r['exit_reason'].split('(')[0].strip(), tag))
        print()
        print('    Win/Loss: %dW / %dL (%d trades)' % (wins, losses, len(results)))
        print('    Total P&L: Rs %+d (invested: Rs %d)' % (total_rs, total_inv))
        print('    Avg P&L: %+.2f%% per trade' % avg_pnl)

    return results


# ── Run all days ──
all_day_results = {}
for sim_date in DAYS_TO_SIM:
    day_results = run_day_simulation(sim_date)
    all_day_results[str(sim_date)] = day_results

# ── Grand Summary ──
print('\n' + '=' * 80)
print('  GRAND SUMMARY — Top 10 GMM → ALL PE  (3 days)')
print('=' * 80)

grand_total = 0
grand_wins = 0
grand_losses = 0
grand_trades = 0
grand_invested = 0

for d in DAYS_TO_SIM:
    ds = str(d)
    res = all_day_results[ds]
    if not res:
        print('  %s: NO TRADES' % ds)
        continue
    wins = sum(1 for r in res if r['pnl_pct'] > 0)
    losses = len(res) - wins
    total = sum(r['pnl_rs'] for r in res)
    inv = sum(r['invested'] for r in res)
    avg = sum(r['pnl_pct'] for r in res) / len(res)
    grand_total += total
    grand_wins += wins
    grand_losses += losses
    grand_trades += len(res)
    grand_invested += inv
    print('  %s:  Rs %+7d  %dW/%dL  avg %+.2f%%  invested Rs %d' % (
        ds, total, wins, losses, avg, inv))

print()
print('  3-DAY TOTAL: Rs %+d  %dW/%dL (%d trades)' % (
    grand_total, grand_wins, grand_losses, grand_trades))
if grand_trades > 0:
    print('  Avg P&L per trade: Rs %+d' % (grand_total // grand_trades))
    print('  Win rate: %.0f%%' % (grand_wins / grand_trades * 100))
    print('  Total invested: Rs %d' % grand_invested)
    print('  ROI: %+.2f%%' % (grand_total / grand_invested * 100 if grand_invested > 0 else 0))
