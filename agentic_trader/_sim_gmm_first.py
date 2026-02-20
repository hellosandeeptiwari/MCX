"""
GMM-FIRST PIPELINE SIMULATION v2 — Score-Rank Architecture

The idea: Models should HELP each other, not work in isolation.

  Step 1: GMM scores ALL 207 stocks by anomaly (how "unusual" is the pattern?)
          → Rank by anomaly score, take top 30 most unusual
  Step 2: XGB Direction ONLY on top 30 → decides CE or PE (NO Gate filter!)
          → GMM already selected "interesting" stocks, no need for Move/Flat gate
  Step 3: Combined rank = anomaly × P(direction)
  Step 4: Sector-diversified top 7

Key difference from current pipeline:
  - No smart scoring formula (eliminates CE bias)
  - No Move/Flat gate (GMM replaces it — it already selects "unusual" = likely to move)
  - GMM SELECTS candidates, XGB Direction CONFIRMS direction
  - Models work together: GMM finds unusual → XGB says CE or PE

Exit rules: +18% profit, -28% SL, 10-candle time stop, 12-candle speed gate
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

TODAY = date(2026, 2, 19)

# Exit thresholds (same as exit_manager.py)
QUICK_PROFIT_PCT = 0.18
STOPLOSS_PCT = -0.28
TIME_STOP_CANDLES = 10
SPEED_GATE_CANDLES = 12
SPEED_GATE_MIN_PCT = 0.03

MAX_TRADES = 7
MAX_PER_SECTOR = 2
GMM_TOP_N = 30  # Take top 30 most unusual stocks from GMM

print('=' * 80)
print('  GMM-FIRST PIPELINE SIMULATION — ' + str(TODAY))
print('  "Let GMM detect abnormal first, then XGB decides direction"')
print('=' * 80)

# ─── Load models and data ───
from ml_models.predictor import MovePredictor
from ml_models.down_risk_detector import DownRiskDetector
from ml_models.feature_engineering import compute_features, get_feature_names, get_sector_for_symbol
from ml_models.data_fetcher import get_kite_client, get_instrument_token

predictor = MovePredictor()
if not predictor.ready:
    print('ERROR: MovePredictor not ready')
    sys.exit(1)
print('XGB models loaded: ' + predictor.model_type)

gmm_detector = DownRiskDetector()
if not gmm_detector.load():
    print('ERROR: DownRiskDetector not loaded')
    sys.exit(1)
print('GMM detector loaded: UP=' + ('OK' if gmm_detector.up_detector._is_trained else 'FAIL') +
      ', FLAT=' + ('OK' if gmm_detector.flat_detector._is_trained else 'FAIL'))

kite = get_kite_client()
if not kite:
    print('ERROR: No Kite client')
    sys.exit(1)
print('Kite connected.')

# Load NFO instruments
nfo = kite.instruments('NFO')
print(str(len(nfo)) + ' NFO instruments loaded')

# ─── Step 0: Load all 207 stock candle data + features ───
data_dir = 'ml_models/data/candles_5min'
foi_dir = 'ml_models/data/futures_oi'
nifty_df = pd.read_parquet(os.path.join(data_dir, 'NIFTY50.parquet'))

import json as _json

parquets = [f for f in os.listdir(data_dir)
            if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
symbols = sorted([f.replace('.parquet', '') for f in parquets])
print(str(len(symbols)) + ' stocks to scan\n')

# ─── Step 1: Run GMM on ALL stocks → find "abnormal" regime ───
print('=' * 80)
print('  STEP 1: GMM ANOMALY SCORING — rank ALL stocks by how unusual they look')
print('=' * 80)

feature_names = get_feature_names()
gmm_results = {}
feature_cache = {}  # cache features for step 2

# Cache file to avoid recomputing GMM scores every run
GMM_CACHE_FILE = '_gmm_cache_' + str(TODAY) + '.json'
if os.path.exists(GMM_CACHE_FILE):
    print('  Loading cached GMM scores from ' + GMM_CACHE_FILE + '...')
    with open(GMM_CACHE_FILE, 'r') as f:
        gmm_results = _json.load(f)
    # Still need feature_cache for XGB step — load DFs only for top N
    print('  Cached: ' + str(len(gmm_results)) + ' stocks')
    # Build feature_cache lazily (only for stocks we'll use in Step 2)
else:
    print('  Computing GMM scores for all stocks (first run, will cache)...')
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print('  [' + str(i + 1) + '/' + str(len(symbols)) + ']...')
        try:
            df = pd.read_parquet(os.path.join(data_dir, sym + '.parquet'))
            df = df[df['date'].dt.date <= TODAY]
            if len(df) < 100:
                continue

            # Compute features
            feat_df = compute_features(df, symbol=sym)
            if feat_df.empty:
                continue

            # Get last row's features
            last_row = feat_df.iloc[-1:]
            X = np.nan_to_num(last_row[feature_names].values.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)

            # Run GMM against both regimes
            result_up = gmm_detector.predict_single(X, 'UP')
            result_flat = gmm_detector.predict_single(X, 'FLAT')

            up_score = float(result_up['anomaly_score'][0])
            flat_score = float(result_flat['anomaly_score'][0])
            up_flag = bool(result_up['down_risk_flag'][0])
            flat_flag = bool(result_flat['down_risk_flag'][0])

            is_abnormal = up_flag or flat_flag
            max_anomaly = max(up_score, flat_score)
            regime_used = 'UP' if up_score >= flat_score else 'FLAT'

            gmm_results[sym] = {
                'up_score': round(up_score, 4),
                'flat_score': round(flat_score, 4),
                'up_flag': up_flag,
                'flat_flag': flat_flag,
                'is_abnormal': is_abnormal,
                'max_anomaly': round(max_anomaly, 4),
                'regime_used': regime_used,
                'bucket': str(result_up['confidence_bucket'][0]) if regime_used == 'UP'
                          else str(result_flat['confidence_bucket'][0]),
            }
            feature_cache[sym] = (df, X)

        except Exception as e:
            pass

    # Save cache
    with open(GMM_CACHE_FILE, 'w') as f:
        _json.dump(gmm_results, f)
    print('  Saved GMM cache to ' + GMM_CACHE_FILE)

abnormal_syms = [s for s, r in gmm_results.items() if r['is_abnormal']]
normal_syms = [s for s, r in gmm_results.items() if not r['is_abnormal']]

# Sort ALL stocks by anomaly score (most unusual first)
all_ranked = sorted(gmm_results.keys(), key=lambda s: gmm_results[s]['max_anomaly'], reverse=True)

print('\n  GMM RESULTS:')
print('    Total scanned: ' + str(len(gmm_results)))
print('    FLAGGED abnormal (binary): ' + str(len(abnormal_syms)) + ' (for reference)')
print('    Score range: ' + str(gmm_results[all_ranked[-1]]['max_anomaly']) + ' to ' + str(gmm_results[all_ranked[0]]['max_anomaly']))
print()

# Take TOP N by anomaly score (regardless of binary flag)
top_n = all_ranked[:GMM_TOP_N]
print('  Taking TOP ' + str(GMM_TOP_N) + ' most unusual stocks for XGB confirmation:')
print('  Rank  Symbol         AnomalyScore  UP_score  FLAT_score  Flag?')
print('  ' + '-' * 70)
for j, s in enumerate(top_n, 1):
    r = gmm_results[s]
    flag_str = 'FLAG' if r['is_abnormal'] else ''
    print('  ' + str(j).rjust(3) + '.  ' + s.ljust(14) +
          ' ' + str(r['max_anomaly']).ljust(10) +
          '    ' + str(r['up_score']).ljust(8) +
          '  ' + str(r['flat_score']).ljust(10) +
          '  ' + flag_str)

# ─── Step 2: Run XGB Direction ONLY on top N stocks (NO Gate!) ───
print('\n' + '=' * 80)
print('  STEP 2: XGB DIRECTION on TOP ' + str(GMM_TOP_N) + ' — CE or PE? (no Move/Flat gate)')
print('=' * 80)

candidates = []

for sym in top_n:
    try:
        # Lazy-load dataframe if not in cache (when using JSON cache)
        if sym in feature_cache:
            df, X_cached = feature_cache[sym]
        else:
            df = pd.read_parquet(os.path.join(data_dir, sym + '.parquet'))
            df = df[df['date'].dt.date <= TODAY]

        # Load supplementary data
        foi_path = os.path.join(foi_dir, sym + '_futures_oi.parquet')
        foi_df = pd.read_parquet(foi_path) if os.path.exists(foi_path) else None
        sector = get_sector_for_symbol(sym)
        sector_key = 'SECTOR_' + sector.upper().replace(' ', '_') if sector != 'Other' else None
        sector_df = None
        if sector_key:
            sp = os.path.join(data_dir, sector_key + '.parquet')
            if os.path.exists(sp):
                sector_df = pd.read_parquet(sp)

        # Full prediction via predictor (Gate + Direction)
        pred = predictor.get_titan_signals(df, futures_oi_df=foi_df,
                                           nifty_5min_df=nifty_df, sector_5min_df=sector_df)
        if not pred:
            continue

        signal = pred.get('ml_signal', 'UNKNOWN')
        ml_conf = pred.get('ml_confidence', 0)
        p_move = pred.get('ml_move_prob', 0)
        prob_up = pred.get('ml_prob_up', 0)
        prob_down = pred.get('ml_prob_down', 0)

        # NO Gate filter! GMM already selected unusual stocks.
        # Direction decides CE or PE based purely on prob_up vs prob_down
        if prob_up > prob_down:
            direction = 'BUY'
            option_type = 'CE'
            dir_prob = prob_up
        else:
            direction = 'SELL'
            option_type = 'PE'
            dir_prob = prob_down

        # Very loose floor — just avoid random noise
        if dir_prob < 0.30:
            continue

        # Combined rank: anomaly × P(direction) — two models cooperating
        rank_score = round(gmm_results[sym]['max_anomaly'] * dir_prob * 100, 4)

        gmm_info = gmm_results[sym]

        candidates.append({
            'sym': sym,
            'sector': sector,
            'direction': direction,
            'option': option_type,
            'signal': signal,
            'p_move': round(p_move, 4),
            'dir_prob': round(dir_prob, 4),
            'ml_conf': round(ml_conf, 4),
            'rank_score': rank_score,
            'anomaly_score': gmm_info['max_anomaly'],
            'gmm_bucket': gmm_info['bucket'],
        })

    except Exception:
        pass

# Sort by rank_score (pure probability)
candidates.sort(key=lambda c: c['rank_score'], reverse=True)

ce_cands = [c for c in candidates if c['option'] == 'CE']
pe_cands = [c for c in candidates if c['option'] == 'PE']

print('\n  XGB RESULTS on ABNORMAL stocks:')
print('    Directional candidates: ' + str(len(candidates)) + ' (' + str(len(ce_cands)) + ' CE + ' + str(len(pe_cands)) + ' PE) from top ' + str(GMM_TOP_N) + ' GMM (no gate!)')
print()

print('  ALL CANDIDATES ranked by P(MOVE) x P(direction):')
print('  Rank  Symbol         Opt   RankScore  P(move) DirP   Conf   Anomaly  Bucket')
print('  ' + '-' * 85)
for j, c in enumerate(candidates[:30], 1):
    print('  ' + str(j).rjust(3) + '.  ' + c['sym'].ljust(14) + ' ' + c['option'] +
          '   ' + str(c['rank_score']).rjust(7) +
          '    ' + str(round(c['p_move'] * 100)).rjust(3) + '%' +
          '   ' + str(round(c['dir_prob'] * 100)).rjust(3) + '%' +
          '   ' + str(round(c['ml_conf'] * 100)).rjust(3) + '%' +
          '   ' + str(c['anomaly_score']).ljust(7) +
          '  ' + c['gmm_bucket'])

# ─── Step 3: Sector-diversified selection ───
print('\n' + '=' * 80)
print('  STEP 3: SELECT TOP ' + str(MAX_TRADES) + ' (sector-diversified)')
print('=' * 80)

sector_counts = {}
selected = []
for c in candidates:
    if len(selected) >= MAX_TRADES:
        break
    sec = c['sector']
    if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
        continue
    sector_counts[sec] = sector_counts.get(sec, 0) + 1
    selected.append(c)

print('\n  GMM-FIRST PICKS:')
for j, c in enumerate(selected, 1):
    print('  #' + str(j) + ' ' + c['sym'].ljust(14) + ' ' + c['option'] + ' BUY  rank=' +
          str(c['rank_score']) + '  P(move)=' + str(round(c['p_move'] * 100)) + '%' +
          '  P(dir)=' + str(round(c['dir_prob'] * 100)) + '%' +
          '  anomaly=' + str(c['anomaly_score']) +
          '  sector=' + c['sector'])

ce_sel = sum(1 for c in selected if c['option'] == 'CE')
pe_sel = sum(1 for c in selected if c['option'] == 'PE')
print('\n  Direction mix: ' + str(ce_sel) + ' CE + ' + str(pe_sel) + ' PE')

# ─── Step 4: Simulate with actual option premiums ───
print('\n' + '=' * 80)
print('  STEP 4: SIMULATE WITH ACTUAL OPTION PREMIUMS')
print('=' * 80)


def find_atm_option(sym, option_type, spot_price, nfo_instruments):
    """Find ATM option for nearest weekly expiry."""
    opts = [i for i in nfo_instruments
            if i['name'] == sym
            and i['instrument_type'] == option_type
            and i['segment'] == 'NFO-OPT']
    if not opts:
        return None
    future_opts = [o for o in opts if o['expiry'] >= TODAY]
    if not future_opts:
        return None
    expiries = sorted(set(o['expiry'] for o in future_opts))
    nearest_expiry = None
    for exp in expiries:
        if (exp - TODAY).days <= 7:
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


def simulate_option_trade(kite_client, inst, entry_candle_idx):
    """Simulate option trade with actual premium data."""
    token = inst['token']
    data = kite_client.historical_data(token, TODAY, TODAY, '5minute')
    if not data or len(data) <= entry_candle_idx:
        return {'error': 'No option data (' + str(len(data) if data else 0) + ' candles)'}
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
            exit_reason = 'STOP_LOSS (-28%)'
            exit_candle = j
            break
        if high >= target_price:
            exit_price = target_price
            exit_reason = 'QUICK_PROFIT (+18%)'
            exit_candle = j
            break
        if candles_held >= TIME_STOP_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < 0.09:
                exit_price = close
                exit_reason = 'TIME_STOP (' + str(candles_held) + ' candles)'
                exit_candle = j
                break
        if candles_held >= SPEED_GATE_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < SPEED_GATE_MIN_PCT:
                exit_price = close
                exit_reason = 'SPEED_GATE (' + str(candles_held) + ' candles)'
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


# Simulate each pick at staggered entry times
# Entry candle 1=09:20, 2=09:25, 3=09:30, 4=09:35, 5=09:40
# We'll enter each trade at candle 5 (09:40) for fair comparison with existing sim
ENTRY_CANDLE = 5

results = []
for pick in selected:
    sym = pick['sym']
    opt_type = pick['option']

    print('\n  ' + sym + ' ' + opt_type + ' BUY (rank=' + str(pick['rank_score']) +
          ', anomaly=' + str(pick['anomaly_score']) + ')')

    # Get spot price
    spot_token = get_instrument_token(kite, sym)
    if spot_token == 0:
        print('    SKIP: No spot token')
        continue

    time.sleep(0.35)
    spot_data = kite.historical_data(spot_token, TODAY, TODAY, '5minute')
    if not spot_data or len(spot_data) <= ENTRY_CANDLE:
        print('    SKIP: No spot data')
        continue

    spot_at_entry = spot_data[ENTRY_CANDLE]['close']
    print('    Spot@09:40 = ' + str(round(spot_at_entry, 1)))

    # Find ATM option
    inst = find_atm_option(sym, opt_type, spot_at_entry, nfo)
    if not inst:
        print('    SKIP: No ATM ' + opt_type + ' found')
        continue

    print('    Option: ' + inst['tradingsymbol'] + ' (strike=' + str(inst['strike']) +
          ', exp=' + inst['expiry'] + ', lot=' + str(inst['lot_size']) + ')')

    time.sleep(0.35)
    result = simulate_option_trade(kite, inst, ENTRY_CANDLE)

    if 'error' in result:
        print('    ERROR: ' + result['error'])
        continue

    # Position sizing: 2% risk
    lot_size = inst['lot_size']
    premium_per_lot = result['entry_price'] * lot_size
    capital = 500000
    risk_per_trade = capital * 0.02
    max_lots = max(1, int(risk_per_trade / (premium_per_lot * 0.28)))
    lots = min(max_lots, 15)
    total_premium = result['entry_price'] * lot_size * lots
    pnl_rupees = result['pnl_abs'] * lot_size * lots

    tag = 'WIN' if result['pnl_pct'] > 0 else 'LOSS'

    print('    Entry: ' + str(result['entry_price']) + ' -> Exit: ' + str(result['exit_price']))
    print('    Premium P&L: ' + ('+' if result['pnl_pct'] >= 0 else '') + str(result['pnl_pct']) + '% ' + tag)
    print('    Exit: ' + result['exit_reason'] + ' | ' + str(result['candles_held']) + ' candles (' +
          str(result['time_held_min']) + ' min)')
    print('    Size: ' + str(lots) + ' lots x ' + str(lot_size) + ' = ' + str(lots * lot_size) + ' qty')
    print('    P&L: Rs ' + ('+' if pnl_rupees >= 0 else '') + str(round(pnl_rupees)) +
          ' (invested: Rs ' + str(round(total_premium)) + ')')

    results.append({
        'symbol': sym, 'option': opt_type, 'rank_score': pick['rank_score'],
        'anomaly': pick['anomaly_score'], 'sector': pick['sector'],
        'signal': pick['signal'], 'direction': pick['direction'],
        'strike': inst['strike'], 'expiry': inst['expiry'],
        'tradingsymbol': inst['tradingsymbol'],
        'entry': result['entry_price'], 'exit': result['exit_price'],
        'pnl_pct': result['pnl_pct'], 'pnl_rs': round(pnl_rupees),
        'exit_reason': result['exit_reason'],
        'candles_held': result['candles_held'],
        'max_pct': result['max_pct'], 'lots': lots, 'lot_size': lot_size,
        'invested': round(total_premium),
    })


# ─── Summary ───
print('\n' + '=' * 80)
print('  GMM-FIRST PIPELINE RESULTS')
print('=' * 80)

if results:
    wins = sum(1 for r in results if r['pnl_pct'] > 0)
    losses = len(results) - wins
    total_rs = sum(r['pnl_rs'] for r in results)
    total_invested = sum(r['invested'] for r in results)
    avg_pnl = sum(r['pnl_pct'] for r in results) / len(results)
    avg_hold = sum(r['candles_held'] for r in results) / len(results)
    ce_count = sum(1 for r in results if r['option'] == 'CE')
    pe_count = sum(1 for r in results if r['option'] == 'PE')

    for r in results:
        tag = 'WIN' if r['pnl_pct'] > 0 else 'LOSS'
        print('    ' + r['symbol'].ljust(14) + ' ' + r['option'] + ' ' +
              r['tradingsymbol'].ljust(25) +
              ': ' + ('+' if r['pnl_pct'] >= 0 else '') + str(r['pnl_pct']) + '% = Rs ' +
              ('+' if r['pnl_rs'] >= 0 else '') + str(r['pnl_rs']) +
              ' (' + r['exit_reason'].split('(')[0].strip() + ', ' +
              str(r['candles_held']) + ' candles) ' + tag)

    print()
    print('    Direction: ' + str(ce_count) + ' CE + ' + str(pe_count) + ' PE')
    print('    Win/Loss: ' + str(wins) + 'W / ' + str(losses) + 'L')
    print('    Total P&L: Rs ' + ('+' if total_rs >= 0 else '') + str(total_rs))
    print('    Avg P&L: ' + ('+' if avg_pnl >= 0 else '') + str(round(avg_pnl, 2)) + '% per trade')
    print('    Avg hold: ' + str(round(avg_hold, 1)) + ' candles')

    by_reason = {}
    for r in results:
        reason = r['exit_reason'].split('(')[0].strip()
        by_reason[reason] = by_reason.get(reason, 0) + 1
    print('    Exit reasons: ' + str(by_reason))

    # CE vs PE breakdown
    ce_results = [r for r in results if r['option'] == 'CE']
    pe_results = [r for r in results if r['option'] == 'PE']
    if ce_results:
        ce_pnl = sum(r['pnl_rs'] for r in ce_results)
        print('\n    CE subtotal: Rs ' + ('+' if ce_pnl >= 0 else '') + str(ce_pnl) +
              ' (' + str(sum(1 for r in ce_results if r['pnl_pct'] > 0)) + 'W/' +
              str(sum(1 for r in ce_results if r['pnl_pct'] <= 0)) + 'L)')
    if pe_results:
        pe_pnl = sum(r['pnl_rs'] for r in pe_results)
        print('    PE subtotal: Rs ' + ('+' if pe_pnl >= 0 else '') + str(pe_pnl) +
              ' (' + str(sum(1 for r in pe_results if r['pnl_pct'] > 0)) + 'W/' +
              str(sum(1 for r in pe_results if r['pnl_pct'] <= 0)) + 'L)')

# Compare with old pipeline
print('\n' + '=' * 80)
print('  COMPARISON: GMM-FIRST vs CURRENT PIPELINE')
print('=' * 80)
print('  CURRENT pipeline (all CE):    Rs -24,524 (2W/5L) — all 7 CE on a crash day')
if results:
    ce_info = str(ce_count) + ' CE + ' + str(pe_count) + ' PE'
    print('  GMM-FIRST pipeline (' + ce_info + '): Rs ' + ('+' if total_rs >= 0 else '') + str(total_rs) +
          ' (' + str(wins) + 'W/' + str(losses) + 'L)')
    diff = total_rs - (-24524)
    print('  Difference: Rs ' + ('+' if diff >= 0 else '') + str(diff) + ' in favor of ' +
          ('GMM-FIRST' if diff > 0 else 'CURRENT'))
