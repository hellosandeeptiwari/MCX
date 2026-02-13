"""
Replay today's candle data through OLD vs NEW indicator logic.
Fetches actual 5-min candles from Kite API for each traded symbol,
then computes what the OLD broken pipeline produced vs the NEW fixed pipeline.
"""
import json, os, sys
from datetime import datetime, timedelta
import pandas as pd

# Setup Kite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from kiteconnect import KiteConnect
kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

# Get today's traded symbols
with open('trade_history.json') as f:
    all_trades = json.load(f)
today_trades = [t for t in all_trades if '2026-02-10' in t.get('closed_at','') or '2026-02-10' in t.get('opened_at','')]

# Unique underlyings
underlyings = list(set(t.get('underlying','') for t in today_trades if t.get('underlying')))
print(f"Symbols to analyze: {underlyings}")
print(f"Total trades: {len(today_trades)}")

# Get instrument tokens
instruments = kite.instruments("NSE")
token_map = {}
for sym in underlyings:
    ts = sym.split(":")[1]
    for inst in instruments:
        if inst['tradingsymbol'] == ts:
            token_map[sym] = inst['instrument_token']
            break

print(f"Tokens found: {len(token_map)}/{len(underlyings)}")

# For each symbol, fetch daily + intraday data AND compute OLD vs NEW indicators
results = []

for sym in underlyings:
    if sym not in token_map:
        print(f"  ⚠️ No token for {sym}")
        continue
    
    token = token_map[sym]
    ts_name = sym.split(":")[1]
    
    try:
        # Daily data (60 days)
        to_date = datetime(2026, 2, 10)
        from_date = to_date - timedelta(days=60)
        daily_data = kite.historical_data(token, from_date, to_date, "day")
        
        # Intraday 5-min data for today
        today_start = datetime(2026, 2, 10, 9, 15)
        today_end = datetime(2026, 2, 10, 15, 30)
        intraday_data = kite.historical_data(token, today_start, today_end, "5minute")
        
        if not daily_data or not intraday_data:
            print(f"  ⚠️ No data for {sym}")
            continue
        
        df = pd.DataFrame(daily_data)
        idf = pd.DataFrame(intraday_data)
        
        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['volume']
        
        # ATR (daily)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily_atr = tr.rolling(14).mean().iloc[-1]
        
        # === COMPUTE 5-MIN ATR (for NEW logic) ===
        i_tr1 = idf['high'] - idf['low']
        i_tr2 = abs(idf['high'] - idf['close'].shift(1))
        i_tr3 = abs(idf['low'] - idf['close'].shift(1))
        i_tr = pd.concat([i_tr1, i_tr2, i_tr3], axis=1).max(axis=1)
        intraday_atr = i_tr.rolling(14).mean().iloc[-1] if len(i_tr) >= 14 else i_tr.mean()
        
        # === VOLUME COMPARISON ===
        current_volume_daily = vol.iloc[-1]  # Today's daily bar volume
        volume_5d_avg_old = vol.tail(5).mean()  # OLD: includes today
        volume_5d_avg_new = vol.iloc[-6:-1].mean() if len(vol) >= 6 else vol.iloc[:-1].mean()  # NEW: excludes today
        
        today_intraday_vol = idf['volume'].sum()
        
        # For trades at various times, let's check at representative times
        # Use each trade's entry time to compute what market looked like then
        sym_trades = [t for t in today_trades if t.get('underlying') == sym]
        
        for trade in sym_trades:
            entry_time = datetime.fromisoformat(trade['timestamp'])
            pnl = trade.get('pnl', 0)
            
            # Get candles up to entry time
            candles_at_entry = idf[idf['date'] <= entry_time]
            if len(candles_at_entry) < 3:
                continue
            
            # --- Volume at entry time ---
            vol_at_entry = candles_at_entry['volume'].sum()
            elapsed_min = max((entry_time - datetime(2026, 2, 10, 9, 15)).total_seconds() / 60, 1)
            day_frac = min(elapsed_min / 375.0, 1.0)
            
            old_volume_ratio = current_volume_daily / volume_5d_avg_old if volume_5d_avg_old > 0 else 1
            new_normalized_vol = vol_at_entry / day_frac
            new_volume_ratio = new_normalized_vol / volume_5d_avg_new if volume_5d_avg_new > 0 else 1
            
            old_vol_regime = "LOW" if old_volume_ratio < 0.5 else "NORMAL" if old_volume_ratio < 1.2 else "HIGH" if old_volume_ratio < 2.0 else "EXPLOSIVE"
            new_vol_regime = "LOW" if new_volume_ratio < 0.5 else "NORMAL" if new_volume_ratio < 1.2 else "HIGH" if new_volume_ratio < 2.0 else "EXPLOSIVE"
            
            # --- EMA at entry time ---
            ic = candles_at_entry['close']
            ema9 = ic.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = ic.ewm(span=21, adjust=False).mean().iloc[-1]
            ema_spread = abs(ema9 - ema21) / ema21 * 100 if ema21 > 0 else 0
            
            ema9_series = ic.ewm(span=9, adjust=False).mean()
            ema21_series = ic.ewm(span=21, adjust=False).mean()
            
            if len(ema9_series) >= 5 and len(ema21_series) >= 5:
                recent = abs(ema9_series.iloc[-5:] - ema21_series.iloc[-5:]) / ema21_series.iloc[-5:] * 100
                
                # OLD logic
                old_compressed = (recent < 0.15).sum()
                if old_compressed >= 5:
                    old_ema_regime = "COMPRESSED"
                elif ema9 > ema21:
                    old_ema_regime = "EXPANDING_BULL"
                else:
                    old_ema_regime = "EXPANDING_BEAR"
                
                # NEW logic
                new_compressed = (recent < 0.04).sum()
                if new_compressed >= 4:
                    new_ema_regime = "COMPRESSED"
                elif ema_spread >= 0.08:
                    new_ema_regime = "EXPANDING"
                else:
                    new_ema_regime = "NORMAL"
            else:
                old_ema_regime = "COMPRESSED"
                new_ema_regime = "NORMAL"
            
            # --- ORB Hold at entry ---
            orb_candles = idf.head(3)
            orb_high = orb_candles['high'].max()
            orb_low = orb_candles['low'].min()
            ltp = candles_at_entry['close'].iloc[-1]
            
            if ltp > orb_high:
                orb_signal = "BREAKOUT_UP"
            elif ltp < orb_low:
                orb_signal = "BREAKOUT_DOWN"
            else:
                orb_signal = "INSIDE_ORB"
            
            # ORB hold candles (NEW)
            post_orb = candles_at_entry.iloc[3:] if len(candles_at_entry) > 3 else pd.DataFrame()
            orb_hold = 0
            if len(post_orb) > 0 and orb_signal != "INSIDE_ORB":
                if orb_signal == "BREAKOUT_UP":
                    orb_hold = int((post_orb['low'] >= orb_high * 0.998).sum())
                else:
                    orb_hold = int((post_orb['high'] <= orb_low * 1.002).sum())
            
            # --- Follow-through ---
            # OLD: break on first non-confirming
            old_ft = 0
            if orb_signal == "BREAKOUT_UP" and len(candles_at_entry) > 3:
                po = candles_at_entry.iloc[3:]
                for j in range(len(po)):
                    if po.iloc[j]['close'] > po.iloc[j]['open']:
                        old_ft += 1
                    else:
                        break
            elif orb_signal == "BREAKOUT_DOWN" and len(candles_at_entry) > 3:
                po = candles_at_entry.iloc[3:]
                for j in range(len(po)):
                    if po.iloc[j]['close'] < po.iloc[j]['open']:
                        old_ft += 1
                    else:
                        break
            
            # NEW: doji-tolerant
            new_ft = 0
            misses = 0
            if orb_signal != "INSIDE_ORB" and len(candles_at_entry) > 3:
                po = candles_at_entry.iloc[3:]
                is_bull = orb_signal == "BREAKOUT_UP"
                for j in range(len(po)):
                    c_open = po.iloc[j]['open']
                    c_close = po.iloc[j]['close']
                    c_high = po.iloc[j]['high']
                    c_low = po.iloc[j]['low']
                    body = abs(c_close - c_open)
                    rng = c_high - c_low if c_high > c_low else 0.01
                    is_doji = body / rng < 0.15
                    if is_doji:
                        continue
                    if is_bull and c_close > c_open:
                        new_ft += 1
                    elif not is_bull and c_close < c_open:
                        new_ft += 1
                    else:
                        misses += 1
                        if misses > 1:
                            break
            
            # --- Range expansion ---
            if len(candles_at_entry) >= 1:
                last_c = candles_at_entry.iloc[-1]
                body = abs(last_c['close'] - last_c['open'])
                old_rexp = body / daily_atr if daily_atr > 0 else 0
                new_rexp = body / intraday_atr if intraday_atr > 0 else 0
            else:
                old_rexp = 0
                new_rexp = 0
            
            # --- VWAP steepening ---
            if len(candles_at_entry) >= 10:
                mid1 = candles_at_entry['close'].iloc[-10:-5].mean()
                mid2 = candles_at_entry['close'].iloc[-5:].mean()
                vol1 = candles_at_entry['volume'].iloc[-10:-5].mean()
                vol2 = candles_at_entry['volume'].iloc[-5:].mean()
                old_steep = vol2 > vol1 * 1.2 and abs(mid2 - mid1) > daily_atr * 0.3
                new_steep = vol2 > vol1 * 1.2 and abs(mid2 - mid1) > intraday_atr * 2.0
            else:
                old_steep = False
                new_steep = False
            
            # --- ADX (NEW only) ---
            adx_val = 20.0
            if len(df) >= 28:
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
                atr_smooth = tr.rolling(14).mean()
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth)
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                dx = dx.replace([float('inf'), float('-inf')], 0).fillna(0)
                adx_series = dx.rolling(14).mean()
                if len(adx_series.dropna()) > 0:
                    adx_val = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 20.0
            
            # --- VWAP numeric (for trend_following scoring) ---
            itp = (candles_at_entry['high'] + candles_at_entry['low'] + candles_at_entry['close']) / 3
            ivol = candles_at_entry['volume']
            i_cum_tpv = (itp * ivol).cumsum()
            i_cum_vol = ivol.cumsum()
            vwap_series = (i_cum_tpv / i_cum_vol).replace([float('inf'), float('-inf')], ltp).fillna(ltp)
            
            if len(vwap_series) >= 6:
                vwap_change_pct = (vwap_series.iloc[-1] - vwap_series.iloc[-6]) / vwap_series.iloc[-6] * 100
            else:
                vwap_change_pct = 0
            
            results.append({
                'sym': sym, 'time': entry_time.strftime('%H:%M'),
                'pnl': pnl, 'candles': len(candles_at_entry),
                'old_vol': old_vol_regime, 'new_vol': new_vol_regime,
                'old_ema': old_ema_regime, 'new_ema': new_ema_regime,
                'ema_spread': round(ema_spread, 4),
                'orb_signal': orb_signal, 'orb_hold': orb_hold,
                'old_ft': old_ft, 'new_ft': new_ft,
                'old_rexp': round(old_rexp, 3), 'new_rexp': round(new_rexp, 2),
                'old_steep': old_steep, 'new_steep': new_steep,
                'adx': round(adx_val, 1),
                'vwap_pct': round(vwap_change_pct, 4),
                'daily_atr': round(daily_atr, 2),
                'intraday_atr': round(intraday_atr, 2),
            })
            
        print(f"  ✅ {sym}: {len(idf)} candles, {len(sym_trades)} trades")
        
    except Exception as e:
        print(f"  ❌ {sym}: {e}")

# === PRINT RESULTS ===
print(f"\n{'='*140}")
print("ACTUAL CANDLE-BASED COMPARISON: OLD vs NEW INDICATOR VALUES")
print(f"{'='*140}")
print(f"{'Symbol':14s} {'Time':>5s} {'PnL':>8s} | {'Vol OLD':>8s}{'→':>1s}{'NEW':>8s} | {'EMA OLD':>14s}{'→':>1s}{'NEW':>10s}{'Sprd%':>7s} | {'FT O→N':>7s} {'RExp O→N':>12s} {'Steep':>6s} | {'ORBHld':>6s} {'ADX':>5s} {'VWAP%':>7s}")
print("-"*140)

for r in results:
    vol_changed = "📍" if r['old_vol'] != r['new_vol'] else "  "
    ema_changed = "📍" if r['old_ema'] != r['new_ema'] else "  "
    ft_changed = "📍" if r['old_ft'] != r['new_ft'] else "  "
    
    sym_short = r['sym'].replace('NSE:','')
    print(f"{sym_short:14s} {r['time']:>5s} {r['pnl']:>+8.0f} | {r['old_vol']:>8s}→{r['new_vol']:>8s}{vol_changed} | {r['old_ema']:>14s}→{r['new_ema']:>10s}{ema_changed}{r['ema_spread']:>6.3f}% | {r['old_ft']:>2d}→{r['new_ft']:<2d}{ft_changed} {r['old_rexp']:>5.3f}→{r['new_rexp']:>5.2f} {str(r['new_steep']):>6s} | {r['orb_hold']:>6d} {r['adx']:>5.1f} {r['vwap_pct']:>+7.3f}")

print(f"\n{'='*140}")
print("KEY OBSERVATIONS")
print(f"{'='*140}")

# Aggregate stats
vol_changes = sum(1 for r in results if r['old_vol'] != r['new_vol'])
ema_changes = sum(1 for r in results if r['old_ema'] != r['new_ema'])
ft_changes = sum(1 for r in results if r['old_ft'] != r['new_ft'])

print(f"  Volume regime changed:  {vol_changes}/{len(results)} trades")
print(f"  EMA regime changed:     {ema_changes}/{len(results)} trades")
print(f"  Follow-through changed: {ft_changes}/{len(results)} trades")

# Which EMA regime changes from COMPRESSED to something else
compressed_to_other = [r for r in results if 'COMPRESSED' in r['old_ema'] and r['new_ema'] != 'COMPRESSED']
print(f"\n  COMPRESSED → other:  {len(compressed_to_other)}")
for r in compressed_to_other:
    sym = r['sym'].replace('NSE:','')
    print(f"    {sym:14s} → {r['new_ema']:10s} (spread={r['ema_spread']:.3f}%)  PnL={r['pnl']:+,.0f}")

# ORB hold stats
orb_holds = [r for r in results if r['orb_hold'] > 0]
print(f"\n  ORB holds > 0:       {len(orb_holds)}/{len(results)}")
for r in orb_holds:
    sym = r['sym'].replace('NSE:','')
    print(f"    {sym:14s} hold={r['orb_hold']} candles  PnL={r['pnl']:+,.0f}")

# ADX stats
high_adx = [r for r in results if r['adx'] >= 25]
print(f"\n  ADX >= 25:           {len(high_adx)}/{len(results)}")
for r in high_adx:
    sym = r['sym'].replace('NSE:','')
    print(f"    {sym:14s} ADX={r['adx']:.1f}  PnL={r['pnl']:+,.0f}")

# NEW follow-through > 0 where OLD was 0
ft_unlocked = [r for r in results if r['old_ft'] == 0 and r['new_ft'] > 0]
print(f"\n  FT unlocked (0→N):   {len(ft_unlocked)}/{len(results)}")
for r in ft_unlocked:
    sym = r['sym'].replace('NSE:','')
    print(f"    {sym:14s} 0→{r['new_ft']}  PnL={r['pnl']:+,.0f}")

# Range expansion improvement
rexp_improved = [r for r in results if r['new_rexp'] >= 0.5 and r['old_rexp'] < 0.5]
print(f"\n  RangeExp unlocked:   {len(rexp_improved)}/{len(results)}")
for r in rexp_improved:
    sym = r['sym'].replace('NSE:','')
    print(f"    {sym:14s} {r['old_rexp']:.3f}→{r['new_rexp']:.2f}  PnL={r['pnl']:+,.0f}")

print(f"\n  ATR comparison: Daily ATR vs Intraday ATR")
for r in results:
    sym = r['sym'].replace('NSE:','')
    ratio = r['daily_atr'] / r['intraday_atr'] if r['intraday_atr'] > 0 else 0
    print(f"    {sym:14s} Daily={r['daily_atr']:>8.2f}  Intraday={r['intraday_atr']:>6.2f}  Ratio={ratio:.1f}x")

# Save results for further analysis
with open('candle_replay_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to candle_replay_results.json")
