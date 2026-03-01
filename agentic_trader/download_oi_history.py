"""
DOWNLOAD HISTORICAL OPTION OI DATA — NSE F&O Bhav Copy + DhanHQ

Downloads daily strike-level Option OI from NSE archives and computes
per-stock OI features for GMM model training:
  - pcr_oi: Put-Call Ratio by OI
  - pcr_oi_change: PCR computed from OI change
  - oi_buildup_strength: -1 (short buildup) to +1 (long buildup) 
  - spot_vs_max_pain: (spot - max_pain) / spot * 100
  - iv_skew: ATM put IV - call IV (estimated from prices)
  - atm_iv: ATM average IV
  - call_resistance_dist: Distance from top-call-OI strike (%)
  - total_oi_change_pct: Daily change in total OI (%)

Data source: NSE F&O Bhav Copy (new format)
URL: https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip

Contains per-contract: Symbol, Expiry, Strike, OptionType, OHLC, SettlementPrice,
UnderlyingPrice, OpenInterest, ChangeInOI, Volume

Usage:
    python download_oi_history.py                    # Download 6 months
    python download_oi_history.py --months 12        # Download 12 months
    python download_oi_history.py --from 2025-09-01  # From specific date
"""

import os
import sys
import csv
import io
import json
import time
import math
import zipfile
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger('oi_history')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'ml_models' / 'data' / 'options_oi'
CACHE_DIR = BASE_DIR / 'ml_models' / 'data' / 'nse_bhav_cache'

# ─── NSE Config ─────────────────────────────────────────────────────
NSE_HOME = "https://www.nseindia.com"
# New-format bhav copy URL (works as of Feb 2026)
BHAV_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Column mapping for new-format bhav copy
COL_MAP = {
    'TradDt': 'trade_date',
    'TckrSymb': 'symbol',
    'FinInstrmTp': 'instrument',  # STO=stock option, STF=stock future, IDO=index option, IDF=index future
    'XpryDt': 'expiry',
    'StrkPric': 'strike',
    'OptnTp': 'option_type',      # CE or PE
    'ClsPric': 'close',
    'SttlmPric': 'settle_price',
    'UndrlygPric': 'underlying',
    'OpnIntrst': 'oi',
    'ChngInOpnIntrst': 'oi_change',
    'TtlTradgVol': 'volume',
    'TtlTrfVal': 'turnover',
}


# ─── NSE Session Manager ───────────────────────────────────────────

class NSESession:
    """Manages NSE session with cookie refresh."""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._last_cookie_time = 0
        self._cookie_ttl = 120  # Refresh cookies every 2 minutes
        self._last_request = 0
        self._min_gap = 1.5
    
    def _ensure_cookies(self):
        if time.time() - self._last_cookie_time > self._cookie_ttl:
            try:
                self._session.get(NSE_HOME, timeout=15)
                self._last_cookie_time = time.time()
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Cookie refresh failed: {e}")
    
    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)
        self._last_request = time.time()
    
    def get(self, url: str, **kwargs) -> requests.Response:
        self._ensure_cookies()
        self._throttle()
        return self._session.get(url, **kwargs)


# ─── Trading Calendar ──────────────────────────────────────────────

def get_trading_dates(from_date: date, to_date: date) -> List[date]:
    """Generate list of potential trading dates (weekdays only).
    NSE holidays will return 404 which we handle gracefully.
    """
    dates = []
    current = from_date
    while current <= to_date:
        if current.weekday() < 5:  # Monday=0 to Friday=4
            dates.append(current)
        current += timedelta(days=1)
    return dates


# ─── Download Bhav Copies ──────────────────────────────────────────

def download_bhav_copy(session: NSESession, trade_date: date) -> Optional[pd.DataFrame]:
    """Download and parse a single day's F&O bhav copy from NSE.
    
    Returns DataFrame with option rows only (CE/PE), or None if unavailable.
    """
    date_str = trade_date.strftime('%Y%m%d')
    cache_file = CACHE_DIR / f'bhav_{date_str}.parquet'
    
    # Use cache if exists
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                return df
        except Exception:
            pass
    
    url = BHAV_URL_TEMPLATE.format(date=date_str)
    
    try:
        resp = session.get(url, timeout=30)
        
        if resp.status_code == 404:
            return None  # Holiday or no data
        
        if resp.status_code != 200:
            logger.warning(f"{trade_date}: HTTP {resp.status_code}")
            return None
        
        # Unzip
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]
        csv_data = z.read(csv_name).decode('utf-8')
        
        # Parse
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = []
        
        for raw in reader:
            inst = raw.get('FinInstrmTp', '').strip()
            optn = raw.get('OptnTp', '').strip()
            
            # Only keep stock options (STO) and index options (IDO)
            if inst not in ('STO', 'IDO'):
                continue
            if optn not in ('CE', 'PE'):
                continue
            
            try:
                row = {
                    'trade_date': trade_date.isoformat(),
                    'symbol': raw['TckrSymb'].strip(),
                    'instrument': inst,
                    'expiry': raw['XpryDt'].strip(),
                    'strike': float(raw['StrkPric'].strip() or 0),
                    'option_type': optn,
                    'close': float(raw['ClsPric'].strip() or 0),
                    'settle_price': float(raw['SttlmPric'].strip() or 0),
                    'underlying': float(raw['UndrlygPric'].strip() or 0),
                    'oi': int(float(raw['OpnIntrst'].strip() or 0)),
                    'oi_change': int(float(raw['ChngInOpnIntrst'].strip() or 0)),
                    'volume': int(float(raw['TtlTradgVol'].strip() or 0)),
                }
                rows.append(row)
            except (ValueError, KeyError) as e:
                continue
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows)
        
        # Cache to parquet for re-runs
        df.to_parquet(cache_file, index=False)
        
        return df
        
    except zipfile.BadZipFile:
        logger.warning(f"{trade_date}: Invalid zip file")
        return None
    except Exception as e:
        logger.error(f"{trade_date}: {e}")
        return None


def download_all_bhav_copies(from_date: date, to_date: date) -> Dict[str, pd.DataFrame]:
    """Download bhav copies for a date range.
    
    Returns dict: date_str -> DataFrame of option rows
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    dates = get_trading_dates(from_date, to_date)
    session = NSESession()
    
    results = {}
    total = len(dates)
    success = 0
    cached = 0
    
    print(f"📥 Downloading NSE F&O bhav copies: {from_date} to {to_date} ({total} potential trading days)")
    
    for i, d in enumerate(dates, 1):
        date_str = d.isoformat()
        cache_file = CACHE_DIR / f'bhav_{d.strftime("%Y%m%d")}.parquet'
        
        is_cached = cache_file.exists()
        df = download_bhav_copy(session, d)
        
        if df is not None and len(df) > 0:
            results[date_str] = df
            success += 1
            if is_cached:
                cached += 1
            symbols = df['symbol'].nunique()
            opt_rows = len(df)
            print(f"  [{i}/{total}] {d}: ✅ {opt_rows:,} options, {symbols} symbols {'(cached)' if is_cached else ''}")
        else:
            if d.weekday() < 5:
                print(f"  [{i}/{total}] {d}: ⏭️ holiday/no data")
        
        # Progress every 20 days
        if i % 20 == 0:
            print(f"  ... progress: {success}/{i} days downloaded ({cached} cached)")
    
    print(f"\n✅ Downloaded {success}/{total} trading days ({cached} from cache)")
    return results


# ─── Compute Per-Stock Daily OI Features ───────────────────────────

def _compute_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame, 
                      spot: float) -> float:
    """Compute max pain strike.
    Max pain = strike where total loss for option writers is minimized.
    """
    if calls_df.empty and puts_df.empty:
        return spot
    
    # Get all unique strikes
    all_strikes = sorted(set(
        list(calls_df['strike'].unique()) + list(puts_df['strike'].unique())
    ))
    
    if not all_strikes:
        return spot
    
    min_pain = float('inf')
    max_pain_strike = spot
    
    for test_strike in all_strikes:
        total_pain = 0
        
        # Pain to call writers: for each CE strike, if test_strike > CE_strike,
        # call buyers profit = (test_strike - ce_strike) * oi
        for _, row in calls_df.iterrows():
            if test_strike > row['strike']:
                total_pain += (test_strike - row['strike']) * row['oi']
        
        # Pain to put writers: for each PE strike, if test_strike < PE_strike,
        # put buyers profit = (pe_strike - test_strike) * oi
        for _, row in puts_df.iterrows():
            if test_strike < row['strike']:
                total_pain += (row['strike'] - test_strike) * row['oi']
        
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = test_strike
    
    return max_pain_strike


def _estimate_iv_from_price(option_price: float, spot: float, strike: float,
                            days_to_expiry: float, option_type: str,
                            risk_free: float = 0.065) -> float:
    """Estimate implied volatility using Newton-Raphson on Black-Scholes.
    
    Simplified but functional. Returns IV as percentage (e.g., 25.0 = 25%).
    Returns 0 if cannot converge or inputs are degenerate.
    """
    if option_price <= 0 or spot <= 0 or strike <= 0 or days_to_expiry <= 0:
        return 0.0
    
    T = days_to_expiry / 365.0
    
    # Intrinsic value check
    if option_type == 'CE':
        intrinsic = max(spot - strike, 0)
    else:
        intrinsic = max(strike - spot, 0)
    
    # If option price is below intrinsic, IV can't be computed
    if option_price < intrinsic * 0.5:
        return 0.0
    
    try:
        from scipy.stats import norm
        
        def bs_price(sigma):
            if sigma <= 0:
                return 0
            d1 = (math.log(spot / strike) + (risk_free + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if option_type == 'CE':
                return spot * norm.cdf(d1) - strike * math.exp(-risk_free * T) * norm.cdf(d2)
            else:
                return strike * math.exp(-risk_free * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        def bs_vega(sigma):
            if sigma <= 0:
                return 0
            d1 = (math.log(spot / strike) + (risk_free + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            return spot * math.sqrt(T) * norm.pdf(d1)
        
        # Newton-Raphson
        sigma = 0.3  # Initial guess
        for _ in range(50):
            price = bs_price(sigma)
            vega = bs_vega(sigma)
            if vega < 1e-10:
                break
            sigma = sigma - (price - option_price) / vega
            if sigma <= 0.01:
                sigma = 0.01
            if sigma > 5.0:
                return 0.0
        
        return sigma * 100  # Convert to percentage
        
    except Exception:
        return 0.0


def compute_stock_oi_features(day_df: pd.DataFrame) -> List[dict]:
    """Compute per-stock OI features from a single day's bhav copy data.
    
    For each stock, computes:
    - pcr_oi: Total put OI / Total call OI
    - pcr_oi_change: Total put OI change / Total call OI change  
    - pcr_volume: Total put volume / Total call volume
    - oi_buildup_strength: Directional OI interpretation (-1 to +1)
    - spot_vs_max_pain: (spot - max_pain) / spot * 100
    - iv_skew: ATM put IV - call IV (positive = bearish fear)
    - atm_iv: Average ATM implied volatility
    - call_resistance_dist: (top_call_OI_strike - spot) / spot * 100
    - put_support_dist: (spot - top_put_OI_strike) / spot * 100
    - total_oi: Sum of all call+put OI
    - total_oi_change_pct: % change in total OI
    """
    features = []
    
    # Group by symbol, pick nearest expiry per symbol
    for symbol in day_df['symbol'].unique():
        sym_df = day_df[day_df['symbol'] == symbol]
        
        if len(sym_df) == 0:
            continue
        
        # Get underlying price (should be same across all rows)
        spot = sym_df['underlying'].iloc[0]
        if spot <= 0:
            continue
        
        trade_date = sym_df['trade_date'].iloc[0]
        
        # Pick nearest expiry (most liquid, closest to spot)
        expiries = sorted(sym_df['expiry'].unique())
        today = datetime.strptime(trade_date, '%Y-%m-%d').date()
        
        # Find nearest future expiry
        best_expiry = expiries[0]
        for exp_str in expiries:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if exp_date >= today:
                    best_expiry = exp_str
                    break
            except ValueError:
                continue
        
        exp_df = sym_df[sym_df['expiry'] == best_expiry]
        calls = exp_df[exp_df['option_type'] == 'CE'].copy()
        puts = exp_df[exp_df['option_type'] == 'PE'].copy()
        
        if len(calls) == 0 and len(puts) == 0:
            continue
        
        # Total OI
        total_call_oi = calls['oi'].sum() if len(calls) > 0 else 0
        total_put_oi = puts['oi'].sum() if len(puts) > 0 else 0
        total_oi = total_call_oi + total_put_oi
        
        # PCR OI
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # PCR OI Change
        total_call_oi_chg = calls['oi_change'].sum() if len(calls) > 0 else 0
        total_put_oi_chg = puts['oi_change'].sum() if len(puts) > 0 else 0
        if abs(total_call_oi_chg) > 0:
            pcr_oi_change = total_put_oi_chg / abs(total_call_oi_chg)
        else:
            pcr_oi_change = 0.0
        
        # PCR Volume
        total_call_vol = calls['volume'].sum() if len(calls) > 0 else 0
        total_put_vol = puts['volume'].sum() if len(puts) > 0 else 0
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
        
        # OI Buildup strength
        # Positive = bullish (put OI adding more than call OI = writers confident)
        # Negative = bearish (call OI adding more = hedging/shorting)
        total_oi_change = total_call_oi_chg + total_put_oi_chg
        if total_oi > 0:
            # Ratio of change to total, directional
            if total_oi_change > 0:
                # OI increasing — check if put-heavy (bullish) or call-heavy (bearish)
                oi_buildup = (total_put_oi_chg - total_call_oi_chg) / (abs(total_put_oi_chg) + abs(total_call_oi_chg) + 1)
            else:
                # OI decreasing — unwinding
                oi_buildup = -(total_call_oi_chg - total_put_oi_chg) / (abs(total_put_oi_chg) + abs(total_call_oi_chg) + 1)
        else:
            oi_buildup = 0.0
        
        # Max Pain
        max_pain = _compute_max_pain(calls, puts, spot)
        spot_vs_max_pain = (spot - max_pain) / spot * 100 if spot > 0 else 0
        
        # Find ATM strikes (closest to spot)
        all_strikes = sorted(set(
            list(calls['strike'].unique()) + list(puts['strike'].unique())
        ))
        
        if all_strikes:
            atm_strike = min(all_strikes, key=lambda x: abs(x - spot))
        else:
            atm_strike = spot
        
        # IV estimation at ATM
        atm_ce = calls[calls['strike'] == atm_strike]
        atm_pe = puts[puts['strike'] == atm_strike]
        
        try:
            exp_date = datetime.strptime(best_expiry, '%Y-%m-%d').date()
            dte = max((exp_date - today).days, 1)
        except ValueError:
            dte = 30
        
        ce_iv = 0.0
        pe_iv = 0.0
        
        if len(atm_ce) > 0:
            ce_price = atm_ce['settle_price'].iloc[0]
            if ce_price <= 0:
                ce_price = atm_ce['close'].iloc[0]
            ce_iv = _estimate_iv_from_price(ce_price, spot, atm_strike, dte, 'CE')
        
        if len(atm_pe) > 0:
            pe_price = atm_pe['settle_price'].iloc[0]
            if pe_price <= 0:
                pe_price = atm_pe['close'].iloc[0]
            pe_iv = _estimate_iv_from_price(pe_price, spot, atm_strike, dte, 'PE')
        
        iv_skew = pe_iv - ce_iv  # Positive = bearish fear premium
        atm_iv = (ce_iv + pe_iv) / 2 if (ce_iv > 0 or pe_iv > 0) else 0.0
        
        # Call resistance: strike with max call OI above spot
        call_resistance_dist = 0.0
        if len(calls) > 0:
            above_spot = calls[calls['strike'] > spot]
            if len(above_spot) > 0:
                top_call_strike = above_spot.loc[above_spot['oi'].idxmax(), 'strike']
                call_resistance_dist = (top_call_strike - spot) / spot * 100
        
        # Put support: strike with max put OI below spot
        put_support_dist = 0.0
        if len(puts) > 0:
            below_spot = puts[puts['strike'] < spot]
            if len(below_spot) > 0:
                top_put_strike = below_spot.loc[below_spot['oi'].idxmax(), 'strike']
                put_support_dist = (spot - top_put_strike) / spot * 100
        
        # Total OI change %
        total_oi_change_pct = 0.0
        prev_oi = total_oi - total_oi_change
        if prev_oi > 0:
            total_oi_change_pct = total_oi_change / prev_oi * 100
        
        feature_row = {
            'trade_date': trade_date,
            'symbol': symbol,
            'instrument': sym_df['instrument'].iloc[0],
            'underlying': spot,
            'expiry': best_expiry,
            'dte': dte,
            'total_call_oi': int(total_call_oi),
            'total_put_oi': int(total_put_oi),
            'total_oi': int(total_oi),
            'pcr_oi': round(pcr_oi, 4),
            'pcr_oi_change': round(pcr_oi_change, 4),
            'pcr_volume': round(pcr_volume, 4),
            'oi_buildup_strength': round(oi_buildup, 4),
            'spot_vs_max_pain': round(spot_vs_max_pain, 4),
            'max_pain': max_pain,
            'iv_skew': round(iv_skew, 2),
            'atm_iv': round(atm_iv, 2),
            'atm_ce_iv': round(ce_iv, 2),
            'atm_pe_iv': round(pe_iv, 2),
            'call_resistance_dist': round(call_resistance_dist, 4),
            'put_support_dist': round(put_support_dist, 4),
            'total_oi_change_pct': round(total_oi_change_pct, 4),
            'total_call_oi_change': int(total_call_oi_chg),
            'total_put_oi_change': int(total_put_oi_chg),
            'total_call_volume': int(total_call_vol),
            'total_put_volume': int(total_put_vol),
        }
        features.append(feature_row)
    
    return features


def process_all_days(bhav_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Process all downloaded bhav copy data into per-stock daily OI features."""
    
    all_features = []
    total_days = len(bhav_data)
    
    print(f"\n📊 Computing OI features for {total_days} trading days...")
    
    for i, (date_str, day_df) in enumerate(sorted(bhav_data.items()), 1):
        features = compute_stock_oi_features(day_df)
        all_features.extend(features)
        
        if i % 10 == 0 or i == total_days:
            print(f"  [{i}/{total_days}] {date_str}: {len(features)} stocks processed")
    
    if not all_features:
        print("❌ No features computed!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    
    # Sort by date and symbol
    df = df.sort_values(['trade_date', 'symbol']).reset_index(drop=True)
    
    print(f"\n✅ Computed: {len(df):,} stock-day records")
    print(f"   Stocks: {df['symbol'].nunique()}")
    print(f"   Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
    
    # Show sample stats
    print(f"\n📊 Feature statistics:")
    for col in ['pcr_oi', 'iv_skew', 'atm_iv', 'spot_vs_max_pain', 'oi_buildup_strength']:
        if col in df.columns:
            vals = df[col].replace(0, np.nan).dropna()
            if len(vals) > 0:
                print(f"   {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                      f"min={vals.min():.3f}, max={vals.max():.3f}")
    
    return df


def save_features(df: pd.DataFrame):
    """Save features as parquet files (per-stock and combined)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Save combined file
    combined_path = DATA_DIR / 'all_options_oi_features.parquet'
    df.to_parquet(combined_path, index=False)
    print(f"\n💾 Saved combined: {combined_path}")
    print(f"   Size: {combined_path.stat().st_size / 1024:.1f} KB, {len(df):,} rows")
    
    # 2. Save per-stock files (for training pipeline alignment)
    stock_count = 0
    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy()
        if len(sym_df) > 0:
            stock_path = DATA_DIR / f'{symbol}_options_oi.parquet'
            sym_df.to_parquet(stock_path, index=False)
            stock_count += 1
    
    print(f"   Per-stock files: {stock_count} stocks saved to {DATA_DIR}")
    
    # 3. Save summary JSON for quick reference
    summary = {
        'last_updated': datetime.now().isoformat(),
        'date_range': [df['trade_date'].min(), df['trade_date'].max()],
        'trading_days': df['trade_date'].nunique(),
        'total_records': len(df),
        'stocks': sorted(df[df['instrument'] == 'STO']['symbol'].unique().tolist()),
        'indices': sorted(df[df['instrument'] == 'IDO']['symbol'].unique().tolist()),
        'features': [c for c in df.columns if c not in ('trade_date', 'symbol', 'instrument', 'expiry')],
    }
    
    summary_path = DATA_DIR / 'oi_data_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   Summary: {summary_path}")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Download historical option OI data from NSE')
    parser.add_argument('--months', type=int, default=6, help='Months of history to download (default: 6)')
    parser.add_argument('--from-date', type=str, help='Start date YYYY-MM-DD (overrides --months)')
    parser.add_argument('--to-date', type=str, help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download, use cached bhav copies only')
    args = parser.parse_args()
    
    to_date = date.today()
    if args.to_date:
        to_date = date.fromisoformat(args.to_date)
    
    if args.from_date:
        from_date = date.fromisoformat(args.from_date)
    else:
        from_date = to_date - timedelta(days=args.months * 30)
    
    print(f"{'='*60}")
    print(f"📥 NSE F&O Option OI Historical Data Downloader")
    print(f"{'='*60}")
    print(f"  Period: {from_date} to {to_date} (~{(to_date - from_date).days} days)")
    print(f"  Output: {DATA_DIR}")
    print()
    
    # Step 1: Download bhav copies
    if args.skip_download:
        print("⏭️ Skipping download, using cached data...")
        bhav_data = {}
        for f in sorted(CACHE_DIR.glob('bhav_*.parquet')):
            try:
                d_str = f.stem.replace('bhav_', '')
                d = datetime.strptime(d_str, '%Y%m%d').date()
                if from_date <= d <= to_date:
                    df = pd.read_parquet(f)
                    bhav_data[d.isoformat()] = df
            except Exception:
                continue
        print(f"  Loaded {len(bhav_data)} cached days")
    else:
        bhav_data = download_all_bhav_copies(from_date, to_date)
    
    if not bhav_data:
        print("❌ No data downloaded. Check internet and NSE availability.")
        return
    
    # Step 2: Compute features
    features_df = process_all_days(bhav_data)
    
    if features_df.empty:
        print("❌ No features computed.")
        return
    
    # Step 3: Save
    save_features(features_df)
    
    print(f"\n{'='*60}")
    print(f"✅ DONE! Option OI features ready for GMM training")
    print(f"   Next: Wire into feature_engineering.py to include in model")
    print(f"{'='*60}")


# ─── LOADER FUNCTIONS (called by trainer.py) ───────────────────────

def load_options_oi_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Load pre-computed daily option OI features for a symbol.
    
    Called by trainer.py during training data preparation.
    
    Returns:
        DataFrame with trade_date, symbol, pcr_oi, iv_skew, etc.
        or None if not available.
    """
    path = DATA_DIR / f'{symbol}_options_oi.parquet'
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_all_options_oi_daily(symbols: list = None) -> Dict[str, pd.DataFrame]:
    """Load daily option OI features for all symbols.
    
    If no symbols specified, scans the options_oi directory for all available
    parquet files.
    
    Returns:
        dict: symbol -> DataFrame of daily OI features
    """
    if not DATA_DIR.exists():
        return {}
    
    if symbols is None:
        # Discover all available parquets dynamically
        symbols = [f.stem.replace('_options_oi', '')
                   for f in DATA_DIR.glob('*_options_oi.parquet')]
    
    result = {}
    for sym in symbols:
        df = load_options_oi_daily(sym)
        if df is not None and len(df) > 0:
            result[sym] = df
    
    return result


if __name__ == '__main__':
    main()
