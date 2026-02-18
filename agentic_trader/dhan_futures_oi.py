"""
FUTURES OI DATA FETCHER — Backfill historical futures OI from DhanHQ

DhanHQ provides historical OHLCV+OI for futures contracts that Kite lacks.
This module:
1. Downloads DhanHQ instrument master to find NSE futures security IDs
2. Fetches daily + intraday futures data with OI for all universe stocks
3. Saves to parquet for ML training backfill
4. Computes derived features: OI buildup, basis, etc.

Usage:
    python dhan_futures_oi.py              # Backfill all stocks
    python dhan_futures_oi.py --symbol SBIN  # Backfill single stock
"""

import os
import csv
import io
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger('dhan_fut_oi')

DHAN_API = "https://api.dhan.co/v2"
INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'futures_oi')

# Config — credentials from .env only
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Stocks we need futures data for (matches APPROVED_UNIVERSE)
UNIVERSE_STOCKS = [
    'SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK',
    'BANKBARODA', 'PNB', 'UNIONBANK', 'CANBK', 'YESBANK',
    'RELIANCE', 'INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM',
    'BHARTIARTL', 'IDEA',
    'M&M', 'MARUTI', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
    'SUNPHARMA', 'DRREDDY', 'CIPLA',
    'TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'JINDALSTEL', 'NMDC',
    'NATIONALUM', 'SAIL', 'HINDCOPPER',
    'ONGC', 'BPCL',
    'ASIANPAINT', 'TITAN', 'ITC', 'HINDUNILVR', 'ETERNAL',
    'BAJFINANCE', 'BAJAJFINSV',
    'NHPC', 'POWERGRID', 'NTPC', 'ADANIPOWER', 'TATAPOWER',
    'MCX', 'IRFC',
]

# DhanHQ equity security IDs (from dhan_oi_fetcher.py DHAN_SCRIP_MAP)
EQUITY_SCRIP_MAP = {
    'SBIN': 3045, 'HDFCBANK': 1333, 'ICICIBANK': 4963, 'AXISBANK': 5900,
    'KOTAKBANK': 1922, 'INDUSINDBK': 5258, 'BANKBARODA': 4668, 'PNB': 10666,
    'RELIANCE': 2885, 'INFY': 1594, 'TCS': 11536, 'WIPRO': 3787,
    'HCLTECH': 7229, 'TECHM': 13538, 'LTIM': 17818,
    'BHARTIARTL': 10604, 'M&M': 2031, 'MARUTI': 10999, 'BAJAJ-AUTO': 16669,
    'HEROMOTOCO': 345, 'EICHERMOT': 9013,
    'SUNPHARMA': 3351, 'DRREDDY': 881, 'CIPLA': 694,
    'TATASTEEL': 3499, 'HINDALCO': 1363, 'JSWSTEEL': 11723, 'VEDL': 3063,
    'JINDALSTEL': 6733, 'NMDC': 15332, 'SAIL': 2963,
    'ONGC': 2475, 'BPCL': 526,
    'ASIANPAINT': 236, 'TITAN': 3506, 'ITC': 1660, 'HINDUNILVR': 1394,
    'BAJFINANCE': 317, 'BAJAJFINSV': 16573,
    'NHPC': 15337, 'POWERGRID': 14977, 'NTPC': 11630, 'TATAPOWER': 3426,
    'MCX': 14932,
}


def _load_config() -> dict:
    """Load DhanHQ credentials from .env."""
    client_id = os.environ.get('DHAN_CLIENT_ID', '')
    access_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
    if client_id and access_token:
        return {'client_id': client_id, 'access_token': access_token}
    return {}


class FuturesOIFetcher:
    """Fetch historical futures OI data from DhanHQ."""
    
    def __init__(self):
        cfg = _load_config()
        self.client_id = cfg.get('client_id', '')
        self.access_token = cfg.get('access_token', '')
        self.ready = bool(self.client_id and self.access_token)
        
        self._headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id,
        }
        
        self._last_call = 0
        self._min_interval = 1.5  # Rate limit: ~40 req/min
        
        # Contract ID cache: symbol -> list of {id, expiry, type}
        self._contracts_cache = None
        
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()
    
    def _request(self, method: str, endpoint: str, json_body: dict = None) -> Tuple[int, dict]:
        if not self.ready:
            return 0, {'error': 'DhanHQ not configured'}
        self._rate_limit()
        url = f"{DHAN_API}{endpoint}"
        try:
            if method == 'GET':
                resp = requests.get(url, headers=self._headers, timeout=15)
            else:
                resp = requests.post(url, headers=self._headers, json=json_body, timeout=15)
            try:
                data = resp.json()
            except Exception:
                data = {'raw': resp.text}
            return resp.status_code, data
        except Exception as e:
            return 0, {'error': str(e)}
    
    # ================================================================
    # INSTRUMENT MASTER — find futures contract IDs
    # ================================================================
    
    def load_instrument_master(self, force_refresh: bool = False) -> Dict[str, List[dict]]:
        """Download DhanHQ instrument master and find NSE futures contracts.
        
        Returns:
            dict: symbol -> list of {id, sym, expiry, type} sorted by expiry
        """
        cache_file = os.path.join(DATA_DIR, 'futures_contracts.json')
        
        # Use cache if less than 1 day old
        if not force_refresh and os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if time.time() - mtime < 86400:
                with open(cache_file, 'r') as f:
                    self._contracts_cache = json.load(f)
                return self._contracts_cache
        
        print("📥 Downloading DhanHQ instrument master (~34MB)...")
        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=120)
        if resp.status_code != 200:
            print(f"❌ Failed to download instrument master: {resp.status_code}")
            return {}
        
        reader = csv.reader(io.StringIO(resp.text))
        header = next(reader)
        # Columns: 0=EXCH_ID, 1=SEGMENT, 2=SECURITY_ID, 3=INSTRUMENT_NAME,
        # 5=TRADING_SYMBOL, 8=EXPIRY_DATE, 13=EXCH_INSTRUMENT_TYPE
        
        contracts = {}  # symbol -> list of contracts
        
        for row in reader:
            if len(row) < 14:
                continue
            exch = row[0]
            sec_id = row[2]
            trading_sym = row[5]
            expiry = row[8]
            inst_type = row[13]
            
            # Only NSE futures (not BSE which has low OI)
            if exch != 'NSE':
                continue
            if inst_type not in ('FUT', 'FUTSTK', 'FUTIDX'):
                continue
            
            # Extract base symbol from trading symbol like "SBIN-Feb2026-FUT"
            base = trading_sym.split('-')[0] if '-' in trading_sym else trading_sym
            
            if base not in UNIVERSE_STOCKS and base not in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
                continue
            
            if base not in contracts:
                contracts[base] = []
            
            contracts[base].append({
                'id': sec_id,
                'sym': trading_sym,
                'expiry': expiry,
                'type': inst_type,
            })
        
        # Sort each by expiry
        for sym in contracts:
            contracts[sym].sort(key=lambda x: x['expiry'])
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(contracts, f, indent=2)
        
        self._contracts_cache = contracts
        print(f"✅ Found futures contracts for {len(contracts)} symbols")
        for sym, ctrs in sorted(contracts.items())[:5]:
            print(f"   {sym}: {len(ctrs)} contracts, nearest={ctrs[0]['sym']}")
        
        return contracts
    
    def get_nearest_contract(self, symbol: str) -> Optional[dict]:
        """Get the nearest-month (most liquid) futures contract for a symbol."""
        if self._contracts_cache is None:
            self.load_instrument_master()
        
        ctrs = self._contracts_cache.get(symbol, [])
        if not ctrs:
            return None
        
        # Find nearest expiry that hasn't expired yet
        now_str = datetime.now().strftime('%Y-%m-%d')
        for c in ctrs:
            exp_date = c['expiry'][:10]  # YYYY-MM-DD
            if exp_date >= now_str:
                return c
        
        # If all expired, return the last one (most recent)
        return ctrs[-1]
    
    # ================================================================
    # FETCH HISTORICAL FUTURES DATA WITH OI
    # ================================================================
    
    def fetch_daily_futures_oi(self, symbol: str, months_back: int = 36) -> Optional[pd.DataFrame]:
        """Fetch daily futures OHLCV+OI for a symbol.
        
        Uses the nearest-month contract. DhanHQ provides up to 3 years of
        continuous futures data with OI.
        
        Args:
            symbol: Stock symbol (e.g., 'SBIN')
            months_back: How many months to look back (default 36 = 3 years)
        
        Returns:
            DataFrame with date, open, high, low, close, volume, oi
        """
        contract = self.get_nearest_contract(symbol)
        if not contract:
            logger.warning(f"No futures contract found for {symbol}")
            return None
        
        from_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        body = {
            'securityId': contract['id'],
            'exchangeSegment': 'NSE_FNO',
            'instrument': 'FUTSTK',
            'oi': True,
            'fromDate': from_date,
            'toDate': to_date,
        }
        
        status, data = self._request('POST', '/charts/historical', json_body=body)
        
        if status != 200 or not data.get('timestamp'):
            # Try FUTIDX instrument type (for indices like NIFTY)
            body['instrument'] = 'FUTIDX'
            status, data = self._request('POST', '/charts/historical', json_body=body)
        
        if status != 200 or not data.get('timestamp'):
            logger.warning(f"Failed to fetch futures daily for {symbol}: {status}")
            return None
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['timestamp'], unit='s'),
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
            'fut_oi': data.get('open_interest', [0] * len(data['timestamp'])),
        })
        
        df = df.sort_values('date').reset_index(drop=True)
        df['symbol'] = symbol
        return df
    
    def fetch_intraday_futures_oi(self, symbol: str, days_back: int = 85,
                                  interval: int = 5) -> Optional[pd.DataFrame]:
        """Fetch intraday (5-min) futures candles with OI.
        
        DhanHQ allows 90 days per request for intraday.
        
        Args:
            symbol: Stock symbol
            days_back: How many days to look back (max 90 per chunk)
            interval: Candle interval in minutes (5)
        
        Returns:
            DataFrame with date, open, high, low, close, volume, fut_oi
        """
        contract = self.get_nearest_contract(symbol)
        if not contract:
            return None
        
        # Fetch in chunks of 85 days
        all_dfs = []
        end = datetime.now()
        start = end - timedelta(days=days_back)
        
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=85), end)
            from_str = current.strftime('%Y-%m-%d 09:15:00')
            to_str = chunk_end.strftime('%Y-%m-%d 15:30:00')
            
            body = {
                'securityId': contract['id'],
                'exchangeSegment': 'NSE_FNO',
                'instrument': 'FUTSTK',
                'interval': str(interval),
                'oi': True,
                'fromDate': from_str,
                'toDate': to_str,
            }
            
            status, data = self._request('POST', '/charts/intraday', json_body=body)
            
            if status != 200 or not data.get('timestamp'):
                # Try FUTIDX
                body['instrument'] = 'FUTIDX'
                status, data = self._request('POST', '/charts/intraday', json_body=body)
            
            if status == 200 and data.get('timestamp'):
                chunk_df = pd.DataFrame({
                    'date': pd.to_datetime(data['timestamp'], unit='s'),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume'],
                    'fut_oi': data.get('open_interest', [0] * len(data['timestamp'])),
                })
                all_dfs.append(chunk_df)
            
            current = chunk_end
            time.sleep(0.5)
        
        if not all_dfs:
            return None
        
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        df['symbol'] = symbol
        return df
    
    def fetch_equity_spot(self, symbol: str, months_back: int = 36) -> Optional[pd.DataFrame]:
        """Fetch daily equity spot price for basis calculation."""
        eq_id = EQUITY_SCRIP_MAP.get(symbol)
        if not eq_id:
            return None
        
        from_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        body = {
            'securityId': str(eq_id),
            'exchangeSegment': 'NSE_EQ',
            'instrument': 'EQUITY',
            'fromDate': from_date,
            'toDate': to_date,
        }
        
        status, data = self._request('POST', '/charts/historical', json_body=body)
        
        if status != 200 or not data.get('timestamp'):
            return None
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['timestamp'], unit='s'),
            'spot_close': data['close'],
            'spot_volume': data['volume'],
        })
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    # ================================================================
    # COMPUTE DERIVED FEATURES
    # ================================================================
    
    def compute_daily_features(self, symbol: str, months_back: int = 36) -> Optional[pd.DataFrame]:
        """Fetch and compute daily futures OI features for a symbol.
        
        Features computed:
        - fut_oi_change_pct: % change in OI from previous day
        - fut_oi_buildup: OI interpretation
            +1 = long buildup (OI up + price up)
            -1 = short buildup (OI up + price down)
            +0.5 = short covering (OI down + price up)
            -0.5 = long unwinding (OI down + price down)
        - fut_basis_pct: (futures - spot) / spot * 100
        - fut_oi_5d_trend: 5-day OI change % (trend)
        - fut_vol_ratio: Futures volume / Equity volume
        
        Returns:
            DataFrame with date, symbol, and 5 feature columns
        """
        # Fetch futures daily
        fut_df = self.fetch_daily_futures_oi(symbol, months_back)
        if fut_df is None or len(fut_df) < 5:
            logger.warning(f"Insufficient futures data for {symbol}")
            return None
        
        # Fetch equity spot
        spot_df = self.fetch_equity_spot(symbol, months_back)
        
        features = fut_df[['date', 'symbol']].copy()
        
        # 1. OI change %
        features['fut_oi_change_pct'] = fut_df['fut_oi'].pct_change() * 100
        
        # 2. OI buildup interpretation
        price_change = fut_df['close'].diff()
        oi_change = fut_df['fut_oi'].diff()
        
        buildup = np.zeros(len(fut_df))
        for i in range(1, len(fut_df)):
            if oi_change.iloc[i] > 0 and price_change.iloc[i] > 0:
                buildup[i] = 1.0    # Long buildup
            elif oi_change.iloc[i] > 0 and price_change.iloc[i] < 0:
                buildup[i] = -1.0   # Short buildup
            elif oi_change.iloc[i] < 0 and price_change.iloc[i] > 0:
                buildup[i] = 0.5    # Short covering
            elif oi_change.iloc[i] < 0 and price_change.iloc[i] < 0:
                buildup[i] = -0.5   # Long unwinding
        features['fut_oi_buildup'] = buildup
        
        # 3. Basis % (futures premium/discount vs spot)
        if spot_df is not None and len(spot_df) > 0:
            # Normalize dates for merge (strip time)
            features['merge_date'] = features['date'].dt.normalize()
            spot_df['merge_date'] = spot_df['date'].dt.normalize()
            
            merged = features.merge(
                spot_df[['merge_date', 'spot_close', 'spot_volume']],
                on='merge_date', how='left'
            )
            
            features['fut_basis_pct'] = np.where(
                merged['spot_close'] > 0,
                (fut_df['close'] - merged['spot_close']) / merged['spot_close'] * 100,
                0
            )
            
            # 5. Volume ratio
            features['fut_vol_ratio'] = np.where(
                merged['spot_volume'] > 0,
                fut_df['volume'] / merged['spot_volume'],
                0
            )
            
            features.drop(columns=['merge_date'], inplace=True)
        else:
            features['fut_basis_pct'] = 0.0
            features['fut_vol_ratio'] = 0.0
        
        # 4. 5-day OI trend
        features['fut_oi_5d_trend'] = (
            fut_df['fut_oi'].pct_change(5) * 100
        )
        
        # Clean up
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    # ================================================================
    # BACKFILL — fetch and save for all universe stocks
    # ================================================================
    
    def backfill_all(self, months_back: int = 36) -> dict:
        """Fetch and save futures OI features for all universe stocks.
        
        Returns:
            dict: symbol -> {'rows': int, 'status': 'ok'|'failed'|'no_contract'}
        """
        # Ensure we have instrument master
        contracts = self.load_instrument_master()
        
        results = {}
        total = len(UNIVERSE_STOCKS)
        
        for i, symbol in enumerate(UNIVERSE_STOCKS, 1):
            print(f"[{i}/{total}] {symbol}...", end=" ", flush=True)
            
            if symbol not in contracts:
                print("⏭️ no NSE futures contract")
                results[symbol] = {'rows': 0, 'status': 'no_contract'}
                continue
            
            try:
                features = self.compute_daily_features(symbol, months_back)
                
                if features is not None and len(features) > 0:
                    # Save to parquet
                    out_path = os.path.join(DATA_DIR, f'{symbol}_futures_oi.parquet')
                    features.to_parquet(out_path, index=False)
                    print(f"✅ {len(features)} rows saved")
                    results[symbol] = {'rows': len(features), 'status': 'ok'}
                else:
                    print("⚠️ no data returned")
                    results[symbol] = {'rows': 0, 'status': 'failed'}
                    
            except Exception as e:
                print(f"❌ {e}")
                results[symbol] = {'rows': 0, 'status': 'failed', 'error': str(e)}
        
        # Summary
        ok = sum(1 for v in results.values() if v['status'] == 'ok')
        total_rows = sum(v['rows'] for v in results.values())
        print(f"\n{'='*50}")
        print(f"✅ Backfilled {ok}/{total} stocks, {total_rows:,} total rows")
        print(f"📁 Saved to: {DATA_DIR}")
        
        return results
    
    def backfill_intraday_all(self, days_back: int = 85) -> dict:
        """Fetch and save intraday futures OI for all universe stocks.
        
        This provides 5-min-level OI for much richer ML features.
        WARNING: This is API-intensive (~2 calls per stock).
        """
        contracts = self.load_instrument_master()
        results = {}
        total = len(UNIVERSE_STOCKS)
        
        for i, symbol in enumerate(UNIVERSE_STOCKS, 1):
            print(f"[{i}/{total}] {symbol} intraday...", end=" ", flush=True)
            
            if symbol not in contracts:
                print("⏭️ no contract")
                results[symbol] = {'rows': 0, 'status': 'no_contract'}
                continue
            
            try:
                df = self.fetch_intraday_futures_oi(symbol, days_back)
                
                if df is not None and len(df) > 0:
                    out_path = os.path.join(DATA_DIR, f'{symbol}_intraday_oi.parquet')
                    df.to_parquet(out_path, index=False)
                    print(f"✅ {len(df)} candles saved")
                    results[symbol] = {'rows': len(df), 'status': 'ok'}
                else:
                    print("⚠️ no data")
                    results[symbol] = {'rows': 0, 'status': 'failed'}
                    
            except Exception as e:
                print(f"❌ {e}")
                results[symbol] = {'rows': 0, 'status': 'failed'}
        
        ok = sum(1 for v in results.values() if v['status'] == 'ok')
        total_rows = sum(v['rows'] for v in results.values())
        print(f"\n{'='*50}")
        print(f"✅ Intraday backfill: {ok}/{total} stocks, {total_rows:,} candles")
        
        return results


# ================================================================
# LOADER — for trainer.py to load backfilled data
# ================================================================

def load_futures_oi_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Load pre-computed daily futures OI features for a symbol."""
    path = os.path.join(DATA_DIR, f'{symbol}_futures_oi.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_futures_oi_intraday(symbol: str) -> Optional[pd.DataFrame]:
    """Load intraday futures OI candles for a symbol."""
    path = os.path.join(DATA_DIR, f'{symbol}_intraday_oi.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_all_futures_oi_daily(symbols: list = None) -> Dict[str, pd.DataFrame]:
    """Load daily futures OI features for all symbols."""
    if symbols is None:
        symbols = UNIVERSE_STOCKS
    
    result = {}
    for sym in symbols:
        df = load_futures_oi_daily(sym)
        if df is not None:
            result[sym] = df
    
    return result


if __name__ == '__main__':
    import sys
    
    fetcher = FuturesOIFetcher()
    
    if not fetcher.ready:
        print("❌ DhanHQ not configured. Check DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env")
        sys.exit(1)
    
    # Parse args
    if '--symbol' in sys.argv:
        idx = sys.argv.index('--symbol')
        symbol = sys.argv[idx + 1]
        print(f"=== Backfilling {symbol} ===")
        features = fetcher.compute_daily_features(symbol)
        if features is not None:
            print(f"\n{features.tail(5).to_string()}")
            out_path = os.path.join(DATA_DIR, f'{symbol}_futures_oi.parquet')
            features.to_parquet(out_path, index=False)
            print(f"\n✅ Saved {len(features)} rows to {out_path}")
    elif '--intraday' in sys.argv:
        fetcher.backfill_intraday_all()
    else:
        fetcher.backfill_all()
