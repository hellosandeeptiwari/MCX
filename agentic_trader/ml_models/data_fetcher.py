"""
DATA FETCHER — Download historical 5-min candles from Kite API

Downloads OHLCV candles for F&O stocks and caches to disk.
Kite allows chunked requests of 60 days each for 5-min data, supporting
up to 1+ year of lookback.

Usage:
    python -m ml_models.data_fetcher              # Fetch all F&O stocks
    python -m ml_models.data_fetcher --symbols SBIN RELIANCE  # Specific stocks
    python -m ml_models.data_fetcher --days 365    # Last 365 days
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

# Add parent to path for Kite imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CANDLE_DIR = os.path.join(DATA_DIR, 'candles_5min')

# Top F&O stocks to download (sorted by liquidity/relevance)
DEFAULT_SYMBOLS = [
    # Tier-1 (most liquid)
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "BAJFINANCE", "RELIANCE", "BHARTIARTL", "INFY", "TCS",
    # Tier-2 (high beta)
    "TATASTEEL", "JSWSTEEL", "JINDALSTEL", "HINDALCO",
    "LT", "MARUTI", "TITAN", "SUNPHARMA",
    "ONGC", "NTPC", "ITC", "TATAMOTORS", "CIPLA", "IDEA",
    # High-volume wildcards
    "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY",
    "DRREDDY", "DIVISLAB", "AUROPHARMA", "BIOCON", "LUPIN",
    "WIPRO", "HCLTECH", "TECHM", "LTIM",
    "VEDL", "NMDC", "SAIL", "NATIONALUM", "HINDZINC",
    "POWERGRID", "TATAPOWER", "ADANIENT",
    "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR",
    "BANKBARODA", "PNB", "IDFCFIRSTB", "INDUSINDBK",
]


def get_kite_client():
    """Get authenticated Kite client from Titan's existing session"""
    try:
        from zerodha_tools import get_tools
        tools = get_tools(paper_mode=True)
        if hasattr(tools, 'kite') and tools.kite:
            return tools.kite
    except Exception:
        pass
    
    # Fallback: try to create from env vars / saved token file
    try:
        from kiteconnect import KiteConnect
        api_key = os.environ.get('ZERODHA_API_KEY', '')
        # .kite_access_token lives in agentic_trader/ (parent of ml_models/)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        access_token_file = os.path.join(parent_dir, '.kite_access_token')
        access_token = ''
        if os.path.exists(access_token_file):
            with open(access_token_file, 'r') as f:
                access_token = f.read().strip()
        
        if not api_key:
            # Try loading from .env
            env_file = os.path.join(parent_dir, '.env')
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('ZERODHA_API_KEY='):
                            api_key = line.strip().split('=', 1)[1]
        
        if api_key and access_token:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            return kite
    except Exception:
        pass
    
    return None


def get_instrument_token(kite, symbol: str, exchange: str = "NSE") -> int:
    """Resolve symbol to instrument token using cached instruments"""
    cache_file = os.path.join(DATA_DIR, 'instruments_cache.json')
    instruments = {}
    
    # Try cache first (valid for 24h)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400:  # 24h
            with open(cache_file, 'r') as f:
                instruments = json.load(f)
    
    cache_key = f"{exchange}:{symbol}"
    if cache_key in instruments:
        return instruments[cache_key]
    
    # Fetch from Kite API
    print(f"   Fetching instrument tokens from Kite API...")
    try:
        all_instruments = kite.instruments(exchange)
        for inst in all_instruments:
            key = f"{exchange}:{inst['tradingsymbol']}"
            instruments[key] = inst['instrument_token']
        
        # Save cache
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(instruments, f)
        
        return instruments.get(cache_key, 0)
    except Exception as e:
        print(f"   ⚠️ Failed to fetch instruments: {e}")
        return 0


def fetch_candles(kite, symbol: str, days: int = 365, interval: str = "5minute") -> pd.DataFrame:
    """Fetch historical candles for a single stock.
    
    Args:
        kite: Authenticated KiteConnect client
        symbol: Trading symbol (e.g., "SBIN")
        days: Number of calendar days to fetch (chunked at 55 days per request)
        interval: Candle interval
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    token = get_instrument_token(kite, symbol)
    if not token:
        print(f"   ❌ No instrument token for {symbol}")
        return pd.DataFrame()
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    all_candles = []
    
    # Kite allows max 60 days per 5-min request, so chunk if needed
    chunk_size = 55  # days per request (stay under 60-day limit)
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size), end_date)
        
        try:
            time.sleep(0.35)  # Respect rate limit
            candles = kite.historical_data(
                instrument_token=token,
                from_date=current_start.strftime('%Y-%m-%d'),
                to_date=current_end.strftime('%Y-%m-%d'),
                interval=interval
            )
            if candles:
                all_candles.extend(candles)
        except Exception as e:
            print(f"   ⚠️ Error fetching {symbol} ({current_start} to {current_end}): {e}")
        
        current_start = current_end + timedelta(days=1)
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates (overlapping chunks)
    df = df.drop_duplicates(subset='date').reset_index(drop=True)
    
    return df


def fetch_and_save_all(symbols: list = None, days: int = 365):
    """Fetch candles for all symbols and save to disk.
    
    Saves each stock as: data/candles_5min/{SYMBOL}.parquet
    Also saves a metadata file with fetch timestamps.
    """
    kite = get_kite_client()
    if not kite:
        print("❌ No Kite client available. Run Titan first to authenticate.")
        return False
    
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    os.makedirs(CANDLE_DIR, exist_ok=True)
    
    metadata = {}
    success = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"  FETCHING {len(symbols)} STOCKS × {days} DAYS OF 5-MIN CANDLES")
    print(f"{'='*60}")
    print(f"  Estimated time: ~{len(symbols) * 2 * 0.35 / 60:.1f} minutes")
    print(f"  Save location: {CANDLE_DIR}")
    print()
    
    for i, symbol in enumerate(symbols):
        progress = f"[{i+1}/{len(symbols)}]"
        
        # Check if already fetched recently (within 1 day)
        parquet_path = os.path.join(CANDLE_DIR, f"{symbol}.parquet")
        if os.path.exists(parquet_path):
            mtime = os.path.getmtime(parquet_path)
            if time.time() - mtime < 86400:
                print(f"  {progress} {symbol:<15s} — cached (skip)")
                success += 1
                continue
        
        print(f"  {progress} {symbol:<15s} — fetching...", end="", flush=True)
        
        try:
            df = fetch_candles(kite, symbol, days=days)
            
            if len(df) > 0:
                df.to_parquet(parquet_path, index=False)
                
                trading_days = df['date'].dt.date.nunique()
                print(f" ✅ {len(df):,} candles, {trading_days} days")
                
                metadata[symbol] = {
                    'candles': len(df),
                    'trading_days': trading_days,
                    'first_date': str(df['date'].iloc[0]),
                    'last_date': str(df['date'].iloc[-1]),
                    'fetched_at': datetime.now().isoformat()
                }
                success += 1
            else:
                print(f" ❌ no data")
                failed += 1
        except Exception as e:
            print(f" ❌ {str(e)[:50]}")
            failed += 1
    
    # Save metadata
    meta_path = os.path.join(DATA_DIR, 'fetch_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  DONE: {success} fetched, {failed} failed")
    total_candles = sum(m['candles'] for m in metadata.values())
    print(f"  Total candles: {total_candles:,}")
    print(f"  Saved to: {CANDLE_DIR}")
    print(f"{'='*60}\n")
    
    return success > 0


def load_candles(symbol: str) -> pd.DataFrame:
    """Load cached candles for a symbol."""
    path = os.path.join(CANDLE_DIR, f"{symbol}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_all_candles(symbols: list = None) -> dict:
    """Load all cached candle data. Returns {symbol: DataFrame}."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    result = {}
    for symbol in symbols:
        df = load_candles(symbol)
        if len(df) > 0:
            result[symbol] = df
    
    return result


# ══════════════════════════════════════════════════════════════
#  DAILY CANDLES (500+ days — Kite allows 2000 days)
# ══════════════════════════════════════════════════════════════

DAILY_DIR = os.path.join(DATA_DIR, 'candles_daily')


def fetch_daily_candles(kite, symbol: str, days: int = 500) -> pd.DataFrame:
    """Fetch daily candles for a stock. Kite allows up to 2000 days."""
    token = get_instrument_token(kite, symbol)
    if not token:
        return pd.DataFrame()
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    try:
        time.sleep(0.35)
        candles = kite.historical_data(
            instrument_token=token,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d'),
            interval='day'
        )
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.drop_duplicates(subset='date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"   Error fetching daily {symbol}: {str(e)[:60]}")
        return pd.DataFrame()


def fetch_and_save_daily(symbols: list = None, days: int = 500):
    """Fetch daily candles for all symbols and save to disk."""
    kite = get_kite_client()
    if not kite:
        print("No Kite client available.")
        return False
    
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    os.makedirs(DAILY_DIR, exist_ok=True)
    
    success = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"  FETCHING {len(symbols)} STOCKS x {days} DAYS OF DAILY CANDLES")
    print(f"{'='*60}")
    print(f"  Save location: {DAILY_DIR}")
    print()
    
    for i, symbol in enumerate(symbols):
        progress = f"[{i+1}/{len(symbols)}]"
        
        # Check cache (valid for 1 day)
        parquet_path = os.path.join(DAILY_DIR, f"{symbol}.parquet")
        if os.path.exists(parquet_path):
            mtime = os.path.getmtime(parquet_path)
            if time.time() - mtime < 86400:
                print(f"  {progress} {symbol:<15s} -- cached (skip)")
                success += 1
                continue
        
        print(f"  {progress} {symbol:<15s} -- fetching...", end="", flush=True)
        
        try:
            df = fetch_daily_candles(kite, symbol, days=days)
            if len(df) > 0:
                df.to_parquet(parquet_path, index=False)
                trading_days = len(df)
                print(f" OK {trading_days} days ({df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
                success += 1
            else:
                print(f" NO DATA")
                failed += 1
        except Exception as e:
            print(f" ERROR {str(e)[:50]}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"  DONE: {success} fetched, {failed} failed")
    print(f"  Saved to: {DAILY_DIR}")
    print(f"{'='*60}\n")
    
    return success > 0


def load_daily_candles(symbol: str) -> pd.DataFrame:
    """Load cached daily candles for a symbol."""
    path = os.path.join(DAILY_DIR, f"{symbol}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_all_daily(symbols: list = None) -> dict:
    """Load all cached daily candle data. Returns {symbol: DataFrame}."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    result = {}
    for symbol in symbols:
        df = load_daily_candles(symbol)
        if len(df) > 0:
            result[symbol] = df
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch Kite historical candles')
    parser.add_argument('--symbols', nargs='+', default=None, help='Specific symbols')
    parser.add_argument('--days', type=int, default=365, help='Days of 5-min history')
    parser.add_argument('--daily', action='store_true', help='Fetch daily candles (500 days)')
    parser.add_argument('--daily-days', type=int, default=500, help='Days of daily history')
    args = parser.parse_args()
    
    if args.daily:
        fetch_and_save_daily(symbols=args.symbols, days=args.daily_days)
    else:
        fetch_and_save_all(symbols=args.symbols, days=args.days)
