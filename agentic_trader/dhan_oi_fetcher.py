"""
DHAN OI FETCHER — Authenticated OI + Greeks from DhanHQ API

Richer than NSE public API:
  - Full Greeks per strike (delta, gamma, theta, vega)
  - Implied Volatility per strike
  - Best bid/ask per strike  
  - OI + previous_oi → compute OI change
  - Volume + previous_volume
  - Average price, prev close
  - Works for NSE FNO, BSE FNO, MCX Commodity, and all Indices

DhanHQ API endpoints:
  POST /v2/optionchain       → Full option chain with OI, Greeks, IV, bid/ask
  POST /v2/optionchain/expirylist → Active expiry dates
  POST /v2/marketfeed/quote   → Market depth + OI for individual instruments

Rate limits:
  - Option Chain: 1 request per 3 seconds (per unique request)
  - Token validity: 24 hours (auto-renew supported)

FAIL-SAFE: Returns empty/neutral on ANY error. Never blocks Titan.
Falls back to NSE fetcher if DhanHQ is unavailable.

Usage:
    from dhan_oi_fetcher import get_dhan_oi_fetcher
    
    fetcher = get_dhan_oi_fetcher()
    data = fetcher.fetch("SBIN")
    # data = {
    #   'symbol': 'SBIN',
    #   'source': 'DHAN',
    #   'spot_price': 785.50,
    #   'timestamp': '2026-02-15T10:30:00',
    #   'total_call_oi': 12500000,
    #   'total_put_oi': 9800000,
    #   'total_call_oi_change': 250000,
    #   'total_put_oi_change': -180000,
    #   'pcr_oi': 0.784,
    #   'pcr_volume': ...,
    #   'max_pain': 780.0,
    #   'oi_buildup_signal': 'LONG_BUILDUP',
    #   'oi_buildup_strength': 0.72,
    #   'strikes': [
    #     {'strike': 780, 'ce_oi': 500000, 'ce_oi_change': 25000,
    #      'ce_greeks': {'delta': 0.53, 'gamma': 0.001, 'theta': -15.1, 'vega': 12.2},
    #      'ce_iv': 22.5, 'pe_oi': ..., ...},
    #     ...
    #   ],
    # }
"""

import os
import json
import time
import csv
import io
import threading
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

logger = logging.getLogger("dhan_oi_fetcher")


# ─── Symbol-to-Dhan SecurityID mapping ─────────────────────────────
# Indices use IDX_I, stocks use NSE_FNO
# These are the UNDERLYING security IDs (not option contract IDs)

DHAN_SCRIP_MAP = {
    # === INDICES ===
    'NIFTY':       {'scrip_id': 13,    'segment': 'IDX_I'},
    'NIFTY 50':    {'scrip_id': 13,    'segment': 'IDX_I'},
    'BANKNIFTY':   {'scrip_id': 25,    'segment': 'IDX_I'},
    'NIFTY BANK':  {'scrip_id': 25,    'segment': 'IDX_I'},
    'FINNIFTY':    {'scrip_id': 27,    'segment': 'IDX_I'},
    'MIDCPNIFTY':  {'scrip_id': 442,   'segment': 'IDX_I'},
    
    # === F&O STOCKS (Top 50 by options liquidity) ===
    'ADANIENT':    {'scrip_id': 25,    'segment': 'NSE_FNO'},
    'ADANIPORTS':  {'scrip_id': 15083, 'segment': 'NSE_FNO'},
    'APOLLOHOSP':  {'scrip_id': 157,   'segment': 'NSE_FNO'},
    'ASIANPAINT':  {'scrip_id': 236,   'segment': 'NSE_FNO'},
    'AXISBANK':    {'scrip_id': 5900,  'segment': 'NSE_FNO'},
    'BAJAJ-AUTO':  {'scrip_id': 16669, 'segment': 'NSE_FNO'},
    'BAJFINANCE':  {'scrip_id': 317,   'segment': 'NSE_FNO'},
    'BAJAJFINSV':  {'scrip_id': 16675, 'segment': 'NSE_FNO'},
    'BHARTIARTL':  {'scrip_id': 10604, 'segment': 'NSE_FNO'},
    'BPCL':        {'scrip_id': 526,   'segment': 'NSE_FNO'},
    'BRITANNIA':   {'scrip_id': 547,   'segment': 'NSE_FNO'},
    'CIPLA':       {'scrip_id': 694,   'segment': 'NSE_FNO'},
    'COALINDIA':   {'scrip_id': 20374, 'segment': 'NSE_FNO'},
    'DIVISLAB':    {'scrip_id': 10940, 'segment': 'NSE_FNO'},
    'DRREDDY':     {'scrip_id': 881,   'segment': 'NSE_FNO'},
    'EICHERMOT':   {'scrip_id': 910,   'segment': 'NSE_FNO'},
    'GRASIM':      {'scrip_id': 1232,  'segment': 'NSE_FNO'},
    'HCLTECH':     {'scrip_id': 7229,  'segment': 'NSE_FNO'},
    'HDFCBANK':    {'scrip_id': 1333,  'segment': 'NSE_FNO'},
    'HDFCLIFE':    {'scrip_id': 467,   'segment': 'NSE_FNO'},
    'HEROMOTOCO':  {'scrip_id': 1348,  'segment': 'NSE_FNO'},
    'HINDALCO':    {'scrip_id': 1363,  'segment': 'NSE_FNO'},
    'HINDUNILVR':  {'scrip_id': 1394,  'segment': 'NSE_FNO'},
    'ICICIBANK':   {'scrip_id': 4963,  'segment': 'NSE_FNO'},
    'INDUSINDBK':  {'scrip_id': 5258,  'segment': 'NSE_FNO'},
    'INFY':        {'scrip_id': 1594,  'segment': 'NSE_FNO'},
    'ITC':         {'scrip_id': 1660,  'segment': 'NSE_FNO'},
    'JSWSTEEL':    {'scrip_id': 11723, 'segment': 'NSE_FNO'},
    'KOTAKBANK':   {'scrip_id': 1922,  'segment': 'NSE_FNO'},
    'LT':          {'scrip_id': 11483, 'segment': 'NSE_FNO'},
    'LTM':         {'scrip_id': 17818, 'segment': 'NSE_FNO'},
    'M&M':         {'scrip_id': 2031,  'segment': 'NSE_FNO'},
    'MARUTI':      {'scrip_id': 10999, 'segment': 'NSE_FNO'},
    'NESTLEIND':   {'scrip_id': 17963, 'segment': 'NSE_FNO'},
    'NTPC':        {'scrip_id': 11630, 'segment': 'NSE_FNO'},
    'ONGC':        {'scrip_id': 2475,  'segment': 'NSE_FNO'},
    'POWERGRID':   {'scrip_id': 14977, 'segment': 'NSE_FNO'},
    'RELIANCE':    {'scrip_id': 2885,  'segment': 'NSE_FNO'},
    'SBILIFE':     {'scrip_id': 21808, 'segment': 'NSE_FNO'},
    'SBIN':        {'scrip_id': 3045,  'segment': 'NSE_FNO'},
    'SUNPHARMA':   {'scrip_id': 3351,  'segment': 'NSE_FNO'},
    'TATAMOTORS':  {'scrip_id': 3456,  'segment': 'NSE_FNO'},
    'TATASTEEL':   {'scrip_id': 3499,  'segment': 'NSE_FNO'},
    'TCS':         {'scrip_id': 11536, 'segment': 'NSE_FNO'},
    'TECHM':       {'scrip_id': 13538, 'segment': 'NSE_FNO'},
    'TITAN':       {'scrip_id': 3506,  'segment': 'NSE_FNO'},
    'ULTRACEMCO':  {'scrip_id': 11532, 'segment': 'NSE_FNO'},
    'WIPRO':       {'scrip_id': 3787,  'segment': 'NSE_FNO'},
    
    # === MCX Commodity ===
    'CRUDEOIL':    {'scrip_id': 444,   'segment': 'MCX_COMM'},
    'NATURALGAS':  {'scrip_id': 460,   'segment': 'MCX_COMM'},
    'GOLD':        {'scrip_id': 445,   'segment': 'MCX_COMM'},
    'SILVER':      {'scrip_id': 456,   'segment': 'MCX_COMM'},
}


# ─── Credentials ───────────────────────────────────────────────────
# Read from .env only


def _load_credentials() -> dict:
    """Load DhanHQ credentials from .env."""
    env_client = os.environ.get('DHAN_CLIENT_ID', '')
    env_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
    if env_client and env_token:
        return {'client_id': env_client, 'access_token': env_token}
    return {}


class DhanOIFetcher:
    """Fetch option chain OI + Greeks from DhanHQ authenticated API.
    
    FAIL-SAFE: Returns empty dict on any error. Never blocks.
    Thread-safe with request locking.
    
    Features over NSE:
    - Full Greeks per strike (delta, gamma, theta, vega)
    - Bid/ask per strike
    - More reliable than NSE (no cookie dance, no anti-bot blocks)
    - Covers MCX Commodity options too
    - Previous OI + previous volume for change computation
    """
    
    BASE_URL = "https://api.dhan.co/v2"
    CHAIN_URL = f"{BASE_URL}/optionchain"
    EXPIRY_URL = f"{BASE_URL}/optionchain/expirylist"
    PROFILE_URL = f"{BASE_URL}/profile"
    RENEW_URL = f"{BASE_URL}/RenewToken"
    QUOTE_URL = f"{BASE_URL}/marketfeed/quote"
    
    # Rate limit: 1 unique request per 3 seconds (option chain)
    MIN_REQUEST_GAP = 3.2  # slightly over to be safe
    # Market quote has separate rate limit: 1 req/sec, 1000 instruments
    QUOTE_REQUEST_GAP = 1.1
    
    # Cache TTL: 90 seconds (faster refresh for watcher pipeline)
    CACHE_TTL = 90
    
    # Circuit breaker: disable after N consecutive failures
    MAX_CONSECUTIVE_FAILURES = 5
    
    def __init__(self, client_id: str = None, access_token: str = None):
        """
        Args:
            client_id: DhanHQ client ID. Reads from .env if not given.
            access_token: JWT access token. Reads from .env if not given.
        """
        # Load from config if not provided
        if not client_id or not access_token:
            creds = _load_credentials()
            client_id = client_id or creds.get('client_id', '')
            access_token = access_token or creds.get('access_token', '')
        
        self.client_id = client_id
        self.access_token = access_token
        self.ready = bool(client_id and access_token)
        self.data_plan_active = None  # checked on first use
        
        # Rate limiting
        self._last_request_time = 0.0
        self._last_quote_time = 0.0  # separate throttle for market quote
        self._lock = threading.Lock()
        self._quote_lock = threading.Lock()
        
        # Cache: {key: (timestamp, data)}
        self._cache: Dict[str, Tuple[float, dict]] = {}
        
        # Market quote cache: {cache_key: (timestamp, data)}
        self._quote_cache: Dict[str, Tuple[float, dict]] = {}
        self._QUOTE_CACHE_TTL = 60  # 60s for real-time quote data
        
        # OI timeseries: {security_id: [(timestamp, oi), ...]}
        # Stores OI snapshots for computing intraday rate-of-change
        self._oi_timeseries: Dict[str, list] = {}
        self._OI_TIMESERIES_MAX_AGE = 1800  # keep last 30 minutes
        
        # Circuit breaker
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0
        
        # Expiry cache: {symbol: (timestamp, [expiry_dates])}
        self._expiry_cache: Dict[str, Tuple[float, list]] = {}
        self._EXPIRY_CACHE_TTL = 3600  # 1 hour
        
        if self.ready:
            logger.info(f"DhanOIFetcher initialized (client_id={client_id[:4]}...)")
        else:
            logger.warning("DhanOIFetcher: No credentials. Set client_id + access_token.")
    
    def _headers(self) -> dict:
        """API request headers."""
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id,
        }
    
    def _throttle(self):
        """Enforce rate limit: 1 request per 3 seconds."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.MIN_REQUEST_GAP:
                time.sleep(self.MIN_REQUEST_GAP - elapsed)
            self._last_request_time = time.time()
    
    def _throttle_quote(self):
        """Enforce rate limit for market quote: 1 request per second."""
        with self._quote_lock:
            now = time.time()
            elapsed = now - self._last_quote_time
            if elapsed < self.QUOTE_REQUEST_GAP:
                time.sleep(self.QUOTE_REQUEST_GAP - elapsed)
            self._last_quote_time = time.time()
    
    def _check_circuit_breaker(self) -> bool:
        """Returns True if circuit is CLOSED (OK to proceed)."""
        if not self._circuit_open:
            return True
        if time.time() > self._circuit_open_until:
            # Half-open: allow one attempt
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("DhanOI: Circuit breaker half-open, retrying...")
            return True
        return False
    
    def _record_success(self):
        """Reset failure counter on success."""
        self._consecutive_failures = 0
        self._circuit_open = False
    
    def _record_failure(self, error_msg: str = ""):
        """Track failure for circuit breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 300  # 5 min cooldown
            logger.warning(f"DhanOI: Circuit OPEN after {self._consecutive_failures} failures. "
                          f"Cooldown 5 min. Last error: {error_msg}")
    
    def _resolve_symbol(self, symbol: str) -> Optional[dict]:
        """Resolve symbol to DhanHQ scrip_id + segment.
        
        Handles:
          - "NSE:SBIN" → strip exchange prefix
          - "SBIN" → direct lookup
          - Case-insensitive
          
        Returns:
            {'scrip_id': int, 'segment': str} or None
        """
        # Strip exchange prefix
        clean = symbol.strip()
        if ':' in clean:
            clean = clean.split(':', 1)[1]
        clean = clean.upper().strip()
        
        # Direct lookup
        if clean in DHAN_SCRIP_MAP:
            return DHAN_SCRIP_MAP[clean]
        
        # Try common aliases
        aliases = {
            'NIFTY50': 'NIFTY',
            'NIFTY 50': 'NIFTY',
            'NIFTYBANK': 'BANKNIFTY',
            'NIFTY BANK': 'BANKNIFTY',
            'BANK NIFTY': 'BANKNIFTY',
        }
        if clean in aliases:
            return DHAN_SCRIP_MAP.get(aliases[clean])
        
        return None
    
    def check_data_plan(self) -> bool:
        """Check if Data API subscription is active.
        
        Returns True if data plan is active, False otherwise.
        Caches result after first check.
        """
        if self.data_plan_active is not None:
            return self.data_plan_active
        
        if not self.ready:
            self.data_plan_active = False
            return False
        
        try:
            r = requests.get(self.PROFILE_URL, headers=self._headers(), timeout=10)
            if r.status_code == 200:
                profile = r.json()
                self.data_plan_active = profile.get('dataPlan', '').lower() == 'active'
                if not self.data_plan_active:
                    logger.warning("DhanOI: Data Plan is INACTIVE. "
                                  "Subscribe at web.dhan.co to use Option Chain API.")
                else:
                    logger.info("DhanOI: Data Plan ACTIVE ✓")
                return self.data_plan_active
        except Exception as e:
            logger.warning(f"DhanOI: Profile check failed: {e}")
        
        self.data_plan_active = False
        return False
    
    def renew_token(self) -> bool:
        """Renew access token for another 24 hours.
        
        Must be called while current token is still valid.
        Uses dhanClientId header (per DhanHQ docs).
        Returns True if renewal successful.
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'access-token': self.access_token,
                'dhanClientId': self.client_id,
            }
            r = requests.post(self.RENEW_URL, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                new_token = data.get('accessToken', '')
                if new_token:
                    self.access_token = new_token
                    logger.info(f"DhanOI: Token renewed in-memory, expires {data.get('expiryTime', '?')}")
                    return True
            else:
                logger.warning(f"DhanOI: Token renewal failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"DhanOI: Token renewal error: {e}")
        return False
    
    def fetch_expiries(self, symbol: str) -> list:
        """Fetch active expiry dates for an underlying.
        
        Returns list of date strings ["2026-02-19", "2026-02-27", ...] or empty.
        """
        try:
            resolved = self._resolve_symbol(symbol)
            if not resolved:
                return []
            
            cache_key = f"expiry:{resolved['scrip_id']}:{resolved['segment']}"
            cached = self._expiry_cache.get(cache_key)
            if cached:
                ts, expiries = cached
                if time.time() - ts < self._EXPIRY_CACHE_TTL:
                    return expiries
            
            self._throttle()
            
            payload = {
                'UnderlyingScrip': resolved['scrip_id'],
                'UnderlyingSeg': resolved['segment'],
            }
            
            r = requests.post(self.EXPIRY_URL, headers=self._headers(),
                            json=payload, timeout=15)
            
            if r.status_code == 200:
                data = r.json()
                expiries = data.get('data', [])
                if expiries:
                    self._expiry_cache[cache_key] = (time.time(), expiries)
                    self._record_success()
                    return expiries
            else:
                self._record_failure(f"expiry {r.status_code}: {r.text[:100]}")
                
        except Exception as e:
            self._record_failure(str(e))
        
        return []
    
    def fetch(self, symbol: str, expiry: str = None) -> dict:
        """Fetch full option chain OI + Greeks for a symbol.
        
        FAIL-SAFE: Returns empty dict on any error.
        
        Args:
            symbol: e.g., "SBIN", "NSE:SBIN", "NIFTY", "BANKNIFTY"
            expiry: specific expiry date "YYYY-MM-DD". If None, uses nearest.
            
        Returns:
            Parsed OI data dict (see module docstring) or empty dict.
        """
        try:
            if not self.ready:
                return {}
            
            if not self._check_circuit_breaker():
                return {}
            
            # Resolve symbol
            resolved = self._resolve_symbol(symbol)
            if not resolved:
                logger.debug(f"DhanOI: Unknown symbol '{symbol}'")
                return {}
            
            # Check cache
            cache_key = f"chain:{resolved['scrip_id']}:{resolved['segment']}:{expiry or 'nearest'}"
            cached = self._cache.get(cache_key)
            if cached:
                ts, data = cached
                if time.time() - ts < self.CACHE_TTL:
                    return data
            
            # Get nearest expiry if not specified
            if not expiry:
                expiries = self.fetch_expiries(symbol)
                if expiries:
                    expiry = expiries[0]  # nearest
                else:
                    return {}
            
            # Fetch option chain
            self._throttle()
            
            payload = {
                'UnderlyingScrip': resolved['scrip_id'],
                'UnderlyingSeg': resolved['segment'],
                'Expiry': expiry,
            }
            
            r = requests.post(self.CHAIN_URL, headers=self._headers(),
                            json=payload, timeout=15)
            
            if r.status_code == 200:
                raw = r.json()
                if raw.get('status') == 'success':
                    result = self._parse_chain(raw.get('data', {}), symbol, expiry)
                    if result:
                        result['source'] = 'DHAN'
                        self._cache[cache_key] = (time.time(), result)
                        self._record_success()
                        return result
            elif r.status_code == 401:
                error_data = r.json() if r.text else {}
                err = error_data.get('data', {})
                if '806' in str(err) or 'not Subscribed' in str(err):
                    logger.warning("DhanOI: Data APIs not subscribed (Error 806)")
                    self.data_plan_active = False
                elif '807' in str(err) or '809' in str(err) or 'expired' in str(err).lower():
                    logger.warning("DhanOI: Token expired/invalid — attempting renewal")
                    # Try renewing token immediately before giving up
                    if self.renew_token():
                        logger.info("DhanOI: Token renewed after 401 — retrying fetch")
                        return self.fetch(symbol, expiry)  # Retry once with fresh token
                    # Renewal failed — check env for externally-refreshed token
                    import os as _dhan_os
                    _env_token = _dhan_os.environ.get('DHAN_ACCESS_TOKEN', '')
                    if _env_token and _env_token != self.access_token:
                        self.access_token = _env_token
                        logger.info("DhanOI: Picked up refreshed token from env — retrying")
                        return self.fetch(symbol, expiry)
                else:
                    logger.warning(f"DhanOI: Auth error: {r.text[:200]}")
                self._record_failure(f"{r.status_code}: {r.text[:100]}")
            else:
                self._record_failure(f"{r.status_code}: {r.text[:100]}")
            
        except Exception as e:
            self._record_failure(str(e))
        
        return {}
    
    def _parse_chain(self, data: dict, symbol: str, expiry: str) -> dict:
        """Parse DhanHQ option chain response into structured format.
        
        DhanHQ response structure:
          data.last_price: float (underlying LTP)
          data.oc.{strike}.ce/pe: {oi, previous_oi, volume, previous_volume,
              greeks: {delta, theta, gamma, vega}, implied_volatility,
              last_price, top_bid_price, top_ask_price, security_id, ...}
        """
        try:
            spot_price = data.get('last_price', 0)
            oc = data.get('oc', {})
            
            if not oc:
                return {}
            
            strikes = []
            total_call_oi = 0
            total_put_oi = 0
            total_call_oi_change = 0
            total_put_oi_change = 0
            total_call_volume = 0
            total_put_volume = 0
            
            for strike_str, strike_data in oc.items():
                try:
                    strike_price = float(strike_str)
                except (ValueError, TypeError):
                    continue
                
                ce = strike_data.get('ce', {}) or {}
                pe = strike_data.get('pe', {}) or {}
                
                # Call data
                ce_oi = ce.get('oi', 0) or 0
                ce_prev_oi = ce.get('previous_oi', 0) or 0
                ce_oi_change = ce_oi - ce_prev_oi
                ce_volume = ce.get('volume', 0) or 0
                ce_prev_volume = ce.get('previous_volume', 0) or 0
                ce_iv = ce.get('implied_volatility', 0) or 0
                ce_ltp = ce.get('last_price', 0) or 0
                ce_avg_price = ce.get('average_price', 0) or 0
                ce_prev_close = ce.get('previous_close_price', 0) or 0
                ce_bid = ce.get('top_bid_price', 0) or 0
                ce_ask = ce.get('top_ask_price', 0) or 0
                ce_bid_qty = ce.get('top_bid_quantity', 0) or 0
                ce_ask_qty = ce.get('top_ask_quantity', 0) or 0
                ce_sec_id = ce.get('security_id', 0) or 0
                ce_greeks = ce.get('greeks', {}) or {}
                
                # Put data
                pe_oi = pe.get('oi', 0) or 0
                pe_prev_oi = pe.get('previous_oi', 0) or 0
                pe_oi_change = pe_oi - pe_prev_oi
                pe_volume = pe.get('volume', 0) or 0
                pe_prev_volume = pe.get('previous_volume', 0) or 0
                pe_iv = pe.get('implied_volatility', 0) or 0
                pe_ltp = pe.get('last_price', 0) or 0
                pe_avg_price = pe.get('average_price', 0) or 0
                pe_prev_close = pe.get('previous_close_price', 0) or 0
                pe_bid = pe.get('top_bid_price', 0) or 0
                pe_ask = pe.get('top_ask_price', 0) or 0
                pe_bid_qty = pe.get('top_bid_quantity', 0) or 0
                pe_ask_qty = pe.get('top_ask_quantity', 0) or 0
                pe_sec_id = pe.get('security_id', 0) or 0
                pe_greeks = pe.get('greeks', {}) or {}
                
                # Accumulate
                total_call_oi += ce_oi
                total_put_oi += pe_oi
                total_call_oi_change += ce_oi_change
                total_put_oi_change += pe_oi_change
                total_call_volume += ce_volume
                total_put_volume += pe_volume
                
                strikes.append({
                    'strike': strike_price,
                    # Call side
                    'ce_oi': ce_oi,
                    'ce_oi_change': ce_oi_change,
                    'ce_prev_oi': ce_prev_oi,
                    'ce_volume': ce_volume,
                    'ce_prev_volume': ce_prev_volume,
                    'ce_iv': ce_iv,
                    'ce_ltp': ce_ltp,
                    'ce_avg_price': ce_avg_price,
                    'ce_prev_close': ce_prev_close,
                    'ce_bid': ce_bid,
                    'ce_ask': ce_ask,
                    'ce_bid_qty': ce_bid_qty,
                    'ce_ask_qty': ce_ask_qty,
                    'ce_security_id': ce_sec_id,
                    'ce_greeks': {
                        'delta': ce_greeks.get('delta', 0),
                        'gamma': ce_greeks.get('gamma', 0),
                        'theta': ce_greeks.get('theta', 0),
                        'vega': ce_greeks.get('vega', 0),
                    },
                    # Put side
                    'pe_oi': pe_oi,
                    'pe_oi_change': pe_oi_change,
                    'pe_prev_oi': pe_prev_oi,
                    'pe_volume': pe_volume,
                    'pe_prev_volume': pe_prev_volume,
                    'pe_iv': pe_iv,
                    'pe_ltp': pe_ltp,
                    'pe_avg_price': pe_avg_price,
                    'pe_prev_close': pe_prev_close,
                    'pe_bid': pe_bid,
                    'pe_ask': pe_ask,
                    'pe_bid_qty': pe_bid_qty,
                    'pe_ask_qty': pe_ask_qty,
                    'pe_security_id': pe_sec_id,
                    'pe_greeks': {
                        'delta': pe_greeks.get('delta', 0),
                        'gamma': pe_greeks.get('gamma', 0),
                        'theta': pe_greeks.get('theta', 0),
                        'vega': pe_greeks.get('vega', 0),
                    },
                    # Change
                    'ce_change': ce_ltp - ce_prev_close if ce_prev_close else 0,
                    'pe_change': pe_ltp - pe_prev_close if pe_prev_close else 0,
                })
            
            if not strikes:
                return {}
            
            # Sort by strike
            strikes.sort(key=lambda x: x['strike'])
            
            # === Computed Metrics ===
            pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 1.0
            pcr_oi_change = round(total_put_oi_change / total_call_oi_change, 3) if total_call_oi_change != 0 else 0.0
            pcr_volume = round(total_put_volume / total_call_volume, 3) if total_call_volume > 0 else 1.0
            
            # Top OI strikes
            top_call_oi = sorted(
                [(s['strike'], s['ce_oi']) for s in strikes if s['ce_oi'] > 0],
                key=lambda x: x[1], reverse=True
            )[:5]
            top_put_oi = sorted(
                [(s['strike'], s['pe_oi']) for s in strikes if s['pe_oi'] > 0],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            call_resistance = top_call_oi[0][0] if top_call_oi else 0
            put_support = top_put_oi[0][0] if top_put_oi else 0
            
            # Top OI change strikes
            top_call_oi_change = sorted(
                [(s['strike'], s['ce_oi_change']) for s in strikes if s['ce_oi_change'] > 0],
                key=lambda x: x[1], reverse=True
            )[:5]
            top_put_oi_change = sorted(
                [(s['strike'], s['pe_oi_change']) for s in strikes if s['pe_oi_change'] > 0],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            # Max pain
            max_pain = self._calc_max_pain(strikes)
            
            # IV skew
            iv_skew = self._calc_iv_skew(strikes, spot_price)
            
            # ATM Greeks summary
            atm_greeks = self._atm_greeks(strikes, spot_price)
            
            # ═══ Market Quote Enrichment ═══
            # Fetch oi_day_high/low, buy/sell qty, 5-level depth for ATM zone
            # This is a SEPARATE API (1 req/sec, 1000 instruments) — not option chain
            _quote_enrichment = {}
            try:
                if spot_price > 0:
                    # Collect ATM zone security IDs (±5 strikes)
                    _atm_sorted = sorted(strikes, key=lambda s: abs(s['strike'] - spot_price))
                    _atm_zone = _atm_sorted[:10]
                    _sec_ids = []
                    _sid_to_strike = {}  # map security_id → (strike, 'ce'|'pe')
                    for s in _atm_zone:
                        ce_sid = s.get('ce_security_id', 0)
                        pe_sid = s.get('pe_security_id', 0)
                        if ce_sid:
                            _sec_ids.append(ce_sid)
                            _sid_to_strike[ce_sid] = (s['strike'], 'ce')
                        if pe_sid:
                            _sec_ids.append(pe_sid)
                            _sid_to_strike[pe_sid] = (s['strike'], 'pe')
                    
                    if _sec_ids:
                        _quotes = self.fetch_market_quotes(_sec_ids, segment='NSE_FNO')
                        if _quotes:
                            # Aggregate ATM zone metrics
                            _total_buy_qty = 0
                            _total_sell_qty = 0
                            _ce_at_day_high = 0
                            _pe_at_day_high = 0
                            _atm_enriched = 0
                            _ce_oi_velocities = []
                            _pe_oi_velocities = []
                            
                            for sid, q in _quotes.items():
                                if sid not in _sid_to_strike:
                                    continue
                                strike_val, opt_type = _sid_to_strike[sid]
                                _total_buy_qty += q.get('buy_quantity', 0)
                                _total_sell_qty += q.get('sell_quantity', 0)
                                
                                # Check if OI is at intraday high (fresh buildup)
                                _oi = q.get('oi', 0)
                                _oi_dh = q.get('oi_day_high', 0)
                                if _oi > 0 and _oi_dh > 0 and _oi >= _oi_dh * 0.98:
                                    if opt_type == 'ce':
                                        _ce_at_day_high += 1
                                    else:
                                        _pe_at_day_high += 1
                                
                                # Enrich strike data with depth
                                for s in strikes:
                                    if s['strike'] == strike_val:
                                        if opt_type == 'ce':
                                            s['ce_buy_qty_total'] = q.get('buy_quantity', 0)
                                            s['ce_sell_qty_total'] = q.get('sell_quantity', 0)
                                            s['ce_oi_day_high'] = q.get('oi_day_high', 0)
                                            s['ce_oi_day_low'] = q.get('oi_day_low', 0)
                                            s['ce_depth'] = q.get('depth', {})
                                        else:
                                            s['pe_buy_qty_total'] = q.get('buy_quantity', 0)
                                            s['pe_sell_qty_total'] = q.get('sell_quantity', 0)
                                            s['pe_oi_day_high'] = q.get('oi_day_high', 0)
                                            s['pe_oi_day_low'] = q.get('oi_day_low', 0)
                                            s['pe_depth'] = q.get('depth', {})
                                        break
                                
                                # OI velocity
                                vel = self._compute_oi_velocity(str(sid))
                                if vel is not None:
                                    if opt_type == 'ce':
                                        _ce_oi_velocities.append(vel)
                                    else:
                                        _pe_oi_velocities.append(vel)
                                _atm_enriched += 1
                            
                            _quote_enrichment = {
                                'atm_buy_qty': _total_buy_qty,
                                'atm_sell_qty': _total_sell_qty,
                                'ce_at_day_high_count': _ce_at_day_high,
                                'pe_at_day_high_count': _pe_at_day_high,
                                'atm_enriched_count': _atm_enriched,
                                'ce_oi_velocity': sum(_ce_oi_velocities) / len(_ce_oi_velocities) if _ce_oi_velocities else None,
                                'pe_oi_velocity': sum(_pe_oi_velocities) / len(_pe_oi_velocities) if _pe_oi_velocities else None,
                            }
                            logger.debug(f"DhanOI: Quote enrichment for {symbol}: "
                                        f"buy/sell={_total_buy_qty}/{_total_sell_qty}, "
                                        f"CE@DH={_ce_at_day_high}, PE@DH={_pe_at_day_high}")
            except Exception as e:
                logger.debug(f"DhanOI: Quote enrichment failed: {e}")
            
            # OI buildup signal
            buildup_signal, buildup_strength = self._detect_oi_buildup(
                total_call_oi_change, total_put_oi_change,
                total_call_volume, total_put_volume,
                spot_price, strikes, _quote_enrichment
            )
            
            # Clean symbol
            clean_symbol = symbol.split(':')[-1] if ':' in symbol else symbol
            
            return {
                'symbol': clean_symbol,
                'spot_price': spot_price,
                'timestamp': datetime.now().isoformat(),
                'expiry': expiry,
                
                # Totals
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'total_call_oi_change': total_call_oi_change,
                'total_put_oi_change': total_put_oi_change,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                
                # Ratios
                'pcr_oi': pcr_oi,
                'pcr_oi_change': pcr_oi_change,
                'pcr_volume': pcr_volume,
                
                # Key levels
                'max_pain': max_pain,
                'call_resistance': call_resistance,
                'put_support': put_support,
                'iv_skew': iv_skew,
                
                # Top strikes
                'top_call_oi_strikes': top_call_oi,
                'top_put_oi_strikes': top_put_oi,
                'top_call_oi_change_strikes': top_call_oi_change,
                'top_put_oi_change_strikes': top_put_oi_change,
                
                # Buildup signal
                'oi_buildup_signal': buildup_signal,
                'oi_buildup_strength': buildup_strength,
                
                # Participant identification (v4: writer vs buyer)
                'oi_participant_id': getattr(self, '_last_participant_id', 'UNKNOWN'),
                'oi_participant_detail': getattr(self, '_last_participant_detail', {}),
                
                # Greeks (ATM summary)
                'atm_greeks': atm_greeks,
                
                # Market Quote enrichment (oi_day_high/low, buy/sell)
                'quote_enrichment': _quote_enrichment,
                
                # Full strike data
                'strikes': strikes,
            }
            
        except Exception as e:
            logger.debug(f"DhanOI: Parse error: {e}")
            return {}
    
    def _atm_greeks(self, strikes: list, spot: float) -> dict:
        """Extract ATM option Greeks for easy GPT consumption."""
        try:
            if not strikes or spot <= 0:
                return {}
            
            atm = min(strikes, key=lambda s: abs(s['strike'] - spot))
            
            return {
                'strike': atm['strike'],
                'ce_delta': atm['ce_greeks']['delta'],
                'ce_gamma': atm['ce_greeks']['gamma'],
                'ce_theta': atm['ce_greeks']['theta'],
                'ce_vega': atm['ce_greeks']['vega'],
                'pe_delta': atm['pe_greeks']['delta'],
                'pe_gamma': atm['pe_greeks']['gamma'],
                'pe_theta': atm['pe_greeks']['theta'],
                'pe_vega': atm['pe_greeks']['vega'],
                'ce_iv': atm.get('ce_iv', 0),
                'pe_iv': atm.get('pe_iv', 0),
            }
        except Exception:
            return {}
    
    @staticmethod
    def find_optimal_strike(direction: str, strikes: list, spot_price: float) -> dict:
        """OI Heatmap Strike Picker — find where institutional action is concentrated.

        Scores each strike in ATM ± 2 zone on 5 factors:
          1. Volume (30 pts)  — liquidity for quick entry/exit
          2. Spread (25 pts)  — tighter bid-ask = less slippage
          3. Delta  (20 pts)  — peak at ~0.50 (ATM sweet spot)
          4. Support (15 pts) — contra-side OI writing = support/resistance building
          5. Momentum (10 pts) — same-side buying activity vs writing (can be negative)

        Args:
            direction: 'BUY' (bullish → buy CE) or 'SELL' (bearish → buy PE)
            strikes: Full per-strike data list from fetch()
            spot_price: Current spot price

        Returns:
            {'selection': 'ATM'|'ITM_1'|'OTM_1'|..., 'strike': float,
             'reason': str, 'score': float, 'atm_strike': float}
        """
        _default = {'selection': 'ATM', 'strike': 0, 'reason': 'no_data', 'score': 0,
                     'atm_strike': 0, 'details': {}}
        try:
            if not strikes or spot_price <= 0:
                return _default

            # Sort by strike price
            sorted_all = sorted(strikes, key=lambda s: s['strike'])
            if len(sorted_all) < 3:
                return _default

            # Find ATM index
            atm_idx = min(range(len(sorted_all)),
                          key=lambda i: abs(sorted_all[i]['strike'] - spot_price))

            # Zone: ATM ± 2
            z_start = max(0, atm_idx - 2)
            z_end = min(len(sorted_all), atm_idx + 3)
            zone = sorted_all[z_start:z_end]
            atm_in_zone = atm_idx - z_start  # ATM position within zone

            is_ce = (direction == 'BUY')

            # Collect zone values for normalisation
            if is_ce:
                volumes = [s.get('ce_volume', 0) for s in zone]
            else:
                volumes = [s.get('pe_volume', 0) for s in zone]
            max_vol = max(volumes) if volumes else 1

            # Contra-OI normalisation
            if is_ce:
                contra_ois = [abs(s.get('pe_oi_change', 0)) for s in zone]
                same_ois = [abs(s.get('ce_oi_change', 0)) for s in zone]
            else:
                contra_ois = [abs(s.get('ce_oi_change', 0)) for s in zone]
                same_ois = [abs(s.get('pe_oi_change', 0)) for s in zone]
            max_contra = max(contra_ois) if contra_ois else 1
            max_same = max(same_ois) if same_ois else 1

            best_score = -999
            best_idx = atm_in_zone
            best_details = {}

            for i, s in enumerate(zone):
                if is_ce:
                    vol = s.get('ce_volume', 0)
                    bid = s.get('ce_bid', 0)
                    ask = s.get('ce_ask', 0)
                    delta = abs(s.get('ce_greeks', {}).get('delta', 0))
                    oi_chg = s.get('ce_oi_change', 0)
                    px_chg = s.get('ce_change', 0)
                    contra_oi_chg = s.get('pe_oi_change', 0)
                    contra_px_chg = s.get('pe_change', 0)
                else:
                    vol = s.get('pe_volume', 0)
                    bid = s.get('pe_bid', 0)
                    ask = s.get('pe_ask', 0)
                    delta = abs(s.get('pe_greeks', {}).get('delta', 0))
                    oi_chg = s.get('pe_oi_change', 0)
                    px_chg = s.get('pe_change', 0)
                    contra_oi_chg = s.get('ce_oi_change', 0)
                    contra_px_chg = s.get('ce_change', 0)

                # === 1. VOLUME (0-30) ===
                vol_score = (vol / max_vol * 30) if max_vol > 0 else 15

                # === 2. SPREAD (0-25) ===
                mid = (bid + ask) / 2
                if mid > 0:
                    spread_pct = (ask - bid) / mid
                    spread_score = max(0.0, 25 * (1 - spread_pct / 0.05))
                else:
                    spread_score = 0.0

                # === 3. DELTA (0-20) — sweet spot at 0.45-0.55 ===
                delta_score = max(0.0, 20 * (1 - abs(delta - 0.50) / 0.30))

                # === 4. CONTRA-SIDE SUPPORT (0-15) ===
                #   Contra OI up + contra price down = writing (support/resistance) → good
                if contra_oi_chg > 0 and contra_px_chg <= 0:
                    support_score = min(15.0, (contra_oi_chg / max_contra * 15) if max_contra > 0 else 0)
                else:
                    support_score = 0.0

                # === 5. SAME-SIDE MOMENTUM (-10 to +10) ===
                #   OI up + price up = buying activity (good)
                #   OI up + price down = writing (resistance → bad)
                if oi_chg > 0:
                    norm = (oi_chg / max_same * 10) if max_same > 0 else 0
                    momentum_score = min(10.0, norm) if px_chg >= 0 else -min(10.0, norm)
                else:
                    momentum_score = 0.0

                total = vol_score + spread_score + delta_score + support_score + momentum_score

                if total > best_score:
                    best_score = total
                    best_idx = i
                    best_details = {
                        'vol': round(vol_score, 1), 'spread': round(spread_score, 1),
                        'delta': round(delta_score, 1), 'support': round(support_score, 1),
                        'momentum': round(momentum_score, 1), 'total': round(total, 1),
                    }

            # ── Map zone-offset → StrikeSelection label ──
            offset = best_idx - atm_in_zone  # positive = higher strike
            if offset == 0:
                selection = 'ATM'
            elif is_ce:
                # CE: lower strike = ITM, higher = OTM
                sel_map = {-1: 'ITM_1', -2: 'ITM_2', 1: 'OTM_1', 2: 'OTM_2'}
                selection = sel_map.get(offset, 'ITM_2' if offset < -2 else 'OTM_2')
            else:
                # PE: higher strike = ITM, lower = OTM
                sel_map = {1: 'ITM_1', 2: 'ITM_2', -1: 'OTM_1', -2: 'OTM_2'}
                selection = sel_map.get(offset, 'ITM_2' if offset > 2 else 'OTM_2')

            return {
                'selection': selection,
                'strike': zone[best_idx]['strike'],
                'reason': (f"V={best_details.get('vol', 0)} Sp={best_details.get('spread', 0)} "
                           f"D={best_details.get('delta', 0)} Su={best_details.get('support', 0)} "
                           f"M={best_details.get('momentum', 0)}"),
                'score': round(best_score, 1),
                'atm_strike': zone[atm_in_zone]['strike'],
                'details': best_details,
            }
        except Exception:
            return _default

    def _calc_max_pain(self, strikes: list) -> float:
        """Calculate max pain strike."""
        try:
            if not strikes:
                return 0.0
            
            min_pain = float('inf')
            max_pain_strike = 0.0
            
            for test in strikes:
                test_strike = test['strike']
                total_pain = 0
                
                for s in strikes:
                    if test_strike > s['strike'] and s['ce_oi'] > 0:
                        total_pain += (test_strike - s['strike']) * s['ce_oi']
                    if test_strike < s['strike'] and s['pe_oi'] > 0:
                        total_pain += (s['strike'] - test_strike) * s['pe_oi']
                
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception:
            return 0.0
    
    def _calc_iv_skew(self, strikes: list, spot: float) -> float:
        """ATM IV skew = put IV - call IV. Positive → bearish fear."""
        try:
            if not strikes or spot <= 0:
                return 0.0
            
            atm = min(strikes, key=lambda s: abs(s['strike'] - spot))
            ce_iv = atm.get('ce_iv', 0)
            pe_iv = atm.get('pe_iv', 0)
            
            if ce_iv > 0 and pe_iv > 0:
                return round(pe_iv - ce_iv, 2)
            return 0.0
        except Exception:
            return 0.0
    
    def _detect_oi_buildup(self, call_oi_chg: int, put_oi_chg: int,
                           call_vol: int, put_vol: int,
                           spot: float, strikes: list,
                           quote_enrichment: dict = None) -> Tuple[str, float]:
        """Detect OI buildup pattern — ENHANCED v3.
        
        Improvements over v2:
          7) OI day-high detection (fresh intraday buildup via Market Quote API)
          8) Exchange buy/sell queue imbalance (aggregate pending orders)
          9) Intraday OI velocity (rate-of-change from timeseries cache)
        
        v2 features (retained):
          1) Price-direction cross-reference (textbook OI interpretation)
          2) ATM-concentrated OI analysis (5 strikes each side of ATM)
          3) IV skew confirmation (put IV vs call IV supports direction)
          4) OI-change concentration (single-strike dominance = noise filter)
          5) Bid/ask imbalance (writing vs buying heuristic)
          6) Expiry-week noise guard (lower strength near expiry)
        """
        if quote_enrichment is None:
            quote_enrichment = {}
        try:
            net_oi_change = put_oi_chg - call_oi_chg
            total_oi_change = abs(put_oi_chg) + abs(call_oi_chg)
            
            if total_oi_change == 0:
                return 'NEUTRAL', 0.0
            
            oi_ratio = net_oi_change / total_oi_change if total_oi_change > 0 else 0
            vol_ratio = put_vol / call_vol if call_vol > 0 else 1.0
            
            # ── ATM-concentrated OI analysis (±5 strikes around spot) ──
            # OI changes far OTM are noise (hedging, straddles).
            # Only ATM-zone OI represents directional conviction.
            atm_ce_chg = 0
            atm_pe_chg = 0
            atm_ce_vol = 0
            atm_pe_vol = 0
            _atm_iv_skew = 0.0
            _atm_count = 0
            _atm_ce_bid_qty = 0
            _atm_ce_ask_qty = 0
            _atm_pe_bid_qty = 0
            _atm_pe_ask_qty = 0
            _concentration_max_ce = 0  # max single-strike CE OI change
            _concentration_max_pe = 0  # max single-strike PE OI change
            
            if strikes and spot > 0:
                # Sort by proximity to spot, take ±5 strikes
                by_dist = sorted(strikes, key=lambda s: abs(s['strike'] - spot))
                atm_zone = by_dist[:10]  # 5 ITM + 5 OTM effectively
                
                for s in atm_zone:
                    atm_ce_chg += s.get('ce_oi_change', 0)
                    atm_pe_chg += s.get('pe_oi_change', 0)
                    atm_ce_vol += s.get('ce_volume', 0)
                    atm_pe_vol += s.get('pe_volume', 0)
                    _atm_ce_bid_qty += s.get('ce_bid_qty', 0)
                    _atm_ce_ask_qty += s.get('ce_ask_qty', 0)
                    _atm_pe_bid_qty += s.get('pe_bid_qty', 0)
                    _atm_pe_ask_qty += s.get('pe_ask_qty', 0)
                    _concentration_max_ce = max(_concentration_max_ce, abs(s.get('ce_oi_change', 0)))
                    _concentration_max_pe = max(_concentration_max_pe, abs(s.get('pe_oi_change', 0)))
                
                # ATM IV skew (nearest strike)
                if atm_zone:
                    _atm = atm_zone[0]
                    _ce_iv = _atm.get('ce_iv', 0) or 0
                    _pe_iv = _atm.get('pe_iv', 0) or 0
                    if _ce_iv > 0 and _pe_iv > 0:
                        _atm_iv_skew = _pe_iv - _ce_iv  # positive = put IV > call IV = bearish
                    _atm_count = len(atm_zone)
            
            # Use ATM data if available, else fall back to totals
            _use_atm = _atm_count >= 4
            _eff_ce_chg = atm_ce_chg if _use_atm else call_oi_chg
            _eff_pe_chg = atm_pe_chg if _use_atm else put_oi_chg
            _eff_ce_vol = atm_ce_vol if _use_atm else call_vol
            _eff_pe_vol = atm_pe_vol if _use_atm else put_vol
            
            _eff_net = _eff_pe_chg - _eff_ce_chg
            _eff_total = abs(_eff_pe_chg) + abs(_eff_ce_chg)
            _eff_ratio = _eff_net / _eff_total if _eff_total > 0 else 0
            _eff_vol_ratio = _eff_pe_vol / _eff_ce_vol if _eff_ce_vol > 0 else 1.0
            
            # ── OI concentration check (single-strike dominance = noise) ──
            # If >80% of total OI change comes from one strike, it's likely
            # a large institutional hedge, not broad market positioning.
            _conc_ce = _concentration_max_ce / abs(_eff_ce_chg) if abs(_eff_ce_chg) > 0 else 0
            _conc_pe = _concentration_max_pe / abs(_eff_pe_chg) if abs(_eff_pe_chg) > 0 else 0
            _single_strike_noise = max(_conc_ce, _conc_pe) > 0.85
            
            # ── Price-direction from real-time spot vs ATM strike ──
            # Old logic inferred from daily premium changes — stale and misleading.
            # New: spot above ATM strike = UP, below = DOWN. Cross-check via CE/PE LTP.
            _price_dir = 0  # +1 = up, -1 = down, 0 = unclear
            if strikes and spot > 0:
                _atm_s = min(strikes, key=lambda s: abs(s['strike'] - spot))
                _atm_strike = _atm_s.get('strike', 0)
                _ce_ltp = _atm_s.get('ce_ltp', 0) or 0
                _pe_ltp = _atm_s.get('pe_ltp', 0) or 0
                if _atm_strike > 0:
                    _spot_vs_strike = spot - _atm_strike
                    # Primary: direct spot vs strike
                    if abs(_spot_vs_strike) > _atm_strike * 0.002:  # >0.2% away
                        _price_dir = 1 if _spot_vs_strike > 0 else -1
                    # Tiebreaker: CE vs PE LTP at ATM (put-call parity)
                    elif _ce_ltp > 0 and _pe_ltp > 0:
                        _price_dir = 1 if _ce_ltp > _pe_ltp else (-1 if _pe_ltp > _ce_ltp else 0)
            
            # ════════════════════════════════════════════════════════════
            # PREMIUM-OI CROSS-ANALYSIS — Writer vs Buyer Identification
            # ════════════════════════════════════════════════════════════
            # The golden rule: OI increase alone is ambiguous.
            #   OI ↑ + Premium ↓/flat = WRITERS adding (supply ↑, selling)
            #   OI ↑ + Premium ↑      = BUYERS adding (demand ↑, buying)
            # This matters because:
            #   PE OI ↑ by WRITERS = BULLISH (support being built)
            #   PE OI ↑ by BUYERS  = BEARISH (protection/hedging)
            #   CE OI ↑ by WRITERS = BEARISH (resistance being built)
            #   CE OI ↑ by BUYERS  = BULLISH (aggressive call buying)
            #
            # v4.1: Uses RECENT-WINDOW deltas (last 10 min) when timeseries
            # data is available, so premium changes reflect the same window
            # as the OI changes.  Falls back to whole-day (LTP - prev_close)
            # only on cold start (first ~10 min of tracking).
            # ────────────────────────────────────────────────────────────
            _ce_writer_oi = 0   # CE OI added by writers (premium ↓ while OI ↑)
            _ce_buyer_oi = 0    # CE OI added by buyers (premium ↑ while OI ↑)
            _pe_writer_oi = 0   # PE OI added by writers
            _pe_buyer_oi = 0    # PE OI added by buyers
            _participant_id = 'UNKNOWN'  # WRITER_DOMINANT | BUYER_DOMINANT | MIXED
            _recent_window_used = 0      # How many strikes used recent delta
            _daily_fallback_used = 0     # How many fell back to whole-day change
            
            if _use_atm and strikes and spot > 0:
                by_dist = sorted(strikes, key=lambda s: abs(s['strike'] - spot))
                _pid_zone = by_dist[:10]
                for _ps in _pid_zone:
                    _ps_ce_oi_chg = _ps.get('ce_oi_change', 0)
                    _ps_pe_oi_chg = _ps.get('pe_oi_change', 0)
                    _ps_ce_sid = str(_ps.get('ce_security_id', 0))
                    _ps_pe_sid = str(_ps.get('pe_security_id', 0))
                    
                    # ── CE side: try recent-window delta first ──
                    if _ps_ce_oi_chg > 0:
                        _ce_recent = self._compute_recent_delta(_ps_ce_sid) if _ps_ce_sid != '0' else None
                        if _ce_recent and _ce_recent['oi_delta'] > 0:
                            # Recent window has OI increasing — use recent premium delta
                            _ce_prem_pct = _ce_recent['premium_pct']
                            _recent_window_used += 1
                        else:
                            # Fallback: whole-day change
                            _ps_ce_prem_chg = _ps.get('ce_change', 0) or 0
                            _ps_ce_ltp = _ps.get('ce_ltp', 0) or 0
                            _ce_prem_pct = (_ps_ce_prem_chg / _ps_ce_ltp * 100) if _ps_ce_ltp > 1 else 0
                            _daily_fallback_used += 1
                        
                        if _ce_prem_pct <= -1.0:
                            _ce_writer_oi += _ps_ce_oi_chg
                        elif _ce_prem_pct >= 1.0:
                            _ce_buyer_oi += _ps_ce_oi_chg
                    
                    # ── PE side: try recent-window delta first ──
                    if _ps_pe_oi_chg > 0:
                        _pe_recent = self._compute_recent_delta(_ps_pe_sid) if _ps_pe_sid != '0' else None
                        if _pe_recent and _pe_recent['oi_delta'] > 0:
                            _pe_prem_pct = _pe_recent['premium_pct']
                            _recent_window_used += 1
                        else:
                            _ps_pe_prem_chg = _ps.get('pe_change', 0) or 0
                            _ps_pe_ltp = _ps.get('pe_ltp', 0) or 0
                            _pe_prem_pct = (_ps_pe_prem_chg / _ps_pe_ltp * 100) if _ps_pe_ltp > 1 else 0
                            _daily_fallback_used += 1
                        
                        if _pe_prem_pct <= -1.0:
                            _pe_writer_oi += _ps_pe_oi_chg
                        elif _pe_prem_pct >= 1.0:
                            _pe_buyer_oi += _ps_pe_oi_chg
                
                # Determine dominant participant for the ACTIVE side
                _total_writer = _ce_writer_oi + _pe_writer_oi
                _total_buyer = _ce_buyer_oi + _pe_buyer_oi
                _total_classified = _total_writer + _total_buyer
                if _total_classified > 0:
                    _writer_pct = _total_writer / _total_classified
                    if _writer_pct >= 0.65:
                        _participant_id = 'WRITER_DOMINANT'
                    elif _writer_pct <= 0.35:
                        _participant_id = 'BUYER_DOMINANT'
                    else:
                        _participant_id = 'MIXED'
            
            # Store for return in parent dict
            self._last_participant_id = _participant_id
            self._last_participant_detail = {
                'ce_writer_oi': _ce_writer_oi, 'ce_buyer_oi': _ce_buyer_oi,
                'pe_writer_oi': _pe_writer_oi, 'pe_buyer_oi': _pe_buyer_oi,
                'participant_id': _participant_id,
                'recent_window_used': _recent_window_used,
                'daily_fallback_used': _daily_fallback_used,
            }
            
            # ── Bid/Ask writing heuristic ──
            # Large ask_qty on CEs = writers selling CEs = bearish positioning
            # Large bid_qty on PEs = writers selling PEs = bullish support
            _writer_bias = 0  # +1 = writers bullish, -1 = writers bearish
            if _atm_ce_ask_qty > 0 and _atm_pe_ask_qty > 0:
                _ce_ba_ratio = _atm_ce_ask_qty / max(_atm_ce_bid_qty, 1)
                _pe_ba_ratio = _atm_pe_ask_qty / max(_atm_pe_bid_qty, 1)
                # CE ask >> bid = heavy CE writing = bearish
                if _ce_ba_ratio > 1.5 and _pe_ba_ratio < 1.2:
                    _writer_bias = -1
                # PE ask >> bid = heavy PE writing = bullish support
                elif _pe_ba_ratio > 1.5 and _ce_ba_ratio < 1.2:
                    _writer_bias = 1
            
            # ── IV skew confirmation ──
            # Positive skew (put IV > call IV) = bearish fear = confirms SHORT_BUILDUP
            # Negative skew (call IV > put IV) = bullish greed = confirms LONG_BUILDUP
            _iv_confirms_bull = _atm_iv_skew < -2.0
            _iv_confirms_bear = _atm_iv_skew > 2.0
            
            # ── Market Quote enrichment (v3) ──
            # 7) OI at day high = fresh intraday buildup happening NOW
            # 8) Exchange buy/sell queue imbalance
            # 9) OI velocity (intraday rate-of-change)
            _fresh_ce_buildup = False  # CE OI at intraday highs
            _fresh_pe_buildup = False  # PE OI at intraday highs
            _exchange_pressure = 0     # +1 = buy pressure, -1 = sell pressure
            _oi_velocity_bull = False   # PE OI accelerating (bullish)
            _oi_velocity_bear = False   # CE OI accelerating (bearish)
            
            if quote_enrichment:
                _ce_dh = quote_enrichment.get('ce_at_day_high_count', 0)
                _pe_dh = quote_enrichment.get('pe_at_day_high_count', 0)
                # If 3+ ATM PE strikes have OI at day high → fresh put writing → bullish
                _fresh_pe_buildup = _pe_dh >= 3
                # If 3+ ATM CE strikes have OI at day high → fresh call writing → bearish
                _fresh_ce_buildup = _ce_dh >= 3
                
                # Exchange aggregate buy vs sell queue
                _buy_q = quote_enrichment.get('atm_buy_qty', 0)
                _sell_q = quote_enrichment.get('atm_sell_qty', 0)
                if _buy_q > 0 and _sell_q > 0:
                    _bs_ratio = _buy_q / _sell_q
                    if _bs_ratio > 1.5:
                        _exchange_pressure = 1   # heavy buy queue = demand
                    elif _bs_ratio < 0.67:
                        _exchange_pressure = -1  # heavy sell queue = supply
                
                # OI velocity from timeseries
                _ce_vel = quote_enrichment.get('ce_oi_velocity')
                _pe_vel = quote_enrichment.get('pe_oi_velocity')
                # PE OI accelerating faster than CE = put writing increasing = bullish
                if _pe_vel is not None and _ce_vel is not None:
                    if _pe_vel > 0.02 and _pe_vel > _ce_vel:
                        _oi_velocity_bull = True
                    elif _ce_vel > 0.02 and _ce_vel > _pe_vel:
                        _oi_velocity_bear = True
            
            # ════════════════════════════════════════════════════════════
            # SIGNAL CLASSIFICATION — using ATM-zone effective values
            # ════════════════════════════════════════════════════════════
            signal = 'NEUTRAL'
            strength = 0.0
            _THRESHOLD = 0.25  # lowered from 0.3 for ATM-only data (less noise)
            
            if _eff_pe_chg > 0 and _eff_ratio > _THRESHOLD:
                # Put OI increasing more than call OI
                # Classic assumption: put WRITERS adding → support → bullish
                # BUT if PE BUYERS dominate → hedging/fear → actually bearish!
                signal = 'LONG_BUILDUP'
                _base = abs(_eff_ratio) * 0.6
                
                # Writer vs Buyer check on PE side
                _pe_is_buyer_driven = (_pe_buyer_oi > _pe_writer_oi * 1.5) and _pe_buyer_oi > 0
                _pe_is_writer_confirmed = (_pe_writer_oi > _pe_buyer_oi * 1.5) and _pe_writer_oi > 0
                
                if _pe_is_buyer_driven:
                    # PE OI rising because BUYERS are buying puts = bearish hedge
                    # Flip signal: this is NOT long buildup, it's put buying
                    signal = 'SHORT_BUILDUP'
                    _base *= 0.80  # Lower base — buyer-driven OI is less sticky
                elif _pe_is_writer_confirmed:
                    # PE OI rising because WRITERS are selling puts = genuine support
                    _base += 0.12  # Writer-confirmed boost
                
                # Confirmations boost strength
                if _eff_vol_ratio > 0.8:    _base += 0.10  # volume confirms
                if _iv_confirms_bull:        _base += 0.10  # IV skew confirms
                if _price_dir == 1:          _base += 0.10  # price moving up confirms
                if _writer_bias == 1:        _base += 0.08  # bid/ask shows writing
                if _fresh_pe_buildup:        _base += 0.12  # PE OI at day high = fresh!
                if _oi_velocity_bull:        _base += 0.08  # PE OI accelerating
                if _exchange_pressure == 1:  _base += 0.06  # buy queue dominance
                if _single_strike_noise:     _base -= 0.15  # concentrated = less reliable
                strength = min(1.0, max(0.1, _base))
            
            elif _eff_ce_chg > 0 and _eff_ratio < -_THRESHOLD:
                # Call OI increasing more than put OI
                # Classic assumption: call WRITERS adding → resistance → bearish
                # BUT if CE BUYERS dominate → aggressive call buying → bullish!
                signal = 'SHORT_BUILDUP'
                _base = abs(_eff_ratio) * 0.6
                
                # Writer vs Buyer check on CE side
                _ce_is_buyer_driven = (_ce_buyer_oi > _ce_writer_oi * 1.5) and _ce_buyer_oi > 0
                _ce_is_writer_confirmed = (_ce_writer_oi > _ce_buyer_oi * 1.5) and _ce_writer_oi > 0
                
                if _ce_is_buyer_driven:
                    # CE OI rising because BUYERS are buying calls = bullish
                    signal = 'LONG_BUILDUP'
                    _base *= 0.80  # Lower base — buyer-driven OI is less sticky
                elif _ce_is_writer_confirmed:
                    # CE OI rising because WRITERS selling calls = genuine resistance
                    _base += 0.12  # Writer-confirmed boost
                
                if _eff_vol_ratio < 1.2:     _base += 0.10
                if _iv_confirms_bear:         _base += 0.10
                if _price_dir == -1:          _base += 0.10
                if _writer_bias == -1:        _base += 0.08
                if _fresh_ce_buildup:         _base += 0.12  # CE OI at day high = fresh!
                if _oi_velocity_bear:         _base += 0.08  # CE OI accelerating
                if _exchange_pressure == -1:  _base += 0.06  # sell queue dominance
                if _single_strike_noise:      _base -= 0.15
                strength = min(1.0, max(0.1, _base))
            
            elif _eff_ce_chg < 0 and abs(_eff_ce_chg) > abs(_eff_pe_chg):
                # Call OI decreasing = call writers exiting = short covering
                signal = 'SHORT_COVERING'
                _base = abs(_eff_ratio) * 0.5
                if _price_dir == 1:           _base += 0.10
                if _iv_confirms_bull:         _base += 0.08
                if _exchange_pressure == 1:   _base += 0.06
                strength = min(0.85, max(0.1, _base))
            
            elif _eff_pe_chg < 0 and abs(_eff_pe_chg) > abs(_eff_ce_chg):
                # Put OI decreasing = put writers exiting = long unwinding
                signal = 'LONG_UNWINDING'
                _base = abs(_eff_ratio) * 0.5
                if _price_dir == -1:          _base += 0.10
                if _iv_confirms_bear:         _base += 0.08
                if _exchange_pressure == -1:  _base += 0.06
                strength = min(0.85, max(0.1, _base))
            
            # ── Expiry-week noise guard ──
            # Near expiry, OI changes are dominated by rollover/decay.
            # Reduce strength by 30% if within 2 days of expiry.
            try:
                from datetime import datetime as _dt
                _today = _dt.now().date()
                _day_of_week = _today.weekday()  # Thu=3
                # Weekly expiry is Thursday. If today is Wed(2) or Thu(3), reduce
                if _day_of_week in (2, 3):
                    strength *= 0.7
            except Exception:
                pass
            
            # ── Cross-validate: price direction vs OI signal ──
            # If price is clearly moving opposite to OI signal, downgrade
            if signal == 'LONG_BUILDUP' and _price_dir == -1:
                strength *= 0.7  # bullish OI but price falling — less reliable
            elif signal == 'SHORT_BUILDUP' and _price_dir == 1:
                strength *= 0.7  # bearish OI but price rising — less reliable
            
            # ── Participant confidence modifier (v4) ──
            # If participant analysis is conclusive, adjust final strength
            if _participant_id == 'WRITER_DOMINANT':
                strength *= 1.10  # Writers = stickier positions, higher conviction
            elif _participant_id == 'BUYER_DOMINANT':
                strength *= 0.85  # Buyers = can exit quickly, lower conviction
            # MIXED / UNKNOWN = no adjustment
            
            return signal, round(min(1.0, strength), 3)
            
        except Exception:
            return 'NEUTRAL', 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # MARKET QUOTE API — oi_day_high/low, buy/sell qty, 5-level depth
    # Endpoint: POST /v2/marketfeed/quote
    # Rate: 1 req/sec, up to 1000 instruments per request
    # ═══════════════════════════════════════════════════════════════════
    
    def fetch_market_quotes(self, security_ids: List[int], segment: str = "NSE_FNO") -> Dict[int, dict]:
        """Batch-fetch real-time market quotes for multiple security IDs.
        
        Returns OI day high/low, buy/sell quantities, 5-level depth, volume.
        """
        if not self.ready or not security_ids:
            return {}
        
        try:
            # Cache check
            _ids_key = ','.join(str(s) for s in sorted(security_ids[:50]))
            cache_key = f"quote:{segment}:{_ids_key}"
            cached = self._quote_cache.get(cache_key)
            if cached:
                ts, data = cached
                if time.time() - ts < self._QUOTE_CACHE_TTL:
                    return data
            
            self._throttle_quote()
            
            payload = {segment: [int(sid) for sid in security_ids[:1000]]}
            
            r = requests.post(self.QUOTE_URL, headers=self._headers(),
                            json=payload, timeout=10)
            
            if r.status_code != 200:
                logger.debug(f"DhanOI: Market quote HTTP {r.status_code}: {r.text[:100]}")
                return {}
            
            raw = r.json()
            if raw.get('status') != 'success':
                return {}
            
            seg_data = raw.get('data', {}).get(segment, {})
            result = {}
            
            for sid_str, q in seg_data.items():
                sid = int(sid_str)
                oi = q.get('oi', 0) or 0
                result[sid] = {
                    'oi': oi,
                    'oi_day_high': q.get('oi_day_high', 0) or 0,
                    'oi_day_low': q.get('oi_day_low', 0) or 0,
                    'buy_quantity': q.get('buy_quantity', 0) or 0,
                    'sell_quantity': q.get('sell_quantity', 0) or 0,
                    'volume': q.get('volume', 0) or 0,
                    'last_price': q.get('last_price', 0) or 0,
                    'net_change': q.get('net_change', 0) or 0,
                    'depth': q.get('depth', {}),
                    'ohlc': q.get('ohlc', {}),
                }
                
                # Record OI + premium snapshot for timeseries
                if oi > 0:
                    _ltp = q.get('last_price', 0) or 0
                    self._record_oi_snapshot(str(sid), oi, float(_ltp))
            
            if result:
                self._quote_cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception as e:
            logger.debug(f"DhanOI: Market quote error: {e}")
            return {}
    
    def _record_oi_snapshot(self, security_id: str, oi: int, premium: float = 0.0):
        """Store OI + premium value in timeseries for intraday analysis.
        
        Each entry is (timestamp, oi, premium).  Premium is used for
        recent-window writer vs buyer identification — avoids relying
        on whole-day change which can lag actual OI buildup moment.
        """
        now = time.time()
        if security_id not in self._oi_timeseries:
            self._oi_timeseries[security_id] = []
        
        series = self._oi_timeseries[security_id]
        series.append((now, oi, premium))
        
        # Prune entries older than max age
        cutoff = now - self._OI_TIMESERIES_MAX_AGE
        self._oi_timeseries[security_id] = [e for e in series if e[0] >= cutoff]
    
    def _compute_oi_velocity(self, security_id: str, lookback_seconds: int = 900) -> Optional[float]:
        """Compute OI rate-of-change over the lookback window.
        
        Returns:
            Fractional change (e.g., 0.05 = 5% increase) or None if insufficient data.
        """
        series = self._oi_timeseries.get(str(security_id), [])
        if len(series) < 2:
            return None
        
        now = time.time()
        cutoff = now - lookback_seconds
        
        old_entries = [e for e in series if e[0] >= cutoff]
        if len(old_entries) < 2:
            return None
        
        old_oi = old_entries[0][1]
        new_oi = old_entries[-1][1]
        
        if old_oi == 0:
            return None
        
        return (new_oi - old_oi) / old_oi
    
    def _compute_recent_delta(self, security_id: str, lookback_seconds: int = 600) -> Optional[dict]:
        """Compute RECENT OI and premium deltas over a short lookback window.
        
        This solves the premium-lag problem: instead of using LTP - prev_close
        (which reflects the ENTIRE day), we compare the last two snapshots
        to see what happened IN THE SAME WINDOW as the OI change.
        
        Args:
            security_id: DhanHQ security ID (string)
            lookback_seconds: window size (default 600s = 10 min)
            
        Returns:
            dict with {oi_delta, premium_delta, oi_pct, premium_pct,
                       old_oi, new_oi, old_premium, new_premium}
            or None if insufficient data (need >= 2 entries with premium > 0).
        """
        series = self._oi_timeseries.get(str(security_id), [])
        if len(series) < 2:
            return None
        
        now = time.time()
        cutoff = now - lookback_seconds
        
        # Only entries in the recent window that have premium data
        recent = [e for e in series if e[0] >= cutoff and len(e) >= 3 and e[2] > 0]
        if len(recent) < 2:
            return None
        
        old_ts, old_oi, old_prem = recent[0]
        new_ts, new_oi, new_prem = recent[-1]
        
        # Need meaningful time gap (at least 60 seconds)
        if (new_ts - old_ts) < 60:
            return None
        
        oi_delta = new_oi - old_oi
        prem_delta = new_prem - old_prem
        oi_pct = (oi_delta / old_oi * 100) if old_oi > 0 else 0.0
        prem_pct = (prem_delta / old_prem * 100) if old_prem > 0 else 0.0
        
        return {
            'oi_delta': oi_delta,
            'premium_delta': round(prem_delta, 2),
            'oi_pct': round(oi_pct, 2),
            'premium_pct': round(prem_pct, 2),
            'old_oi': old_oi,
            'new_oi': new_oi,
            'old_premium': round(old_prem, 2),
            'new_premium': round(new_prem, 2),
            'window_secs': round(new_ts - old_ts),
        }

    def fetch_batch(self, symbols: List[str], max_symbols: int = 15) -> Dict[str, dict]:
        """Fetch OI data for multiple symbols.
        
        Respects rate limits (3s gap per request).
        """
        results = {}
        for sym in symbols[:max_symbols]:
            data = self.fetch(sym)
            if data:
                results[sym] = data
        return results
    
    def get_snapshot_for_logging(self, symbol: str) -> dict:
        """Compact snapshot for JSONL logging (strips full strike data)."""
        try:
            data = self.fetch(symbol)
            if not data:
                return {}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'source': 'DHAN',
                'symbol': data.get('symbol', symbol),
                'spot_price': data.get('spot_price'),
                'expiry': data.get('expiry'),
                
                'total_call_oi': data.get('total_call_oi'),
                'total_put_oi': data.get('total_put_oi'),
                'total_call_oi_change': data.get('total_call_oi_change'),
                'total_put_oi_change': data.get('total_put_oi_change'),
                'total_call_volume': data.get('total_call_volume'),
                'total_put_volume': data.get('total_put_volume'),
                
                'pcr_oi': data.get('pcr_oi'),
                'pcr_oi_change': data.get('pcr_oi_change'),
                'pcr_volume': data.get('pcr_volume'),
                
                'max_pain': data.get('max_pain'),
                'call_resistance': data.get('call_resistance'),
                'put_support': data.get('put_support'),
                'iv_skew': data.get('iv_skew'),
                
                'top3_call_oi': data.get('top_call_oi_strikes', [])[:3],
                'top3_put_oi': data.get('top_put_oi_strikes', [])[:3],
                'top3_call_oi_change': data.get('top_call_oi_change_strikes', [])[:3],
                'top3_put_oi_change': data.get('top_put_oi_change_strikes', [])[:3],
                
                'oi_buildup_signal': data.get('oi_buildup_signal'),
                'oi_buildup_strength': data.get('oi_buildup_strength'),
                
                # DhanHQ exclusive: ATM Greeks
                'atm_greeks': data.get('atm_greeks', {}),
            }
        except Exception:
            return {}
    
    def to_flow_analyzer_format(self, symbol: str) -> dict:
        """Convert DhanHQ data to OptionsFlowAnalyzer-compatible format.
        
        Drop-in compatible with apply_oi_overlay() and NSE fetcher's format.
        """
        try:
            data = self.fetch(symbol)
            if not data:
                return self._neutral_flow()
            
            pcr = data.get('pcr_oi', 1.0)
            iv_skew = data.get('iv_skew', 0.0)
            max_pain = data.get('max_pain', 0.0)
            buildup = data.get('oi_buildup_signal', 'NEUTRAL')
            strength = data.get('oi_buildup_strength', 0.0)
            spot = data.get('spot_price', 0.0)
            
            bias = 'NEUTRAL'
            confidence = 0.5
            boost = 0
            
            # Primary: OI buildup pattern
            if buildup == 'LONG_BUILDUP':
                bias = 'BULLISH'
                confidence = 0.5 + strength * 0.3
                boost = min(4, int(strength * 5))
            elif buildup == 'SHORT_BUILDUP':
                bias = 'BEARISH'
                confidence = 0.5 + strength * 0.3
                boost = max(-4, -int(strength * 5))
            elif buildup == 'SHORT_COVERING':
                bias = 'BULLISH'
                confidence = 0.5 + strength * 0.2
                boost = min(3, int(strength * 4))
            elif buildup == 'LONG_UNWINDING':
                bias = 'BEARISH'
                confidence = 0.5 + strength * 0.2
                boost = max(-3, -int(strength * 4))
            
            # Secondary: PCR
            if bias == 'NEUTRAL':
                if pcr >= 1.5:
                    bias = 'BULLISH'
                    confidence = min(0.75, 0.5 + (pcr - 1.0) * 0.12)
                    boost = 3
                elif pcr >= 1.2:
                    bias = 'BULLISH'
                    confidence = 0.58
                    boost = 2
                elif pcr <= 0.5:
                    bias = 'BEARISH'
                    confidence = min(0.75, 0.5 + (1.0 - pcr) * 0.12)
                    boost = -3
                elif pcr <= 0.7:
                    bias = 'BEARISH'
                    confidence = 0.58
                    boost = -2
            
            # IV skew adjustment
            if iv_skew > 5:
                if bias == 'NEUTRAL':
                    bias = 'BEARISH'
                confidence = min(0.85, confidence + 0.08)
                boost = max(-4, boost - 1)
            elif iv_skew < -5:
                if bias == 'NEUTRAL':
                    bias = 'BULLISH'
                confidence = min(0.85, confidence + 0.08)
                boost = min(4, boost + 1)
            
            boost = max(-4, min(4, boost))
            
            # GPT line
            gpt_parts = [f"PCR:{pcr:.2f}"]
            if abs(iv_skew) > 2:
                gpt_parts.append(f"IVskew:{iv_skew:+.1f}%")
            if max_pain > 0:
                gpt_parts.append(f"MaxPain:{max_pain:.0f}")
            
            # ATM Greeks for GPT
            atm_g = data.get('atm_greeks', {})
            if atm_g.get('ce_delta'):
                gpt_parts.append(f"ATMδ:{atm_g['ce_delta']:.2f}/{atm_g.get('pe_delta', 0):.2f}")
            if atm_g.get('ce_vega'):
                gpt_parts.append(f"Vega:{atm_g['ce_vega']:.1f}")
            
            gpt_line = f"📊OI[Dhan]:{bias}({','.join(gpt_parts)})"
            
            if buildup != 'NEUTRAL':
                gpt_line += f"|{buildup}({strength:.0%})"
            
            ce_chg = data.get('total_call_oi_change', 0)
            pe_chg = data.get('total_put_oi_change', 0)
            if ce_chg != 0 or pe_chg != 0:
                gpt_line += f"|ΔOI:CE{ce_chg:+,}/PE{pe_chg:+,}"
            
            # Market Quote enrichment indicators
            _qe = data.get('quote_enrichment', {})
            if _qe:
                _tags = []
                _ce_dh = _qe.get('ce_at_day_high_count', 0)
                _pe_dh = _qe.get('pe_at_day_high_count', 0)
                if _pe_dh >= 3:
                    _tags.append(f"PE@DH:{_pe_dh}")
                if _ce_dh >= 3:
                    _tags.append(f"CE@DH:{_ce_dh}")
                _buy_q = _qe.get('atm_buy_qty', 0)
                _sell_q = _qe.get('atm_sell_qty', 0)
                if _buy_q > 0 and _sell_q > 0:
                    _tags.append(f"B/S:{_buy_q}/{_sell_q}")
                if _tags:
                    gpt_line += f"|MQ:{'|'.join(_tags)}"
            
            # Participant identification (v4.1: writer vs buyer with recent-window)
            _pid = data.get('oi_participant_id', 'UNKNOWN')
            _pid_detail = data.get('oi_participant_detail', {})
            if _pid != 'UNKNOWN':
                _cw = _pid_detail.get('ce_writer_oi', 0)
                _cb = _pid_detail.get('ce_buyer_oi', 0)
                _pw = _pid_detail.get('pe_writer_oi', 0)
                _pb = _pid_detail.get('pe_buyer_oi', 0)
                _rw = _pid_detail.get('recent_window_used', 0)
                _df = _pid_detail.get('daily_fallback_used', 0)
                _src_tag = f"R{_rw}" if _rw > 0 else f"D{_df}"
                gpt_line += f"|PID:{_pid}(CW:{_cw:,}/CB:{_cb:,}/PW:{_pw:,}/PB:{_pb:,}|{_src_tag})"
            
            return {
                'flow_bias': bias,
                'flow_confidence': round(confidence, 3),
                'pcr_oi': round(pcr, 3),
                'iv_skew': round(iv_skew, 2),
                'max_pain': max_pain,
                'call_resistance': data.get('call_resistance', 0),
                'put_support': data.get('put_support', 0),
                'spot_price': spot,
                'flow_score_boost': boost,
                'flow_gpt_line': gpt_line,
                'source': 'DHAN',
                'atm_greeks': atm_g,
                # OI change fields (same as NSE enrichment)
                'nse_total_call_oi_change': ce_chg,
                'nse_total_put_oi_change': pe_chg,
                'nse_pcr_volume': data.get('pcr_volume', 1.0),
                'nse_oi_buildup': buildup,
                'nse_oi_buildup_strength': strength,
                'nse_enriched': True,
                'oi_participant_id': _pid,
                'oi_participant_detail': _pid_detail,
            }
            
        except Exception:
            return self._neutral_flow()
    
    def _neutral_flow(self) -> dict:
        """Default neutral result for flow analyzer format."""
        return {
            'flow_bias': 'NEUTRAL',
            'flow_confidence': 0.5,
            'pcr_oi': 1.0,
            'iv_skew': 0.0,
            'max_pain': 0.0,
            'call_resistance': 0,
            'put_support': 0,
            'spot_price': 0,
            'flow_score_boost': 0,
            'flow_gpt_line': '📊OI:NEUTRAL(no data)',
            'source': 'NONE',
        }


# ─── Singleton ──────────────────────────────────────────────────────

_instance: Optional[DhanOIFetcher] = None
_instance_lock = threading.Lock()


def get_dhan_oi_fetcher(client_id: str = None, access_token: str = None) -> DhanOIFetcher:
    """Get or create singleton DhanOIFetcher.
    
    First call initializes with credentials.
    Subsequent calls return same instance.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = DhanOIFetcher(client_id, access_token)
        return _instance


# ─── CLI Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    
    fetcher = get_dhan_oi_fetcher()
    
    if not fetcher.ready:
        print("❌ No DhanHQ credentials. Run: save_credentials(client_id, access_token)")
        sys.exit(1)
    
    print(f"✅ DhanOIFetcher ready (client_id: {fetcher.client_id[:4]}...)")
    
    # Check data plan
    dp = fetcher.check_data_plan()
    print(f"📊 Data Plan: {'ACTIVE ✓' if dp else 'INACTIVE ✗ — subscribe at web.dhan.co'}")
    
    if not dp:
        print("\n⚠ Option Chain API requires Data Plan subscription.")
        print("  Go to web.dhan.co → Data API Plans → Subscribe")
        sys.exit(0)
    
    # Test with NIFTY
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    print(f"\n🔍 Fetching option chain for {symbol}...")
    
    data = fetcher.fetch(symbol)
    if data:
        print(f"  Spot: {data.get('spot_price')}")
        print(f"  Expiry: {data.get('expiry')}")
        print(f"  CE OI: {data.get('total_call_oi'):,}")
        print(f"  PE OI: {data.get('total_put_oi'):,}")
        print(f"  CE OI Δ: {data.get('total_call_oi_change'):+,}")
        print(f"  PE OI Δ: {data.get('total_put_oi_change'):+,}")
        print(f"  PCR: {data.get('pcr_oi')}")
        print(f"  Max Pain: {data.get('max_pain')}")
        print(f"  IV Skew: {data.get('iv_skew')}")
        print(f"  Buildup: {data.get('oi_buildup_signal')} ({data.get('oi_buildup_strength'):.0%})")
        print(f"  ATM Greeks: {data.get('atm_greeks')}")
        print(f"  # Strikes: {len(data.get('strikes', []))}")
    else:
        print("  ❌ No data returned")
