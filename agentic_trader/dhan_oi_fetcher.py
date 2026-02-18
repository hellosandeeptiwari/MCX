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
    'LTIM':        {'scrip_id': 17818, 'segment': 'NSE_FNO'},
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
    
    # Rate limit: 1 unique request per 3 seconds
    MIN_REQUEST_GAP = 3.2  # slightly over to be safe
    
    # Cache TTL: 3 minutes (OI updates slowly)
    CACHE_TTL = 180
    
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
        self._lock = threading.Lock()
        
        # Cache: {key: (timestamp, data)}
        self._cache: Dict[str, Tuple[float, dict]] = {}
        
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
        Returns True if renewal successful.
        """
        try:
            r = requests.post(self.RENEW_URL, headers=self._headers(), timeout=10)
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
                    logger.warning("DhanOI: Token expired/invalid")
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
            
            # OI buildup signal
            buildup_signal, buildup_strength = self._detect_oi_buildup(
                total_call_oi_change, total_put_oi_change,
                total_call_volume, total_put_volume,
                spot_price, strikes
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
                
                # Greeks (ATM summary)
                'atm_greeks': atm_greeks,
                
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
                           spot: float, strikes: list) -> Tuple[str, float]:
        """Detect OI buildup pattern.
        
        Same logic as NSE fetcher for consistency:
        - LONG_BUILDUP: Put OI increasing, support building → bullish
        - SHORT_BUILDUP: Call OI increasing, resistance building → bearish
        - SHORT_COVERING: Call OI decreasing → bullish
        - LONG_UNWINDING: Put OI decreasing → bearish
        """
        try:
            net_oi_change = put_oi_chg - call_oi_chg
            total_oi_change = abs(put_oi_chg) + abs(call_oi_chg)
            
            if total_oi_change == 0:
                return 'NEUTRAL', 0.0
            
            oi_ratio = net_oi_change / total_oi_change if total_oi_change > 0 else 0
            vol_ratio = put_vol / call_vol if call_vol > 0 else 1.0
            
            signal = 'NEUTRAL'
            strength = 0.0
            
            if put_oi_chg > 0 and oi_ratio > 0.3:
                if vol_ratio > 0.8:
                    signal = 'LONG_BUILDUP'
                    strength = min(1.0, abs(oi_ratio) * 0.8 + 0.2)
                else:
                    signal = 'LONG_BUILDUP'
                    strength = min(0.7, abs(oi_ratio) * 0.6)
            
            elif call_oi_chg > 0 and oi_ratio < -0.3:
                if vol_ratio < 1.2:
                    signal = 'SHORT_BUILDUP'
                    strength = min(1.0, abs(oi_ratio) * 0.8 + 0.2)
                else:
                    signal = 'SHORT_BUILDUP'
                    strength = min(0.7, abs(oi_ratio) * 0.6)
            
            elif call_oi_chg < 0 and abs(call_oi_chg) > abs(put_oi_chg):
                signal = 'SHORT_COVERING'
                strength = min(0.8, abs(oi_ratio) * 0.7)
            
            elif put_oi_chg < 0 and abs(put_oi_chg) > abs(call_oi_chg):
                signal = 'LONG_UNWINDING'
                strength = min(0.8, abs(oi_ratio) * 0.7)
            
            return signal, round(strength, 3)
            
        except Exception:
            return 'NEUTRAL', 0.0
    
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
