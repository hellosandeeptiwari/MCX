"""
NSE OI FETCHER — Strike-wise OI + OI Change from NSE Public API

Pulls richer OI data than Kite provides:
  - OI change per strike (fresh buildup/unwinding detection)
  - Volume per strike (call/put)
  - Total market-wide OI aggregates
  - Max pain, PCR, IV skew — all from NSE directly

NSE API endpoints:
  /api/option-chain-equities?symbol=SBIN   → Stock option chains
  /api/option-chain-indices?symbol=NIFTY   → Index option chains

Anti-blocking measures:
  - Session cookie obtained from homepage first
  - Browser-like headers
  - 1.5s minimum gap between requests
  - Aggressive caching (3-min TTL)
  - Retry with backoff on 403/429

FAIL-SAFE: Returns empty/neutral on ANY error. Never blocks Titan.

Usage:
    from nse_oi_fetcher import NSEOIFetcher
    
    fetcher = NSEOIFetcher()
    data = fetcher.fetch("SBIN")
    # data = {
    #   'symbol': 'SBIN',
    #   'spot_price': 785.50,
    #   'timestamp': '2026-02-15T10:30:00',
    #   'total_call_oi': 12500000,
    #   'total_put_oi': 9800000,
    #   'total_call_oi_change': 250000,
    #   'total_put_oi_change': -180000,
    #   'pcr_oi': 0.784,
    #   'pcr_oi_change': ...,
    #   'pcr_volume': ...,
    #   'max_pain': 780.0,
    #   'strikes': [
    #     {'strike': 780, 'ce_oi': 500000, 'ce_oi_change': 25000, 'ce_volume': ...,
    #      'ce_iv': 22.5, 'pe_oi': ..., 'pe_oi_change': ..., 'pe_volume': ..., 'pe_iv': ...},
    #     ...
    #   ],
    #   'top_call_oi_strikes': [(800, 1200000), (820, 900000), ...],
    #   'top_put_oi_strikes': [(760, 1100000), (740, 800000), ...],
    #   'oi_buildup_signal': 'LONG_BUILDUP' | 'SHORT_BUILDUP' | 'LONG_UNWINDING' | 'SHORT_COVERING' | 'NEUTRAL',
    #   'oi_buildup_strength': 0.0 to 1.0,
    # }
"""

import time
import requests
import threading
from datetime import datetime
from typing import Dict, Optional, List, Tuple


class NSEOIFetcher:
    """Fetch option chain OI data from NSE public API.
    
    FAIL-SAFE: Returns empty dict on any error. Never blocks.
    Thread-safe with session locking.
    """
    
    BASE_URL = "https://www.nseindia.com"
    EQUITY_CHAIN_URL = f"{BASE_URL}/api/option-chain-equities"
    INDEX_CHAIN_URL = f"{BASE_URL}/api/option-chain-indices"
    
    # Index symbols that use the index endpoint
    INDEX_SYMBOLS = {"NIFTY", "NIFTY 50", "BANKNIFTY", "NIFTY BANK", "FINNIFTY", "MIDCPNIFTY"}
    
    # Browser-like headers to avoid blocks
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }
    
    CACHE_TTL = 180  # 3 minutes (NSE updates every ~3 min during market hours)
    MIN_REQUEST_GAP = 1.5  # seconds between requests
    MAX_RETRIES = 2
    
    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()
        self._last_request_time = 0.0
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._session_valid = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5  # Disable after 5 consecutive failures
        self.ready = True
        
    def _init_session(self) -> bool:
        """Initialize session by hitting NSE homepage to get cookies.
        
        Returns True if session is ready.
        """
        try:
            with self._session_lock:
                if self._session_valid and self._session:
                    return True
                    
                self._session = requests.Session()
                self._session.headers.update(self.HEADERS)
                
                # Hit homepage to get session cookies
                resp = self._session.get(
                    self.BASE_URL,
                    timeout=10,
                    allow_redirects=True
                )
                
                if resp.status_code == 200:
                    self._session_valid = True
                    self._consecutive_failures = 0
                    return True
                else:
                    self._session_valid = False
                    return False
                    
        except Exception:
            self._session_valid = False
            return False
    
    def _throttle(self):
        """Enforce minimum gap between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_GAP:
            time.sleep(self.MIN_REQUEST_GAP - elapsed)
        self._last_request_time = time.time()
    
    def _get_raw_chain(self, symbol: str) -> Optional[dict]:
        """Fetch raw option chain JSON from NSE.
        
        Returns None on failure.
        """
        if self._consecutive_failures >= self._max_consecutive_failures:
            return None  # Circuit breaker
            
        # Determine endpoint
        clean_symbol = symbol.replace("NSE:", "").replace("NFO:", "").strip()
        
        if clean_symbol.upper() in self.INDEX_SYMBOLS:
            url = self.INDEX_CHAIN_URL
            # Map our symbol names to NSE's
            nse_symbol = clean_symbol.upper()
            if nse_symbol == "NIFTY 50":
                nse_symbol = "NIFTY"
            elif nse_symbol == "NIFTY BANK":
                nse_symbol = "BANKNIFTY"
        else:
            url = self.EQUITY_CHAIN_URL
            nse_symbol = clean_symbol.upper()
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if not self._session_valid or not self._session:
                    if not self._init_session():
                        continue
                
                self._throttle()
                
                resp = self._session.get(
                    url,
                    params={"symbol": nse_symbol},
                    timeout=10,
                    allow_redirects=True
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    if 'records' in data and data['records'].get('data'):
                        self._consecutive_failures = 0
                        return data
                    else:
                        # Empty response (weekend/holiday) — NOT a failure
                        return None
                        
                elif resp.status_code in (401, 403):
                    # Session expired, re-init
                    self._session_valid = False
                    self._session = None
                    if attempt < self.MAX_RETRIES:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        self._consecutive_failures += 1
                        return None
                        
                elif resp.status_code == 429:
                    # Rate limited
                    if attempt < self.MAX_RETRIES:
                        time.sleep(5 * (attempt + 1))
                        continue
                    else:
                        self._consecutive_failures += 1
                        return None
                else:
                    self._consecutive_failures += 1
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES:
                    time.sleep(2)
                    continue
                self._consecutive_failures += 1
                return None
            except Exception:
                self._consecutive_failures += 1
                return None
        
        return None
    
    def fetch(self, symbol: str) -> dict:
        """Fetch and parse OI data for a symbol.
        
        FAIL-SAFE: Returns empty dict on any error.
        
        Args:
            symbol: e.g., "SBIN", "NSE:SBIN", "NIFTY", "NSE:NIFTY 50"
            
        Returns:
            Parsed OI data dict (see module docstring for format)
        """
        try:
            clean_symbol = symbol.replace("NSE:", "").replace("NFO:", "").strip().upper()
            
            # Check cache
            cache_key = clean_symbol
            if cache_key in self._cache:
                cached_time, cached_data = self._cache[cache_key]
                if (time.time() - cached_time) < self.CACHE_TTL:
                    return cached_data
            
            # Fetch raw data
            raw = self._get_raw_chain(symbol)
            if not raw:
                return {}
            
            # Parse
            result = self._parse_chain(raw, clean_symbol)
            
            # Cache
            if result:
                self._cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception:
            return {}
    
    def fetch_batch(self, symbols: List[str], max_symbols: int = 15) -> Dict[str, dict]:
        """Fetch OI data for multiple symbols.
        
        Respects rate limits with throttling between each request.
        
        Args:
            symbols: List of symbols
            max_symbols: Max to fetch (limits API calls)
            
        Returns:
            {symbol: oi_data, ...}
        """
        results = {}
        for sym in symbols[:max_symbols]:
            data = self.fetch(sym)
            if data:
                results[sym] = data
        return results
    
    def _parse_chain(self, raw: dict, symbol: str) -> dict:
        """Parse NSE raw JSON into structured OI data.
        
        Returns empty dict on parse failure.
        """
        try:
            records = raw.get('records', {})
            data_rows = records.get('data', [])
            
            if not data_rows:
                return {}
            
            # Spot price
            spot_price = 0.0
            underlying_value = records.get('underlyingValue', 0)
            if underlying_value:
                spot_price = float(underlying_value)
            
            # Timestamp from NSE
            nse_timestamp = records.get('timestamp', '')
            
            # Expiry dates available
            expiry_dates = records.get('expiryDates', [])
            
            # Use nearest expiry only (most liquid, most relevant OI)
            nearest_expiry = expiry_dates[0] if expiry_dates else None
            
            # Filter to nearest expiry
            if nearest_expiry:
                filtered_rows = [r for r in data_rows if r.get('expiryDate') == nearest_expiry]
            else:
                filtered_rows = data_rows
            
            if not filtered_rows:
                return {}
            
            # Parse strike-level data
            strikes = []
            total_call_oi = 0
            total_put_oi = 0
            total_call_oi_change = 0
            total_put_oi_change = 0
            total_call_volume = 0
            total_put_volume = 0
            
            for row in filtered_rows:
                strike_price = row.get('strikePrice', 0)
                
                ce = row.get('CE', {})
                pe = row.get('PE', {})
                
                ce_oi = ce.get('openInterest', 0) if ce else 0
                ce_oi_change = ce.get('changeinOpenInterest', 0) if ce else 0
                ce_volume = ce.get('totalTradedVolume', 0) if ce else 0
                ce_iv = ce.get('impliedVolatility', 0) if ce else 0
                ce_ltp = ce.get('lastPrice', 0) if ce else 0
                ce_change = ce.get('change', 0) if ce else 0
                
                pe_oi = pe.get('openInterest', 0) if pe else 0
                pe_oi_change = pe.get('changeinOpenInterest', 0) if pe else 0
                pe_volume = pe.get('totalTradedVolume', 0) if pe else 0
                pe_iv = pe.get('impliedVolatility', 0) if pe else 0
                pe_ltp = pe.get('lastPrice', 0) if pe else 0
                pe_change = pe.get('change', 0) if pe else 0
                
                total_call_oi += ce_oi
                total_put_oi += pe_oi
                total_call_oi_change += ce_oi_change
                total_put_oi_change += pe_oi_change
                total_call_volume += ce_volume
                total_put_volume += pe_volume
                
                strikes.append({
                    'strike': strike_price,
                    'ce_oi': ce_oi,
                    'ce_oi_change': ce_oi_change,
                    'ce_volume': ce_volume,
                    'ce_iv': ce_iv,
                    'ce_ltp': ce_ltp,
                    'ce_change': ce_change,
                    'pe_oi': pe_oi,
                    'pe_oi_change': pe_oi_change,
                    'pe_volume': pe_volume,
                    'pe_iv': pe_iv,
                    'pe_ltp': pe_ltp,
                    'pe_change': pe_change,
                })
            
            # Sort strikes
            strikes.sort(key=lambda x: x['strike'])
            
            # === Computed Metrics ===
            
            # PCR
            pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 1.0
            pcr_oi_change = round(total_put_oi_change / total_call_oi_change, 3) if total_call_oi_change != 0 else 0.0
            pcr_volume = round(total_put_volume / total_call_volume, 3) if total_call_volume > 0 else 1.0
            
            # Top OI strikes (resistance/support)
            call_oi_sorted = sorted(
                [(s['strike'], s['ce_oi']) for s in strikes if s['ce_oi'] > 0],
                key=lambda x: x[1], reverse=True
            )
            put_oi_sorted = sorted(
                [(s['strike'], s['pe_oi']) for s in strikes if s['pe_oi'] > 0],
                key=lambda x: x[1], reverse=True
            )
            
            top_call_oi = call_oi_sorted[:5]
            top_put_oi = put_oi_sorted[:5]
            
            call_resistance = top_call_oi[0][0] if top_call_oi else 0
            put_support = top_put_oi[0][0] if top_put_oi else 0
            
            # Top OI change strikes (fresh buildup)
            call_oi_change_sorted = sorted(
                [(s['strike'], s['ce_oi_change']) for s in strikes if s['ce_oi_change'] > 0],
                key=lambda x: x[1], reverse=True
            )
            put_oi_change_sorted = sorted(
                [(s['strike'], s['pe_oi_change']) for s in strikes if s['pe_oi_change'] > 0],
                key=lambda x: x[1], reverse=True
            )
            
            top_call_oi_change = call_oi_change_sorted[:5]
            top_put_oi_change = put_oi_change_sorted[:5]
            
            # Max pain
            max_pain = self._calc_max_pain(strikes)
            
            # IV Skew (ATM)
            iv_skew = self._calc_iv_skew(strikes, spot_price)
            
            # OI buildup signal
            buildup_signal, buildup_strength = self._detect_oi_buildup(
                total_call_oi_change, total_put_oi_change,
                total_call_volume, total_put_volume,
                spot_price, strikes
            )
            
            result = {
                'symbol': symbol,
                'spot_price': spot_price,
                'timestamp': nse_timestamp,
                'fetch_time': datetime.now().isoformat(),
                'expiry': nearest_expiry or '',
                
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
                
                # Full strike data (for snapshot logging)
                'strikes': strikes,
            }
            
            return result
            
        except Exception:
            return {}
    
    def _calc_max_pain(self, strikes: list) -> float:
        """Calculate max pain strike from parsed strike data."""
        try:
            if not strikes:
                return 0.0
            
            min_pain = float('inf')
            max_pain_strike = 0.0
            
            for test in strikes:
                test_strike = test['strike']
                total_pain = 0
                
                for s in strikes:
                    # Call pain: if price > strike, call buyers profit
                    if test_strike > s['strike'] and s['ce_oi'] > 0:
                        total_pain += (test_strike - s['strike']) * s['ce_oi']
                    # Put pain: if price < strike, put buyers profit
                    if test_strike < s['strike'] and s['pe_oi'] > 0:
                        total_pain += (s['strike'] - test_strike) * s['pe_oi']
                
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception:
            return 0.0
    
    def _calc_iv_skew(self, strikes: list, spot: float) -> float:
        """Calculate ATM IV skew (put IV - call IV). Positive = bearish fear."""
        try:
            if not strikes or spot <= 0:
                return 0.0
            
            # Find ATM strike (closest to spot)
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
        """Detect OI buildup pattern from OI changes.
        
        Patterns:
        - LONG_BUILDUP: Put OI increasing + Call OI stable/decreasing + spot near support
        - SHORT_BUILDUP: Call OI increasing + Put OI stable/decreasing + spot near resistance
        - SHORT_COVERING: Call OI decreasing + spot moving up
        - LONG_UNWINDING: Put OI decreasing + spot moving down
        - NEUTRAL: No clear pattern
        
        Returns:
            (signal, strength) where strength is 0.0 to 1.0
        """
        try:
            net_oi_change = put_oi_chg - call_oi_chg
            total_oi_change = abs(put_oi_chg) + abs(call_oi_chg)
            
            if total_oi_change == 0:
                return 'NEUTRAL', 0.0
            
            # Ratio of net change to total change (-1.0 to +1.0)
            # Positive = more puts being added = bullish support building
            # Negative = more calls being added = bearish, resistance building
            oi_ratio = net_oi_change / total_oi_change if total_oi_change > 0 else 0
            
            # Volume ratio for confirmation
            vol_ratio = put_vol / call_vol if call_vol > 0 else 1.0
            
            signal = 'NEUTRAL'
            strength = 0.0
            
            if put_oi_chg > 0 and oi_ratio > 0.3:
                # Significant put OI building → support → bullish
                if vol_ratio > 0.8:
                    signal = 'LONG_BUILDUP'
                    strength = min(1.0, abs(oi_ratio) * 0.8 + 0.2)
                else:
                    signal = 'LONG_BUILDUP'
                    strength = min(0.7, abs(oi_ratio) * 0.6)
            
            elif call_oi_chg > 0 and oi_ratio < -0.3:
                # Significant call OI building → resistance → bearish
                if vol_ratio < 1.2:
                    signal = 'SHORT_BUILDUP'
                    strength = min(1.0, abs(oi_ratio) * 0.8 + 0.2)
                else:
                    signal = 'SHORT_BUILDUP'
                    strength = min(0.7, abs(oi_ratio) * 0.6)
            
            elif call_oi_chg < 0 and abs(call_oi_chg) > abs(put_oi_chg):
                # Call OI decreasing = call writers exiting = short covering rally
                signal = 'SHORT_COVERING'
                strength = min(0.8, abs(oi_ratio) * 0.7)
            
            elif put_oi_chg < 0 and abs(put_oi_chg) > abs(call_oi_chg):
                # Put OI decreasing = put writers exiting = long unwinding
                signal = 'LONG_UNWINDING'
                strength = min(0.8, abs(oi_ratio) * 0.7)
            
            return signal, round(strength, 3)
            
        except Exception:
            return 'NEUTRAL', 0.0
    
    def get_snapshot_for_logging(self, symbol: str) -> dict:
        """Get a compact snapshot dict suitable for JSONL logging.
        
        Strips full strike data to keep logs manageable.
        """
        try:
            data = self.fetch(symbol)
            if not data:
                return {}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': data.get('symbol', symbol),
                'spot_price': data.get('spot_price'),
                'expiry': data.get('expiry'),
                'nse_timestamp': data.get('timestamp'),
                
                # Totals
                'total_call_oi': data.get('total_call_oi'),
                'total_put_oi': data.get('total_put_oi'),
                'total_call_oi_change': data.get('total_call_oi_change'),
                'total_put_oi_change': data.get('total_put_oi_change'),
                'total_call_volume': data.get('total_call_volume'),
                'total_put_volume': data.get('total_put_volume'),
                
                # Ratios
                'pcr_oi': data.get('pcr_oi'),
                'pcr_oi_change': data.get('pcr_oi_change'),
                'pcr_volume': data.get('pcr_volume'),
                
                # Key levels
                'max_pain': data.get('max_pain'),
                'call_resistance': data.get('call_resistance'),
                'put_support': data.get('put_support'),
                'iv_skew': data.get('iv_skew'),
                
                # Top 3 strikes (compact)
                'top3_call_oi': data.get('top_call_oi_strikes', [])[:3],
                'top3_put_oi': data.get('top_put_oi_strikes', [])[:3],
                'top3_call_oi_change': data.get('top_call_oi_change_strikes', [])[:3],
                'top3_put_oi_change': data.get('top_put_oi_change_strikes', [])[:3],
                
                # Buildup
                'oi_buildup_signal': data.get('oi_buildup_signal'),
                'oi_buildup_strength': data.get('oi_buildup_strength'),
            }
        except Exception:
            return {}
    
    def to_flow_analyzer_format(self, symbol: str) -> dict:
        """Convert NSE data to the format expected by OptionsFlowAnalyzer.
        
        This allows NSE data to be used as a DROP-IN replacement or
        supplement for the Kite-based flow analyzer.
        
        Returns dict compatible with apply_oi_overlay():
          flow_bias, flow_confidence, pcr_oi, iv_skew, max_pain, etc.
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
            
            # Determine bias from BOTH OI change patterns AND PCR
            bias = 'NEUTRAL'
            confidence = 0.5
            boost = 0
            
            # Primary signal: OI buildup pattern (uses OI CHANGE — unavailable in Kite)
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
            
            # Secondary: PCR confirmation/override (if buildup is neutral)
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
            
            # Clamp
            boost = max(-4, min(4, boost))
            confidence = round(min(0.9, confidence), 3)
            
            # Spot vs max pain
            mp_dist = ((spot - max_pain) / max_pain * 100) if max_pain > 0 else 0
            
            # Build GPT line
            gpt_parts = [f"PCR:{pcr:.2f}"]
            if data.get('total_call_oi_change', 0) != 0 or data.get('total_put_oi_change', 0) != 0:
                gpt_parts.append(f"ΔOI:CE{data.get('total_call_oi_change', 0):+,}/PE{data.get('total_put_oi_change', 0):+,}")
            if abs(iv_skew) > 2:
                gpt_parts.append(f"IVskew:{iv_skew:+.1f}")
            if max_pain > 0:
                gpt_parts.append(f"MaxPain:{max_pain:.0f}")
            if buildup != 'NEUTRAL':
                gpt_parts.append(f"Buildup:{buildup}({strength:.0%})")
            
            gpt_line = f"📊NSE_OI:{bias}({','.join(gpt_parts)})"
            
            return {
                'flow_bias': bias,
                'flow_confidence': confidence,
                'pcr_oi': pcr,
                'iv_skew': iv_skew,
                'max_pain': max_pain,
                'call_resistance': data.get('call_resistance', 0),
                'put_support': data.get('put_support', 0),
                'spot_price': spot,
                'spot_vs_max_pain_pct': round(mp_dist, 2),
                'flow_score_boost': boost,
                'flow_gpt_line': gpt_line,
                # NSE-exclusive fields (richer than Kite)
                'nse_oi_buildup': buildup,
                'nse_oi_buildup_strength': strength,
                'nse_total_call_oi_change': data.get('total_call_oi_change', 0),
                'nse_total_put_oi_change': data.get('total_put_oi_change', 0),
                'nse_pcr_volume': data.get('pcr_volume', 1.0),
                'nse_pcr_oi_change': data.get('pcr_oi_change', 0.0),
            }
            
        except Exception:
            return self._neutral_flow()
    
    def _neutral_flow(self) -> dict:
        """Neutral flow data — zero effect on Titan."""
        return {
            'flow_bias': 'NEUTRAL',
            'flow_confidence': 0.0,
            'pcr_oi': 1.0,
            'iv_skew': 0.0,
            'max_pain': 0.0,
            'call_resistance': 0,
            'put_support': 0,
            'spot_price': 0.0,
            'spot_vs_max_pain_pct': 0.0,
            'flow_score_boost': 0,
            'flow_gpt_line': '',
            'nse_oi_buildup': 'NEUTRAL',
            'nse_oi_buildup_strength': 0.0,
            'nse_total_call_oi_change': 0,
            'nse_total_put_oi_change': 0,
            'nse_pcr_volume': 1.0,
            'nse_pcr_oi_change': 0.0,
        }
    
    def reset(self):
        """Reset session and cache. Use after prolonged failures."""
        self._session = None
        self._session_valid = False
        self._cache.clear()
        self._consecutive_failures = 0


# === Singleton ===
_nse_fetcher_instance = None


def get_nse_oi_fetcher() -> NSEOIFetcher:
    """Get or create singleton NSE OI fetcher."""
    global _nse_fetcher_instance
    if _nse_fetcher_instance is None:
        _nse_fetcher_instance = NSEOIFetcher()
    return _nse_fetcher_instance


if __name__ == '__main__':
    print("=" * 60)
    print("NSE OI FETCHER — Standalone Test")
    print("=" * 60)
    
    fetcher = NSEOIFetcher()
    
    test_symbols = ["SBIN", "RELIANCE", "NIFTY"]
    
    for sym in test_symbols:
        print(f"\n--- {sym} ---")
        data = fetcher.fetch(sym)
        
        if data:
            print(f"  Spot: ₹{data.get('spot_price', 0):.2f}")
            print(f"  Expiry: {data.get('expiry', 'N/A')}")
            print(f"  PCR (OI): {data.get('pcr_oi', 0):.3f}")
            print(f"  PCR (Volume): {data.get('pcr_volume', 0):.3f}")
            print(f"  PCR (OI Change): {data.get('pcr_oi_change', 0):.3f}")
            print(f"  Total Call OI: {data.get('total_call_oi', 0):,}")
            print(f"  Total Put OI: {data.get('total_put_oi', 0):,}")
            print(f"  Call OI Change: {data.get('total_call_oi_change', 0):+,}")
            print(f"  Put OI Change: {data.get('total_put_oi_change', 0):+,}")
            print(f"  Max Pain: ₹{data.get('max_pain', 0):.0f}")
            print(f"  Call Resistance: ₹{data.get('call_resistance', 0):.0f}")
            print(f"  Put Support: ₹{data.get('put_support', 0):.0f}")
            print(f"  IV Skew: {data.get('iv_skew', 0):+.2f}")
            print(f"  OI Buildup: {data.get('oi_buildup_signal', 'N/A')} ({data.get('oi_buildup_strength', 0):.0%})")
            
            top_calls = data.get('top_call_oi_strikes', [])[:3]
            top_puts = data.get('top_put_oi_strikes', [])[:3]
            if top_calls:
                print(f"  Top Call OI: {', '.join(f'₹{s[0]:.0f}({s[1]:,})' for s in top_calls)}")
            if top_puts:
                print(f"  Top Put OI: {', '.join(f'₹{s[0]:.0f}({s[1]:,})' for s in top_puts)}")
            
            # Also test flow analyzer format
            flow = fetcher.to_flow_analyzer_format(sym)
            print(f"  Flow: {flow.get('flow_bias')} (conf:{flow.get('flow_confidence'):.2f}, boost:{flow.get('flow_score_boost'):+d})")
            print(f"  GPT: {flow.get('flow_gpt_line', '')}")
        else:
            print("  ❌ No data (NSE may be down or market closed)")
        
        time.sleep(1)  # Be polite
    
    print("\n✅ Done")
