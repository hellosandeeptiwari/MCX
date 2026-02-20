"""
KITE TICKER - REAL-TIME WEBSOCKET STREAMING FOR TITAN
=====================================================
Replaces REST polling (kite.quote/kite.ltp) with WebSocket streaming.

Features:
1. Subscribes to all universe stocks + NIFTY/BANKNIFTY via KiteTicker
2. Maintains in-memory LTP cache updated in real-time (~ms latency)
3. Falls back to REST API if WebSocket disconnects
4. Provides get_ltp() / get_quotes() that read from cache (zero API calls)
5. Auto-reconnects on disconnect
6. Thread-safe — can be accessed from scan loop, exit manager, etc.

Modes:
- LTP mode: 8 bytes per instrument (cheapest)
- QUOTE mode: 44 bytes (OHLC, volume, OI — no depth)
- FULL mode: 184 bytes (includes 5-level market depth)

Limits: 3000 instruments per connection, 3 connections per API key
"""

import time
import threading
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict

try:
    from kiteconnect import KiteTicker as _KiteTicker
    HAS_KITE_TICKER = True
except ImportError:
    HAS_KITE_TICKER = False
    print("⚠️ KiteTicker not available — falling back to REST polling")


class TitanTicker:
    """
    Real-time market data streaming via Kite WebSocket.
    
    Usage:
        ticker = TitanTicker(api_key, access_token, kite_client)
        ticker.start()
        ticker.subscribe_symbols(["NSE:SBIN", "NSE:RELIANCE", ...])
        
        # Get data (zero API calls — reads from in-memory cache)
        ltp = ticker.get_ltp("NSE:SBIN")
        batch = ticker.get_ltp_batch(["NSE:SBIN", "NSE:RELIANCE"])
        quote = ticker.get_quote("NSE:SBIN")  # OHLC + volume + OI
    """
    
    def __init__(self, api_key: str, access_token: str, kite_client=None):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = kite_client  # Fallback for REST calls
        
        # === DATA STORES (thread-safe via lock) ===
        self._lock = threading.RLock()
        self._ltp_cache: Dict[int, float] = {}           # token -> last_price
        self._quote_cache: Dict[int, Dict] = {}           # token -> full quote dict
        self._last_update: Dict[int, float] = {}          # token -> timestamp
        
        # === SYMBOL MAPPING ===
        self._symbol_to_token: Dict[str, int] = {}        # "NSE:SBIN" -> 779521
        self._token_to_symbol: Dict[int, str] = {}        # 779521 -> "NSE:SBIN"
        self._subscribed_tokens: Set[int] = set()
        
        # === STATE ===
        self._ws: Optional[_KiteTicker] = None if not HAS_KITE_TICKER else None
        self._connected = False
        self._running = False
        self._reconnect_count = 0
        self._max_reconnects = 50
        self._tick_count = 0
        self._last_tick_time = 0
        self._started_at = None
        
        # === INSTRUMENT MAP (loaded once) ===
        self._instruments_loaded = False
        self._instrument_map: Dict[str, int] = {}  # "NSE:SBIN" -> instrument_token
        
        # Stats
        self._stats = {
            'ticks_received': 0,
            'reconnects': 0,
            'errors': 0,
            'fallback_calls': 0,
            'cache_hits': 0,
        }
    
    def _load_instruments(self):
        """Load instrument token mapping from Kite (once per day)"""
        if self._instruments_loaded or not self.kite:
            return
        
        try:
            # Load NSE + NFO instruments
            for exchange in ["NSE", "NFO"]:
                instruments = self.kite.instruments(exchange)
                for inst in instruments:
                    key = f"{inst['exchange']}:{inst['tradingsymbol']}"
                    self._instrument_map[key] = inst['instrument_token']
            
            # Also add index instruments
            try:
                nse_instruments = self.kite.instruments("NSE")
                for inst in nse_instruments:
                    if inst.get('segment') == 'INDICES':
                        key = f"NSE:{inst['tradingsymbol']}"
                        self._instrument_map[key] = inst['instrument_token']
            except:
                pass
            
            self._instruments_loaded = True
            print(f"🔌 Ticker: Loaded {len(self._instrument_map)} instrument tokens")
        except Exception as e:
            print(f"⚠️ Ticker: Failed to load instruments: {e}")
    
    def _resolve_token(self, symbol: str) -> Optional[int]:
        """Resolve symbol to instrument_token"""
        if symbol in self._symbol_to_token:
            return self._symbol_to_token[symbol]
        
        if not self._instruments_loaded:
            self._load_instruments()
        
        token = self._instrument_map.get(symbol)
        if token:
            self._symbol_to_token[symbol] = token
            self._token_to_symbol[token] = symbol
        return token
    
    def start(self):
        """Start WebSocket connection in background thread"""
        if not HAS_KITE_TICKER:
            print("⚠️ KiteTicker not available — ticker running in REST-fallback mode")
            self._running = True
            return
        
        if self._running:
            return
        
        self._load_instruments()
        
        self._ws = _KiteTicker(self.api_key, self.access_token)
        
        # Register callbacks
        self._ws.on_ticks = self._on_ticks
        self._ws.on_connect = self._on_connect
        self._ws.on_close = self._on_close
        self._ws.on_error = self._on_error
        self._ws.on_reconnect = self._on_reconnect
        self._ws.on_noreconnect = self._on_noreconnect
        self._ws.on_order_update = self._on_order_update
        
        self._running = True
        self._started_at = time.time()
        
        # Run WebSocket in daemon thread (won't block main process)
        self._ws_thread = threading.Thread(
            target=self._ws.connect,
            kwargs={'threaded': True},
            daemon=True,
            name="TitanTicker-WS"
        )
        self._ws_thread.start()
        print("🔌 Ticker: WebSocket connecting...")
    
    def stop(self):
        """Stop WebSocket connection"""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except:
                pass
        self._connected = False
        print("🔌 Ticker: Stopped")
    
    def subscribe_symbols(self, symbols: List[str], mode: str = 'quote'):
        """
        Subscribe to real-time data for given symbols.
        
        Args:
            symbols: List of "EXCHANGE:TRADINGSYMBOL" strings
            mode: 'ltp', 'quote', or 'full'
        """
        tokens_to_sub = []
        for sym in symbols:
            token = self._resolve_token(sym)
            if token and token not in self._subscribed_tokens:
                tokens_to_sub.append(token)
                self._subscribed_tokens.add(token)
        
        if not tokens_to_sub:
            return
        
        if self._connected and self._ws:
            try:
                self._ws.subscribe(tokens_to_sub)
                # Set mode
                mode_map = {'ltp': 'ltp', 'quote': 'quote', 'full': 'full'}
                ws_mode = mode_map.get(mode, 'quote')
                self._ws.set_mode(ws_mode, tokens_to_sub)
                print(f"🔌 Ticker: Subscribed {len(tokens_to_sub)} new instruments ({mode} mode)")
            except Exception as e:
                print(f"⚠️ Ticker subscribe error: {e}")
        else:
            # Will subscribe on connect
            pass
    
    def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        tokens = []
        for sym in symbols:
            token = self._symbol_to_token.get(sym)
            if token:
                tokens.append(token)
                self._subscribed_tokens.discard(token)
        
        if tokens and self._connected and self._ws:
            try:
                self._ws.unsubscribe(tokens)
            except:
                pass
    
    # ===========================================================
    # DATA ACCESS — Zero API calls, reads from in-memory cache
    # ===========================================================
    
    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get last traded price from cache. Returns None if not subscribed/no data.
        Falls back to REST kite.ltp() if cache miss.
        """
        token = self._symbol_to_token.get(symbol) or self._resolve_token(symbol)
        
        if token:
            with self._lock:
                ltp = self._ltp_cache.get(token)
                if ltp is not None:
                    self._stats['cache_hits'] += 1
                    return ltp
        
        # Cache miss — REST fallback
        if self.kite:
            try:
                self._stats['fallback_calls'] += 1
                data = self.kite.ltp([symbol])
                ltp = data.get(symbol, {}).get('last_price')
                if ltp and token:
                    with self._lock:
                        self._ltp_cache[token] = ltp
                        self._last_update[token] = time.time()
                return ltp
            except:
                return None
        return None
    
    def get_ltp_batch(self, symbols: List[str], max_age_seconds: float = 60.0) -> Dict[str, float]:
        """
        Get LTPs for multiple symbols. Uses cache first, REST fallback for misses.
        Stale cache entries (older than max_age_seconds) are treated as misses.
        """
        result = {}
        cache_misses = []
        now = time.time()
        
        for sym in symbols:
            token = self._symbol_to_token.get(sym) or self._resolve_token(sym)
            if token:
                with self._lock:
                    ltp = self._ltp_cache.get(token)
                    last_ts = self._last_update.get(token, 0)
                    if ltp is not None and (now - last_ts) <= max_age_seconds:
                        result[sym] = ltp
                        self._stats['cache_hits'] += 1
                        continue
            cache_misses.append(sym)
        
        # REST fallback for cache misses
        if cache_misses and self.kite:
            try:
                self._stats['fallback_calls'] += 1
                data = self.kite.ltp(cache_misses)
                for sym in cache_misses:
                    if sym in data:
                        ltp = data[sym].get('last_price')
                        if ltp:
                            result[sym] = ltp
                            token = self._symbol_to_token.get(sym)
                            if token:
                                with self._lock:
                                    self._ltp_cache[token] = ltp
                                    self._last_update[token] = time.time()
            except:
                pass
        
        return result
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get full quote (OHLC, volume, OI, LTP) from cache.
        Returns None if not subscribed in quote/full mode.
        """
        token = self._symbol_to_token.get(symbol) or self._resolve_token(symbol)
        if token:
            with self._lock:
                quote = self._quote_cache.get(token)
                if quote:
                    self._stats['cache_hits'] += 1
                    return quote
        
        # REST fallback
        if self.kite:
            try:
                self._stats['fallback_calls'] += 1
                data = self.kite.quote([symbol])
                return data.get(symbol)
            except:
                return None
        return None
    
    def get_quote_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols from cache + REST fallback"""
        result = {}
        cache_misses = []
        
        for sym in symbols:
            token = self._symbol_to_token.get(sym) or self._resolve_token(sym)
            if token:
                with self._lock:
                    quote = self._quote_cache.get(token)
                    if quote:
                        result[sym] = quote
                        self._stats['cache_hits'] += 1
                        continue
            cache_misses.append(sym)
        
        if cache_misses and self.kite:
            try:
                self._stats['fallback_calls'] += 1
                data = self.kite.quote(cache_misses)
                for sym in cache_misses:
                    if sym in data:
                        result[sym] = data[sym]
            except:
                pass
        
        return result
    
    def is_fresh(self, symbol: str, max_age_seconds: float = 5.0) -> bool:
        """Check if cached data for a symbol is fresh (within max_age)"""
        token = self._symbol_to_token.get(symbol)
        if not token:
            return False
        with self._lock:
            last = self._last_update.get(token, 0)
        return (time.time() - last) < max_age_seconds
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    @property
    def stats(self) -> Dict:
        """Get ticker stats for dashboard"""
        with self._lock:
            cached_count = len(self._ltp_cache)
        
        uptime = time.time() - self._started_at if self._started_at else 0
        return {
            'connected': self._connected,
            'subscribed': len(self._subscribed_tokens),
            'cached': cached_count,
            'ticks': self._stats['ticks_received'],
            'cache_hits': self._stats['cache_hits'],
            'fallbacks': self._stats['fallback_calls'],
            'reconnects': self._stats['reconnects'],
            'errors': self._stats['errors'],
            'uptime_min': round(uptime / 60, 1),
            'tps': round(self._stats['ticks_received'] / max(uptime, 1), 1),
        }
    
    # ===========================================================
    # WEBSOCKET CALLBACKS
    # ===========================================================
    
    def _on_ticks(self, ws, ticks):
        """Called when tick data is received"""
        with self._lock:
            for tick in ticks:
                token = tick.get('instrument_token')
                if token is None:
                    continue
                
                # Always update LTP
                ltp = tick.get('last_price')
                if ltp:
                    self._ltp_cache[token] = ltp
                
                # Update full quote if available
                if 'ohlc' in tick or 'volume' in tick:
                    self._quote_cache[token] = {
                        'instrument_token': token,
                        'last_price': ltp,
                        'ohlc': tick.get('ohlc', {}),
                        'volume': tick.get('volume_traded', tick.get('volume', 0)),
                        'oi': tick.get('oi', 0),
                        'oi_day_high': tick.get('oi_day_high', 0),
                        'oi_day_low': tick.get('oi_day_low', 0),
                        'last_trade_time': tick.get('last_trade_time'),
                        'last_quantity': tick.get('last_traded_quantity', tick.get('last_quantity', 0)),
                        'buy_quantity': tick.get('total_buy_quantity', tick.get('buy_quantity', 0)),
                        'sell_quantity': tick.get('total_sell_quantity', tick.get('sell_quantity', 0)),
                        'average_price': tick.get('average_traded_price', tick.get('average_price', 0)),
                        'depth': tick.get('depth', {}),
                    }
                
                self._last_update[token] = time.time()
                self._stats['ticks_received'] += 1
    
    def _on_connect(self, ws, response):
        """Called on WebSocket connect"""
        self._connected = True
        print(f"🔌 Ticker: WebSocket CONNECTED")
        
        # Subscribe to all previously requested tokens
        if self._subscribed_tokens:
            tokens_list = list(self._subscribed_tokens)
            try:
                ws.subscribe(tokens_list)
                ws.set_mode('quote', tokens_list)
                print(f"🔌 Ticker: Re-subscribed {len(tokens_list)} instruments")
            except Exception as e:
                print(f"⚠️ Ticker: Re-subscribe error: {e}")
    
    def _on_close(self, ws, code, reason):
        """Called on WebSocket close"""
        self._connected = False
        if self._running:
            print(f"🔌 Ticker: Connection closed ({code}: {reason}) — will reconnect")
    
    def _on_error(self, ws, code, reason):
        """Called on WebSocket error"""
        self._stats['errors'] += 1
        print(f"⚠️ Ticker error: {code} — {reason}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Called on reconnect attempt"""
        self._stats['reconnects'] += 1
        self._reconnect_count = attempts_count
    
    def _on_noreconnect(self, ws):
        """Called when max reconnect attempts reached"""
        self._connected = False
        print("❌ Ticker: Max reconnect attempts reached — falling back to REST")
    
    def _on_order_update(self, ws, data):
        """
        Called when order update is received via WebSocket.
        This replaces the need for HTTP postback webhooks.
        """
        order_id = data.get('order_id', 'unknown')
        status = data.get('status', 'unknown')
        symbol = data.get('tradingsymbol', 'unknown')
        print(f"📋 Order update via WS: {symbol} #{order_id} → {status}")
        
        # Store for the main loop to process
        if not hasattr(self, '_order_updates'):
            self._order_updates = []
        self._order_updates.append({
            'order_id': order_id,
            'status': status,
            'symbol': symbol,
            'data': data,
            'received_at': time.time()
        })
    
    def get_pending_order_updates(self) -> List[Dict]:
        """Get and clear pending order updates received via WebSocket"""
        if not hasattr(self, '_order_updates'):
            return []
        updates = self._order_updates
        self._order_updates = []
        return updates

    # ===========================================================
    # F&O FUTURES SUBSCRIPTION — OI DATA FOR ENTIRE UNIVERSE
    # ===========================================================

    def subscribe_fo_futures(self, nfo_instruments: List[Dict] = None):
        """
        Subscribe to ALL near-month F&O futures for real-time OI data.

        This eliminates per-stock REST calls in _get_futures_oi_quick().
        Subscribes in 'quote' mode → OI, volume, OHLC for ~200 futures.
        Well within the 3000-instrument-per-connection limit.

        Args:
            nfo_instruments: Pre-fetched kite.instruments("NFO") list.
                             If None, will fetch from Kite API.
        """
        try:
            if nfo_instruments is None and self.kite:
                nfo_instruments = self.kite.instruments("NFO")

            if not nfo_instruments:
                print("⚠️ Ticker: No NFO instruments available for futures subscription")
                return

            # Find nearest-month futures for each stock
            from collections import defaultdict
            stock_futures = defaultdict(list)
            for inst in nfo_instruments:
                if inst.get('instrument_type') == 'FUT' and inst.get('segment') == 'NFO-FUT':
                    stock_futures[inst['name']].append(inst)

            # Pick nearest expiry for each stock
            fut_symbols = []
            self._futures_map = {}  # "NSE:SBIN" -> "NFO:SBIN26FEBFUT" (for lookups)
            self._futures_tokens = {}  # "NSE:SBIN" -> instrument_token of its future

            for stock_name, futs in stock_futures.items():
                futs.sort(key=lambda x: x['expiry'])
                nearest = futs[0]
                fut_sym = f"NFO:{nearest['tradingsymbol']}"
                nse_sym = f"NSE:{stock_name}"
                self._futures_map[nse_sym] = fut_sym
                token = nearest['instrument_token']
                self._futures_tokens[nse_sym] = token
                # Register in symbol-token maps
                self._symbol_to_token[fut_sym] = token
                self._token_to_symbol[token] = fut_sym
                fut_symbols.append(fut_sym)

            # Subscribe all futures in quote mode
            self.subscribe_symbols(fut_symbols, mode='quote')
            print(f"🔌 Ticker: Subscribed {len(fut_symbols)} near-month futures for OI streaming")

        except Exception as e:
            print(f"⚠️ Ticker: Futures subscription error: {e}")

    def get_futures_oi(self, equity_symbol: str) -> Optional[Dict]:
        """
        Get futures OI data for an equity symbol from WebSocket cache.

        Args:
            equity_symbol: "NSE:SBIN" format

        Returns:
            Dict with oi, oi_day_high, oi_day_low, volume, ltp, or None if not cached.
        """
        if not hasattr(self, '_futures_tokens'):
            return None

        token = self._futures_tokens.get(equity_symbol)
        if not token:
            return None

        with self._lock:
            quote = self._quote_cache.get(token)
            if quote:
                self._stats['cache_hits'] += 1
                return {
                    'has_futures': True,
                    'futures_symbol': self._futures_map.get(equity_symbol, ''),
                    'oi': quote.get('oi', 0),
                    'oi_day_high': quote.get('oi_day_high', 0),
                    'oi_day_low': quote.get('oi_day_low', 0),
                    'volume': quote.get('volume', 0),
                    'ltp': quote.get('last_price', 0),
                }

        return None  # Not in cache — caller should REST fallback

    def get_futures_oi_batch(self, equity_symbols: List[str]) -> Dict[str, Dict]:
        """
        Batch-read futures OI for multiple equity symbols from WebSocket cache.
        Returns {symbol: oi_dict} for all cache hits.
        """
        result = {}
        if not hasattr(self, '_futures_tokens'):
            return result

        with self._lock:
            for sym in equity_symbols:
                token = self._futures_tokens.get(sym)
                if token:
                    quote = self._quote_cache.get(token)
                    if quote:
                        result[sym] = {
                            'has_futures': True,
                            'futures_symbol': self._futures_map.get(sym, ''),
                            'oi': quote.get('oi', 0),
                            'oi_day_high': quote.get('oi_day_high', 0),
                            'oi_day_low': quote.get('oi_day_low', 0),
                            'volume': quote.get('volume', 0),
                            'ltp': quote.get('last_price', 0),
                        }
                        self._stats['cache_hits'] += 1
        return result


# === SINGLETON ===
_ticker_instance: Optional[TitanTicker] = None

def get_ticker(api_key: str = None, access_token: str = None, kite_client=None) -> TitanTicker:
    """Get or create the global TitanTicker instance"""
    global _ticker_instance
    if _ticker_instance is None and api_key and access_token:
        _ticker_instance = TitanTicker(api_key, access_token, kite_client)
    return _ticker_instance


def reset_ticker():
    """Reset the global ticker (for testing)"""
    global _ticker_instance
    if _ticker_instance:
        _ticker_instance.stop()
    _ticker_instance = None
