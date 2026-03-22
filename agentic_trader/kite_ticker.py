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
import json
import os
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
        
        # === BREAKOUT WATCHER (initialized later via attach_breakout_watcher) ===
        self._breakout_watcher: Optional['BreakoutWatcher'] = None
    
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/index_instruments]: {e}")
            
            self._instruments_loaded = True
            # print(f"🔌 Ticker: Loaded {len(self._instrument_map)} instrument tokens")
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
        # Save watcher state before shutdown (restart resilience)
        if self._breakout_watcher:
            try:
                self._breakout_watcher.save_state()
                print("   💾 Watcher: state saved on shutdown")
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/save_state_shutdown]: {e}")
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/ws_close]: {e}")
        self._connected = False
    
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/unsubscribe]: {e}")
    
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/get_ltp_REST]: {symbol} — {e}")
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/get_ltp_batch_REST]: {len(cache_misses)} misses — {e}")
        
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/get_quote_REST]: {symbol} — {e}")
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
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/get_quote_batch_REST]: {len(cache_misses)} misses — {e}")
        
        return result
    
    def is_fresh(self, symbol: str, max_age_seconds: float = 5.0) -> bool:
        """Check if cached data for a symbol is fresh (within max_age)"""
        token = self._symbol_to_token.get(symbol)
        if not token:
            return False
        with self._lock:
            last = self._last_update.get(token, 0)
        return (time.time() - last) < max_age_seconds
    
    def attach_breakout_watcher(self, config: dict) -> 'BreakoutWatcher':
        """
        Create and attach a BreakoutWatcher that runs inside the tick thread.
        
        Args:
            config: BREAKOUT_WATCHER config dict from config.py
            
        Returns:
            The BreakoutWatcher instance (also stored on self._breakout_watcher)
        """
        watcher = BreakoutWatcher(config)
        self._breakout_watcher = watcher
        if watcher._enabled:
            print(f"   ⚡ Breakout Watcher: ACTIVE (spike≥{config.get('price_spike_pct', 0.8)}%, "
                  f"slow_grind≥{config.get('slow_grind_pct', 1.0)}%/5min, "
                  f"sustain={config.get('sustain_seconds', 15)}s, "
                  f"cooldown={config.get('cooldown_seconds', 180)}s+escalation, "
                  f"queue={config.get('queue_size', 100)}, "
                  f"bypass≥{config.get('priority_bypass_pct', 2.0)}%)")
        return watcher
    
    @property
    def breakout_watcher(self) -> Optional['BreakoutWatcher']:
        """Access the attached breakout watcher"""
        return self._breakout_watcher
    
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
                
                # Periodic tick count log (every 10,000 ticks)
                if self._stats['ticks_received'] % 10000 == 0:
                    _bw_stats = self._breakout_watcher.stats if self._breakout_watcher else {}
                    print(f"📊 Ticker: {self._stats['ticks_received']:,} ticks | "
                          f"spikes={_bw_stats.get('spikes_detected', 0)} "
                          f"extremes={_bw_stats.get('extremes_detected', 0)} "
                          f"vol_surges={_bw_stats.get('vol_surges_detected', 0)} "
                          f"sustain_pass={_bw_stats.get('sustain_passed', 0)} "
                          f"sustain_fail={_bw_stats.get('sustain_failed', 0)} "
                          f"queued={_bw_stats.get('queued', 0)}")
                
                # === BREAKOUT WATCHER: feed every equity tick ===
                if self._breakout_watcher:
                    sym = self._token_to_symbol.get(token)
                    if sym and sym.startswith('NSE:'):
                        if ':NIFTY' not in sym:
                            self._breakout_watcher.process_tick(sym, tick, self._token_to_symbol)
                        elif sym == 'NSE:NIFTY 50':
                            # Feed NIFTY 50 data for Earlybird market-context gating
                            _n_ltp = tick.get('last_price', 0)
                            if _n_ltp > 0:
                                self._breakout_watcher._nifty_ltp = _n_ltp
                            _n_ohlc = tick.get('ohlc', {})
                            if _n_ohlc:
                                _n_close = _n_ohlc.get('close', 0)
                                _n_open = _n_ohlc.get('open', 0)
                                if _n_close > 0 and self._breakout_watcher._nifty_prev_close == 0:
                                    self._breakout_watcher._nifty_prev_close = _n_close
                                if _n_open > 0 and self._breakout_watcher._nifty_day_open == 0:
                                    self._breakout_watcher._nifty_day_open = _n_open
    
    def _on_connect(self, ws, response):
        """Called on WebSocket connect"""
        self._connected = True
        print(f"🔌 Ticker: WebSocket CONNECTED — subscribing {len(self._subscribed_tokens)} instruments")
        
        # Subscribe to all previously requested tokens
        if self._subscribed_tokens:
            tokens_list = list(self._subscribed_tokens)
            try:
                ws.subscribe(tokens_list)
                ws.set_mode('quote', tokens_list)
                print(f"🔌 Ticker: Subscribed {len(tokens_list)} instruments OK")
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
        # print(f"📋 Order update via WS: {symbol} #{order_id} → {status}")
        
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
            # print(f"🔌 Ticker: Subscribed {len(fut_symbols)} near-month futures for OI streaming")

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
                    'buy_quantity': quote.get('buy_quantity', 0),
                    'sell_quantity': quote.get('sell_quantity', 0),
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
                            'buy_quantity': quote.get('buy_quantity', 0),
                            'sell_quantity': quote.get('sell_quantity', 0),
                        }
                        self._stats['cache_hits'] += 1
        return result


# ===========================================================
# BREAKOUT WATCHER — WebSocket-Driven Fast Entry Detection
# ===========================================================
# Runs inside the existing TitanTicker tick thread.
# Monitors ALL subscribed equities for breakout signals.
# Pushes triggered symbols to a thread-safe queue.
# Main loop drains the queue within 1 second and runs full pipeline.
# ===========================================================

import queue


class PriorityTriggerQueue:
    """
    Thread-safe priority queue for breakout triggers.
    
    Uses abs(move_pct) as priority — bigger moves get processed first.
    When full, evicts the LOWEST-priority item instead of dropping the new one,
    ensuring large movers (like VBL's 3% spike) are never lost to noise.
    
    Thread-safety: guarded by a threading.Lock since both the WebSocket thread
    (_fire_trigger) and the main thread (drain) access the internal list.
    """

    def __init__(self, maxsize: int = 100):
        self._items: list = []          # [(priority, seq, trigger_data), ...] sorted desc
        self._maxsize = maxsize
        self._seq = 0
        self._lock = threading.Lock()

    def put(self, trigger_data: dict, priority: float):
        """
        Add a trigger by priority.  If full, evict lowest-priority item
        only if the new item has higher priority.
        
        Returns:
            (was_added: bool, evicted_symbol: str | None)
        """
        with self._lock:
            self._seq += 1

            if len(self._items) < self._maxsize:
                self._items.append((priority, self._seq, trigger_data))
                self._items.sort(key=lambda x: -x[0])      # highest priority first
                return True, None

            # Queue full — compare with the weakest (last) item
            min_priority = self._items[-1][0]
            if priority > min_priority:
                evicted = self._items.pop()                 # remove weakest
                evicted_sym = evicted[2].get('symbol', '?')
                self._items.append((priority, self._seq, trigger_data))
                self._items.sort(key=lambda x: -x[0])
                return True, evicted_sym

            return False, None                              # new item too weak

    def drain(self) -> list:
        """Drain ALL items, returned highest-priority first. Thread-safe."""
        with self._lock:
            result = [item[2] for item in self._items]
            self._items.clear()
            return result

    def empty(self) -> bool:
        with self._lock:
            return len(self._items) == 0

    @property
    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()


class BreakoutWatcher:
    """
    Watches WebSocket ticks for breakout conditions and queues them
    for the main trading loop to process through the full pipeline.
    
    Thread-safety: All methods called from TitanTicker's _on_ticks (WS thread).
    Only drain_queue() is called from the main thread.
    """
    
    def __init__(self, config: dict):
        self._config = config  # Store full config for runtime lookups
        self._enabled = config.get('enabled', False)
        
        # Trigger thresholds
        self._spike_pct = config.get('price_spike_pct', 0.8)
        self._day_extreme = config.get('day_extreme_trigger', True)
        self._vol_surge_x = config.get('volume_surge_multiplier', 2.5)
        self._vol_surge_min_move = config.get('volume_surge_min_move_pct', 0.3)
        
        self._day_ext_min_move = config.get('day_extreme_min_move_pct', 0.4)
        
        # Sustain filter (signal-aware times)
        self._sustain_secs = config.get('sustain_seconds', 60)
        self._sustain_secs_extreme = config.get('sustain_seconds_extreme', 45)
        self._sustain_secs_volume = config.get('sustain_seconds_volume', 45)
        self._sustain_secs_grind = config.get('sustain_seconds_grind', 15)
        self._sustain_recheck_pct = config.get('sustain_recheck_pct', 0.5)
        self._sustain_recheck_pct_volume = config.get('sustain_recheck_pct_volume', 0.15)
        
        # Slow grind: 5-minute baseline to detect persistent moves
        self._slow_grind_pct = config.get('slow_grind_pct', 1.0)  # 1% move over 5min = slow grind
        
        # Cooldown
        self._cooldown_secs = config.get('cooldown_seconds', 180)
        self._max_per_min = config.get('max_triggers_per_minute', 10)
        
        # Priority queue settings
        self._queue_size = config.get('queue_size', 100)
        self._priority_bypass_pct = config.get('priority_bypass_pct', 1.5)
        
        # Active window
        self._active_after = config.get('active_after', '09:20')
        self._active_until = config.get('active_until', '15:10')
        
        # --- Internal state (accessed from WS thread only) ---
        self._lock = threading.Lock()
        self._queue = PriorityTriggerQueue(maxsize=self._queue_size)
        
        # Baseline prices: symbol → {price, timestamp} — snapshot at subscribe or periodic reset
        self._baselines: Dict[str, Dict] = {}  # sym → {'price': float, 'ts': float}  (60s window)
        self._baselines_long: Dict[str, Dict] = {}  # sym → {'price': float, 'ts': float}  (5min window)
        
        # Sustain pending: symbol → {trigger_price, trigger_ts, trigger_type, baseline_price}
        self._pending: Dict[str, Dict] = {}
        
        # Cooldown tracker: symbol → last_trigger_timestamp + trigger type for priority escalation
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_trigger_type: Dict[str, str] = {}  # sym → trigger_type that set cooldown
        
        # Volume rolling average: symbol → deque of recent volume DELTAS (for surge detection)
        from collections import deque
        self._vol_delta_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))  # 50 ticks for stable avg
        self._prev_cumulative_vol: Dict[str, int] = {}  # sym → last cumulative volume
        self._init_ts = time.time()  # Track startup time for warmup-aware detection
        
        # Grind volume tracking: cumulative volume at grind baseline start
        self._grind_start_vol: Dict[str, int] = {}  # sym → cumulative vol when long baseline set
        self._grind_start_avg_delta: Dict[str, float] = {}  # sym → avg vol delta at grind start
        
        # Day extremes: track the REPORTED ohlc.high/low from the exchange
        # Separate dict so the update step doesn't clobber the comparison values
        self._prev_reported_high: Dict[str, float] = {}  # sym → last seen ohlc.high
        self._prev_reported_low: Dict[str, float] = {}   # sym → last seen ohlc.low
        self._day_high_break_count: Dict[str, int] = {}   # sym → how many times new day high fired
        self._day_low_break_count: Dict[str, int] = {}    # sym → how many times new day low fired
        self._last_day_high_break_ts: Dict[str, float] = {}  # sym → epoch of last NEW_DAY_HIGH fire
        self._last_day_low_break_ts: Dict[str, float] = {}   # sym → epoch of last NEW_DAY_LOW fire
        
        # Price history for spike acceleration detection: last few LTPs per symbol
        self._recent_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Global trigger rate limiter
        self._recent_triggers: list = []  # list of timestamps (global)
        self._per_symbol_triggers: Dict[str, list] = {}  # sym → list of timestamps
        
        # Stats
        self._stats = {
            'ticks_processed': 0,
            'spikes_detected': 0,
            'extremes_detected': 0,
            'vol_surges_detected': 0,
            'sustain_passed': 0,
            'sustain_failed': 0,
            'cooldown_blocked': 0,
            'rate_limited': 0,
            'queued': 0,
        }
        # Log-batching: limit noisy watcher messages per 5-min window
        self._log_window_start = time.time()
        self._log_grind_count = 0      # 🐢 SLOW GRIND detected
        self._log_sustain_count = 0    # ✅ SUSTAINED ... queuing
        self._log_grind_suppressed = []   # symbols suppressed
        self._log_sustain_suppressed = [] # symbols suppressed
        self._LOG_GRIND_MAX = 3    # show first N per window
        self._LOG_SUSTAIN_MAX = 5  # show first N per window
        self._LOG_WINDOW_SECS = 300  # 5 min window
        
        # OI Forward-Looking Confirmation — async prefetch during sustain
        # When a trigger enters sustain pending, we launch a bg thread to fetch
        # OI data.  By the time sustain completes (15-60s), the OI result is ready.
        # OI confirmation adjusts sustain thresholds and enriches trigger metadata.
        self._oi_analyzer_ref = None    # Set via attach_oi_analyzer()
        self._oi_prefetch_results: Dict[str, dict] = {}  # sym → {result, ts, direction, signal, strength}
        self._oi_prefetch_lock = threading.Lock()
        # Config: how much OI confirmation / contradiction adjusts the sustain bar
        self._oi_confirm_sustain_ease = config.get('oi_confirm_sustain_ease_pct', 0.15)   # lower recheck by 0.15% when OI confirms
        self._oi_contradict_sustain_tighten = config.get('oi_contradict_sustain_tighten_pct', 0.20)  # raise recheck by 0.20% when OI contradicts
        self._oi_min_strength_for_confirm = config.get('oi_min_strength_for_confirm', 0.35)  # minimum OI strength to count as confirmation

        # === EARLYBIRD STATE (Opening Volatility 09:15-09:45) ===
        from config import EARLYBIRD_COMMON, EARLYBIRD_A, EARLYBIRD_B, EARLYBIRD_C
        self._eb_common = EARLYBIRD_COMMON
        self._eb_a = EARLYBIRD_A
        self._eb_b = EARLYBIRD_B
        self._eb_c = EARLYBIRD_C
        self._earlybird_enabled = EARLYBIRD_COMMON.get('enabled', False)
        self._earlybird_trades_fired = 0  # Count of earlybird triggers queued today
        self._prev_close: Dict[str, float] = {}  # sym → previous day close price (from OHLC)
        self._day_open: Dict[str, float] = {}     # sym → today's open price (from OHLC)
        self._earlybird_fired: Dict[str, set] = {'A': set(), 'B': set(), 'C': set()}  # Per-mode fired symbols
        # NIFTY 50 tracking for market-context gating
        self._nifty_prev_close: float = 0.0    # NIFTY 50 previous day close
        self._nifty_day_open: float = 0.0       # NIFTY 50 today's open
        self._nifty_ltp: float = 0.0            # NIFTY 50 latest tick price

        # State persistence — survive mid-session restarts
        self._state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watcher_state.json')
        self._last_save_ts = 0.0
        self._SAVE_INTERVAL = 60  # Save state every 60s
        self._post_restart = False  # True for first 2 min after loading persisted state
        self._post_restart_until = 0.0
        
        # Try to restore state from a previous session (same trading day)
        self._try_load_state()
    
    # === OI FORWARD-LOOKING CONFIRMATION ===

    def attach_oi_analyzer(self, oi_analyzer):
        """Provide the OptionsFlowAnalyzer reference for OI confirmation.
        Called from autonomous_trader after both ticker and OI analyzer are ready."""
        self._oi_analyzer_ref = oi_analyzer
        print(f"   🔬 Watcher: OI analyzer attached — sustain OI confirmation ACTIVE")

    def _launch_oi_prefetch(self, symbol: str, move_direction: str):
        """Launch a background thread to fetch OI data for a symbol entering sustain.
        
        Non-blocking: the OI fetch runs on a daemon thread, result is stored
        in _oi_prefetch_results for the sustain check to read when ready.
        
        Args:
            symbol: "NSE:SYMBOL" format
            move_direction: 'UP' or 'DOWN' (price move direction)
        """
        if not self._oi_analyzer_ref:
            return
        # Don't re-fetch if we already have a recent result (< 30s old)
        with self._oi_prefetch_lock:
            existing = self._oi_prefetch_results.get(symbol)
            if existing and (time.time() - existing.get('ts', 0)) < 30:
                return

        def _fetch_oi():
            try:
                result = self._oi_analyzer_ref.analyze(symbol)
                if not result:
                    return
                sig = result.get('nse_oi_buildup', 'NEUTRAL')
                strength = result.get('nse_oi_buildup_strength', 0.0)
                bias = result.get('flow_bias', 'NEUTRAL')
                # Determine if OI supports the price move direction
                _oi_bullish = sig in ('LONG_BUILDUP', 'SHORT_COVERING')
                _oi_bearish = sig in ('SHORT_BUILDUP', 'LONG_UNWINDING')
                if move_direction == 'UP':
                    confirms = _oi_bullish
                    contradicts = _oi_bearish
                else:
                    confirms = _oi_bearish
                    contradicts = _oi_bullish
                with self._oi_prefetch_lock:
                    self._oi_prefetch_results[symbol] = {
                        'result': result,
                        'ts': time.time(),
                        'signal': sig,
                        'strength': strength,
                        'bias': bias,
                        'confirms': confirms,
                        'contradicts': contradicts,
                        'move_direction': move_direction,
                        'participant': result.get('oi_participant_id', ''),
                    }
            except Exception as e:
                print(f"⚠️ FALLBACK [ticker/oi_prefetch]: {symbol[-8:] if len(symbol) > 8 else symbol} — {e}")

        t = threading.Thread(target=_fetch_oi, daemon=True, name=f"oi-prefetch-{symbol[-8:]}")
        t.start()

    def _get_oi_confirmation(self, symbol: str) -> Optional[dict]:
        """Get pre-fetched OI confirmation result for a symbol.
        
        Returns:
            dict with {confirms, contradicts, signal, strength, bias, participant}
            or None if not available / too old.
        """
        with self._oi_prefetch_lock:
            result = self._oi_prefetch_results.get(symbol)
            if not result:
                return None
            # Stale check: OI data older than 90s is not useful
            if (time.time() - result.get('ts', 0)) > 90:
                return None
            return result

    def _cleanup_oi_prefetch(self, symbol: str):
        """Remove OI prefetch result after use (or on sustain failure)."""
        with self._oi_prefetch_lock:
            self._oi_prefetch_results.pop(symbol, None)

    # === STATE PERSISTENCE ===
    
    def save_state(self):
        """Save critical watcher state to disk for restart resilience.
        
        Only persists state that's expensive to rebuild:
        - Cooldowns (prevents duplicate triggers after restart)
        - Break counts (prevents stale day extremes from getting fresh-break bonus)
        - Day extremes (prev high/low for comparison)
        - Stats (continuity in monitoring)
        Baselines and volume history rebuild naturally from ticks within seconds.
        """
        now = time.time()
        state = {
            'saved_at': now,
            'saved_date': datetime.now().strftime('%Y-%m-%d'),
            'saved_time': datetime.now().strftime('%H:%M:%S'),
            # Cooldowns: sym → timestamp + type
            'cooldowns': {s: t for s, t in self._cooldowns.items() if now - t < self._cooldown_secs},
            'cooldown_types': dict(self._cooldown_trigger_type),
            # Break counts: essential — prevents 5th break looking like 1st
            'day_high_breaks': dict(self._day_high_break_count),
            'day_low_breaks': dict(self._day_low_break_count),
            # Break timestamps: for consolidation lookback (quiet time before break)
            'last_high_break_ts': dict(self._last_day_high_break_ts),
            'last_low_break_ts': dict(self._last_day_low_break_ts),
            # Day extremes: so we don't re-trigger on the same high/low
            'prev_high': dict(self._prev_reported_high),
            'prev_low': dict(self._prev_reported_low),
            # Stats: continuity
            'stats': dict(self._stats),
            # Per-symbol trigger timestamps (prune old ones)
            'per_sym_triggers': {s: [t for t in ts if now - t < 60]
                                 for s, ts in self._per_symbol_triggers.items()},
        }
        try:
            _tmp = self._state_file + '.tmp'
            with open(_tmp, 'w') as f:
                json.dump(state, f)
            os.replace(_tmp, self._state_file)  # Atomic on POSIX
            self._last_save_ts = now
        except Exception as e:
            print(f"   ⚠️ Watcher: state save failed: {e}")
    
    def _try_load_state(self):
        """Load persisted state if from same trading day and recent enough.
        
        Rules:
        - Must be same calendar date (state is day-specific)
        - Must be < 30 min old (stale state is worse than no state)
        - Sets post_restart flag for first 2 min (triggers carry a caution tag)
        """
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            
            # Validate: same day?
            saved_date = state.get('saved_date', '')
            today = datetime.now().strftime('%Y-%m-%d')
            if saved_date != today:
                print(f"   ℹ️ Watcher: state file from {saved_date} (not today) — skipping")
                return
            
            # Validate: recent enough? (< 30 min)
            saved_at = state.get('saved_at', 0)
            age_min = (time.time() - saved_at) / 60
            if age_min > 30:
                print(f"   ℹ️ Watcher: state file {age_min:.0f}min old (>30min) — skipping")
                return
            
            # Restore cooldowns (only those still active)
            now = time.time()
            _cd = state.get('cooldowns', {})
            for sym, ts in _cd.items():
                if now - ts < self._cooldown_secs:
                    self._cooldowns[sym] = ts
            self._cooldown_trigger_type.update(state.get('cooldown_types', {}))
            
            # Restore break counts
            self._day_high_break_count.update(state.get('day_high_breaks', {}))
            self._day_low_break_count.update(state.get('day_low_breaks', {}))
            # Restore break timestamps (consolidation lookback)
            self._last_day_high_break_ts.update(state.get('last_high_break_ts', {}))
            self._last_day_low_break_ts.update(state.get('last_low_break_ts', {}))
            
            # Restore day extremes
            self._prev_reported_high.update(state.get('prev_high', {}))
            self._prev_reported_low.update(state.get('prev_low', {}))
            
            # Restore stats
            for k, v in state.get('stats', {}).items():
                self._stats[k] = v
            
            # Restore per-symbol trigger timestamps (prune expired)
            for sym, ts_list in state.get('per_sym_triggers', {}).items():
                valid = [t for t in ts_list if now - t < 60]
                if valid:
                    self._per_symbol_triggers[sym] = valid
            
            # Set post-restart warmup: 2 min caution period
            self._post_restart = True
            self._post_restart_until = now + 120
            
            _n_cd = len(self._cooldowns)
            _n_hb = sum(self._day_high_break_count.values())
            _n_lb = sum(self._day_low_break_count.values())
            print(f"   🔄 Watcher: RESTORED state from {state.get('saved_time', '?')} "
                  f"({age_min:.0f}min ago) — {_n_cd} cooldowns, {_n_hb} high breaks, "
                  f"{_n_lb} low breaks, warmup=2min")
        except Exception as e:
            print(f"   ⚠️ Watcher: state load failed: {e}")
    
    def _maybe_auto_save(self):
        """Called from process_tick — saves state every _SAVE_INTERVAL seconds."""
        now = time.time()
        if now - self._last_save_ts >= self._SAVE_INTERVAL:
            self.save_state()
        # Clear post-restart flag after warmup period
        if self._post_restart and now >= self._post_restart_until:
            self._post_restart = False
            print("   ✅ Watcher: post-restart warmup complete — full sensitivity active")
    
    def _is_active_window(self) -> bool:
        """Check if current time is within the active trigger window"""
        now = datetime.now()
        h_after, m_after = map(int, self._active_after.split(':'))
        h_until, m_until = map(int, self._active_until.split(':'))
        start = now.replace(hour=h_after, minute=m_after, second=0)
        end = now.replace(hour=h_until, minute=m_until, second=0)
        return start <= now <= end
    
    def _check_rate_limit(self, symbol: str = None) -> bool:
        """Return True if we're under per-symbol AND global per-minute limits.
        
        Per-symbol: max 3 triggers/min per stock (prevents one noisy stock flooding)
        Global: max 30 triggers/min total (safety net for crash days)
        Individual stocks should NOT be dropped because other stocks ate the quota.
        """
        now = time.time()
        # Prune old timestamps (older than 60s)
        self._recent_triggers = [t for t in self._recent_triggers if now - t < 60]
        
        # Global safety cap (generous — 30/min, not the old 20)
        if len(self._recent_triggers) >= 30:
            return False
        
        # Per-symbol cap: 3/min per stock
        if symbol:
            _sym_triggers = self._per_symbol_triggers.get(symbol, [])
            _sym_triggers = [t for t in _sym_triggers if now - t < 60]
            self._per_symbol_triggers[symbol] = _sym_triggers
            if len(_sym_triggers) >= 3:
                return False
        
        return True
    
    def _is_cooled_down(self, symbol: str) -> bool:
        """Return True if symbol is past its cooldown period"""
        last = self._cooldowns.get(symbol, 0)
        return (time.time() - last) >= self._cooldown_secs

    def mark_grind_traded(self, symbol: str):
        """Reset the grind trend-origin baseline after a trade is actually placed.
        Called by autonomous_trader when a SLOW_GRIND trade successfully enters.
        This ensures the baseline accumulates through blocked triggers but resets
        once the system has acted on the trend."""
        now = time.time()
        ltp = None
        # Get latest price from pending or baseline
        bl = self._baselines_long.get(symbol)
        if bl:
            ltp = bl.get('price', 0)
        # Reset to current state
        if symbol in self._baselines_long:
            self._baselines_long[symbol] = {'price': ltp or 0, 'ts': now, 'peak_move': 0.0, 'min_price': ltp or 0, 'max_price': ltp or 0}
        _cum_vol = self._prev_cumulative_vol.get(symbol, 0)
        self._grind_start_vol[symbol] = _cum_vol
        _hist = self._vol_delta_history.get(symbol)
        self._grind_start_avg_delta[symbol] = (sum(_hist) / len(_hist)) if _hist and len(_hist) >= 3 else 0
    
    def process_tick(self, symbol: str, tick: dict, token_to_symbol: Dict[int, str]):
        """
        Called from TitanTicker._on_ticks for each tick.
        Detects breakout conditions and manages the sustain → queue pipeline.
        
        Args:
            symbol: "NSE:SYMBOL" format
            tick: Raw tick dict from KiteTicker
            token_to_symbol: Token→symbol mapping for reverse lookups
        """
        if not self._enabled:
            return
        
        self._stats['ticks_processed'] += 1
        
        # Auto-save state periodically (every 60s) + clear post-restart warmup
        self._maybe_auto_save()
        
        # Flush log-batch window every 5 min — print summary of suppressed messages
        _now_t = time.time()
        if _now_t - self._log_window_start >= self._LOG_WINDOW_SECS:
            if self._log_grind_suppressed:
                print(f"   🐢 Watcher: +{len(self._log_grind_suppressed)} more SLOW GRIND detected ({', '.join(self._log_grind_suppressed[:8])}{'...' if len(self._log_grind_suppressed) > 8 else ''})")
            if self._log_sustain_suppressed:
                print(f"   ✅ Watcher: +{len(self._log_sustain_suppressed)} more SUSTAINED queued ({', '.join(self._log_sustain_suppressed[:8])}{'...' if len(self._log_sustain_suppressed) > 8 else ''})")
            self._log_window_start = _now_t
            self._log_grind_count = 0
            self._log_sustain_count = 0
            self._log_grind_suppressed = []
            self._log_sustain_suppressed = []
        
        ltp = tick.get('last_price')
        if not ltp or ltp <= 0:
            return
        
        now = time.time()
        
        # === UPDATE DAY EXTREMES (extract OHLC high/low from exchange) ===
        ohlc = tick.get('ohlc', {})
        _tick_day_high = ohlc.get('high', 0) if ohlc else 0
        _tick_day_low = ohlc.get('low', 0) if ohlc else 0
        
        # === TRACK PREV CLOSE & DAY OPEN (for Earlybird gap detection) ===
        if ohlc:
            _ohlc_close = ohlc.get('close', 0)  # Previous day's close
            _ohlc_open = ohlc.get('open', 0)     # Today's open
            if _ohlc_close > 0 and symbol not in self._prev_close:
                self._prev_close[symbol] = _ohlc_close
            if _ohlc_open > 0 and symbol not in self._day_open:
                self._day_open[symbol] = _ohlc_open
        
        # === COMPUTE VOLUME DELTA (convert cumulative → per-tick delta) ===
        _cum_vol = tick.get('volume_traded', tick.get('volume', 0))
        _vol_delta = 0
        if _cum_vol > 0:
            _prev_cv = self._prev_cumulative_vol.get(symbol, 0)
            _vol_delta = max(0, _cum_vol - _prev_cv) if _prev_cv > 0 else 0
            self._prev_cumulative_vol[symbol] = _cum_vol
            if _vol_delta > 0:
                self._vol_delta_history[symbol].append(_vol_delta)
        
        # === UPDATE BASELINES ===
        # Short baseline (60s) — existing, detects fast spikes
        bl = self._baselines.get(symbol)
        if bl is None or (now - bl['ts']) > 60:
            self._baselines[symbol] = {'price': ltp, 'ts': now}
            # Don't return yet — check long baseline first
        
        # Long baseline (rolling) — detects slow grinds that the 60s window misses.
        # ROLLING design: baseline only resets when (a) grind fires, (b) price REVERSES
        # significantly (>50% retrace from peak), or (c) max age (10min or 30min if trending).
        # This way a steady 0.2%/min grind accumulates to 1% over 5 min and fires.
        bl_long = self._baselines_long.get(symbol)
        if bl_long is None:
            self._baselines_long[symbol] = {'price': ltp, 'ts': now, 'peak_move': 0.0, 'min_price': ltp, 'max_price': ltp}
            self._grind_start_vol[symbol] = _cum_vol
            _hist = self._vol_delta_history.get(symbol)
            self._grind_start_avg_delta[symbol] = (sum(_hist) / len(_hist)) if _hist and len(_hist) >= 3 else 0
        else:
            _long_move = (ltp - bl_long['price']) / bl_long['price'] * 100 if bl_long['price'] > 0 else 0
            _age = now - bl_long['ts']

            # Track peak move from baseline (absolute, for retrace detection)
            _peak_move = bl_long.get('peak_move', 0.0)
            if abs(_long_move) > abs(_peak_move):
                bl_long['peak_move'] = _long_move
                _peak_move = _long_move

            # Track min/max prices for spike-vs-grind move efficiency
            if ltp < bl_long.get('min_price', ltp):
                bl_long['min_price'] = ltp
            if ltp > bl_long.get('max_price', ltp):
                bl_long['max_price'] = ltp
            
            # Volume confirmation: compare vol rate during grind vs before grind
            _grind_vol_confirmed = False
            _grind_vol_ratio = 1.0
            _grind_start_cv = self._grind_start_vol.get(symbol, 0)
            _grind_start_ad = self._grind_start_avg_delta.get(symbol, 0)
            if _age > 30 and _grind_start_ad > 0 and _cum_vol > _grind_start_cv:
                _grind_vol_total = _cum_vol - _grind_start_cv
                _grind_vol_rate = _grind_vol_total / _age  # vol/sec during grind
                _baseline_vol_rate = _grind_start_ad  # avg delta/tick before grind
                _grind_vol_ratio = _grind_vol_rate / _baseline_vol_rate if _baseline_vol_rate > 0 else 1.0
                _grind_vol_confirmed = _grind_vol_ratio >= 1.15  # 15%+ more volume = real interest (was 1.2)
            
            # Velocity: how fast the grind is moving (pct per minute)
            _velocity = abs(_long_move) / (_age / 60) if _age > 10 else 0
            
            # Track peak velocity on every tick (not just at detection)
            _prev_peak_vel = bl_long.get('peak_velocity', 0)
            if _velocity > _prev_peak_vel:
                bl_long['peak_velocity'] = _velocity
            
            # Adaptive grind threshold: lower when volume confirms the move
            _effective_grind_pct = self._slow_grind_pct * 0.65 if _grind_vol_confirmed else self._slow_grind_pct
            
            # Check if slow grind threshold crossed
            # TREND-ORIGIN MEMORY: Don't reset baseline on grind fire.
            # Keep the trend origin so re-triggers after cooldown see the
            # FULL cumulative move (e.g. 2.5% not 0.6%). The cooldown
            # check here prevents useless churn during the cooldown window.
            # [FIX Mar 19] Grind ≠ spike. After 09:45 require 5-min buildup
            # at ≤0.4%/min. Before 09:45 keep 3-min/0.8 for opening grinds.
            from datetime import time as _dtime
            _now_t = datetime.now().time()
            _post_open = _now_t >= _dtime(9, 45)
            _grind_min_age = 300 if _post_open else 180   # 5 min after 09:45, 3 min before
            _grind_max_vel = 0.40 if _post_open else 0.80  # tighter slope after open settles
            _grind_cooled = self._is_cooled_down(symbol)
            _grind_age_ok = _age >= _grind_min_age
            _grind_vel_ok = _velocity <= _grind_max_vel

            # === SPIKE-vs-GRIND SMART FILTER ===
            # A real grind is STEADY and LINEAR. Spikes/craters are bursts.
            #
            # 1. MOVE EFFICIENCY: net_move / (high - low) price range.
            #    Grind A→B steadily: eff ~0.85-1.0
            #    Spike+retrace A→C→B: eff <0.60
            #    V-shape crater: eff <0.50
            #
            # 2. CONCENTRATION: fraction of total move in last 60s.
            #    Grind: ~20% per minute over 5 min.
            #    Late spike from flat base: 80-100% in last 60s. Cap at 50%.
            _bl_min = bl_long.get('min_price', bl_long['price'])
            _bl_max = bl_long.get('max_price', ltp)
            _bl_range = _bl_max - _bl_min
            _net_abs = abs(ltp - bl_long['price'])
            _move_efficiency = (_net_abs / _bl_range) if _bl_range > 0.01 else 1.0
            _efficiency_ok = _move_efficiency >= 0.60

            _concentration_ok = True
            _concentration = 0.0
            if abs(_long_move) > 0.3:
                _sb_conc = self._baselines.get(symbol)
                if _sb_conc and _sb_conc['price'] > 0 and (now - _sb_conc['ts']) >= 15:
                    _sb_move_abs = abs((ltp - _sb_conc['price']) / _sb_conc['price'] * 100)
                    if abs(_long_move) > 0.01:
                        _concentration = _sb_move_abs / abs(_long_move)
                        _concentration_ok = _concentration <= 0.50

            if abs(_long_move) >= _effective_grind_pct and symbol not in self._pending and _grind_cooled and _grind_age_ok and _grind_vel_ok and _efficiency_ok and _concentration_ok:
                _sg_type = 'SLOW_GRIND_UP' if _long_move > 0 else 'SLOW_GRIND_DOWN'
                self._stats['slow_grinds_detected'] = self._stats.get('slow_grinds_detected', 0) + 1
                self._pending[symbol] = {
                    'trigger_price': ltp,
                    'trigger_ts': now,
                    'trigger_type': _sg_type,
                    'baseline_price': bl_long['price'],
                    'move_pct': _long_move,
                    'velocity': round(_velocity, 3),
                    'vol_confirmed': _grind_vol_confirmed,
                    'vol_ratio': round(_grind_vol_ratio, 2),
                    'grind_age_s': round(_age, 0),
                    'move_efficiency': round(_move_efficiency, 3),
                    'concentration': round(_concentration, 3),
                }
                # Slope quality signals for end-of-slope detection downstream
                _grind_open = ohlc.get('open', 0) if ohlc else 0
                if _grind_open > 0:
                    self._pending[symbol]['intraday_pct'] = round((ltp - _grind_open) / _grind_open * 100, 2)
                if _tick_day_high > _tick_day_low:
                    self._pending[symbol]['day_range_position'] = round(
                        max(0, min(1, (ltp - _tick_day_low) / (_tick_day_high - _tick_day_low))), 3)
                # Peak velocity: track highest velocity seen for this grind baseline
                _prev_peak_vel = bl_long.get('peak_velocity', 0)
                if _velocity > _prev_peak_vel:
                    bl_long['peak_velocity'] = _velocity
                self._pending[symbol]['peak_velocity'] = round(bl_long.get('peak_velocity', _velocity), 3)
                # SB move: how much did price move in the LAST 60s?
                # If SB << LB the grind has stalled (slope flattening)
                _sb = self._baselines.get(symbol)
                if _sb and _sb['price'] > 0:
                    _sb_move = abs((ltp - _sb['price']) / _sb['price'] * 100)
                    self._pending[symbol]['sb_move_pct'] = round(_sb_move, 3)
                _mins = _age / 60
                _vc_tag = " VOL✓" if _grind_vol_confirmed else ""
                self._log_grind_count += 1
                if self._log_grind_count <= self._LOG_GRIND_MAX:
                    print(f"   🐢 Watcher: {symbol} SLOW GRIND detected ({_long_move:+.1f}% over {_mins:.1f}min, vel={_velocity:.2f}%/min{_vc_tag}, eff={_move_efficiency:.0%}, conc={_concentration:.0%}) → sustain check")
                else:
                    self._log_grind_suppressed.append(symbol.replace('NSE:', ''))
                # TREND-ORIGIN: Do NOT reset baseline here. The origin is
                # preserved so the next trigger (after cooldown) sees the full
                # cumulative grind. Baseline only resets on reversion/staleness
                # or when the trade is actually placed (via mark_grind_traded).
            elif abs(_long_move) < 0.15 or (abs(_peak_move) >= 0.3 and abs(_long_move) < abs(_peak_move) * 0.5):
                # Baseline reversion — the grind has died. Two conditions:
                # (a) Price drifted back to within 0.15% of baseline (flat/reversed), OR
                # (b) Price retraced >50% of its peak move from baseline (momentum fading).
                # Old logic: reset at 0.25% flat — killed legitimate grinds with normal noise.
                self._baselines_long[symbol] = {'price': ltp, 'ts': now, 'peak_move': 0.0, 'min_price': ltp, 'max_price': ltp}
                self._grind_start_vol[symbol] = _cum_vol
                _hist = self._vol_delta_history.get(symbol)
                self._grind_start_avg_delta[symbol] = (sum(_hist) / len(_hist)) if _hist and len(_hist) >= 3 else 0
            elif _age > 600:
                # Stale protection: max 10 min baseline age.
                # But if the stock is actively trending (move > threshold),
                # extend to 30 min — this IS the trend we want to track.
                _still_trending = abs(_long_move) >= self._slow_grind_pct
                _stale_limit = 1800 if _still_trending else 600
                if _age > _stale_limit:
                    self._baselines_long[symbol] = {'price': ltp, 'ts': now, 'peak_move': 0.0, 'min_price': ltp, 'max_price': ltp}
                    self._grind_start_vol[symbol] = _cum_vol
                    _hist = self._vol_delta_history.get(symbol)
                    self._grind_start_avg_delta[symbol] = (sum(_hist) / len(_hist)) if _hist and len(_hist) >= 3 else 0
        
        # If short baseline was just reset, skip trigger detection this tick
        if bl is None or (now - self._baselines[symbol]['ts']) < 0.01:
            return
        
        # === CHECK SUSTAIN PENDING ===
        pending = self._pending.get(symbol)
        if pending:
            elapsed = now - pending['trigger_ts']
            baseline_price = pending['baseline_price']
            if baseline_price > 0:
                _cur_move = abs(ltp - baseline_price) / baseline_price * 100
                # Track peak move during sustain window for retrace detection
                _prev_peak = pending.get('_peak_move_pct', 0)
                if _cur_move > _prev_peak:
                    pending['_peak_move_pct'] = _cur_move
            # Signal-aware sustain time: stronger signals need less proof
            _ttype_s = pending.get('trigger_type', '')
            if 'EARLYBIRD' in _ttype_s:
                _effective_sustain = pending.get('_earlybird_sustain', 10)
            elif 'GRIND' in _ttype_s:
                _effective_sustain = self._sustain_secs_grind
            elif 'DAY' in _ttype_s:
                _effective_sustain = self._sustain_secs_extreme
            elif 'VOLUME' in _ttype_s:
                _effective_sustain = self._sustain_secs_volume
            else:
                _effective_sustain = self._sustain_secs
            if elapsed >= _effective_sustain:
                # Time's up — check if price still holds
                move_pct = round(abs(ltp - baseline_price) / baseline_price * 100, 1)
                # Volume surges use a lower sustain bar (they just need price not to crash)
                _ttype = pending.get('trigger_type', '')
                # Earlybird uses per-mode sustain hold threshold
                if 'EARLYBIRD' in _ttype:
                    _eb_mode = pending.get('earlybird_mode', 'B')
                    if _eb_mode == 'A':
                        _recheck = self._eb_a.get('sustain_min_hold_pct', 0.4)
                    elif _eb_mode == 'C':
                        _recheck = self._eb_c.get('sustain_min_hold_pct', 0.7)
                    else:
                        _recheck = self._eb_b.get('sustain_min_hold_pct', 0.5)
                elif 'VOLUME' in _ttype:
                    _recheck = self._sustain_recheck_pct_volume
                else:
                    _recheck = self._sustain_recheck_pct
                # Early market hardening: require larger sustained move before 09:55
                # NOTE: Earlybird triggers SKIP early market hardening (they ARE early market)
                if 'EARLYBIRD' not in _ttype:
                    _em_end = self._config.get('early_market_end', '09:55')
                    _em_h, _em_m = int(_em_end.split(':')[0]), int(_em_end.split(':')[1])
                    _now_dt = datetime.now()
                    if _now_dt.hour < _em_h or (_now_dt.hour == _em_h and _now_dt.minute < _em_m):
                        _em_min_sustain = self._config.get('early_market_min_sustain_pct', 1.0)
                        _recheck = max(_recheck, _em_min_sustain)
                _peak_move = pending.get('_peak_move_pct', move_pct)
                # Anti-retrace: if price retraced >50% of its peak move, the move is fading
                _retrace_max = self._config.get('sustain_retrace_max_pct', 50.0)
                _retraced_pct = ((1 - move_pct / _peak_move) * 100) if _peak_move > 0 else 0
                _retrace_fail = _peak_move > 0 and _retraced_pct > _retrace_max

                # === OI FORWARD-LOOKING CONFIRMATION ===
                # Check async OI prefetch result (launched when trigger entered sustain).
                # OI buildup in the SAME direction as price move = forward confirmation
                # that institutional money supports this move → ease the sustain bar.
                # OI buildup AGAINST the price move = contradiction → tighten the bar.
                _oi_confirm = self._get_oi_confirmation(symbol)
                _oi_tag = ''
                _oi_adjusted = False
                if _oi_confirm and _oi_confirm.get('strength', 0) >= self._oi_min_strength_for_confirm:
                    _oi_sig = _oi_confirm.get('signal', 'NEUTRAL')
                    _oi_str = _oi_confirm.get('strength', 0)
                    if _oi_confirm.get('confirms'):
                        # OI confirms the price move — writers/buyers building in the
                        # same direction.  This is forward-looking evidence that the
                        # move will continue.  Ease the sustain recheck threshold.
                        _recheck = max(0.1, _recheck - self._oi_confirm_sustain_ease)
                        _oi_tag = f' OI✓{_oi_sig}({_oi_str:.2f})'
                        _oi_adjusted = True
                        pending['_oi_confirmed'] = True
                        pending['_oi_signal'] = _oi_sig
                        pending['_oi_strength'] = _oi_str
                        pending['_oi_participant'] = _oi_confirm.get('participant', '')
                    elif _oi_confirm.get('contradicts'):
                        # OI contradicts the price move — institutional money is
                        # positioned against this direction.  Raise the bar.
                        _recheck = _recheck + self._oi_contradict_sustain_tighten
                        _oi_tag = f' OI✗{_oi_sig}({_oi_str:.2f})'
                        _oi_adjusted = True
                        pending['_oi_contradicted'] = True
                        pending['_oi_signal'] = _oi_sig
                        pending['_oi_strength'] = _oi_str
                elif _oi_confirm:
                    # OI data available but strength too low to matter
                    _oi_tag = f' OI~{_oi_confirm.get("signal", "?")}(weak)'

                if move_pct >= _recheck and not _retrace_fail:
                    # SUSTAINED — push to queue
                    # Record the sustained move and peak for exhaustion analysis downstream
                    pending['_sustain_held_pct'] = move_pct
                    self._stats['sustain_passed'] += 1
                    if _oi_adjusted and _oi_confirm and _oi_confirm.get('confirms'):
                        self._stats['oi_confirm_sustain_passed'] = self._stats.get('oi_confirm_sustain_passed', 0) + 1
                    self._log_sustain_count += 1
                    if self._log_sustain_count <= self._LOG_SUSTAIN_MAX:
                        print(f"   ✅ Watcher: {symbol} SUSTAINED {_ttype} ({move_pct:.1f}% held vs {_recheck}%, peak={_peak_move:.1f}%, retrace={_retraced_pct:.0f}%{_oi_tag}) — queuing")
                    else:
                        self._log_sustain_suppressed.append(symbol.replace('NSE:', ''))
                    self._cleanup_oi_prefetch(symbol)
                    self._fire_trigger(symbol, ltp, pending['trigger_type'], pending)
                else:
                    # Failed sustain — retrace (batch-logged to reduce noise)
                    self._stats['sustain_failed'] += 1
                    if _oi_adjusted and _oi_confirm and _oi_confirm.get('contradicts'):
                        self._stats['oi_contradict_sustain_failed'] = self._stats.get('oi_contradict_sustain_failed', 0) + 1
                    _fail_reason = f'retrace {_retraced_pct:.0f}%>{_retrace_max:.0f}%' if _retrace_fail else f'{move_pct:.1f}%<{_recheck}%'
                    if 'VOLUME' not in _ttype:
                        _pf = self._stats.get('_price_sustain_fails', 0) + 1
                        self._stats['_price_sustain_fails'] = _pf
                        if _pf <= 3 or _pf % 25 == 0:
                            print(f"   ❌ Watcher: {symbol} sustain FAILED {_ttype} ({_fail_reason}, peak={_peak_move:.1f}%{_oi_tag}) [#{_pf}]")
                    else:
                        _vf = self._stats.get('_vol_sustain_fails', 0) + 1
                        self._stats['_vol_sustain_fails'] = _vf
                        if _vf % 50 == 0:
                            print(f"   📉 Watcher: {_vf} volume-surge sustain fails so far (latest: {symbol} {_fail_reason}{_oi_tag})")
                    self._cleanup_oi_prefetch(symbol)
                del self._pending[symbol]
            return  # While pending, don't check new triggers for this symbol
        
        # === EARLYBIRD DETECTION (09:16-09:45 opening volatility) ===
        # Three modes tracked separately:
        #   A = Gap continuation (highest quality, larger sizing)
        #   B = Strong opening directional move (medium quality)
        #   C = Opening spike (lowest quality, stricter sustain)
        # Market-context: compare stock move vs NIFTY 50 to detect beta vs idiosyncratic.
        _earlybird_triggered = False
        if self._earlybird_enabled:
            from datetime import time as _eb_time
            _eb_com = self._eb_common
            _eb_now_t = datetime.now().time()
            _eb_start_parts = _eb_com.get('start_time', '09:16').split(':')
            _eb_end_parts = _eb_com.get('end_time', '09:45').split(':')
            _eb_start = _eb_time(int(_eb_start_parts[0]), int(_eb_start_parts[1]))
            _eb_end = _eb_time(int(_eb_end_parts[0]), int(_eb_end_parts[1]))
            
            if _eb_start <= _eb_now_t <= _eb_end:
                _eb_prev_close = self._prev_close.get(symbol, 0)
                _eb_day_open = self._day_open.get(symbol, 0)
                _eb_baseline = bl['price'] if bl else 0
                
                if _eb_prev_close > 0 and _eb_day_open > 0 and _eb_baseline > 0:
                    # Core price metrics
                    _gap_pct = (ltp - _eb_prev_close) / _eb_prev_close * 100
                    _open_move_pct = (ltp - _eb_day_open) / _eb_day_open * 100
                    _eb_move_from_bl = (ltp - _eb_baseline) / _eb_baseline * 100
                    
                    # --- Market Context: NIFTY 50 change from open ---
                    _nifty_change = 0.0
                    _is_idiosyncratic = False
                    _is_beta_driven = False
                    if self._nifty_day_open > 0 and self._nifty_ltp > 0:
                        _nifty_change = (self._nifty_ltp - self._nifty_day_open) / self._nifty_day_open * 100
                        _idx_thresh = _eb_com.get('index_same_dir_threshold', 0.3)
                        # Stock and index moving same direction both ≥ threshold = beta-driven
                        if abs(_nifty_change) >= _idx_thresh and abs(_open_move_pct) >= 0.3:
                            if (_nifty_change > 0 and _open_move_pct > 0) or (_nifty_change < 0 and _open_move_pct < 0):
                                _is_beta_driven = True
                        # Stock moving but index flat/opposite = idiosyncratic
                        if abs(_open_move_pct) >= 0.5 and (abs(_nifty_change) < _idx_thresh or
                            (_nifty_change > 0 and _open_move_pct < 0) or (_nifty_change < 0 and _open_move_pct > 0)):
                            _is_idiosyncratic = True
                    
                    # Volume check (shared baseline — modes can override)
                    _eb_base_vol_ratio = _eb_com.get('min_vol_ratio', 1.5)
                    _eb_vol_ok = True  # Assume OK if not enough history
                    _eb_vol_ratio_actual = 0.0
                    if len(self._vol_delta_history[symbol]) >= 3:
                        _eb_avg_vol = sum(self._vol_delta_history[symbol]) / len(self._vol_delta_history[symbol])
                        if _eb_avg_vol > 0:
                            _eb_vol_ratio_actual = _vol_delta / _eb_avg_vol
                    
                    _has_gap = abs(_gap_pct) >= self._eb_a.get('gap_open_pct', 0.8)
                    _strong_gap = abs(_gap_pct) >= self._eb_a.get('gap_strong_pct', 1.5)
                    
                    # --- MODE A: Gap Continuation ---
                    _eb_a_cfg = self._eb_a
                    if symbol not in self._earlybird_fired['A'] and _has_gap:
                        _gap_cont_min = _eb_a_cfg.get('gap_continuation_min', 0.35)
                        # Gap direction must match current move direction AND price continues
                        if ((_gap_pct > 0 and _open_move_pct >= _gap_cont_min) or
                            (_gap_pct < 0 and _open_move_pct <= -_gap_cont_min)):
                            # Volume check for mode A (uses common threshold)
                            if _eb_vol_ratio_actual >= _eb_base_vol_ratio or len(self._vol_delta_history[symbol]) < 3:
                                _eb_dir_move = _open_move_pct
                                _eb_ttype = 'EARLYBIRD_A_UP' if _eb_dir_move > 0 else 'EARLYBIRD_A_DOWN'
                                _eb_sustain = _eb_a_cfg.get('sustain_seconds', 8)
                                _eb_reason = f'GAP({_gap_pct:+.1f}%)+CONT({_open_move_pct:+.1f}%)'
                                self._pending[symbol] = {
                                    'trigger_price': ltp, 'trigger_ts': now,
                                    'trigger_type': _eb_ttype, 'baseline_price': _eb_baseline,
                                    'move_pct': _eb_dir_move,
                                    'gap_pct': round(_gap_pct, 2), 'open_move_pct': round(_open_move_pct, 2),
                                    'has_gap': True, 'strong_gap': _strong_gap,
                                    'earlybird_mode': 'A', 'earlybird_reason': _eb_reason,
                                    '_earlybird_sustain': _eb_sustain,
                                    'nifty_change_pct': round(_nifty_change, 2),
                                    'is_idiosyncratic': _is_idiosyncratic,
                                    'is_beta_driven': _is_beta_driven,
                                }
                                _earlybird_triggered = True
                                self._earlybird_fired['A'].add(symbol)
                                self._earlybird_trades_fired += 1
                                self._stats['earlybird_a_detected'] = self._stats.get('earlybird_a_detected', 0) + 1
                                _ctx_tag = '★IDIO' if _is_idiosyncratic else ('βETA' if _is_beta_driven else 'NEUTRAL')
                                print(f"   🐦A Watcher: {symbol} {_eb_ttype} ({_eb_reason}) nifty={_nifty_change:+.1f}% [{_ctx_tag}] → sustain {_eb_sustain}s")
                    
                    # --- MODE B: Strong Opening Move (only if A didn't fire) ---
                    _eb_b_cfg = self._eb_b
                    if not _earlybird_triggered and symbol not in self._earlybird_fired['B']:
                        _eb_open_move_thresh = _eb_b_cfg.get('opening_move_pct', 0.7)
                        if abs(_open_move_pct) >= _eb_open_move_thresh:
                            # Volume check for mode B
                            if _eb_vol_ratio_actual >= _eb_base_vol_ratio or len(self._vol_delta_history[symbol]) < 3:
                                _eb_dir_move = _open_move_pct
                                _eb_ttype = 'EARLYBIRD_B_UP' if _eb_dir_move > 0 else 'EARLYBIRD_B_DOWN'
                                _eb_sustain = _eb_b_cfg.get('sustain_seconds', 10)
                                _eb_reason = f'OPEN_MOVE({_open_move_pct:+.1f}%)'
                                self._pending[symbol] = {
                                    'trigger_price': ltp, 'trigger_ts': now,
                                    'trigger_type': _eb_ttype, 'baseline_price': _eb_baseline,
                                    'move_pct': _eb_dir_move,
                                    'gap_pct': round(_gap_pct, 2), 'open_move_pct': round(_open_move_pct, 2),
                                    'has_gap': _has_gap, 'strong_gap': _strong_gap,
                                    'earlybird_mode': 'B', 'earlybird_reason': _eb_reason,
                                    '_earlybird_sustain': _eb_sustain,
                                    'nifty_change_pct': round(_nifty_change, 2),
                                    'is_idiosyncratic': _is_idiosyncratic,
                                    'is_beta_driven': _is_beta_driven,
                                }
                                _earlybird_triggered = True
                                self._earlybird_fired['B'].add(symbol)
                                self._earlybird_trades_fired += 1
                                self._stats['earlybird_b_detected'] = self._stats.get('earlybird_b_detected', 0) + 1
                                _ctx_tag = '★IDIO' if _is_idiosyncratic else ('βETA' if _is_beta_driven else 'NEUTRAL')
                                print(f"   🐦B Watcher: {symbol} {_eb_ttype} ({_eb_reason}) nifty={_nifty_change:+.1f}% [{_ctx_tag}] → sustain {_eb_sustain}s")
                    
                    # --- MODE C: Opening Spike (only if A and B didn't fire) ---
                    _eb_c_cfg = self._eb_c
                    if not _earlybird_triggered and symbol not in self._earlybird_fired['C']:
                        _eb_spike_thresh = _eb_c_cfg.get('opening_spike_pct', 1.2)
                        if abs(_eb_move_from_bl) >= _eb_spike_thresh:
                            # Mode C requires stricter volume
                            _eb_c_vol_ratio = _eb_c_cfg.get('min_vol_ratio', 2.0)
                            if _eb_vol_ratio_actual >= _eb_c_vol_ratio or len(self._vol_delta_history[symbol]) < 3:
                                _eb_dir_move = _eb_move_from_bl
                                _eb_ttype = 'EARLYBIRD_C_UP' if _eb_dir_move > 0 else 'EARLYBIRD_C_DOWN'
                                _eb_sustain = _eb_c_cfg.get('sustain_seconds', 15)
                                _eb_reason = f'OPEN_SPIKE({_eb_move_from_bl:+.1f}%)'
                                self._pending[symbol] = {
                                    'trigger_price': ltp, 'trigger_ts': now,
                                    'trigger_type': _eb_ttype, 'baseline_price': _eb_baseline,
                                    'move_pct': _eb_dir_move,
                                    'gap_pct': round(_gap_pct, 2), 'open_move_pct': round(_open_move_pct, 2),
                                    'has_gap': _has_gap, 'strong_gap': _strong_gap,
                                    'earlybird_mode': 'C', 'earlybird_reason': _eb_reason,
                                    '_earlybird_sustain': _eb_sustain,
                                    'nifty_change_pct': round(_nifty_change, 2),
                                    'is_idiosyncratic': _is_idiosyncratic,
                                    'is_beta_driven': _is_beta_driven,
                                }
                                _earlybird_triggered = True
                                self._earlybird_fired['C'].add(symbol)
                                self._earlybird_trades_fired += 1
                                self._stats['earlybird_c_detected'] = self._stats.get('earlybird_c_detected', 0) + 1
                                _ctx_tag = '★IDIO' if _is_idiosyncratic else ('βETA' if _is_beta_driven else 'NEUTRAL')
                                print(f"   🐦C Watcher: {symbol} {_eb_ttype} ({_eb_reason}) nifty={_nifty_change:+.1f}% [{_ctx_tag}] → sustain {_eb_sustain}s")
        
        if _earlybird_triggered:
            return  # Earlybird entered sustain — skip normal trigger detection this tick
        
        # === DETECT TRIGGERS ===
        baseline_price = bl['price']
        if baseline_price <= 0:
            return
        
        move_pct = (ltp - baseline_price) / baseline_price * 100
        trigger_type = None
        _spike_plus_surge = False
        _spike_surge_ratio = 0.0
        
        # Track recent prices for acceleration detection
        self._recent_prices[symbol].append((now, ltp))
        
        # 1) PRICE SPIKE: moved ≥ spike_pct from baseline
        #    [FIX Mar 10] Add volume confirmation — reject spikes on dry volume.
        #    Add acceleration tracking and spike magnitude to trigger metadata.
        #    [FIX Mar 16] Time-aware threshold: higher bar before 09:45 (opening noise)
        _open_spike_pct = self._config.get('price_spike_pct_open', 1.0)
        _open_cutoff = self._config.get('price_spike_open_until', '09:45')
        _h, _m = int(_open_cutoff.split(':')[0]), int(_open_cutoff.split(':')[1])
        _now_t = datetime.now()
        _effective_spike_pct = _open_spike_pct if (_now_t.hour < _h or (_now_t.hour == _h and _now_t.minute < _m)) else self._spike_pct
        if abs(move_pct) >= _effective_spike_pct:
            # Volume confirmation: require current vol delta ≥ 50% of rolling avg
            # Prevents fake spikes from illiquid ticks or stale-price jumps
            _spike_vol_ok = True
            if len(self._vol_delta_history[symbol]) >= 3:
                _s_avg = sum(self._vol_delta_history[symbol]) / len(self._vol_delta_history[symbol])
                if _s_avg > 0 and _vol_delta < _s_avg * 0.5:
                    _spike_vol_ok = False  # Spike on dry volume — likely fake
            
            if _spike_vol_ok:
                trigger_type = 'PRICE_SPIKE_UP' if move_pct > 0 else 'PRICE_SPIKE_DOWN'
                self._stats['spikes_detected'] += 1
                # Compute acceleration: compare speed of last 5 ticks vs prior 5
                _spike_accel = 0.0
                _rp = list(self._recent_prices[symbol])
                if len(_rp) >= 6:
                    _mid = len(_rp) // 2
                    _recent_half = _rp[_mid:]
                    _older_half = _rp[:_mid]
                    if len(_recent_half) >= 2 and len(_older_half) >= 2:
                        _r_speed = abs(_recent_half[-1][1] - _recent_half[0][1]) / max(0.1, _recent_half[-1][0] - _recent_half[0][0])
                        _o_speed = abs(_older_half[-1][1] - _older_half[0][1]) / max(0.1, _older_half[-1][0] - _older_half[0][0])
                        if _o_speed > 0:
                            _spike_accel = round(_r_speed / _o_speed, 2)  # >1 = accelerating
                
                # === SPIKE + SURGE CO-FIRE: check if VOLUME_SURGE also met ===
                # When both conditions fire simultaneously, trade gets double lot.
                # PRICE_SPIKE preempts VOLUME_SURGE by priority, but if volume is
                # ALSO surging, it signals massive institutional conviction → 2x sizing.
                _spike_plus_surge = False
                _spike_surge_ratio = 0.0
                if _vol_delta > 0 and len(self._vol_delta_history[symbol]) >= 5:
                    _sps_hist = self._vol_delta_history[symbol]
                    _sps_avg = sum(_sps_hist) / len(_sps_hist)
                    _sps_warmup = (now - self._init_ts) < 600
                    _sps_mult = max(1.8, self._vol_surge_x * 0.6) if _sps_warmup else self._vol_surge_x
                    if _sps_avg > 0 and _vol_delta >= _sps_avg * _sps_mult:
                        _sps_recent_3 = list(_sps_hist)[-3:] if len(_sps_hist) >= 3 else list(_sps_hist)
                        _sps_elev = sum(1 for d in _sps_recent_3 if d >= _sps_avg * 2.0)
                        if _sps_elev >= 2:
                            _spike_plus_surge = True
                            _spike_surge_ratio = round(_vol_delta / _sps_avg, 1)
        
        # 2) VOLUME SURGE: current tick's volume DELTA ≥ N× rolling average delta
        #    [FIX Mar 6] Also require minimum price move — explosive volume with
        #    only 0.1% move = absorption, not breakout. BDL had EXPLOSIVE vol but
        #    tiny move, reversed in 4 min.
        #    [FIX Mar 10] Require 2+ consecutive elevated ticks to filter single-tick anomalies.
        #    Add surge_ratio and depth_imbalance to trigger data for pipeline scoring.
        #    [FIX Mar 17] Promoted to #2 priority (was #3 behind DAY_EXT). Volume surge
        #    signals institutional conviction — should not be preempted by day extremes.
        if not trigger_type and _vol_delta > 0 and len(self._vol_delta_history[symbol]) >= 5:
            _hist = self._vol_delta_history[symbol]
            _avg_delta = sum(_hist) / len(_hist)
            # Warmup: first 10 min after restart, baselines are thin — use lower multiplier
            _warmup = (now - self._init_ts) < 600
            _surge_mult = max(1.8, self._vol_surge_x * 0.6) if _warmup else self._vol_surge_x
            if _avg_delta > 0 and _vol_delta >= _avg_delta * _surge_mult:
                if abs(move_pct) >= self._vol_surge_min_move:
                    # Consecutive tick check: at least 2 of last 3 ticks must be ≥ 2x avg
                    # Prevents single-tick glitches from triggering
                    _recent_3 = list(_hist)[-3:] if len(_hist) >= 3 else list(_hist)
                    _elevated_count = sum(1 for d in _recent_3 if d >= _avg_delta * 2.0)
                    if _elevated_count >= 2:
                        trigger_type = 'VOLUME_SURGE'
                        self._stats['vol_surges_detected'] += 1
                        # Compute surge ratio and depth imbalance for pipeline
                        _surge_ratio = round(_vol_delta / _avg_delta, 1) if _avg_delta > 0 else 0
                        _buy_qty = tick.get('buy_quantity', 0)
                        _sell_qty = tick.get('sell_quantity', 0)
                        _depth_imbalance = 0.0
                        if _buy_qty + _sell_qty > 0:
                            _depth_imbalance = round((_buy_qty - _sell_qty) / (_buy_qty + _sell_qty), 3)
                # else: volume spike but price hasn't moved enough — absorption, skip
        
        # 3) DAY EXTREME: exchange-reported ohlc.high/low INCREASED since last tick
        #    [FIX Mar 10] Track break count — 1st/2nd break = strong, 5th+ = noise.
        #    Require margin above old extreme (not just ₹0.01). Check volume on break.
        #    [FIX Mar 17] Demoted to #3 priority (was #2). Volume surge takes precedence.
        #    Always update tracked high/low values even if another trigger already fired.
        #    [FIX Mar 17b] Hardened: raised break margin, require vol ≥1.2x avg,
        #    wick-body check (LTP must still be above old extreme), late-day margin 2x.
        #    [FIX Mar 17c] Consolidation lookback: track quiet time since last break.
        _day_ext_quiet_min = 60.0  # default if not computed
        if self._day_extreme and _tick_day_high > 0 and _tick_day_low > 0:
            _prev_h = self._prev_reported_high.get(symbol, 0)
            _prev_l = self._prev_reported_low.get(symbol, 0)
            _now_ext = datetime.now()
            _late_day = _now_ext.hour >= 14  # after 14:00 — require higher margin
            
            if not trigger_type and _prev_h > 0 and _tick_day_high > _prev_h and abs(move_pct) >= self._day_ext_min_move:
                _break_margin = (_tick_day_high - _prev_h) / _prev_h * 100 if _prev_h > 0 else 0
                _hbc = self._day_high_break_count.get(symbol, 0) + 1
                self._day_high_break_count[symbol] = _hbc
                # Break margin gate: 0.12% for fresh breaks, 0.20% for repeated
                # Late-day (after 14:00): double the margin — thin-volume extensions are traps
                _base_margin = 0.12 if _hbc <= 3 else 0.20
                _margin_ok = _break_margin >= (_base_margin * 2 if _late_day else _base_margin)
                # Volume gate: require vol ≥ 1.2× average (not just "not dead")
                _ext_vol_ok = True
                if len(self._vol_delta_history[symbol]) >= 3:
                    _e_avg = sum(self._vol_delta_history[symbol]) / len(self._vol_delta_history[symbol])
                    if _e_avg > 0 and _vol_delta < _e_avg * 1.2:
                        _ext_vol_ok = False  # Breakout on below-average volume — skip
                # Wick-body check: LTP must still be AT or ABOVE the OLD high
                # If ohlc.high made a new high but LTP already retraced below prev high,
                # the break was a wick (intra-tick spike that reversed) — not a real breakout
                _wick_ok = ltp >= _prev_h
                if _margin_ok and _ext_vol_ok and _wick_ok:
                    trigger_type = 'NEW_DAY_HIGH'
                    self._stats['extremes_detected'] += 1
                    _day_ext_break_count = _hbc
                    _day_ext_break_margin = round(_break_margin, 3)
                    # Consolidation lookback: how long since the last day-high break?
                    # Market opens 09:15 IST — use that as reference for first break
                    _mkt_open_ts = datetime.now().replace(hour=9, minute=15, second=0).timestamp()
                    _last_h_ts = self._last_day_high_break_ts.get(symbol, _mkt_open_ts)
                    _day_ext_quiet_min = round((now - _last_h_ts) / 60, 1)
                    self._last_day_high_break_ts[symbol] = now
            elif not trigger_type and _prev_l > 0 and _tick_day_low < _prev_l and abs(move_pct) >= self._day_ext_min_move:
                _break_margin = (_prev_l - _tick_day_low) / _prev_l * 100 if _prev_l > 0 else 0
                _lbc = self._day_low_break_count.get(symbol, 0) + 1
                self._day_low_break_count[symbol] = _lbc
                _base_margin = 0.12 if _lbc <= 3 else 0.20
                _margin_ok = _break_margin >= (_base_margin * 2 if _late_day else _base_margin)
                _ext_vol_ok = True
                if len(self._vol_delta_history[symbol]) >= 3:
                    _e_avg = sum(self._vol_delta_history[symbol]) / len(self._vol_delta_history[symbol])
                    if _e_avg > 0 and _vol_delta < _e_avg * 1.2:
                        _ext_vol_ok = False
                # Wick-body check: LTP must still be AT or BELOW the OLD low
                _wick_ok = ltp <= _prev_l
                if _margin_ok and _ext_vol_ok and _wick_ok:
                    trigger_type = 'NEW_DAY_LOW'
                    self._stats['extremes_detected'] += 1
                    _day_ext_break_count = _lbc
                    _day_ext_break_margin = round(_break_margin, 3)
                    _mkt_open_ts = datetime.now().replace(hour=9, minute=15, second=0).timestamp()
                    _last_l_ts = self._last_day_low_break_ts.get(symbol, _mkt_open_ts)
                    _day_ext_quiet_min = round((now - _last_l_ts) / 60, 1)
                    self._last_day_low_break_ts[symbol] = now
            # Always update the tracked values
            self._prev_reported_high[symbol] = _tick_day_high
            self._prev_reported_low[symbol] = _tick_day_low
        
        # === ENTER SUSTAIN PHASE ===
        if trigger_type:
            # Build trigger metadata — enriched for pipeline scoring
            _trigger_meta = {
                'trigger_price': ltp,
                'trigger_ts': now,
                'trigger_type': trigger_type,
                'baseline_price': baseline_price,
                'move_pct': move_pct,
            }
            # Add depth imbalance for all triggers
            _buy_qty = tick.get('buy_quantity', 0)
            _sell_qty = tick.get('sell_quantity', 0)
            if _buy_qty + _sell_qty > 0:
                _trigger_meta['depth_imbalance'] = round((_buy_qty - _sell_qty) / (_buy_qty + _sell_qty), 3)
            # PRICE_SPIKE: add acceleration and spike magnitude
            if 'SPIKE' in trigger_type:
                _trigger_meta['spike_accel'] = _spike_accel
                _trigger_meta['spike_magnitude'] = round(abs(move_pct), 2)
                # Co-fire: PRICE_SPIKE + VOLUME_SURGE both met → double lot signal
                if _spike_plus_surge:
                    _trigger_meta['spike_plus_surge'] = True
                    _trigger_meta['surge_ratio'] = _spike_surge_ratio
            # VOLUME_SURGE: add surge ratio
            elif trigger_type == 'VOLUME_SURGE':
                _trigger_meta['surge_ratio'] = _surge_ratio
            # NEW_DAY_HIGH/LOW: add break count, margin, and consolidation quiet time
            elif 'DAY' in trigger_type:
                _trigger_meta['break_count'] = _day_ext_break_count
                _trigger_meta['break_margin'] = _day_ext_break_margin
                _trigger_meta['quiet_minutes'] = _day_ext_quiet_min
            self._pending[symbol] = _trigger_meta
            # === OI FORWARD-LOOKING PREFETCH ===
            # Launch async OI fetch NOW so it's ready by the time sustain completes.
            # The 15-60s sustain window gives plenty of time for the OI HTTP call.
            _move_dir = 'UP' if move_pct > 0 else 'DOWN'
            self._launch_oi_prefetch(symbol, _move_dir)
    
    def _fire_trigger(self, symbol: str, ltp: float, trigger_type: str, pending: dict):
        """Push a confirmed trigger to the priority queue after passing gates.
        
        Priority = abs(move_pct).  When queue is full, the WEAKEST trigger
        is evicted — ensuring large movers are never dropped by noise.
        
        Big moves (≥ priority_bypass_pct) bypass the rate limit so they
        ALWAYS enter the queue even during flood conditions (crash days).
        """
        # NOTE: No active-window gate here — let triggers queue up freely.
        # The drain gate (watcher_start in _process_breakout_triggers) controls
        # when queued triggers are actually processed. This prevents losing
        # valid pre-09:20 sustained triggers that would otherwise be silently dropped.
        
        # Gate: cooldown (per-symbol) — with priority escalation
        # Higher-priority trigger types can bypass cooldown set by weaker triggers.
        # Priority order: PRICE_SPIKE > VOLUME_SURGE > SLOW_GRIND > NEW_DAY_LOW/HIGH
        # [FIX Mar 17] Volume surge promoted above grind and day-ext (institutional signal)
        _TRIGGER_PRIORITY = {
            'NEW_DAY_HIGH': 1, 'NEW_DAY_LOW': 1,
            'SLOW_GRIND_UP': 2, 'SLOW_GRIND_DOWN': 2,
            'VOLUME_SURGE': 3,
            'PRICE_SPIKE_UP': 4, 'PRICE_SPIKE_DOWN': 4,
            'EARLYBIRD_UP': 5, 'EARLYBIRD_DOWN': 5,
            'EARLYBIRD_C_UP': 5, 'EARLYBIRD_C_DOWN': 5,
            'EARLYBIRD_B_UP': 6, 'EARLYBIRD_B_DOWN': 6,
            'EARLYBIRD_A_UP': 7, 'EARLYBIRD_A_DOWN': 7,
        }
        if not self._is_cooled_down(symbol):
            # Check if this trigger outranks the one that set the cooldown
            _new_priority = _TRIGGER_PRIORITY.get(trigger_type, 1)
            _prev_type = self._cooldown_trigger_type.get(symbol, 'VOLUME_SURGE')
            _prev_priority = _TRIGGER_PRIORITY.get(_prev_type, 1)
            if _new_priority > _prev_priority:
                print(f"   🔄 Watcher: {symbol} ESCALATING through cooldown "
                      f"({trigger_type}[p{_new_priority}] > {_prev_type}[p{_prev_priority}])")
            else:
                self._stats['cooldown_blocked'] += 1
                _cb = self._stats['cooldown_blocked']
                if _cb <= 3 or _cb % 25 == 0:
                    print(f"   ⏳ Watcher: {symbol} blocked by cooldown — skipping [#{_cb}]")
                return
        
        move_pct_abs = abs(pending.get('move_pct', 0))
        
        # Gate: rate limit — BIG moves bypass this entirely
        if not self._check_rate_limit(symbol):
            if move_pct_abs < self._priority_bypass_pct:
                self._stats['rate_limited'] += 1
                return
            # Big move → bypass rate limit
            print(f"   🔥 Watcher: {symbol} BYPASSING rate limit "
                  f"(move={move_pct_abs:.1f}% ≥ {self._priority_bypass_pct}% threshold)")
        
        # All gates passed — queue with priority
        now = time.time()
        self._cooldowns[symbol] = now
        self._cooldown_trigger_type[symbol] = trigger_type  # Track what type set this cooldown
        self._recent_triggers.append(now)
        # Track per-symbol triggers for per-symbol rate limit
        if symbol not in self._per_symbol_triggers:
            self._per_symbol_triggers[symbol] = []
        self._per_symbol_triggers[symbol].append(now)
        
        trigger_data = {
            'symbol': symbol,
            'ltp': ltp,
            'trigger_type': trigger_type,
            'move_pct': pending.get('move_pct', 0),
            'baseline_price': pending.get('baseline_price', 0),
            'trigger_price': pending.get('trigger_price', 0),
            'detected_at': now,
            'detected_time': datetime.now().strftime('%H:%M:%S'),
        }
        # Pass through enriched metadata from process_tick
        for _meta_key in ('velocity', 'vol_confirmed', 'vol_ratio', 'grind_age_s',
                          'intraday_pct', 'day_range_position',
                          'peak_velocity', 'sb_move_pct',
                          'surge_ratio', 'depth_imbalance',
                          'spike_accel', 'spike_magnitude', 'spike_plus_surge',
                          'break_count', 'break_margin',
                          '_peak_move_pct', '_sustain_held_pct',
                          '_oi_confirmed', '_oi_contradicted',
                          '_oi_signal', '_oi_strength', '_oi_participant',
                          'gap_pct', 'open_move_pct', 'has_gap', 'strong_gap',
                          'earlybird_reason'):
            if _meta_key in pending:
                trigger_data[_meta_key] = pending[_meta_key]
        
        # Tag triggers during post-restart warmup (baselines are thin)
        if self._post_restart:
            trigger_data['post_restart'] = True
        
        priority = move_pct_abs
        # OI confirmation boost: if OI confirms the move, give it priority uplift
        # so OI-confirmed triggers are processed before unconfirmed ones.
        if pending.get('_oi_confirmed') and pending.get('_oi_strength', 0) >= 0.50:
            priority = priority * 1.25  # 25% priority boost for strong OI confirmation
        added, evicted = self._queue.put(trigger_data, priority)
        
        if added:
            self._stats['queued'] += 1
            if evicted:
                print(f"   📤 Watcher: {symbol} QUEUED (evicted weaker {evicted}) — "
                      f"{trigger_type}, {pending.get('move_pct', 0):+.1f}%")
            else:
                _qc = self._stats['queued']
                if _qc <= 5 or _qc % 25 == 0:
                    print(f"   📤 Watcher: {symbol} QUEUED for main thread "
                          f"({trigger_type}, {pending.get('move_pct', 0):+.1f}%) [#{_qc}]")
        else:
            self._stats['priority_dropped'] = self._stats.get('priority_dropped', 0) + 1
            print(f"   ⚠️ Watcher: {symbol} NOT QUEUED — weaker than all "
                  f"{self._queue.qsize} queued items (move={move_pct_abs:.1f}%)")
    
    def drain_queue(self) -> List[Dict]:
        """
        Drain all pending triggers from the priority queue.
        Called from the MAIN THREAD only (between scan cycles).
        
        Returns:
            List of trigger dicts sorted by priority (biggest move first).
        """
        return self._queue.drain()
    
    def set_enabled(self, enabled: bool):
        """Enable/disable the watcher at runtime"""
        self._enabled = enabled
    
    @property 
    def stats(self) -> dict:
        return dict(self._stats)
    
    def reset_day(self):
        """Reset daily state (call at market open)"""
        self._baselines.clear()
        self._baselines_long.clear()
        self._pending.clear()
        self._cooldowns.clear()
        self._cooldown_trigger_type.clear()
        self._per_symbol_triggers.clear()
        self._vol_delta_history.clear()
        self._prev_cumulative_vol.clear()
        self._prev_reported_high.clear()
        self._prev_reported_low.clear()
        self._day_high_break_count.clear()
        self._day_low_break_count.clear()
        self._recent_prices.clear()
        self._grind_start_vol.clear()
        self._grind_start_avg_delta.clear()
        self._recent_triggers.clear()
        self._queue.clear()
        self._post_restart = False
        with self._oi_prefetch_lock:
            self._oi_prefetch_results.clear()
        for k in self._stats:
            self._stats[k] = 0
        # Reset Earlybird state
        self._earlybird_trades_fired = 0
        self._prev_close.clear()
        self._day_open.clear()
        self._earlybird_fired = {'A': set(), 'B': set(), 'C': set()}
        self._nifty_prev_close = 0.0
        self._nifty_day_open = 0.0
        self._nifty_ltp = 0.0
        # Delete previous day's state file
        if os.path.exists(self._state_file):
            try:
                os.remove(self._state_file)
            except OSError:
                pass


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
