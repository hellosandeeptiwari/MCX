"""
DATA MANAGER MODULE
Handles all data fetching: quotes, historical candles, WebSocket ticks
"""

import json
import os
from datetime import datetime, timedelta
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
from typing import Dict, List, Callable, Optional
import threading
import time

from config import (
    API_KEY, API_SECRET, TOKEN_FILE, UNIVERSE,
    CANDLE_INTERVAL, TREND_INTERVAL
)


class DataManager:
    """Centralized data manager for the trading agent"""
    
    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY)
        self.ticker = None
        self.access_token = None
        self.instruments_cache = {}
        self.ltp_cache = {}  # Symbol -> LTP
        self.tick_callbacks = []  # Functions to call on tick
        self.connected = False
        
        # Load saved token
        self._load_token()
    
    def _load_token(self):
        """Load access token from file"""
        try:
            token_path = os.path.join(os.path.dirname(__file__), '..', TOKEN_FILE)
            with open(token_path, 'r') as f:
                data = json.load(f)
            
            # Check if token is from today
            if data.get('date') == str(datetime.now().date()):
                self.access_token = data['access_token']
                self.kite.set_access_token(self.access_token)
                print("✅ Access token loaded successfully")
                return True
            else:
                print("⚠️ Token expired. Need new login.")
                return False
        except FileNotFoundError:
            print("⚠️ No saved token found. Need to login.")
            return False
    
    def generate_session(self, request_token: str) -> bool:
        """Generate new session with request token"""
        try:
            data = self.kite.generate_session(request_token, api_secret=API_SECRET)
            self.access_token = data['access_token']
            self.kite.set_access_token(self.access_token)
            
            # Save token
            token_path = os.path.join(os.path.dirname(__file__), '..', TOKEN_FILE)
            with open(token_path, 'w') as f:
                json.dump({
                    'access_token': self.access_token,
                    'date': str(datetime.now().date())
                }, f)
            
            print("✅ New session created and saved")
            return True
        except Exception as e:
            print(f"❌ Session error: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        try:
            self.kite.profile()
            return True
        except:
            return False
    
    def get_instruments(self, exchange: str = "NSE") -> List[dict]:
        """Get all instruments for an exchange (cached)"""
        if exchange not in self.instruments_cache:
            self.instruments_cache[exchange] = self.kite.instruments(exchange)
        return self.instruments_cache[exchange]
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol like NSE:RELIANCE"""
        exchange, tradingsymbol = symbol.split(":")
        instruments = self.get_instruments(exchange)
        
        for inst in instruments:
            if inst['tradingsymbol'] == tradingsymbol:
                return inst['instrument_token']
        return None
    
    def get_quotes(self, symbols: List[str] = None) -> Dict:
        """Get quotes for symbols"""
        if symbols is None:
            symbols = UNIVERSE
        
        try:
            quotes = self.kite.quote(symbols)
            
            # Update LTP cache
            for sym, data in quotes.items():
                self.ltp_cache[sym] = data.get('last_price', 0)
            
            return quotes
        except Exception as e:
            print(f"❌ Quote error: {e}")
            return {}
    
    def get_ltp(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get just LTP for symbols (lighter call)"""
        if symbols is None:
            symbols = UNIVERSE
        
        try:
            ltp_data = self.kite.ltp(symbols)
            result = {}
            for sym, data in ltp_data.items():
                result[sym] = data.get('last_price', 0)
                self.ltp_cache[sym] = result[sym]
            return result
        except Exception as e:
            print(f"❌ LTP error: {e}")
            return {}
    
    def get_historical_data(
        self, 
        symbol: str, 
        interval: str = None,
        days: int = 30,
        from_date: datetime = None,
        to_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get historical candle data
        
        Args:
            symbol: e.g., "NSE:RELIANCE"
            interval: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day
            days: Number of days of data (if from_date not specified)
            from_date: Start date
            to_date: End date
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if interval is None:
            interval = CANDLE_INTERVAL
        
        if to_date is None:
            to_date = datetime.now()
        
        if from_date is None:
            from_date = to_date - timedelta(days=days)
        
        try:
            token = self.get_instrument_token(symbol)
            if token is None:
                print(f"❌ Symbol not found: {symbol}")
                return pd.DataFrame()
            
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Historical data error for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ohlc_for_universe(self, interval: str = None, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get historical data for all symbols in universe"""
        result = {}
        for symbol in UNIVERSE:
            df = self.get_historical_data(symbol, interval, days)
            if not df.empty:
                result[symbol] = df
        return result
    
    # ========== WEBSOCKET METHODS ==========
    
    def register_tick_callback(self, callback: Callable):
        """Register a function to be called on each tick"""
        self.tick_callbacks.append(callback)
    
    def _on_ticks(self, ws, ticks):
        """Called when ticks are received"""
        for tick in ticks:
            token = tick['instrument_token']
            # Find symbol from token (reverse lookup)
            for sym in UNIVERSE:
                if self.get_instrument_token(sym) == token:
                    self.ltp_cache[sym] = tick.get('last_price', 0)
                    tick['symbol'] = sym
                    break
        
        # Call registered callbacks
        for callback in self.tick_callbacks:
            try:
                callback(ticks)
            except Exception as e:
                print(f"❌ Tick callback error: {e}")
    
    def _on_connect(self, ws, response):
        """Called when WebSocket connects"""
        self.connected = True
        print("✅ WebSocket connected")
        
        # Subscribe to universe
        tokens = []
        for sym in UNIVERSE:
            token = self.get_instrument_token(sym)
            if token:
                tokens.append(token)
        
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            print(f"📡 Subscribed to {len(tokens)} instruments")
    
    def _on_close(self, ws, code, reason):
        """Called when WebSocket closes"""
        self.connected = False
        print(f"⚠️ WebSocket closed: {reason}")
    
    def _on_error(self, ws, code, reason):
        """Called on WebSocket error"""
        print(f"❌ WebSocket error: {code} - {reason}")
    
    def start_websocket(self):
        """Start WebSocket connection for real-time ticks"""
        if not self.access_token:
            print("❌ Cannot start WebSocket: No access token")
            return False
        
        self.ticker = KiteTicker(API_KEY, self.access_token)
        
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close
        self.ticker.on_error = self._on_error
        
        # Run in background thread
        def run_ticker():
            self.ticker.connect(threaded=True)
        
        thread = threading.Thread(target=run_ticker, daemon=True)
        thread.start()
        
        print("🚀 WebSocket starting...")
        return True
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.ticker:
            self.ticker.close()
            self.connected = False
            print("🛑 WebSocket stopped")
    
    def get_cached_ltp(self, symbol: str) -> float:
        """Get LTP from cache (for fast access)"""
        return self.ltp_cache.get(symbol, 0)


# Singleton instance
_data_manager = None

def get_data_manager() -> DataManager:
    """Get singleton DataManager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


if __name__ == "__main__":
    # Test the data manager
    dm = get_data_manager()
    
    if dm.is_authenticated():
        print("\n📊 Testing Data Manager...")
        
        # Test quotes
        quotes = dm.get_quotes(["NSE:RELIANCE", "NSE:MCX"])
        for sym, q in quotes.items():
            print(f"{sym}: ₹{q['last_price']:,.2f}")
        
        # Test historical
        print("\n📈 Historical data for MCX (5 days):")
        df = dm.get_historical_data("NSE:MCX", "day", days=5)
        print(df.tail())
    else:
        print("❌ Not authenticated. Need to login first.")
