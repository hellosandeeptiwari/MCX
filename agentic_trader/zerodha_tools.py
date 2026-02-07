"""
ZERODHA TOOLS FOR THE AGENT
Provides structured tools that the LLM agent can call
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from kiteconnect import KiteConnect
import pandas as pd
import os

from config import (
    ZERODHA_API_KEY, ZERODHA_API_SECRET, 
    HARD_RULES, TRADING_HOURS, APPROVED_UNIVERSE, FNO_CONFIG
)
from execution_guard import get_execution_guard
from idempotent_order_engine import get_idempotent_engine
from correlation_guard import get_correlation_guard
from regime_score import get_regime_scorer
from position_reconciliation import get_position_reconciliation
from data_health_gate import get_data_health_gate
from options_trader import get_options_trader, OptionType, StrikeSelection, ExpirySelection, get_intraday_scorer, IntradaySignal


@dataclass
class AccountState:
    """Current account state"""
    available_margin: float
    used_margin: float
    total_equity: float
    start_of_day_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: List[Dict]
    pending_orders: List[Dict]
    positions_count: int
    daily_loss: float
    daily_loss_pct: float
    can_trade: bool
    reason: str


@dataclass
class MarketData:
    """Market data for a symbol"""
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_pct: float
    timestamp: str
    is_stale: bool
    # Indicators
    sma_20: float
    sma_50: float
    rsi_14: float
    atr_14: float
    # Enhanced context for better decisions
    trend: str = "SIDEWAYS"
    high_20d: float = 0
    low_20d: float = 0
    high_5d: float = 0
    low_5d: float = 0
    prev_high: float = 0
    prev_low: float = 0
    prev_close: float = 0
    resistance_1: float = 0
    resistance_2: float = 0
    support_1: float = 0
    support_2: float = 0
    atr_target: float = 0
    atr_stoploss: float = 0
    volume_ratio: float = 1.0
    volume_signal: str = "NORMAL"
    
    # === REGIME DETECTION SIGNALS ===
    # VWAP Analysis
    vwap: float = 0                    # Volume Weighted Average Price
    vwap_slope: str = "FLAT"           # RISING, FALLING, FLAT
    price_vs_vwap: str = "AT_VWAP"     # ABOVE_VWAP, BELOW_VWAP, AT_VWAP
    
    # EMA Compression/Expansion
    ema_9: float = 0
    ema_21: float = 0
    ema_spread: float = 0              # % difference between 9 and 21 EMA
    ema_regime: str = "NEUTRAL"        # COMPRESSED, EXPANDING_BULL, EXPANDING_BEAR
    
    # Opening Range Breakout (ORB)
    orb_high: float = 0                # First 15-min high
    orb_low: float = 0                 # First 15-min low
    orb_signal: str = "INSIDE_ORB"     # BREAKOUT_UP, BREAKOUT_DOWN, INSIDE_ORB
    orb_strength: float = 0            # % move from ORB level
    
    # Volume Analysis
    volume_5d_avg: float = 0           # 5-day average volume
    volume_vs_avg: float = 1.0         # Today's volume / 5-day avg
    volume_regime: str = "NORMAL"      # LOW, NORMAL, HIGH, EXPLOSIVE
    
    # === CHOP FILTER (Anti-Whipsaw) ===
    chop_zone: bool = False            # True = NO TRADE zone (choppy)
    chop_reason: str = ""              # Why it's choppy
    atr_range_ratio: float = 1.0       # Current range / ATR (low = compressed)
    orb_reentries: int = 0             # Count of ORB re-entries (chop signal)
    
    # === HIGHER TIMEFRAME ALIGNMENT ===
    htf_trend: str = "NEUTRAL"         # BULLISH, BEARISH, NEUTRAL (based on longer EMA)
    htf_ema_slope: str = "FLAT"        # RISING, FALLING, FLAT
    htf_alignment: str = "NEUTRAL"     # ALIGNED, CONFLICTING, NEUTRAL


@dataclass 
class TradePlan:
    """A candidate trade plan"""
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    risk_amount: float
    risk_pct: float
    rationale: str
    confidence: str  # LOW, MEDIUM, HIGH
    rule_checks: Dict[str, bool]
    approved: bool = False


@dataclass
class RiskCheck:
    """Risk validation result"""
    rule: str
    passed: bool
    current_value: Any
    limit: Any
    message: str


class ZerodhaTools:
    """
    Tools that the LLM agent can call to interact with Zerodha
    """
    
    # Trade tracking file path
    TRADES_FILE = os.path.join(os.path.dirname(__file__), 'active_trades.json')
    TRADE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'trade_history.json')
    
    def __init__(self, paper_mode: bool = True, paper_capital: float = None):
        self.kite = KiteConnect(api_key=ZERODHA_API_KEY)
        self.access_token = None
        self.start_of_day_equity = None
        self.last_api_call = 0
        self.api_call_count = 0
        
        # Paper trading state
        self.paper_mode = paper_mode
        self.paper_capital = paper_capital or HARD_RULES["CAPITAL"]
        self.paper_positions = []
        self.paper_orders = []
        self.paper_pnl = 0
        
        # Load saved token
        self._load_token()
        
        # Load active trades from file
        self._load_active_trades()
    
    def _load_active_trades(self):
        """Load active trades from JSON file"""
        try:
            if os.path.exists(self.TRADES_FILE):
                with open(self.TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    # Only load trades from today
                    today = str(datetime.now().date())
                    if data.get('date') == today:
                        self.paper_positions = data.get('active_trades', [])
                        self.paper_pnl = data.get('realized_pnl', 0)
                        print(f"📂 Loaded {len(self.paper_positions)} active trades from file")
                    else:
                        # New day - start fresh
                        self.paper_positions = []
                        self.paper_pnl = 0
                        print(f"📅 New trading day - starting fresh")
            else:
                self.paper_positions = []
                self.paper_pnl = 0
        except Exception as e:
            print(f"⚠️ Error loading trades: {e}")
            self.paper_positions = []
            self.paper_pnl = 0
    
    def _save_active_trades(self):
        """Save active trades to JSON file"""
        try:
            data = {
                'date': str(datetime.now().date()),
                'last_updated': datetime.now().isoformat(),
                'active_trades': self.paper_positions,
                'realized_pnl': self.paper_pnl,
                'paper_capital': self.paper_capital
            }
            with open(self.TRADES_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving trades: {e}")
    
    def _save_to_history(self, trade: Dict, result: str, pnl: float):
        """Save completed trade to history file"""
        try:
            history = []
            if os.path.exists(self.TRADE_HISTORY_FILE):
                with open(self.TRADE_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            
            trade_record = {
                **trade,
                'result': result,  # 'TARGET_HIT', 'STOPLOSS_HIT', 'MANUAL_EXIT', 'EOD_EXIT'
                'pnl': pnl,
                'closed_at': datetime.now().isoformat()
            }
            history.append(trade_record)
            
            with open(self.TRADE_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving to history: {e}")
    
    def is_symbol_in_active_trades(self, symbol: str) -> bool:
        """Check if symbol already has an active trade"""
        for trade in self.paper_positions:
            if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                return True
        return False
    
    def get_active_trade(self, symbol: str) -> Optional[Dict]:
        """Get active trade for a symbol if exists"""
        for trade in self.paper_positions:
            if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                return trade
        return None
    
    def update_trade_status(self, symbol: str, status: str, exit_price: float = None, pnl: float = None):
        """Update trade status and move to history if closed"""
        for i, trade in enumerate(self.paper_positions):
            if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                trade['status'] = status
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now().isoformat()
                
                if pnl is not None:
                    trade['pnl'] = pnl
                    self.paper_pnl += pnl
                
                # Save to history and remove from active
                if status in ['TARGET_HIT', 'STOPLOSS_HIT', 'MANUAL_EXIT', 'EOD_EXIT']:
                    self._save_to_history(trade, status, pnl or 0)
                    self.paper_positions.pop(i)
                
                self._save_active_trades()
                return True
        return False
    
    def check_and_update_trades(self) -> List[Dict]:
        """Check all active trades for target/stoploss hits and update"""
        updates = []
        
        if not self.paper_positions:
            return updates
        
        # Get current prices for all active symbols
        symbols = [t['symbol'] for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        if not symbols:
            return updates
        
        try:
            quotes = self.kite.quote(symbols)
            
            for trade in self.paper_positions[:]:  # Copy list to avoid modification during iteration
                if trade.get('status', 'OPEN') != 'OPEN':
                    continue
                
                symbol = trade['symbol']
                if symbol not in quotes:
                    continue
                
                ltp = quotes[symbol]['last_price']
                entry = trade['avg_price']
                sl = trade['stop_loss']
                target = trade.get('target', entry * 1.02)
                qty = trade['quantity']
                side = trade['side']
                
                # Calculate P&L
                if side == 'BUY':
                    pnl = (ltp - entry) * qty
                    # Check target hit (price went above target)
                    if ltp >= target:
                        self.update_trade_status(symbol, 'TARGET_HIT', ltp, pnl)
                        updates.append({
                            'symbol': symbol, 
                            'result': 'TARGET_HIT', 
                            'pnl': pnl,
                            'entry': entry,
                            'exit': ltp
                        })
                    # Check stoploss hit (price went below SL)
                    elif ltp <= sl:
                        self.update_trade_status(symbol, 'STOPLOSS_HIT', ltp, pnl)
                        updates.append({
                            'symbol': symbol, 
                            'result': 'STOPLOSS_HIT', 
                            'pnl': pnl,
                            'entry': entry,
                            'exit': ltp
                        })
                else:  # SHORT
                    pnl = (entry - ltp) * qty
                    # Check target hit (price went below target for shorts)
                    if ltp <= target:
                        self.update_trade_status(symbol, 'TARGET_HIT', ltp, pnl)
                        updates.append({
                            'symbol': symbol, 
                            'result': 'TARGET_HIT', 
                            'pnl': pnl,
                            'entry': entry,
                            'exit': ltp
                        })
                    # Check stoploss hit (price went above SL for shorts)
                    elif ltp >= sl:
                        self.update_trade_status(symbol, 'STOPLOSS_HIT', ltp, pnl)
                        updates.append({
                            'symbol': symbol, 
                            'result': 'STOPLOSS_HIT', 
                            'pnl': pnl,
                            'entry': entry,
                            'exit': ltp
                        })
        except Exception as e:
            print(f"⚠️ Error checking trades: {e}")
        
        return updates
    
    def _load_token(self):
        """Load access token from file"""
        try:
            token_path = os.path.join(os.path.dirname(__file__), '..', 'zerodha_token.json')
            with open(token_path, 'r') as f:
                data = json.load(f)
            
            if data.get('date') == str(datetime.now().date()):
                self.access_token = data['access_token']
                self.kite.set_access_token(self.access_token)
                return True
        except:
            pass
        return False
    
    def authenticate(self):
        """Authenticate with Zerodha - run once daily"""
        print("\n🔐 Zerodha Authentication")
        print("="*50)
        
        login_url = self.kite.login_url()
        print(f"\n1. Open this URL in your browser:")
        print(f"   {login_url}")
        
        print(f"\n2. Login with your Zerodha credentials")
        print(f"\n3. After login, copy the 'request_token' from the redirect URL")
        print(f"   (It's in the URL like: ?request_token=XXXXXX)")
        
        request_token = input("\n4. Paste request_token here: ").strip()
        
        try:
            data = self.kite.generate_session(request_token, api_secret=ZERODHA_API_SECRET)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save token
            token_path = os.path.join(os.path.dirname(__file__), '..', 'zerodha_token.json')
            with open(token_path, 'w') as f:
                json.dump({
                    'access_token': self.access_token,
                    'date': str(datetime.now().date())
                }, f)
            
            print(f"\n✅ Authentication successful!")
            print(f"   Token saved for today: {datetime.now().date()}")
            
            # Test the connection
            profile = self.kite.profile()
            print(f"   Logged in as: {profile.get('user_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            return False
    
    def _rate_limit(self):
        """Enforce API rate limiting"""
        min_interval = HARD_RULES["API_RATE_LIMIT_MS"] / 1000
        elapsed = time.time() - self.last_api_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_api_call = time.time()
        self.api_call_count += 1
    
    def _is_trading_hours(self) -> tuple[bool, str]:
        """Check if within trading hours"""
        now = datetime.now().time()
        start = datetime.strptime(TRADING_HOURS["start"], "%H:%M").time()
        end = datetime.strptime(TRADING_HOURS["end"], "%H:%M").time()
        no_new = datetime.strptime(TRADING_HOURS["no_new_after"], "%H:%M").time()
        
        if now < start:
            return False, f"Market not open yet (opens at {TRADING_HOURS['start']})"
        if now > end:
            return False, f"Market closed (closed at {TRADING_HOURS['end']})"
        if now > no_new:
            return False, f"No new trades after {TRADING_HOURS['no_new_after']}"
        
        return True, "Within trading hours"
    
    def get_account_state(self) -> Dict:
        """
        Tool: Get current account state
        Returns margins, positions, open orders, and risk status
        """
        self._rate_limit()
        
        try:
            # In paper mode, use simulated capital
            if self.paper_mode:
                # Calculate used margin from paper positions
                used = sum(abs(p.get('quantity', 0) * p.get('avg_price', 0)) for p in self.paper_positions)
                available = self.paper_capital - used + self.paper_pnl
                total_equity = self.paper_capital + self.paper_pnl
                
                # Initialize start of day equity
                if self.start_of_day_equity is None:
                    self.start_of_day_equity = self.paper_capital
                
                realized_pnl = self.paper_pnl
                unrealized_pnl = 0  # Would need LTP to calculate
                open_positions = self.paper_positions
                pending_orders = self.paper_orders
                
            else:
                # Get real margins from Zerodha
                margins = self.kite.margins()
                equity_margin = margins.get('equity', {})
                available = equity_margin.get('available', {}).get('live_balance', 0)
                used = equity_margin.get('utilised', {}).get('debits', 0)
                
                # Get positions
                positions = self.kite.positions()
                day_positions = positions.get('day', [])
                
                # Calculate P&L
                realized_pnl = sum(p.get('realised', 0) for p in day_positions)
                unrealized_pnl = sum(p.get('unrealised', 0) for p in day_positions)
                
                # Get open orders
                orders = self.kite.orders()
                pending_orders = [o for o in orders if o['status'] in ['OPEN', 'TRIGGER PENDING']]
                
                # Count open positions (non-zero quantity)
                open_positions = [p for p in day_positions if p['quantity'] != 0]
                
                # Calculate daily loss
                total_equity = available + used
                if self.start_of_day_equity is None:
                    self.start_of_day_equity = total_equity - realized_pnl - unrealized_pnl
            
            daily_pnl = realized_pnl + unrealized_pnl
            daily_loss = min(0, daily_pnl)
            daily_loss_pct = abs(daily_loss) / self.start_of_day_equity if self.start_of_day_equity > 0 else 0
            
            # Check if can trade
            can_trade = True
            reason = "OK"
            
            # Check daily loss limit
            if daily_loss_pct >= HARD_RULES["MAX_DAILY_LOSS"]:
                can_trade = False
                reason = f"Daily loss limit hit ({daily_loss_pct*100:.2f}% >= {HARD_RULES['MAX_DAILY_LOSS']*100}%)"
            
            # Check max positions
            positions_count = len(open_positions) if isinstance(open_positions, list) else 0
            if positions_count >= HARD_RULES["MAX_POSITIONS"]:
                can_trade = False
                reason = f"Max positions reached ({positions_count} >= {HARD_RULES['MAX_POSITIONS']})"
            
            # Check trading hours
            in_hours, hours_reason = self._is_trading_hours()
            if not in_hours:
                can_trade = False
                reason = hours_reason
            
            # Format positions for output
            if self.paper_mode:
                formatted_positions = self.paper_positions
                formatted_orders = self.paper_orders
            else:
                formatted_positions = [{
                    'symbol': f"{p['exchange']}:{p['tradingsymbol']}",
                    'quantity': p['quantity'],
                    'avg_price': p['average_price'],
                    'ltp': p['last_price'],
                    'pnl': p['pnl'],
                    'side': 'LONG' if p['quantity'] > 0 else 'SHORT'
                } for p in open_positions]
                formatted_orders = [{
                    'order_id': o['order_id'],
                    'symbol': f"{o['exchange']}:{o['tradingsymbol']}",
                    'side': o['transaction_type'],
                    'quantity': o['quantity'],
                    'price': o['price'],
                    'status': o['status']
                } for o in pending_orders]
            
            state = AccountState(
                available_margin=available,
                used_margin=used,
                total_equity=total_equity,
                start_of_day_equity=self.start_of_day_equity,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                open_positions=formatted_positions,
                pending_orders=formatted_orders,
                positions_count=positions_count,
                daily_loss=daily_loss,
                daily_loss_pct=daily_loss_pct,
                can_trade=can_trade,
                reason=reason
            )
            
            return asdict(state)
            
        except Exception as e:
            return {"error": str(e), "can_trade": False, "reason": f"API Error: {e}"}
    
    def get_volume_analysis(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Tool: Analyze intraday momentum using FUTURES OI (not delivery volume)
        
        Why Futures OI matters for EOD:
        - Futures = 100% speculative, no delivery
        - Long buildup = fresh longs, expect continuation
        - Short buildup = fresh shorts, expect drop
        - Short covering = shorts buying back, expect rally
        - Long unwinding = longs exiting, expect weakness
        
        Order book depth shows PENDING orders (intent to buy/sell)
        """
        self._rate_limit()
        
        try:
            # Get quote data (includes volume, bid/ask)
            quotes = self.kite.quote(symbols)
            
            analysis = {}
            
            for symbol in symbols:
                if symbol not in quotes:
                    continue
                
                q = quotes[symbol]
                ltp = q.get('last_price', 0)
                volume = q.get('volume', 0)
                
                # OHLC data
                ohlc = q.get('ohlc', {})
                open_price = ohlc.get('open', ltp)
                high = ohlc.get('high', ltp)
                low = ohlc.get('low', ltp)
                prev_close = ohlc.get('close', ltp)
                
                # Change calculations
                change_from_open = ((ltp - open_price) / open_price * 100) if open_price else 0
                
                # Bid-Ask analysis (PENDING orders = real-time intent)
                depth = q.get('depth', {})
                buy_depth = depth.get('buy', [])
                sell_depth = depth.get('sell', [])
                
                # Calculate buy vs sell PENDING volume
                buy_pending = sum(b.get('quantity', 0) for b in buy_depth) if buy_depth else 0
                sell_pending = sum(s.get('quantity', 0) for s in sell_depth) if sell_depth else 0
                total_pending = buy_pending + sell_pending
                
                # Pending order imbalance (this is REAL-TIME intent, not executed)
                if total_pending > 0:
                    buy_pressure = buy_pending / total_pending
                    sell_pressure = sell_pending / total_pending
                else:
                    buy_pressure = 0.5
                    sell_pressure = 0.5
                
                # Intraday range analysis
                day_range = high - low if high > low else 0.01
                position_in_range = (ltp - low) / day_range if day_range > 0 else 0.5
                
                # Get FUTURES OI for this stock (the real speculative indicator)
                oi_data = self._get_futures_oi_quick(symbol)
                
                # Initialize predictions
                eod_prediction = "NEUTRAL"
                eod_confidence = "LOW"
                eod_reasoning = []
                
                if oi_data.get('has_futures'):
                    oi_signal = oi_data.get('oi_signal', 'NEUTRAL')
                    
                    # OI-based EOD prediction (most reliable for intraday)
                    if oi_signal == "LONG_BUILDUP":
                        eod_prediction = "UP"
                        eod_confidence = "HIGH"
                        eod_reasoning.append(f"Long buildup in futures (OI↑ + Price↑)")
                    
                    elif oi_signal == "SHORT_BUILDUP":
                        eod_prediction = "DOWN"
                        eod_confidence = "HIGH"
                        eod_reasoning.append(f"Short buildup in futures (OI↑ + Price↓)")
                    
                    elif oi_signal == "SHORT_COVERING":
                        eod_prediction = "UP"
                        eod_confidence = "HIGH"
                        eod_reasoning.append(f"Short covering rally (OI↓ + Price↑)")
                    
                    elif oi_signal == "LONG_UNWINDING":
                        eod_prediction = "DOWN"
                        eod_confidence = "MEDIUM"
                        eod_reasoning.append(f"Long unwinding (OI↓ + Price↓)")
                
                else:
                    # For non-F&O stocks, use order book imbalance (less reliable)
                    if buy_pressure > 0.65:
                        eod_prediction = "UP"
                        eod_confidence = "LOW"
                        eod_reasoning.append(f"Strong buy orders pending ({buy_pressure*100:.0f}%)")
                    elif sell_pressure > 0.65:
                        eod_prediction = "DOWN"
                        eod_confidence = "LOW"
                        eod_reasoning.append(f"Strong sell orders pending ({sell_pressure*100:.0f}%)")
                
                # Additional: Price near day's low with time running out = potential squeeze
                now = datetime.now()
                if now.hour >= 14:  # After 2 PM
                    if position_in_range < 0.3 and buy_pressure > 0.55:
                        eod_reasoning.append("Near day low, late session - potential short squeeze")
                        if eod_confidence == "LOW":
                            eod_confidence = "MEDIUM"
                    elif position_in_range > 0.7 and sell_pressure > 0.55:
                        eod_reasoning.append("Near day high, late session - potential profit booking")
                        if eod_confidence == "LOW":
                            eod_confidence = "MEDIUM"
                
                # Trading signal
                if eod_prediction == "UP" and eod_confidence in ["MEDIUM", "HIGH"]:
                    trade_signal = "BUY_FOR_EOD"
                    strategy = f"Buy for EOD close higher - {', '.join(eod_reasoning)}"
                elif eod_prediction == "DOWN" and eod_confidence in ["MEDIUM", "HIGH"]:
                    trade_signal = "SHORT_FOR_EOD"
                    strategy = f"Short for EOD close lower - {', '.join(eod_reasoning)}"
                else:
                    trade_signal = "NO_SIGNAL"
                    strategy = "No clear EOD edge from OI/order flow"
                
                analysis[symbol] = {
                    "ltp": ltp,
                    "change_from_open": round(change_from_open, 2),
                    "volume": volume,
                    "buy_pending_qty": buy_pending,
                    "sell_pending_qty": sell_pending,
                    "buy_pressure_pct": round(buy_pressure * 100, 1),
                    "sell_pressure_pct": round(sell_pressure * 100, 1),
                    "position_in_range": round(position_in_range * 100, 1),
                    "has_futures_oi": oi_data.get('has_futures', False),
                    "oi_signal": oi_data.get('oi_signal', 'N/A'),
                    "eod_prediction": eod_prediction,
                    "eod_confidence": eod_confidence,
                    "eod_reasoning": eod_reasoning,
                    "trade_signal": trade_signal,
                    "strategy": strategy
                }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_futures_oi_quick(self, symbol: str) -> Dict:
        """Quick futures OI check for a symbol"""
        try:
            if ":" in symbol:
                _, stock = symbol.split(":")
            else:
                stock = symbol
            
            # Check if this is an F&O stock
            fno_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
                         "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "TATAMOTORS",
                         "ITC", "TATASTEEL", "HINDUNILVR", "MARUTI", "TITAN",
                         "LT", "SUNPHARMA", "BHARTIARTL", "ONGC", "COALINDIA",
                         "TATAPOWER", "BANKBARODA", "PNB", "MCX"]
            
            if stock not in fno_stocks:
                return {"has_futures": False}
            
            # Get current quote for price change
            quote = self.kite.quote([symbol])
            if symbol not in quote:
                return {"has_futures": False}
            
            q = quote[symbol]
            ltp = q.get('last_price', 0)
            ohlc = q.get('ohlc', {})
            prev_close = ohlc.get('close', ltp)
            change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
            
            # Get instruments to find futures
            instruments = self.kite.instruments("NFO")
            
            # Find current month futures
            futures = [i for i in instruments if 
                      i['name'] == stock and 
                      i['instrument_type'] == 'FUT' and
                      i['segment'] == 'NFO-FUT']
            
            if not futures:
                return {"has_futures": False}
            
            # Sort by expiry, get nearest
            futures.sort(key=lambda x: x['expiry'])
            current_fut = futures[0]
            
            fut_symbol = f"NFO:{current_fut['tradingsymbol']}"
            fut_quote = self.kite.quote([fut_symbol])
            
            if fut_symbol not in fut_quote:
                return {"has_futures": False}
            
            fq = fut_quote[fut_symbol]
            oi = fq.get('oi', 0)
            oi_day_high = fq.get('oi_day_high', oi)
            oi_day_low = fq.get('oi_day_low', oi)
            
            # OI change from day's low (intraday buildup)
            oi_change_pct = ((oi - oi_day_low) / oi_day_low * 100) if oi_day_low else 0
            
            # Determine OI signal
            if change_pct > 0.3 and oi_change_pct > 2:
                oi_signal = "LONG_BUILDUP"
            elif change_pct < -0.3 and oi_change_pct > 2:
                oi_signal = "SHORT_BUILDUP"
            elif change_pct < -0.3 and oi_change_pct < -2:
                oi_signal = "LONG_UNWINDING"
            elif change_pct > 0.3 and oi_change_pct < -2:
                oi_signal = "SHORT_COVERING"
            else:
                oi_signal = "NEUTRAL"
            
            return {
                "has_futures": True,
                "futures_symbol": fut_symbol,
                "oi": oi,
                "oi_change_pct": round(oi_change_pct, 2),
                "price_change_pct": round(change_pct, 2),
                "oi_signal": oi_signal
            }
            
        except Exception as e:
            return {"has_futures": False, "error": str(e)}
    
    def get_oi_analysis(self, symbol: str) -> Dict:
        """
        Tool: Get Open Interest analysis for F&O stocks
        
        OI Analysis:
        - Long Build-up: Price UP + OI UP = Strong Bullish
        - Short Build-up: Price DOWN + OI UP = Strong Bearish
        - Long Unwinding: Price DOWN + OI DOWN = Weak (longs exiting)
        - Short Covering: Price UP + OI DOWN = Rally (shorts covering)
        """
        self._rate_limit()
        
        try:
            # Get the underlying symbol without exchange
            if ":" in symbol:
                _, stock = symbol.split(":")
            else:
                stock = symbol
            
            # Get current quote
            quote = self.kite.quote([symbol])
            if symbol not in quote:
                return {"error": f"Symbol not found: {symbol}"}
            
            q = quote[symbol]
            ltp = q.get('last_price', 0)
            ohlc = q.get('ohlc', {})
            prev_close = ohlc.get('close', ltp)
            change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
            
            # Try to get F&O instruments for this stock
            # Find current month expiry futures
            try:
                instruments = self.kite.instruments("NFO")
                
                # Find futures for this stock
                futures = [i for i in instruments if 
                          i['name'] == stock and 
                          i['instrument_type'] == 'FUT' and
                          i['segment'] == 'NFO-FUT']
                
                if not futures:
                    return {
                        "symbol": symbol,
                        "ltp": ltp,
                        "change_pct": round(change_pct, 2),
                        "oi_analysis": "N/A - Not an F&O stock",
                        "has_fno": False
                    }
                
                # Sort by expiry to get current month
                futures.sort(key=lambda x: x['expiry'])
                current_fut = futures[0] if futures else None
                
                if current_fut:
                    fut_symbol = f"NFO:{current_fut['tradingsymbol']}"
                    fut_quote = self.kite.quote([fut_symbol])
                    
                    if fut_symbol in fut_quote:
                        fq = fut_quote[fut_symbol]
                        oi = fq.get('oi', 0)
                        oi_day_high = fq.get('oi_day_high', oi)
                        oi_day_low = fq.get('oi_day_low', oi)
                        
                        # Calculate OI change (approximate)
                        oi_change = oi - oi_day_low if oi_day_low else 0
                        oi_change_pct = (oi_change / oi_day_low * 100) if oi_day_low else 0
                        
                        # Determine OI interpretation
                        if change_pct > 0.5 and oi_change_pct > 1:
                            oi_signal = "LONG_BUILDUP"
                            interpretation = "🟢 New longs being created - BULLISH"
                            trade_bias = "BUY"
                        elif change_pct < -0.5 and oi_change_pct > 1:
                            oi_signal = "SHORT_BUILDUP"
                            interpretation = "🔴 New shorts being created - BEARISH"
                            trade_bias = "SHORT"
                        elif change_pct < -0.5 and oi_change_pct < -1:
                            oi_signal = "LONG_UNWINDING"
                            interpretation = "🟡 Longs exiting - Weak"
                            trade_bias = "AVOID"
                        elif change_pct > 0.5 and oi_change_pct < -1:
                            oi_signal = "SHORT_COVERING"
                            interpretation = "🟢 Shorts covering - Rally mode"
                            trade_bias = "BUY"
                        else:
                            oi_signal = "NEUTRAL"
                            interpretation = "⚪ No clear OI signal"
                            trade_bias = "NEUTRAL"
                        
                        return {
                            "symbol": symbol,
                            "ltp": ltp,
                            "change_pct": round(change_pct, 2),
                            "has_fno": True,
                            "futures_symbol": fut_symbol,
                            "oi": oi,
                            "oi_day_high": oi_day_high,
                            "oi_day_low": oi_day_low,
                            "oi_change": oi_change,
                            "oi_change_pct": round(oi_change_pct, 2),
                            "oi_signal": oi_signal,
                            "interpretation": interpretation,
                            "trade_bias": trade_bias
                        }
                
            except Exception as e:
                pass  # F&O data not available
            
            return {
                "symbol": symbol,
                "ltp": ltp,
                "change_pct": round(change_pct, 2),
                "oi_analysis": "Could not fetch OI data",
                "has_fno": False
            }
            
        except Exception as e:
            return {"error": str(e)}

    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Tool: Get market data for symbols
        Returns OHLCV + technical indicators
        """
        # Filter to approved universe
        valid_symbols = [s for s in symbols if s in APPROVED_UNIVERSE]
        if not valid_symbols:
            return {"error": "No valid symbols in approved universe"}
        
        self._rate_limit()
        
        result = {}
        
        try:
            # Get quotes
            quotes = self.kite.quote(valid_symbols)
            
            for symbol in valid_symbols:
                if symbol not in quotes:
                    continue
                
                q = quotes[symbol]
                ohlc = q.get('ohlc', {})
                
                # Check if data is stale
                last_trade_time = q.get('last_trade_time')
                is_stale = False
                if last_trade_time:
                    if isinstance(last_trade_time, str):
                        last_trade_time = datetime.fromisoformat(last_trade_time)
                    age = (datetime.now() - last_trade_time).seconds
                    is_stale = age > HARD_RULES["STALE_DATA_SECONDS"]
                
                # Get historical data for indicators
                indicators = self._calculate_indicators(symbol)
                
                # === DATA HEALTH GATE: Add health check result ===
                health_gate = get_data_health_gate()
                health_result = health_gate.check_health(symbol, {
                    'ltp': q.get('last_price', 0),
                    'volume': q.get('volume', 0),
                    'volume_ratio': indicators.get('volume_ratio', 1.0),
                    'timestamp': str(datetime.now()),
                    'last_trade_time': q.get('last_trade_time'),
                    'vwap': indicators.get('vwap', 0),
                    'ema_9': indicators.get('ema_9', 0),
                    'ema_21': indicators.get('ema_21', 0),
                    'sma_20': indicators.get('sma_20', 0),
                    'atr_14': indicators.get('atr_14', 0),
                    'rsi_14': indicators.get('rsi_14', 50)
                })
                
                # Use health gate stale check instead of simple is_stale
                is_stale = not health_result.can_trade or is_stale
                
                data = MarketData(
                    symbol=symbol,
                    ltp=q.get('last_price', 0),
                    open=ohlc.get('open', 0),
                    high=ohlc.get('high', 0),
                    low=ohlc.get('low', 0),
                    close=ohlc.get('close', 0),
                    volume=q.get('volume', 0),
                    change_pct=q.get('change', 0),
                    timestamp=str(datetime.now()),
                    is_stale=is_stale,
                    sma_20=indicators.get('sma_20', 0),
                    sma_50=indicators.get('sma_50', 0),
                    rsi_14=indicators.get('rsi_14', 50),
                    atr_14=indicators.get('atr_14', 0),
                    # Enhanced context
                    trend=indicators.get('trend', 'SIDEWAYS'),
                    high_20d=indicators.get('high_20d', 0),
                    low_20d=indicators.get('low_20d', 0),
                    high_5d=indicators.get('high_5d', 0),
                    low_5d=indicators.get('low_5d', 0),
                    prev_high=indicators.get('prev_high', 0),
                    prev_low=indicators.get('prev_low', 0),
                    prev_close=indicators.get('prev_close', 0),
                    resistance_1=indicators.get('resistance_1', 0),
                    resistance_2=indicators.get('resistance_2', 0),
                    support_1=indicators.get('support_1', 0),
                    support_2=indicators.get('support_2', 0),
                    atr_target=indicators.get('atr_target', 0),
                    atr_stoploss=indicators.get('atr_stoploss', 0),
                    volume_ratio=indicators.get('volume_ratio', 1.0),
                    volume_signal=indicators.get('volume_signal', 'NORMAL'),
                    # === REGIME DETECTION ===
                    vwap=indicators.get('vwap', 0),
                    vwap_slope=indicators.get('vwap_slope', 'FLAT'),
                    price_vs_vwap=indicators.get('price_vs_vwap', 'AT_VWAP'),
                    ema_9=indicators.get('ema_9', 0),
                    ema_21=indicators.get('ema_21', 0),
                    ema_spread=indicators.get('ema_spread', 0),
                    ema_regime=indicators.get('ema_regime', 'NEUTRAL'),
                    orb_high=indicators.get('orb_high', 0),
                    orb_low=indicators.get('orb_low', 0),
                    orb_signal=indicators.get('orb_signal', 'INSIDE_ORB'),
                    orb_strength=indicators.get('orb_strength', 0),
                    volume_5d_avg=indicators.get('volume_5d_avg', 0),
                    volume_vs_avg=indicators.get('volume_vs_avg', 1.0),
                    volume_regime=indicators.get('volume_regime', 'NORMAL'),
                    # === CHOP FILTER ===
                    chop_zone=indicators.get('chop_zone', False),
                    chop_reason=indicators.get('chop_reason', ''),
                    atr_range_ratio=indicators.get('atr_range_ratio', 1.0),
                    orb_reentries=indicators.get('orb_reentries', 0),
                    # === HTF ALIGNMENT ===
                    htf_trend=indicators.get('htf_trend', 'NEUTRAL'),
                    htf_ema_slope=indicators.get('htf_ema_slope', 'FLAT'),
                    htf_alignment=indicators.get('htf_alignment', 'NEUTRAL')
                )
                
                result[symbol] = asdict(data)
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_indicators(self, symbol: str) -> Dict:
        """Calculate technical indicators for a symbol"""
        self._rate_limit()
        
        try:
            # Get instrument token
            exchange, tradingsymbol = symbol.split(":")
            instruments = self.kite.instruments(exchange)
            token = None
            for inst in instruments:
                if inst['tradingsymbol'] == tradingsymbol:
                    token = inst['instrument_token']
                    break
            
            if not token:
                return {}
            
            # Get historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=60)
            
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            close = df['close']
            high = df['high']
            low = df['low']
            
            # SMA
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.mean()
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 14 else 50
            
            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()
            
            # Additional context for informed decisions
            high_20d = high.tail(20).max()
            low_20d = low.tail(20).min()
            high_5d = high.tail(5).max()
            low_5d = low.tail(5).min()
            
            # Previous day
            prev_high = high.iloc[-2] if len(high) > 1 else high.iloc[-1]
            prev_low = low.iloc[-2] if len(low) > 1 else low.iloc[-1]
            prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
            
            # Trend detection
            current_price = close.iloc[-1]
            trend = "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "SIDEWAYS"
            
            # Volume analysis
            vol = df['volume']
            avg_volume_20 = vol.tail(20).mean()
            current_volume = vol.iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # === REGIME DETECTION CALCULATIONS ===
            
            # 1. EMA 9 and 21 for compression/expansion
            ema_9_series = close.ewm(span=9, adjust=False).mean()
            ema_21_series = close.ewm(span=21, adjust=False).mean()
            ema_9 = ema_9_series.iloc[-1]
            ema_21 = ema_21_series.iloc[-1]
            ema_spread = abs(ema_9 - ema_21) / ema_21 * 100 if ema_21 > 0 else 0
            
            # EMA regime detection - TIME-QUALIFIED COMPRESSION
            # Check if compression has persisted for 5+ candles
            compression_threshold = 0.3  # % spread threshold
            if len(ema_9_series) >= 5:
                recent_spreads = abs(ema_9_series.iloc[-5:] - ema_21_series.iloc[-5:]) / ema_21_series.iloc[-5:] * 100
                candles_compressed = (recent_spreads < compression_threshold).sum()
                
                if candles_compressed >= 5:
                    ema_regime = "COMPRESSED"  # Squeeze for 5+ candles - valid breakout setup
                elif ema_9 > ema_21:
                    ema_regime = "EXPANDING_BULL"
                else:
                    ema_regime = "EXPANDING_BEAR"
            else:
                ema_regime = "NEUTRAL"
            
            # 2. VWAP calculation (for daily data, approximate)
            typical_price = (high + low + close) / 3
            cumulative_tpv = (typical_price * vol).cumsum()
            cumulative_vol = vol.cumsum()
            vwap_series = cumulative_tpv / cumulative_vol
            vwap = vwap_series.iloc[-1] if len(vwap_series) > 0 else current_price
            
            # VWAP slope over 5 candles (not tick-to-tick to avoid chop)
            if len(vwap_series) >= 5:
                vwap_5_ago = vwap_series.iloc[-5]
                vwap_now = vwap_series.iloc[-1]
                vwap_change_pct = (vwap_now - vwap_5_ago) / vwap_5_ago * 100 if vwap_5_ago > 0 else 0
                
                if vwap_change_pct > 0.5:  # >0.5% rise over 5 candles
                    vwap_slope = "RISING"
                elif vwap_change_pct < -0.5:  # <-0.5% fall over 5 candles
                    vwap_slope = "FALLING"
                else:
                    vwap_slope = "FLAT"
            else:
                vwap_slope = "FLAT"
            
            # Price vs VWAP
            if current_price > vwap * 1.005:
                price_vs_vwap = "ABOVE_VWAP"
            elif current_price < vwap * 0.995:
                price_vs_vwap = "BELOW_VWAP"
            else:
                price_vs_vwap = "AT_VWAP"
            
            # 3. Opening Range Breakout (using previous day high/low as proxy for ORB)
            orb_high = prev_high
            orb_low = prev_low
            orb_range = orb_high - orb_low
            
            if current_price > orb_high:
                orb_signal = "BREAKOUT_UP"
                orb_strength = (current_price - orb_high) / orb_range * 100 if orb_range > 0 else 0
            elif current_price < orb_low:
                orb_signal = "BREAKOUT_DOWN"
                orb_strength = (orb_low - current_price) / orb_range * 100 if orb_range > 0 else 0
            else:
                orb_signal = "INSIDE_ORB"
                orb_strength = 0
            
            # 4. Volume relative to 5-day average
            volume_5d_avg = vol.tail(5).mean() if len(vol) >= 5 else vol.mean()
            volume_vs_avg = current_volume / volume_5d_avg if volume_5d_avg > 0 else 1.0
            
            if volume_vs_avg < 0.5:
                volume_regime = "LOW"
            elif volume_vs_avg < 1.2:
                volume_regime = "NORMAL"
            elif volume_vs_avg < 2.0:
                volume_regime = "HIGH"
            else:
                volume_regime = "EXPLOSIVE"
            
            # === 5. CHOP FILTER (Anti-Whipsaw) ===
            chop_zone = False
            chop_reason = ""
            
            # Calculate ATR/Range ratio for last 5 candles
            if len(df) >= 5:
                recent_high = high.iloc[-5:].max()
                recent_low = low.iloc[-5:].min()
                recent_range = recent_high - recent_low
                atr_range_ratio = recent_range / atr if atr > 0 else 1.0
            else:
                atr_range_ratio = 1.0
            
            # Count ORB re-entries (price crossing back into ORB multiple times)
            orb_reentries = 0
            if len(df) >= 10:
                for i in range(-10, 0):
                    price_i = close.iloc[i]
                    price_prev = close.iloc[i-1] if i > -10 else close.iloc[i]
                    # Check if crossed back into ORB from outside
                    if orb_low <= price_i <= orb_high:
                        if price_prev > orb_high or price_prev < orb_low:
                            orb_reentries += 1
            
            # CHOP Zone detection (any of these = NO TRADE)
            # 1. VWAP is flat AND volume is not high
            if vwap_slope == "FLAT" and volume_regime in ["LOW", "NORMAL"]:
                chop_zone = True
                chop_reason = "VWAP_FLAT+LOW_VOL"
            
            # 2. Range too compressed (< 0.5x ATR over 5 candles)
            elif atr_range_ratio < 0.5:
                chop_zone = True
                chop_reason = "COMPRESSED_RANGE"
            
            # 3. ORB containment with multiple re-entries (whipsaw)
            elif orb_reentries >= 3 and volume_regime in ["LOW", "NORMAL"]:
                chop_zone = True
                chop_reason = "ORB_WHIPSAW"
            
            # === 6. HIGHER TIMEFRAME (HTF) ALIGNMENT ===
            # Use 50-period EMA as HTF proxy (simulates longer timeframe trend)
            htf_trend = "NEUTRAL"
            htf_ema_slope = "FLAT"
            htf_alignment = "NEUTRAL"
            
            if len(df) >= 50:
                ema_50 = close.ewm(span=50, adjust=False).mean()
                ema_50_now = ema_50.iloc[-1]
                ema_50_10_ago = ema_50.iloc[-10] if len(ema_50) >= 10 else ema_50.iloc[0]
                
                # HTF trend based on price vs 50 EMA
                if current_price > ema_50_now * 1.005:
                    htf_trend = "BULLISH"
                elif current_price < ema_50_now * 0.995:
                    htf_trend = "BEARISH"
                else:
                    htf_trend = "NEUTRAL"
                
                # 50 EMA slope (over 10 periods)
                ema_50_change = (ema_50_now - ema_50_10_ago) / ema_50_10_ago * 100 if ema_50_10_ago > 0 else 0
                if ema_50_change > 0.5:
                    htf_ema_slope = "RISING"
                elif ema_50_change < -0.5:
                    htf_ema_slope = "FALLING"
                else:
                    htf_ema_slope = "FLAT"
                
                # Alignment check - does intraday signal match HTF?
                # Will be set by caller based on intended trade direction
                if htf_trend == "BULLISH" and htf_ema_slope == "RISING":
                    htf_alignment = "BULLISH_ALIGNED"
                elif htf_trend == "BEARISH" and htf_ema_slope == "FALLING":
                    htf_alignment = "BEARISH_ALIGNED"
                elif htf_ema_slope == "FLAT":
                    htf_alignment = "NEUTRAL"
                else:
                    htf_alignment = "MIXED"
            
            # Support/Resistance (swing points)
            resistance_1 = high_5d
            resistance_2 = high_20d
            support_1 = low_5d
            support_2 = low_20d
            
            # ATR-based targets
            atr_target = round(current_price + (atr * 1.5), 2)
            atr_stoploss = round(current_price - (atr * 1.0), 2)
            
            return {
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'rsi_14': round(rsi, 2),
                'atr_14': round(atr, 2),
                # Trend
                'trend': trend,
                'high_20d': round(high_20d, 2),
                'low_20d': round(low_20d, 2),
                'high_5d': round(high_5d, 2),
                'low_5d': round(low_5d, 2),
                'prev_high': round(prev_high, 2),
                'prev_low': round(prev_low, 2),
                'prev_close': round(prev_close, 2),
                'resistance_1': round(resistance_1, 2),
                'resistance_2': round(resistance_2, 2),
                'support_1': round(support_1, 2),
                'support_2': round(support_2, 2),
                'atr_target': atr_target,
                'atr_stoploss': atr_stoploss,
                'volume_ratio': round(volume_ratio, 2),
                'volume_signal': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.5 else 'NORMAL',
                # === NEW REGIME DETECTION FIELDS ===
                'vwap': round(vwap, 2),
                'vwap_slope': vwap_slope,
                'price_vs_vwap': price_vs_vwap,
                'ema_9': round(ema_9, 2),
                'ema_21': round(ema_21, 2),
                'ema_spread': round(ema_spread, 2),
                'ema_regime': ema_regime,
                'orb_high': round(orb_high, 2),
                'orb_low': round(orb_low, 2),
                'orb_signal': orb_signal,
                'orb_strength': round(orb_strength, 2),
                'volume_5d_avg': round(volume_5d_avg, 0),
                'volume_vs_avg': round(volume_vs_avg, 2),
                'volume_regime': volume_regime,
                # === CHOP FILTER FIELDS ===
                'chop_zone': chop_zone,
                'chop_reason': chop_reason,
                'atr_range_ratio': round(atr_range_ratio, 2),
                'orb_reentries': orb_reentries,
                # === HTF ALIGNMENT FIELDS ===
                'htf_trend': htf_trend,
                'htf_ema_slope': htf_ema_slope,
                'htf_alignment': htf_alignment
            }
            
        except Exception as e:
            return {}
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float, 
        capital: float,
        lot_size: int = 1,
        atr: float = 0,
        volume_regime: str = "NORMAL",
        spread_bps: float = 0,
        available_margin: float = 0
    ) -> Dict:
        """
        Tool: Calculate adaptive position size based on R-risk and volatility
        
        Key improvements:
        1. R-based sizing: risk_amount = equity * risk_per_trade
        2. Volatility adjustment: Reduce size in high vol
        3. Spread penalty: Reduce size if spread is wide
        4. Margin cap: Don't exceed available margin
        5. Liquidity cap: Don't take too large a position
        """
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {
                "error": "Stop loss equals entry price", 
                "quantity": 0,
                "reason": "Invalid stop loss"
            }
        
        # === BASE R-RISK CALCULATION ===
        # Risk 0.5-1% of capital per trade
        base_risk_pct = HARD_RULES["RISK_PER_TRADE"]  # e.g., 0.005 = 0.5%
        max_risk = capital * base_risk_pct
        
        # Base quantity from R-risk
        base_quantity = int(max_risk / risk_per_share)
        
        # === VOLATILITY ADJUSTMENT ===
        # High volatility = reduce position size for survivability
        vol_multiplier = 1.0
        vol_warning = ""
        
        if atr > 0 and entry_price > 0:
            atr_pct = (atr / entry_price) * 100
            
            if atr_pct > 3.0:  # Very high volatility (>3% daily ATR)
                vol_multiplier = 0.5
                vol_warning = f"High vol ({atr_pct:.1f}% ATR) - size reduced 50%"
            elif atr_pct > 2.0:  # High volatility
                vol_multiplier = 0.7
                vol_warning = f"Elevated vol ({atr_pct:.1f}% ATR) - size reduced 30%"
            elif atr_pct > 1.5:  # Moderate volatility
                vol_multiplier = 0.85
                vol_warning = f"Moderate vol ({atr_pct:.1f}% ATR) - size reduced 15%"
        
        # === VOLUME REGIME ADJUSTMENT ===
        # EXPLOSIVE = allow normal, HIGH = normal, NORMAL = reduce, LOW = block
        regime_multiplier = 1.0
        regime_warning = ""
        
        if volume_regime == "LOW":
            return {
                "error": "Volume too low for safe entry",
                "quantity": 0,
                "reason": "LOW volume regime - trading blocked"
            }
        elif volume_regime == "NORMAL":
            regime_multiplier = 0.8
            regime_warning = "Normal volume - conservative sizing"
        elif volume_regime == "EXPLOSIVE":
            regime_multiplier = 1.0  # Full size OK
        
        # === SPREAD PENALTY ===
        # Wide spreads eat into profits - reduce size
        spread_multiplier = 1.0
        spread_warning = ""
        
        if spread_bps > 30:  # Wide spread
            spread_multiplier = 0.7
            spread_warning = f"Wide spread ({spread_bps:.0f} bps) - size reduced"
        elif spread_bps > 20:
            spread_multiplier = 0.85
        
        # === APPLY ALL ADJUSTMENTS ===
        adjusted_quantity = int(base_quantity * vol_multiplier * regime_multiplier * spread_multiplier)
        
        # === CAP BY MARGIN ===
        if available_margin > 0:
            # Assume 5x leverage for F&O, 1x for equity
            max_position_value = available_margin * 0.8  # Use only 80% of margin
            max_qty_by_margin = int(max_position_value / entry_price)
            
            if adjusted_quantity > max_qty_by_margin:
                adjusted_quantity = max_qty_by_margin
        
        # === CAP BY MAX POSITION SIZE ===
        # Never risk more than 25% of capital in one position
        max_position_pct = 0.25
        max_position_value = capital * max_position_pct
        max_qty_by_position = int(max_position_value / entry_price)
        
        if adjusted_quantity > max_qty_by_position:
            adjusted_quantity = max_qty_by_position
        
        # === ROUND TO LOT SIZE ===
        if lot_size > 1:
            adjusted_quantity = (adjusted_quantity // lot_size) * lot_size
        
        # Minimum quantity check
        if adjusted_quantity < 1:
            adjusted_quantity = 1 if lot_size == 1 else lot_size
        
        # === FINAL CALCULATIONS ===
        actual_risk = risk_per_share * adjusted_quantity
        actual_risk_pct = (actual_risk / capital) * 100
        position_value = entry_price * adjusted_quantity
        position_pct = (position_value / capital) * 100
        
        # Collect warnings
        warnings = []
        if vol_warning:
            warnings.append(vol_warning)
        if regime_warning:
            warnings.append(regime_warning)
        if spread_warning:
            warnings.append(spread_warning)
        
        return {
            "quantity": adjusted_quantity,
            "risk_per_share": round(risk_per_share, 2),
            "max_risk_allowed": round(max_risk, 2),
            "actual_risk": round(actual_risk, 2),
            "actual_risk_pct": round(actual_risk_pct, 3),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            # Adjustment factors
            "base_quantity": base_quantity,
            "vol_multiplier": vol_multiplier,
            "regime_multiplier": regime_multiplier,
            "spread_multiplier": spread_multiplier,
            "warnings": warnings,
            "sizing_method": "R-RISK_ADAPTIVE"
        }
    
    def validate_trade(self, trade_plan: Dict) -> Dict:
        """
        Tool: Validate a trade plan against all hard rules
        Returns pass/fail for each rule
        """
        checks = []
        all_passed = True
        
        # Get current account state
        account = self.get_account_state()
        
        # 1. Check if trading is allowed
        if not account.get('can_trade', False):
            checks.append(RiskCheck(
                rule="TRADING_ALLOWED",
                passed=False,
                current_value=account.get('reason'),
                limit="Must be able to trade",
                message=f"Trading blocked: {account.get('reason')}"
            ))
            all_passed = False
        else:
            checks.append(RiskCheck(
                rule="TRADING_ALLOWED",
                passed=True,
                current_value="OK",
                limit="Must be able to trade",
                message="Trading is allowed"
            ))
        
        # 2. Check symbol is in approved universe
        symbol = trade_plan.get('symbol', '')
        in_universe = symbol in APPROVED_UNIVERSE
        checks.append(RiskCheck(
            rule="APPROVED_UNIVERSE",
            passed=in_universe,
            current_value=symbol,
            limit=f"Must be in {len(APPROVED_UNIVERSE)} approved symbols",
            message="Symbol approved" if in_universe else f"{symbol} not in approved universe"
        ))
        if not in_universe:
            all_passed = False
        
        # 3. Check risk per trade
        risk_pct = trade_plan.get('risk_pct', 1)
        risk_ok = risk_pct <= HARD_RULES["RISK_PER_TRADE"] * 100
        checks.append(RiskCheck(
            rule="RISK_PER_TRADE",
            passed=risk_ok,
            current_value=f"{risk_pct:.3f}%",
            limit=f"<= {HARD_RULES['RISK_PER_TRADE']*100}%",
            message="Risk within limit" if risk_ok else "Risk exceeds limit"
        ))
        if not risk_ok:
            all_passed = False
        
        # 4. Check max positions
        current_positions = account.get('positions_count', 0)
        positions_ok = current_positions < HARD_RULES["MAX_POSITIONS"]
        checks.append(RiskCheck(
            rule="MAX_POSITIONS",
            passed=positions_ok,
            current_value=current_positions,
            limit=f"< {HARD_RULES['MAX_POSITIONS']}",
            message="Position limit OK" if positions_ok else "Max positions reached"
        ))
        if not positions_ok:
            all_passed = False
        
        # 5. Check stop loss exists and is in correct direction
        has_sl = trade_plan.get('stop_loss', 0) > 0
        entry_price = trade_plan.get('entry_price', 0)
        stop_loss = trade_plan.get('stop_loss', 0)
        side = trade_plan.get('side', 'BUY')
        
        sl_direction_ok = True
        sl_message = "Stop loss set"
        
        if has_sl and entry_price > 0:
            if side == 'BUY':
                # For long: SL must be BELOW entry
                if stop_loss >= entry_price:
                    sl_direction_ok = False
                    sl_message = f"BUY trade: SL (₹{stop_loss}) must be BELOW entry (₹{entry_price})"
            elif side == 'SELL':
                # For short: SL must be ABOVE entry
                if stop_loss <= entry_price:
                    sl_direction_ok = False
                    sl_message = f"SELL trade: SL (₹{stop_loss}) must be ABOVE entry (₹{entry_price})"
        
        sl_valid = has_sl and sl_direction_ok
        checks.append(RiskCheck(
            rule="STOP_LOSS_REQUIRED",
            passed=sl_valid,
            current_value=f"SL: ₹{stop_loss}, Entry: ₹{entry_price}, Side: {side}",
            limit="Must have correctly placed stop loss",
            message=sl_message if sl_valid else sl_message
        ))
        if not sl_valid:
            all_passed = False
        
        # 6. Check daily loss limit
        daily_loss_pct = account.get('daily_loss_pct', 0)
        daily_ok = daily_loss_pct < HARD_RULES["MAX_DAILY_LOSS"]
        checks.append(RiskCheck(
            rule="DAILY_LOSS_LIMIT",
            passed=daily_ok,
            current_value=f"{daily_loss_pct*100:.2f}%",
            limit=f"< {HARD_RULES['MAX_DAILY_LOSS']*100}%",
            message="Daily loss OK" if daily_ok else "Daily loss limit reached"
        ))
        if not daily_ok:
            all_passed = False
        
        return {
            "all_passed": all_passed,
            "checks": [asdict(c) for c in checks],
            "trade_approved": all_passed,
            "message": "✅ All checks passed - trade can proceed" if all_passed else "❌ Trade blocked - rule violations detected"
        }
    
    def place_order(self, order: Dict) -> Dict:
        """
        Tool: Place an order with Zerodha (or simulate in paper mode)
        Only works if validate_trade passes, execution guard approves, and no duplicate exists
        Uses idempotent order engine to prevent duplicate orders on reconnect/retry
        """
        symbol = order.get('symbol', '')
        side = order.get('side', 'BUY')
        quantity = order.get('quantity', 0)
        strategy = order.get('strategy', 'MANUAL')
        setup_id = order.get('setup_id', order.get('rationale', 'DEFAULT')[:20])
        
        # === POSITION RECONCILIATION CHECK ===
        recon = get_position_reconciliation(kite=self.kite, paper_mode=self.paper_mode)
        recon_can_trade, recon_reason = recon.can_trade()
        if not recon_can_trade:
            return {
                "success": False,
                "error": f"RECONCILIATION BLOCK: {recon_reason}",
                "state": recon.state.value,
                "action": "Resolve mismatch before new trades"
            }
        
        # === DATA HEALTH GATE CHECK ===
        health_gate = get_data_health_gate()
        try:
            md = self.get_market_data([symbol])
            if symbol in md and isinstance(md[symbol], dict):
                health_ok, health_reason = health_gate.can_trade(symbol, md[symbol])
                if not health_ok:
                    return {
                        "success": False,
                        "error": f"DATA HEALTH BLOCK: {health_reason}",
                        "stale_counter": health_gate.get_stale_counter(symbol),
                        "action": "Wait for healthy data"
                    }
        except Exception as e:
            print(f"   ⚠️ Data health check failed: {e}")
        
        # === IDEMPOTENT ORDER CHECK ===
        idempotent_engine = get_idempotent_engine()
        
        # Create order intent
        order_intent = idempotent_engine.create_intent(
            symbol=symbol,
            direction=side,
            strategy=strategy,
            setup_id=setup_id
        )
        
        # Check if this order intent was already placed
        try:
            # Get broker orders for duplicate check
            broker_open_orders = self.kite.orders() if not self.paper_mode else []
            can_place, reason = idempotent_engine.can_place_order(
                intent=order_intent,
                broker_open_orders=broker_open_orders
            )
            
            if not can_place:
                return {
                    "success": False,
                    "error": f"IDEMPOTENT BLOCK: {reason}",
                    "client_order_id": order_intent.generate_client_order_id(),
                    "intent": {
                        "symbol": symbol,
                        "direction": side,
                        "strategy": strategy,
                        "setup_id": setup_id
                    }
                }
        except Exception as e:
            print(f"   ⚠️ Idempotency check failed: {e} - proceeding with caution")
        
        # Store intent for later recording
        order['_order_intent'] = order_intent
        
        # CHECK FOR DUPLICATE TRADE (active position check)
        if self.is_symbol_in_active_trades(symbol):
            existing = self.get_active_trade(symbol)
            return {
                "success": False,
                "error": f"DUPLICATE TRADE BLOCKED: {symbol} already has an active position",
                "existing_trade": {
                    "symbol": existing['symbol'],
                    "side": existing['side'],
                    "entry": existing['avg_price'],
                    "stop_loss": existing['stop_loss'],
                    "target": existing.get('target'),
                    "opened_at": existing.get('timestamp')
                }
            }
        
        # === CORRELATION & INDEX GUARD CHECK ===
        correlation_guard = get_correlation_guard()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        corr_check = correlation_guard.can_trade(
            symbol=symbol,
            active_positions=active_positions
        )
        
        if not corr_check.can_trade:
            return {
                "success": False,
                "error": f"CORRELATION BLOCK: {corr_check.reason}",
                "beta_category": corr_check.beta_category,
                "correlated_positions": corr_check.correlated_positions,
                "warnings": corr_check.warnings
            }
        
        # Log correlation warnings if any
        if corr_check.warnings:
            for w in corr_check.warnings:
                print(f"   ⚠️ Correlation: {w}")
        
        # === REGIME SCORE CHECK ===
        regime_scorer = get_regime_scorer()
        
        # Get market data for scoring
        market_data = {}
        try:
            md = self.get_market_data([symbol])
            if symbol in md and isinstance(md[symbol], dict):
                market_data = md[symbol]
        except:
            pass
        
        # Calculate regime score
        regime_result = regime_scorer.calculate_score(
            symbol=symbol,
            direction=side,
            trade_type=strategy,
            market_data=market_data
        )
        
        if not regime_result.passes_threshold:
            return {
                "success": False,
                "error": f"REGIME SCORE BLOCK: {regime_result.total_score}/{regime_result.threshold} - {regime_result.final_verdict}",
                "score": regime_result.total_score,
                "threshold": regime_result.threshold,
                "confidence": regime_result.confidence,
                "summary": regime_result.summary,
                "breakdown": [{"name": c.name, "points": c.points, "reason": c.reason} for c in regime_result.components]
            }
        
        print(f"   📊 Regime Score: {regime_result.total_score}/{regime_result.threshold} ({regime_result.confidence})")
        order['_regime_score'] = regime_result.total_score
        order['_regime_confidence'] = regime_result.confidence
        
        # First validate
        validation = self.validate_trade(order)
        if not validation['all_passed']:
            return {
                "success": False,
                "error": "Trade validation failed",
                "validation": validation
            }
        
        # === EXECUTION GUARD: Check spread and get execution policy ===
        execution_guard = get_execution_guard()
        try:
            quote = self.kite.quote([symbol])
            ltp = quote[symbol]['last_price']
            depth = quote[symbol].get('depth', {})
            
            # Get bid/ask from depth
            buy_depth = depth.get('buy', [{}])
            sell_depth = depth.get('sell', [{}])
            
            bid = buy_depth[0].get('price', ltp * 0.999) if buy_depth else ltp * 0.999
            ask = sell_depth[0].get('price', ltp * 1.001) if sell_depth else ltp * 1.001
            bid_qty = buy_depth[0].get('quantity', 0) if buy_depth else 0
            ask_qty = sell_depth[0].get('quantity', 0) if sell_depth else 0
            
            # Get volume regime from market data
            market_data = self.get_market_data([symbol])
            volume_regime = market_data.get(symbol, {}).get('volume_regime', 'NORMAL')
            
            # Check execution policy
            exec_policy = execution_guard.get_execution_policy(
                symbol=symbol,
                side=side,
                quantity=quantity,
                ltp=ltp,
                bid=bid,
                ask=ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                volume_regime=volume_regime
            )
            
            if not exec_policy.can_execute:
                return {
                    "success": False,
                    "error": f"EXECUTION BLOCKED: {exec_policy.reason}",
                    "execution_policy": {
                        "order_type": exec_policy.order_type,
                        "reason": exec_policy.reason,
                        "warnings": exec_policy.warnings
                    }
                }
            
            # Log warnings if any
            if exec_policy.warnings:
                print(f"   ⚠️ Execution warnings: {', '.join(exec_policy.warnings)}")
            
            # Store expected entry price for slippage calculation
            expected_entry = ask if side == 'BUY' else bid
            order['expected_entry'] = expected_entry
            order['volume_regime'] = volume_regime
            order['execution_order_type'] = exec_policy.order_type
            order['max_slippage_pct'] = exec_policy.max_slippage_pct
            
            # === ADAPTIVE POSITION SIZING: Recalculate quantity ===
            # Get market data for ATR and spread
            atr = market_data.get(symbol, {}).get('atr_14', 0)
            spread_bps = ((ask - bid) / ((ask + bid) / 2)) * 10000 if ask > 0 and bid > 0 else 0
            
            entry_price = ltp
            stop_loss = order.get('stop_loss', ltp * 0.99 if side == 'BUY' else ltp * 1.01)
            
            # Calculate adaptive position size
            sizing = self.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                capital=self.paper_capital if hasattr(self, 'paper_capital') else 100000,
                lot_size=1,
                atr=atr,
                volume_regime=volume_regime,
                spread_bps=spread_bps,
                available_margin=0  # Will use default caps
            )
            
            if 'error' in sizing:
                return {
                    "success": False,
                    "error": f"SIZING BLOCKED: {sizing.get('reason', sizing.get('error'))}",
                    "sizing": sizing
                }
            
            # Override quantity with adaptive sizing
            original_qty = order.get('quantity', 0)
            order['quantity'] = sizing['quantity']
            
            if original_qty != sizing['quantity']:
                print(f"   📊 Adaptive sizing: {original_qty} → {sizing['quantity']}")
                if sizing.get('warnings'):
                    for w in sizing['warnings']:
                        print(f"      ⚠️ {w}")
            
        except Exception as e:
            print(f"   ⚠️ Execution guard check failed: {e} - proceeding with defaults")
        
        self._rate_limit()
        
        try:
            # PAPER MODE: Simulate the order
            if self.paper_mode:
                import random
                paper_order_id = f"PAPER_{random.randint(100000, 999999)}"
                
                # ALWAYS use CURRENT LTP as entry price (not what LLM passed)
                try:
                    quote = self.kite.quote([symbol])
                    current_ltp = quote[symbol]['last_price']
                except:
                    current_ltp = order.get('entry_price', 0)
                
                # Calculate proper SL and target based on ACTUAL entry
                side = order['side']
                if side == 'BUY':
                    # For BUY: SL below entry, target above
                    stop_loss = current_ltp * 0.99  # 1% below
                    target = current_ltp * 1.015     # 1.5% above
                else:
                    # For SELL/SHORT: SL above entry, target below
                    stop_loss = current_ltp * 1.01  # 1% above
                    target = current_ltp * 0.985    # 1.5% below
                
                # Add to paper positions
                position = {
                    'symbol': symbol,
                    'quantity': order['quantity'],
                    'avg_price': current_ltp,  # ACTUAL market price
                    'side': side,
                    'stop_loss': stop_loss,
                    'target': target,
                    'order_id': paper_order_id,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': order.get('rationale', 'No rationale provided'),
                    'volume_regime': order.get('volume_regime', 'NORMAL'),
                    'expected_entry': order.get('expected_entry', current_ltp)
                }
                self.paper_positions.append(position)
                
                # SAVE TO FILE immediately
                self._save_active_trades()
                
                # Record slippage
                expected_entry = order.get('expected_entry', current_ltp)
                execution_guard.record_slippage(
                    symbol=symbol,
                    side=side,
                    expected_price=expected_entry,
                    actual_price=current_ltp,
                    volume_regime=order.get('volume_regime', 'NORMAL'),
                    order_type=order.get('execution_order_type', 'MARKET')
                )
                
                # Update paper capital (reduce by position value)
                position_value = order['quantity'] * current_ltp
                
                print(f"   ✅ {side} {order['quantity']} {symbol} @ ₹{current_ltp:.2f}")
                print(f"      SL: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                
                # === RECORD ORDER IN IDEMPOTENT ENGINE ===
                order_intent = order.get('_order_intent')
                if order_intent:
                    idempotent_engine.record_order(
                        intent=order_intent,
                        broker_order_id=paper_order_id,
                        quantity=order['quantity'],
                        price=current_ltp,
                        status="COMPLETE"
                    )
                
                return {
                    "success": True,
                    "paper_trade": True,
                    "order_id": paper_order_id,
                    "sl_order_id": f"PAPER_SL_{random.randint(100000, 999999)}",
                    "message": f"📝 PAPER ORDER: {side} {order['quantity']} {symbol} @ ₹{current_ltp:.2f}",
                    "position_value": position_value,
                    "entry_price": current_ltp,
                    "stop_loss": stop_loss,
                    "target": target,
                    "client_order_id": order_intent.generate_client_order_id() if order_intent else None,
                    "details": order
                }
            
            # LIVE MODE: Place real order
            exchange, tradingsymbol = symbol.split(":")
            
            # Place main order
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if order['side'] == 'BUY' else self.kite.TRANSACTION_TYPE_SELL,
                quantity=order['quantity'],
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY
            )
            
            # Place SL order
            sl_side = self.kite.TRANSACTION_TYPE_SELL if order['side'] == 'BUY' else self.kite.TRANSACTION_TYPE_BUY
            sl_trigger = order['stop_loss'] * (0.999 if order['side'] == 'BUY' else 1.001)
            
            sl_order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=sl_side,
                quantity=order['quantity'],
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_SLM,
                trigger_price=sl_trigger,
                validity=self.kite.VALIDITY_DAY
            )
            
            # === RECORD ORDER IN IDEMPOTENT ENGINE ===
            order_intent = order.get('_order_intent')
            if order_intent:
                idempotent_engine.record_order(
                    intent=order_intent,
                    broker_order_id=str(order_id),
                    quantity=order['quantity'],
                    price=order.get('entry_price', 0),
                    status="OPEN"
                )
            
            return {
                "success": True,
                "order_id": order_id,
                "sl_order_id": sl_order_id,
                "message": f"Order placed: {order['side']} {order['quantity']} {symbol}",
                "client_order_id": order_intent.generate_client_order_id() if order_intent else None,
                "details": order
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def place_option_order(self, underlying: str, direction: str, 
                          option_type: str = None, 
                          strike_selection: str = "ATM",
                          expiry_selection: str = "CURRENT_WEEK",
                          rationale: str = "",
                          use_intraday_scoring: bool = True) -> Dict:
        """
        Tool: Place an option order instead of equity order
        
        INTRADAY SIGNALS ARE HIGHEST PRIORITY for decision making.
        When use_intraday_scoring=True, the system analyzes:
        - ORB breakout (25 points) - HIGHEST
        - Volume regime (20 points) 
        - VWAP position/trend (15 points)
        - EMA regime (15 points)
        - HTF alignment (10 points)
        - RSI extremes (10 points)
        - Price momentum (5 points)
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            direction: 'BUY' or 'SELL' signal on underlying
            option_type: 'CE' or 'PE' (None = auto based on direction)
            strike_selection: 'ATM', 'ITM_1', 'ITM_2', 'OTM_1', 'OTM_2'
            expiry_selection: 'CURRENT_WEEK', 'NEXT_WEEK', 'CURRENT_MONTH'
            rationale: Trade rationale
            use_intraday_scoring: If True, uses intraday signals for decision (default: True)
            
        Returns:
            Dict with order result including Greeks and intraday decision
        """
        from options_trader import get_intraday_scorer, IntradaySignal
        
        # Check if symbol is F&O eligible
        options_trader = get_options_trader(
            kite=self.kite, 
            capital=getattr(self, 'paper_capital', 100000),
            paper_mode=self.paper_mode
        )
        
        if not options_trader.should_use_options(underlying):
            return {
                "success": False,
                "error": f"{underlying} is not F&O eligible",
                "action": "Use equity order instead"
            }
        
        # === GET INTRADAY MARKET DATA ===
        market_data = {}
        if use_intraday_scoring:
            try:
                md = self.get_market_data([underlying])
                if underlying in md and isinstance(md[underlying], dict):
                    market_data = md[underlying]
                    print(f"   📊 Using intraday signals for {underlying}")
            except Exception as e:
                print(f"   ⚠️ Could not get intraday data: {e}")
        
        # Parse enums
        opt_type = None
        if option_type:
            opt_type = OptionType[option_type.upper()]
        
        strike_sel = StrikeSelection[strike_selection.upper()] if strike_selection != "AUTO" else None
        expiry_sel = ExpirySelection[expiry_selection.upper()] if expiry_selection != "AUTO" else None
        
        # === CREATE OPTION ORDER WITH INTRADAY SCORING ===
        plan = options_trader.create_option_order(
            underlying=underlying,
            direction=direction,
            option_type=opt_type,
            strike_selection=strike_sel,
            expiry_selection=expiry_sel,
            market_data=market_data if use_intraday_scoring else None
        )
        
        if plan is None:
            return {
                "success": False,
                "error": f"Could not create option order for {underlying}",
                "reason": "No suitable strikes or chain unavailable"
            }
        
        # Validate risk - max premium check
        max_premium_per_trade = 15000  # ₹15K per option trade
        if plan.total_premium > max_premium_per_trade:
            return {
                "success": False,
                "error": f"Premium ₹{plan.total_premium:.0f} exceeds limit ₹{max_premium_per_trade}",
                "action": "Reduce quantity or choose different strike"
            }
        
        # Check total options exposure
        portfolio_greeks = options_trader.get_portfolio_greeks()
        max_total_exposure = 50000  # ₹50K total option exposure
        current_exposure = sum(p.get('total_premium', 0) for p in options_trader.positions if p.get('status') == 'OPEN')
        
        if current_exposure + plan.total_premium > max_total_exposure:
            return {
                "success": False,
                "error": f"Total option exposure would exceed ₹{max_total_exposure}",
                "current_exposure": current_exposure,
                "new_premium": plan.total_premium
            }
        
        # Execute the order
        result = options_trader.execute_option_order(plan)
        
        if result.get('success'):
            print(f"   📊 OPTION ORDER: {plan.contract.symbol}")
            print(f"      Premium: ₹{plan.total_premium:.0f} | Lots: {plan.quantity}")
            print(f"      Greeks: {plan.greeks_summary}")
            print(f"      Target: ₹{plan.target_premium:.2f} | SL: ₹{plan.stoploss_premium:.2f}")
            
            # Add to paper positions if paper mode
            if self.paper_mode:
                option_position = {
                    'symbol': plan.contract.symbol,
                    'underlying': plan.underlying,
                    'quantity': plan.quantity * plan.contract.lot_size,
                    'lots': plan.quantity,
                    'avg_price': plan.contract.ltp,
                    'side': direction,
                    'option_type': plan.contract.option_type.value,
                    'strike': plan.contract.strike,
                    'expiry': plan.contract.expiry.isoformat() if plan.contract.expiry else None,
                    'stop_loss': plan.stoploss_premium,
                    'target': plan.target_premium,
                    'order_id': result.get('order_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'is_option': True,
                    'total_premium': plan.total_premium,
                    'max_loss': plan.max_loss,
                    'breakeven': plan.breakeven,
                    'delta': plan.contract.delta,
                    'theta': plan.contract.theta,
                    'iv': plan.contract.iv,
                    'rationale': rationale or plan.rationale
                }
                self.paper_positions.append(option_position)
                self._save_active_trades()
        
        return result
    
    def get_option_greeks_update(self, symbol: str = None) -> Dict:
        """
        Tool: Get current Greeks for option positions
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with Greeks updates and exit signals
        """
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 100000),
            paper_mode=self.paper_mode
        )
        
        if symbol:
            # Check specific position
            for pos in options_trader.positions:
                if pos.get('symbol') == symbol and pos.get('status') == 'OPEN':
                    exit_signal = options_trader.check_option_exit(symbol)
                    return {
                        "symbol": symbol,
                        "position": pos,
                        "exit_signal": exit_signal
                    }
            return {"error": f"No open option position for {symbol}"}
        
        # Get all Greeks
        return {
            "portfolio_greeks": options_trader.get_portfolio_greeks(),
            "positions": [p for p in options_trader.positions if p.get('status') == 'OPEN']
        }
    
    def check_option_exits(self) -> List[Dict]:
        """
        Tool: Check all option positions for exit signals
        
        Returns:
            List of positions that need attention
        """
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 100000),
            paper_mode=self.paper_mode
        )
        
        exit_signals = []
        
        for pos in options_trader.positions:
            if pos.get('status') != 'OPEN':
                continue
            
            symbol = pos.get('symbol')
            signal = options_trader.check_option_exit(symbol)
            
            if signal and signal.get('should_exit'):
                exit_signals.append({
                    "symbol": symbol,
                    "reason": signal.get('reason'),
                    "exit_type": signal.get('exit_type'),
                    "current_pnl": signal.get('current_pnl', 0),
                    "days_to_expiry": signal.get('days_to_expiry', 0),
                    "delta": pos.get('delta', 0),
                    "theta": pos.get('theta', 0)
                })
        
        return exit_signals

    def get_portfolio_risk(self) -> Dict:
        """
        Tool: Get current portfolio risk exposure
        """
        account = self.get_account_state()
        
        total_exposure = 0
        total_risk = 0
        
        for pos in account.get('open_positions', []):
            qty = abs(pos['quantity'])
            price = pos['ltp']
            exposure = qty * price
            total_exposure += exposure
            # Assume 2% risk per position if SL unknown
            total_risk += exposure * 0.02
        
        capital = account.get('total_equity', 0)
        
        return {
            "total_equity": round(capital, 2),
            "total_exposure": round(total_exposure, 2),
            "exposure_pct": round(total_exposure / capital * 100, 2) if capital > 0 else 0,
            "estimated_risk": round(total_risk, 2),
            "risk_pct": round(total_risk / capital * 100, 2) if capital > 0 else 0,
            "positions_count": len(account.get('open_positions', [])),
            "max_positions": HARD_RULES["MAX_POSITIONS"],
            "daily_pnl": round(account.get('realized_pnl', 0) + account.get('unrealized_pnl', 0), 2),
            "daily_loss_limit_remaining": round(
                (HARD_RULES["MAX_DAILY_LOSS"] * capital) - abs(account.get('daily_loss', 0)), 2
            )
        }
    
    def get_options_chain(self, symbol: str, bias: str = "neutral") -> Dict:
        """
        Tool: Get options chain for a stock with best option recommendations
        
        Args:
            symbol: Stock symbol like 'NSE:RELIANCE'
            bias: 'bullish', 'bearish', or 'neutral'
        
        Returns:
            Options chain with ATM strikes and recommendations
        """
        self._rate_limit()
        
        try:
            # Extract trading symbol
            exchange, tradingsymbol = symbol.split(":")
            
            # Get spot price
            quote = self.kite.quote([symbol])
            spot_price = quote[symbol]['last_price']
            
            # Get all instruments
            instruments = self.kite.instruments("NFO")
            
            # Filter options for this stock
            stock_options = [i for i in instruments 
                           if i['name'] == tradingsymbol 
                           and i['instrument_type'] in ['CE', 'PE']]
            
            if not stock_options:
                return {"error": f"No options found for {tradingsymbol}"}
            
            # Get nearest expiry
            expiries = sorted(set([i['expiry'] for i in stock_options]))
            nearest_expiry = expiries[0]
            
            # Filter to nearest expiry
            options = [o for o in stock_options if o['expiry'] == nearest_expiry]
            
            # Find ATM strike
            strikes = sorted(set([o['strike'] for o in options]))
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
            atm_idx = strikes.index(atm_strike)
            
            # Get relevant strikes (ATM ± 2)
            relevant_strikes = strikes[max(0, atm_idx-2):atm_idx+3]
            
            # Get tokens for relevant options
            relevant_options = [o for o in options if o['strike'] in relevant_strikes]
            tokens = [o['instrument_token'] for o in relevant_options]
            
            # Get quotes
            quotes = self.kite.quote(tokens)
            
            # Build chain
            chain = {}
            for opt in relevant_options:
                strike = opt['strike']
                if strike not in chain:
                    chain[strike] = {'CE': None, 'PE': None}
                
                token = opt['instrument_token']
                q = quotes.get(str(token), quotes.get(token, {}))
                
                chain[strike][opt['instrument_type']] = {
                    'symbol': f"NFO:{opt['tradingsymbol']}",
                    'token': token,
                    'ltp': q.get('last_price', 0),
                    'bid': q.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                    'ask': q.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                    'oi': q.get('oi', 0),
                    'volume': q.get('volume', 0),
                    'lot_size': opt['lot_size']
                }
            
            # Recommend best option based on bias
            recommendation = None
            if bias == "bullish":
                ce = chain[atm_strike].get('CE')
                if ce:
                    recommendation = {
                        "action": "BUY",
                        "option_type": "CE",
                        "strike": atm_strike,
                        "symbol": ce['symbol'],
                        "premium": ce['ltp'],
                        "lot_size": ce['lot_size'],
                        "cost_per_lot": ce['ltp'] * ce['lot_size'],
                        "rationale": f"ATM Call for bullish view on {tradingsymbol}"
                    }
            elif bias == "bearish":
                pe = chain[atm_strike].get('PE')
                if pe:
                    recommendation = {
                        "action": "BUY",
                        "option_type": "PE",
                        "strike": atm_strike,
                        "symbol": pe['symbol'],
                        "premium": pe['ltp'],
                        "lot_size": pe['lot_size'],
                        "cost_per_lot": pe['ltp'] * pe['lot_size'],
                        "rationale": f"ATM Put for bearish view on {tradingsymbol}"
                    }
            
            return {
                "symbol": symbol,
                "spot_price": spot_price,
                "expiry": str(nearest_expiry),
                "atm_strike": atm_strike,
                "chain": chain,
                "recommendation": recommendation,
                "fno_enabled": FNO_CONFIG.get('enabled', False)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def place_option_order(self, option_symbol: str, action: str, quantity: int, 
                           premium: float, stop_loss_pct: float = 30) -> Dict:
        """
        Tool: Place an option order
        
        Args:
            option_symbol: Full option symbol like 'NFO:RELIANCE25FEB2500CE'
            action: 'BUY' only (no selling allowed)
            quantity: Number of lots
            premium: Expected premium per share
            stop_loss_pct: Stop loss as % of premium (default 30%)
        """
        if action != "BUY":
            return {"error": "Only BUY orders allowed for options"}
        
        self._rate_limit()
        
        try:
            exchange, tradingsymbol = option_symbol.split(":")
            
            # Get lot size (estimate for paper mode)
            lot_size = 1000  # Default lot size estimate
            
            if not self.paper_mode:
                # Get actual lot size from instruments
                instruments = self.kite.instruments("NFO")
                for inst in instruments:
                    if inst['tradingsymbol'] == tradingsymbol:
                        lot_size = inst['lot_size']
                        break
            
            total_qty = quantity * lot_size
            
            # PAPER MODE: Simulate the option order
            if self.paper_mode:
                import random
                paper_order_id = f"PAPER_OPT_{random.randint(100000, 999999)}"
                
                position = {
                    'symbol': option_symbol,
                    'quantity': total_qty,
                    'lots': quantity,
                    'lot_size': lot_size,
                    'avg_price': premium,
                    'side': 'BUY',
                    'stop_loss': premium * (1 - stop_loss_pct/100),
                    'type': 'OPTION',
                    'order_id': paper_order_id,
                    'timestamp': datetime.now().isoformat()
                }
                self.paper_positions.append(position)
                
                cost = premium * total_qty
                
                return {
                    "success": True,
                    "paper_trade": True,
                    "order_id": paper_order_id,
                    "message": f"📝 PAPER OPTION: BUY {quantity} lots of {tradingsymbol} @ ₹{premium:.2f}",
                    "details": {
                        "symbol": option_symbol,
                        "lots": quantity,
                        "lot_size": lot_size,
                        "total_qty": total_qty,
                        "expected_premium": premium,
                        "expected_cost": cost,
                        "stop_loss_at": premium * (1 - stop_loss_pct/100)
                    }
                }
            
            # LIVE MODE: Place real order
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=total_qty,
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            
            return {
                "success": True,
                "order_id": order_id,
                "message": f"Option order placed: BUY {quantity} lots of {tradingsymbol}",
                "details": {
                    "symbol": option_symbol,
                    "lots": quantity,
                    "lot_size": lot_size,
                    "total_qty": total_qty,
                    "expected_premium": premium,
                    "expected_cost": premium * total_qty,
                    "stop_loss_at": premium * (1 - stop_loss_pct/100)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Create singleton instance
_tools = None

def get_tools(paper_mode: bool = True, paper_capital: float = None) -> ZerodhaTools:
    global _tools
    if _tools is None:
        _tools = ZerodhaTools(paper_mode=paper_mode, paper_capital=paper_capital)
    return _tools

def reset_tools():
    """Reset the singleton to allow new configuration"""
    global _tools
    _tools = None


if __name__ == "__main__":
    # Test the tools
    tools = get_tools()
    
    print("\n" + "="*60)
    print("ZERODHA TOOLS TEST")
    print("="*60)
    
    # Test account state
    print("\n📊 Account State:")
    account = tools.get_account_state()
    if 'error' not in account:
        print(f"   Available Margin: ₹{account['available_margin']:,.2f}")
        print(f"   Open Positions: {account['positions_count']}")
        print(f"   Daily P&L: ₹{account['realized_pnl'] + account['unrealized_pnl']:,.2f}")
        print(f"   Can Trade: {account['can_trade']} ({account['reason']})")
    else:
        print(f"   Error: {account['error']}")
    
    # Test position sizing
    print("\n📐 Position Size Calculation:")
    sizing = tools.calculate_position_size(
        entry_price=2500,
        stop_loss=2450,
        capital=200000,
        lot_size=1
    )
    print(f"   Entry: ₹2500, SL: ₹2450")
    print(f"   Quantity: {sizing['quantity']}")
    print(f"   Risk: ₹{sizing['actual_risk']} ({sizing['actual_risk_pct']}%)")
    
    # Test trade validation
    print("\n✅ Trade Validation:")
    validation = tools.validate_trade({
        'symbol': 'NSE:RELIANCE',
        'side': 'BUY',
        'entry_price': 2500,
        'stop_loss': 2450,
        'target': 2600,
        'quantity': 20,
        'risk_pct': 0.5
    })
    print(f"   All Passed: {validation['all_passed']}")
    for check in validation['checks']:
        emoji = "✅" if check['passed'] else "❌"
        print(f"   {emoji} {check['rule']}: {check['message']}")
