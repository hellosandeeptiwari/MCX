"""
ZERODHA TOOLS FOR THE AGENT
Provides structured tools that the LLM agent can call
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from config import GTT_CONFIG, AUTOSLICE_ENABLED, calc_brokerage
from execution_guard import get_execution_guard
from idempotent_order_engine import get_idempotent_engine
from correlation_guard import get_correlation_guard
from regime_score import get_regime_scorer
from safe_io import atomic_json_save
from state_db import get_state_db
from position_reconciliation import get_position_reconciliation
from data_health_gate import get_data_health_gate
from options_trader import get_options_trader, OptionType, StrikeSelection, ExpirySelection, get_intraday_scorer, IntradaySignal
from kite_ticker import get_ticker, TitanTicker


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
    ema_regime: str = "NORMAL"         # COMPRESSED, EXPANDING, NORMAL
    
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

    # === TREND ENGINE FIELDS (must flow through to TrendFollowing) ===
    vwap_change_pct: float = 0.0       # Numeric VWAP change % (precise, not lossy string)
    orb_hold_candles: int = 0          # Post-ORB candles holding breakout level
    adx: float = 20.0                  # ADX-14 trend strength (>25 = strong)

    # === ACCELERATION FIELDS (must flow through to IntradayOptionScorer) ===
    follow_through_candles: int = 0    # Confirming candles post-breakout
    range_expansion_ratio: float = 0.0 # Last candle body / 5-min ATR
    vwap_slope_steepening: bool = False # Price acceleration with rising volume

    # === PULLBACK / VWAP DISTANCE FIELDS (for TrendFollowing pullback scorer) ===
    vwap_distance_pct: float = 0.0     # ((ltp - vwap) / vwap) * 100 — signed
    pullback_depth_pct: float = 0.0    # Most recent pullback depth % from swing high/low
    pullback_candles: int = 0          # Candles in last pullback

    # === BOS / SWEEP (STRUCTURE) FIELDS ===
    bos_signal: str = 'NONE'           # BOS_HIGH, BOS_LOW, NONE — price closed beyond swing level
    sweep_signal: str = 'NONE'         # SWEEP_HIGH, SWEEP_LOW, NONE — wicked beyond but closed inside
    swing_high_level: float = 0.0      # Prior swing high used for BOS/sweep detection
    swing_low_level: float = 0.0       # Prior swing low used for BOS/sweep detection


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
    # Trade logging is handled by centralized TradeLedger (trade_ledger.py)
    
    def __init__(self, paper_mode: bool = True, paper_capital: float = None):
        self.kite = KiteConnect(api_key=ZERODHA_API_KEY, timeout=15)
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
        self._positions_lock = threading.RLock()  # Thread safety for paper_positions (reentrant)
        self._scorer_rejected_symbols = set()  # Track scorer rejections for retry filtering
        self._exit_cooldowns: dict = {}  # underlying -> datetime of last exit (20-min re-entry cooldown)
        
        # Load saved token
        self._load_token()
        
        # === LIVE MODE: Sync capital from broker margins ===
        if not paper_mode and self.access_token:
            try:
                margins = self.kite.margins()
                equity_margin = margins.get('equity', {})
                live_balance = equity_margin.get('available', {}).get('live_balance', 0)
                if live_balance > 0:
                    self.paper_capital = live_balance
                    print(f"   💰 LIVE capital synced from broker: ₹{live_balance:,.0f}")
                else:
                    print(f"   ⚠️ Could not fetch live balance, using config capital: ₹{self.paper_capital:,.0f}")
            except Exception as e:
                print(f"   ⚠️ Broker margin fetch failed ({e}), using config capital: ₹{self.paper_capital:,.0f}")
        
        # === INITIALIZE KITE TICKER (WebSocket streaming) ===
        self.ticker: TitanTicker = None
        if self.access_token:
            try:
                self.ticker = get_ticker(ZERODHA_API_KEY, self.access_token, self.kite)
                self.ticker.start()
                # Subscribe to universe stocks
                from config import TIER_1_OPTIONS, TIER_2_OPTIONS
                all_symbols = TIER_1_OPTIONS + TIER_2_OPTIONS
                # Add NIFTY 50 index
                all_symbols.append("NSE:NIFTY 50")
                # Add sectoral indices for sector cross-validation
                _sector_indices = [
                    "NSE:NIFTY METAL", "NSE:NIFTY IT", "NSE:NIFTY BANK",
                    "NSE:NIFTY AUTO", "NSE:NIFTY PHARMA", "NSE:NIFTY ENERGY",
                    "NSE:NIFTY FMCG", "NSE:NIFTY REALTY", "NSE:NIFTY PSU BANK",
                    "NSE:NIFTY INFRA", "NSE:NIFTY COMMODITIES",
                ]
                all_symbols.extend(_sector_indices)
                self.ticker.subscribe_symbols(all_symbols, mode='quote')
                # Subscribe ALL near-month F&O futures for real-time OI data
                # This eliminates per-stock REST calls in _get_futures_oi_quick()
                try:
                    _nfo_instruments = self.kite.instruments("NFO")
                    self.ticker.subscribe_fo_futures(_nfo_instruments)
                    # Also subscribe ALL F&O equity symbols (for full universe scanning)
                    _fo_equities = set()
                    for inst in _nfo_instruments:
                        if inst.get('instrument_type') == 'FUT' and inst.get('segment') == 'NFO-FUT':
                            _fo_equities.add(f"NSE:{inst['name']}")
                    _extra_equities = [s for s in _fo_equities if s not in all_symbols]
                    if _extra_equities:
                        self.ticker.subscribe_symbols(_extra_equities, mode='quote')
                        # print(f"🔌 Ticker: Subscribed {len(_extra_equities)} additional F&O equities (full universe)")
                except Exception as e:
                    print(f"⚠️ Ticker: Futures/equity subscription error (non-fatal): {e}")
                
                # === ATTACH BREAKOUT WATCHER ===
                try:
                    from config import BREAKOUT_WATCHER
                    if BREAKOUT_WATCHER.get('enabled', False):
                        self.ticker.attach_breakout_watcher(BREAKOUT_WATCHER)
                except Exception as e:
                    print(f"⚠️ Breakout watcher init failed (non-fatal): {e}")
            except Exception as e:
                print(f"⚠️ Ticker init failed (REST fallback active): {e}")
                self.ticker = None
        
        # Load active trades from file
        self._load_active_trades()
        # Subscribe option contract tokens for existing positions to WebSocket
        self._subscribe_position_symbols()
    
    def _load_active_trades(self):
        """Load active trades from SQLite (falls back to JSON for migration)"""
        try:
            today = str(datetime.now().date())
            positions, pnl, cap = get_state_db().load_active_trades(today)
            if positions or pnl != 0.0:
                self.paper_positions = positions
                self.paper_pnl = pnl
                if cap > 0:
                    self.paper_capital = cap
                return
        except Exception as e:
            print(f"⚠️ Error loading trades from SQLite: {e}")
        # Fallback: legacy JSON
        try:
            if os.path.exists(self.TRADES_FILE):
                with open(self.TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    today = str(datetime.now().date())
                    if data.get('date') == today:
                        self.paper_positions = data.get('active_trades', [])
                        self.paper_pnl = data.get('realized_pnl', 0)
                    else:
                        self.paper_positions = []
                        self.paper_pnl = 0
            else:
                self.paper_positions = []
                self.paper_pnl = 0
        except Exception as e:
            print(f"⚠️ Error loading trades from JSON: {e}")
            self.paper_positions = []
            self.paper_pnl = 0
    
    def _subscribe_position_symbols(self):
        """Subscribe all option contract tokens from active positions to WebSocket.
        This enables real-time LTP streaming for PnL dashboard instead of REST fallback."""
        if not self.ticker:
            return
        option_syms = set()
        for t in self.paper_positions:
            if t.get('status', 'OPEN') != 'OPEN':
                continue
            if t.get('is_iron_condor'):
                for k in ('sold_ce_symbol', 'hedge_ce_symbol', 'sold_pe_symbol', 'hedge_pe_symbol'):
                    s = t.get(k, '')
                    if s:
                        option_syms.add(s)
            elif t.get('is_credit_spread'):
                for k in ('sold_symbol', 'hedge_symbol'):
                    s = t.get(k, '')
                    if s:
                        option_syms.add(s)
            elif t.get('is_debit_spread'):
                for k in ('buy_symbol', 'sell_symbol'):
                    s = t.get(k, '')
                    if s:
                        option_syms.add(s)
            elif t.get('is_option'):
                s = t.get('symbol', '')
                if s and ':' in s:
                    option_syms.add(s)
        if option_syms:
            self.ticker.subscribe_symbols(list(option_syms), mode='quote')
            # print(f"🔌 Ticker: Subscribed {len(option_syms)} option contracts for live PnL")

    def _save_active_trades(self):
        """Save active trades to SQLite (caller must hold _positions_lock)"""
        try:
            get_state_db().save_active_trades(
                self.paper_positions,
                self.paper_pnl,
                self.paper_capital,
            )
        except Exception as e:
            print(f"⚠️ Error saving trades: {e}")
    
    # =================================================================
    # AUTOSLICE — Freeze Quantity Protection
    # =================================================================
    
    def _place_order_autoslice(self, **kwargs):
        """
        Place order with autoslice=true to auto-split orders exceeding exchange freeze limits.
        Without autoslice, orders >1800 qty NIFTY (or similar) get REJECTED.
        
        Uses route monkey-patching since the kiteconnect Python SDK doesn't expose
        autoslice as a keyword argument (it's a URL query parameter).
        """
        if not AUTOSLICE_ENABLED or not hasattr(self.kite, '_routes'):
            return self.kite.place_order(**kwargs)
        
        orig_route = self.kite._routes.get("orders.place", "/orders/{variety}")
        try:
            if "autoslice" not in orig_route:
                self.kite._routes["orders.place"] = "/orders/{variety}?autoslice=true"
            return self.kite.place_order(**kwargs)
        finally:
            self.kite._routes["orders.place"] = orig_route
    
    # =================================================================
    # GTT SAFETY NET — Server-Side SL + Target (survives crashes)
    # =================================================================
    
    GTT_LOG_FILE = os.path.join(os.path.dirname(__file__), 'gtt_safety_log.json')
    
    def _place_gtt_safety_net(self, symbol: str, side: str, quantity: int,
                               entry_price: float, sl_price: float, target_price: float,
                               product: str = "MIS", tag: str = "TITAN") -> Optional[int]:
        """
        Place a GTT TWO-LEG (OCO) order as a crash-protection safety net.
        
        The GTT lives on Zerodha's servers. If Titan crashes:
        - SL trigger fires → exits at SL (server-side)
        - Target trigger fires → exits at target (server-side)
        - Whichever fires first cancels the other (OCO)
        
        Args:
            symbol: "NSE:SBIN" or "NFO:SBIN26FEB180CE"
            side: Original trade side ("BUY" or "SELL")
            quantity: Trade quantity
            entry_price: Entry price (used as last_price for GTT)
            sl_price: Stop loss price
            target_price: Target price
            product: "MIS" or "NRML"
            tag: Trade tag for identification
            
        Returns:
            GTT trigger_id if placed successfully, None otherwise
        """
        if self.paper_mode or not GTT_CONFIG.get('enabled', False):
            return None
        
        try:
            exchange, tradingsymbol = symbol.split(":")
            
            # Determine exit direction (opposite of entry)
            exit_type = self.kite.TRANSACTION_TYPE_SELL if side == "BUY" else self.kite.TRANSACTION_TYPE_BUY
            
            # Apply buffers to avoid double-triggering with primary SL-M
            sl_buffer = GTT_CONFIG.get('sl_buffer_pct', 1.0) / 100
            target_buffer = GTT_CONFIG.get('target_buffer_pct', 0.5) / 100
            limit_buffer = GTT_CONFIG.get('limit_price_buffer_pct', 2.0) / 100
            
            if side == "BUY":
                # BUY trade: SL is below entry, target is above entry
                gtt_sl_trigger = round(sl_price * (1 - sl_buffer), 2)      # Wider SL than primary
                gtt_target_trigger = round(target_price * (1 - target_buffer), 2)  # Slightly tighter target
                # LIMIT prices with buffer to ensure fill in gaps
                sl_limit_price = round(gtt_sl_trigger * (1 - limit_buffer), 2)     # Sell below trigger
                target_limit_price = round(gtt_target_trigger * (1 - limit_buffer), 2)
            else:
                # SELL trade: SL is above entry, target is below entry
                gtt_sl_trigger = round(sl_price * (1 + sl_buffer), 2)
                gtt_target_trigger = round(target_price * (1 + target_buffer), 2)
                sl_limit_price = round(gtt_sl_trigger * (1 + limit_buffer), 2)     # Buy above trigger
                target_limit_price = round(gtt_target_trigger * (1 + limit_buffer), 2)
            
            # Ensure trigger values are in correct order for TWO-LEG
            # First trigger = lower value, Second trigger = upper value
            if gtt_sl_trigger < gtt_target_trigger:
                trigger_values = [gtt_sl_trigger, gtt_target_trigger]
                orders = [
                    {
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "transaction_type": exit_type,
                        "quantity": quantity,
                        "order_type": self.kite.ORDER_TYPE_LIMIT,
                        "product": product,
                        "price": sl_limit_price
                    },
                    {
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "transaction_type": exit_type,
                        "quantity": quantity,
                        "order_type": self.kite.ORDER_TYPE_LIMIT,
                        "product": product,
                        "price": target_limit_price
                    }
                ]
            else:
                trigger_values = [gtt_target_trigger, gtt_sl_trigger]
                orders = [
                    {
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "transaction_type": exit_type,
                        "quantity": quantity,
                        "order_type": self.kite.ORDER_TYPE_LIMIT,
                        "product": product,
                        "price": target_limit_price
                    },
                    {
                        "exchange": exchange,
                        "tradingsymbol": tradingsymbol,
                        "transaction_type": exit_type,
                        "quantity": quantity,
                        "order_type": self.kite.ORDER_TYPE_LIMIT,
                        "product": product,
                        "price": sl_limit_price
                    }
                ]
            
            # Place GTT TWO-LEG (OCO)
            result = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_OCO,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                trigger_values=trigger_values,
                last_price=entry_price,
                orders=orders
            )
            
            trigger_id = result.get('trigger_id')
            
            if GTT_CONFIG.get('log_gtt_events', True):
                self._log_gtt_event('PLACED', symbol, trigger_id, {
                    'side': side, 'quantity': quantity,
                    'entry': entry_price, 'sl': sl_price, 'target': target_price,
                    'gtt_sl_trigger': gtt_sl_trigger, 'gtt_target_trigger': gtt_target_trigger,
                    'tag': tag
                })
                # print(f"   🛡️ GTT safety net placed: {symbol} trigger_id={trigger_id} "
                #       f"(SL@{gtt_sl_trigger} / Target@{gtt_target_trigger})")
            
            return trigger_id
            
        except Exception as e:
            print(f"   ⚠️ GTT safety net failed for {symbol}: {e}")
            if GTT_CONFIG.get('log_gtt_events', True):
                self._log_gtt_event('FAILED', symbol, None, {'error': str(e), 'tag': tag})
            return None
    
    def _cancel_gtt(self, trigger_id: int, symbol: str = "") -> bool:
        """
        Cancel a GTT order (called when trade exits normally).
        
        Args:
            trigger_id: The GTT trigger ID to cancel
            symbol: Symbol for logging (optional)
            
        Returns:
            True if cancelled successfully
        """
        if self.paper_mode or not trigger_id:
            return False
        
        try:
            self.kite.delete_gtt(trigger_id)
            if GTT_CONFIG.get('log_gtt_events', True):
                self._log_gtt_event('CANCELLED', symbol, trigger_id, {'reason': 'trade_exited_normally'})
                # print(f"   🛡️ GTT cancelled: {symbol} trigger_id={trigger_id}")
            return True
        except Exception as e:
            # GTT may have already triggered or expired
            print(f"   ⚠️ GTT cancel failed for {symbol} (trigger_id={trigger_id}): {e}")
            self._log_gtt_event('CANCEL_FAILED', symbol, trigger_id, {'error': str(e)})
            return False
    
    def _cleanup_orphaned_gtts(self):
        """
        On startup, check for GTTs that are still active but the trade was already closed.
        This handles the case where Titan crashed and the GTT persisted.
        """
        if self.paper_mode or not GTT_CONFIG.get('cleanup_on_startup', True):
            return
        
        try:
            gtts = self.kite.get_gtts()
            active_symbols = set()
            for trade in self.paper_positions:
                if trade.get('status', 'OPEN') == 'OPEN':
                    active_symbols.add(trade.get('symbol', ''))
            
            orphaned = 0
            for gtt in gtts:
                if gtt.get('status') != 'active':
                    continue
                # Check if this GTT belongs to Titan (check orders for our tradingsymbols)
                gtt_orders = gtt.get('orders', [])
                if not gtt_orders:
                    continue
                gtt_symbol = f"{gtt_orders[0].get('exchange', '')}:{gtt_orders[0].get('tradingsymbol', '')}"
                
                # If no active trade for this symbol, the GTT is orphaned
                if gtt_symbol not in active_symbols:
                    try:
                        self.kite.delete_gtt(gtt['id'])
                        orphaned += 1
                        self._log_gtt_event('ORPHAN_CLEANUP', gtt_symbol, gtt['id'], 
                                          {'gtt_status': gtt.get('status')})
                    except Exception as e:
                        print(f"   ⚠️ Failed to cleanup orphaned GTT {gtt['id']}: {e}")
            
            if orphaned > 0:
                # print(f"   🛡️ Cleaned up {orphaned} orphaned GTT(s) from previous session")
                pass
        except Exception as e:
            print(f"   ⚠️ GTT orphan cleanup failed: {e}")
    
    def _log_gtt_event(self, event: str, symbol: str, trigger_id: Optional[int], details: Dict):
        """Log GTT events to gtt_safety_log.json"""
        try:
            log = []
            if os.path.exists(self.GTT_LOG_FILE):
                with open(self.GTT_LOG_FILE, 'r') as f:
                    log = json.load(f)
            
            log.append({
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'symbol': symbol,
                'trigger_id': trigger_id,
                **details
            })
            
            # Keep last 500 events
            if len(log) > 500:
                log = log[-500:]
            
            atomic_json_save(self.GTT_LOG_FILE, log)
        except Exception:
            pass  # Never crash on logging
    
    def _log_entry_to_ledger(self, position: Dict):
        """Log a new trade entry to centralized Trade Ledger (single source of truth).
        
        Called from EVERY position append point (options, equity, spreads, condors).
        Mirrors _save_to_history which centralizes all exits.
        The position dict already has all the data we need.
        """
        try:
            from trade_ledger import get_trade_ledger
            _ml = position.get('entry_metadata', {})
            _md = position.get('xgb_model', {})
            _gm = position.get('gmm_model', {})
            
            # Determine strategy type from position flags
            if position.get('is_iron_condor'):
                _strat = 'IRON_CONDOR'
            elif position.get('is_credit_spread'):
                _strat = 'CREDIT_SPREAD'
            elif position.get('is_debit_spread'):
                _strat = 'DEBIT_SPREAD'
            elif position.get('is_option'):
                _strat = position.get('strategy_type', 'NAKED_OPTION')
            else:
                _strat = position.get('strategy_type', 'EQUITY')
            
            get_trade_ledger().log_entry(
                symbol=position.get('symbol', ''),
                underlying=position.get('underlying', position.get('symbol', '')),
                direction=position.get('direction', position.get('side', '')),
                source=position.get('setup_type', position.get('strategy_type', 'MANUAL')),
                smart_score=position.get('smart_score') or 0,
                pre_score=position.get('p_score') or position.get('entry_score', 0),
                final_score=position.get('entry_score') or position.get('p_score', 0),
                dr_score=position.get('dr_score') or 0,
                dr_flag=position.get('dr_flag', False),
                up_flag=position.get('up_flag', False),
                down_flag=position.get('down_flag', False),
                gate_prob=position.get('ml_move_prob') or 0,
                gmm_action=position.get('gmm_action', _gm.get('action', '')),
                ml_direction=position.get('ml_scored_direction', _md.get('scored_direction', '')),
                ml_move_prob=position.get('ml_move_prob') or _md.get('ml_move_prob', 0),
                ml_confidence=position.get('ml_confidence') or _md.get('confidence', ''),
                xgb_disagrees=position.get('ml_xgb_disagrees', False),
                sector=position.get('sector', ''),
                strategy_type=_strat,
                score_tier=position.get('score_tier', ''),
                is_sniper=position.get('is_sniper', False),
                lot_multiplier=position.get('lot_multiplier', 1.0),
                option_symbol=position.get('symbol', ''),
                strike=position.get('strike', 0),
                option_type=position.get('option_type', ''),
                expiry=position.get('expiry', '') or '',
                entry_price=position.get('avg_price', 0),
                quantity=position.get('quantity', 0),
                lots=position.get('lots', 0),
                stop_loss=position.get('stop_loss', 0),
                target=position.get('target', 0),
                total_premium=position.get('total_premium', 0),
                delta=position.get('delta', position.get('net_delta', 0)),
                iv=position.get('iv', 0),
                rationale=position.get('rationale', ''),
                order_id=position.get('order_id', ''),
                trade_id=position.get('trade_id', ''),
            )
        except Exception as e:
            print(f"⚠️ TradeLedger entry log error: {e}")

    def _save_to_history(self, trade: Dict, result: str, pnl: float, exit_detail: Dict = None):
        """Save completed trade to centralized Trade Ledger (single source of truth)"""
        try:
            from trade_ledger import get_trade_ledger
            _ed = exit_detail or {}
            _entry_price = trade.get('avg_price', 0)
            _exit_price = trade.get('exit_price', 0)
            _qty = trade.get('quantity', 0)
            _pnl_pct = (pnl / (_entry_price * _qty) * 100) if _entry_price > 0 and _qty > 0 else 0
            _hold_mins = 0
            try:
                _et = trade.get('timestamp', '')
                if _et:
                    _hold_mins = int((datetime.now() - datetime.fromisoformat(_et)).total_seconds() / 60)
            except Exception:
                pass
            get_trade_ledger().log_exit(
                symbol=trade.get('symbol', ''),
                underlying=trade.get('underlying', trade.get('symbol', '')),
                direction=trade.get('direction', trade.get('side', '')),
                source=trade.get('setup_type', trade.get('strategy_type', '')),
                sector=trade.get('sector', ''),
                exit_type=result,
                entry_price=_entry_price,
                exit_price=_exit_price,
                quantity=_qty,
                pnl=pnl,
                pnl_pct=_pnl_pct,
                smart_score=trade.get('smart_score', trade.get('entry_metadata', {}).get('smart_score', 0)),
                final_score=trade.get('entry_score', 0),
                dr_score=trade.get('dr_score', trade.get('entry_metadata', {}).get('dr_score', 0)),
                score_tier=trade.get('score_tier', ''),
                strategy_type=trade.get('strategy_type', ''),
                is_sniper=trade.get('is_sniper', False),
                candles_held=_ed.get('candles_held', 0),
                r_multiple=_ed.get('r_multiple_achieved', 0),
                max_favorable=_ed.get('max_favorable_excursion', 0),
                exit_reason=_ed.get('exit_reason', result),
                breakeven_applied=_ed.get('breakeven_applied', False),
                trailing_active=_ed.get('trailing_active', False),
                partial_booked=_ed.get('partial_booked', False),
                current_sl=_ed.get('current_sl_at_exit', 0),
                hold_minutes=_hold_mins,
                order_id=trade.get('order_id', ''),
                trade_id=trade.get('trade_id', ''),
                entry_time=trade.get('timestamp', ''),
                # TIE: pass thesis invalidation metadata via extra dict
                extra={k: v for k, v in _ed.items() if k in (
                    'thesis_check', 'thesis_reason', 'underlying_ltp_at_exit'
                )} or None,
            )
        except Exception as e:
            print(f"⚠️ Error saving to trade ledger: {e}")
    
    def is_symbol_in_active_trades(self, symbol: str) -> bool:
        """Check if symbol already has an active trade"""
        with self._positions_lock:
            for trade in self.paper_positions:
                if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                    return True
        return False
    
    def get_active_trade(self, symbol: str) -> Optional[Dict]:
        """Get active trade for a symbol if exists"""
        with self._positions_lock:
            for trade in self.paper_positions:
                if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                    return trade
        return None
    
    def _execute_live_exit(self, trade: Dict, exit_qty: int = None):
        """Place real exit order(s) on Zerodha to close a live position.
        
        Handles single legs (options/equity) AND multi-leg spreads/condors.
        Also cancels pending SL-M orders placed at entry.
        Called by update_trade_status and partial_exit_trade in LIVE mode.
        
        Args:
            trade: The trade dict with symbol, quantity, side, order_id, sl_order_id
            exit_qty: Quantity to exit (None = full quantity)
        """
        if self.paper_mode:
            return  # Paper mode doesn't need real orders
        
        symbol = trade.get('symbol', '')
        qty = exit_qty or trade.get('quantity', 0)
        side = trade.get('side', 'BUY')
        
        if qty <= 0:
            print(f"   ⚠️ Live exit skipped for {symbol}: qty=0")
            return
        
        # === MULTI-LEG SPREAD/CONDOR EXIT ===
        if '|' in symbol:
            self._execute_live_spread_exit(trade, exit_qty)
            return
        
        try:
            exchange, tradingsymbol = symbol.split(':')
            
            # 1. Cancel pending SL-M order (placed at entry)
            sl_order_id = trade.get('sl_order_id')
            if sl_order_id and not str(sl_order_id).startswith('PAPER_'):
                try:
                    self.kite.cancel_order(
                        variety=self.kite.VARIETY_REGULAR,
                        order_id=sl_order_id
                    )
                    print(f"   🛑 Cancelled SL order {sl_order_id} for {symbol}")
                except Exception as e:
                    print(f"   ⚠️ SL cancel failed for {symbol} (may have triggered): {e}")
            
            # 2. Place market exit order (opposite side)
            exit_side = self.kite.TRANSACTION_TYPE_SELL if side == 'BUY' else self.kite.TRANSACTION_TYPE_BUY
            
            exit_order_id = self._place_order_autoslice(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=exit_side,
                quantity=qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY,
                market_protection=-1,
                tag='TITAN_EXIT'
            )
            
            print(f"   ✅ Live EXIT order placed: {exit_side} {qty} {symbol} (order_id: {exit_order_id})")
            
        except Exception as e:
            # CRITICAL FAILURE: Position still open at broker!
            print(f"   🚨🚨 CRITICAL: Live exit order FAILED for {symbol}: {e}")
            print(f"   🚨 MANUAL ACTION REQUIRED: Close {qty} {symbol} on Kite app/web!")
            try:
                import json as _json
                critical_log = os.path.join(os.path.dirname(__file__), 'critical_failures.json')
                failures = []
                if os.path.exists(critical_log):
                    with open(critical_log, 'r') as f:
                        failures = _json.load(f)
                failures.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'LIVE_EXIT_FAILED',
                    'symbol': symbol,
                    'quantity': qty,
                    'side': side,
                    'error': str(e)
                })
                with open(critical_log, 'w') as f:
                    _json.dump(failures, f, indent=2)
            except Exception:
                pass

    def _execute_live_spread_exit(self, trade: Dict, exit_qty: int = None):
        """Close all legs of a credit spread or iron condor on Zerodha.
        
        Credit spread: BUY back sold option + SELL the hedge option.
        Iron condor: BUY back both sold options + SELL both hedge options.
        """
        qty = exit_qty or trade.get('quantity', 0)
        
        legs = []
        
        if trade.get('is_iron_condor'):
            # 4 legs: BUY back sold CE/PE, SELL the hedge CE/PE
            for prefix, action in [('sold_ce', 'BUY'), ('sold_pe', 'BUY'), 
                                     ('hedge_ce', 'SELL'), ('hedge_pe', 'SELL')]:
                sym = trade.get(f'{prefix}_symbol')
                if sym:
                    legs.append((sym, action))
        elif trade.get('is_credit_spread'):
            # 2 legs: BUY back sold option, SELL the hedge option
            sold_sym = trade.get('sold_symbol')
            hedge_sym = trade.get('hedge_symbol')
            if sold_sym:
                legs.append((sold_sym, 'BUY'))  # Buy back the short leg
            if hedge_sym:
                legs.append((hedge_sym, 'SELL'))  # Sell the hedge
        elif trade.get('is_debit_spread'):
            # 2 legs: SELL the bought option, BUY back the sold option
            # For debit spreads, we need to check which leg is which
            symbols = trade.get('symbol', '').split('|')
            if len(symbols) == 2:
                legs.append((symbols[0], 'SELL'))  # Sell bought leg
                legs.append((symbols[1], 'BUY'))   # Buy back sold leg
        
        for leg_symbol, leg_action in legs:
            try:
                exchange, tradingsymbol = leg_symbol.split(':')
                tx_type = self.kite.TRANSACTION_TYPE_BUY if leg_action == 'BUY' else self.kite.TRANSACTION_TYPE_SELL
                
                order_id = self._place_order_autoslice(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=tx_type,
                    quantity=qty,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_DAY,
                    market_protection=-1,
                    tag='TITAN_SPRD_EXIT'
                )
                print(f"   ✅ Spread leg exit: {leg_action} {qty} {leg_symbol} (order: {order_id})")
            except Exception as e:
                print(f"   🚨 CRITICAL: Spread leg exit FAILED: {leg_action} {qty} {leg_symbol}: {e}")
                print(f"   🚨 MANUAL ACTION: Close this leg on Kite app/web!")
                try:
                    import json as _json
                    critical_log = os.path.join(os.path.dirname(__file__), 'critical_failures.json')
                    failures = []
                    if os.path.exists(critical_log):
                        with open(critical_log, 'r') as f:
                            failures = _json.load(f)
                    failures.append({
                        'timestamp': datetime.now().isoformat(),
                        'event': 'LIVE_SPREAD_EXIT_FAILED',
                        'symbol': leg_symbol,
                        'action': leg_action,
                        'quantity': qty,
                        'spread_symbol': trade.get('symbol', ''),
                        'error': str(e)
                    })
                    with open(critical_log, 'w') as f:
                        _json.dump(failures, f, indent=2)
                except Exception:
                    pass

    def update_trade_status(self, symbol: str, status: str, exit_price: float = None, pnl: float = None, exit_detail: Dict = None):
        """Update trade status and move to history if closed.
        
        In LIVE mode, also places a real SELL order on Zerodha to close the position
        and cancels the pending SL order.
        
        Args:
            exit_detail: Optional dict with exit context from ExitManager:
                candles_held, r_multiple_achieved, max_favorable_excursion, exit_reason
        """
        with self._positions_lock:
            for i, trade in enumerate(self.paper_positions):
                if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                    trade['status'] = status
                    trade['exit_price'] = exit_price
                    trade['exit_time'] = datetime.now().isoformat()
                    
                    if pnl is not None:
                        trade['pnl'] = pnl
                        self.paper_pnl += pnl
                    
                    # Save to history and remove from active
                    # ROBUST: Archive ANY trade with exit_price+pnl (no whitelist gaps)
                    _known_exit_statuses = {
                        'TARGET_HIT', 'STOPLOSS_HIT', 'MANUAL_EXIT', 'EOD_EXIT',
                        'SL_HIT', 'OPTION_SPEED_GATE', 'TIME_STOP', 'TRAILING_SL',
                        'SESSION_CUTOFF', 'GREEKS_EXIT', 'EXPIRY_FORCE_EXIT', 'PARTIAL_PROFIT',
                        'IC_TARGET_HIT', 'IC_SL_HIT', 'IC_TIME_EXIT', 'IC_BREAKOUT_EXIT', 'IC_EOD_EXIT',
                        'SPREAD_TRAIL_SL', 'THETA_DECAY_WARNING',
                        'DEBIT_SPREAD_SL', 'DEBIT_SPREAD_TARGET', 'DEBIT_SPREAD_TIME_EXIT',
                        'DEBIT_SPREAD_TRAIL_SL', 'DEBIT_SPREAD_MAX_PROFIT',
                        'SNIPE_TRAILING_SL', 'SNIPE_TIME_GUARD', 'CONVICTION_REVERSAL',
                        # Thesis Invalidation Engine (TIE) exit types
                        'THESIS_INVALID_R_COLLAPSE', 'THESIS_INVALID_NEVER_SHOWED_LIFE',
                        'THESIS_INVALID_IV_CRUSH', 'THESIS_INVALID_UNDERLYING_BOS',
                        'THESIS_INVALID_MAX_PAIN_CEILING',
                        # Portfolio-level kill-all profit booking
                        'PROFIT_TARGET_EXIT',
                    }
                    # Catch-all: if exit_price and pnl are set, treat as closed even for unknown statuses
                    is_closed = status in _known_exit_statuses or (exit_price is not None and pnl is not None)
                    if is_closed:
                        # === RECORD RE-ENTRY COOLDOWN (20-min block on same underlying) ===
                        _cooldown_underlying = trade.get('underlying', '')
                        if _cooldown_underlying:
                            from config import HARD_RULES as _HR
                            _cd_mins = _HR.get('REENTRY_COOLDOWN_MINUTES', 20)
                            self._exit_cooldowns[_cooldown_underlying] = datetime.now()
                            # print(f"   ⏳ Cooldown: {_cooldown_underlying} blocked for {_cd_mins} min re-entry")
                        
                        # === LIVE MODE: Place real exit order on Zerodha ===
                        if not self.paper_mode and status != 'STOPLOSS_HIT' and status != 'SL_HIT':
                            # Don't place exit for SL_HIT — the SL-M order already triggered at broker
                            self._execute_live_exit(trade)
                        
                        self._save_to_history(trade, status, pnl or 0, exit_detail=exit_detail)
                        if status != 'PARTIAL_PROFIT':
                            # === CANCEL GTT SAFETY NET ON TRADE EXIT ===
                            gtt_id = trade.get('gtt_trigger_id')
                            if gtt_id and GTT_CONFIG.get('cleanup_on_exit', True):
                                self._cancel_gtt(gtt_id, symbol)
                            self.paper_positions.pop(i)
                    
                    self._save_active_trades()
                    return True
        return False
    
    def partial_exit_trade(self, symbol: str, exit_qty: int, exit_price: float, partial_pnl: float):
        """Partially exit a trade — reduce quantity and record partial P&L.
        
        In LIVE mode, places a real partial sell order and updates the broker SL order
        to reflect the remaining quantity.
        """
        with self._positions_lock:
            for trade in self.paper_positions:
                if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                    original_qty = trade['quantity']
                    remaining_qty = original_qty - exit_qty
                    
                    # === LIVE MODE: Place real partial exit order ===
                    if not self.paper_mode:
                        self._execute_live_exit(trade, exit_qty=exit_qty)
                        # Update SL order to remaining quantity
                        sl_order_id = trade.get('sl_order_id')
                        if sl_order_id and not str(sl_order_id).startswith('PAPER_'):
                            try:
                                self.kite.modify_order(
                                    variety=self.kite.VARIETY_REGULAR,
                                    order_id=sl_order_id,
                                    quantity=remaining_qty
                                )
                                # print(f"   📊 Updated SL order qty to {remaining_qty} for {symbol}")
                            except Exception as e:
                                print(f"   ⚠️ Failed to modify SL order for {symbol}: {e}")
                    
                    # Save partial exit to history
                    partial_record = dict(trade)
                    partial_record['quantity'] = exit_qty
                    partial_record['exit_price'] = exit_price
                    partial_record['exit_time'] = datetime.now().isoformat()
                    partial_record['pnl'] = partial_pnl
                    partial_record['status'] = 'PARTIAL_PROFIT'
                    partial_record['partial_exit'] = True
                    partial_record['original_qty'] = original_qty
                    partial_record['remaining_qty'] = remaining_qty
                    self._save_to_history(partial_record, 'PARTIAL_PROFIT', partial_pnl)
                    
                    # Update the live position with reduced qty
                    trade['quantity'] = remaining_qty
                    trade['total_premium'] = remaining_qty * trade['avg_price']
                    
                    # Track realized P&L
                    self.paper_pnl += partial_pnl
                    self._save_active_trades()
                    
                    # print(f"   📊 Partial exit saved: {exit_qty} exited, {remaining_qty} remaining")
                    return True
        return False
    
    def check_and_update_trades(self) -> List[Dict]:
        """Check all active trades for target/stoploss hits and update"""
        updates = []
        
        with self._positions_lock:
            if not self.paper_positions:
                return updates
            
            # Get current prices for all active symbols
            symbols = [t['symbol'] for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not symbols:
            return updates
        
        try:
            # Try WebSocket cache first (zero API calls), fallback to REST
            if self.ticker and self.ticker.connected:
                quotes = {}
                ltp_batch = self.ticker.get_ltp_batch(symbols)
                for sym, ltp in ltp_batch.items():
                    quotes[sym] = {'last_price': ltp}
            else:
                quotes = self.kite.ltp(symbols)
            
            with self._positions_lock:
                for trade in self.paper_positions[:]:  # Copy list to avoid modification during iteration
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    
                    # Skip iron condors — they have dedicated monitoring in autonomous_trader
                    if trade.get('is_iron_condor', False):
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
                    
                    # Calculate P&L (with 0.6% brokerage on total turnover)
                    brokerage = calc_brokerage(entry, ltp, qty)
                    if side == 'BUY':
                        pnl = (ltp - entry) * qty - brokerage
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
                        pnl = (entry - ltp) * qty - brokerage
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
        """Load access token from .env (ZERODHA_ACCESS_TOKEN)"""
        env_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "")
        if env_token:
            self.access_token = env_token
            self.kite.set_access_token(self.access_token)
            try:
                self.kite.profile()
                return True
            except Exception:
                print(f"⚠️ .env ZERODHA_ACCESS_TOKEN is invalid/expired")
                self.access_token = None
        
        # No valid token — check if we can prompt interactively
        import sys
        if not sys.stdin.isatty():
            # Headless mode (systemd, Docker, CI) — cannot prompt for input
            print(f"\n⚠️ No valid Kite access token. Running HEADLESS — skipping interactive auth.")
            print(f"   Set ZERODHA_ACCESS_TOKEN in .env or run auth manually.")
            return False
        
        print(f"\n⚠️ No valid Kite access token found.")
        print(f"   Update ZERODHA_ACCESS_TOKEN in .env, or start interactive auth...\n")
        return self.authenticate()
    
    def authenticate(self):
        """Authenticate with Zerodha - run once daily"""
        import sys
        if not sys.stdin.isatty():
            print("\n⚠️ Cannot authenticate interactively in headless mode.")
            print("   Set ZERODHA_ACCESS_TOKEN in .env and restart.")
            return False
        
        print("\n🔐 Zerodha Authentication")
        print("="*50)
        
        login_url = self.kite.login_url()
        print(f"\n1. Open this URL in your browser:")
        print(f"   {login_url}")
        
        print(f"\n2. Login with your Zerodha credentials")
        print(f"\n3. After login, copy the 'request_token' from the redirect URL")
        print(f"   (It's in the URL like: ?request_token=XXXXXX)")
        
        try:
            request_token = input("\n4. Paste request_token here: ").strip()
        except (EOFError, OSError):
            print("\n⚠️ Cannot read input (headless/pipe mode). Set ZERODHA_ACCESS_TOKEN in .env.")
            return False
        
        try:
            data = self.kite.generate_session(request_token, api_secret=ZERODHA_API_SECRET)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save token to .env
            self._update_env_token(self.access_token)
            
            print(f"\n✅ Authentication successful!")
            print(f"   Token saved to .env for: {datetime.now().date()}")
            
            # Test the connection
            profile = self.kite.profile()
            print(f"   Logged in as: {profile.get('user_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            return False
    
    def _update_env_token(self, access_token):
        """Update ZERODHA_ACCESS_TOKEN in .env file"""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            updated = False
            new_lines = []
            for line in lines:
                if line.startswith('ZERODHA_ACCESS_TOKEN='):
                    new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
                    updated = True
                else:
                    new_lines.append(line)
            
            if not updated:
                new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
            
            with open(env_path, 'w') as f:
                f.writelines(new_lines)
        
        # Also update current process environment
        os.environ['ZERODHA_ACCESS_TOKEN'] = access_token
    
    def _rate_limit(self):
        """Enforce API rate limiting using a token-bucket approach (thread-safe).
        
        Kite allows ~3 requests/second. Instead of serialising all threads through
        a single 350ms sleep, we keep a deque of recent call timestamps and only
        sleep when the bucket is full (3 calls in the last 1.05s window).
        This lets multiple threads fire concurrently up to the rate cap.
        """
        if not hasattr(self, '_rate_lock'):
            self._rate_lock = threading.Lock()
        if not hasattr(self, '_api_call_times'):
            from collections import deque
            self._api_call_times = deque()  # timestamps of recent API calls
        
        MAX_CALLS_PER_WINDOW = 3
        WINDOW_SEC = 1.05  # slightly above 1s for safety margin
        
        while True:
            with self._rate_lock:
                now = time.time()
                # Purge timestamps older than the window
                while self._api_call_times and (now - self._api_call_times[0]) > WINDOW_SEC:
                    self._api_call_times.popleft()
                
                if len(self._api_call_times) < MAX_CALLS_PER_WINDOW:
                    # Bucket has room — register this call and proceed
                    self._api_call_times.append(now)
                    self.last_api_call = now
                    self.api_call_count += 1
                    return  # Done — thread can fire its API call
                else:
                    # Bucket full — compute how long until a slot opens
                    sleep_time = WINDOW_SEC - (now - self._api_call_times[0])
            
            # Sleep OUTSIDE the lock so other threads aren't blocked
            if sleep_time > 0:
                time.sleep(sleep_time)
    
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
        """Quick futures OI check — WebSocket cache first, REST fallback.
        
        With ticker subscribed to ALL ~200 near-month futures, this is now
        a zero-API-call cache read for any F&O stock. No hardcoded list needed.
        """
        try:
            if ":" not in symbol:
                symbol = f"NSE:{symbol}"
            _, stock = symbol.split(":")

            # ---- TIER 1: Read from WebSocket cache (0 API calls, ~0ms) ----
            if self.ticker and hasattr(self.ticker, '_futures_tokens'):
                oi_data = self.ticker.get_futures_oi(symbol)
                if oi_data and oi_data.get('oi', 0) > 0:
                    # Get equity price change from ticker cache too
                    eq_ltp = self.ticker.get_ltp(symbol)
                    eq_quote = self.ticker.get_quote(symbol)
                    if eq_quote:
                        prev_close = eq_quote.get('ohlc', {}).get('close', eq_ltp or 0)
                        change_pct = ((eq_ltp - prev_close) / prev_close * 100) if prev_close and eq_ltp else 0
                    elif eq_ltp:
                        change_pct = 0  # Have LTP but no prev_close from cache
                    else:
                        change_pct = 0

                    oi = oi_data['oi']
                    oi_day_low = oi_data.get('oi_day_low', oi)
                    oi_change_pct = ((oi - oi_day_low) / oi_day_low * 100) if oi_day_low else 0

                    # Classify OI signal
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
                        "futures_symbol": oi_data.get('futures_symbol', ''),
                        "oi": oi,
                        "oi_change_pct": round(oi_change_pct, 2),
                        "price_change_pct": round(change_pct, 2),
                        "oi_signal": oi_signal
                    }

            # ---- TIER 2: REST fallback (2 API calls) ----
            # Get instruments to find futures (CACHED — NFO instruments don't change intraday)
            if not hasattr(self, '_nfo_instruments_cache'):
                self._nfo_instruments_cache = None
                self._nfo_instruments_ts = 0
            if self._nfo_instruments_cache is None or (time.time() - self._nfo_instruments_ts) > 1800:
                self._nfo_instruments_cache = self.kite.instruments("NFO")
                self._nfo_instruments_ts = time.time()
            instruments = self._nfo_instruments_cache

            # Find current month futures — dynamic, works for ANY F&O stock
            futures = [i for i in instruments if
                      i['name'] == stock and
                      i['instrument_type'] == 'FUT' and
                      i['segment'] == 'NFO-FUT']

            if not futures:
                return {"has_futures": False}

            futures.sort(key=lambda x: x['expiry'])
            current_fut = futures[0]
            fut_symbol = f"NFO:{current_fut['tradingsymbol']}"

            self._rate_limit()
            quote = self.kite.quote([symbol])
            if symbol not in quote:
                return {"has_futures": False}
            q = quote[symbol]
            ltp = q.get('last_price', 0)
            prev_close = q.get('ohlc', {}).get('close', ltp)
            change_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0

            self._rate_limit()
            fut_quote = self.kite.quote([fut_symbol])
            if fut_symbol not in fut_quote:
                return {"has_futures": False}

            fq = fut_quote[fut_symbol]
            oi = fq.get('oi', 0)
            oi_day_high = fq.get('oi_day_high', oi)
            oi_day_low = fq.get('oi_day_low', oi)

            oi_change_pct = ((oi - oi_day_low) / oi_day_low * 100) if oi_day_low else 0

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

    def get_market_data(self, symbols: List[str], force_fresh: bool = False) -> Dict[str, Dict]:
        """
        Tool: Get market data for symbols
        Returns OHLCV + technical indicators
        
        Args:
            force_fresh: If True, bypass indicator cache entirely (used by breakout watcher
                         for 1-3 stocks where fresh RSI/VWAP/ADX is critical).
        """
        # Filter to approved universe + any extra symbols passed by caller (wild-cards)
        valid_symbols = [s for s in symbols if s in APPROVED_UNIVERSE or s.startswith("NSE:")]
        if not valid_symbols:
            return {"error": "No valid symbols"}
        
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
                
                # indicators fetched in parallel below — just store quote for now
                result[symbol] = {'_quote': q, '_ohlc': ohlc, '_is_stale': is_stale}
            
            # === PARALLEL INDICATOR CALCULATION ===
            # Each _calculate_indicators makes 2 API calls (~1.5s/stock).
            # With 8 workers we overlap network latency for full F&O universe scan.
            _symbols_to_calc = list(result.keys())
            _indicators_map = {}
            _t0 = time.time()
            
            def _calc_one(sym):
                try:
                    return sym, self._calculate_indicators(sym, force_fresh=force_fresh)
                except Exception as e:
                    # print(f"   ⚠️ Indicator error {sym}: {e}")
                    return sym, None
            
            from config import FULL_FNO_SCAN
            _nworkers = FULL_FNO_SCAN.get('indicator_threads', 8)
            with ThreadPoolExecutor(max_workers=_nworkers) as executor:
                futures = {executor.submit(_calc_one, s): s for s in _symbols_to_calc}
                for future in as_completed(futures):
                    sym, ind = future.result()
                    if ind:
                        _indicators_map[sym] = ind
            
            _elapsed = time.time() - _t0
            _cached = sum(1 for s in _symbols_to_calc if hasattr(self, '_ind_cache') and s in self._ind_cache and (time.time() - self._ind_cache[s][0]) < 300)
            _today_str = datetime.now().strftime('%Y-%m-%d')
            _daily_cached = sum(1 for s in _symbols_to_calc if hasattr(self, '_daily_data_cache') and s in self._daily_data_cache and self._daily_data_cache[s][0] == _today_str)
            # print(f"   ⚡ Indicators: {len(_indicators_map)}/{len(_symbols_to_calc)} stocks in {_elapsed:.1f}s ({_cached} ind-cached, {_daily_cached} daily-cached)")
            
            # === REBUILD RESULT with indicators ===
            final_result = {}
            for symbol in _symbols_to_calc:
                indicators = _indicators_map.get(symbol)
                if not indicators:
                    continue
                
                _saved = result[symbol]
                q = _saved['_quote']
                ohlc = _saved['_ohlc']
                is_stale = _saved['_is_stale']
                
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
                
                # Compute change% properly: (LTP - prev_close) / prev_close * 100
                # Kite q['change'] is ABSOLUTE rupee change, NOT percentage!
                _ltp = q.get('last_price', 0)
                _prev_close = ohlc.get('close', 0)
                _change_pct = ((_ltp - _prev_close) / _prev_close * 100) if _prev_close > 0 and _ltp > 0 else 0

                data = MarketData(
                    symbol=symbol,
                    ltp=_ltp,
                    open=ohlc.get('open', 0),
                    high=ohlc.get('high', 0),
                    low=ohlc.get('low', 0),
                    close=ohlc.get('close', 0),
                    volume=q.get('volume', 0),
                    change_pct=round(_change_pct, 2),
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
                    ema_regime=indicators.get('ema_regime', 'NORMAL'),
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
                    htf_alignment=indicators.get('htf_alignment', 'NEUTRAL'),
                    # === TREND ENGINE FIELDS ===
                    vwap_change_pct=indicators.get('vwap_change_pct', 0.0),
                    orb_hold_candles=indicators.get('orb_hold_candles', 0),
                    adx=indicators.get('adx', 20.0),
                    # === ACCELERATION FIELDS ===
                    follow_through_candles=indicators.get('follow_through_candles', 0),
                    range_expansion_ratio=indicators.get('range_expansion_ratio', 0.0),
                    vwap_slope_steepening=indicators.get('vwap_slope_steepening', False),
                    # === PULLBACK / VWAP DISTANCE ===
                    vwap_distance_pct=indicators.get('vwap_distance_pct', 0.0),
                    pullback_depth_pct=indicators.get('pullback_depth_pct', 0.0),
                    pullback_candles=indicators.get('pullback_candles', 0)
                )
                
                final_result[symbol] = asdict(data)
            
            return final_result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_indicators(self, symbol: str, force_fresh: bool = False) -> Dict:
        """Calculate technical indicators for a symbol using BOTH daily and intraday data.
        
        Results are cached for INDICATOR_CACHE_TTL seconds to avoid redundant
        API calls when the same stock appears in consecutive scan cycles.
        
        Args:
            force_fresh: If True, skip cache lookup (breakout watcher needs live indicators).
        """
        # --- Indicator cache (10-minute TTL — scans every 3-5min, so ~50% cache hit rate) ---
        INDICATOR_CACHE_TTL = 600  # seconds
        if not hasattr(self, '_ind_cache'):
            self._ind_cache: Dict[str, tuple] = {}  # symbol -> (timestamp, result)
        
        if not force_fresh:
            _cached = self._ind_cache.get(symbol)
            if _cached:
                _cache_age = time.time() - _cached[0]
                if _cache_age < INDICATOR_CACHE_TTL:
                    return _cached[1]  # Cache hit — skip API calls
        
        self._rate_limit()
        
        try:
            # Get instrument token
            exchange, tradingsymbol = symbol.split(":")
            
            # Cache instruments list to avoid repeated API calls (thread-safe)
            if not hasattr(self, '_instrument_cache'):
                self._instrument_cache = {}
            if not hasattr(self, '_inst_lock'):
                self._inst_lock = threading.Lock()
            if exchange not in self._instrument_cache:
                with self._inst_lock:
                    if exchange not in self._instrument_cache:  # double-check after lock
                        self._instrument_cache[exchange] = self.kite.instruments(exchange)
            
            # O(1) token lookup via dict (built once per exchange, thread-safe)
            if not hasattr(self, '_token_dict'):
                self._token_dict = {}
            if exchange not in self._token_dict:
                with self._inst_lock:
                    if exchange not in self._token_dict:
                        self._token_dict[exchange] = {
                            inst['tradingsymbol']: inst['instrument_token']
                            for inst in self._instrument_cache[exchange]
                        }
            
            token = self._token_dict.get(exchange, {}).get(tradingsymbol)
            
            if not token:
                return {}
            
            # Get DAILY historical data (for trend, SMA, RSI, ATR + ML features need >=50 candles)
            # Daily data is cached for the entire trading day — it doesn't change intraday.
            # This eliminates ~50% of API calls on subsequent scan cycles.
            if not hasattr(self, '_daily_data_cache'):
                self._daily_data_cache = {}  # symbol -> (date_str, data_list)
            if not hasattr(self, '_daily_data_lock'):
                self._daily_data_lock = threading.Lock()
            
            _today_str = datetime.now().strftime('%Y-%m-%d')
            _daily_hit = False
            _cached_daily = self._daily_data_cache.get(symbol)
            if _cached_daily and _cached_daily[0] == _today_str:
                data = _cached_daily[1]
                _daily_hit = True
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=120)
                
                data = None
                for attempt in range(3):
                    try:
                        self._rate_limit()
                        data = self.kite.historical_data(
                            instrument_token=token,
                            from_date=from_date,
                            to_date=to_date,
                            interval="day"
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            wait = (attempt + 1) * 2  # 2s, 4s
                            # print(f"   ⏳ {symbol} daily data retry {attempt+1}/2 (waiting {wait}s)")
                            time.sleep(wait)
                        else:
                            raise
                
                if data:
                    with self._daily_data_lock:
                        self._daily_data_cache[symbol] = (_today_str, data)
            
            if not data:
                return {}
            
            # === FETCH INTRADAY DATA for proper VWAP & ORB ===
            # Early session (before 09:45): use 3-minute candles for faster indicator maturation
            # After 09:45: use standard 5-minute candles
            from config import EARLY_SESSION
            _now_time = datetime.now()
            _early_end = _now_time.replace(hour=int(EARLY_SESSION['end_time'].split(':')[0]), 
                                           minute=int(EARLY_SESSION['end_time'].split(':')[1]), 
                                           second=0, microsecond=0)
            _is_early_session = EARLY_SESSION.get('enabled', True) and _now_time < _early_end
            
            if _is_early_session:
                _candle_interval = EARLY_SESSION.get('candle_interval', '3minute')
                _orb_candle_count = EARLY_SESSION.get('orb_candle_count', 5)  # 15 min = 5 x 3min
                _vwap_slope_lookback = EARLY_SESSION.get('vwap_slope_lookback', 10)  # 30 min = 10 x 3min
                _momentum_lookback = EARLY_SESSION.get('momentum_lookback', 5)  # 15 min = 5 x 3min
            else:
                _candle_interval = '5minute'
                _orb_candle_count = 3   # 15 min = 3 x 5min
                _vwap_slope_lookback = 6  # 30 min = 6 x 5min
                _momentum_lookback = 3   # 15 min = 3 x 5min
            
            intraday_df = None
            try:
                today = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
                intraday_data = None
                for attempt in range(3):
                    try:
                        self._rate_limit()
                        intraday_data = self.kite.historical_data(
                            instrument_token=token,
                            from_date=today,
                            to_date=datetime.now(),
                            interval=_candle_interval
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            wait = (attempt + 1) * 2
                            # print(f"   ⏳ {symbol} intraday data retry {attempt+1}/2 (waiting {wait}s)")
                            time.sleep(wait)
                        else:
                            raise
                if intraday_data and len(intraday_data) >= 2:
                    intraday_df = pd.DataFrame(intraday_data)
                    # Cache intraday candles for ML predictor (separate from indicator cache)
                    if not hasattr(self, '_candle_cache'):
                        self._candle_cache = {}
                    self._candle_cache[symbol] = intraday_df.copy()
            except Exception:
                pass  # Fall back to daily-based calculations
            
            df = pd.DataFrame(data)
            
            # Cache daily candles for ML predictor (so ML can run even when intraday < 50)
            if not hasattr(self, '_daily_cache'):
                self._daily_cache = {}
            self._daily_cache[symbol] = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
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
            
            # ADX (Average Directional Index) - trend strength
            adx = 20.0  # Default
            if len(df) >= 28:
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
                atr_smooth = tr.rolling(14).mean()
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth)
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                dx = dx.replace([float('inf'), float('-inf')], 0).fillna(0)
                adx_series = dx.rolling(14).mean()
                if len(adx_series.dropna()) > 0:
                    adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 20.0
            
            # === INTRADAY ADX OVERRIDE ===
            # Daily ADX is blind to today's price action — a stock crashing 4% intraday
            # still shows daily ADX=20 because it looks at past 14+ daily bars.
            # Fix: if today's move from prev_close exceeds N× daily ATR, override ADX
            # to reflect the real intraday trend strength.
            #
            # Tiers (conservative — only boosts, never reduces):
            #   Move ≥ 2.0× ATR → ADX = max(daily, 45)  "Very strong intraday trend"
            #   Move ≥ 1.5× ATR → ADX = max(daily, 35)  "Strong intraday trend"
            #   Move ≥ 1.0× ATR → ADX = max(daily, 28)  "Moderate intraday trend"
            #
            # Also checks intraday candle consistency: if 70%+ candles move in same
            # direction, it confirms directional conviction (not just a gap).
            _daily_adx = adx  # preserve original for logging
            _adx_overridden = False
            _prev_close_for_adx = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
            _current_price_for_adx = close.iloc[-1]
            if atr > 0 and _prev_close_for_adx > 0:
                _todays_move = abs(_current_price_for_adx - _prev_close_for_adx)
                _atr_multiple = _todays_move / atr
                
                # Check intraday directional consistency (anti-whipsaw)
                _directional_pct = 0.0
                if intraday_df is not None and len(intraday_df) >= 3:
                    _bodies = intraday_df['close'] - intraday_df['open']
                    _bullish_count = (_bodies > 0).sum()
                    _bearish_count = (_bodies < 0).sum()
                    _total = len(_bodies)
                    _directional_pct = max(_bullish_count, _bearish_count) / _total if _total > 0 else 0
                
                # Only override if move is significant AND directionally consistent
                # OR if move is very large (gap + continuation, direction inherently clear)
                _direction_confirmed = _directional_pct >= 0.6 or _atr_multiple >= 2.0
                
                if _direction_confirmed:
                    if _atr_multiple >= 2.0:
                        adx = max(adx, 45.0)
                    elif _atr_multiple >= 1.5:
                        adx = max(adx, 35.0)
                    elif _atr_multiple >= 1.0:
                        adx = max(adx, 28.0)
                    
                    if adx > _daily_adx:
                        _adx_overridden = True
            
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
            
            # Volume analysis (TIME-NORMALIZED for intraday partial-day bias)
            vol = df['volume']
            raw_today_volume = vol.iloc[-1]
            
            # Time-normalize today's volume to full-day equivalent
            _now = datetime.now()
            _market_open = _now.replace(hour=9, minute=15, second=0, microsecond=0)
            _full_day_min = 375.0  # 9:15 to 15:30
            _elapsed_min = max((_now - _market_open).total_seconds() / 60.0, 1.0)
            if _elapsed_min < _full_day_min:
                # Cap multiplier at 12.5x (= 30-min minimum extrapolation)
                # Prevents absurd projections in first few minutes
                _proj_mult = min(_full_day_min / _elapsed_min, 12.5)
                current_volume = raw_today_volume * _proj_mult
            else:
                current_volume = raw_today_volume  # After market close, no projection
            
            # Exclude today from averages (today is partial / projected)
            avg_volume_20 = vol.iloc[-21:-1].mean() if len(vol) >= 21 else vol.iloc[:-1].mean() if len(vol) >= 2 else vol.mean()
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # === REGIME DETECTION CALCULATIONS ===
            
            # 1. EMA 9 and 21 for compression/expansion
            # Use INTRADAY 5-min candle EMAs if available for better regime detection
            if intraday_df is not None and len(intraday_df) >= 21:
                i_close = intraday_df['close']
                ema_9_series = i_close.ewm(span=9, adjust=False).mean()
                ema_21_series = i_close.ewm(span=21, adjust=False).mean()
                ema_9 = ema_9_series.iloc[-1]
                ema_21 = ema_21_series.iloc[-1]
                ema_spread = abs(ema_9 - ema_21) / ema_21 * 100 if ema_21 > 0 else 0
                
                compression_threshold = 0.04  # 5-min EMAs: ≤0.04% spread = truly compressed
                recent_spreads = abs(ema_9_series.iloc[-5:] - ema_21_series.iloc[-5:]) / ema_21_series.iloc[-5:] * 100
                candles_compressed = (recent_spreads < compression_threshold).sum()
                
                if candles_compressed >= 4:
                    ema_regime = "COMPRESSED"
                elif ema_spread >= 0.08:  # Meaningful divergence on 5-min
                    ema_regime = "EXPANDING"
                else:
                    ema_regime = "NORMAL"
            else:
                # Fallback: daily EMAs
                ema_9_series = close.ewm(span=9, adjust=False).mean()
                ema_21_series = close.ewm(span=21, adjust=False).mean()
                ema_9 = ema_9_series.iloc[-1]
                ema_21 = ema_21_series.iloc[-1]
                ema_spread = abs(ema_9 - ema_21) / ema_21 * 100 if ema_21 > 0 else 0
                
                compression_threshold = 0.3  # % spread threshold for daily
                if len(ema_9_series) >= 5:
                    recent_spreads = abs(ema_9_series.iloc[-5:] - ema_21_series.iloc[-5:]) / ema_21_series.iloc[-5:] * 100
                    candles_compressed = (recent_spreads < compression_threshold).sum()
                    
                    if candles_compressed >= 4:
                        ema_regime = "COMPRESSED"
                    elif ema_spread >= 0.5:  # Meaningful divergence on daily
                        ema_regime = "EXPANDING"
                    else:
                        ema_regime = "NORMAL"
                else:
                    ema_regime = "NORMAL"
            
            # 2. VWAP calculation - USE INTRADAY DATA if available
            if intraday_df is not None and len(intraday_df) >= 2:
                # REAL intraday VWAP from today's candles
                itp = (intraday_df['high'] + intraday_df['low'] + intraday_df['close']) / 3
                ivol = intraday_df['volume']
                i_cum_tpv = (itp * ivol).cumsum()
                i_cum_vol = ivol.cumsum()
                vwap_series = i_cum_tpv / i_cum_vol
                vwap_series = vwap_series.replace([float('inf'), float('-inf')], current_price).fillna(current_price)
                vwap = vwap_series.iloc[-1]
                
                # VWAP slope over lookback window (30 min equivalent)
                if len(vwap_series) >= _vwap_slope_lookback:
                    vwap_back = vwap_series.iloc[-_vwap_slope_lookback]
                    vwap_now = vwap_series.iloc[-1]
                    vwap_change_pct = (vwap_now - vwap_back) / vwap_back * 100 if vwap_back > 0 else 0
                    
                    if vwap_change_pct > 0.15:
                        vwap_slope = "RISING"
                    elif vwap_change_pct < -0.15:
                        vwap_slope = "FALLING"
                    else:
                        vwap_slope = "FLAT"
                else:
                    vwap_slope = "FLAT"
                    vwap_change_pct = 0.0
            else:
                # Fallback: daily VWAP (less accurate for intraday)
                typical_price = (high + low + close) / 3
                cumulative_tpv = (typical_price * vol).cumsum()
                cumulative_vol = vol.cumsum()
                vwap_series_daily = cumulative_tpv / cumulative_vol
                vwap = vwap_series_daily.iloc[-1] if len(vwap_series_daily) > 0 else current_price
                
                if len(vwap_series_daily) >= 5:
                    vwap_5_ago = vwap_series_daily.iloc[-5]
                    vwap_now = vwap_series_daily.iloc[-1]
                    vwap_change_pct = (vwap_now - vwap_5_ago) / vwap_5_ago * 100 if vwap_5_ago > 0 else 0
                    vwap_slope = "RISING" if vwap_change_pct > 0.5 else "FALLING" if vwap_change_pct < -0.5 else "FLAT"
                else:
                    vwap_slope = "FLAT"
                    vwap_change_pct = 0.0
            
            # Price vs VWAP
            if current_price > vwap * 1.003:
                price_vs_vwap = "ABOVE_VWAP"
            elif current_price < vwap * 0.997:
                price_vs_vwap = "BELOW_VWAP"
            else:
                price_vs_vwap = "AT_VWAP"
            
            # 3. Opening Range Breakout - USE INTRADAY DATA if available
            if intraday_df is not None and len(intraday_df) >= _orb_candle_count:
                # TRUE ORB: First 15 minutes (dynamic: 3x5min or 5x3min)
                orb_candles = intraday_df.head(_orb_candle_count)
                orb_high = orb_candles['high'].max()
                orb_low = orb_candles['low'].min()
            elif intraday_df is not None and len(intraday_df) >= 2:
                # Partial ORB: use whatever candles we have (ORB still forming)
                orb_candles = intraday_df
                orb_high = orb_candles['high'].max()
                orb_low = orb_candles['low'].min()
            else:
                # Fallback: previous day high/low
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
            
            # ORB hold: count candles that held the breakout level
            orb_hold_candles = 0
            if intraday_df is not None and len(intraday_df) > _orb_candle_count and orb_signal != "INSIDE_ORB":
                post_orb_candles = intraday_df.iloc[_orb_candle_count:]
                if orb_signal == "BREAKOUT_UP":
                    orb_hold_candles = int((post_orb_candles['low'] >= orb_high * 0.998).sum())
                elif orb_signal == "BREAKOUT_DOWN":
                    orb_hold_candles = int((post_orb_candles['high'] <= orb_low * 1.002).sum())
            
            # 4. Volume relative to 5-day average (TIME-NORMALIZED)
            # Exclude today from avg (today is partial), use projected volume for comparison
            volume_5d_avg = vol.iloc[-6:-1].mean() if len(vol) >= 6 else vol.iloc[:-1].mean() if len(vol) >= 2 else vol.mean()
            volume_vs_avg = current_volume / volume_5d_avg if volume_5d_avg > 0 else 1.0
            
            if volume_vs_avg < 0.5:
                volume_regime = "LOW"
            elif volume_vs_avg < 1.2:
                volume_regime = "NORMAL"
            elif volume_vs_avg < 2.0:
                volume_regime = "HIGH"
            else:
                volume_regime = "EXPLOSIVE"
            
            # 4b. RECENT VOLUME RATE — last 3 candles vs session average candle volume
            # Answers: "Is volume intense RIGHT NOW or was it front-loaded earlier?"
            recent_vol_rate = 1.0  # default: same as session average
            if intraday_df is not None and len(intraday_df) >= 5:
                candle_volumes = intraday_df['volume'].values
                avg_candle_vol = candle_volumes.mean()
                recent_3_avg = candle_volumes[-3:].mean()
                recent_vol_rate = recent_3_avg / avg_candle_vol if avg_candle_vol > 0 else 1.0
            
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
            
            # CHOP Zone detection (tightened: only trigger on clear chop)
            # 1. VWAP is flat AND volume is LOW (not NORMAL - intraday vol is biased low)
            if vwap_slope == "FLAT" and volume_regime == "LOW":
                chop_zone = True
                chop_reason = "VWAP_FLAT+LOW_VOL"
            
            # 2. Range too compressed (< 0.4x ATR over 5 candles) 
            elif atr_range_ratio < 0.4:
                chop_zone = True
                chop_reason = "COMPRESSED_RANGE"
            
            # 3. ORB containment with many re-entries (whipsaw)
            elif orb_reentries >= 4 and volume_regime == "LOW":
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
            
            # === ACCELERATION / FOLLOW-THROUGH METRICS (for IntradayOptionScorer) ===
            follow_through_candles = 0
            range_expansion_ratio = 0.0
            vwap_slope_steepening = False
            intraday_atr = atr / 8.66  # Default: daily ATR / sqrt(75 candles/day)
            
            if intraday_df is not None and len(intraday_df) >= 6:
                idf = intraday_df
                
                # Calculate intraday ATR from available candles
                idf_tr = pd.concat([
                    idf['high'] - idf['low'],
                    abs(idf['high'] - idf['close'].shift(1)),
                    abs(idf['low'] - idf['close'].shift(1))
                ], axis=1).max(axis=1)
                
                if len(idf) >= 14:
                    _iatr = idf_tr.rolling(14).mean().iloc[-1]
                else:
                    # Fewer than 14 candles: use mean of available TRs (better than daily/15)
                    _iatr = idf_tr.iloc[1:].mean()  # Skip first (no shift value)
                if not pd.isna(_iatr) and _iatr > 0:
                    intraday_atr = _iatr
                
                # Follow-through: count confirming candles post-ORB (doji-tolerant)
                if orb_signal in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
                    post_orb = idf.iloc[_orb_candle_count:]  # Skip ORB period (dynamic)
                    confirming = 0
                    non_confirming = 0
                    for i in range(len(post_orb)):
                        c = post_orb.iloc[i]
                        body = c['close'] - c['open']
                        candle_range = c['high'] - c['low']
                        is_confirming = (body > 0) if orb_signal == "BREAKOUT_UP" else (body < 0)
                        is_doji = abs(body) < candle_range * 0.15 if candle_range > 0 else True
                        
                        if is_confirming:
                            confirming += 1
                        elif is_doji:
                            pass  # Dojis don't break follow-through
                        else:
                            non_confirming += 1
                            if non_confirming > 1:  # Tolerate 1 counter candle
                                break
                    follow_through_candles = confirming
                
                # Range expansion: last candle body / 5-min ATR (proper scale)
                last_candle = idf.iloc[-1]
                candle_body = abs(last_candle['close'] - last_candle['open'])
                range_expansion_ratio = candle_body / intraday_atr if intraday_atr > 0 else 0.0
                
                # VWAP slope steepening: price acceleration with rising volume
                if len(idf) >= 10:
                    idf_c = idf['close']
                    idf_v = idf['volume']
                    mid1 = idf_c.iloc[-10:-5].mean()
                    mid2 = idf_c.iloc[-5:].mean()
                    vol1 = idf_v.iloc[-10:-5].mean()
                    vol2 = idf_v.iloc[-5:].mean()
                    # Threshold: 2x 5-min ATR movement with volume acceleration
                    move_threshold = intraday_atr * 2.0
                    if vol2 > vol1 * 1.15 and abs(mid2 - mid1) > move_threshold:
                        vwap_slope_steepening = True
            
            # === VWAP DISTANCE (signed % from VWAP) ===
            vwap_distance_pct = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
            
            # === PULLBACK DETECTION (from intraday candles) ===
            pullback_depth_pct = 0.0
            pullback_candles = 0
            if intraday_df is not None and len(intraday_df) >= 6 and orb_signal != "INSIDE_ORB":
                idf_close = intraday_df['close'].values
                idf_high = intraday_df['high'].values
                idf_low = intraday_df['low'].values
                n = len(idf_close)
                
                if orb_signal == "BREAKOUT_UP":
                    # Find the highest high after ORB, then measure pullback from there
                    post_orb_highs = idf_high[_orb_candle_count:]  # skip ORB candles
                    if len(post_orb_highs) > 0:
                        swing_high_idx = int(post_orb_highs.argmax())  # relative to post_orb
                        swing_high = float(post_orb_highs[swing_high_idx])
                        # Measure pullback from swing high to current low
                        if swing_high > 0 and swing_high_idx < len(post_orb_highs) - 1:
                            # There are candles after the swing high — measure pullback
                            after_swing = idf_low[_orb_candle_count + swing_high_idx + 1:]
                            if len(after_swing) > 0:
                                lowest_after = float(after_swing.min())
                                pullback_depth_pct = (swing_high - lowest_after) / swing_high * 100
                                # Count candles pulling back (close < prev close)
                                pb_candles = 0
                                closes_after = idf_close[_orb_candle_count + swing_high_idx + 1:]
                                for k in range(len(closes_after)):
                                    prev_c = idf_close[_orb_candle_count + swing_high_idx + k] if k == 0 else closes_after[k - 1]
                                    if closes_after[k] < prev_c:
                                        pb_candles += 1
                                    else:
                                        break  # pullback ended
                                pullback_candles = pb_candles
                
                elif orb_signal == "BREAKOUT_DOWN":
                    # Find the lowest low after ORB, then measure pullback from there
                    post_orb_lows = idf_low[_orb_candle_count:]
                    if len(post_orb_lows) > 0:
                        swing_low_idx = int(post_orb_lows.argmin())
                        swing_low = float(post_orb_lows[swing_low_idx])
                        if swing_low > 0 and swing_low_idx < len(post_orb_lows) - 1:
                            after_swing = idf_high[_orb_candle_count + swing_low_idx + 1:]
                            if len(after_swing) > 0:
                                highest_after = float(after_swing.max())
                                pullback_depth_pct = (highest_after - swing_low) / swing_low * 100
                                pb_candles = 0
                                closes_after = idf_close[_orb_candle_count + swing_low_idx + 1:]
                                for k in range(len(closes_after)):
                                    prev_c = idf_close[_orb_candle_count + swing_low_idx + k] if k == 0 else closes_after[k - 1]
                                    if closes_after[k] > prev_c:
                                        pb_candles += 1
                                    else:
                                        break
                                pullback_candles = pb_candles
            
            # === MOMENTUM 15m: price change over lookback candles (15 min equivalent) ===
            momentum_15m = 0.0
            if intraday_df is not None and len(intraday_df) >= (_momentum_lookback + 1):
                close_now = float(intraday_df['close'].iloc[-1])
                close_back = float(intraday_df['close'].iloc[-(_momentum_lookback + 1)])  # lookback candles back
                if close_back > 0:
                    momentum_15m = ((close_now - close_back) / close_back) * 100
            
            # === BOS / SWEEP (STRUCTURE) DETECTION ===
            # BOS: Did price break AND close beyond the last swing high/low?
            #      → Confirms real breakout (structural break).
            # Sweep: Did price wick beyond swing level but close back inside?
            #      → Classic liquidity grab / stop-hunt / fake breakout.
            # Lookback: 10-15 candles (30-45 min on 3-min chart) for swing detection.
            # Uses a simple fractal approach: swing high = candle whose high is
            # higher than the N candles on each side (left-confirmed only, right=1).
            bos_signal = 'NONE'
            sweep_signal = 'NONE'
            _struct_swing_high = 0.0
            _struct_swing_low = 0.0
            
            if intraday_df is not None and len(intraday_df) >= 8:
                _s_highs = intraday_df['high'].values
                _s_lows = intraday_df['low'].values
                _s_closes = intraday_df['close'].values
                _s_n = len(_s_highs)
                _swing_lookback = min(15, _s_n - 2)  # 15 candles max, leave room for current
                
                # --- Find most recent SWING HIGH (fractal: high > 3 bars left & 1 bar right) ---
                # Scan backwards from second-to-last candle (last candle is current/forming)
                for _si in range(_s_n - 2, max(2, _s_n - _swing_lookback - 1), -1):
                    if (_s_highs[_si] > _s_highs[_si - 1] and 
                        _s_highs[_si] > _s_highs[_si - 2] and
                        _s_highs[_si] > _s_highs[_si - 3] if _si >= 3 else _s_highs[_si] > _s_highs[_si - 1]):
                        # Confirm it's also higher than the bar after it (right side)
                        if _si + 1 < _s_n and _s_highs[_si] > _s_highs[_si + 1]:
                            _struct_swing_high = float(_s_highs[_si])
                            break
                
                # --- Find most recent SWING LOW (fractal: low < 3 bars left & 1 bar right) ---
                for _si in range(_s_n - 2, max(2, _s_n - _swing_lookback - 1), -1):
                    if (_s_lows[_si] < _s_lows[_si - 1] and 
                        _s_lows[_si] < _s_lows[_si - 2] and
                        _s_lows[_si] < _s_lows[_si - 3] if _si >= 3 else _s_lows[_si] < _s_lows[_si - 1]):
                        if _si + 1 < _s_n and _s_lows[_si] < _s_lows[_si + 1]:
                            _struct_swing_low = float(_s_lows[_si])
                            break
                
                # --- Current candle (last bar) ---
                _cur_high = float(_s_highs[-1])
                _cur_low = float(_s_lows[-1])
                _cur_close = float(_s_closes[-1])
                
                # --- SWEEP vs BOS detection on SWING HIGH ---
                if _struct_swing_high > 0 and _cur_high > _struct_swing_high:
                    if _cur_close > _struct_swing_high:
                        # Closed above swing high → BOS (real breakout)
                        bos_signal = 'BOS_HIGH'
                    else:
                        # Wicked above but closed back below → SWEEP (liquidity grab)
                        sweep_signal = 'SWEEP_HIGH'
                
                # --- SWEEP vs BOS detection on SWING LOW ---
                if _struct_swing_low > 0 and _cur_low < _struct_swing_low:
                    if _cur_close < _struct_swing_low:
                        # Closed below swing low → BOS (real breakdown)
                        bos_signal = 'BOS_LOW'
                    else:
                        # Wicked below but closed back above → SWEEP (stop hunt)
                        sweep_signal = 'SWEEP_LOW'
            
            _ind_result = {
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
                'recent_vol_rate': round(recent_vol_rate, 2),
                # === CHOP FILTER FIELDS ===
                'chop_zone': chop_zone,
                'chop_reason': chop_reason,
                'atr_range_ratio': round(atr_range_ratio, 2),
                'orb_reentries': orb_reentries,
                # === HTF ALIGNMENT FIELDS ===
                'htf_trend': htf_trend,
                'htf_ema_slope': htf_ema_slope,
                'htf_alignment': htf_alignment,
                # === TREND SIGNAL FIELDS (for TrendFollowing engine) ===
                'vwap_change_pct': round(vwap_change_pct, 4),
                'orb_hold_candles': orb_hold_candles,
                'adx': round(adx, 1),
                'adx_daily': round(_daily_adx, 1),
                'adx_overridden': _adx_overridden,
                # === ACCELERATION FIELDS ===
                'follow_through_candles': follow_through_candles,
                'range_expansion_ratio': round(range_expansion_ratio, 2),
                'vwap_slope_steepening': vwap_slope_steepening,
                # === PULLBACK / VWAP DISTANCE ===
                'vwap_distance_pct': round(vwap_distance_pct, 3),
                'pullback_depth_pct': round(pullback_depth_pct, 3),
                'pullback_candles': pullback_candles,
                # === MOMENTUM ===
                'momentum_15m': round(momentum_15m, 4),
                # === BOS / SWEEP (STRUCTURE) ===
                'bos_signal': bos_signal,
                'sweep_signal': sweep_signal,
                'swing_high_level': round(_struct_swing_high, 2),
                'swing_low_level': round(_struct_swing_low, 2),
            }
            
            # === FUTURES OI SIGNAL (zero API calls — WebSocket cache) ===
            # Pro traders check OI flow for institutional direction confirmation
            try:
                _oi_quick = self._get_futures_oi_quick(symbol)
                if _oi_quick.get('has_futures'):
                    _ind_result['oi_signal'] = _oi_quick.get('oi_signal', 'NEUTRAL')
                    _ind_result['oi_change_pct'] = _oi_quick.get('oi_change_pct', 0)
                else:
                    _ind_result['oi_signal'] = 'NO_FUTURES'
                    _ind_result['oi_change_pct'] = 0
            except Exception:
                _ind_result['oi_signal'] = 'ERROR'
                _ind_result['oi_change_pct'] = 0
            
            # Store in cache for TTL reuse
            self._ind_cache[symbol] = (time.time(), _ind_result)
            return _ind_result
            
        except Exception as e:
            import traceback
            print(f"\n⚠️ INDICATOR CALCULATION FAILED for {symbol}: {e}")
            traceback.print_exc()
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
            # print(f"   ⚠️ Data health check failed: {e}")
            pass
        
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
            # print(f"   ⚠️ Idempotency check failed: {e} - proceeding with caution")
            pass
        
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
                # print(f"   ⚠️ Correlation: {w}")
                pass
        
        # === REGIME SCORE CHECK ===
        regime_scorer = get_regime_scorer()
        
        # Get market data for scoring
        market_data = {}
        try:
            md = self.get_market_data([symbol])
            if symbol in md and isinstance(md[symbol], dict):
                market_data = md[symbol]
        except Exception:
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
        
        # print(f"   📊 Regime Score: {regime_result.total_score}/{regime_result.threshold} ({regime_result.confidence})")
        order['_regime_score'] = regime_result.total_score
        order['_regime_confidence'] = regime_result.confidence
        
        # === INTRADAY SIGNAL SCORING FOR CASH EQUITY ===
        # Same IntradayOptionScorer used for options, but with lower thresholds
        # Prevents blindly placing cash trades without signal quality check
        from config import CASH_INTRADAY_CONFIG
        cash_scoring_enabled = CASH_INTRADAY_CONFIG.get('enabled', True)
        
        if cash_scoring_enabled and market_data:
            try:
                scorer = get_intraday_scorer()
                
                # Build IntradaySignal from market_data (same as options flow)
                intraday_signal = IntradaySignal(
                    symbol=symbol,
                    orb_signal=market_data.get('orb_signal', 'INSIDE_ORB'),
                    vwap_position=market_data.get('price_vs_vwap', market_data.get('vwap_position', 'AT_VWAP')),
                    vwap_trend=market_data.get('vwap_slope', market_data.get('vwap_trend', 'FLAT')),
                    ema_regime=market_data.get('ema_regime', 'NORMAL'),
                    volume_regime=market_data.get('volume_regime', 'NORMAL'),
                    rsi=market_data.get('rsi_14', 50.0),
                    price_momentum=market_data.get('momentum_15m', 0.0),
                    htf_alignment=market_data.get('htf_alignment', 'NEUTRAL'),
                    chop_zone=market_data.get('chop_zone', False),
                    follow_through_candles=market_data.get('follow_through_candles', 0),
                    range_expansion_ratio=market_data.get('range_expansion_ratio', 0.0),
                    vwap_slope_steepening=market_data.get('vwap_slope_steepening', False),
                    atr=market_data.get('atr_14', 0.0)
                )
                
                # Score the signal (skip microstructure for cash equity)
                cash_decision = scorer.score_intraday_signal(
                    signal=intraday_signal,
                    market_data=market_data,
                    option_data=None,  # No option microstructure for cash
                    caller_direction=side
                )
                
                cash_min_score = CASH_INTRADAY_CONFIG.get('min_score', 40)
                cash_min_conviction = CASH_INTRADAY_CONFIG.get('min_conviction_points', 5)
                
                # Extract directional conviction from decision reasons
                actual_score = cash_decision.confidence_score
                
                # Check minimum score
                if actual_score < cash_min_score:
                    self._scorer_rejected_symbols.add(symbol.replace('NSE:', ''))
                    if CASH_INTRADAY_CONFIG.get('log_rejections', True):
                        # print(f"   🚫 CASH INTRADAY BLOCK: {symbol} score {actual_score:.0f}/{cash_min_score}")
                        # for r in cash_decision.reasons[:5]:
                        #     print(f"      {r}")
                        # for w in cash_decision.warnings[:5]:
                        #     print(f"      {w}")
                        pass
                    return {
                        "success": False,
                        "error": f"CASH INTRADAY BLOCK: Score {actual_score:.0f}/{cash_min_score} - signals too weak for entry",
                        "intraday_score": actual_score,
                        "threshold": cash_min_score,
                        "direction": cash_decision.recommended_direction,
                        "reasons": cash_decision.reasons[:5],
                        "warnings": cash_decision.warnings[:5],
                        "action": "SKIP - intraday signals insufficient for cash trade"
                    }
                
                # Check direction alignment — agent said BUY but signals say SELL (or vice versa)
                if cash_decision.recommended_direction != "HOLD" and cash_decision.recommended_direction != side:
                    if CASH_INTRADAY_CONFIG.get('log_rejections', True):
                        # print(f"   🚫 CASH DIRECTION CONFLICT: Agent wants {side} but signals say {cash_decision.recommended_direction}")
                        pass
                    return {
                        "success": False,
                        "error": f"DIRECTION CONFLICT: Agent wants {side} but intraday signals say {cash_decision.recommended_direction} (score: {actual_score:.0f})",
                        "intraday_score": actual_score,
                        "agent_direction": side,
                        "signal_direction": cash_decision.recommended_direction,
                        "action": "SKIP - direction mismatch between analysis and signals"
                    }
                
                # print(f"   ✅ Cash Intraday Score: {actual_score:.0f}/{cash_min_score} ({cash_decision.recommended_direction}) — PASSED")
                order['_intraday_score'] = actual_score
                order['_intraday_direction'] = cash_decision.recommended_direction
                
            except Exception as e:
                print(f"   ⚠️ Cash intraday scoring failed: {e} — allowing trade (fallback)")
        elif cash_scoring_enabled and not market_data:
            print(f"   ⚠️ No market data for cash scoring — allowing trade (no data)")
        
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
                # print(f"   ⚠️ Execution warnings: {', '.join(exec_policy.warnings)}")
                pass
            
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
                capital=(self.paper_capital + getattr(self, 'paper_pnl', 0)) if hasattr(self, 'paper_capital') else 100000,
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
                # print(f"   📊 Adaptive sizing: {original_qty} → {sizing['quantity']}")
                # if sizing.get('warnings'):
                #     for w in sizing['warnings']:
                #         print(f"      ⚠️ {w}")
                pass
            
        except Exception as e:
            return {
                "success": False,
                "error": f"EXECUTION GUARD FAILED: {e} - order blocked for safety",
                "action": "Execution guard could not validate spread/sizing. Retry next cycle."
            }
        
        self._rate_limit()
        
        try:
            # PAPER MODE: Simulate the order
            if self.paper_mode:
                import random
                paper_order_id = f"PAPER_{random.randint(100000, 999999)}"
                
                # ALWAYS use CURRENT LTP as entry price (not what LLM passed)
                try:
                    quote = self.kite.ltp([symbol])
                    current_ltp = quote[symbol]['last_price']
                except Exception:
                    current_ltp = order.get('entry_price', 0)
                
                # Use order's computed SL/target (from ATR/regime/trend), fallback to 1%/1.5% only if missing
                side = order['side']
                if side == 'BUY':
                    stop_loss = order.get('stop_loss') or (current_ltp * 0.99)
                    target = order.get('target') or (current_ltp * 1.015)
                else:
                    stop_loss = order.get('stop_loss') or (current_ltp * 1.01)
                    target = order.get('target') or (current_ltp * 0.985)
                
                # Add to paper positions (thread-safe)
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
                with self._positions_lock:
                    self.paper_positions.append(position)
                    self._save_active_trades()
                self._log_entry_to_ledger(position)
                
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
            
            # Place main order (with autoslice for freeze qty protection)
            order_id = self._place_order_autoslice(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if order['side'] == 'BUY' else self.kite.TRANSACTION_TYPE_SELL,
                quantity=order['quantity'],
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET,
                validity=self.kite.VALIDITY_DAY,
                market_protection=-1,
                tag='TITAN_EQ'
            )
            
            # Place SL order (with autoslice)
            sl_side = self.kite.TRANSACTION_TYPE_SELL if order['side'] == 'BUY' else self.kite.TRANSACTION_TYPE_BUY
            sl_trigger = order['stop_loss'] * (0.999 if order['side'] == 'BUY' else 1.001)
            
            sl_order_id = self._place_order_autoslice(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=sl_side,
                quantity=order['quantity'],
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_SLM,
                trigger_price=sl_trigger,
                validity=self.kite.VALIDITY_DAY,
                tag='TITAN_SL'
            )
            
            # Get fill price from broker
            try:
                import time as _time
                _time.sleep(0.5)  # Wait for order to fill
                broker_orders = self.kite.orders()
                fill_price = current_ltp
                for bo in broker_orders:
                    if str(bo.get('order_id')) == str(order_id) and bo.get('status') == 'COMPLETE':
                        fill_price = bo.get('average_price', current_ltp)
                        break
            except Exception:
                fill_price = current_ltp
            
            # === STORE LIVE POSITION IN TRACKING (same structure as paper) ===
            stop_loss = order.get('stop_loss', fill_price * 0.94)
            target = order.get('target', fill_price * 1.10)
            
            live_position = {
                'symbol': symbol,
                'quantity': order['quantity'],
                'avg_price': fill_price,
                'side': order['side'],
                'stop_loss': stop_loss,
                'target': target,
                'order_id': str(order_id),
                'sl_order_id': str(sl_order_id),
                'timestamp': datetime.now().isoformat(),
                'status': 'OPEN',
                'rationale': order.get('rationale', ''),
                'volume_regime': order.get('volume_regime', 'NORMAL'),
                'expected_entry': order.get('expected_entry', fill_price),
                'is_live': True,
            }
            with self._positions_lock:
                self.paper_positions.append(live_position)
                self._save_active_trades()
            self._log_entry_to_ledger(live_position)
            
            # === GTT SAFETY NET (server-side backup SL + target) ===
            gtt_trigger_id = None
            if GTT_CONFIG.get('equity_gtt', True):
                gtt_trigger_id = self._place_gtt_safety_net(
                    symbol=symbol,
                    side=order['side'],
                    quantity=order['quantity'],
                    entry_price=fill_price,
                    sl_price=stop_loss,
                    target_price=target,
                    product="MIS",
                    tag='TITAN_EQ'
                )
                # Store GTT ID in the trade record for cleanup on exit
                if gtt_trigger_id:
                    with self._positions_lock:
                        for trade in self.paper_positions:
                            if trade.get('order_id') == str(order_id) or trade.get('symbol') == symbol:
                                trade['gtt_trigger_id'] = gtt_trigger_id
                                break
                        self._save_active_trades()
            
            # === RECORD ORDER IN IDEMPOTENT ENGINE ===
            order_intent = order.get('_order_intent')
            if order_intent:
                idempotent_engine.record_order(
                    intent=order_intent,
                    broker_order_id=str(order_id),
                    quantity=order['quantity'],
                    price=fill_price,
                    status="OPEN"
                )
            
            print(f"   ✅ LIVE {order['side']} {order['quantity']} {symbol} @ ₹{fill_price:.2f}")
            print(f"      SL: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
            print(f"      Order ID: {order_id} | SL Order: {sl_order_id}")
            
            return {
                "success": True,
                "order_id": order_id,
                "sl_order_id": sl_order_id,
                "gtt_trigger_id": gtt_trigger_id,
                "entry_price": fill_price,
                "message": f"LIVE Order placed: {order['side']} {order['quantity']} {symbol} @ ₹{fill_price:.2f}",
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
                          use_intraday_scoring: bool = True,
                          lot_multiplier: float = 1.0,
                          setup_type: str = "",
                          ml_data: dict = None,
                          sector: str = "",
                          pre_fetched_market_data: dict = None) -> Dict:
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
        
        # === SHARED SAFETY GATES (same as place_order) ===
        # 1. Trading hours check
        from config import TRADING_HOURS
        now = datetime.now()
        no_new_after = datetime.strptime(TRADING_HOURS['no_new_after'], '%H:%M').time()
        market_start = datetime.strptime(TRADING_HOURS['start'], '%H:%M').time()
        if now.time() < market_start or now.time() > no_new_after:
            return {
                "success": False,
                "error": f"TRADING HOURS BLOCK: Current time {now.strftime('%H:%M')} outside {TRADING_HOURS['start']}-{TRADING_HOURS['no_new_after']}",
                "action": "Wait for trading hours"
            }
        
        # 2. Risk governor check (daily loss, consecutive losses, cooldown)
        from risk_governor import get_risk_governor
        risk_gov = get_risk_governor()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        trade_perm = risk_gov.can_trade_general(active_positions=active_positions, setup_type=setup_type)
        if not trade_perm.allowed:
            return {
                "success": False,
                "error": f"RISK GOVERNOR BLOCK: {trade_perm.reason}",
                "action": "Risk limit reached - no new trades"
            }
        
        # 2b. Regime-aware position limit (fewer positions in MIXED market)
        _open_count = len([t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN'])
        _regime_max = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)  # Default to MIXED (conservative)
        # If market_breadth was injected into market_data, use it
        try:
            _md_check = self.get_market_data([underlying])
            _breadth_check = _md_check.get(underlying, {}).get('market_breadth', 'MIXED') if isinstance(_md_check.get(underlying), dict) else 'MIXED'
            if _breadth_check in ('BULLISH', 'BEARISH'):
                _regime_max = HARD_RULES.get('MAX_POSITIONS_TRENDING', 12)
            else:
                _regime_max = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)
        except Exception:
            pass  # Use conservative default
        if _open_count >= _regime_max:
            return {
                "success": False,
                "error": f"REGIME POSITION LIMIT: {_open_count} open >= {_regime_max} max (market regime limit)",
                "action": "Too many positions for current market regime"
            }
        
        # 3. Position reconciliation check
        recon = get_position_reconciliation(kite=self.kite, paper_mode=self.paper_mode)
        recon_can_trade, recon_reason = recon.can_trade()
        if not recon_can_trade:
            return {
                "success": False,
                "error": f"RECONCILIATION BLOCK: {recon_reason}",
                "action": "Resolve mismatch before new trades"
            }
        
        # 4. Data health gate check
        health_gate = get_data_health_gate()
        try:
            md_health = self.get_market_data([underlying])
            if underlying in md_health and isinstance(md_health[underlying], dict):
                health_ok, health_reason = health_gate.can_trade(underlying, md_health[underlying])
                if not health_ok:
                    return {
                        "success": False,
                        "error": f"DATA HEALTH BLOCK: {health_reason}",
                        "action": "Wait for healthy data"
                    }
        except Exception as e:
            # print(f"   ⚠️ Data health check failed for option: {e}")
            pass
        
        # 5. Correlation guard check
        correlation_guard = get_correlation_guard()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        corr_check = correlation_guard.can_trade(
            symbol=underlying,
            active_positions=active_positions
        )
        if not corr_check.can_trade:
            return {
                "success": False,
                "error": f"CORRELATION BLOCK: {corr_check.reason}",
                "action": "Too many correlated positions"
            }
        
        # === CHECK FOR DUPLICATE OPTION ON SAME UNDERLYING ===
        for pos in self.paper_positions:
            if pos.get('status', 'OPEN') == 'OPEN' and pos.get('is_option') and pos.get('underlying') == underlying:
                return {
                    "success": False,
                    "error": f"DUPLICATE BLOCKED: Already have option position on {underlying} ({pos['symbol']})",
                    "action": "Skip - already holding option on this underlying"
                }
        
        # === RE-ENTRY COOLDOWN CHECK (20-min block after exit on same underlying) ===
        _cd_time = self._exit_cooldowns.get(underlying)
        if _cd_time:
            from config import HARD_RULES as _HR
            _cd_mins = _HR.get('REENTRY_COOLDOWN_MINUTES', 20)
            _elapsed = (datetime.now() - _cd_time).total_seconds() / 60
            if _elapsed < _cd_mins:
                _remaining = round(_cd_mins - _elapsed, 1)
                # print(f"   ⏳ COOLDOWN BLOCK: {underlying} exited {_elapsed:.1f} min ago, {_remaining} min remaining")
                return {
                    "success": False,
                    "error": f"COOLDOWN: {underlying} exited {_elapsed:.1f} min ago ({_remaining} min left of {_cd_mins}-min cooldown)",
                    "action": "Skip - re-entry cooldown active"
                }
            else:
                del self._exit_cooldowns[underlying]  # Cooldown expired, clean up
        
        # === GET INTRADAY MARKET DATA ===
        market_data = {}
        # Use pre-fetched market data if provided (e.g., from TEST_XGB for IV crush gate)
        if pre_fetched_market_data and isinstance(pre_fetched_market_data, dict) and len(pre_fetched_market_data) > 3:
            market_data = pre_fetched_market_data
        elif use_intraday_scoring:
            try:
                md = self.get_market_data([underlying])
                if underlying in md and isinstance(md[underlying], dict):
                    market_data = md[underlying]
                    # print(f"   📊 Using intraday signals for {underlying}")
                else:
                    print(f"   ⚠️ No indicator data for {underlying} — rejecting option order")
                    self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
                    return {
                        "success": False,
                        "error": f"Indicator data unavailable for {underlying}. Cannot score option trade without market signals.",
                        "reason": "Data health gate: indicators failed or timed out.",
                        "action": "SKIP - do NOT retry this symbol this cycle"
                    }
            except Exception as e:
                print(f"   ⚠️ Could not get intraday data: {e}")
                self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
                return {
                    "success": False,
                    "error": f"Failed to fetch market data for {underlying}: {e}",
                    "reason": "Data fetch error.",
                    "action": "SKIP - do NOT retry this symbol this cycle"
                }
        
        # Parse enums
        opt_type = None
        if option_type:
            opt_type = OptionType[option_type.upper()]
        
        # === LOOKUP CACHED CYCLE DECISION (avoid re-scoring) ===
        _cached = getattr(self, '_cached_cycle_decisions', {}).get(underlying, None)
        
        # === EXPIRY DAY SHIELD — Block/restrict 0DTE entries ===
        from config import EXPIRY_SHIELD_CONFIG
        if EXPIRY_SHIELD_CONFIG.get('enabled', False):
            try:
                from expiry_shield import expiry_entry_gate
                # Get nearest expiry to determine if today is expiry day
                _nearest_expiry = options_trader.chain_fetcher.get_nearest_expiry(
                    underlying.replace('NSE:', ''), ExpirySelection.CURRENT_WEEK
                )
                _expiry_iso = _nearest_expiry.isoformat() if _nearest_expiry else ''
                _is_spread = False  # Will be determined by routing below
                _gate_ok, _gate_reason = expiry_entry_gate('naked_option', _is_spread, _expiry_iso)
                if not _gate_ok:
                    # print(f"   🛡️ {_gate_reason}")
                    return {
                        "success": False,
                        "error": _gate_reason,
                        "action": "EXPIRY SHIELD: Wait or use different expiry"
                    }
                elif _gate_reason == "EXPIRY_DAY_TRADE":
                    # print(f"   ⚡ Expiry Day Shield: 0DTE entry — SL will be tightened, force-exit by 14:45")
                    pass
            except Exception as e:
                print(f"   ⚠️ Expiry shield check failed (proceeding): {e}")
        
        # === ROUTING REVERSAL: ELITE/ORB go naked-first, spreads as fallback ===
        # High-conviction directional setups should NOT be capped by credit spreads
        _naked_first = setup_type in ('ELITE', 'ORB_BREAKOUT')
        if _naked_first:
            print(f"   🎯 {setup_type} NAKED-FIRST: Skipping spread auto-route for {underlying} — directional conviction")
        
        # === AUTO-TRY DEBIT SPREAD FIRST (directional momentum play) ===
        from config import DEBIT_SPREAD_CONFIG
        if not _naked_first and DEBIT_SPREAD_CONFIG.get('enabled', False):
            try:
                # print(f"   🔄 Auto-trying debit spread FIRST for {underlying} ({direction})...")
                debit_result = self.place_debit_spread(
                    underlying=underlying,
                    direction=direction,
                    rationale=rationale or "Auto-routed from option order (debit spread primary)",
                    pre_fetched_market_data=market_data if market_data else None,
                    cached_decision=_cached
                )
                if debit_result.get('success'):
                    print(f"   ✅ Debit spread placed instead of naked buy!")
                    debit_result['auto_routed'] = True
                    debit_result['original_tool'] = 'place_option_order'
                    return debit_result
                else:
                    debit_reason = debit_result.get('error', 'Unknown')
                    # print(f"   🔄 Debit spread not viable ({debit_reason}), checking credit spread...")
            except Exception as e:
                # print(f"   ⚠️ Debit spread attempt failed ({e}), checking credit spread...")
                pass
        
        # === AUTO-TRY CREDIT SPREAD SECOND (theta-positive hedge) ===
        from config import CREDIT_SPREAD_CONFIG
        if not _naked_first and CREDIT_SPREAD_CONFIG.get('enabled', False) and CREDIT_SPREAD_CONFIG.get('primary_strategy', False):
            try:
                # print(f"   🔄 Auto-trying credit spread for {underlying} ({direction})...")
                spread_result = self.place_credit_spread(
                    underlying=underlying,
                    direction=direction,
                    rationale=rationale or "Auto-routed: debit spread failed, trying credit spread",
                    pre_fetched_market_data=market_data if market_data else None,
                    cached_decision=_cached
                )
                if spread_result.get('success'):
                    print(f"   ✅ Credit spread placed instead of naked buy!")
                    spread_result['auto_routed'] = True
                    spread_result['original_tool'] = 'place_option_order'
                    spread_result['fallback_from'] = 'debit_spread'
                    return spread_result
                else:
                    fallback_reason = spread_result.get('error', 'Unknown')
                    # print(f"   🔄 Credit spread not viable for {underlying}, falling back to naked buy...")
            except Exception as e:
                # print(f"   ⚠️ Credit spread attempt failed ({e}), falling back to naked buy...")
                pass
        
        strike_sel = StrikeSelection[strike_selection.upper()] if strike_selection != "AUTO" else None
        expiry_sel = ExpirySelection[expiry_selection.upper()] if expiry_selection != "AUTO" else None
        
        # === CREATE OPTION ORDER WITH INTRADAY SCORING ===
        # Pass market_data for IV crush gate even when not using intraday scoring
        # (e.g., TEST_XGB needs ATR for RV computation in IV/RV ratio check)
        _md_for_order = market_data if (use_intraday_scoring or pre_fetched_market_data) else None
        plan = options_trader.create_option_order(
            underlying=underlying,
            direction=direction,
            option_type=opt_type,
            strike_selection=strike_sel,
            expiry_selection=expiry_sel,
            market_data=_md_for_order,
            cached_decision=_cached,
            setup_type=setup_type
        )
        
        if plan is None:
            # Track rejection so autonomous_trader won't retry this symbol
            self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
            
            # === ELITE/ORB FALLBACK: Naked failed → now try spreads ===
            if _naked_first:
                print(f"   🔄 {setup_type} naked option rejected for {underlying}, trying spread fallback...")
                # Try debit spread
                from config import DEBIT_SPREAD_CONFIG as _nf_debit_cfg
                if _nf_debit_cfg.get('enabled', False):
                    try:
                        _nf_debit = self.place_debit_spread(
                            underlying=underlying, direction=direction,
                            rationale=rationale or f"{setup_type} fallback: naked rejected, trying debit spread",
                            pre_fetched_market_data=market_data if market_data else None,
                            cached_decision=_cached
                        )
                        if _nf_debit.get('success'):
                            print(f"   ✅ {setup_type} debit spread fallback placed for {underlying}!")
                            _nf_debit['auto_routed'] = True
                            _nf_debit['original_tool'] = 'place_option_order'
                            _nf_debit['fallback_from'] = f'{setup_type}_naked_rejected'
                            return _nf_debit
                    except Exception:
                        pass
                # Try credit spread
                from config import CREDIT_SPREAD_CONFIG as _nf_credit_cfg
                if _nf_credit_cfg.get('enabled', False) and _nf_credit_cfg.get('primary_strategy', False):
                    try:
                        _nf_credit = self.place_credit_spread(
                            underlying=underlying, direction=direction,
                            rationale=rationale or f"{setup_type} fallback: naked rejected, trying credit spread",
                            pre_fetched_market_data=market_data if market_data else None,
                            cached_decision=_cached
                        )
                        if _nf_credit.get('success'):
                            print(f"   ✅ {setup_type} credit spread fallback placed for {underlying}!")
                            _nf_credit['auto_routed'] = True
                            _nf_credit['original_tool'] = 'place_option_order'
                            _nf_credit['fallback_from'] = f'{setup_type}_naked_rejected'
                            return _nf_credit
                    except Exception:
                        pass
            
            # === AUTO-TRY IRON CONDOR for LOW-SCORE stocks (choppy theta harvest) ===
            from config import IRON_CONDOR_CONFIG
            if IRON_CONDOR_CONFIG.get('enabled', False):
                rejected_score = getattr(options_trader, '_last_rejected_score', 999)
                max_ic_score = IRON_CONDOR_CONFIG.get('max_directional_score', 45)
                min_ic_score = IRON_CONDOR_CONFIG.get('min_directional_score', 15)
                if min_ic_score <= rejected_score <= max_ic_score:
                    # DTE pre-check: skip IC if no eligible expiry today (0DTE only)
                    ic_dte_ok = getattr(self, '_ic_dte_eligible', None)
                    if ic_dte_ok is None:
                        # Compute once per session
                        ic_dte_ok = False
                        try:
                            # ExpirySelection already imported at module level (line 26)
                            idx_mode = IRON_CONDOR_CONFIG.get('index_mode', {})
                            stk_mode = IRON_CONDOR_CONFIG.get('stock_mode', {})
                            today_date = datetime.now().date()
                            # Quick check: index weekly
                            for idx_sym in ['NSE:NIFTY 50']:
                                try:
                                    idx_exp = options_trader.chain_fetcher.get_nearest_expiry(idx_sym, ExpirySelection[idx_mode.get('prefer_expiry', 'CURRENT_WEEK')])
                                    if idx_exp:
                                        from datetime import date as _date
                                        exp_d = idx_exp if isinstance(idx_exp, _date) and not isinstance(idx_exp, datetime) else idx_exp.date() if hasattr(idx_exp, 'date') else idx_exp
                                        if (exp_d - today_date).days <= idx_mode.get('max_dte', 2):
                                            ic_dte_ok = True
                                except Exception:
                                    pass
                            if not ic_dte_ok:
                                try:
                                    stk_exp = options_trader.chain_fetcher.get_nearest_expiry("NSE:RELIANCE", ExpirySelection[stk_mode.get('prefer_expiry', 'CURRENT_MONTH')])
                                    if stk_exp:
                                        from datetime import date as _date
                                        exp_d = stk_exp if isinstance(stk_exp, _date) and not isinstance(stk_exp, datetime) else stk_exp.date() if hasattr(stk_exp, 'date') else stk_exp
                                        if (exp_d - today_date).days <= stk_mode.get('max_dte', 15):
                                            ic_dte_ok = True
                                except Exception:
                                    pass
                            self._ic_dte_eligible = ic_dte_ok
                            if not ic_dte_ok:
                                # print(f"   🦅 IC DTE check: No expiry within limits — IC disabled for today")
                                pass
                        except Exception:
                            self._ic_dte_eligible = False
                            ic_dte_ok = False
                    
                    if not ic_dte_ok:
                        pass  # Silently skip — already printed once
                    else:
                        # print(f"   🦅 Score {rejected_score:.0f} too low for directional, trying IRON CONDOR...")
                        try:
                            ic_result = self.place_iron_condor(
                                underlying=underlying,
                                rationale=rationale or f"Auto-routed: directional score {rejected_score:.0f} → iron condor",
                                directional_score=rejected_score,
                                pre_fetched_market_data=market_data if market_data else None
                            )
                            if ic_result.get('success'):
                                print(f"   ✅ Iron Condor placed instead of directional trade!")
                                ic_result['auto_routed'] = True
                                ic_result['original_tool'] = 'place_option_order'
                                ic_result['fallback_from'] = 'directional_rejected'
                                return ic_result
                            else:
                                # print(f"   🔄 Iron Condor not viable: {ic_result.get('error', 'Unknown')}")
                                pass
                        except Exception as e:
                            # print(f"   ⚠️ Iron Condor attempt failed: {e}")
                            pass
            
            return {
                "success": False,
                "error": f"Intraday scorer REJECTED {underlying} - score below threshold. Do NOT retry this symbol.",
                "reason": "Intraday signals insufficient for options entry. Wait for stronger setup.",
                "action": "SKIP - look for other opportunities"
            }
        
        # === PENNY PREMIUM GATE (Feb 24 fix) ===
        # Block options with LTP below minimum — prevents position-size bombs
        # e.g. ₹0.94 option × 64,500 units: ₹0.10 move = ₹6,450 loss
        from config import HARD_RULES as _hr
        _min_premium = _hr.get('MIN_OPTION_PREMIUM', 3.0)
        if plan.contract.ltp < _min_premium:
            print(f"   🚫 PENNY PREMIUM BLOCKED: {plan.contract.symbol} LTP=₹{plan.contract.ltp:.2f} < min ₹{_min_premium}")
            return {
                "success": False,
                "error": f"Option premium ₹{plan.contract.ltp:.2f} below minimum ₹{_min_premium} — penny options create position-size bombs",
                "action": "Skip — premium too cheap, excessive unit exposure risk"
            }
        
        # === IV CRUSH GUARD (Feb 25 fix — IOC 22.8% IV trap) ===
        # Prevents buying low-IV options that are vulnerable to IV compression,
        # especially on high-breadth days where uncertainty is already low.
        # IOC CE bought at 22.8% IV on 47/50 bullish day → IV compressed → -10.7% premium loss.
        from config import IV_CRUSH_GUARD as _iv_cfg
        if _iv_cfg.get('enabled', True):
            try:
                _opt_iv = plan.contract.iv       # IV as decimal (e.g., 0.228 = 22.8%)
                _opt_ltp = plan.contract.ltp      # Option premium (₹)
                _opt_delta = plan.contract.delta   # Delta
                _opt_vega = plan.contract.vega     # Vega
                _sym = plan.contract.symbol
                
                if _opt_iv is not None and _opt_iv > 0:
                    # --- Sub-Gate 1: Absolute IV Floor ---
                    _iv_floor = _iv_cfg.get('min_iv_floor', 0.25)
                    
                    # --- Sub-Gate 2: Breadth-Adjusted IV Floor ---
                    # Check market breadth to raise the IV floor on extreme-breadth days
                    _effective_iv_floor = _iv_floor
                    _breadth_label = 'MIXED'
                    _breadth_ratio = 0.0
                    try:
                        _md_iv = self.get_market_data([underlying])
                        _breadth_label = _md_iv.get(underlying, {}).get('market_breadth', 'MIXED') if isinstance(_md_iv.get(underlying), dict) else 'MIXED'
                        # Compute breadth ratio from paper_positions context or market data
                        # Use a lightweight heuristic: BULLISH/BEARISH = extreme, MIXED = normal
                        if _breadth_label in ('BULLISH', 'BEARISH'):
                            _breadth_ratio = 0.90  # Assume ~90% when regime is strongly directional
                        else:
                            _breadth_ratio = 0.55  # Mixed regime = ~55%
                    except Exception:
                        pass
                    
                    _extreme_thresh = _iv_cfg.get('breadth_extreme_threshold', 0.85)
                    if _breadth_ratio >= _extreme_thresh:
                        _extreme_floor = _iv_cfg.get('breadth_extreme_iv_floor', 0.32)
                        _effective_iv_floor = max(_effective_iv_floor, _extreme_floor)
                    
                    if _opt_iv < _effective_iv_floor:
                        _iv_pct = _opt_iv * 100
                        _floor_pct = _effective_iv_floor * 100
                        _reason = f"IV {_iv_pct:.1f}% < floor {_floor_pct:.0f}%"
                        if _effective_iv_floor > _iv_floor:
                            _reason += f" (raised from {_iv_floor*100:.0f}% due to {_breadth_label} breadth)"
                        # print(f"   🧊 IV CRUSH GUARD BLOCKED: {_sym} — {_reason}")
                        # print(f"      Low IV = vega trap. Premium will shrink if IV compresses further.")
                        return {
                            "success": False,
                            "error": f"IV CRUSH GUARD: {_reason}. "
                                     f"Buying options at low IV exposes to vega loss if IV compresses. "
                                     f"IOC-type trap: direction right but premium falls.",
                            "action": "Skip — wait for higher IV or use spreads to neutralize vega"
                        }
                    
                    # --- Sub-Gate 3: Premium Floor (IV-safety) ---
                    _min_prem_iv = _iv_cfg.get('min_premium_for_iv_safety', 8.0)
                    if _opt_ltp and _opt_ltp < _min_prem_iv and _opt_iv < 0.35:
                        # print(f"   🧊 IV CRUSH GUARD (PREMIUM): {_sym} — LTP ₹{_opt_ltp:.2f} < ₹{_min_prem_iv:.0f} "
                        #       f"with IV {_opt_iv*100:.1f}%. Cheap + low IV = amplified vega risk.")
                        return {
                            "success": False,
                            "error": f"IV CRUSH GUARD: Premium ₹{_opt_ltp:.2f} < ₹{_min_prem_iv:.0f} with IV {_opt_iv*100:.1f}%. "
                                     f"Cheap OTM options suffer disproportionately from IV drops "
                                     f"(₹1 IV drop on ₹{_opt_ltp:.1f} = {1/_opt_ltp*100:.0f}% loss).",
                            "action": "Skip — choose deeper ITM or higher-premium strike"
                        }
                    
                    # --- Sub-Gate 4: Vega/Delta Ratio ---
                    _vd_iv_thresh = _iv_cfg.get('vega_delta_iv_threshold', 0.30)
                    _max_vd = _iv_cfg.get('max_vega_delta_ratio', 0.50)
                    if (_opt_iv < _vd_iv_thresh and _opt_delta and abs(_opt_delta) > 0.01 
                            and _opt_vega and abs(_opt_vega) > 0):
                        _vd_ratio = abs(_opt_vega) / abs(_opt_delta)
                        if _vd_ratio > _max_vd:
                            # print(f"   🧊 IV CRUSH GUARD (VEGA/Δ): {_sym} — |vega/delta|={_vd_ratio:.2f} > {_max_vd:.2f} "
                            #       f"with IV {_opt_iv*100:.1f}%. This is a vol bet, not a direction bet.")
                            return {
                                "success": False,
                                "error": f"IV CRUSH GUARD: Vega/Delta ratio {_vd_ratio:.2f} > {_max_vd:.2f} at IV {_opt_iv*100:.1f}%. "
                                         f"Vega dominates delta — trade profits need IV to rise, not just direction.",
                                "action": "Skip — choose higher-delta (deeper ITM) strike to reduce vega exposure"
                            }
                else:
                    # IV not available — log but don't block
                    # print(f"   ⚠️ IV CRUSH GUARD: IV not available for {plan.contract.symbol}, skipping IV checks")
                    pass
            except Exception as _iv_err:
                # print(f"   ⚠️ IV Crush Guard check failed (proceeding): {_iv_err}")
                pass
        
        # === MAX UNITS GATE (Feb 24 fix) ===
        # Hard cap on total units (lots × lot_size) to prevent outsized notional exposure
        _max_units = _hr.get('MAX_UNITS_PER_TRADE', 30000)
        _total_units = plan.quantity * plan.contract.lot_size
        if _total_units > _max_units:
            _new_lots = max(1, _max_units // plan.contract.lot_size)
            # print(f"   ⚠️ UNITS CAP: {plan.contract.symbol} {_total_units:,} units > {_max_units:,} max → capped to {_new_lots} lots ({_new_lots * plan.contract.lot_size:,} units)")
            plan.quantity = _new_lots
            plan.total_premium = _new_lots * plan.premium_per_lot
            plan.max_loss = plan.total_premium
        
        # === THETA BLEED ENTRY GATES (Feb 24 fix) ===
        # Three gates to prevent naked option buys that will bleed to theta:
        #   1. Theta/premium ratio gate — block if daily theta > 5% of premium
        #   2. DTE floor — block if DTE < 3 on non-expiry days
        #   3. Afternoon score multiplier — on expiry day after 1PM, require 1.5× score
        from config import THETA_ENTRY_GATE as _theta_cfg
        if _theta_cfg.get('enabled', True):
            try:
                _opt_ltp = plan.contract.ltp
                _opt_theta = plan.contract.theta
                _opt_expiry = plan.contract.expiry
                
                # Compute DTE for this contract
                _dte = -1
                if _opt_expiry:
                    from datetime import date as _date_cls
                    _exp_d = _opt_expiry.date() if hasattr(_opt_expiry, 'date') else _opt_expiry
                    _dte = (_exp_d - _date_cls.today()).days
                
                # Gate 1: Theta/premium ratio — block if theta eats >5% of premium per day
                if _opt_theta and _opt_ltp and _opt_ltp > 0:
                    _theta_pct = abs(_opt_theta) / _opt_ltp * 100
                    _max_theta_pct = _theta_cfg.get('max_theta_pct_of_premium', 5.0)
                    if _theta_pct > _max_theta_pct:
                        # print(f"   🕐 THETA GATE BLOCKED: {plan.contract.symbol} — daily θ=₹{_opt_theta:.2f} "
                        #       f"= {_theta_pct:.1f}% of LTP ₹{_opt_ltp:.2f} (max {_max_theta_pct}%)")
                        return {
                            "success": False,
                            "error": f"THETA GATE: Daily theta ₹{_opt_theta:.2f} = {_theta_pct:.1f}% of premium (>{_max_theta_pct}%). "
                                     f"Option will decay faster than directional move can compensate.",
                            "action": "Skip — choose higher DTE or deeper ITM option"
                        }
                
                # Gate 2: DTE floor — don't buy naked options with DTE < 3 (non-expiry days)
                # On expiry day (DTE=0), expiry_shield already handles entry restrictions
                _min_dte = _theta_cfg.get('min_dte_naked_buy', 3)
                from config import EXPIRY_SHIELD_CONFIG as _exp_cfg
                _is_expiry_day = _exp_cfg.get('is_monthly_expiry', False)
                if _dte >= 0 and _dte < _min_dte and not _is_expiry_day:
                    # print(f"   🕐 DTE FLOOR BLOCKED: {plan.contract.symbol} — DTE={_dte} < min {_min_dte} "
                    #       f"(theta curve steepens quadratically near expiry)")
                    return {
                        "success": False,
                        "error": f"DTE FLOOR: Option expires in {_dte} day(s), min DTE for naked buy is {_min_dte}. "
                                 f"Time decay accelerates exponentially near expiry.",
                        "action": "Skip — select further-dated expiry or use spread instead"
                    }
                
                # Gate 3: Afternoon theta multiplier — EXPIRY DAY ONLY, after 1PM
                # On expiry day afternoon, theta accelerates sharply. Require higher score.
                if _is_expiry_day:
                    _pm_after = _theta_cfg.get('expiry_day_pm_after', '13:00')
                    _pm_time = datetime.strptime(_pm_after, '%H:%M').time()
                    _now_time = datetime.now().time()
                    if _now_time >= _pm_time:
                        _score_mult = _theta_cfg.get('expiry_day_pm_score_multiplier', 1.5)
                        _entry_score = getattr(plan, 'entry_metadata', {}).get('entry_score', 0)
                        # Get the normal threshold from scorer
                        try:
                            from options_trader import get_intraday_scorer
                            _scorer = get_intraday_scorer()
                            _normal_threshold = getattr(_scorer, 'BLOCK_THRESHOLD', 57)
                        except Exception:
                            _normal_threshold = 57
                        _raised_threshold = _normal_threshold * _score_mult
                        if _entry_score < _raised_threshold:
                            # print(f"   🕐 EXPIRY PM GATE: {plan.contract.symbol} — Score {_entry_score:.0f} < "
                            #       f"{_raised_threshold:.0f} (normal {_normal_threshold}×{_score_mult} after {_pm_after} on expiry day)")
                            return {
                                "success": False,
                                "error": f"EXPIRY PM GATE: Score {_entry_score:.0f} < {_raised_threshold:.0f} "
                                         f"(need {_score_mult}× normal threshold after {_pm_after} on expiry day). "
                                         f"Afternoon theta decay is extreme.",
                                "action": "Skip — only highest-conviction trades allowed in expiry afternoon"
                            }
                        else:
                            # print(f"   ⚡ EXPIRY PM: Score {_entry_score:.0f} ≥ {_raised_threshold:.0f} "
                            #       f"— passed afternoon theta gate on expiry day")
                            pass
            except Exception as _theta_err:
                print(f"   ⚠️ Theta entry gate check failed (proceeding): {_theta_err}")
        
        # Validate risk - max premium check
        max_premium_per_trade = 150000  # ₹1.5 lakh per option trade
        if plan.total_premium > max_premium_per_trade:
            return {
                "success": False,
                "error": f"Premium ₹{plan.total_premium:.0f} exceeds limit ₹{max_premium_per_trade}",
                "action": "Reduce quantity or choose different strike"
            }
        
        # Check total options exposure — use paper_positions (source of truth, synced on exits)
        # options_trader.positions is stale (never removes exited trades)
        # GMM_SNIPER trades have separate ₹3L capital pool
        # Initialize both pools (only the relevant one is checked)
        current_exposure = 0
        max_total_exposure = 500000
        sniper_exposure = 0
        max_sniper_exposure = 270000
        if setup_type == 'GMM_SNIPER':
            from config import GMM_SNIPER as _sniper_cfg
            sniper_capital = _sniper_cfg.get('separate_capital', 300000)
            sniper_max_pct = _sniper_cfg.get('max_exposure_pct', 90) / 100.0
            max_sniper_exposure = sniper_capital * sniper_max_pct
            sniper_exposure = sum(p.get('total_premium', 0) for p in self.paper_positions 
                                if p.get('status', 'OPEN') == 'OPEN' and p.get('is_sniper', False))
            if sniper_exposure + plan.total_premium > max_sniper_exposure:
                return {
                    "success": False,
                    "error": f"Sniper capital exhausted: ₹{sniper_exposure:,.0f} + ₹{plan.total_premium:,.0f} > ₹{max_sniper_exposure:,.0f} (₹{sniper_capital/100000:.0f}L×{sniper_max_pct:.0%})",
                    "current_exposure": sniper_exposure,
                    "new_premium": plan.total_premium
                }
        else:
            max_total_exposure = 500000  # ₹5 lakh total option exposure (full capital)
            current_exposure = sum(p.get('total_premium', 0) for p in self.paper_positions 
                                 if p.get('status', 'OPEN') == 'OPEN' and not p.get('is_sniper', False))
            if current_exposure + plan.total_premium > max_total_exposure:
                return {
                    "success": False,
                    "error": f"Total option exposure would exceed ₹{max_total_exposure}",
                    "current_exposure": current_exposure,
                    "new_premium": plan.total_premium
                }
        
        # === APPLY LOT MULTIPLIER (for GMM sniper / high-conviction trades) ===
        if lot_multiplier and lot_multiplier != 1.0:
            original_lots = plan.quantity
            new_lots = max(1, round(plan.quantity * lot_multiplier))
            # Re-check premium cap against appropriate capital pool
            new_premium = new_lots * plan.premium_per_lot
            if setup_type == 'GMM_SNIPER':
                _can_afford = new_premium <= max_premium_per_trade and sniper_exposure + new_premium <= max_sniper_exposure
            else:
                _can_afford = new_premium <= max_premium_per_trade and current_exposure + new_premium <= max_total_exposure
            if _can_afford:
                plan.quantity = new_lots
                plan.total_premium = new_premium
                plan.max_loss = new_premium
                # print(f"   🎯 LOT MULTIPLIER: {lot_multiplier}x → {original_lots} → {new_lots} lots (₹{new_premium:,.0f})")
            else:
                # print(f"   ⚠️ Lot multiplier {lot_multiplier}x would exceed limits, keeping {original_lots} lots")
                pass

        # Execute the order
        result = options_trader.execute_option_order(plan)
        
        if result.get('success'):
            print(f"   📊 OPTION ORDER: {plan.contract.symbol}")
            print(f"      Premium: ₹{plan.total_premium:.0f} | Lots: {plan.quantity}")
            print(f"      Greeks: {plan.greeks_summary}")
            print(f"      Target: ₹{plan.target_premium:.2f} | SL: ₹{plan.stoploss_premium:.2f}")
            
            # Add to positions tracking (both paper and live mode)
            if self.paper_mode:
                # Options are always BOUGHT (BUY CE for bullish, BUY PE for bearish)
                # The direction (BUY/SELL) indicates the market view, not the option transaction
                option_side = 'BUY'  # We always buy options (debit), never write/sell them
                option_position = {
                    'symbol': plan.contract.symbol,
                    'underlying': plan.underlying,
                    'quantity': plan.quantity * plan.contract.lot_size,
                    'lots': plan.quantity,
                    'avg_price': plan.contract.ltp,
                    'side': option_side,
                    'direction': plan.direction,  # Final direction from scorer (may differ from LLM's original)
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
                    'rationale': rationale or plan.rationale,
                    # === ENTRY METADATA (for post-trade review) ===
                    'entry_metadata': getattr(plan, 'entry_metadata', {}),
                    'trade_id': getattr(plan, 'entry_metadata', {}).get('trade_id', ''),
                    'entry_score': getattr(plan, 'entry_metadata', {}).get('entry_score', 0),
                    'score_tier': getattr(plan, 'entry_metadata', {}).get('score_tier', 'unknown'),
                    'strategy_type': getattr(plan, 'entry_metadata', {}).get('strategy_type', 'NAKED_OPTION'),
                    'setup_type': setup_type or 'MANUAL',
                    'is_sniper': setup_type == 'GMM_SNIPER',
                    'lot_multiplier': lot_multiplier if lot_multiplier != 1.0 else 1.0,
                    # === TOP-LEVEL ML SCORES (easy access) ===
                    'smart_score': (ml_data or {}).get('smart_score', None),
                    'p_score': (ml_data or {}).get('p_score', None),
                    'dr_score': (ml_data or {}).get('dr_score', None),
                    'ml_move_prob': (ml_data or {}).get('ml_move_prob', None),
                    'ml_confidence': (ml_data or {}).get('ml_confidence', None),
                    # === ML MODEL DATA (XGB + VAE+GMM) ===
                    'xgb_model': (ml_data or {}).get('xgb_model', {}),
                    'gmm_model': (ml_data or {}).get('gmm_model', {}),
                    'ml_scored_direction': (ml_data or {}).get('scored_direction', ''),
                    'ml_xgb_disagrees': (ml_data or {}).get('xgb_disagrees', False),
                    'sector': sector or '',
                }
                with self._positions_lock:
                    self.paper_positions.append(option_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(option_position)
            else:
                # LIVE MODE: Track option position + get fill price
                option_side = 'BUY'
                try:
                    import time as _time
                    _time.sleep(0.5)
                    broker_orders = self.kite.orders()
                    fill_price = plan.contract.ltp
                    for bo in broker_orders:
                        if str(bo.get('order_id')) == str(result.get('order_id')) and bo.get('status') == 'COMPLETE':
                            fill_price = bo.get('average_price', plan.contract.ltp)
                            break
                except Exception:
                    fill_price = plan.contract.ltp
                
                option_position = {
                    'symbol': plan.contract.symbol,
                    'underlying': plan.underlying,
                    'quantity': plan.quantity * plan.contract.lot_size,
                    'lots': plan.quantity,
                    'avg_price': fill_price,
                    'side': option_side,
                    'direction': plan.direction,
                    'option_type': plan.contract.option_type.value,
                    'strike': plan.contract.strike,
                    'expiry': plan.contract.expiry.isoformat() if plan.contract.expiry else None,
                    'stop_loss': plan.stoploss_premium,
                    'target': plan.target_premium,
                    'order_id': result.get('order_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'is_option': True,
                    'is_live': True,
                    'total_premium': fill_price * plan.quantity * plan.contract.lot_size,
                    'max_loss': plan.max_loss,
                    'breakeven': plan.breakeven,
                    'delta': plan.contract.delta,
                    'theta': plan.contract.theta,
                    'iv': plan.contract.iv,
                    'rationale': rationale or plan.rationale,
                    'entry_metadata': getattr(plan, 'entry_metadata', {}),
                    'trade_id': getattr(plan, 'entry_metadata', {}).get('trade_id', ''),
                    'entry_score': getattr(plan, 'entry_metadata', {}).get('entry_score', 0),
                    'score_tier': getattr(plan, 'entry_metadata', {}).get('score_tier', 'unknown'),
                    'strategy_type': getattr(plan, 'entry_metadata', {}).get('strategy_type', 'NAKED_OPTION'),
                    'setup_type': setup_type or 'MANUAL',
                    'is_sniper': setup_type == 'GMM_SNIPER',
                    'lot_multiplier': lot_multiplier if lot_multiplier != 1.0 else 1.0,
                    # === TOP-LEVEL ML SCORES (easy access) ===
                    'smart_score': (ml_data or {}).get('smart_score', None),
                    'p_score': (ml_data or {}).get('p_score', None),
                    'dr_score': (ml_data or {}).get('dr_score', None),
                    'ml_move_prob': (ml_data or {}).get('ml_move_prob', None),
                    'ml_confidence': (ml_data or {}).get('ml_confidence', None),
                    # === ML MODEL DATA (XGB + VAE+GMM) ===
                    'xgb_model': (ml_data or {}).get('xgb_model', {}),
                    'gmm_model': (ml_data or {}).get('gmm_model', {}),
                    'ml_scored_direction': (ml_data or {}).get('scored_direction', ''),
                    'ml_xgb_disagrees': (ml_data or {}).get('xgb_disagrees', False),
                    'sector': sector or '',
                }
                
                # Place SL-M order for the option
                try:
                    exchange, tradingsymbol = plan.contract.symbol.split(':')
                    sl_trigger = plan.stoploss_premium * 0.999  # Slightly wider
                    sl_order_id = self._place_order_autoslice(
                        variety=self.kite.VARIETY_REGULAR,
                        exchange=exchange,
                        tradingsymbol=tradingsymbol,
                        transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                        quantity=plan.quantity * plan.contract.lot_size,
                        product=self.kite.PRODUCT_MIS,
                        order_type=self.kite.ORDER_TYPE_SLM,
                        trigger_price=sl_trigger,
                        validity=self.kite.VALIDITY_DAY,
                        tag='TITAN_OPT_SL'
                    )
                    option_position['sl_order_id'] = str(sl_order_id)
                    print(f"      🛡️ Live SL order placed: trigger ₹{sl_trigger:.2f} (order: {sl_order_id})")
                except Exception as e:
                    print(f"      ⚠️ Failed to place SL order for option: {e}")
                
                # Place GTT safety net for option
                gtt_trigger_id = self._place_gtt_safety_net(
                    symbol=plan.contract.symbol,
                    side=option_side,
                    quantity=plan.quantity * plan.contract.lot_size,
                    entry_price=fill_price,
                    sl_price=plan.stoploss_premium,
                    target_price=plan.target_premium,
                    product="MIS",
                    tag='TITAN_OPT'
                )
                if gtt_trigger_id:
                    option_position['gtt_trigger_id'] = gtt_trigger_id
                
                with self._positions_lock:
                    self.paper_positions.append(option_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(option_position)
                
                print(f"   ✅ LIVE OPTION: {option_side} {plan.quantity * plan.contract.lot_size} {plan.contract.symbol} @ ₹{fill_price:.2f}")
        
        # Subscribe the new option contract to WebSocket for real-time PnL
        self._subscribe_position_symbols()
        return result
    
    def place_credit_spread(self, underlying: str, direction: str,
                            spread_width: int = None,
                            rationale: str = "",
                            pre_fetched_market_data: dict = None,
                            cached_decision: dict = None) -> Dict:
        """
        Tool: Place a credit spread (SELL option + BUY hedge) — theta-positive strategy
        
        Credit spreads SELL an option to collect premium and BUY a further OTM option
        as a hedge. Theta (time decay) works IN OUR FAVOR.
        
        BULLISH view → Bull Put Spread: SELL OTM PE + BUY further OTM PE
        BEARISH view → Bear Call Spread: SELL OTM CE + BUY further OTM CE
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            direction: 'BUY' (bullish → bull put spread) or 'SELL' (bearish → bear call spread)
            spread_width: Strikes apart for hedge (default: 2 from config)
            rationale: Trade rationale
            
        Returns:
            Dict with spread execution result, credit received, max risk
        """
        from options_trader import get_options_trader, CreditSpreadPlan
        
        # === SAFETY GATES (same as place_option_order) ===
        # 1. Trading hours
        from config import TRADING_HOURS, CREDIT_SPREAD_CONFIG
        now = datetime.now()
        no_new_after = datetime.strptime(TRADING_HOURS['no_new_after'], '%H:%M').time()
        market_start = datetime.strptime(TRADING_HOURS['start'], '%H:%M').time()
        if now.time() < market_start or now.time() > no_new_after:
            return {"success": False, "error": f"Outside trading hours {TRADING_HOURS['start']}-{TRADING_HOURS['no_new_after']}"}
        
        # 2. Risk governor
        from risk_governor import get_risk_governor
        risk_gov = get_risk_governor()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        trade_perm = risk_gov.can_trade_general(active_positions=active_positions)
        if not trade_perm.allowed:
            return {"success": False, "error": f"RISK GOVERNOR: {trade_perm.reason}"}
        
        # 3. Position reconciliation check
        recon = get_position_reconciliation(kite=self.kite, paper_mode=self.paper_mode)
        recon_can_trade, recon_reason = recon.can_trade()
        if not recon_can_trade:
            return {"success": False, "error": f"RECONCILIATION BLOCK: {recon_reason}"}
        
        # 4. Correlation guard check
        correlation_guard = get_correlation_guard()
        corr_check = correlation_guard.can_trade(
            symbol=underlying,
            active_positions=active_positions
        )
        if not corr_check.can_trade:
            return {"success": False, "error": f"CORRELATION BLOCK: {corr_check.reason}"}
        
        # 5. Check for duplicate spreads on same underlying
        for pos in self.paper_positions:
            if pos.get('status', 'OPEN') == 'OPEN' and pos.get('is_credit_spread') and pos.get('underlying') == underlying:
                return {"success": False, "error": f"Already have credit spread on {underlying}"}
            if pos.get('status', 'OPEN') == 'OPEN' and pos.get('is_option') and pos.get('underlying') == underlying:
                return {"success": False, "error": f"Already have option position on {underlying}"}
        
        # 5b. Re-entry cooldown check (20-min block after exit)
        _cd_time = self._exit_cooldowns.get(underlying)
        if _cd_time:
            from config import HARD_RULES as _HR
            _cd_mins = _HR.get('REENTRY_COOLDOWN_MINUTES', 20)
            _elapsed = (datetime.now() - _cd_time).total_seconds() / 60
            if _elapsed < _cd_mins:
                _remaining = round(_cd_mins - _elapsed, 1)
                # print(f"   ⏳ COOLDOWN BLOCK: {underlying} exited {_elapsed:.1f} min ago, {_remaining} min remaining")
                return {"success": False, "error": f"COOLDOWN: {underlying} exited {_elapsed:.1f} min ago ({_remaining} min left)"}
            else:
                del self._exit_cooldowns[underlying]
        
        # 6. Get market data for scoring (use pre-fetched if available)
        market_data = {}
        if pre_fetched_market_data and isinstance(pre_fetched_market_data, dict) and len(pre_fetched_market_data) > 3:
            market_data = pre_fetched_market_data
            # print(f"   📊 Using pre-fetched market data for {underlying} (saved API call)")
        else:
            try:
                md = self.get_market_data([underlying])
                if underlying in md and isinstance(md[underlying], dict):
                    market_data = md[underlying]
                else:
                    self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
                    return {"success": False, "error": f"No indicator data for {underlying}"}
            except Exception as e:
                self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
                return {"success": False, "error": f"Market data fetch failed: {e}"}
        
        # 7. Data health gate check
        health_gate = get_data_health_gate()
        try:
            health_ok, health_reason = health_gate.can_trade(underlying, market_data)
            if not health_ok:
                return {"success": False, "error": f"DATA HEALTH BLOCK: {health_reason}"}
        except Exception as e:
            # print(f"   ⚠️ Data health check failed for spread: {e}")
            pass
        
        # === CREATE CREDIT SPREAD PLAN ===
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 500000),
            paper_mode=self.paper_mode
        )
        
        plan = options_trader.create_credit_spread(
            underlying=underlying,
            direction=direction,
            market_data=market_data,
            spread_width_strikes=spread_width,
            cached_decision=cached_decision
        )
        
        if plan is None:
            # Check if we should fall back to naked buy
            if CREDIT_SPREAD_CONFIG.get('fallback_to_buy', True):
                # print(f"   🔄 Credit spread not viable for {underlying}, checking naked buy fallback...")
                # Only fallback if score is high enough
                buy_threshold = CREDIT_SPREAD_CONFIG.get('buy_only_score_threshold', 70)
                # Try regular option order (it has its own scoring)
                return {
                    "success": False,
                    "error": f"Credit spread not viable for {underlying}",
                    "fallback": "place_option_order",
                    "action": f"Try place_option_order() if score >= {buy_threshold}"
                }
            
            self._scorer_rejected_symbols.add(underlying.replace('NSE:', ''))
            return {"success": False, "error": f"Credit spread rejected for {underlying}"}
        
        # === DhanHQ MARGIN PRE-CHECK (advisory) ===
        try:
            from dhan_risk_tools import get_dhan_risk_tools
            drt = get_dhan_risk_tools()
            if drt.ready:
                fund = drt.get_fund_limit()
                avail = float(fund.get('availabelBalance', 0))
                if avail > 0 and plan.max_risk > 0:
                    margin_ratio = plan.max_risk / avail if avail else 999
                    if margin_ratio > 0.5:
                        print(f"   ⚠️ DhanHQ: Spread max_risk ₹{plan.max_risk:,.0f} = {margin_ratio:.0%} of available ₹{avail:,.0f}")
                    else:
                        print(f"   ✅ DhanHQ: Margin OK — risk ₹{plan.max_risk:,.0f} / available ₹{avail:,.0f}")
        except Exception:
            pass  # Non-blocking advisory
        
        # === EXECUTE THE SPREAD ===
        result = options_trader.execute_credit_spread(plan)
        
        if result.get('success'):
            print(f"   ✅ CREDIT SPREAD PLACED: {plan.spread_type}")
            print(f"      SELL: {plan.sold_contract.symbol} @ ₹{plan.sold_contract.ltp:.2f}")
            print(f"      BUY:  {plan.hedge_contract.symbol} @ ₹{plan.hedge_contract.ltp:.2f}")
            print(f"      Credit: ₹{plan.net_credit_total:,.0f} | Max Risk: ₹{plan.max_risk:,.0f}")
            print(f"      NetΘ: {plan.net_theta:+.2f}/day | DTE: {plan.dte}")
            
            # Add to positions tracking (paper AND live mode)
            if self.paper_mode:
                spread_position = {
                    'symbol': f"{plan.sold_contract.symbol}|{plan.hedge_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_credit_spread': True,
                    'spread_type': plan.spread_type,
                    'direction': plan.direction,
                    'sold_symbol': plan.sold_contract.symbol,
                    'sold_strike': plan.sold_contract.strike,
                    'sold_premium': plan.sold_contract.ltp,
                    'hedge_symbol': plan.hedge_contract.symbol,
                    'hedge_strike': plan.hedge_contract.strike,
                    'hedge_premium': plan.hedge_contract.ltp,
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.net_credit,  # Credit received per share
                    'side': 'SELL',  # Net seller
                    'net_credit': plan.net_credit,
                    'net_credit_total': plan.net_credit_total,
                    'max_risk': plan.max_risk,
                    'spread_width': plan.spread_width,
                    'stop_loss': plan.stop_loss_debit,
                    'target': plan.target_credit,
                    'breakeven': plan.breakeven,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'credit_pct': plan.credit_pct,
                    'dte': plan.dte,
                    'is_option': True,
                    'order_id': result.get('spread_id', result.get('sold_order_id')),
                    'spread_id': result.get('spread_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': 0,  # Net credit strategy - no premium paid
                }
                with self._positions_lock:
                    self.paper_positions.append(spread_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(spread_position)
            else:
                # LIVE MODE: Track credit spread position (full metadata parity with paper)
                spread_position = {
                    'symbol': f"{plan.sold_contract.symbol}|{plan.hedge_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_credit_spread': True,
                    'is_live': True,
                    'spread_type': plan.spread_type,
                    'direction': plan.direction,
                    'sold_symbol': plan.sold_contract.symbol,
                    'sold_strike': plan.sold_contract.strike,
                    'sold_premium': plan.sold_contract.ltp,
                    'hedge_symbol': plan.hedge_contract.symbol,
                    'hedge_strike': plan.hedge_contract.strike,
                    'hedge_premium': plan.hedge_contract.ltp,
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.net_credit,
                    'side': 'SELL',
                    'net_credit': plan.net_credit,
                    'net_credit_total': plan.net_credit_total,
                    'max_risk': plan.max_risk,
                    'spread_width': plan.spread_width,
                    'stop_loss': plan.stop_loss_debit,
                    'target': plan.target_credit,
                    'breakeven': plan.breakeven,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'credit_pct': plan.credit_pct,
                    'dte': plan.dte,
                    'is_option': True,
                    'order_id': result.get('spread_id', result.get('sold_order_id')),
                    'sold_order_id': result.get('sold_order_id'),
                    'hedge_order_id': result.get('hedge_order_id'),
                    'spread_id': result.get('spread_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': 0,
                    # === ENTRY METADATA (parity with paper mode) ===
                    'strategy_type': 'CREDIT_SPREAD',
                    'trade_id': f"CS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'entry_score': getattr(plan, '_score', 0),
                    'score_tier': 'premium' if getattr(plan, '_score', 0) >= 70 else 'standard',
                    'entry_metadata': {
                        'strategy_type': 'CREDIT_SPREAD',
                        'spread_type': plan.spread_type,
                        'credit_pct': plan.credit_pct,
                        'dte': plan.dte,
                    },
                }
                with self._positions_lock:
                    self.paper_positions.append(spread_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(spread_position)
                print(f"   ✅ LIVE credit spread tracked: {plan.spread_type}")
        
        # Subscribe spread leg contracts to WebSocket for real-time PnL
        self._subscribe_position_symbols()
        return result
    
    def place_debit_spread(self, underlying: str, direction: str,
                           spread_width: int = None,
                           rationale: str = "",
                           pre_fetched_market_data: dict = None,
                           cached_decision: dict = None) -> Dict:
        """
        Tool: Place an intraday debit spread on a momentum mover.
        
        BUY near-ATM option + SELL further OTM option in direction of move.
        Profits from strong continuation. Cheaper than naked buy, defined risk.
        
        BULLISH → Bull Call Spread: BUY ATM CE + SELL OTM CE
        BEARISH → Bear Put Spread: BUY ATM PE + SELL OTM PE
        
        Smart candle gates filter: FT≥2, ADX≥28, ORB<120%, RangeExp<0.50.
        Tiered sizing: premium (score≥70) gets 3+ lots, standard gets 2+.
        R:R = 2.67:1 (SL 30%, Target 80%).
        
        Args:
            underlying: e.g., "NSE:EICHERMOT"
            direction: 'BUY' (bullish) or 'SELL' (bearish)
            spread_width: Strikes apart for sell leg (default: 3 from config)
            rationale: Trade rationale
        """
        from options_trader import get_options_trader, DebitSpreadPlan
        from config import TRADING_HOURS, DEBIT_SPREAD_CONFIG
        
        if not DEBIT_SPREAD_CONFIG.get('enabled', False):
            return {"success": False, "error": "Debit spreads disabled in config"}
        
        # === SAFETY GATES ===
        # 1. Trading hours + debit spread time cutoff
        now = datetime.now()
        no_new_after = datetime.strptime(TRADING_HOURS['no_new_after'], '%H:%M').time()
        market_start = datetime.strptime(TRADING_HOURS['start'], '%H:%M').time()
        debit_cutoff = datetime.strptime(DEBIT_SPREAD_CONFIG.get('no_entry_after', '14:00'), '%H:%M').time()
        
        if now.time() < market_start or now.time() > no_new_after:
            return {"success": False, "error": f"Outside trading hours"}
        if now.time() > debit_cutoff:
            return {"success": False, "error": f"Past debit spread cutoff ({DEBIT_SPREAD_CONFIG['no_entry_after']}) — not enough time"}
        
        # 2. Risk governor
        from risk_governor import get_risk_governor
        risk_gov = get_risk_governor()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        trade_perm = risk_gov.can_trade_general(active_positions=active_positions)
        if not trade_perm.allowed:
            return {"success": False, "error": f"RISK GOVERNOR: {trade_perm.reason}"}
        
        # 3. Position reconciliation
        recon = get_position_reconciliation(kite=self.kite, paper_mode=self.paper_mode)
        recon_can_trade, recon_reason = recon.can_trade()
        if not recon_can_trade:
            return {"success": False, "error": f"RECONCILIATION BLOCK: {recon_reason}"}
        
        # 4. Correlation guard
        correlation_guard = get_correlation_guard()
        corr_check = correlation_guard.can_trade(
            symbol=underlying,
            active_positions=active_positions
        )
        if not corr_check.can_trade:
            return {"success": False, "error": f"CORRELATION BLOCK: {corr_check.reason}"}
        
        # 5. Check for duplicate on same underlying
        for pos in self.paper_positions:
            if pos.get('status', 'OPEN') == 'OPEN' and pos.get('underlying') == underlying:
                return {"success": False, "error": f"Already have position on {underlying}"}
        
        # 5b. Re-entry cooldown check (20-min block after exit)
        _cd_time = self._exit_cooldowns.get(underlying)
        if _cd_time:
            from config import HARD_RULES as _HR
            _cd_mins = _HR.get('REENTRY_COOLDOWN_MINUTES', 20)
            _elapsed = (datetime.now() - _cd_time).total_seconds() / 60
            if _elapsed < _cd_mins:
                _remaining = round(_cd_mins - _elapsed, 1)
                # print(f"   ⏳ COOLDOWN BLOCK: {underlying} exited {_elapsed:.1f} min ago, {_remaining} min remaining")
                return {"success": False, "error": f"COOLDOWN: {underlying} exited {_elapsed:.1f} min ago ({_remaining} min left)"}
            else:
                del self._exit_cooldowns[underlying]
        
        # 6. Get market data
        market_data = {}
        if pre_fetched_market_data and isinstance(pre_fetched_market_data, dict) and len(pre_fetched_market_data) > 3:
            market_data = pre_fetched_market_data
            # print(f"   📊 Using pre-fetched market data for {underlying} (saved API call)")
        else:
            try:
                md = self.get_market_data([underlying])
                if underlying in md and isinstance(md[underlying], dict):
                    market_data = md[underlying]
                else:
                    return {"success": False, "error": f"No indicator data for {underlying}"}
            except Exception as e:
                return {"success": False, "error": f"Market data fetch failed: {e}"}
        
        # 7. Data health gate
        health_gate = get_data_health_gate()
        try:
            health_ok, health_reason = health_gate.can_trade(underlying, market_data)
            if not health_ok:
                return {"success": False, "error": f"DATA HEALTH BLOCK: {health_reason}"}
        except Exception as e:
            # print(f"   ⚠️ Data health check failed for debit spread: {e}")
            pass
        
        # === CREATE DEBIT SPREAD PLAN ===
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 500000),
            paper_mode=self.paper_mode
        )
        
        plan = options_trader.create_debit_spread(
            underlying=underlying,
            direction=direction,
            market_data=market_data,
            spread_width_strikes=spread_width,
            cached_decision=cached_decision
        )
        
        if plan is None:
            return {"success": False, "error": f"Debit spread not viable for {underlying} (check move%, score, liquidity)"}
        
        # === DhanHQ MARGIN PRE-CHECK (advisory) ===
        try:
            from dhan_risk_tools import get_dhan_risk_tools
            drt = get_dhan_risk_tools()
            if drt.ready and hasattr(plan, 'net_debit_total') and plan.net_debit_total > 0:
                fund = drt.get_fund_limit()
                avail = float(fund.get('availabelBalance', 0))
                if avail > 0:
                    cost_ratio = plan.net_debit_total / avail if avail else 999
                    if cost_ratio > 0.3:
                        print(f"   ⚠️ DhanHQ: Debit ₹{plan.net_debit_total:,.0f} = {cost_ratio:.0%} of available ₹{avail:,.0f}")
                    else:
                        print(f"   ✅ DhanHQ: Margin OK — debit ₹{plan.net_debit_total:,.0f} / available ₹{avail:,.0f}")
        except Exception:
            pass  # Non-blocking advisory
        
        # === EXECUTE ===
        result = options_trader.execute_debit_spread(plan)
        
        if result.get('success'):
            print(f"   ✅ DEBIT SPREAD PLACED: {plan.spread_type}")
            print(f"      BUY:  {plan.buy_contract.symbol} @ ₹{plan.buy_contract.ltp:.2f}")
            print(f"      SELL: {plan.sell_contract.symbol} @ ₹{plan.sell_contract.ltp:.2f}")
            print(f"      Debit: ₹{plan.net_debit_total:,.0f} | Max Profit: ₹{plan.max_profit:,.0f}")
            print(f"      NetΔ: {plan.net_delta:+.2f} | Move: {plan.move_pct:+.1f}% | DTE: {plan.dte}")
            
            if self.paper_mode:
                debit_position = {
                    'symbol': f"{plan.buy_contract.symbol}|{plan.sell_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_debit_spread': True,
                    'is_credit_spread': False,
                    'spread_type': plan.spread_type,
                    'direction': plan.direction,
                    'buy_symbol': plan.buy_contract.symbol,
                    'buy_strike': plan.buy_contract.strike,
                    'buy_premium': plan.buy_contract.ltp,
                    'sell_symbol': plan.sell_contract.symbol,
                    'sell_strike': plan.sell_contract.strike,
                    'sell_premium': plan.sell_contract.ltp,
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.net_debit,  # Debit paid per share
                    'side': 'BUY',  # Net buyer
                    'net_debit': plan.net_debit,
                    'net_debit_total': plan.net_debit_total,
                    'max_profit': plan.max_profit,
                    'max_loss': plan.max_loss,
                    'spread_width': plan.spread_width,
                    'stop_loss': plan.stop_loss_value,
                    'target': plan.target_value,
                    'breakeven': plan.breakeven,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'move_pct': plan.move_pct,
                    'dte': plan.dte,
                    'is_option': True,
                    'order_id': result.get('spread_id', result.get('buy_order_id')),
                    'spread_id': result.get('spread_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': plan.net_debit_total,
                    # === ENTRY METADATA for debit spreads ===
                    'strategy_type': 'DEBIT_SPREAD',
                    'trade_id': f"DS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'entry_score': getattr(plan, '_score', 0),
                    'score_tier': 'premium' if getattr(plan, '_score', 0) >= 70 else 'standard',
                    'entry_metadata': {
                        'strategy_type': 'DEBIT_SPREAD',
                        'proactive': 'proactive' in (rationale or '').lower(),
                        'move_pct': plan.move_pct,
                        'dte': plan.dte,
                    },
                }
                with self._positions_lock:
                    self.paper_positions.append(debit_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(debit_position)
            else:
                # LIVE MODE: Track debit spread position
                debit_position = {
                    'symbol': f"{plan.buy_contract.symbol}|{plan.sell_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_debit_spread': True,
                    'is_live': True,
                    'is_credit_spread': False,
                    'spread_type': plan.spread_type,
                    'direction': plan.direction,
                    'buy_symbol': plan.buy_contract.symbol,
                    'buy_strike': plan.buy_contract.strike,
                    'buy_premium': plan.buy_contract.ltp,
                    'sell_symbol': plan.sell_contract.symbol,
                    'sell_strike': plan.sell_contract.strike,
                    'sell_premium': plan.sell_contract.ltp,
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.net_debit,
                    'side': 'BUY',
                    'net_debit': plan.net_debit,
                    'net_debit_total': plan.net_debit_total,
                    'max_profit': plan.max_profit,
                    'max_loss': plan.max_loss,
                    'spread_width': plan.spread_width,
                    'stop_loss': plan.stop_loss_value,
                    'target': plan.target_value,
                    'breakeven': plan.breakeven,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'move_pct': plan.move_pct,
                    'dte': plan.dte,
                    'is_option': True,
                    'order_id': result.get('spread_id', result.get('buy_order_id')),
                    'buy_order_id': result.get('buy_order_id'),
                    'sell_order_id': result.get('sell_order_id'),
                    'spread_id': result.get('spread_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': plan.net_debit_total,
                    'strategy_type': 'DEBIT_SPREAD',
                    'trade_id': f"DS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'entry_score': getattr(plan, '_score', 0),
                    'score_tier': 'premium' if getattr(plan, '_score', 0) >= 70 else 'standard',
                    'entry_metadata': {
                        'strategy_type': 'DEBIT_SPREAD',
                        'proactive': 'proactive' in (rationale or '').lower(),
                        'move_pct': plan.move_pct,
                        'dte': plan.dte,
                    },
                }
                with self._positions_lock:
                    self.paper_positions.append(debit_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(debit_position)
                print(f"   ✅ LIVE debit spread tracked: {plan.spread_type}")
        
        # Subscribe spread leg contracts to WebSocket for real-time PnL
        self._subscribe_position_symbols()
        return result
    
    # =================================================================
    # THESIS HEDGE PROTOCOL — CONVERT NAKED OPTION → DEBIT SPREAD
    # =================================================================
    
    def convert_naked_to_spread(self, trade: Dict, tie_check: str = "") -> Dict:
        """
        Convert an existing naked option into a debit spread by selling an OTM leg.
        Called by THP when TIE fires on a hedgeable check (R_COLLAPSE, NEVER_SHOWED_LIFE, UNDERLYING_BOS).
        
        The naked option becomes the BUY leg; we add a SELL leg further OTM.
        This caps max loss at (buy_entry - sell_premium) and gives time for recovery.
        
        Args:
            trade: The existing naked option position dict (from paper_positions)
            tie_check: The TIE check that triggered this (for logging)
            
        Returns:
            {"success": True, ...} or {"success": False, "error": "..."}
        """
        from options_trader import get_options_trader, OptionType, ExpirySelection
        from config import THESIS_HEDGE_CONFIG
        
        if not THESIS_HEDGE_CONFIG.get('enabled', False):
            return {"success": False, "error": "THP disabled in config"}
        
        symbol = trade.get('symbol', '')
        underlying = trade.get('underlying', '')
        option_type_str = trade.get('option_type', '')  # 'CE' or 'PE'
        buy_strike = trade.get('strike', 0)
        buy_entry_price = trade.get('avg_price', 0)
        expiry_str = trade.get('expiry', '')
        lot_size_from_trade = trade.get('lot_size', 0) or (trade.get('quantity', 0) // max(1, trade.get('lots', 1)))
        lots = trade.get('lots', 1)
        direction = trade.get('direction', '')
        
        if not all([underlying, option_type_str, buy_strike, buy_entry_price, expiry_str]):
            missing_fields = []
            if not underlying: missing_fields.append('underlying')
            if not option_type_str: missing_fields.append('option_type')
            if not buy_strike: missing_fields.append('strike')
            if not buy_entry_price: missing_fields.append('avg_price')
            if not expiry_str: missing_fields.append('expiry')
            return {"success": False, "error": f"Incomplete trade data for hedge: {symbol} — missing: {missing_fields}"}
        
        # Parse option type
        opt_type = OptionType.CE if option_type_str == 'CE' else OptionType.PE
        
        # Parse expiry
        from datetime import date as _date
        try:
            expiry = datetime.fromisoformat(expiry_str)
        except (ValueError, TypeError):
            try:
                expiry = datetime.strptime(str(expiry_str), '%Y-%m-%d')
            except Exception:
                return {"success": False, "error": f"Cannot parse expiry '{expiry_str}'"}
        
        # CRITICAL: Normalize to date object for Zerodha instrument matching
        # Zerodha instruments store expiry as datetime.date; datetime != date in Python 3
        from datetime import date as _date_type
        expiry_date = expiry.date() if isinstance(expiry, datetime) and not isinstance(expiry, _date_type) else expiry
        if hasattr(expiry_date, 'hour'):  # Still a datetime, force to date
            expiry_date = expiry_date.date()
        
        # Check DTE — on 0DTE, hedging is still valuable to cap max loss
        # but warn that sell leg premium may be thin
        dte = (expiry_date - datetime.now().date()).days
        if dte < 0:
            return {"success": False, "error": f"DTE={dte} — option already expired"}
        if dte == 0:
            print(f"   ⚡ THP 0DTE: Hedging on expiry day — sell leg premium may be thin but caps max loss")
        
        print(f"\n🛡️ THP: Converting {symbol} to debit spread (TIE: {tie_check})")
        print(f"   Buy leg: {option_type_str} {buy_strike} @ ₹{buy_entry_price:.2f}")
        
        # === FETCH OPTION CHAIN ===
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 500000),
            paper_mode=self.paper_mode
        )
        # Pass expiry_date (date object) to match Zerodha instrument format
        chain = options_trader.chain_fetcher.fetch_option_chain(underlying, expiry_date)
        if chain is None:
            return {"success": False, "error": f"Cannot fetch option chain for {underlying} exp={expiry_date} (type={type(expiry_date).__name__})"}
        
        # === FIND SELL LEG ===
        # Get all strikes of the same option type, sorted
        # Use expiry_date for comparison (contracts may store date or datetime)
        strikes = sorted(set(
            c.strike for c in chain.contracts
            if c.option_type == opt_type and (
                c.expiry == expiry_date or 
                (hasattr(c.expiry, 'date') and c.expiry.date() == expiry_date) or
                (hasattr(expiry_date, 'date') and c.expiry == expiry_date.date())
            )
        ))
        if not strikes:
            # Enhanced diagnostics — log chain contents for debugging
            all_expiries = set(str(c.expiry) + f"({type(c.expiry).__name__})" for c in chain.contracts[:5])
            print(f"   ⚠️ THP chain diag: {len(chain.contracts)} contracts, expiry_date={expiry_date}({type(expiry_date).__name__}), sample expiries={all_expiries}")
            return {"success": False, "error": f"No {option_type_str} strikes in chain for {underlying} (chain has {len(chain.contracts)} contracts, exp={expiry_date})"}
        
        sell_offset = THESIS_HEDGE_CONFIG.get('sell_strike_offset', 3)
        min_hedge_premium = THESIS_HEDGE_CONFIG.get('min_hedge_premium', 3.0)
        min_hedge_premium_pct = THESIS_HEDGE_CONFIG.get('min_hedge_premium_pct', 15)
        min_oi = THESIS_HEDGE_CONFIG.get('min_oi', 500)
        max_bid_ask_pct = THESIS_HEDGE_CONFIG.get('max_bid_ask_pct', 5.0)
        
        # Find the index of our buy strike (or nearest)
        buy_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - buy_strike))
        
        # Sell leg is further OTM:
        # CE (bullish): sell higher strike (buy_idx + offset)
        # PE (bearish): sell lower strike (buy_idx - offset)
        if opt_type == OptionType.CE:
            sell_idx = buy_idx + sell_offset
        else:
            sell_idx = buy_idx - sell_offset
        
        if sell_idx < 0 or sell_idx >= len(strikes):
            return {"success": False, "error": f"Sell leg index {sell_idx} out of range (total strikes: {len(strikes)})"}
        
        sell_strike = strikes[sell_idx]
        sell_contract = chain.get_contract(sell_strike, opt_type, expiry)
        if sell_contract is None:
            return {"success": False, "error": f"No contract at sell strike {sell_strike} {option_type_str}"}
        
        sell_premium = sell_contract.ltp
        sell_oi = sell_contract.oi
        sell_bid = sell_contract.bid
        sell_ask = sell_contract.ask
        
        # === LIQUIDITY VALIDATION ===
        if sell_oi < min_oi:
            return {"success": False, "error": f"Sell leg OI={sell_oi} < {min_oi} — illiquid"}
        
        if sell_premium > 0 and sell_bid > 0 and sell_ask > 0:
            bid_ask_pct = ((sell_ask - sell_bid) / sell_premium) * 100
            if bid_ask_pct > max_bid_ask_pct:
                return {"success": False, "error": f"Sell leg bid-ask spread {bid_ask_pct:.1f}% > {max_bid_ask_pct}% — illiquid"}
        
        # === MINIMUM PREMIUM CHECK ===
        if sell_premium < min_hedge_premium:
            return {"success": False, "error": f"Sell leg premium ₹{sell_premium:.2f} < ₹{min_hedge_premium:.2f} — too cheap to hedge"}
        
        premium_ratio_pct = (sell_premium / buy_entry_price * 100) if buy_entry_price > 0 else 0
        if premium_ratio_pct < min_hedge_premium_pct:
            return {"success": False, "error": f"Sell leg only {premium_ratio_pct:.1f}% of buy entry — need ≥{min_hedge_premium_pct}%"}
        
        # === COMPUTE SPREAD METRICS ===
        spread_width = abs(buy_strike - sell_strike)
        # Net debit = what we paid for buy leg - what we receive for sell leg
        net_debit = buy_entry_price - sell_premium
        if net_debit < 0:
            net_debit = 0  # Can't have negative debit
        net_debit_total = net_debit * lots * lot_size_from_trade
        
        print(f"   Sell leg: {option_type_str} {sell_strike} @ ₹{sell_premium:.2f} (OI: {sell_oi:,})")
        print(f"   Spread: width={spread_width} | net_debit=₹{net_debit:.2f}/share | total=₹{net_debit_total:,.0f}")
        
        # === EXECUTE SELL LEG ===
        import random
        sell_symbol = sell_contract.symbol
        quantity = lots * lot_size_from_trade
        
        if self.paper_mode:
            sell_order_id = f"PAPER_THP_{random.randint(100000, 999999)}"
            print(f"   📝 PAPER sell order: {sell_symbol} × {quantity} @ ₹{sell_premium:.2f}")
        else:
            # LIVE: Place sell (SELL) order for the OTM leg
            try:
                nfo_symbol = sell_symbol.replace("NFO:", "")
                sell_order_id = self._place_order_autoslice(
                    tradingsymbol=nfo_symbol,
                    exchange="NFO",
                    transaction_type="SELL",
                    quantity=quantity,
                    order_type="MARKET",
                    product="MIS",
                    tag="THP_HEDGE",
                )
                print(f"   🔴 LIVE sell order placed: {sell_symbol} × {quantity} → order_id={sell_order_id}")
            except Exception as e:
                return {"success": False, "error": f"Sell order failed: {e}"}
        
        # === COMPUTE HEDGED EXIT LEVELS ===
        hedged_sl_pct = THESIS_HEDGE_CONFIG.get('hedged_sl_pct', 50)
        hedged_target_pct = THESIS_HEDGE_CONFIG.get('hedged_target_pct', 150)
        
        # SL: spread value drops to (1 - sl_pct/100) * net_debit
        hedged_sl_value = net_debit * (1 - hedged_sl_pct / 100)
        if hedged_sl_value < 0:
            hedged_sl_value = 0
        
        # Target: spread value rises to (hedged_target_pct/100) * net_debit
        hedged_target_value = net_debit * (hedged_target_pct / 100)
        
        # === UPDATE POSITION DICT IN-PLACE ===
        with self._positions_lock:
            # Find the trade in paper_positions and update
            for pos in self.paper_positions:
                if pos.get('symbol') == symbol and pos.get('status') == 'OPEN':
                    # Convert to debit spread
                    old_symbol = pos['symbol']
                    pos['symbol'] = f"{old_symbol}|{sell_symbol}"
                    pos['is_debit_spread'] = True
                    pos['hedged_from_tie'] = True
                    pos['tie_check'] = tie_check
                    pos['hedge_timestamp'] = datetime.now().isoformat()
                    # Spread leg info
                    pos['buy_symbol'] = old_symbol
                    pos['buy_strike'] = buy_strike
                    pos['buy_premium'] = buy_entry_price
                    pos['sell_symbol'] = sell_symbol
                    pos['sell_strike'] = sell_strike
                    pos['sell_premium'] = sell_premium
                    pos['sell_order_id'] = sell_order_id
                    # Spread metrics
                    pos['net_debit'] = net_debit
                    pos['net_debit_total'] = net_debit_total
                    pos['spread_width'] = spread_width
                    pos['spread_type'] = f"{'BULL_CALL' if opt_type == OptionType.CE else 'BEAR_PUT'}_SPREAD"
                    # New exit levels
                    pos['stop_loss'] = hedged_sl_value
                    pos['target'] = hedged_target_value
                    pos['avg_price'] = net_debit  # Now tracking net debit instead of buy premium
                    pos['strategy_type'] = 'THP_HEDGED_SPREAD'
                    break
            self._save_active_trades()
        
        # Subscribe sell leg to WebSocket
        self._subscribe_position_symbols()
        
        print(f"   ✅ THP HEDGE COMPLETE: {symbol} → debit spread")
        print(f"      SL: ₹{hedged_sl_value:.2f} | Target: ₹{hedged_target_value:.2f}")
        
        return {
            "success": True,
            "symbol": f"{symbol}|{sell_symbol}",
            "sell_symbol": sell_symbol,
            "sell_strike": sell_strike,
            "sell_premium": sell_premium,
            "sell_order_id": sell_order_id,
            "net_debit": net_debit,
            "net_debit_total": net_debit_total,
            "spread_width": spread_width,
            "hedged_sl": hedged_sl_value,
            "hedged_target": hedged_target_value,
            "tie_check": tie_check,
        }
    
    def unwind_hedge(self, trade: Dict, sell_leg_ltp: float) -> Dict:
        """
        Unwind a THP hedge by buying back the sold leg.
        Converts the debit spread back to a naked option — restores full upside.
        
        Called when the buy leg has recovered past entry price, indicating
        the original thesis has re-validated and the hedge is now capping profits.
        
        Args:
            trade: The THP-hedged position dict (is_debit_spread=True, hedged_from_tie=True)
            sell_leg_ltp: Current LTP of the sell leg (cost to buy back)
            
        Returns:
            {"success": True, ...} or {"success": False, "error": "..."}
        """
        symbol = trade.get('symbol', '')
        buy_symbol = trade.get('buy_symbol', '')
        sell_symbol = trade.get('sell_symbol', '')
        sell_entry_premium = trade.get('sell_premium', 0)  # What we received when we sold
        buy_entry_price = trade.get('buy_premium', 0)       # Original buy leg entry
        lots = trade.get('lots', 1)
        lot_size = trade.get('lot_size', 0) or (trade.get('quantity', 0) // max(1, lots))
        quantity = lots * lot_size
        
        if not all([buy_symbol, sell_symbol, sell_leg_ltp > 0]):
            return {"success": False, "error": "Missing data for unwind"}
        
        # Cost of buying back the sold leg
        buyback_cost_per_share = sell_leg_ltp - sell_entry_premium  # Positive = loss on hedge leg
        buyback_cost_total = buyback_cost_per_share * quantity
        
        print(f"\n🔓 THP UNWIND: Buying back sold leg to restore full upside")
        print(f"   Sell leg: {sell_symbol}")
        print(f"   Sold at: ₹{sell_entry_premium:.2f} → Buy back at: ₹{sell_leg_ltp:.2f}")
        print(f"   Hedge leg cost: ₹{buyback_cost_per_share:+.2f}/share (₹{buyback_cost_total:+,.0f} total)")
        
        # === EXECUTE BUYBACK ===
        import random
        if self.paper_mode:
            buyback_order_id = f"PAPER_UNWIND_{random.randint(100000, 999999)}"
            print(f"   📝 PAPER buyback: {sell_symbol} × {quantity} @ ₹{sell_leg_ltp:.2f}")
        else:
            try:
                nfo_symbol = sell_symbol.replace("NFO:", "")
                buyback_order_id = self._place_order_autoslice(
                    tradingsymbol=nfo_symbol,
                    exchange="NFO",
                    transaction_type="BUY",  # Buy back the sold leg
                    quantity=quantity,
                    order_type="MARKET",
                    product="MIS",
                    tag="THP_UNWIND",
                )
                print(f"   🟢 LIVE buyback placed: {sell_symbol} × {quantity} → order_id={buyback_order_id}")
            except Exception as e:
                return {"success": False, "error": f"Buyback order failed: {e}"}
        
        # === REVERT POSITION DICT TO NAKED OPTION ===
        with self._positions_lock:
            for pos in self.paper_positions:
                if pos.get('symbol') == symbol and pos.get('status') == 'OPEN':
                    # Revert symbol to buy leg only
                    pos['symbol'] = buy_symbol
                    pos['is_debit_spread'] = False
                    pos['hedged_from_tie'] = False
                    pos['hedge_unwound'] = True
                    pos['unwind_timestamp'] = datetime.now().isoformat()
                    pos['unwind_cost_per_share'] = buyback_cost_per_share
                    pos['unwind_cost_total'] = buyback_cost_total
                    # Restore naked option fields
                    pos['avg_price'] = buy_entry_price
                    pos['stop_loss'] = buy_entry_price * 0.85  # Fresh 15% SL from entry
                    pos['target'] = buy_entry_price * 1.40     # Fresh 40% target
                    pos['strategy_type'] = 'NAKED_OPTION'
                    # Clean up spread fields
                    for key in ['sell_symbol', 'sell_strike', 'sell_premium', 'sell_order_id',
                                'net_debit', 'net_debit_total', 'spread_width', 'spread_type',
                                'tie_check', 'hedge_timestamp']:
                        pos.pop(key, None)
                    break
            self._save_active_trades()
        
        self._subscribe_position_symbols()
        
        print(f"   ✅ UNWIND COMPLETE: {buy_symbol} is naked again — full upside restored")
        
        return {
            "success": True,
            "symbol": buy_symbol,
            "old_symbol": symbol,
            "buyback_cost_per_share": buyback_cost_per_share,
            "buyback_cost_total": buyback_cost_total,
            "buyback_order_id": buyback_order_id,
            "buy_entry_price": buy_entry_price,
        }
    
    # =================================================================
    # IRON CONDOR — NEUTRAL THETA HARVEST (CHOPPY STOCKS)
    # =================================================================
    
    def place_iron_condor(self, underlying: str,
                          rationale: str = "",
                          directional_score: float = 0,
                          pre_fetched_market_data: dict = None) -> Dict:
        """
        Tool: Place an Iron Condor on a choppy/range-bound stock to harvest theta.
        
        4 legs: SELL OTM CE + BUY further OTM CE + SELL OTM PE + BUY further OTM PE
        Profits when stock stays INSIDE the sold strikes (range-bound).
        
        Best for stocks with low directional score (< 45) and small intraday move.
        
        Args:
            underlying: e.g., "NSE:BHEL"
            rationale: Trade rationale
            directional_score: The low score from directional scoring
            pre_fetched_market_data: Pre-fetched market data dict
            
        Returns:
            Dict with IC execution result
        """
        from options_trader import get_options_trader, IronCondorPlan
        from config import TRADING_HOURS, IRON_CONDOR_CONFIG
        
        if not IRON_CONDOR_CONFIG.get('enabled', False):
            return {"success": False, "error": "Iron Condor disabled in config"}
        
        # === SAFETY GATES ===
        now = datetime.now()
        no_new_after = datetime.strptime(TRADING_HOURS['no_new_after'], '%H:%M').time()
        market_start = datetime.strptime(TRADING_HOURS['start'], '%H:%M').time()
        ic_cutoff = datetime.strptime(IRON_CONDOR_CONFIG.get('no_entry_after', '13:30'), '%H:%M').time()
        
        if now.time() < market_start or now.time() > no_new_after:
            return {"success": False, "error": "Outside trading hours"}
        if now.time() > ic_cutoff:
            return {"success": False, "error": f"Past IC cutoff ({IRON_CONDOR_CONFIG['no_entry_after']})"}
        
        # Risk governor
        from risk_governor import get_risk_governor
        risk_gov = get_risk_governor()
        active_positions = [t for t in self.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        trade_perm = risk_gov.can_trade_general(active_positions=active_positions)
        if not trade_perm.allowed:
            return {"success": False, "error": f"RISK GOVERNOR: {trade_perm.reason}"}
        
        # Duplicate check
        for pos in self.paper_positions:
            if pos.get('status', 'OPEN') == 'OPEN' and pos.get('underlying') == underlying:
                return {"success": False, "error": f"Already have position on {underlying}"}
        
        # Re-entry cooldown check (20-min block after exit)
        _cd_time = self._exit_cooldowns.get(underlying)
        if _cd_time:
            from config import HARD_RULES as _HR
            _cd_mins = _HR.get('REENTRY_COOLDOWN_MINUTES', 20)
            _elapsed = (datetime.now() - _cd_time).total_seconds() / 60
            if _elapsed < _cd_mins:
                _remaining = round(_cd_mins - _elapsed, 1)
                # print(f"   ⏳ COOLDOWN BLOCK: {underlying} exited {_elapsed:.1f} min ago, {_remaining} min remaining")
                return {"success": False, "error": f"COOLDOWN: {underlying} exited {_elapsed:.1f} min ago ({_remaining} min left)"}
            else:
                del self._exit_cooldowns[underlying]
        
        # Get market data
        market_data = {}
        if pre_fetched_market_data and isinstance(pre_fetched_market_data, dict) and len(pre_fetched_market_data) > 3:
            market_data = pre_fetched_market_data
        else:
            try:
                md = self.get_market_data([underlying])
                if underlying in md and isinstance(md[underlying], dict):
                    market_data = md[underlying]
                else:
                    return {"success": False, "error": f"No indicator data for {underlying}"}
            except Exception as e:
                return {"success": False, "error": f"Market data fetch failed: {e}"}
        
        # === CREATE IRON CONDOR PLAN ===
        options_trader = get_options_trader(
            kite=self.kite,
            capital=getattr(self, 'paper_capital', 500000),
            paper_mode=self.paper_mode
        )
        
        plan = options_trader.create_iron_condor(
            underlying=underlying,
            market_data=market_data,
            directional_score=directional_score
        )
        
        if plan is None:
            return {"success": False, "error": f"Iron Condor not viable for {underlying}"}
        
        # === DhanHQ MARGIN PRE-CHECK (advisory) ===
        try:
            from dhan_risk_tools import get_dhan_risk_tools
            drt = get_dhan_risk_tools()
            if drt.ready:
                fund = drt.get_fund_limit()
                avail = float(fund.get('availabelBalance', 0))
                if avail > 0 and plan.max_risk > 0:
                    margin_ratio = plan.max_risk / avail if avail else 999
                    if margin_ratio > 0.5:
                        print(f"   ⚠️ DhanHQ: IC max_risk ₹{plan.max_risk:,.0f} = {margin_ratio:.0%} of available ₹{avail:,.0f}")
                    else:
                        print(f"   ✅ DhanHQ: Margin OK — risk ₹{plan.max_risk:,.0f} / available ₹{avail:,.0f}")
        except Exception:
            pass  # Non-blocking advisory
        
        # === EXECUTE ===
        result = options_trader.execute_iron_condor(plan)
        
        if result.get('success'):
            print(f"   ✅ IRON CONDOR PLACED: {plan.underlying}")
            print(f"      PE wing: SELL {plan.sold_pe_contract.symbol} + BUY {plan.hedge_pe_contract.symbol}")
            print(f"      CE wing: SELL {plan.sold_ce_contract.symbol} + BUY {plan.hedge_ce_contract.symbol}")
            print(f"      Credit: ₹{plan.total_credit_amount:,.0f} | Risk: ₹{plan.max_risk:,.0f}")
            print(f"      Zone: ₹{plan.lower_breakeven:.0f} — ₹{plan.upper_breakeven:.0f}")
            
            if self.paper_mode:
                ic_position = {
                    'symbol': f"{plan.sold_pe_contract.symbol}|{plan.sold_ce_contract.symbol}|{plan.hedge_pe_contract.symbol}|{plan.hedge_ce_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_iron_condor': True,
                    'is_credit_spread': False,
                    'is_debit_spread': False,
                    'strategy_type': 'IRON_CONDOR',
                    'direction': 'NEUTRAL',
                    # Upper wing
                    'sold_ce_symbol': plan.sold_ce_contract.symbol,
                    'sold_ce_strike': plan.sold_ce_contract.strike,
                    'sold_ce_premium': plan.sold_ce_contract.ltp,
                    'hedge_ce_symbol': plan.hedge_ce_contract.symbol,
                    'hedge_ce_strike': plan.hedge_ce_contract.strike,
                    'hedge_ce_premium': plan.hedge_ce_contract.ltp,
                    # Lower wing
                    'sold_pe_symbol': plan.sold_pe_contract.symbol,
                    'sold_pe_strike': plan.sold_pe_contract.strike,
                    'sold_pe_premium': plan.sold_pe_contract.ltp,
                    'hedge_pe_symbol': plan.hedge_pe_contract.symbol,
                    'hedge_pe_strike': plan.hedge_pe_contract.strike,
                    'hedge_pe_premium': plan.hedge_pe_contract.ltp,
                    # Sizing
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.total_credit,
                    'side': 'SELL',
                    'total_credit': plan.total_credit,
                    'total_credit_amount': plan.total_credit_amount,
                    'max_risk': plan.max_risk,
                    'ce_wing_width': plan.ce_wing_width,
                    'pe_wing_width': plan.pe_wing_width,
                    'upper_breakeven': plan.upper_breakeven,
                    'lower_breakeven': plan.lower_breakeven,
                    'profit_zone_width': plan.profit_zone_width,
                    'stop_loss': plan.stop_loss_debit,
                    'target': plan.target_buyback,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'credit_pct': plan.credit_pct,
                    'dte': plan.dte,
                    'directional_score': plan.directional_score,
                    'is_option': True,
                    'order_id': result.get('condor_id'),
                    'condor_id': result.get('condor_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': 0,
                    'entry_metadata': {
                        'strategy_type': 'IRON_CONDOR',
                        'directional_score': plan.directional_score,
                        'profit_zone': f"₹{plan.lower_breakeven:.0f}-₹{plan.upper_breakeven:.0f}",
                    },
                }
                with self._positions_lock:
                    self.paper_positions.append(ic_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(ic_position)
            else:
                # LIVE MODE: Track iron condor position
                ic_position = {
                    'symbol': f"{plan.sold_pe_contract.symbol}|{plan.sold_ce_contract.symbol}|{plan.hedge_pe_contract.symbol}|{plan.hedge_ce_contract.symbol}",
                    'underlying': plan.underlying,
                    'is_iron_condor': True,
                    'is_live': True,
                    'is_credit_spread': False,
                    'is_debit_spread': False,
                    'strategy_type': 'IRON_CONDOR',
                    'direction': 'NEUTRAL',
                    'sold_ce_symbol': plan.sold_ce_contract.symbol,
                    'sold_ce_strike': plan.sold_ce_contract.strike,
                    'sold_ce_premium': plan.sold_ce_contract.ltp,
                    'hedge_ce_symbol': plan.hedge_ce_contract.symbol,
                    'hedge_ce_strike': plan.hedge_ce_contract.strike,
                    'hedge_ce_premium': plan.hedge_ce_contract.ltp,
                    'sold_pe_symbol': plan.sold_pe_contract.symbol,
                    'sold_pe_strike': plan.sold_pe_contract.strike,
                    'sold_pe_premium': plan.sold_pe_contract.ltp,
                    'hedge_pe_symbol': plan.hedge_pe_contract.symbol,
                    'hedge_pe_strike': plan.hedge_pe_contract.strike,
                    'hedge_pe_premium': plan.hedge_pe_contract.ltp,
                    'quantity': plan.quantity * plan.lot_size,
                    'lots': plan.quantity,
                    'lot_size': plan.lot_size,
                    'avg_price': plan.total_credit,
                    'side': 'SELL',
                    'total_credit': plan.total_credit,
                    'total_credit_amount': plan.total_credit_amount,
                    'max_risk': plan.max_risk,
                    'ce_wing_width': plan.ce_wing_width,
                    'pe_wing_width': plan.pe_wing_width,
                    'upper_breakeven': plan.upper_breakeven,
                    'lower_breakeven': plan.lower_breakeven,
                    'profit_zone_width': plan.profit_zone_width,
                    'stop_loss': plan.stop_loss_debit,
                    'target': plan.target_buyback,
                    'net_delta': plan.net_delta,
                    'net_theta': plan.net_theta,
                    'net_vega': plan.net_vega,
                    'credit_pct': plan.credit_pct,
                    'dte': plan.dte,
                    'directional_score': plan.directional_score,
                    'is_option': True,
                    'order_id': result.get('condor_id'),
                    'condor_id': result.get('condor_id'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'OPEN',
                    'rationale': rationale or plan.rationale,
                    'total_premium': 0,
                    'entry_metadata': {
                        'strategy_type': 'IRON_CONDOR',
                        'directional_score': plan.directional_score,
                        'profit_zone': f"₹{plan.lower_breakeven:.0f}-₹{plan.upper_breakeven:.0f}",
                    },
                }
                with self._positions_lock:
                    self.paper_positions.append(ic_position)
                    self._save_active_trades()
                self._log_entry_to_ledger(ic_position)
                print(f"   ✅ LIVE iron condor tracked: {plan.underlying}")
        
        # Subscribe IC leg contracts to WebSocket for real-time PnL
        self._subscribe_position_symbols()
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
                    return {
                        "symbol": symbol,
                        "position": pos
                    }
            return {"error": f"No open option position for {symbol}"}
        
        # Get all Greeks
        return {
            "portfolio_greeks": options_trader.get_greeks_summary(),
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
        
        # Sync: Feed option positions from paper_positions into options_trader
        # so check_option_exits sees all positions (not just ones placed via OptionsTrader)
        with self._positions_lock:
            option_positions = [p for p in self.paper_positions 
                                if p.get('is_option', False) and p.get('status', 'OPEN') == 'OPEN']
        
        # Build positions list that check_option_exits expects
        for pos in option_positions:
            # Check if already tracked in options_trader.positions
            already_tracked = any(
                op.get('symbol') == pos.get('symbol') and op.get('status') == 'OPEN'
                for op in options_trader.positions
            )
            if not already_tracked:
                # Add to options_trader.positions with expected fields
                # Include expiry + timestamp for DTE-aware theta decay checks
                options_trader.positions.append({
                    'symbol': pos['symbol'],
                    'entry_premium': pos.get('avg_price', 0),
                    'target_premium': pos.get('target', pos.get('avg_price', 0) * 1.5),
                    'stoploss_premium': pos.get('stop_loss', pos.get('avg_price', 0) * 0.7),
                    'quantity': pos.get('lots', pos.get('quantity', 1)),
                    'lot_size': pos.get('lot_size', 1),
                    'greeks': pos.get('greeks', {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'iv': 0}),
                    'status': 'OPEN',
                    'timestamp': pos.get('timestamp', ''),
                    'expiry': pos.get('expiry', ''),
                })
        
        # Use the existing check_option_exits method that checks all positions
        all_exits = options_trader.check_option_exits()
        
        exit_signals = []
        for signal in all_exits:
            sig_type = signal.get('signal', '')
            if sig_type in ('TARGET_HIT', 'STOPLOSS_HIT', 'THETA_DECAY_WARNING'):
                exit_signals.append({
                    "symbol": signal['symbol'],
                    "reason": sig_type,
                    "exit_type": sig_type,
                    "current_pnl": signal.get('pnl_value', 0),
                    "should_exit": sig_type in ('TARGET_HIT', 'STOPLOSS_HIT', 'THETA_DECAY_WARNING'),
                })
            elif sig_type == 'THETA_DECAY_INFO':
                # Informational only (2+ DTE) — log but don't exit
                pass  # Silently skip — no action needed
        
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
            
            # Get spot price (ltp is lighter than full quote)
            quote = self.kite.ltp([symbol])
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
    
    def _place_option_order_legacy(self, option_symbol: str, action: str, quantity: int, 
                           premium: float, stop_loss_pct: float = 30) -> Dict:
        """
        LEGACY: Place an option order by specific symbol (used internally)
        For GPT tool calls, use place_option_order(underlying, direction) instead.
        
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
                with self._positions_lock:
                    self.paper_positions.append(position)
                    self._save_active_trades()
                self._log_entry_to_ledger(position)
                
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
            # LIVE MODE: Place real order (with autoslice for freeze qty protection)
            order_id = self._place_order_autoslice(
                variety=self.kite.VARIETY_REGULAR,
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=total_qty,
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET,
                market_protection=-1,
                tag='TITAN_OPT'
            )
            
            # === GTT SAFETY NET for naked option buys ===
            gtt_trigger_id = None
            if GTT_CONFIG.get('option_gtt', True):
                option_sl = premium * (1 - stop_loss_pct/100)
                option_target = premium * 1.5  # 50% profit target for safety net
                gtt_trigger_id = self._place_gtt_safety_net(
                    symbol=option_symbol,
                    side="BUY",
                    quantity=total_qty,
                    entry_price=premium,
                    sl_price=option_sl,
                    target_price=option_target,
                    product="MIS",
                    tag='TITAN_OPT'
                )
                # Store in trade record
                if gtt_trigger_id:
                    with self._positions_lock:
                        for trade in self.paper_positions:
                            if trade.get('symbol') == option_symbol and trade.get('status', 'OPEN') == 'OPEN':
                                trade['gtt_trigger_id'] = gtt_trigger_id
                                break
                        self._save_active_trades()
            
            return {
                "success": True,
                "order_id": order_id,
                "gtt_trigger_id": gtt_trigger_id,
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
        capital=500000,
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
