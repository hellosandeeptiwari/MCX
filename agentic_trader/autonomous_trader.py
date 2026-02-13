"""
AUTONOMOUS TRADING BOT
Fully autonomous - makes decisions and executes without approval

⚠️ WARNING: This bot trades REAL MONEY automatically!
- It will place orders without asking
- It can lose your entire capital
- Past performance doesn't guarantee future results

USE AT YOUR OWN RISK!
"""

import time
import json
import threading
from datetime import datetime, timedelta
import schedule
import sys
import os

# Fix Windows cp1252 encoding for emoji output
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure CWD is the script's directory so relative file paths (risk_state.json, active_trades.json) resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from config import HARD_RULES, APPROVED_UNIVERSE, TRADING_HOURS, FNO_CONFIG, TIER_1_OPTIONS, TIER_2_OPTIONS, FULL_FNO_SCAN
from llm_agent import TradingAgent
from zerodha_tools import get_tools, reset_tools
from market_scanner import get_market_scanner
from options_trader import update_fno_lot_sizes
from exit_manager import get_exit_manager, calculate_structure_sl
from execution_guard import get_execution_guard
from risk_governor import get_risk_governor, SystemState
from correlation_guard import get_correlation_guard
from regime_score import get_regime_scorer
from position_reconciliation import get_position_reconciliation
from data_health_gate import get_data_health_gate


class AutonomousTrader:
    """
    Fully autonomous trading bot that:
    1. Scans market every 5 minutes
    2. Identifies opportunities using GPT
    3. Executes trades automatically
    4. Manages risk and exits
    """
    
    def __init__(self, capital: float = 10000, paper_mode: bool = True):
        """
        Args:
            capital: Starting capital
            paper_mode: If True, simulates trades (no real orders)
        """
        self.capital = capital
        self.paper_mode = paper_mode
        self.start_capital = capital
        self.daily_pnl = 0
        self._pnl_lock = threading.Lock()  # Thread-safe P&L updates
        self.trades_today = []
        self.positions = []
        
        # Initialize agent with auto_execute=True
        print("\n" + "="*60)
        print("🤖 AUTONOMOUS TRADING BOT")
        print("="*60)
        print(f"\n  Capital: ₹{capital:,}")
        print(f"  Mode: {'📝 PAPER TRADING' if paper_mode else '💰 LIVE TRADING'}")
        print(f"  Risk per trade: {HARD_RULES['RISK_PER_TRADE']*100}%")
        print(f"  Max daily loss: {HARD_RULES['MAX_DAILY_LOSS']*100}%")
        print(f"  Max positions: {HARD_RULES['MAX_POSITIONS']}")
        print(f"\n  Universe: {len(APPROVED_UNIVERSE)} stocks ({len(TIER_1_OPTIONS)} Tier-1 + {len(TIER_2_OPTIONS)} Tier-2)")
        print(f"  Scanner: ALL F&O stocks (~200) scanned each cycle for wild-card movers")
        
        if not paper_mode:
            print("\n  ⚠️  LIVE MODE - Real orders will be placed!")
            confirm = input("  Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("  Aborted.")
                sys.exit(0)
        
        # Reset tools singleton to use new configuration
        reset_tools()
        # Auto-execute is ON for both paper and live trading
        self.agent = TradingAgent(auto_execute=True, paper_mode=paper_mode, paper_capital=capital)
        self.tools = get_tools(paper_mode=paper_mode, paper_capital=capital)
        
        # === RESTORE daily P&L from persisted realized P&L ===
        # On restart, zerodha_tools loads paper_pnl from active_trades.json.
        # Sync self.daily_pnl and self.capital so the display is correct.
        persisted_pnl = getattr(self.tools, 'paper_pnl', 0) or 0
        if persisted_pnl != 0:
            with self._pnl_lock:
                self.daily_pnl = persisted_pnl
                self.capital = capital + persisted_pnl
            print(f"  📊 Restored daily P&L: ₹{persisted_pnl:+,.0f} | Capital: ₹{self.capital:,.0f}")
        
        # Real-time monitoring
        self.monitor_running = False
        self.monitor_thread = None
        self.monitor_interval = 3  # Check every 3 seconds
        
        # ORB trade tracking - once per direction per symbol per day
        # Persisted to disk so mid-day restarts don't allow duplicate ORB entries
        self._orb_state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'orb_trades_state.json')
        self.orb_trades_today, self.orb_tracking_date = self._load_orb_state()
        
        # Exit Manager for consistent exits
        self.exit_manager = get_exit_manager()
        
        # Execution Guard for order quality
        self.execution_guard = get_execution_guard()
        
        # Risk Governor for account-level risk
        self.risk_governor = get_risk_governor(capital)
        
        # Correlation Guard for hidden overexposure
        self.correlation_guard = get_correlation_guard()
        
        # Regime Scorer for trade quality confidence
        self.regime_scorer = get_regime_scorer()
        
        # Position Reconciliation for broker sync (critical for live)
        self.position_recon = get_position_reconciliation(
            kite=self.tools.kite if hasattr(self.tools, 'kite') else None,
            paper_mode=paper_mode,
            check_interval=10
        )
        
        # Data Health Gate for data quality validation
        self.data_health_gate = get_data_health_gate()
        
        # Market Scanner for dynamic F&O stock discovery
        self.market_scanner = get_market_scanner(kite=self.tools.kite)
        self._wildcard_symbols = []  # Wild-card symbols from last scan
        self._wildcard_scores = {}   # Scanner scores for gate bypass
        self._wildcard_change = {}   # % change for momentum detection
        
        # Pre-populate lot sizes from Kite API at startup
        try:
            update_fno_lot_sizes(self.market_scanner.get_lot_map())
        except Exception as e:
            print(f"⚠️ Dynamic lot size fetch failed at startup (will retry on scan): {e}")
        
        # Start reconciliation loop
        self.position_recon.start()
        
        # Sync exit manager with existing positions (crash recovery)
        self._sync_exit_manager_with_positions()
        
        # === GTT ORPHAN CLEANUP (crash recovery) ===
        # If Titan crashed previously, some GTTs may still be active on Zerodha
        # for positions that were already closed. Clean them up.
        try:
            from config import GTT_CONFIG
            if GTT_CONFIG.get('cleanup_on_startup', True) and not paper_mode:
                self.tools._cleanup_orphaned_gtts()
        except Exception as e:
            print(f"  ⚠️ GTT startup cleanup skipped: {e}")
        
        print("\n  ✅ Bot initialized!")
        print("  🟢 Auto-execution: ON")
        print("  ⚡ Real-time monitoring: ENABLED (every 3 sec)")
        print("  📊 Exit Manager: ACTIVE")
        print("  🛡️ Execution Guard: ACTIVE")
        print("  ⚖️ Risk Governor: ACTIVE")
        print("  🔗 Correlation Guard: ACTIVE")
        print("  📈 Regime Scorer: ACTIVE")
        print("  🔄 Position Reconciliation: ACTIVE (every 10s)")
        print("  🛡️ Data Health Gate: ACTIVE")
        print("  📊 Options Trading: ACTIVE (F&O stocks)")
        print("  🛡️ GTT Safety Net: ACTIVE (server-side SL+target)")
        print("  📦 Autoslice: ACTIVE (freeze qty protection)")
        print("  ⚡ IOC Validity: ACTIVE (spread legs)")
        print("="*60)
    
    def _save_orb_state(self):
        """Persist ORB trade tracking to disk for restart safety"""
        try:
            state = {
                'date': str(self.orb_tracking_date),
                'trades': self.orb_trades_today
            }
            with open(self._orb_state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"⚠️ Failed to save ORB state: {e}")

    def _load_orb_state(self):
        """Load persisted ORB state; returns empty if file missing or stale date"""
        today = datetime.now().date()
        try:
            if os.path.exists(self._orb_state_file):
                with open(self._orb_state_file, 'r') as f:
                    state = json.load(f)
                saved_date = state.get('date', '')
                if saved_date == str(today):
                    trades = state.get('trades', {})
                    count = sum(1 for s in trades.values() for d, v in s.items() if v)
                    if count:
                        print(f"📊 ORB state restored: {count} direction(s) already traded today")
                    return trades, today
        except Exception as e:
            print(f"⚠️ Failed to load ORB state: {e}")
        return {}, today

    def _reset_orb_tracker_if_new_day(self):
        """Reset ORB tracker at start of new trading day"""
        today = datetime.now().date()
        if today != self.orb_tracking_date:
            self.orb_trades_today = {}
            self.orb_tracking_date = today
            self._save_orb_state()
            print(f"📅 New trading day - ORB tracker reset")
    
    def _is_orb_trade_allowed(self, symbol: str, direction: str) -> bool:
        """Check if ORB trade is allowed (once per direction per symbol per day)"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        return not self.orb_trades_today[symbol].get(direction, False)
    
    def _sync_exit_manager_with_positions(self):
        """
        Sync exit manager with paper_positions on startup.
        If exit_manager_state.json is missing/empty but we have active positions,
        re-register them so exit logic (breakeven, trailing SL, time stop) works.
        """
        active = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        active_symbols = {t.get('symbol', '') for t in active}
        
        # Clean ghost positions from exit manager (in state file but not in paper_positions)
        ghost_symbols = [sym for sym in list(self.exit_manager.trade_states.keys()) if sym not in active_symbols]
        for sym in ghost_symbols:
            self.exit_manager.remove_trade(sym)
            print(f"🧹 Exit Manager: Removed ghost position {sym} (not in active trades)")
        
        if not active:
            return
        
        already_tracked = set(self.exit_manager.trade_states.keys())
        registered = 0
        
        for trade in active:
            # Skip IC trades — they have their own dedicated monitoring
            if trade.get('is_iron_condor', False):
                continue
            symbol = trade.get('symbol', '')
            if symbol in already_tracked:
                # Estimate candle count from entry time if still 0
                state = self.exit_manager.trade_states[symbol]
                if state.candles_since_entry == 0 and state.entry_time:
                    elapsed_min = (datetime.now() - state.entry_time).total_seconds() / 60
                    estimated_candles = max(0, int(elapsed_min / 5))  # 5-min candles
                    state.candles_since_entry = estimated_candles
                continue  # Already restored from state file
            
            entry = trade.get('avg_price', trade.get('entry_price', 0))
            sl = trade.get('stop_loss', 0)
            target = trade.get('target', 0)
            qty = trade.get('quantity', 0)
            side = trade.get('side', 'BUY')
            
            if entry > 0 and sl > 0 and target > 0:
                self.exit_manager.register_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    quantity=qty
                )
                # Mark credit spread fields on the trade state
                if trade.get('is_credit_spread'):
                    state = self.exit_manager.trade_states.get(symbol)
                    if state:
                        state.is_credit_spread = True
                        state.net_credit = trade.get('net_credit', 0)
                        state.spread_width = trade.get('spread_width', 0)
                # Mark debit spread fields on the trade state
                if trade.get('is_debit_spread'):
                    state = self.exit_manager.trade_states.get(symbol)
                    if state:
                        state.is_debit_spread = True
                        state.net_debit = trade.get('net_debit', 0)
                        state.spread_width = trade.get('spread_width', 0)
                # Estimate candles from trade entry timestamp
                ts = trade.get('timestamp', '')
                if ts:
                    try:
                        entry_time = datetime.fromisoformat(ts)
                        elapsed_min = (datetime.now() - entry_time).total_seconds() / 60
                        estimated_candles = max(0, int(elapsed_min / 5))
                        state = self.exit_manager.trade_states.get(symbol)
                        if state:
                            state.candles_since_entry = estimated_candles
                            state.entry_time = entry_time
                    except:
                        pass
                registered += 1
        
        if registered:
            print(f"📊 Exit Manager: Synced {registered} existing positions for exit management")
    
    def _mark_orb_trade_taken(self, symbol: str, direction: str):
        """Mark ORB direction as used for symbol today"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        self.orb_trades_today[symbol][direction] = True
        self._save_orb_state()
        print(f"📊 ORB {direction} marked as taken for {symbol} today")
    
    def start_realtime_monitor(self):
        """Start the real-time position monitor in a separate thread"""
        if self.monitor_running:
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._realtime_monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("⚡ Real-time monitor started")
    
    def stop_realtime_monitor(self):
        """Stop the real-time monitor"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⚡ Real-time monitor stopped")
    
    def _realtime_monitor_loop(self):
        """Continuous loop that checks positions every few seconds"""
        candle_timer = 0  # Track time for candle increment
        while self.monitor_running:
            try:
                if self.is_trading_hours():
                    self._check_positions_realtime()
                    self._check_eod_exit()  # Check if need to exit before close
                    
                    # Increment candle counter every ~5 minutes (300s / monitor_interval)
                    candle_timer += self.monitor_interval
                    if candle_timer >= 300:  # 5 minutes = 1 candle
                        candle_timer = 0
                        for state in self.exit_manager.get_all_states():
                            self.exit_manager.increment_candles(state.symbol)
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _check_eod_exit(self):
        """Exit all positions before market close (3:30 PM)"""
        now = datetime.now().time()
        eod_exit_time = datetime.strptime("15:22", "%H:%M").time()  # Exit 8 mins before 3:30
        
        if now >= eod_exit_time:
            with self.tools._positions_lock:
                active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            if active_trades:
                print(f"\n⏰ END OF DAY - Closing all positions...")
                
                # Separate trade types
                regular_trades = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False)]
                spread_trades_eod = [t for t in active_trades if t.get('is_credit_spread', False)]
                debit_spread_trades_eod = [t for t in active_trades if t.get('is_debit_spread', False)]
                ic_trades_eod = [t for t in active_trades if t.get('is_iron_condor', False)]
                
                # Collect all symbols needed for quotes
                all_symbols = [t['symbol'] for t in regular_trades]
                for st in spread_trades_eod:
                    sold_sym = st.get('sold_symbol', '')
                    hedge_sym = st.get('hedge_symbol', '')
                    if sold_sym: all_symbols.append(sold_sym)
                    if hedge_sym: all_symbols.append(hedge_sym)
                for dt in debit_spread_trades_eod:
                    buy_sym = dt.get('buy_symbol', '')
                    sell_sym = dt.get('sell_symbol', '')
                    if buy_sym: all_symbols.append(buy_sym)
                    if sell_sym: all_symbols.append(sell_sym)
                for ict in ic_trades_eod:
                    for leg in ['sold_ce_symbol', 'hedge_ce_symbol', 'sold_pe_symbol', 'hedge_pe_symbol']:
                        sym = ict.get(leg, '')
                        if sym: all_symbols.append(sym)
                
                if not all_symbols:
                    return
                    
                try:
                    # Use ticker cache for EOD pricing if available
                    if self.tools.ticker and self.tools.ticker.connected:
                        ltp_data = self.tools.ticker.get_ltp_batch(all_symbols)
                        quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                        missing = [s for s in all_symbols if s not in quotes]
                        if missing:
                            quotes.update(self.tools.kite.ltp(missing))
                    else:
                        quotes = self.tools.kite.ltp(all_symbols)
                except:
                    return
                
                # --- Close regular trades ---
                for trade in regular_trades:
                    symbol = trade['symbol']
                    if symbol not in quotes:
                        continue
                    
                    ltp = quotes[symbol]['last_price']
                    entry = trade['avg_price']
                    qty = trade['quantity']
                    side = trade['side']
                    
                    if side == 'BUY':
                        pnl = (ltp - entry) * qty
                    else:
                        pnl = (entry - ltp) * qty
                    
                    print(f"   🚪 EOD EXIT: {symbol} @ ₹{ltp:.2f}")
                    print(f"      P&L: ₹{pnl:+,.2f}")
                    
                    # Grab exit manager state for this trade
                    eod_exit_detail = {'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'}
                    try:
                        em_st = self.exit_manager.get_trade_state(symbol)
                        if em_st:
                            eod_exit_detail['candles_held'] = em_st.candles_since_entry
                            eod_exit_detail['r_multiple_achieved'] = round(em_st.max_favorable_move, 3)
                            eod_exit_detail['max_favorable_excursion'] = round(em_st.highest_price, 2)
                    except Exception:
                        pass
                    
                    self.tools.update_trade_status(symbol, 'EOD_EXIT', ltp, pnl, exit_detail=eod_exit_detail)
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close credit spreads (both legs) ---
                for trade in spread_trades_eod:
                    spread_id = trade.get('spread_id', trade['symbol'])
                    sold_sym = trade.get('sold_symbol', '')
                    hedge_sym = trade.get('hedge_symbol', '')
                    qty = trade['quantity']
                    net_credit = trade.get('net_credit', 0)
                    
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp  # Cost to close the spread
                    pnl = (net_credit - current_debit) * qty
                    
                    print(f"   🚪 EOD EXIT SPREAD: {spread_id}")
                    print(f"      Sold leg: {sold_sym} @ ₹{sold_ltp:.2f} | Hedge: {hedge_sym} @ ₹{hedge_ltp:.2f}")
                    print(f"      Credit: ₹{net_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_debit, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close debit spreads (both legs) ---
                for trade in debit_spread_trades_eod:
                    spread_id = trade.get('spread_id', trade['symbol'])
                    buy_sym = trade.get('buy_symbol', '')
                    sell_sym = trade.get('sell_symbol', '')
                    qty = trade['quantity']
                    net_debit = trade.get('net_debit', 0)
                    
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    pnl = (current_value - net_debit) * qty
                    
                    print(f"   🚪 EOD EXIT DEBIT SPREAD: {spread_id}")
                    print(f"      Buy leg: {buy_sym} @ ₹{buy_ltp:.2f} | Sell: {sell_sym} @ ₹{sell_ltp:.2f}")
                    print(f"      Debit: ₹{net_debit:.2f} → Value: ₹{current_value:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_value, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close Iron Condors (all 4 legs) ---
                for trade in ic_trades_eod:
                    condor_id = trade.get('condor_id', trade['symbol'])
                    sold_ce_sym = trade.get('sold_ce_symbol', '')
                    hedge_ce_sym = trade.get('hedge_ce_symbol', '')
                    sold_pe_sym = trade.get('sold_pe_symbol', '')
                    hedge_pe_sym = trade.get('hedge_pe_symbol', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    
                    sold_ce_ltp = quotes.get(sold_ce_sym, {}).get('last_price', 0) if sold_ce_sym else 0
                    hedge_ce_ltp = quotes.get(hedge_ce_sym, {}).get('last_price', 0) if hedge_ce_sym else 0
                    sold_pe_ltp = quotes.get(sold_pe_sym, {}).get('last_price', 0) if sold_pe_sym else 0
                    hedge_pe_ltp = quotes.get(hedge_pe_sym, {}).get('last_price', 0) if hedge_pe_sym else 0
                    
                    # Current debit to close all 4 legs
                    current_debit = (sold_ce_ltp - hedge_ce_ltp) + (sold_pe_ltp - hedge_pe_ltp)
                    pnl = (total_credit - current_debit) * qty
                    
                    print(f"   🚪 EOD EXIT IRON CONDOR: {condor_id}")
                    print(f"      CE wing: Sold ₹{sold_ce_ltp:.2f} / Hedge ₹{hedge_ce_ltp:.2f}")
                    print(f"      PE wing: Sold ₹{sold_pe_ltp:.2f} / Hedge ₹{hedge_pe_ltp:.2f}")
                    print(f"      Credit: ₹{total_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_debit, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT', 'strategy': 'IRON_CONDOR'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    
                    # Record EOD IC close with risk governor
                    open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != trade['symbol']]
                    unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                    self.risk_governor.record_trade_result(trade['symbol'], pnl, pnl > 0, unrealized_pnl=unrealized)
                    self.risk_governor.update_capital(self.capital)
    
    def _check_positions_realtime(self):
        """Check all positions for target/stoploss hits using Exit Manager"""
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # === SYNC: Register any new trades the agent placed since last check ===
        self._sync_exit_manager_with_positions()
        
        # Separate trade types
        equity_trades = [t for t in active_trades if not t.get('is_option', False)]
        option_trades = [t for t in active_trades if t.get('is_option', False) and not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False)]
        spread_trades = [t for t in active_trades if t.get('is_credit_spread', False)]
        debit_spread_trades = [t for t in active_trades if t.get('is_debit_spread', False)]
        ic_trades = [t for t in active_trades if t.get('is_iron_condor', False)]
        
        # === GET CURRENT PRICES ===
        # For spreads (credit & debit), we need both leg symbols
        # For iron condors, we need all 4 leg symbols
        all_symbols = set()
        for t in active_trades:
            if t.get('is_iron_condor'):
                all_symbols.add(t.get('sold_ce_symbol', ''))
                all_symbols.add(t.get('hedge_ce_symbol', ''))
                all_symbols.add(t.get('sold_pe_symbol', ''))
                all_symbols.add(t.get('hedge_pe_symbol', ''))
            elif t.get('is_credit_spread'):
                all_symbols.add(t.get('sold_symbol', ''))
                all_symbols.add(t.get('hedge_symbol', ''))
            elif t.get('is_debit_spread'):
                all_symbols.add(t.get('buy_symbol', ''))
                all_symbols.add(t.get('sell_symbol', ''))
            else:
                all_symbols.add(t['symbol'])
        all_symbols.discard('')
        all_symbols = list(all_symbols)
        if all_symbols:
            try:
                # Use WebSocket cache first (zero API calls), fallback to REST
                if self.tools.ticker and self.tools.ticker.connected:
                    ltp_data = self.tools.ticker.get_ltp_batch(all_symbols)
                    quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                    # For any symbols not in cache, fall back to REST
                    missing = [s for s in all_symbols if s not in quotes]
                    if missing:
                        rest_data = self.tools.kite.ltp(missing)
                        quotes.update(rest_data)
                else:
                    quotes = self.tools.kite.ltp(all_symbols)
            except Exception as e:
                # If mixed exchange query fails, try separately
                quotes = {}
                if equity_trades:
                    try:
                        eq_syms = [t['symbol'] for t in equity_trades]
                        quotes.update(self.tools.kite.ltp(eq_syms))
                    except:
                        pass
                if option_trades:
                    try:
                        opt_syms = [t['symbol'] for t in option_trades]
                        quotes.update(self.tools.kite.ltp(opt_syms))
                    except:
                        pass
        else:
            quotes = {}
        
        # Print position status every 30 seconds (every 10th check)
        if not hasattr(self, '_monitor_count'):
            self._monitor_count = 0
        self._monitor_count += 1
        
        # Suppress dashboard while scan is running to reduce noise
        if getattr(self, '_scanning', False):
            show_status = False
        else:
            show_status = (self._monitor_count % 10 == 0)  # Every 30 seconds
        
        # === CHECK OPTION EXITS (Greeks-based) ===
        # Only check naked option trades — credit spreads use exit_manager
        if option_trades:
            try:
                option_exits = self.tools.check_option_exits()
                for exit_signal in option_exits:
                    symbol = exit_signal['symbol']
                    # Skip credit spread symbols (contain '|')
                    if '|' in symbol:
                        continue
                    reason = exit_signal.get('reason', 'Greeks exit')
                    
                    # Find the option trade
                    opt_trade = next((t for t in option_trades if t['symbol'] == symbol), None)
                    if opt_trade:
                        # Guard: re-check status to prevent double-close from main thread
                        if opt_trade.get('status', 'OPEN') != 'OPEN':
                            continue
                        entry = opt_trade['avg_price']
                        qty = opt_trade['quantity']
                        # Try to get current price
                        try:
                            opt_quote = self.tools.kite.ltp([symbol])
                            ltp = opt_quote[symbol]['last_price']
                        except:
                            ltp = entry  # Assume flat if can't get price
                        
                        pnl = (ltp - entry) * qty
                        print(f"\n📊 OPTION EXIT: {symbol}")
                        print(f"   Reason: {reason}")
                        print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                        print(f"   P&L: ₹{pnl:+,.0f}")
                        
                        self.tools.update_trade_status(symbol, exit_signal.get('exit_type', 'GREEKS_EXIT'), ltp, pnl)
                        with self._pnl_lock:
                            self.daily_pnl += pnl
                            self.capital += pnl
                        
                        # Record with Risk Governor (include unrealized P&L from open positions)
                        open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != symbol]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol, pnl, pnl > 0, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                        
                        # Notify scorer for per-symbol re-entry prevention
                        try:
                            from options_trader import get_intraday_scorer
                            scorer = get_intraday_scorer()
                            if pnl > 0:
                                scorer.record_symbol_win(symbol)
                            else:
                                scorer.record_symbol_loss(symbol)
                        except Exception:
                            pass
            except Exception as e:
                if show_status:
                    print(f"   ⚠️ Option exit check error: {e}")
        
        # === EXIT MANAGER INTEGRATION ===
        # Check all trades via exit manager (equity AND options AND spreads)
        # Filter out zero-price entries (quote failures) to prevent phantom SL triggers
        price_dict = {}
        for t in active_trades:
            if t.get('is_credit_spread'):
                # For credit spreads: compute current net debit (cost to close)
                sold_sym = t.get('sold_symbol', '')
                hedge_sym = t.get('hedge_symbol', '')
                sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0)
                hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0)
                if sold_ltp > 0 and hedge_ltp > 0:
                    # Current debit to close = sold_ltp - hedge_ltp (buy back sold, sell hedge)
                    current_debit = sold_ltp - hedge_ltp
                    price_dict[t['symbol']] = max(0, current_debit)  # Debit can't be negative
            elif t.get('is_debit_spread'):
                # For debit spreads: compute current spread value
                buy_sym = t.get('buy_symbol', '')
                sell_sym = t.get('sell_symbol', '')
                buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0)
                sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0)
                if buy_ltp > 0 and sell_ltp > 0:
                    # Current value = buy_ltp - sell_ltp (what we'd receive if closing)
                    current_value = buy_ltp - sell_ltp
                    price_dict[t['symbol']] = max(0, current_value)
            else:
                ltp_val = quotes.get(t['symbol'], {}).get('last_price', 0)
                if ltp_val and ltp_val > 0:
                    price_dict[t['symbol']] = ltp_val
        exit_signals = self.exit_manager.check_all_exits(price_dict)
        
        # Process exit signals first (highest priority)
        for signal in exit_signals:
            if signal.should_exit:
                trade = next((t for t in active_trades if t['symbol'] == signal.symbol), None)
                if trade:
                    symbol = signal.symbol
                    # Guard: re-check status to prevent double-close from main thread
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    ltp = price_dict.get(symbol, 0) or signal.exit_price
                    # Guard: never exit at price 0 (quote failure)
                    if ltp <= 0:
                        print(f"   ⚠️ Skipping {symbol} exit — LTP is 0 (quote failure)")
                        continue
                    entry = trade['avg_price']
                    qty = trade['quantity']
                    side = trade['side']
                    
                    # === PARTIAL PROFIT EXIT ===
                    if signal.exit_type == "PARTIAL_PROFIT" and signal.partial_pct > 0:
                        # Get lot size to ensure we exit in whole lots
                        lot_size = trade.get('quantity', qty) // max(1, trade.get('lots', 1))  # shares per lot
                        total_lots = qty // lot_size if lot_size > 0 else 1
                        exit_lots = total_lots // 2  # Exit half the lots
                        exit_qty = exit_lots * lot_size
                        
                        if exit_qty <= 0 or exit_lots < 1:
                            continue
                        # Calculate P&L on partial quantity only
                        if side == 'BUY':
                            partial_pnl = (ltp - entry) * exit_qty
                        else:
                            partial_pnl = (entry - ltp) * exit_qty
                        
                        remaining_qty = qty - exit_qty
                        pnl_pct = partial_pnl / (entry * exit_qty) * 100
                        
                        print(f"\n💰 PARTIAL_PROFIT! {symbol}")
                        print(f"   Reason: {signal.reason}")
                        print(f"   Booked {exit_qty}/{qty} @ ₹{ltp:.2f}")
                        print(f"   Partial P&L: ₹{partial_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                        print(f"   Remaining: {remaining_qty} qty, SL moved to entry ₹{entry:.2f}")
                        
                        # Update trade quantity to remaining
                        self.tools.partial_exit_trade(symbol, exit_qty, ltp, partial_pnl)
                        with self._pnl_lock:
                            self.daily_pnl += partial_pnl
                            self.capital += partial_pnl
                        
                        # Update exit manager state with reduced quantity
                        em_state = self.exit_manager.get_trade_state(symbol)
                        if em_state:
                            em_state.quantity = remaining_qty
                            self.exit_manager._persist_state()
                        
                        # Record partial win with risk governor
                        open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol + "_PARTIAL", partial_pnl, True, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                        continue
                    
                    # === FULL EXIT ===
                    if trade.get('is_credit_spread'):
                        # Credit spread P&L = (net_credit - current_debit) × quantity
                        credit = trade.get('net_credit', 0)
                        current_debit = ltp  # For spreads, ltp stores current net debit
                        pnl = (credit - current_debit) * qty
                        pnl_pct = (credit - current_debit) / credit * 100 if credit > 0 else 0
                    elif trade.get('is_debit_spread'):
                        # Debit spread P&L = (current_value - net_debit) × quantity
                        net_debit = trade.get('net_debit', 0)
                        current_value = ltp  # For debit spreads, ltp stores current spread value
                        pnl = (current_value - net_debit) * qty
                        pnl_pct = (current_value - net_debit) / net_debit * 100 if net_debit > 0 else 0
                    elif side == 'BUY':
                        pnl = (ltp - entry) * qty
                    else:
                        pnl = (entry - ltp) * qty
                    
                    pnl_pct = pnl / (entry * qty) * 100 if not trade.get('is_debit_spread') else pnl_pct
                    was_win = pnl > 0
                    
                    # Exit based on signal type
                    emoji = {
                        'SL_HIT': '❌',
                        'TARGET_HIT': '🎯',
                        'SESSION_CUTOFF': '⏰',
                        'TIME_STOP': '⏱️',
                        'TRAILING_SL': '📈',
                        'PARTIAL_PROFIT': '💰',
                        'OPTION_SPEED_GATE': '🚀',
                        'DEBIT_SPREAD_SL': '❌',
                        'DEBIT_SPREAD_TARGET': '🎯',
                        'DEBIT_SPREAD_TIME_EXIT': '⏰',
                        'DEBIT_SPREAD_TRAIL_SL': '📈',
                        'DEBIT_SPREAD_MAX_PROFIT': '💰',
                    }.get(signal.exit_type, '🚪')
                    
                    print(f"\n{emoji} {signal.exit_type}! {symbol}")
                    print(f"   Reason: {signal.reason}")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    
                    # Grab exit manager state BEFORE removing trade (has candles, R-multiple, etc.)
                    exit_detail = {}
                    try:
                        em_state = self.exit_manager.get_trade_state(symbol)
                        if em_state:
                            exit_detail = {
                                'candles_held': em_state.candles_since_entry,
                                'r_multiple_achieved': round(em_state.max_favorable_move, 3),
                                'max_favorable_excursion': round(em_state.highest_price, 2) if em_state.highest_price else 0,
                                'exit_reason': signal.reason,
                                'exit_type': signal.exit_type,
                                'breakeven_applied': em_state.breakeven_applied,
                                'trailing_active': em_state.trailing_active,
                                'partial_booked': em_state.partial_booked,
                                'current_sl_at_exit': round(em_state.current_sl, 2),
                            }
                            print(f"   📋 Exit context: {em_state.candles_since_entry} candles | MaxR: {em_state.max_favorable_move:.2f} | BE: {em_state.breakeven_applied} | Trail: {em_state.trailing_active}")
                    except Exception as e:
                        print(f"   ⚠️ Could not capture exit detail: {e}")
                    
                    self.tools.update_trade_status(symbol, signal.exit_type, ltp, pnl, exit_detail=exit_detail)
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    
                    # Record with Risk Governor (include unrealized P&L from open positions)
                    open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != symbol]
                    unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                    self.risk_governor.record_trade_result(symbol, pnl, was_win, unrealized_pnl=unrealized)
                    self.risk_governor.update_capital(self.capital)
                    
                    # Notify scorer for per-symbol re-entry prevention
                    try:
                        from options_trader import get_intraday_scorer
                        scorer = get_intraday_scorer()
                        if was_win:
                            scorer.record_symbol_win(symbol)
                        else:
                            scorer.record_symbol_loss(symbol)
                    except Exception:
                        pass
                    
                    # Record slippage if we have expected price
                    if hasattr(trade, 'expected_exit') and getattr(trade, 'expected_exit', None):
                        self.execution_guard.record_slippage(
                            symbol=symbol,
                            side=side,
                            expected_price=trade.expected_exit,
                            actual_price=ltp,
                            volume_regime=trade.get('volume_regime', 'NORMAL'),
                            order_type='MARKET'
                        )
                    
                    # Remove from exit manager
                    self.exit_manager.remove_trade(symbol)
                    continue  # Skip further processing for this trade
        
        # Update exit manager with current SL (sync back from trade state)
        for trade in active_trades:
            symbol = trade['symbol']
            state = self.exit_manager.get_trade_state(symbol)
            if state:
                # Sync trailing SL back to trade
                if state.current_sl != trade['stop_loss']:
                    old_sl = trade['stop_loss']
                    trade['stop_loss'] = state.current_sl
                    self.tools._save_active_trades()
                    
                    # LIVE MODE: Modify the SL order on the exchange
                    if not self.tools.paper_mode:
                        sl_order_id = trade.get('sl_order_id')
                        if sl_order_id:
                            try:
                                new_trigger = state.current_sl * (0.999 if trade['side'] == 'BUY' else 1.001)
                                self.tools.kite.modify_order(
                                    variety=self.tools.kite.VARIETY_REGULAR,
                                    order_id=sl_order_id,
                                    trigger_price=round(new_trigger, 1)
                                )
                                print(f"   🔄 LIVE SL modified on exchange: {symbol} {old_sl} → {state.current_sl}")
                            except Exception as e:
                                print(f"   ⚠️ Failed to modify SL order on exchange for {symbol}: {e}")
        
        # === IRON CONDOR MONITORING ===
        # Dedicated exit logic: target capture, SL, time-based auto-exit, breakout
        if ic_trades:
            try:
                from config import IRON_CONDOR_CONFIG
                ic_auto_exit_str = IRON_CONDOR_CONFIG.get('auto_exit_time', '14:50')
                ic_auto_exit_time = datetime.strptime(ic_auto_exit_str, '%H:%M').time()
                now_time = datetime.now().time()
                
                for trade in ic_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    
                    condor_id = trade.get('condor_id', trade['symbol'])
                    underlying = trade.get('underlying', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    total_credit_amount = trade.get('total_credit_amount', 0)
                    target_buyback = trade.get('target', 0)
                    stop_loss_debit = trade.get('stop_loss', 0)
                    upper_be = trade.get('upper_breakeven', 0)
                    lower_be = trade.get('lower_breakeven', 0)
                    sold_ce_strike = trade.get('sold_ce_strike', 0)
                    sold_pe_strike = trade.get('sold_pe_strike', 0)
                    
                    # Get current LTPs for all 4 legs
                    sold_ce_ltp = quotes.get(trade.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    hedge_ce_ltp = quotes.get(trade.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    sold_pe_ltp = quotes.get(trade.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    hedge_pe_ltp = quotes.get(trade.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                    
                    # Skip if we can't get prices for all legs
                    if not all([sold_ce_ltp, hedge_ce_ltp, sold_pe_ltp, hedge_pe_ltp]):
                        continue
                    
                    # Current cost to close = buy back sold legs, sell hedge legs
                    # P&L = credit collected - current debit to close
                    ce_debit = sold_ce_ltp - hedge_ce_ltp  # Cost to close CE wing
                    pe_debit = sold_pe_ltp - hedge_pe_ltp  # Cost to close PE wing
                    current_debit = ce_debit + pe_debit     # Total cost to close
                    ic_pnl = (total_credit - current_debit) * qty
                    ic_pnl_pct = (total_credit - current_debit) / total_credit * 100 if total_credit > 0 else 0
                    
                    # Get underlying spot price for breakout check
                    underlying_ltp = 0
                    try:
                        ul_quote = quotes.get(underlying, {})
                        if ul_quote:
                            underlying_ltp = ul_quote.get('last_price', 0)
                        if not underlying_ltp:
                            ul_data = self.tools.kite.ltp([underlying])
                            underlying_ltp = ul_data[underlying]['last_price']
                    except:
                        pass
                    
                    exit_reason = None
                    exit_type = None
                    
                    # === EXIT CHECK 1: TIME-BASED AUTO-EXIT (2:50 PM) ===
                    if now_time >= ic_auto_exit_time:
                        exit_reason = f"IC auto-exit at {ic_auto_exit_str} (before EOD vol spike)"
                        exit_type = "IC_TIME_EXIT"
                    
                    # === EXIT CHECK 2: TARGET HIT (credit captured %) ===
                    # If current debit < target_buyback, we've captured enough premium
                    elif current_debit <= target_buyback and target_buyback > 0:
                        exit_reason = f"IC TARGET HIT: debit ₹{current_debit:.2f} ≤ target ₹{target_buyback:.2f} ({ic_pnl_pct:.0f}% captured)"
                        exit_type = "IC_TARGET_HIT"
                    
                    # === EXIT CHECK 3: STOP LOSS (loss exceeds multiplier) ===
                    elif current_debit >= stop_loss_debit and stop_loss_debit > 0:
                        exit_reason = f"IC STOP LOSS: debit ₹{current_debit:.2f} ≥ SL ₹{stop_loss_debit:.2f}"
                        exit_type = "IC_SL_HIT"
                    
                    # === EXIT CHECK 4: BREAKOUT WARNING (price approaching sold strikes) ===
                    elif underlying_ltp > 0 and IRON_CONDOR_CONFIG.get('breakout_exit', True):
                        buffer_pct = IRON_CONDOR_CONFIG.get('breakout_buffer_pct', 0.3) / 100
                        ce_danger = sold_ce_strike * (1 - buffer_pct)
                        pe_danger = sold_pe_strike * (1 + buffer_pct)
                        
                        if underlying_ltp >= ce_danger:
                            exit_reason = f"IC BREAKOUT UP: {underlying} ₹{underlying_ltp:.0f} approaching sold CE strike ₹{sold_ce_strike:.0f}"
                            exit_type = "IC_BREAKOUT_EXIT"
                        elif underlying_ltp <= pe_danger:
                            exit_reason = f"IC BREAKOUT DOWN: {underlying} ₹{underlying_ltp:.0f} approaching sold PE strike ₹{sold_pe_strike:.0f}"
                            exit_type = "IC_BREAKOUT_EXIT"
                    
                    if exit_reason:
                        emoji = "🎯" if exit_type == "IC_TARGET_HIT" else "❌" if exit_type == "IC_SL_HIT" else "⏰" if exit_type == "IC_TIME_EXIT" else "🚨"
                        print(f"\n{emoji} {exit_type}: {condor_id}")
                        print(f"   {exit_reason}")
                        print(f"   CE wing: Sold ₹{sold_ce_ltp:.2f} / Hedge ₹{hedge_ce_ltp:.2f}")
                        print(f"   PE wing: Sold ₹{sold_pe_ltp:.2f} / Hedge ₹{hedge_pe_ltp:.2f}")
                        print(f"   Credit: ₹{total_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{ic_pnl:+,.2f} ({ic_pnl_pct:+.1f}%)")
                        
                        self.tools.update_trade_status(
                            trade['symbol'], exit_type, current_debit, ic_pnl,
                            exit_detail={
                                'exit_reason': exit_reason,
                                'exit_type': exit_type,
                                'strategy': 'IRON_CONDOR',
                                'credit': total_credit,
                                'debit_at_exit': current_debit,
                                'underlying_at_exit': underlying_ltp,
                            }
                        )
                        with self._pnl_lock:
                            self.daily_pnl += ic_pnl
                            self.capital += ic_pnl
                        
                        # Record with risk governor
                        open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != trade['symbol']]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(trade['symbol'], ic_pnl, ic_pnl > 0, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                    
                    elif show_status:
                        status = "🟢" if ic_pnl > 0 else "🔴"
                        spot_info = f" | Spot: ₹{underlying_ltp:.0f}" if underlying_ltp > 0 else ""
                        print(f"   {status} 🦅 IC {underlying}: Credit ₹{total_credit:.2f} → Debit ₹{current_debit:.2f} | P&L: ₹{ic_pnl:+,.0f} ({ic_pnl_pct:+.1f}%){spot_info}")
                        print(f"      Zone: ₹{lower_be:.0f} — ₹{upper_be:.0f} | SL: ₹{stop_loss_debit:.2f} | TGT: ₹{target_buyback:.2f}")
            except Exception as e:
                print(f"   ⚠️ IC monitoring error: {e}")
        
        if show_status and active_trades:
            print(f"\n👁️ LIVE POSITIONS [{datetime.now().strftime('%H:%M:%S')}]:")
            
            # --- Show credit spreads first ---
            if spread_trades:
                print(f"{'Spread':30} {'Credit':>10} {'Debit':>10} {'MaxRisk':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in spread_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    spread_id = trade.get('spread_id', trade['symbol'])
                    sold_sym = trade.get('sold_symbol', '')
                    hedge_sym = trade.get('hedge_symbol', '')
                    qty = trade['quantity']
                    net_credit = trade.get('net_credit', 0)
                    max_risk = trade.get('max_risk', (trade.get('spread_width', 0) - net_credit) * qty)
                    
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp
                    pnl = (net_credit - current_debit) * qty
                    pnl_pct = (net_credit - current_debit) / net_credit * 100 if net_credit > 0 else 0
                    
                    state = self.exit_manager.get_trade_state(trade['symbol'])
                    status_flags = "θ+ "
                    if state:
                        target_cr = getattr(state, 'target_credit', net_credit * 0.35)
                        sl_debit = getattr(state, 'stop_loss_debit', net_credit * 2)
                        status_flags += f"TGT:{target_cr:.1f} SL:{sl_debit:.1f}"
                    
                    status = "🟢" if pnl > 0 else "🔴"
                    # Shorten spread_id for display
                    display_name = spread_id[:28] if len(spread_id) > 28 else spread_id
                    print(f"{status} {display_name:28} ₹{net_credit:>9.2f} ₹{current_debit:>9.2f} ₹{max_risk:>9,.0f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ SELL {sold_sym} @ ₹{sold_ltp:.2f} | HEDGE {hedge_sym} @ ₹{hedge_ltp:.2f}")
                print()
            
            # --- Show debit spreads ---
            if debit_spread_trades:
                print(f"{'DebitSpread':30} {'Entry':>10} {'Value':>10} {'MaxProfit':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in debit_spread_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    spread_id = trade.get('spread_id', trade['symbol'])
                    buy_sym = trade.get('buy_symbol', '')
                    sell_sym = trade.get('sell_symbol', '')
                    qty = trade['quantity']
                    net_debit = trade.get('net_debit', 0)
                    max_profit = trade.get('max_profit', 0)
                    
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    pnl = (current_value - net_debit) * qty
                    pnl_pct = (current_value - net_debit) / net_debit * 100 if net_debit > 0 else 0
                    
                    state = self.exit_manager.get_trade_state(trade['symbol'])
                    status_flags = "Δ+ "
                    if state:
                        if state.trailing_active:
                            status_flags += "📈TRAIL "
                        if state.breakeven_applied:
                            status_flags += "🔒BE "
                        status_flags += f"TGT:{state.target:.1f} SL:{state.current_sl:.1f}"
                    
                    status = "🟢" if pnl > 0 else "🔴"
                    display_name = spread_id[:28] if len(spread_id) > 28 else spread_id
                    print(f"{status} {display_name:28} ₹{net_debit:>9.2f} ₹{current_value:>9.2f} ₹{max_profit:>9,.0f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ BUY {buy_sym} @ ₹{buy_ltp:.2f} | SELL {sell_sym} @ ₹{sell_ltp:.2f}")
                print()
            
            # --- Show iron condors ---
            ic_open = [t for t in ic_trades if t.get('status', 'OPEN') == 'OPEN']
            if ic_open:
                print(f"{'🦅 Iron Condor':30} {'Credit':>10} {'Debit':>10} {'MaxRisk':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in ic_open:
                    underlying = trade.get('underlying', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    total_credit_amount = trade.get('total_credit_amount', 0)
                    max_risk = trade.get('max_risk', 0)
                    upper_be = trade.get('upper_breakeven', 0)
                    lower_be = trade.get('lower_breakeven', 0)
                    target_buyback = trade.get('target', 0)
                    stop_loss_debit = trade.get('stop_loss', 0)

                    s_ce = quotes.get(trade.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    h_ce = quotes.get(trade.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    s_pe = quotes.get(trade.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    h_pe = quotes.get(trade.get('hedge_pe_symbol', ''), {}).get('last_price', 0)

                    if all([s_ce, h_ce, s_pe, h_pe]):
                        current_debit = (s_ce - h_ce) + (s_pe - h_pe)
                        ic_pnl = (total_credit - current_debit) * qty
                        ic_pnl_pct = (total_credit - current_debit) / total_credit * 100 if total_credit > 0 else 0
                    else:
                        current_debit = 0
                        ic_pnl = 0
                        ic_pnl_pct = 0

                    # Get underlying spot
                    ul_ltp = quotes.get(underlying, {}).get('last_price', 0)
                    spot_str = f" Spot:₹{ul_ltp:.0f}" if ul_ltp > 0 else ""

                    status = "🟢" if ic_pnl >= 0 else "🔴"
                    display_name = f"IC {underlying}"[:28]
                    print(f"{status} {display_name:28} ₹{total_credit:>9.2f} ₹{current_debit:>9.2f} ₹{max_risk:>9,.0f} ₹{ic_pnl:>+10,.0f} ({ic_pnl_pct:+.1f}%) θ+{spot_str}")
                    print(f"   └─ Zone: ₹{lower_be:.0f}—₹{upper_be:.0f} | TGT: ₹{target_buyback:.2f} | SL: ₹{stop_loss_debit:.2f}")
                    print(f"   └─ CE: SELL {trade.get('sold_ce_symbol','')} @₹{s_ce:.2f} | PE: SELL {trade.get('sold_pe_symbol','')} @₹{s_pe:.2f}")
                print()

            # --- Show regular trades ---
            regular_open = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False) and t.get('status', 'OPEN') == 'OPEN']
            if regular_open:
                print(f"{'Symbol':15} {'Side':6} {'Entry':>10} {'LTP':>10} {'SL':>10} {'Target':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
        
        for trade in active_trades:
            if trade.get('is_credit_spread', False) or trade.get('is_debit_spread', False) or trade.get('is_iron_condor', False):
                continue  # Already displayed above
            symbol = trade['symbol']
            if symbol not in quotes:
                continue
            if trade.get('status', 'OPEN') != 'OPEN':
                continue  # Already exited
            
            ltp = quotes[symbol]['last_price']
            entry = trade['avg_price']
            sl = trade['stop_loss']
            target = trade.get('target', entry * (1.02 if trade['side'] == 'BUY' else 0.98))
            qty = trade['quantity']
            side = trade['side']
            
            # Get exit manager state for status
            state = self.exit_manager.get_trade_state(symbol)
            status_flags = ""
            if state:
                if state.breakeven_applied:
                    status_flags += "🔒BE "
                if state.trailing_active:
                    status_flags += "📈TRAIL "
            
            # Calculate current P&L
            if side == 'BUY':
                pnl = (ltp - entry) * qty
                pnl_pct = (ltp - entry) / entry * 100
                
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (ltp - sl) / ltp * 100
                    tgt_dist = (target - ltp) / ltp * 100
                    print(f"{status} {symbol:13} {'BUY':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
            
            else:  # SHORT position
                pnl = (entry - ltp) * qty
                pnl_pct = (entry - ltp) / entry * 100
                
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (sl - ltp) / ltp * 100
                    tgt_dist = (ltp - target) / ltp * 100
                    print(f"{status} {symbol:13} {'SHORT':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
        
        # Print summary after all positions
        if show_status and active_trades:
            # Calculate total P&L including all spread types
            total_pnl = 0
            for t in active_trades:
                if t.get('status', 'OPEN') != 'OPEN':
                    continue
                if t.get('is_credit_spread', False):
                    sold_sym = t.get('sold_symbol', '')
                    hedge_sym = t.get('hedge_symbol', '')
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp
                    total_pnl += (t.get('net_credit', 0) - current_debit) * t['quantity']
                elif t.get('is_debit_spread', False):
                    buy_sym = t.get('buy_symbol', '')
                    sell_sym = t.get('sell_symbol', '')
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    total_pnl += (current_value - t.get('net_debit', 0)) * t['quantity']
                elif t.get('is_iron_condor', False):
                    s_ce = quotes.get(t.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    h_ce = quotes.get(t.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    s_pe = quotes.get(t.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    h_pe = quotes.get(t.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                    if all([s_ce, h_ce, s_pe, h_pe]):
                        ic_debit = (s_ce - h_ce) + (s_pe - h_pe)
                        total_pnl += (t.get('total_credit', 0) - ic_debit) * t['quantity']
                elif t['symbol'] in quotes:
                    if t['side'] == 'BUY':
                        total_pnl += (quotes[t['symbol']]['last_price'] - t['avg_price']) * t['quantity']
                    else:
                        total_pnl += (t['avg_price'] - quotes[t['symbol']]['last_price']) * t['quantity']
            print("-" * 95)
            print(f"📊 TOTAL UNREALIZED P&L: ₹{total_pnl:+,.0f} | Capital: ₹{self.capital:,.0f} | Daily P&L: ₹{self.daily_pnl:+,.0f}")
            # Print exit manager status
            print(self.exit_manager.get_status_summary())
    
    def reset_agent(self):
        """Reset agent to clear conversation history - but KEEP positions"""
        # DON'T reset tools - preserve paper positions!
        # Just create new agent with fresh conversation
        from llm_agent import TradingAgent
        self.agent = TradingAgent(auto_execute=True, paper_mode=self.paper_mode, paper_capital=self.capital)
    
    def _format_watchlist_for_prompt(self) -> str:
        """Format hot watchlist for GPT prompt — these are 55+ stocks that need re-evaluation"""
        try:
            from options_trader import get_hot_watchlist
            wl = get_hot_watchlist()
            if not wl:
                return 'No watchlist stocks — all prior attempts either succeeded or scored too low'
            lines = []
            for sym, entry in sorted(wl.items(), key=lambda x: x[1].get('score', 0), reverse=True):
                lines.append(
                    f"  🔥 {sym}: Score {entry.get('score', 0):.0f}/100 | Dir: {entry.get('direction', '?')} | "
                    f"Conviction: {entry.get('directional_strength', 0):.0f}/8 needed | "
                    f"Seen {entry.get('cycle_count', 1)}x — TRY AGAIN with place_option_order()"
                )
            return chr(10).join(lines)
        except Exception:
            return 'Watchlist unavailable'
    
    def is_trading_hours(self) -> bool:
        """Check if within trading hours"""
        now = datetime.now().time()
        start = datetime.strptime(TRADING_HOURS["start"], "%H:%M").time()
        end = datetime.strptime(TRADING_HOURS["end"], "%H:%M").time()
        return start <= now <= end
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit hit"""
        max_loss = self.start_capital * HARD_RULES["MAX_DAILY_LOSS"]
        if self.daily_pnl <= -max_loss:
            print(f"❌ Daily loss limit hit! P&L: ₹{self.daily_pnl:,.0f}")
            return False
        return True
    
    def scan_and_trade(self):
        """Main trading loop - scan market and execute trades"""
        if not self.is_trading_hours():
            print(f"⏰ {datetime.now().strftime('%H:%M')} - Outside trading hours")
            return
        
        if not self.check_daily_loss_limit():
            return
        
        # === RISK GOVERNOR CHECK ===
        if not self.risk_governor.is_trading_allowed():
            print(self.risk_governor.get_status())
            return
        
        self._scanning = True  # Suppress real-time dashboard during scan
        _cycle_start = time.time()
        print(f"\n{'='*80}")
        print(f"🔍 SCAN CYCLE @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(self.risk_governor.get_status())
        self._rejected_this_cycle = set()  # Reset rejected symbols for new scan
        
        # === HOT WATCHLIST: Clean stale entries & display ===
        try:
            from options_trader import get_hot_watchlist, cleanup_stale_watchlist
            cleanup_stale_watchlist(max_age_minutes=20)
            _watchlist = get_hot_watchlist()
            if _watchlist:
                print(f"\n🔥 HOT WATCHLIST ({len(_watchlist)} stocks warming up):")
                for _ws, _wd in sorted(_watchlist.items(), key=lambda x: x[1].get('score', 0), reverse=True):
                    print(f"   🔥 {_ws}: Score {_wd.get('score', 0):.0f} | {_wd.get('direction', '?')} | Conviction {_wd.get('directional_strength', 0):.0f}/8 | Seen {_wd.get('cycle_count', 1)}x")
        except Exception as e:
            print(f"   ⚠️ Watchlist check error: {e}")
        
        # CHECK AND UPDATE EXISTING TRADES (target/stoploss hits)
        trade_updates = self.tools.check_and_update_trades()
        if trade_updates:
            print(f"\n📊 TRADE UPDATES:")
            for update in trade_updates:
                emoji = "✅" if update['result'] == 'TARGET_HIT' else "❌"
                print(f"   {emoji} {update['symbol']}: {update['result']}")
                print(f"      Entry: ₹{update['entry']:.2f} → Exit: ₹{update['exit']:.2f}")
                print(f"      P&L: ₹{update['pnl']:+,.2f}")
                with self._pnl_lock:
                    self.daily_pnl += update['pnl']
                    self.capital += update['pnl']  # Also update capital (was missing)
        
        # Show current active positions
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        if active_trades:
            print(f"\n📂 ACTIVE POSITIONS ({len(active_trades)}):")
            for t in active_trades:
                print(f"   • {t['symbol']}: {t['side']} {t['quantity']} @ ₹{t['avg_price']:.2f}")
                print(f"     SL: ₹{t['stop_loss']:.2f} | Target: ₹{t.get('target', 0):.2f}")
        
        # Reset agent for fresh analysis each time
        self.reset_agent()
        
        try:
            # === MARKET SCANNER: Discover movers outside fixed universe ===
            try:
                scan_result = self.market_scanner.scan(existing_universe=APPROVED_UNIVERSE)
                print(self.market_scanner.format_scan_summary(scan_result))
                # Collect wild-card symbols to merge into data pipeline
                self._wildcard_symbols = [f"NSE:{w.symbol}" for w in scan_result.wildcards]
                # Store scanner scores for wildcard gate bypass
                self._wildcard_scores = {f"NSE:{w.symbol}": w.score for w in scan_result.wildcards}
                self._wildcard_change = {f"NSE:{w.symbol}": w.change_pct for w in scan_result.wildcards}
                # === DYNAMIC LOT SIZES: ensure every F&O stock has correct lot size ===
                update_fno_lot_sizes(self.market_scanner.get_lot_map())
            except Exception as e:
                print(f"⚠️ Scanner error (non-fatal): {e}")
                scan_result = None
                self._wildcard_symbols = []
                self._wildcard_scores = {}
                self._wildcard_change = {}
            
            # Merge wild-cards into scan universe for this cycle
            # With FULL_FNO_SCAN enabled, we scan ALL F&O stocks (not just curated + wildcards)
            # Pre-filter by change% to only run expensive indicators on movers
            scan_universe = list(APPROVED_UNIVERSE)
            _full_scan_mode = FULL_FNO_SCAN.get('enabled', False)
            
            if _full_scan_mode:
                # === FULL F&O UNIVERSE SCAN ===
                try:
                    _all_fo = self.market_scanner.get_all_fo_symbols()
                    _all_fo_syms = set(_all_fo)
                    
                    # Pre-filter: use scanner results to only include stocks with meaningful movement
                    _min_change = FULL_FNO_SCAN.get('min_change_pct_filter', 0.5)
                    _max_indicator_stocks = FULL_FNO_SCAN.get('max_indicator_stocks', 80)
                    
                    # Start with all curated stocks (always scan these)
                    _full_universe = set(APPROVED_UNIVERSE)
                    
                    # Add all scanner results that pass the pre-filter
                    if scan_result and hasattr(self.market_scanner, '_all_results'):
                        _moving_stocks = sorted(
                            self.market_scanner._all_results,
                            key=lambda r: abs(r.change_pct),
                            reverse=True
                        )
                        for _r in _moving_stocks:
                            if abs(_r.change_pct) >= _min_change or _r.volume > 0:
                                _full_universe.add(_r.nse_symbol)
                            if len(_full_universe) >= _max_indicator_stocks:
                                break
                    
                    scan_universe = list(_full_universe)
                    _skipped = len(_all_fo) - len(scan_universe)
                    print(f"   📡 FULL F&O SCAN: {len(scan_universe)} stocks passing filter (skipped {_skipped} flat/illiquid from {len(_all_fo)} total)")
                    
                except Exception as _e:
                    _all_fo_syms = set()
                    print(f"   ⚠️ Full scan fallback to curated: {_e}")
                    for ws in self._wildcard_symbols:
                        if ws not in scan_universe:
                            scan_universe.append(ws)
            else:
                # === CLASSIC MODE: curated + wildcards ===
                for ws in self._wildcard_symbols:
                    if ws not in scan_universe:
                        scan_universe.append(ws)
                
                try:
                    _all_fo = self.market_scanner.get_all_fo_symbols()
                    _all_fo_syms = set(_all_fo)
                    if len(self._wildcard_symbols) > 0:
                        print(f"   📡 Scan universe: {len(APPROVED_UNIVERSE)} curated + {len(self._wildcard_symbols)} wildcards = {len(scan_universe)} total (from {len(_all_fo)} F&O)")
                except Exception as _e:
                    _all_fo_syms = set()
                    print(f"   ⚠️ Could not get F&O universe: {_e}")
            
            # Get fresh market data for fixed universe + wild-cards
            market_data = self.tools.get_market_data(scan_universe)
            
            # Get volume analysis for EOD predictions
            volume_analysis = self.tools.get_volume_analysis(scan_universe)
            
            # Format ENHANCED data for prompt
            data_summary = []
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'ltp' in data:
                    # Get volume data for this symbol
                    vol_data = volume_analysis.get(symbol, {})
                    eod_pred = vol_data.get('eod_prediction', 'N/A')
                    order_flow = vol_data.get('order_flow', 'N/A')
                    buy_ratio = vol_data.get('buy_ratio', 50)
                    
                    line = f"""
{symbol}:
  Price: ₹{data['ltp']:.2f} | Change: {data.get('change_pct', 0):.2f}% | Trend: {data.get('trend', 'N/A')}
  RSI: {data.get('rsi_14', 50):.0f} | ATR: ₹{data.get('atr_14', 0):.2f} | ADX: {data.get('adx', 20):.0f}
  VWAP: ₹{data.get('vwap', 0):.2f} ({data.get('price_vs_vwap', 'N/A')}) Slope: {data.get('vwap_slope', 'FLAT')}
  EMA9: ₹{data.get('ema_9', 0):.2f} | EMA21: ₹{data.get('ema_21', 0):.2f} | Regime: {data.get('ema_regime', 'N/A')}
  ORB: H=₹{data.get('orb_high', 0):.2f} L=₹{data.get('orb_low', 0):.2f} → {data.get('orb_signal', 'N/A')} (Str:{data.get('orb_strength', 0):.1f}%) Hold:{data.get('orb_hold_candles', 0)}
  Volume: {data.get('volume_regime', 'N/A')} ({data.get('volume_vs_avg', 1.0):.1f}x avg) | Order Flow: {order_flow} | Buy%: {buy_ratio}%
  HTF: {data.get('htf_trend', 'N/A')} ({data.get('htf_alignment', 'N/A')}) | Chop: {'⚠️YES' if data.get('chop_zone', False) else 'NO'}
  Accel: FollowThru:{data.get('follow_through_candles', 0)} RangeExp:{data.get('range_expansion_ratio', 0):.1f} VWAPSteep:{'Y' if data.get('vwap_slope_steepening', False) else 'N'}
  Support: ₹{data.get('support_1', 0):.2f} / ₹{data.get('support_2', 0):.2f}
  Resistance: ₹{data.get('resistance_1', 0):.2f} / ₹{data.get('resistance_2', 0):.2f}
  EOD Prediction: {eod_pred}"""
                    data_summary.append(line)
            
            # Sort by absolute change to show active stocks first
            sorted_data = sorted(market_data.items(), 
                                key=lambda x: abs(x[1].get('change_pct', 0)) if isinstance(x[1], dict) else 0,
                                reverse=True)
            
            # === SCORE ALL F&O stocks (single-pass — cached for trade-time reuse) ===
            # Scores ALL stocks in sorted_data. Cached decisions are passed to
            # place_option_order / place_credit_spread / place_debit_spread so they
            # DON'T re-score. Only microstructure (bid-ask/OI) is fetched at trade time.
            _pre_scores = {}       # symbol → score (for display / GPT prompt)
            _cycle_decisions = {}  # symbol → {decision, direction, market_data}
            try:
                from options_trader import get_intraday_scorer, IntradaySignal
                _scorer = get_intraday_scorer()
                for _sym, _d in sorted_data:
                    if not isinstance(_d, dict) or 'ltp' not in _d:
                        continue
                    try:
                        _sig = IntradaySignal(
                            symbol=_sym,
                            orb_signal=_d.get('orb_signal', 'INSIDE_ORB'),
                            vwap_position=_d.get('price_vs_vwap', _d.get('vwap_position', 'AT_VWAP')),
                            vwap_trend=_d.get('vwap_slope', _d.get('vwap_trend', 'FLAT')),
                            ema_regime=_d.get('ema_regime', 'NORMAL'),
                            volume_regime=_d.get('volume_regime', 'NORMAL'),
                            rsi=_d.get('rsi_14', 50.0),
                            price_momentum=_d.get('momentum_15m', 0.0),
                            htf_alignment=_d.get('htf_alignment', 'NEUTRAL'),
                            chop_zone=_d.get('chop_zone', False),
                            follow_through_candles=_d.get('follow_through_candles', 0),
                            range_expansion_ratio=_d.get('range_expansion_ratio', 0.0),
                            vwap_slope_steepening=_d.get('vwap_slope_steepening', False),
                            atr=_d.get('atr_14', 0.0)
                        )
                        # S3 FIX: Don't bias scorer with mechanical change_pct direction
                        # Let the scorer determine direction from its own signal analysis
                        # change_pct is backward-looking noise, not an analytical view
                        _dir = None
                        _dec = _scorer.score_intraday_signal(_sig, market_data=_d, caller_direction=_dir)
                        _pre_scores[_sym] = _dec.confidence_score
                        # Cache full decision for trade-time reuse (no re-scoring)
                        _cycle_decisions[_sym] = {
                            'decision': _dec,
                            'direction': _dir,
                            'score': _dec.confidence_score,
                        }
                    except Exception:
                        pass
                
                # Attach cached decisions to tools layer for trade-time reuse
                self.tools._cached_cycle_decisions = _cycle_decisions
                
                # Log score distribution
                _scored_above_49 = sum(1 for s in _pre_scores.values() if s >= 49)
                _scored_above_45 = sum(1 for s in _pre_scores.values() if s >= 45)
                print(f"   📊 SCORED {len(_pre_scores)} F&O stocks: {_scored_above_49} score ≥49, {_scored_above_45} score ≥45")
                _top10 = sorted(_pre_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                if _top10:
                    _top10_str = " | ".join(f"{s.replace('NSE:', '')}={v:.0f}" for s, v in _top10)
                    print(f"   🏆 TOP 10: {_top10_str}")
            except Exception as _e:
                print(f"   ⚠️ Scoring failed: {_e}")
                self.tools._cached_cycle_decisions = {}
            
            # Create a quick summary of all stocks for scanning with EOD predictions
            quick_scan = []
            eod_opportunities = []
            regime_signals = []
            
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data:
                    chg = data.get('change_pct', 0)
                    rsi = data.get('rsi_14', 50)
                    trend = data.get('trend', 'N/A')
                    
                    # Get regime detection signals
                    vwap_slope = data.get('vwap_slope', 'FLAT')
                    price_vs_vwap = data.get('price_vs_vwap', 'AT_VWAP')
                    ema_regime = data.get('ema_regime', 'NORMAL')
                    orb_signal = data.get('orb_signal', 'INSIDE_ORB')
                    orb_strength = data.get('orb_strength', 0)
                    volume_regime = data.get('volume_regime', 'NORMAL')
                    
                    # Get CHOP filter signals
                    chop_zone = data.get('chop_zone', False)
                    chop_reason = data.get('chop_reason', '')
                    
                    # Get HTF alignment signals
                    htf_trend = data.get('htf_trend', 'NEUTRAL')
                    htf_ema_slope = data.get('htf_ema_slope', 'FLAT')
                    htf_alignment = data.get('htf_alignment', 'NEUTRAL')
                    
                    # Get volume analysis
                    vol_data = volume_analysis.get(symbol, {})
                    order_flow = vol_data.get('order_flow', 'BALANCED')
                    eod_pred = vol_data.get('eod_prediction', 'NEUTRAL')
                    eod_conf = vol_data.get('eod_confidence', 'LOW')
                    trade_signal = vol_data.get('trade_signal', 'NO_SIGNAL')
                    
                    # Flag potential setups
                    setup = ""
                    
                    # ======= CHOP FILTER - BLOCK ALL TRADES IN CHOP ZONE =======
                    if chop_zone:
                        setup = f"⚠️CHOP-ZONE({chop_reason})"
                        if symbol in self._wildcard_symbols:
                            print(f"   ⭐ WILDCARD CHOP-BLOCKED: {symbol} — {chop_reason}")
                        quick_scan.append(f"{symbol}: {chg:+.2f}% RSI:{rsi:.0f} {trend} {setup}")
                        continue  # Skip further analysis for this symbol
                    
                    # ======= HTF ALIGNMENT CHECK =======
                    # Determine intended trade direction based on signals
                    intended_direction = None
                    htf_blocked = False
                    
                    # REGIME-BASED SETUPS (highest priority)
                    # ORB trades - only once per direction per symbol per day
                    if orb_signal == "BREAKOUT_UP" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "BUY"
                        # HTF Check: Block BUY if HTF is BEARISH (unless explosive volume)
                        if htf_trend == "BEARISH" and htf_ema_slope == "FALLING" and volume_regime != "EXPLOSIVE":
                            htf_blocked = True
                            setup = "⛔HTF-BEAR-BLOCKS-BUY"
                        elif self._is_orb_trade_allowed(symbol, "UP"):
                            setup = "🚀ORB-BREAKOUT-BUY"
                            regime_signals.append(f"  🚀 {symbol}: ORB↑ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BULL | HTF:{htf_trend}")
                        else:
                            setup = "⛔ORB-UP-ALREADY-TAKEN"
                    elif orb_signal == "BREAKOUT_DOWN" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "SELL"
                        # HTF Check: Block SELL if HTF is BULLISH (unless explosive volume)
                        if htf_trend == "BULLISH" and htf_ema_slope == "RISING" and volume_regime != "EXPLOSIVE":
                            htf_blocked = True
                            setup = "⛔HTF-BULL-BLOCKS-SHORT"
                        elif self._is_orb_trade_allowed(symbol, "DOWN"):
                            setup = "🔻ORB-BREAKOUT-SHORT"
                            regime_signals.append(f"  🔻 {symbol}: ORB↓ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BEAR")
                        else:
                            setup = "⛔ORB-DOWN-ALREADY-TAKEN"
                    elif ema_regime == "COMPRESSED" and volume_regime in ["HIGH", "EXPLOSIVE"]:
                        setup = "💥SQUEEZE-PENDING"
                        regime_signals.append(f"  💥 {symbol}: EMA SQUEEZE + High Volume - BREAKOUT IMMINENT | HTF:{htf_trend}")
                    elif price_vs_vwap == "ABOVE_VWAP" and vwap_slope == "RISING" and rsi < 60:
                        # HTF check for VWAP trend buy
                        if htf_trend == "BEARISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-BUY-HTF-CONFLICT"
                        else:
                            setup = "📈VWAP-TREND-BUY"
                    elif price_vs_vwap == "BELOW_VWAP" and vwap_slope == "FALLING" and rsi > 40:
                        # HTF check for VWAP trend short
                        if htf_trend == "BULLISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-SHORT-HTF-CONFLICT"
                        else:
                            setup = "📉VWAP-TREND-SHORT"
                    # Standard RSI setups (also check HTF)
                    elif rsi < 30:
                        if htf_trend == "BEARISH":
                            setup = "⚠️OVERSOLD-HTF-BEAR"  # Weaker signal
                        else:
                            setup = "⚡OVERSOLD-BUY"
                    elif rsi > 70:
                        if htf_trend == "BULLISH":
                            setup = "⚠️OVERBOUGHT-HTF-BULL"  # Weaker signal
                        else:
                            setup = "⚡OVERBOUGHT-SHORT"
                    elif trade_signal == "BUY_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-BUY ({order_flow})"
                        eod_opportunities.append(f"  🟢 {symbol}: EOD↑ - {order_flow}, conf:{eod_conf} | HTF:{htf_trend}")
                    elif trade_signal == "SHORT_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-SHORT ({order_flow})"
                        eod_opportunities.append(f"  🔴 {symbol}: EOD↓ - {order_flow}, conf:{eod_conf} | HTF:{htf_trend}")
                    elif chg < -1.5 and rsi > 45:
                        setup = "📉WEAK-SHORT"
                    elif chg > 1 and rsi < 55:
                        setup = "📈STRONG-BUY"
                    elif chg < 0 and rsi < 35:
                        setup = "🔄BOUNCE-BUY"
                    
                    # Mark if already in active trades
                    if self.tools.is_symbol_in_active_trades(symbol):
                        setup = "🔒ALREADY HOLDING"
                    
                    # Include CHOP and HTF status in scan output
                    htf_icon = "🐂" if htf_trend == "BULLISH" else "🐻" if htf_trend == "BEARISH" else "➖"
                    fno_tag = "[F&O]" if symbol in FNO_CONFIG.get('prefer_options_for', []) else ""
                    # Include pre-computed intraday score for F&O stocks
                    _score_tag = ""
                    if symbol in _pre_scores:
                        _s = _pre_scores[symbol]
                        _score_tag = f" S:{_s:.0f}" + ("✅" if _s >= 49 else "⚠️" if _s >= 40 else "❌")
                    quick_scan.append(f"{symbol}{fno_tag}: {chg:+.2f}% RSI:{rsi:.0f} {trend} ORB:{orb_signal} Vol:{volume_regime} HTF:{htf_icon}{_score_tag} {setup}")
            
            # Print regime signals
            if regime_signals:
                print(f"\n🎯 REGIME SIGNALS (HIGH PRIORITY):")
                for sig in regime_signals[:5]:
                    print(sig)
            
            # Print EOD opportunities
            if eod_opportunities:
                print(f"\n📊 EOD VOLUME ANALYSIS:")
                for opp in eod_opportunities[:5]:  # Top 5
                    print(opp)
            
            # Get list of symbols already in positions
            active_symbols = [t['symbol'] for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            # Get risk governor status for prompt
            risk_status = self.risk_governor.state
            trades_remaining = self.risk_governor.limits.max_trades_per_day - risk_status.trades_today
            
            # Check trade permission before prompting agent
            active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            can_trade_check = self.risk_governor.can_trade_general(
                active_positions=active_positions
            )
            
            if not can_trade_check.allowed:
                print(f"\n⚠️ Trading blocked: {can_trade_check.reason}")
                if can_trade_check.warnings:
                    for w in can_trade_check.warnings:
                        print(f"   ⚠️ {w}")
                return
            
            # Update correlation guard with current positions
            self.correlation_guard.update_positions(active_positions)
            corr_exposure = self.correlation_guard.get_exposure_summary()
            print(f"\n{corr_exposure}")
            
            # Get reconciliation status
            recon_can_trade, recon_reason = self.position_recon.can_trade()
            recon_state = self.position_recon.state.value
            
            # Get data health status
            halted_symbols = self.data_health_gate.get_halted_symbols()
            
            # Display reconciliation and health status
            print(f"\n🔄 RECONCILIATION: {recon_state} - {'✅ Can Trade' if recon_can_trade else '❌ ' + recon_reason}")
            if halted_symbols:
                print(f"🛡️ DATA HEALTH: ⚠️ {len(halted_symbols)} symbols halted: {', '.join(halted_symbols[:5])}")
            else:
                print(f"🛡️ DATA HEALTH: ✅ All symbols healthy")
            
            # Build F&O opportunity list — ONLY stocks that actually exist in NFO instruments
            fno_prefer = FNO_CONFIG.get('prefer_options_for', [])
            fno_prefer_set = set(fno_prefer) | set(self._wildcard_symbols) | _all_fo_syms
            # Filter to ONLY NFO-verified stocks (scanner loaded from actual instruments)
            fno_nfo_verified = _all_fo_syms  # Only symbols actually in NFO instruments file
            fno_opportunities = []
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data and symbol in fno_prefer_set and symbol in fno_nfo_verified:
                    if self.tools.is_symbol_in_active_trades(symbol):
                        continue
                    if data.get('chop_zone', False):
                        continue
                    
                    # Tier-2 / wild-card gate: only trade when stock shows clear directional trend
                    is_tier2 = symbol in TIER_2_OPTIONS
                    is_wildcard = symbol in self._wildcard_symbols
                    if is_tier2 or is_wildcard:
                        trend_state = data.get('trend', 'SIDEWAYS')
                        orb = data.get('orb_signal', 'INSIDE_ORB')
                        vol = data.get('volume_regime', 'NORMAL')
                        chg_pct = abs(data.get('change_pct', 0))
                        # Tier-2 requires: clear trend OR ORB breakout with volume
                        tier2_trending = trend_state in ('BULLISH', 'STRONG_BULLISH', 'BEARISH', 'STRONG_BEARISH')
                        tier2_orb = orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and vol in ('HIGH', 'EXPLOSIVE')
                        
                        # WILDCARD SCANNER TRUST: if scanner scored this highly, 
                        # trust the momentum signal even without clear trend on 5min candles
                        # Strong scanner score (≥25) OR big move (≥2.5%) with volume = let it through
                        wc_score = self._wildcard_scores.get(symbol, 0)
                        wc_change = abs(self._wildcard_change.get(symbol, 0))
                        scanner_trust = is_wildcard and (
                            wc_score >= 25 or  # High scanner composite score
                            (wc_change >= 2.5 and vol in ('HIGH', 'EXPLOSIVE')) or  # Big move + volume
                            (chg_pct >= 3.0)  # 3%+ intraday move = real momentum
                        )
                        
                        if not (tier2_trending or tier2_orb or scanner_trust):
                            if is_wildcard:
                                print(f"   ⭐ WILDCARD FILTERED: {symbol} — trend={trend_state} orb={orb} vol={vol} scanner_score={wc_score:.0f} chg={wc_change:+.1f}% (need trend/ORB/score≥25/chg≥2.5%)")
                            continue  # Skip Tier-2 stocks without clear trend/breakout
                    
                    setup_type = None
                    direction = None
                    orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                    vol_reg = data.get('volume_regime', 'NORMAL')
                    ema_reg = data.get('ema_regime', 'NORMAL')
                    htf_a = data.get('htf_alignment', 'NEUTRAL')
                    rsi = data.get('rsi_14', 50)
                    pvw = data.get('price_vs_vwap', 'AT_VWAP')
                    vs = data.get('vwap_slope', 'FLAT')
                    
                    if orb_sig == "BREAKOUT_UP" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "ORB_BREAKOUT"
                        direction = "BUY"
                    elif orb_sig == "BREAKOUT_DOWN" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "ORB_BREAKOUT"
                        direction = "SELL"
                    elif pvw == "ABOVE_VWAP" and vs == "RISING" and ema_reg in ["EXPANDING", "COMPRESSED"]:
                        setup_type = "VWAP_TREND"
                        direction = "BUY"
                    elif pvw == "BELOW_VWAP" and vs == "FALLING" and ema_reg in ["EXPANDING", "COMPRESSED"]:
                        setup_type = "VWAP_TREND"
                        direction = "SELL"
                    elif pvw == "ABOVE_VWAP" and vs == "RISING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "MOMENTUM"
                        direction = "BUY"
                    elif pvw == "BELOW_VWAP" and vs == "FALLING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "MOMENTUM"
                        direction = "SELL"
                    elif ema_reg == "COMPRESSED" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "EMA_SQUEEZE"
                        direction = "BUY" if data.get('change_pct', 0) > 0 else "SELL"
                    elif rsi < 30 and htf_a != "BEARISH_ALIGNED":
                        setup_type = "RSI_REVERSAL"
                        direction = "BUY"
                    elif rsi > 70 and htf_a != "BULLISH_ALIGNED":
                        setup_type = "RSI_REVERSAL"
                        direction = "SELL"
                    
                    # WILDCARD MOMENTUM SETUP: if no standard setup detected but wildcard has strong momentum
                    if not setup_type and is_wildcard:
                        wc_chg = self._wildcard_change.get(symbol, 0)
                        if abs(wc_chg) >= 2.0 and vol_reg in ['HIGH', 'EXPLOSIVE']:
                            setup_type = "WILDCARD_MOMENTUM"
                            direction = "BUY" if wc_chg > 0 else "SELL"
                    
                    # S4 FIX: VWAP GRIND SETUP — catches INSIDE_ORB stocks grinding with volume
                    # (e.g., HINDALCO -5.7% but never broke ORB range)
                    if not setup_type and orb_sig == "INSIDE_ORB" and vol_reg in ('HIGH', 'EXPLOSIVE'):
                        _grind_ltp = data.get('ltp', 0)
                        _grind_open = data.get('open', _grind_ltp)
                        _grind_vwap = data.get('vwap', 0)
                        _grind_move = abs((_grind_ltp - _grind_open) / _grind_open * 100) if _grind_open > 0 else 0
                        if _grind_move >= 0.8 and _grind_ltp < _grind_vwap and _grind_ltp < _grind_open:
                            setup_type = "VWAP_GRIND"
                            direction = "SELL"
                        elif _grind_move >= 0.8 and _grind_ltp > _grind_vwap and _grind_ltp > _grind_open:
                            setup_type = "VWAP_GRIND"
                            direction = "BUY"
                    
                    if setup_type and direction:
                        opt_type = "CE" if direction == "BUY" else "PE"
                        wc_tag = " [⭐WILDCARD]" if is_wildcard else ""
                        _fno_score = _pre_scores.get(symbol, 0)
                        _fno_score_tag = f" [Score:{_fno_score:.0f}]" if _fno_score > 0 else ""
                        # S5 FIX: Pre-scores don't include microstructure (15pts) because
                        # option_data isn't fetched during bulk scanning. Offset threshold
                        # by 12 (average micro contribution for liquid F&O stocks) to avoid
                        # filtering out stocks that would pass at trade time.
                        _micro_absent_offset = 12
                        if _fno_score > 0 and _fno_score + _micro_absent_offset < 40:
                            continue
                        fno_opportunities.append(
                            f"  🎯 {symbol}: {setup_type} → place_option_order(underlying=\"{symbol}\", direction=\"{direction}\", strike_selection=\"ATM\") [{opt_type}]{_fno_score_tag}{wc_tag}"
                        )
            
            # === PROACTIVE DEBIT SPREAD SCANNER ===
            # Scan for debit spread opportunities INDEPENDENTLY from the cascade
            # This is the key fix: debit spreads are no longer just a fallback
            debit_spread_placed = []
            try:
                from config import DEBIT_SPREAD_CONFIG
                if DEBIT_SPREAD_CONFIG.get('proactive_scan', False) and DEBIT_SPREAD_CONFIG.get('enabled', False):
                    now_time = datetime.now().time()
                    debit_cutoff = datetime.strptime(DEBIT_SPREAD_CONFIG.get('no_entry_after', '14:30'), '%H:%M').time()
                    
                    if now_time < debit_cutoff:
                        proactive_min_score = DEBIT_SPREAD_CONFIG.get('proactive_scan_min_score', 68)
                        proactive_min_move = DEBIT_SPREAD_CONFIG.get('proactive_scan_min_move_pct', 1.5)
                        
                        # Find momentum setups that qualify for debit spreads
                        debit_candidates = []
                        for symbol, data in sorted_data:
                            if not isinstance(data, dict) or 'ltp' not in data:
                                continue
                            # Skip if already holding or in chop zone
                            if self.tools.is_symbol_in_active_trades(symbol):
                                continue
                            if data.get('chop_zone', False):
                                continue
                            # Skip if not F&O eligible
                            if symbol not in fno_prefer_set:
                                continue
                            
                            # Check move % — momentum plays need real movement
                            chg = abs(data.get('change_pct', 0))
                            if chg < proactive_min_move:
                                continue
                            
                            # Check follow-through (strongest winner signal)
                            ft = data.get('follow_through_candles', 0)
                            min_ft = DEBIT_SPREAD_CONFIG.get('min_follow_through_candles', 2)
                            if ft < min_ft:
                                continue
                            
                            # Check ADX
                            adx = data.get('adx_14', data.get('adx', 0))
                            min_adx = DEBIT_SPREAD_CONFIG.get('min_adx', 28)
                            if adx > 0 and adx < min_adx:
                                continue
                            
                            # Determine direction from price action
                            change_pct = data.get('change_pct', 0)
                            direction = "BUY" if change_pct > 0 else "SELL"
                            
                            # Check trend continuation — move must be in direction of trade
                            orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                            vol_reg = data.get('volume_regime', 'NORMAL')
                            
                            # Priority scoring for debit spread candidates
                            ds_priority = 0
                            ds_priority += ft * 10  # Follow-through is king
                            ds_priority += min(adx, 50)  # ADX contribution
                            ds_priority += chg * 5  # Bigger move = better
                            if orb_sig in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
                                ds_priority += 15
                            if vol_reg in ("HIGH", "EXPLOSIVE"):
                                ds_priority += 10
                            
                            debit_candidates.append((symbol, data, direction, ds_priority))
                        
                        # Sort by priority and try top candidates
                        debit_candidates.sort(key=lambda x: x[3], reverse=True)
                        
                        max_debit_entries = 3  # Max 3 proactive debit spreads per scan cycle (was 2)
                        for symbol, data, direction, priority in debit_candidates[:max_debit_entries]:
                            try:
                                print(f"\n   🎯 PROACTIVE DEBIT SPREAD: Trying {symbol} ({direction}) — Priority: {priority:.0f}")
                                
                                # === PRE-FLIGHT LIQUIDITY CHECK ===
                                # Verify option chain has adequate OI + tight bid-ask BEFORE
                                # spending API calls on full debit spread creation
                                try:
                                    from options_trader import get_options_trader
                                    _ot = get_options_trader(
                                        kite=self.tools.kite,
                                        capital=getattr(self.tools, 'paper_capital', 500000),
                                        paper_mode=getattr(self.tools, 'paper_mode', True)
                                    )
                                    is_liquid, liq_reason = _ot.chain_fetcher.quick_check_option_liquidity(
                                        underlying=symbol,
                                        min_oi=DEBIT_SPREAD_CONFIG.get('min_oi', 500),
                                        max_bid_ask_pct=DEBIT_SPREAD_CONFIG.get('max_spread_bid_ask_pct', 5.0)
                                    )
                                    if not is_liquid:
                                        print(f"   ❌ LIQUIDITY PRE-CHECK FAILED for {symbol}: {liq_reason}")
                                        continue
                                    print(f"   ✅ Liquidity OK: {liq_reason}")
                                except Exception as liq_e:
                                    print(f"   ⚠️ Liquidity check skipped (error: {liq_e}) — proceeding anyway")
                                
                                result = self.tools.place_debit_spread(
                                    underlying=symbol,
                                    direction=direction,
                                    rationale=f"Proactive debit spread: FT={data.get('follow_through_candles', 0)} ADX={data.get('adx_14', 0):.0f} Move={data.get('change_pct', 0):+.1f}%",
                                    pre_fetched_market_data=data
                                )
                                if result.get('success'):
                                    print(f"   ✅ PROACTIVE DEBIT SPREAD PLACED on {symbol}!")
                                    debit_spread_placed.append(symbol)
                                else:
                                    print(f"   ℹ️ Debit spread not viable for {symbol}: {result.get('error', 'unknown')}")
                            except Exception as e:
                                print(f"   ⚠️ Debit spread attempt failed for {symbol}: {e}")
                        
                        if debit_spread_placed:
                            print(f"\n   🚀 PROACTIVE DEBIT SPREADS PLACED: {', '.join(debit_spread_placed)}")
                        elif debit_candidates:
                            print(f"\n   ℹ️ {len(debit_candidates)} debit spread candidates found, none viable this cycle")
            except Exception as e:
                print(f"   ⚠️ Proactive debit spread scan error: {e}")
            
            # === PROACTIVE IRON CONDOR SCANNER (INDEX + STOCK) ===
            # Scan NIFTY/BANKNIFTY for IC opportunities (weekly expiry, 0-2 DTE)
            # Also check recently rejected stocks with low scores
            ic_placed = []
            try:
                from config import IRON_CONDOR_CONFIG, IC_INDEX_SYMBOLS
                if IRON_CONDOR_CONFIG.get('enabled', False) and IRON_CONDOR_CONFIG.get('proactive_scan', False):
                    
                    # === DTE PRE-CHECK (once per day): Skip IC scan if no expiry within DTE limits ===
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    if not hasattr(self, '_ic_eligible_date') or self._ic_eligible_date != today_str:
                        self._ic_eligible_today = False
                        self._ic_eligible_date = today_str
                        try:
                            from options_trader import get_options_trader as _get_ot, ExpirySelection
                            _ot_check = _get_ot(
                                kite=self.tools.kite,
                                capital=getattr(self.tools, 'paper_capital', 500000),
                                paper_mode=getattr(self.tools, 'paper_mode', True)
                            )
                            idx_mode = IRON_CONDOR_CONFIG.get('index_mode', {})
                            stk_mode = IRON_CONDOR_CONFIG.get('stock_mode', {})
                            idx_max_dte = idx_mode.get('max_dte', 2)
                            stk_max_dte = stk_mode.get('max_dte', 15)
                            today_date = datetime.now().date()
                            
                            for idx_sym in IC_INDEX_SYMBOLS:
                                try:
                                    idx_expiry_sel = ExpirySelection[idx_mode.get('prefer_expiry', 'CURRENT_WEEK')]
                                    idx_expiry = _ot_check.chain_fetcher.get_nearest_expiry(idx_sym, idx_expiry_sel)
                                    if idx_expiry:
                                        from datetime import date as _date
                                        exp_d = idx_expiry if isinstance(idx_expiry, _date) and not isinstance(idx_expiry, datetime) else idx_expiry.date() if hasattr(idx_expiry, 'date') else idx_expiry
                                        idx_dte = (exp_d - today_date).days
                                        if idx_dte <= idx_max_dte:
                                            self._ic_eligible_today = True
                                            print(f"   🦅 IC eligible today: {idx_sym} expiry {idx_expiry} (DTE={idx_dte}, max={idx_max_dte})")
                                            break
                                except Exception:
                                    pass
                            
                            if not self._ic_eligible_today:
                                try:
                                    stk_expiry_sel = ExpirySelection[stk_mode.get('prefer_expiry', 'CURRENT_MONTH')]
                                    stk_expiry = _ot_check.chain_fetcher.get_nearest_expiry("NSE:RELIANCE", stk_expiry_sel)
                                    if stk_expiry:
                                        from datetime import date as _date
                                        exp_d = stk_expiry if isinstance(stk_expiry, _date) and not isinstance(stk_expiry, datetime) else stk_expiry.date() if hasattr(stk_expiry, 'date') else stk_expiry
                                        stk_dte = (exp_d - today_date).days
                                        if stk_dte <= stk_max_dte:
                                            self._ic_eligible_today = True
                                            print(f"   🦅 IC eligible today: Stocks expiry {stk_expiry} (DTE={stk_dte}, max={stk_max_dte})")
                                except Exception:
                                    pass
                            
                            if not self._ic_eligible_today:
                                print(f"   🦅 IC SKIPPED TODAY: No expiry within DTE limits (index max={idx_max_dte}, stock max={stk_max_dte})")
                        except Exception as dte_err:
                            print(f"   ⚠️ IC DTE pre-check failed: {dte_err} — will skip IC scan")
                            self._ic_eligible_today = False
                    
                    # Only run IC scan if today has eligible expiry AND within time window
                    now_time = datetime.now().time()
                    ic_earliest = datetime.strptime(IRON_CONDOR_CONFIG.get('earliest_entry', '10:30'), '%H:%M').time()
                    ic_cutoff = datetime.strptime(IRON_CONDOR_CONFIG.get('no_entry_after', '12:30'), '%H:%M').time()
                    
                    if self._ic_eligible_today and ic_earliest <= now_time <= ic_cutoff:
                        from options_trader import get_options_trader
                        _ot = get_options_trader(
                            kite=self.tools.kite,
                            capital=getattr(self.tools, 'paper_capital', 500000),
                            paper_mode=getattr(self.tools, 'paper_mode', True)
                        )
                        
                        # --- INDEX IC SCAN (primary — weekly expiry, best profit) ---
                        for idx_symbol in IC_INDEX_SYMBOLS:
                            try:
                                if self.tools.is_symbol_in_active_trades(idx_symbol):
                                    continue
                                
                                # Fetch index market data
                                idx_data_raw = self.tools.get_market_data([idx_symbol])
                                idx_data = idx_data_raw.get(idx_symbol, {})
                                if not idx_data or not isinstance(idx_data, dict):
                                    continue
                                
                                # Check if index is range-bound
                                idx_change = abs(idx_data.get('change_pct', 99))
                                idx_rsi = idx_data.get('rsi_14', 50)
                                max_move = IRON_CONDOR_CONFIG.get('max_intraday_move_pct', 1.2)
                                rsi_range = IRON_CONDOR_CONFIG.get('prefer_rsi_range', [38, 62])
                                
                                if idx_change > max_move:
                                    print(f"   🦅 IC SKIP {idx_symbol}: moved {idx_change:.1f}% (>{max_move}%)")
                                    continue
                                
                                # Assign a synthetic "choppy" score for IC (lower = choppier = better for IC)
                                # Use: low change%, near-50 RSI, flat VWAP = lower score
                                ic_score = 30  # Default choppy score
                                if idx_change < 0.3:
                                    ic_score -= 5  # Very tight range
                                if 45 <= idx_rsi <= 55:
                                    ic_score -= 3  # Dead neutral RSI 
                                if idx_data.get('vwap_slope', 'FLAT') == 'FLAT':
                                    ic_score -= 2  # Flat VWAP
                                ic_score = max(15, min(45, ic_score))
                                
                                print(f"\n   🦅 INDEX IC SCAN: {idx_symbol} | Chg: {idx_change:.2f}% | RSI: {idx_rsi:.0f} | IC Score: {ic_score}")
                                
                                exec_result = self.tools.place_iron_condor(
                                    underlying=idx_symbol,
                                    rationale=f"Proactive index IC: flat range {idx_change:.1f}%, RSI {idx_rsi:.0f}",
                                    directional_score=ic_score,
                                    pre_fetched_market_data=idx_data
                                )
                                
                                if exec_result and exec_result.get('success'):
                                    print(f"   ✅ INDEX IC PLACED on {idx_symbol}!")
                                    ic_placed.append(idx_symbol)
                                else:
                                    print(f"   ℹ️ Index IC not viable for {idx_symbol}: {exec_result.get('error', 'creation failed')}")
                            except Exception as ie:
                                print(f"   ⚠️ Index IC scan error for {idx_symbol}: {ie}")
                        
                        # --- STOCK IC SCAN (SOPHISTICATED MULTI-FACTOR SCORING) ---
                        # Score each F&O stock on 7 IC-quality factors, rank, and pick best
                        if IRON_CONDOR_CONFIG.get('scan_rejected_stocks', True):
                            stock_ic_candidates = []
                            for symbol, data in sorted_data:
                                if not isinstance(data, dict) or 'ltp' not in data:
                                    continue
                                if self.tools.is_symbol_in_active_trades(symbol):
                                    continue
                                if symbol not in fno_prefer_set:
                                    continue
                                if symbol in IC_INDEX_SYMBOLS:
                                    continue  # Already handled above
                                
                                # === IC QUALITY SCORE (0-100) ===
                                # Higher = better IC candidate
                                ic_quality = 0
                                ic_reasons = []
                                
                                chg = abs(data.get('change_pct', 0))
                                rsi_val = data.get('rsi_14', 50)
                                adx_val = data.get('adx_14', data.get('adx', 25))
                                is_chop = data.get('chop_zone', False)
                                vwap_slope = data.get('vwap_slope', 'FLAT')
                                orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                                vol_regime = data.get('volume_regime', 'NORMAL')
                                vol_ratio = data.get('volume_vs_avg', 1.0)
                                ema_regime = data.get('ema_regime', 'NORMAL')
                                atr_14 = data.get('atr_14', 0)
                                ltp = data.get('ltp', 0)
                                range_exp = data.get('range_expansion_ratio', 0.5)
                                
                                # HARD FILTERS — skip immediately
                                if chg > max_move:
                                    continue
                                if orb_sig in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and data.get('orb_strength', 0) > 50:
                                    continue  # Strong directional breakout — not IC material
                                
                                # FACTOR 1: RANGE TIGHTNESS (0-20 pts)
                                # Tighter intraday range = better for IC
                                if chg < 0.3:
                                    ic_quality += 20; ic_reasons.append(f"TIGHT_RANGE({chg:.1f}%)")
                                elif chg < 0.6:
                                    ic_quality += 15; ic_reasons.append(f"NARROW_RANGE({chg:.1f}%)")
                                elif chg < 1.0:
                                    ic_quality += 8; ic_reasons.append(f"MOD_RANGE({chg:.1f}%)")
                                else:
                                    ic_quality += 3  # Still within max_move but wide
                                
                                # FACTOR 2: RSI NEUTRALITY (0-15 pts)
                                # 45-55 = dead center = perfect; 40-60 = good; outside = risky
                                rsi_mid_dist = abs(rsi_val - 50)
                                if rsi_mid_dist <= 5:
                                    ic_quality += 15; ic_reasons.append(f"RSI_DEAD_CENTER({rsi_val:.0f})")
                                elif rsi_mid_dist <= 10:
                                    ic_quality += 10
                                elif rsi_mid_dist <= 15:
                                    ic_quality += 5
                                else:
                                    ic_quality -= 5  # RSI trending — danger for IC
                                
                                # FACTOR 3: ADX LOW = NO TREND (0-15 pts)
                                # ADX < 20 = no trend (ideal); < 25 = weak trend; > 30 = strong trend (bad)
                                if adx_val < 18:
                                    ic_quality += 15; ic_reasons.append(f"NO_TREND(ADX={adx_val:.0f})")
                                elif adx_val < 22:
                                    ic_quality += 12; ic_reasons.append(f"WEAK_TREND(ADX={adx_val:.0f})")
                                elif adx_val < 26:
                                    ic_quality += 8
                                elif adx_val < 30:
                                    ic_quality += 3
                                else:
                                    ic_quality -= 5  # Strong trend — IC is risky
                                    ic_reasons.append(f"STRONG_TREND_WARNING(ADX={adx_val:.0f})")
                                
                                # FACTOR 4: VWAP SLOPE FLAT (0-10 pts)
                                if vwap_slope == 'FLAT':
                                    ic_quality += 10; ic_reasons.append("VWAP_FLAT")
                                elif vwap_slope in ('RISING', 'FALLING'):
                                    ic_quality -= 3  # Trending — not ideal
                                
                                # FACTOR 5: VOLUME DECLINING (0-10 pts) 
                                # Declining volume = dying momentum = perfect for IC
                                if vol_regime == 'LOW' or vol_ratio < 0.7:
                                    ic_quality += 10; ic_reasons.append(f"LOW_VOL({vol_ratio:.1f}x)")
                                elif vol_regime == 'NORMAL' and vol_ratio < 1.2:
                                    ic_quality += 6
                                elif vol_regime == 'HIGH':
                                    ic_quality += 0  # High volume could mean breakout
                                elif vol_regime == 'EXPLOSIVE':
                                    ic_quality -= 8  # Explosive = breakout imminent
                                    ic_reasons.append("EXPLOSIVE_VOL_DANGER")
                                
                                # FACTOR 6: EMA COMPRESSION (0-10 pts)
                                # Compressed EMAs = coiling for a move or stuck in range
                                if ema_regime == 'COMPRESSED':
                                    ic_quality += 10; ic_reasons.append("EMA_COMPRESSED")
                                elif ema_regime == 'NORMAL':
                                    ic_quality += 5
                                elif ema_regime == 'EXPANDING':
                                    ic_quality -= 3  # Expanding = trending
                                
                                # FACTOR 7: RANGE EXPANSION LOW (0-10 pts)
                                # Low range expansion = price not going anywhere
                                if range_exp < 0.2:
                                    ic_quality += 10; ic_reasons.append(f"LOW_EXPANSION({range_exp:.2f})")
                                elif range_exp < 0.35:
                                    ic_quality += 7
                                elif range_exp < 0.5:
                                    ic_quality += 3
                                else:
                                    ic_quality -= 3  # High expansion = trending
                                
                                # FACTOR 8: CHOP ZONE confirmation (0-10 pts)
                                if is_chop:
                                    ic_quality += 10; ic_reasons.append("CHOP_CONFIRMED")
                                
                                # === MINIMUM IC QUALITY GATE ===
                                min_ic_quality = IRON_CONDOR_CONFIG.get('min_ic_quality_score', 50)
                                if ic_quality < min_ic_quality:
                                    continue  # Not choppy enough for IC
                                
                                # Map ic_quality (0-100) to a directional_score (15-45) for IC
                                # Higher IC quality → lower directional score (more neutral)
                                mapped_dir_score = max(15, min(45, 45 - int((ic_quality - 50) * 0.6)))
                                
                                stock_ic_candidates.append((symbol, data, ic_quality, mapped_dir_score, ic_reasons))
                            
                            # Sort by IC quality score DESCENDING (best IC candidates first)
                            stock_ic_candidates.sort(key=lambda x: x[2], reverse=True)
                            
                            max_stock_ic = IRON_CONDOR_CONFIG.get('max_stock_ic_per_cycle', 2)
                            for symbol, data, ic_quality, mapped_score, reasons in stock_ic_candidates[:max_stock_ic]:
                                try:
                                    reasons_str = " | ".join(reasons[:5])
                                    print(f"\n   🦅 STOCK IC SCAN: {symbol} | Quality: {ic_quality}/100 | Chg: {abs(data.get('change_pct', 0)):.2f}% | RSI: {data.get('rsi_14', 50):.0f}")
                                    print(f"      IC Factors: {reasons_str}")
                                    exec_result = self.tools.place_iron_condor(
                                        underlying=symbol,
                                        rationale=f"IC Quality {ic_quality}/100: {reasons_str} | Move {abs(data.get('change_pct', 0)):.1f}% RSI {data.get('rsi_14', 50):.0f}",
                                        directional_score=mapped_score,
                                        pre_fetched_market_data=data
                                    )
                                    if exec_result and exec_result.get('success'):
                                        print(f"   ✅ STOCK IC PLACED on {symbol}! (Quality: {ic_quality})")
                                        ic_placed.append(symbol)
                                    else:
                                        print(f"   ℹ️ Stock IC not viable for {symbol}: {exec_result.get('error', 'creation failed')}")
                                except Exception as se:
                                    print(f"   ⚠️ Stock IC attempt failed for {symbol}: {se}")
                        
                        if ic_placed:
                            print(f"\n   🦅 IRON CONDORS PLACED: {', '.join(ic_placed)}")
                    else:
                        if now_time < ic_earliest:
                            pass  # Silent — too early, no need to print every cycle
                        else:
                            pass  # Silent — past cutoff
            except Exception as e:
                print(f"   ⚠️ Proactive IC scan error: {e}")
            
            # Select top detailed stocks for GPT (most active + those with setups)
            top_detail_symbols = [s for s, _ in sorted_data[:10] if isinstance(market_data.get(s), dict) and 'ltp' in market_data.get(s, {})]
            detailed_data = [line for line in data_summary if any(sym in line for sym in top_detail_symbols)]
            
            # Build scanner wild-card summary for GPT
            wildcard_info = ""
            if scan_result and scan_result.wildcards:
                wc_lines = []
                for w in scan_result.wildcards:
                    wc_lines.append(f"  ⭐ {w.nse_symbol}: {w.change_pct:+.2f}% ₹{w.ltp:.2f} [{w.category}] — OUTSIDE fixed universe, use place_option_order()")
                wildcard_info = "\n".join(wc_lines)
            
            # Build BROAD MARKET HEAT MAP from scanner (all 191 F&O stocks)
            broad_heat = ""
            try:
                broad_heat = self.market_scanner.get_broad_market_heat(
                    existing_universe=set(APPROVED_UNIVERSE)
                )
            except Exception:
                broad_heat = "Scanner heat map unavailable"
            
            # Ask agent to analyze market with FULL CONTEXT + GPT-5.2 reasoning
            # Compute market breadth for macro context
            _up_count = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) > 0.5)
            _down_count = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) < -0.5)
            _flat_count = len(sorted_data) - _up_count - _down_count
            _breadth = "BULLISH" if _up_count > _down_count * 1.5 else "BEARISH" if _down_count > _up_count * 1.5 else "MIXED"
            
            # Compute sector summary
            _sector_map = {
                'IT': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'KPITTECH', 'COFORGE', 'MPHASIS', 'PERSISTENT'],
                'BANKS': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'BANKBARODA', 'PNB', 'IDFCFIRSTB', 'INDUSINDBK', 'FEDERALBNK'],
                'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL', 'NMDC', 'NATIONALUM', 'HINDZINC', 'SAIL'],
                'PHARMA': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'AUROPHARMA', 'BIOCON', 'LUPIN'],
                'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY', 'BHARATFORG'],
                'ENERGY': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIENT'],
                'FMCG': ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO']
            }
            _sector_perf = []
            for _sec, _syms in _sector_map.items():
                _changes = [market_data.get(f"NSE:{s}", {}).get('change_pct', 0) for s in _syms if isinstance(market_data.get(f"NSE:{s}", {}), dict)]
                if _changes:
                    _avg = sum(_changes) / len(_changes)
                    _sector_perf.append(f"  {_sec}: {_avg:+.2f}% avg ({len([c for c in _changes if c > 0])}/{len(_changes)} ↑)")
            
            prompt = f"""ANALYZE → REASON → EXECUTE. Use your GPT-5.2 reasoning depth.

=== 🌐 MARKET BREADTH ===
Breadth: {_breadth} | Up: {_up_count} | Down: {_down_count} | Flat: {_flat_count}
Sector Performance:
{chr(10).join(_sector_perf) if _sector_perf else '  No sector data'}

=== ⚡ F&O READY SIGNALS (HIGHEST PRIORITY) ===
F&O stocks: {', '.join(fno_prefer)}
{chr(10).join(fno_opportunities) if fno_opportunities else 'No F&O setups right now - check CASH stocks below'}

=== 📡 WILD-CARD MOVERS ===
{wildcard_info if wildcard_info else 'No wild-card movers this cycle'}

=== 🌡️ BROAD MARKET HEAT MAP (Top 40 F&O Movers — ⭐NEW = outside curated list) ===
{broad_heat}

=== 📊 MARKET SCAN ({len(quick_scan)} stocks) ===
{chr(10).join(quick_scan[:50])}

=== 🔬 DETAILED TECHNICALS (Top 10 Movers) ===
{chr(10).join(detailed_data[:10])}

=== 🎯 REGIME SIGNALS ===
{chr(10).join(regime_signals) if regime_signals else 'No strong regime signals'}

=== 📊 EOD PREDICTIONS ===
{chr(10).join(eod_opportunities) if eod_opportunities else 'No EOD signals'}

=== � HOT WATCHLIST (scored 55+ but conviction-blocked — RE-EVALUATE these!) ===
{self._format_watchlist_for_prompt()}

=== �🔒 SKIP (already holding) ===
{', '.join(active_symbols) if active_symbols else 'None'}

=== ⚖️ CORRELATION EXPOSURE ===
{corr_exposure}

=== 🏥 SYSTEM HEALTH ===
Reconciliation: {recon_state} {'(CAN TRADE)' if recon_can_trade else '(BLOCKED: ' + recon_reason + ')'}
Data Health: {len(halted_symbols)} halted | {'Halted: ' + ', '.join(halted_symbols[:5]) if halted_symbols else 'All healthy'}

=== 💰 ACCOUNT ===
Capital: Rs{self.capital:,.0f} | Daily P&L: Rs{risk_status.daily_pnl:+,.0f} ({risk_status.daily_pnl_pct:+.2f}%)
Trades: {risk_status.trades_today}/{self.risk_governor.limits.max_trades_per_day} (Remaining: {trades_remaining})
W/L: {risk_status.wins_today}/{risk_status.losses_today} | Consec Losses: {risk_status.consecutive_losses}/{self.risk_governor.limits.max_consecutive_losses}

=== 🧠 YOUR TASK ===
1. Assess MARKET REGIME first (trending/range/mixed day, sector rotation)
2. ONLY pick from the '⚡ F&O READY SIGNALS' section above — these are pre-validated. Do NOT invent your own picks.
3. Look at the S: (Score) tags — ONLY pick stocks scoring ≥49✅. Stocks with ❌ WILL BE REJECTED.
4. Identify TOP 3 setups from the listed opportunities using CONFLUENCE SCORING
5. Check CONTRARIAN risks (chasing? extended? volume divergence?)
6. EXECUTE via tools — place_option_order(underlying, direction) for F&O, place_order() for cash
7. State your reasoning briefly: Setup | Score | Why

⚠️ CRITICAL RULES:
- Do NOT pick stocks outside the F&O READY SIGNALS list. They are NOT tradeable.
- Do NOT pick stocks scoring below 49. The scorer WILL block them.
- If no setups score ≥49, say 'NO TRADES' — do NOT force a trade.

RULES: F&O → place_option_order() | Cash → place_order() | SL 1% | Target 1.5% | Max 3 trades"""

            response = self.agent.run(prompt)
            print(f"\n📊 Agent response:\n{response[:300]}...")
            
            # === AUTO-RETRY: Detect trades mentioned but not placed ===
            # Collect scorer rejections from the tools layer
            if not hasattr(self, '_rejected_this_cycle'):
                self._rejected_this_cycle = set()
            if hasattr(self.tools, '_scorer_rejected_symbols'):
                # Don't permanently blacklist hot watchlist stocks — allow retry next cycle
                try:
                    from options_trader import get_hot_watchlist
                    _wl = get_hot_watchlist()
                    _wl_syms = {s.replace('NSE:', '') for s in _wl.keys()}
                    _hard_rejected = self.tools._scorer_rejected_symbols - _wl_syms
                    _soft_rejected = self.tools._scorer_rejected_symbols & _wl_syms
                    self._rejected_this_cycle.update(_hard_rejected)
                    if _hard_rejected:
                        print(f"   🚫 Scorer rejected (won't retry): {_hard_rejected}")
                    if _soft_rejected:
                        print(f"   🔥 On hot watchlist (will retry next cycle): {_soft_rejected}")
                except Exception:
                    self._rejected_this_cycle.update(self.tools._scorer_rejected_symbols)
                    if self.tools._scorer_rejected_symbols:
                        print(f"   🚫 Scorer rejected (won't retry): {self.tools._scorer_rejected_symbols}")
                self.tools._scorer_rejected_symbols.clear()
            
            if response and not self.agent.get_pending_approvals():
                # Check if response mentions trade symbols but no orders were placed
                import re
                mentioned_symbols = re.findall(r'NSE:([A-Z]+)', response)
                # Build active set from BOTH equity symbols (NSE:X) AND option underlying (NSE:X)
                active_symbols_set = set()
                for t in self.tools.paper_positions:
                    if t.get('status', 'OPEN') != 'OPEN':
                        continue
                    sym = t.get('symbol', '')
                    # Equity: NSE:INFY → INFY
                    if sym.startswith('NSE:'):
                        active_symbols_set.add(sym.replace('NSE:', ''))
                    # Options: NFO:ICICIBANK26FEB1410CE → check underlying field
                    underlying = t.get('underlying', '')
                    if underlying:
                        active_symbols_set.add(underlying.replace('NSE:', ''))
                # Include ALL F&O stocks in retry — scanner covers them all
                retry_eligible = set(APPROVED_UNIVERSE) | set(self._wildcard_symbols) | _all_fo_syms
                unplaced = [s for s in set(mentioned_symbols) 
                           if s not in active_symbols_set 
                           and f'NSE:{s}' in retry_eligible
                           and s not in self._rejected_this_cycle]
                
                if unplaced and len(unplaced) <= 5:
                    print(f"\n🔄 Detected {len(unplaced)} unplaced trades in response: {unplaced}")
                    fno_prefer_set = {s.replace('NSE:', '') for s in FNO_CONFIG.get('prefer_options_for', [])} | {s.replace('NSE:', '') for s in _all_fo_syms}                    
                    # Parse direction from GPT's response text for each symbol
                    def _parse_direction_from_response(text, symbol):
                        """Extract BUY/SELL direction for a symbol from GPT response text"""
                        import re
                        # Look for patterns like "NSE:PAYTM | ... | SELL" or "direction: SELL" near symbol
                        # Search within 200 chars of the symbol mention
                        patterns = [
                            rf'{symbol}[^{{}}]{{0,200}}(?:direction|side|action)[:\s]*["\']?(SELL|BUY|BEARISH|BULLISH)',
                            rf'{symbol}[^{{}}]{{0,200}}(SELL|SHORT|BEARISH|PUT|Breakdown|breakdown|fade)',
                            rf'{symbol}[^{{}}]{{0,200}}(BUY|LONG|BULLISH|CALL|Breakout|breakout|momentum up)',
                            rf'(SELL|SHORT|BEARISH)[^{{}}]{{0,100}}{symbol}',
                            rf'(BUY|LONG|BULLISH)[^{{}}]{{0,100}}{symbol}',
                        ]
                        for i, pattern in enumerate(patterns):
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                found = match.group(1) if i > 0 else match.group(1)
                                if found.upper() in ('SELL', 'SHORT', 'BEARISH', 'PUT', 'BREAKDOWN', 'FADE'):
                                    return 'SELL'
                                elif found.upper() in ('BUY', 'LONG', 'BULLISH', 'CALL', 'BREAKOUT', 'MOMENTUM UP'):
                                    return 'BUY'
                        return None
                    
                    for sym in unplaced[:3]:  # Max 3 retries
                        direction = _parse_direction_from_response(response, sym)
                        if not direction:
                            print(f"   ⚠️ Could not parse direction for {sym} from GPT response, skipping")
                            continue
                        
                        print(f"   🔄 Direct-placing {sym} ({direction}) — parsed from GPT analysis")
                        try:
                            if sym in fno_prefer_set:
                                result = self.tools.place_option_order(
                                    underlying=f"NSE:{sym}",
                                    direction=direction,
                                    strike_selection="ATM",
                                    rationale=f"GPT identified {sym} as top setup ({direction})"
                                )
                            else:
                                # For non-F&O, try option first then cash
                                result = self.tools.place_option_order(
                                    underlying=f"NSE:{sym}",
                                    direction=direction,
                                    strike_selection="ATM",
                                    rationale=f"GPT identified {sym} as top setup ({direction})"
                                )
                            if result:
                                status = "PLACED" if result.get('success') else result.get('error', 'unknown')[:80]
                                print(f"   ✅ Direct-place {sym}: {status}")
                                # If not F&O eligible, blacklist for this session
                                if not result.get('success') and 'not F&O eligible' in result.get('error', ''):
                                    self._rejected_this_cycle.add(sym)
                                    print(f"   🚫 {sym} blacklisted: not F&O eligible")
                            else:
                                print(f"   ❌ Direct-place {sym}: No result returned")
                        except Exception as e:
                            print(f"   ❌ Direct-place {sym} failed: {str(e)[:100]}")
            
            # Check if agent created any trades
            pending = self.agent.get_pending_approvals()
            if pending:
                for trade in pending:
                    self._record_trade(trade)
            
            # Mark ORB trades as taken to prevent re-entry
            # Check active trades for symbols that had ORB signals
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data:
                    orb_signal = data.get('orb_signal', 'INSIDE_ORB')
                    if self.tools.is_symbol_in_active_trades(symbol):
                        if orb_signal == "BREAKOUT_UP":
                            self._mark_orb_trade_taken(symbol, "UP")
                        elif orb_signal == "BREAKOUT_DOWN":
                            self._mark_orb_trade_taken(symbol, "DOWN")
            
            # === DECISION DASHBOARD: Compact summary of what happened this cycle ===
            _cycle_elapsed = time.time() - _cycle_start
            _active_now = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            _options_now = [t for t in _active_now if t.get('is_option')]
            _spreads_now = [t for t in _active_now if t.get('is_credit_spread') or t.get('is_debit_spread')]
            _ics_now = [t for t in _active_now if t.get('is_iron_condor')]
            
            print(f"\n{'='*80}")
            print(f"📋 CYCLE SUMMARY @ {datetime.now().strftime('%H:%M:%S')} ({_cycle_elapsed:.0f}s)")
            print(f"{'='*80}")
            
            # Market Regime (safe access)
            _d_breadth = _breadth if '_breadth' in dir() else 'N/A'
            _d_up = _up_count if '_up_count' in dir() else '?'
            _d_down = _down_count if '_down_count' in dir() else '?'
            _d_flat = _flat_count if '_flat_count' in dir() else '?'
            print(f"🌐 Market: {_d_breadth} | Up:{_d_up} Down:{_d_down} Flat:{_d_flat}")
            
            # Scorer summary (safe access)
            _d_scores = _pre_scores if '_pre_scores' in dir() else {}
            _d_fno = fno_opportunities if 'fno_opportunities' in dir() else []
            
            if _d_scores:
                _top5 = sorted(_d_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                _passed = sum(1 for s in _d_scores.values() if s >= 56)
                _fno_count = len(_d_fno)
                _scan_mode = "FULL F&O" if _full_scan_mode else "CURATED+WC"
                print(f"📊 Scored: {len(_d_scores)} stocks [{_scan_mode}] | Passed(≥56): {_passed} | F&O Ready: {_fno_count}")
                _top5_str = " | ".join(f"{s.replace('NSE:','')}={v:.0f}" for s, v in _top5)
                print(f"🏆 Top 5: {_top5_str}")
            
            # F&O opportunities that went to GPT
            if _d_fno:
                print(f"\n🎯 F&O SIGNALS SENT TO GPT:")
                for opp in _d_fno[:8]:
                    print(f"   {opp.strip()}")
            
            # What got rejected/blocked
            if self._rejected_this_cycle:
                print(f"\n🚫 REJECTED: {', '.join(self._rejected_this_cycle)}")
            
            # Current portfolio state
            _total_open = len(_active_now)
            if _total_open > 0:
                print(f"\n💼 PORTFOLIO: {_total_open} positions ({len(_options_now)} options, {len(_spreads_now)} spreads, {len(_ics_now)} ICs)")
            else:
                print(f"\n💼 PORTFOLIO: Empty — no open positions")
            
            # Hot watchlist
            try:
                from options_trader import get_hot_watchlist
                _wl = get_hot_watchlist()
                if _wl:
                    _wl_str = ", ".join(f"{k.replace('NSE:','')}({v.get('score',0):.0f})" for k, v in sorted(_wl.items(), key=lambda x: x[1].get('score',0), reverse=True)[:5])
                    print(f"🔥 WATCHLIST: {_wl_str}")
            except Exception:
                pass
            
            # P&L and timing
            _risk_status = self.risk_governor.get_current_state(
                self.daily_pnl, self.capital,
                [t for t in self.tools.paper_positions if t.get('status') == 'OPEN']
            )
            print(f"\n💰 P&L: ₹{_risk_status.daily_pnl:+,.0f} ({_risk_status.daily_pnl_pct:+.2f}%) | Trades: {_risk_status.trades_today} | W:{_risk_status.wins_today} L:{_risk_status.losses_today}")
            
            # Ticker stats
            if self.tools.ticker:
                _ts = self.tools.ticker.stats
                _ws_status = "🟢 LIVE" if _ts['connected'] else "🔴 REST"
                _fut_count = len(getattr(self.tools.ticker, '_futures_map', {}))
                print(f"🔌 Ticker: {_ws_status} | Sub:{_ts['subscribed']}(+{_fut_count} futures) | Hits:{_ts['cache_hits']} | Fallbacks:{_ts['fallbacks']} | Ticks:{_ts['ticks']}")
            
            print(f"⏱️ Cycle: {_cycle_elapsed:.0f}s | Next scan in ~{getattr(self, '_normal_interval', 5)}min")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._scanning = False  # Re-enable real-time dashboard
    
    def _record_trade(self, trade: dict):
        """Record a trade (paper or real) with risk checks"""
        # Check with risk governor first
        active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        position_value = trade.get('entry_price', 0) * trade.get('quantity', 0)
        
        permission = self.risk_governor.can_trade(
            symbol=trade['symbol'],
            position_value=position_value,
            active_positions=active_positions
        )
        
        if not permission.allowed:
            print(f"\n⛔ TRADE BLOCKED by Risk Governor:")
            print(f"   Symbol: {trade['symbol']}")
            print(f"   Reason: {permission.reason}")
            for w in permission.warnings:
                print(f"   ⚠️ {w}")
            return  # Don't record blocked trades
        
        # Apply size adjustment if suggested
        if permission.suggested_size_multiplier < 1.0:
            original_qty = trade.get('quantity', 0)
            trade['quantity'] = int(original_qty * permission.suggested_size_multiplier)
            print(f"   📉 Position size reduced: {original_qty} → {trade['quantity']}")
        
        trade['timestamp'] = datetime.now().isoformat()
        trade['paper'] = self.paper_mode
        
        if self.paper_mode:
            print(f"\n📝 PAPER TRADE: {trade['side']} {trade.get('quantity', 0)} {trade['symbol']}")
            print(f"   Entry: ₹{trade.get('entry_price', 0)}")
            print(f"   Stop: ₹{trade.get('stop_loss', 0)}")
            print(f"   Target: ₹{trade.get('target', 0)}")
        
        # Show warnings if any
        if permission.warnings:
            print(f"   ⚠️ Warnings: {', '.join(permission.warnings)}")
        
        # Register with Exit Manager for smart exits
        self.exit_manager.register_trade(
            symbol=trade['symbol'],
            side=trade['side'],
            entry_price=trade.get('entry_price', 0),
            stop_loss=trade.get('stop_loss', 0),
            target=trade.get('target', 0),
            quantity=trade.get('quantity', 0)
        )
        
        self.trades_today.append(trade)
        
        # Save to file
        with open('trade_log.json', 'a') as f:
            f.write(json.dumps(trade) + '\n')
    
    def monitor_positions(self):
        """Monitor open positions for exit signals"""
        if not self.positions:
            return
        
        print(f"👀 Monitoring {len(self.positions)} positions...")
        
        for pos in self.positions[:]:
            try:
                # Get current price
                data = self.tools.get_market_data([pos['symbol']])
                if pos['symbol'] in data:
                    ltp = data[pos['symbol']]['ltp']
                    
                    # Check stop loss
                    if ltp <= pos['stop_loss']:
                        self._exit_position(pos, ltp, "Stop Loss Hit")
                    
                    # Check target
                    elif ltp >= pos['target']:
                        self._exit_position(pos, ltp, "Target Hit")
            
            except Exception as e:
                print(f"❌ Error monitoring {pos['symbol']}: {e}")
    
    def _exit_position(self, pos: dict, exit_price: float, reason: str):
        """Exit a position"""
        pnl = (exit_price - pos['entry_price']) * pos['quantity']
        if pos['side'] == 'SELL':
            pnl = -pnl
        
        with self._pnl_lock:
            self.daily_pnl += pnl
            self.capital += pnl
        
        print(f"\n🚪 EXIT: {pos['symbol']} @ ₹{exit_price:.2f}")
        print(f"   Reason: {reason}")
        print(f"   P&L: ₹{pnl:,.0f}")
        print(f"   Daily P&L: ₹{self.daily_pnl:,.0f}")
        
        self.positions.remove(pos)
    
    def run(self, scan_interval_minutes: int = 5):
        """Run the autonomous trader with dynamic scan intervals.
        
        Early session (before EARLY_SESSION.end_time): scans every 3 minutes
        using 3-minute candles for faster indicator maturation.
        After early session: switches to standard 5-minute scans.
        """
        from config import EARLY_SESSION
        
        # Determine initial scan interval based on time of day
        _now = datetime.now()
        _early_end_parts = EARLY_SESSION['end_time'].split(':')
        _early_end = _now.replace(hour=int(_early_end_parts[0]), minute=int(_early_end_parts[1]), second=0, microsecond=0)
        _is_early = EARLY_SESSION.get('enabled', True) and _now < _early_end
        
        if _is_early:
            current_interval = EARLY_SESSION.get('scan_interval_minutes', 3)
            print(f"\n🚀 Starting autonomous trading (EARLY SESSION MODE)...")
            print(f"   📊 Using {EARLY_SESSION.get('candle_interval', '3minute')} candles until {EARLY_SESSION['end_time']}")
            print(f"   Scanning every {current_interval} minutes (switches to {scan_interval_minutes}min after {EARLY_SESSION['end_time']})")
        else:
            current_interval = scan_interval_minutes
            print(f"\n🚀 Starting autonomous trading...")
            print(f"   Scanning every {current_interval} minutes")
        
        print(f"   Real-time monitoring every {self.monitor_interval} seconds")
        print(f"   Press Ctrl+C to stop\n")
        
        # Start real-time position monitor
        self.start_realtime_monitor()
        
        # Track whether we've switched from early to normal mode
        self._switched_to_normal = not _is_early
        self._normal_interval = scan_interval_minutes
        
        # Schedule tasks with current interval
        schedule.every(current_interval).minutes.do(self.scan_and_trade)
        
        # Initial scan
        self.scan_and_trade()
        
        # Run loop
        try:
            while True:
                schedule.run_pending()
                
                # Check if we need to switch from early session to normal interval
                if not self._switched_to_normal:
                    _check_now = datetime.now()
                    if _check_now >= _early_end:
                        self._switched_to_normal = True
                        schedule.clear()
                        schedule.every(self._normal_interval).minutes.do(self.scan_and_trade)
                        print(f"\n🔄 Early session ended — switching to {self._normal_interval}-minute scan interval with 5-minute candles\n")
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            self.stop_realtime_monitor()
            self._print_summary()
    
    def _print_summary(self):
        """Print trading summary AND save structured daily report"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # === Collect data from trade_history.json ===
        history = []
        try:
            import json
            history_file = os.path.join(os.path.dirname(__file__), 'trade_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    all_history = json.load(f)
                history = [t for t in all_history if t.get('closed_at', '').startswith(today) or t.get('timestamp', '').startswith(today)]
        except Exception:
            pass
        
        total_trades = len(history)
        winners = [t for t in history if t.get('pnl', 0) > 0]
        losers = [t for t in history if t.get('pnl', 0) < 0]
        breakevens = [t for t in history if t.get('pnl', 0) == 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in history)
        avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t['pnl'] for t in losers) / len(losers) if losers else 0
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        # By strategy
        by_strategy = {}
        for t in history:
            strat = t.get('strategy_type', t.get('spread_type', 'NAKED_OPTION'))
            if t.get('is_iron_condor'):
                strat = 'IRON_CONDOR'
            elif t.get('is_credit_spread'):
                strat = 'CREDIT_SPREAD'
            elif t.get('is_debit_spread'):
                strat = 'DEBIT_SPREAD'
            if strat not in by_strategy:
                by_strategy[strat] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('pnl', 0) > 0:
                by_strategy[strat]['wins'] += 1
            elif t.get('pnl', 0) < 0:
                by_strategy[strat]['losses'] += 1
            by_strategy[strat]['pnl'] += t.get('pnl', 0)
        
        # By exit type
        by_exit = {}
        for t in history:
            exit_type = t.get('result', 'UNKNOWN')
            by_exit[exit_type] = by_exit.get(exit_type, 0) + 1
        
        # By score tier
        by_tier = {}
        for t in history:
            tier = t.get('score_tier', t.get('entry_metadata', {}).get('score_tier', 'unknown'))
            if tier not in by_tier:
                by_tier[tier] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('pnl', 0) > 0:
                by_tier[tier]['wins'] += 1
            elif t.get('pnl', 0) < 0:
                by_tier[tier]['losses'] += 1
            by_tier[tier]['pnl'] += t.get('pnl', 0)
        
        # Candle gate analysis — did entry characteristics predict outcome?
        gate_analysis = {'ft_winners_avg': 0, 'ft_losers_avg': 0, 'adx_winners_avg': 0, 'adx_losers_avg': 0}
        for t in history:
            meta = t.get('entry_metadata', {})
            ft = meta.get('follow_through_candles', 0)
            adx = meta.get('adx', 0)
            if t.get('pnl', 0) > 0:
                gate_analysis['ft_winners_avg'] = (gate_analysis['ft_winners_avg'] + ft) / 2 if gate_analysis['ft_winners_avg'] else ft
                gate_analysis['adx_winners_avg'] = (gate_analysis['adx_winners_avg'] + adx) / 2 if gate_analysis['adx_winners_avg'] else adx
            elif t.get('pnl', 0) < 0:
                gate_analysis['ft_losers_avg'] = (gate_analysis['ft_losers_avg'] + ft) / 2 if gate_analysis['ft_losers_avg'] else ft
                gate_analysis['adx_losers_avg'] = (gate_analysis['adx_losers_avg'] + adx) / 2 if gate_analysis['adx_losers_avg'] else adx
        
        # Exit manager quality — candles held, R achieved
        avg_candles_held = 0
        avg_r_achieved = 0
        max_r_left_on_table = 0
        r_trades = 0
        for t in history:
            ed = t.get('exit_detail', {})
            candles = ed.get('candles_held', 0)
            r_mult = ed.get('r_multiple_achieved', 0)
            if candles > 0:
                avg_candles_held += candles
                r_trades += 1
            if r_mult:
                avg_r_achieved += r_mult
        if r_trades > 0:
            avg_candles_held /= r_trades
            avg_r_achieved /= r_trades
        
        # === PRINT TO TERMINAL ===
        print("\n" + "="*60)
        print(f"📊 DAILY TRADING SUMMARY — {today}")
        print("="*60)
        print(f"  Capital: ₹{self.start_capital:,.0f} → ₹{self.capital:,.0f}")
        print(f"  Daily P&L: ₹{self.daily_pnl:,.0f} ({(self.capital/self.start_capital - 1)*100:+.2f}%)")
        print(f"  Trades: {total_trades} | W: {len(winners)} | L: {len(losers)} | BE: {len(breakevens)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Winner: ₹{avg_win:+,.0f} | Avg Loser: ₹{avg_loss:+,.0f}")
        if avg_loss != 0:
            print(f"  Payoff Ratio: {abs(avg_win/avg_loss):.2f}:1")
        
        if by_strategy:
            print(f"\n  📈 BY STRATEGY:")
            for strat, data in by_strategy.items():
                print(f"    {strat}: W{data['wins']}/L{data['losses']} P&L: ₹{data['pnl']:+,.0f}")
        
        if by_exit:
            print(f"\n  🚪 BY EXIT TYPE:")
            for exit_type, count in sorted(by_exit.items(), key=lambda x: x[1], reverse=True):
                print(f"    {exit_type}: {count}")
        
        if by_tier:
            print(f"\n  🏆 BY SCORE TIER:")
            for tier, data in by_tier.items():
                print(f"    {tier}: W{data['wins']}/L{data['losses']} P&L: ₹{data['pnl']:+,.0f}")
        
        if r_trades > 0:
            print(f"\n  ⏱️ EXIT QUALITY:")
            print(f"    Avg Candles Held: {avg_candles_held:.1f}")
            print(f"    Avg R-Multiple: {avg_r_achieved:.2f}")
        
        print("="*60)
        
        # === SAVE TO DAILY SUMMARY FILE ===
        try:
            import json
            summary = {
                'date': today,
                'capital_start': self.start_capital,
                'capital_end': round(self.capital, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'return_pct': round((self.capital/self.start_capital - 1)*100, 2),
                'total_trades': total_trades,
                'wins': len(winners),
                'losses': len(losers),
                'breakevens': len(breakevens),
                'win_rate': round(win_rate, 1),
                'avg_winner': round(avg_win, 2),
                'avg_loser': round(avg_loss, 2),
                'payoff_ratio': round(abs(avg_win/avg_loss), 2) if avg_loss != 0 else 0,
                'by_strategy': by_strategy,
                'by_exit_type': by_exit,
                'by_score_tier': by_tier,
                'gate_analysis': gate_analysis,
                'exit_quality': {
                    'avg_candles_held': round(avg_candles_held, 1),
                    'avg_r_multiple': round(avg_r_achieved, 2),
                },
                'individual_trades': [
                    {
                        'trade_id': t.get('trade_id', ''),
                        'symbol': t.get('symbol', ''),
                        'underlying': t.get('underlying', ''),
                        'side': t.get('side', ''),
                        'entry_score': t.get('entry_score', t.get('entry_metadata', {}).get('entry_score', 0)),
                        'score_tier': t.get('score_tier', 'unknown'),
                        'strategy_type': t.get('strategy_type', ''),
                        'entry': t.get('avg_price', 0),
                        'exit': t.get('exit_price', 0),
                        'pnl': t.get('pnl', 0),
                        'result': t.get('result', ''),
                        'candles_held': t.get('exit_detail', {}).get('candles_held', 0),
                        'r_multiple': t.get('exit_detail', {}).get('r_multiple_achieved', 0),
                        'ft_at_entry': t.get('entry_metadata', {}).get('follow_through_candles', 0),
                        'adx_at_entry': t.get('entry_metadata', {}).get('adx', 0),
                        'orb_at_entry': t.get('entry_metadata', {}).get('orb_strength_pct', 0),
                        'timestamp': t.get('timestamp', ''),
                    }
                    for t in history
                ],
            }
            
            summary_dir = os.path.join(os.path.dirname(__file__), 'daily_summaries')
            os.makedirs(summary_dir, exist_ok=True)
            summary_file = os.path.join(summary_dir, f'daily_summary_{today}.json')
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n  💾 Daily summary saved to: daily_summaries/daily_summary_{today}.json")
            
            # Also append to trade_decisions.log
            from options_trader import TRADE_DECISIONS_LOG
            with open(TRADE_DECISIONS_LOG, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"📊 DAILY SUMMARY — {today}\n")
                f.write(f"   Capital: ₹{self.start_capital:,.0f} → ₹{self.capital:,.0f}\n")
                f.write(f"   P&L: ₹{self.daily_pnl:,.0f} ({(self.capital/self.start_capital - 1)*100:+.2f}%)\n")
                f.write(f"   Trades: {total_trades} | W: {len(winners)} L: {len(losers)}\n")
                f.write(f"   Win Rate: {win_rate:.1f}% | Payoff: {abs(avg_win/avg_loss):.2f}:1\n" if avg_loss != 0 else f"   Win Rate: {win_rate:.1f}%\n")
                for strat, data in by_strategy.items():
                    f.write(f"   {strat}: W{data['wins']}/L{data['losses']} ₹{data['pnl']:+,.0f}\n")
                f.write(f"   Avg Candles: {avg_candles_held:.1f} | Avg R: {avg_r_achieved:.2f}\n")
                f.write(f"{'='*70}\n")
            
            # === QUANT ANALYTICS EOD REPORT (Feb 13) ===
            try:
                from quant_analytics import load_trade_history, generate_full_report
                all_trades = load_trade_history()
                if all_trades:
                    # Today's report
                    today_report = generate_full_report(all_trades, self.start_capital, filter_date=today)
                    report_file = os.path.join(summary_dir, f'quant_report_{today}.txt')
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(today_report)
                    print(f"  📊 Quant analytics report saved to: daily_summaries/quant_report_{today}.txt")
                    
                    # Also generate cumulative all-time report
                    cum_report = generate_full_report(all_trades, self.start_capital)
                    cum_file = os.path.join(summary_dir, 'quant_report_cumulative.txt')
                    with open(cum_file, 'w', encoding='utf-8') as f:
                        f.write(cum_report)
                    print(f"  📊 Cumulative report saved to: daily_summaries/quant_report_cumulative.txt")
            except Exception as e:
                print(f"  ⚠️ Could not generate quant report: {e}")
        except Exception as e:
            print(f"  ⚠️ Could not save daily summary: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Trading Bot')
    parser.add_argument('--capital', type=float, default=200000, help='Starting capital')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper)')
    parser.add_argument('--interval', type=int, default=5, help='Scan interval in minutes')
    
    args = parser.parse_args()
    
    bot = AutonomousTrader(
        capital=args.capital,
        paper_mode=not args.live
    )
    
    bot.run(scan_interval_minutes=args.interval)


if __name__ == "__main__":
    main()
