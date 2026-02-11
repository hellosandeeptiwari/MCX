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

# Ensure CWD is the script's directory so relative file paths (risk_state.json, active_trades.json) resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from config import HARD_RULES, APPROVED_UNIVERSE, TRADING_HOURS, FNO_CONFIG, TIER_1_OPTIONS, TIER_2_OPTIONS
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
        
        # Pre-populate lot sizes from Kite API at startup
        try:
            update_fno_lot_sizes(self.market_scanner.get_lot_map())
        except Exception as e:
            print(f"⚠️ Dynamic lot size fetch failed at startup (will retry on scan): {e}")
        
        # Start reconciliation loop
        self.position_recon.start()
        
        # Sync exit manager with existing positions (crash recovery)
        self._sync_exit_manager_with_positions()
        
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
        """Exit all positions before market close (3:20 PM)"""
        now = datetime.now().time()
        eod_exit_time = datetime.strptime("15:15", "%H:%M").time()  # Exit 5 mins before 3:20
        
        if now >= eod_exit_time:
            with self.tools._positions_lock:
                active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            if active_trades:
                print(f"\n⏰ END OF DAY - Closing all positions...")
                
                # Separate trade types
                regular_trades = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False)]
                spread_trades_eod = [t for t in active_trades if t.get('is_credit_spread', False)]
                debit_spread_trades_eod = [t for t in active_trades if t.get('is_debit_spread', False)]
                
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
                
                if not all_symbols:
                    return
                    
                try:
                    quotes = self.tools.kite.quote(all_symbols)
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
    
    def _check_positions_realtime(self):
        """Check all positions for target/stoploss hits using Exit Manager"""
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # === SYNC: Register any new trades the agent placed since last check ===
        self._sync_exit_manager_with_positions()
        
        # Separate trade types
        equity_trades = [t for t in active_trades if not t.get('is_option', False)]
        option_trades = [t for t in active_trades if t.get('is_option', False) and not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False)]
        spread_trades = [t for t in active_trades if t.get('is_credit_spread', False)]
        debit_spread_trades = [t for t in active_trades if t.get('is_debit_spread', False)]
        
        # === GET CURRENT PRICES ===
        # For spreads (credit & debit), we need both leg symbols
        all_symbols = set()
        for t in active_trades:
            if t.get('is_credit_spread'):
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
                quotes = self.tools.kite.quote(all_symbols)
            except Exception as e:
                # If mixed exchange query fails, try separately
                quotes = {}
                if equity_trades:
                    try:
                        eq_syms = [t['symbol'] for t in equity_trades]
                        quotes.update(self.tools.kite.quote(eq_syms))
                    except:
                        pass
                if option_trades:
                    try:
                        opt_syms = [t['symbol'] for t in option_trades]
                        quotes.update(self.tools.kite.quote(opt_syms))
                    except:
                        pass
        else:
            quotes = {}
        
        # Print position status every 30 seconds (every 10th check)
        if not hasattr(self, '_monitor_count'):
            self._monitor_count = 0
        self._monitor_count += 1
        
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
                            opt_quote = self.tools.kite.quote([symbol])
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
                    if hasattr(trade, 'expected_exit') and trade.expected_exit:
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
            
            # --- Show regular trades ---
            regular_open = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and t.get('status', 'OPEN') == 'OPEN']
            if regular_open:
                print(f"{'Symbol':15} {'Side':6} {'Entry':>10} {'LTP':>10} {'SL':>10} {'Target':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
        
        for trade in active_trades:
            if trade.get('is_credit_spread', False) or trade.get('is_debit_spread', False):
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
        
        print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Scanning market...")
        print(self.risk_governor.get_status())
        self._rejected_this_cycle = set()  # Reset rejected symbols for new scan
        
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
                # === DYNAMIC LOT SIZES: ensure every F&O stock has correct lot size ===
                update_fno_lot_sizes(self.market_scanner.get_lot_map())
            except Exception as e:
                print(f"⚠️ Scanner error (non-fatal): {e}")
                scan_result = None
                self._wildcard_symbols = []
            
            # Merge wild-cards into scan universe for this cycle
            scan_universe = list(APPROVED_UNIVERSE)
            for ws in self._wildcard_symbols:
                if ws not in scan_universe:
                    scan_universe.append(ws)
            
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
                    quick_scan.append(f"{symbol}{fno_tag}: {chg:+.2f}% RSI:{rsi:.0f} {trend} ORB:{orb_signal} Vol:{volume_regime} HTF:{htf_icon} {setup}")
            
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
            
            # Build F&O opportunity list
            fno_prefer = FNO_CONFIG.get('prefer_options_for', [])
            # Wild-card symbols are F&O-eligible (they came from NFO instruments list)
            fno_prefer_set = set(fno_prefer) | set(self._wildcard_symbols)
            fno_opportunities = []
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data and symbol in fno_prefer_set:
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
                        # Tier-2 requires: clear trend OR ORB breakout with volume
                        tier2_trending = trend_state in ('BULLISH', 'STRONG_BULLISH', 'BEARISH', 'STRONG_BEARISH')
                        tier2_orb = orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and vol in ('HIGH', 'EXPLOSIVE')
                        if not (tier2_trending or tier2_orb):
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
                    
                    if setup_type and direction:
                        opt_type = "CE" if direction == "BUY" else "PE"
                        fno_opportunities.append(
                            f"  🎯 {symbol}: {setup_type} → place_option_order(underlying=\"{symbol}\", direction=\"{direction}\", strike_selection=\"ATM\") [{opt_type}]"
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
                        
                        max_debit_entries = 2  # Max 2 proactive debit spreads per scan cycle
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
            
            # Ask agent to analyze market with FULL CONTEXT
            prompt = f"""EXECUTE TRADES NOW - DO NOT just describe trades!

=== ⚡F&O OPTIONS FIRST (HIGHEST PRIORITY) ===
For stocks marked [F&O], use place_option_order() instead of place_order().
F&O stocks: {', '.join(fno_prefer)}

F&O READY SIGNALS:
{chr(10).join(fno_opportunities) if fno_opportunities else 'No F&O setups right now - check CASH stocks below'}

=== 📡 MARKET SCANNER WILD-CARDS ===
{wildcard_info if wildcard_info else 'No wild-card movers outside fixed universe this cycle'}

=== MARKET SCAN ({len(quick_scan)} stocks) ===
{chr(10).join(quick_scan[:25])}

=== DETAILED TECHNICALS (Top Movers) ===
{chr(10).join(detailed_data[:10])}

=== REGIME SIGNALS (HIGH PRIORITY) ===
{chr(10).join(regime_signals) if regime_signals else 'No strong regime signals'}

=== EOD PREDICTIONS ===
{chr(10).join(eod_opportunities) if eod_opportunities else 'No EOD signals'}

ALREADY HOLDING (SKIP): {', '.join(active_symbols) if active_symbols else 'None'}

=== CORRELATION EXPOSURE ===
{corr_exposure}

=== SYSTEM HEALTH ===
Reconciliation: {recon_state} {'(CAN TRADE)' if recon_can_trade else '(BLOCKED: ' + recon_reason + ')'}
Data Health: {len(halted_symbols)} halted | {'Halted: ' + ', '.join(halted_symbols[:5]) if halted_symbols else 'All healthy'}

=== ACCOUNT ===
Capital: Rs{self.capital:,.0f} | Daily P&L: Rs{risk_status.daily_pnl:+,.0f} ({risk_status.daily_pnl_pct:+.2f}%)
Trades: {risk_status.trades_today}/{self.risk_governor.limits.max_trades_per_day} (Remaining: {trades_remaining})
W/L: {risk_status.wins_today}/{risk_status.losses_today} | Consec Losses: {risk_status.consecutive_losses}/{self.risk_governor.limits.max_consecutive_losses}

=== EXECUTION RULES ===
1. F&O stocks → ALWAYS use place_option_order(underlying, direction, strike_selection="ATM")
2. Cash stocks → use place_order(symbol, side, quantity, stop_loss, target, strategy, setup_id)
3. Stop Loss: 1% from entry | Target: 1.5% from entry | Max 3 trades at a time
4. Call tools IMMEDIATELY. Do not describe trades without placing them.
5. Strategy: ORB, VWAP, EMA_SQUEEZE, RSI, EOD | Setup: BREAKOUT_UP, BREAKOUT_DOWN, VWAP_TREND, OVERSOLD, OVERBOUGHT"""

            response = self.agent.run(prompt)
            print(f"\n📊 Agent response:\n{response[:300]}...")
            
            # === AUTO-RETRY: Detect trades mentioned but not placed ===
            # Collect scorer rejections from the tools layer
            if not hasattr(self, '_rejected_this_cycle'):
                self._rejected_this_cycle = set()
            if hasattr(self.tools, '_scorer_rejected_symbols'):
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
                unplaced = [s for s in set(mentioned_symbols) 
                           if s not in active_symbols_set 
                           and f'NSE:{s}' in [sym for sym in APPROVED_UNIVERSE]
                           and s not in self._rejected_this_cycle]
                
                if unplaced and len(unplaced) <= 3:
                    print(f"\n🔄 Detected {len(unplaced)} unplaced trades in response: {unplaced}")
                    fno_prefer_set = {s.replace('NSE:', '') for s in FNO_CONFIG.get('prefer_options_for', [])}
                    for sym in unplaced[:2]:  # Max 2 retries
                        if sym in fno_prefer_set:
                            retry_prompt = f"You mentioned NSE:{sym} as a trade but did NOT call place_option_order(). NSE:{sym} is F&O eligible. Call place_option_order(underlying=\"NSE:{sym}\", direction=<BUY or SELL based on your analysis>, strike_selection=\"ATM\") NOW. Do not explain - just call the tool."
                        else:
                            retry_prompt = f"You mentioned NSE:{sym} as a trade but did NOT call place_order(). Call place_order() for NSE:{sym} NOW with the entry, stop_loss, and target you identified. Do not explain - just call the tool."
                        retry_response = self.agent.run(retry_prompt)
                        print(f"   🔄 Retry for {sym}: {retry_response[:150]}...")
            
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
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
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
        """Run the autonomous trader"""
        print(f"\n🚀 Starting autonomous trading...")
        print(f"   Scanning every {scan_interval_minutes} minutes")
        print(f"   Real-time monitoring every {self.monitor_interval} seconds")
        print(f"   Press Ctrl+C to stop\n")
        
        # Start real-time position monitor
        self.start_realtime_monitor()
        
        # Schedule tasks
        schedule.every(scan_interval_minutes).minutes.do(self.scan_and_trade)
        
        # Initial scan
        self.scan_and_trade()
        
        # Run loop
        try:
            while True:
                schedule.run_pending()
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
            if t.get('is_credit_spread'):
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
