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

sys.path.insert(0, os.path.dirname(__file__))

from config import HARD_RULES, APPROVED_UNIVERSE, TRADING_HOURS
from llm_agent import TradingAgent
from zerodha_tools import get_tools, reset_tools
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
        print(f"\n  Universe: {len(APPROVED_UNIVERSE)} stocks")
        
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
        
        # Real-time monitoring
        self.monitor_running = False
        self.monitor_thread = None
        self.monitor_interval = 3  # Check every 3 seconds
        
        # ORB trade tracking - once per direction per symbol per day
        self.orb_trades_today = {}  # {symbol: {"UP": False, "DOWN": False}}
        self.orb_tracking_date = datetime.now().date()
        
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
        
        # Start reconciliation loop
        self.position_recon.start()
        
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
    
    def _reset_orb_tracker_if_new_day(self):
        """Reset ORB tracker at start of new trading day"""
        today = datetime.now().date()
        if today != self.orb_tracking_date:
            self.orb_trades_today = {}
            self.orb_tracking_date = today
            print(f"📅 New trading day - ORB tracker reset")
    
    def _is_orb_trade_allowed(self, symbol: str, direction: str) -> bool:
        """Check if ORB trade is allowed (once per direction per symbol per day)"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        return not self.orb_trades_today[symbol].get(direction, False)
    
    def _mark_orb_trade_taken(self, symbol: str, direction: str):
        """Mark ORB direction as used for symbol today"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        self.orb_trades_today[symbol][direction] = True
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
        while self.monitor_running:
            try:
                if self.is_trading_hours():
                    self._check_positions_realtime()
                    self._check_eod_exit()  # Check if need to exit before close
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _check_eod_exit(self):
        """Exit all positions before market close (3:15 PM)"""
        now = datetime.now().time()
        eod_exit_time = datetime.strptime("15:10", "%H:%M").time()  # Exit 5 mins before
        
        if now >= eod_exit_time:
            active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            if active_trades:
                print(f"\n⏰ END OF DAY - Closing all positions...")
                
                # Get current prices
                symbols = [t['symbol'] for t in active_trades]
                try:
                    quotes = self.tools.kite.quote(symbols)
                except:
                    return
                
                for trade in active_trades:
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
                    
                    self.tools.update_trade_status(symbol, 'EOD_EXIT', ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
    
    def _check_positions_realtime(self):
        """Check all positions for target/stoploss hits using Exit Manager"""
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # Separate equity and option positions
        equity_trades = [t for t in active_trades if not t.get('is_option', False)]
        option_trades = [t for t in active_trades if t.get('is_option', False)]
        
        # Get current prices for equity positions
        if equity_trades:
            symbols = [t['symbol'] for t in equity_trades]
            try:
                quotes = self.tools.kite.quote(symbols)
            except Exception as e:
                quotes = {}
        else:
            quotes = {}
        
        # Print position status every 30 seconds (every 10th check)
        if not hasattr(self, '_monitor_count'):
            self._monitor_count = 0
        self._monitor_count += 1
        
        show_status = (self._monitor_count % 10 == 0)  # Every 30 seconds
        
        # === CHECK OPTION EXITS (Greeks-based) ===
        if option_trades:
            try:
                option_exits = self.tools.check_option_exits()
                for exit_signal in option_exits:
                    symbol = exit_signal['symbol']
                    reason = exit_signal.get('reason', 'Greeks exit')
                    
                    # Find the option trade
                    opt_trade = next((t for t in option_trades if t['symbol'] == symbol), None)
                    if opt_trade:
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
                        self.daily_pnl += pnl
                        self.capital += pnl
                        
                        # Record with Risk Governor
                        self.risk_governor.record_trade_result(symbol, pnl, pnl > 0)
                        self.risk_governor.update_capital(self.capital)
            except Exception as e:
                if show_status:
                    print(f"   ⚠️ Option exit check error: {e}")
        
        # === EXIT MANAGER INTEGRATION ===
        # Check all trades via exit manager
        price_dict = {t['symbol']: quotes.get(t['symbol'], {}).get('last_price', 0) for t in equity_trades}
        exit_signals = self.exit_manager.check_all_exits(price_dict)
        
        # Process exit signals first (highest priority)
        for signal in exit_signals:
            if signal.should_exit:
                trade = next((t for t in active_trades if t['symbol'] == signal.symbol), None)
                if trade:
                    symbol = signal.symbol
                    ltp = price_dict.get(symbol, 0) or signal.exit_price
                    entry = trade['avg_price']
                    qty = trade['quantity']
                    side = trade['side']
                    
                    if side == 'BUY':
                        pnl = (ltp - entry) * qty
                    else:
                        pnl = (entry - ltp) * qty
                    
                    pnl_pct = pnl / (entry * qty) * 100
                    was_win = pnl > 0
                    
                    # Exit based on signal type
                    emoji = {
                        'SL_HIT': '❌',
                        'TARGET_HIT': '🎯',
                        'SESSION_CUTOFF': '⏰',
                        'TIME_STOP': '⏱️',
                        'TRAILING_SL': '📈'
                    }.get(signal.exit_type, '🚪')
                    
                    print(f"\n{emoji} {signal.exit_type}! {symbol}")
                    print(f"   Reason: {signal.reason}")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    
                    self.tools.update_trade_status(symbol, signal.exit_type, ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
                    
                    # Record with Risk Governor
                    self.risk_governor.record_trade_result(symbol, pnl, was_win)
                    self.risk_governor.update_capital(self.capital)
                    
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
                    trade['stop_loss'] = state.current_sl
                    self.tools._save_active_trades()
        
        if show_status and active_trades:
            print(f"\n👁️ LIVE POSITIONS [{datetime.now().strftime('%H:%M:%S')}]:")
            print(f"{'Symbol':15} {'Side':6} {'Entry':>10} {'LTP':>10} {'SL':>10} {'Target':>10} {'P&L':>12} {'Status'}")
            print("-" * 95)
        
        for trade in active_trades:
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
            total_pnl = sum(
                (quotes[t['symbol']]['last_price'] - t['avg_price']) * t['quantity'] 
                if t['side'] == 'BUY' 
                else (t['avg_price'] - quotes[t['symbol']]['last_price']) * t['quantity']
                for t in active_trades if t['symbol'] in quotes and t.get('status', 'OPEN') == 'OPEN'
            )
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
        
        # CHECK AND UPDATE EXISTING TRADES (target/stoploss hits)
        trade_updates = self.tools.check_and_update_trades()
        if trade_updates:
            print(f"\n📊 TRADE UPDATES:")
            for update in trade_updates:
                emoji = "✅" if update['result'] == 'TARGET_HIT' else "❌"
                print(f"   {emoji} {update['symbol']}: {update['result']}")
                print(f"      Entry: ₹{update['entry']:.2f} → Exit: ₹{update['exit']:.2f}")
                print(f"      P&L: ₹{update['pnl']:+,.2f}")
                self.daily_pnl += update['pnl']
        
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
            # Get fresh market data for ALL stocks (not just first 10)
            market_data = self.tools.get_market_data(APPROVED_UNIVERSE)
            
            # Get volume analysis for EOD predictions
            volume_analysis = self.tools.get_volume_analysis(APPROVED_UNIVERSE)
            
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
  RSI: {data.get('rsi_14', 50):.0f} | Volume: {data.get('volume_signal', 'N/A')}
  Order Flow: {order_flow} | Buy%: {buy_ratio}% | EOD Prediction: {eod_pred}
  SMA20: ₹{data.get('sma_20', 0):.2f} | SMA50: ₹{data.get('sma_50', 0):.2f}
  Support: ₹{data.get('support_1', 0):.2f} / ₹{data.get('support_2', 0):.2f}
  Resistance: ₹{data.get('resistance_1', 0):.2f} / ₹{data.get('resistance_2', 0):.2f}"""
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
                    ema_regime = data.get('ema_regime', 'NEUTRAL')
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
                    if orb_signal == "BREAKOUT_UP" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime == "EXPANDING_BULL":
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
                    elif orb_signal == "BREAKOUT_DOWN" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime == "EXPANDING_BEAR":
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
                    quick_scan.append(f"{symbol}: {chg:+.2f}% RSI:{rsi:.0f} {trend} ORB:{orb_signal} Vol:{volume_regime} HTF:{htf_icon} {setup}")
            
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
            can_trade_check = self.risk_governor.can_trade(
                symbol="NSE:DUMMY",  # Just checking general permission
                position_value=10000,
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
            
            # Ask agent to analyze market with FULL CONTEXT
            prompt = f"""EXECUTE TRADES NOW using place_order tool - DO NOT just describe trades!

MARKET SCAN ({len(quick_scan)} stocks):
{chr(10).join(quick_scan[:20])}

REGIME SIGNALS (HIGHEST PRIORITY):
{chr(10).join(regime_signals) if regime_signals else 'No strong regime signals'}

EOD PREDICTIONS:
{chr(10).join(eod_opportunities) if eod_opportunities else 'No EOD signals'}

ALREADY HOLDING (SKIP):
{', '.join(active_symbols) if active_symbols else 'None'}

=== CORRELATION EXPOSURE ===
{corr_exposure}

=== SYSTEM HEALTH ===
Reconciliation: {recon_state} {'(CAN TRADE)' if recon_can_trade else '(BLOCKED: ' + recon_reason + ')'}
Data Health: {len(halted_symbols)} halted symbols
{f'Halted: {", ".join(halted_symbols[:5])}' if halted_symbols else 'All symbols healthy'}

=== ACCOUNT & RISK STATUS ===
Capital: Rs{self.capital:,.0f}
Daily P&L: Rs{risk_status.daily_pnl:+,.0f} ({risk_status.daily_pnl_pct:+.2f}%)
Trades Today: {risk_status.trades_today}/{self.risk_governor.limits.max_trades_per_day} (Remaining: {trades_remaining})
Win/Loss: {risk_status.wins_today}/{risk_status.losses_today}
Consecutive Losses: {risk_status.consecutive_losses}/{self.risk_governor.limits.max_consecutive_losses}
Risk per Trade: Rs{self.capital * 0.005:,.0f} (0.5%)

=== REGIME SCORE THRESHOLDS ===
- ORB_BREAKOUT: need 70+ score to trade
- VWAP_TREND: need 60+ score to trade
- EMA_SQUEEZE: need 65+ score to trade
- MEAN_REVERSION: need 65+ score to trade

=== TRADING PRIORITY (highest to lowest) ===

1. ORB BREAKOUT (Opening Range Breakout):
   - BREAKOUT_UP + HIGH/EXPLOSIVE volume + EMA expanding = BUY
   - BREAKOUT_DOWN + HIGH/EXPLOSIVE volume + EMA expanding = SELL
   - Most reliable intraday signal

2. EMA SQUEEZE + VOLUME:
   - EMA COMPRESSED (9 and 21 EMA tight) + EXPLOSIVE volume = breakout imminent
   - Wait for direction, then enter with tight SL

3. VWAP TREND TRADES:
   - Price ABOVE_VWAP + VWAP RISING + RSI<60 = BUY
   - Price BELOW_VWAP + VWAP FALLING + RSI>40 = SELL

4. RSI EXTREMES:
   - RSI < 30 = Oversold BUY
   - RSI > 70 = Overbought SELL

5. EOD PLAYS (after 11 AM):
   - OI signals + volume confirmation = quick scalp

RISK RULES:
- Stop Loss: 1% from entry
- Target: 1.5% from entry
- Max 2-3 trades at a time

=== OPTIONS TRADING (F&O STOCKS) ===
For F&O eligible stocks (RELIANCE, TCS, INFY, HDFCBANK, etc.), you can use options:
- place_option_order(underlying="NSE:RELIANCE", direction="BUY", strike_selection="ATM")
- Options auto-select CE for BUY signals, PE for SELL signals
- Strike selections: ATM, ITM_1, ITM_2, OTM_1, OTM_2
- Options have fixed lot sizes (RELIANCE=250, TCS=150, etc.)
- Max premium per trade: Rs15,000 | Max total exposure: Rs50,000

WHEN TO USE OPTIONS:
- High conviction directional views → ATM or ITM_1
- Strong breakouts with volume → OTM_1 for leverage
- Volatile markets → Options limit downside to premium paid

YOU MUST CALL place_order() tool NOW with strategy and setup_id for order tracking!
Example:
place_order(symbol="NSE:INFY", side="BUY", quantity=100, stop_loss=1490, target=1550, strategy="ORB", setup_id="BREAKOUT_UP")

OPTION EXAMPLE:
place_option_order(underlying="NSE:RELIANCE", direction="BUY", strike_selection="ATM", rationale="Strong ORB breakout")

STRATEGY VALUES: ORB, VWAP, EMA_SQUEEZE, RSI, EOD, VOLUME
SETUP_ID VALUES: BREAKOUT_UP, BREAKOUT_DOWN, VWAP_TREND, OVERSOLD, OVERBOUGHT, SQUEEZE_BREAK"""

            response = self.agent.run(prompt)
            print(f"\n📊 Agent response:\n{response[:500]}...")
            
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
        """Print trading summary"""
        print("\n" + "="*60)
        print("📊 TRADING SUMMARY")
        print("="*60)
        print(f"  Start capital: ₹{self.start_capital:,}")
        print(f"  End capital: ₹{self.capital:,}")
        print(f"  Daily P&L: ₹{self.daily_pnl:,.0f}")
        print(f"  Return: {(self.capital/self.start_capital - 1)*100:.2f}%")
        print(f"  Trades: {len(self.trades_today)}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Trading Bot')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
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
