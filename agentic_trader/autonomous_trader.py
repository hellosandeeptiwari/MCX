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
        
        print("\n  ✅ Bot initialized!")
        print("  🟢 Auto-execution: ON")
        print("  ⚡ Real-time monitoring: ENABLED (every 3 sec)")
        print("="*60)
    
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
        """Check all positions for target/stoploss hits"""
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # Get current prices
        symbols = [t['symbol'] for t in active_trades]
        try:
            quotes = self.tools.kite.quote(symbols)
        except Exception as e:
            return  # Silently fail if API error
        
        # Print position status every 30 seconds (every 10th check)
        if not hasattr(self, '_monitor_count'):
            self._monitor_count = 0
        self._monitor_count += 1
        
        show_status = (self._monitor_count % 10 == 0)  # Every 30 seconds
        
        if show_status and active_trades:
            print(f"\n👁️ LIVE POSITIONS [{datetime.now().strftime('%H:%M:%S')}]:")
            print(f"{'Symbol':15} {'Side':6} {'Entry':>10} {'LTP':>10} {'SL':>10} {'Target':>10} {'P&L':>12}")
            print("-" * 85)
        
        for trade in active_trades:
            symbol = trade['symbol']
            if symbol not in quotes:
                continue
            
            ltp = quotes[symbol]['last_price']
            entry = trade['avg_price']
            sl = trade['stop_loss']
            target = trade.get('target', entry * (1.02 if trade['side'] == 'BUY' else 0.98))
            qty = trade['quantity']
            side = trade['side']
            
            # Calculate current P&L
            if side == 'BUY':
                pnl = (ltp - entry) * qty
                pnl_pct = (ltp - entry) / entry * 100
                
                # Show status line
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (ltp - sl) / ltp * 100
                    tgt_dist = (target - ltp) / ltp * 100
                    print(f"{status} {symbol:13} {'BUY':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%)")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
                
                # TARGET HIT
                if ltp >= target:
                    print(f"\n🎯 TARGET HIT! {symbol}")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    self.tools.update_trade_status(symbol, 'TARGET_HIT', ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
                
                # STOPLOSS HIT
                elif ltp <= sl:
                    print(f"\n❌ STOPLOSS HIT! {symbol}")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    self.tools.update_trade_status(symbol, 'STOPLOSS_HIT', ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
                
                # TRAILING STOPLOSS - lock in profits when up 1%+
                elif pnl_pct >= 1.0 and sl < entry:
                    # Move SL to breakeven
                    new_sl = entry * 1.001  # Slightly above entry
                    trade['stop_loss'] = new_sl
                    trade['trailing'] = True
                    self.tools._save_active_trades()
                    print(f"   📈 {symbol}: Trailing SL → Breakeven ₹{new_sl:.2f} (locked!)")
                
                elif pnl_pct >= 1.5 and trade.get('trailing'):
                    # Move SL to lock 0.5% profit
                    new_sl = entry * 1.005
                    if new_sl > sl:
                        trade['stop_loss'] = new_sl
                        self.tools._save_active_trades()
                        print(f"   📈 {symbol}: Trailing SL → ₹{new_sl:.2f} (0.5% locked)")
            
            else:  # SHORT position
                pnl = (entry - ltp) * qty
                pnl_pct = (entry - ltp) / entry * 100
                
                # Show status line
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (sl - ltp) / ltp * 100
                    tgt_dist = (ltp - target) / ltp * 100
                    print(f"{status} {symbol:13} {'SHORT':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%)")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
                
                # TARGET HIT (price went DOWN for shorts)
                if ltp <= target:
                    print(f"\n🎯 TARGET HIT! {symbol} (SHORT)")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    self.tools.update_trade_status(symbol, 'TARGET_HIT', ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
                
                # STOPLOSS HIT (price went UP for shorts)
                elif ltp >= sl:
                    print(f"\n❌ STOPLOSS HIT! {symbol} (SHORT)")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    self.tools.update_trade_status(symbol, 'STOPLOSS_HIT', ltp, pnl)
                    self.daily_pnl += pnl
                    self.capital += pnl
                
                # TRAILING STOPLOSS for shorts - lock in profits when up 1%+
                elif pnl_pct >= 1.0 and sl > entry:
                    # Move SL to breakeven
                    new_sl = entry * 0.999  # Slightly below entry
                    trade['stop_loss'] = new_sl
                    trade['trailing'] = True
                    self.tools._save_active_trades()
                    print(f"   📈 {symbol}: Trailing SL → Breakeven ₹{new_sl:.2f} (locked!)")
                
                elif pnl_pct >= 1.5 and trade.get('trailing'):
                    # Move SL to lock 0.5% profit
                    new_sl = entry * 0.995
                    if new_sl < sl:
                        trade['stop_loss'] = new_sl
                        self.tools._save_active_trades()
                        print(f"   📈 {symbol}: Trailing SL → ₹{new_sl:.2f} (0.5% locked)")
                    self.capital += pnl
        
        # Print summary after all positions
        if show_status and active_trades:
            total_pnl = sum(
                (quotes[t['symbol']]['last_price'] - t['avg_price']) * t['quantity'] 
                if t['side'] == 'BUY' 
                else (t['avg_price'] - quotes[t['symbol']]['last_price']) * t['quantity']
                for t in active_trades if t['symbol'] in quotes
            )
            print("-" * 85)
            print(f"📊 TOTAL UNREALIZED P&L: ₹{total_pnl:+,.0f} | Capital: ₹{self.capital:,.0f} | Daily P&L: ₹{self.daily_pnl:+,.0f}")
    
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
        
        print(f"\n🔍 {datetime.now().strftime('%H:%M:%S')} - Scanning market...")
        
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
            
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data:
                    chg = data.get('change_pct', 0)
                    rsi = data.get('rsi_14', 50)
                    trend = data.get('trend', 'N/A')
                    
                    # Get volume analysis
                    vol_data = volume_analysis.get(symbol, {})
                    order_flow = vol_data.get('order_flow', 'BALANCED')
                    eod_pred = vol_data.get('eod_prediction', 'NEUTRAL')
                    eod_conf = vol_data.get('eod_confidence', 'LOW')
                    trade_signal = vol_data.get('trade_signal', 'NO_SIGNAL')
                    
                    # Flag potential setups
                    setup = ""
                    if rsi < 30:
                        setup = "⚡OVERSOLD-BUY"
                    elif rsi > 70:
                        setup = "⚡OVERBOUGHT-SHORT"
                    elif trade_signal == "BUY_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-BUY ({order_flow})"
                        eod_opportunities.append(f"  🟢 {symbol}: EOD↑ - {order_flow}, conf:{eod_conf}")
                    elif trade_signal == "SHORT_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-SHORT ({order_flow})"
                        eod_opportunities.append(f"  🔴 {symbol}: EOD↓ - {order_flow}, conf:{eod_conf}")
                    elif chg < -1.5 and rsi > 45:
                        setup = "📉WEAK-SHORT"
                    elif chg > 1 and rsi < 55:
                        setup = "📈STRONG-BUY"
                    elif chg < 0 and rsi < 35:
                        setup = "🔄BOUNCE-BUY"
                    
                    # Mark if already in active trades
                    if self.tools.is_symbol_in_active_trades(symbol):
                        setup = "🔒ALREADY HOLDING"
                    
                    quick_scan.append(f"{symbol}: {chg:+.2f}% RSI:{rsi:.0f} {trend} {setup}")
            
            # Print EOD opportunities
            if eod_opportunities:
                print(f"\n📊 EOD VOLUME ANALYSIS:")
                for opp in eod_opportunities[:5]:  # Top 5
                    print(opp)
            
            # Get list of symbols already in positions
            active_symbols = [t['symbol'] for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            # Ask agent to analyze market with FULL CONTEXT
            prompt = f"""EXECUTE TRADES NOW using place_order tool - DO NOT just describe trades!

QUICK SCAN (all {len(quick_scan)} stocks):
{chr(10).join(quick_scan)}

DETAILED DATA (top movers):
{''.join(data_summary[:8])}

� EOD VOLUME PREDICTIONS:
{chr(10).join(eod_opportunities) if eod_opportunities else 'No clear EOD signals'}

🔒 ALREADY HOLDING (DO NOT TRADE THESE):
{', '.join(active_symbols) if active_symbols else 'None - all slots available'}

ACCOUNT:
  Capital: ₹{self.capital:,.0f}
  Today's P&L: ₹{self.daily_pnl:,.0f}
  Open positions: {len(active_symbols)}
  Max position size: ₹{self.capital * 0.3:,.0f}

TRADING STRATEGIES:

1. TECHNICAL SETUPS (RSI-based):
   - ⚡OVERSOLD-BUY = RSI < 30 at support, high probability bounce
   - ⚡OVERBOUGHT-SHORT = RSI > 70 at resistance, high probability reversal
   - 📈STRONG-BUY = Uptrend with room to run
   - 📉WEAK-SHORT = Downtrend with room to fall

2. EOD VOLUME PLAYS (order flow based):
   - 📊EOD-BUY = Buyers dominating, expect close higher
   - 📊EOD-SHORT = Sellers dominating, expect close lower
   - Order Flow shows STRONG_BUYERS or STRONG_SELLERS = high confidence
   - Use for quick scalps before 3:00 PM close

3. STOP LOSS RULES:
   - BUY: SL below support or 1% below entry
   - SHORT: SL above resistance or 1% above entry
   - EOD plays: tighter stops (0.5%) since shorter holding time

4. TARGET RULES:
   - Technical setups: 1.5:1 minimum
   - EOD plays: 0.5-1% target (quick scalp before close)

5. F&O TRADING (for expensive stocks > ₹500):
   - BULLISH → Buy ATM Call (CE)
   - BEARISH → Buy ATM Put (PE)

TIME-BASED STRATEGY:
- 9:20-11:00 AM: Focus on technical setups (RSI, support/resistance)
- 11:00-2:00 PM: Focus on EOD volume plays (order flow based)
- After 2:30 PM: Only exit, no new positions

CRITICAL: Skip any symbol in ALREADY HOLDING!

YOU MUST CALL place_order() tool NOW! Example:
place_order(symbol="NSE:INFY", side="BUY", quantity=100, stop_loss=1490, target=1550)

DO NOT just print analysis - CALL THE TOOL!"""

            response = self.agent.run(prompt)
            print(f"\n📊 Agent response:\n{response[:500]}...")
            
            # Check if agent created any trades
            pending = self.agent.get_pending_approvals()
            if pending:
                for trade in pending:
                    self._record_trade(trade)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _record_trade(self, trade: dict):
        """Record a trade (paper or real)"""
        trade['timestamp'] = datetime.now().isoformat()
        trade['paper'] = self.paper_mode
        
        if self.paper_mode:
            print(f"\n📝 PAPER TRADE: {trade['side']} {trade.get('quantity', 0)} {trade['symbol']}")
            print(f"   Entry: ₹{trade.get('entry_price', 0)}")
            print(f"   Stop: ₹{trade.get('stop_loss', 0)}")
            print(f"   Target: ₹{trade.get('target', 0)}")
        
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
