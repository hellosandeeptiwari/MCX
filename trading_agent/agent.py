"""
TRADING AGENT - MAIN ORCHESTRATOR
Coordinates all modules to run the automated trading system

Features:
- Scans universe for signals
- Validates with risk manager
- Executes trades
- Monitors positions (SL/Target)
- WebSocket for real-time price updates
- Graceful shutdown
"""

import time
import threading
import signal as os_signal
import sys
from datetime import datetime, timedelta
from typing import Optional
import json

# Import all modules
from config import (
    UNIVERSE, CANDLE_INTERVAL, 
    MARKET_OPEN, MARKET_CLOSE, NO_NEW_TRADES_AFTER
)
from data_manager import get_data_manager
from signal_engine import get_signal_engine, SignalType
from risk_manager import get_risk_manager
from execution_engine import get_execution_engine


class TradingAgent:
    """
    Main trading agent that orchestrates everything
    """
    
    def __init__(self, dry_run: bool = True):
        """
        Initialize trading agent
        
        Args:
            dry_run: If True, simulate trades without real execution
        """
        self.dry_run = dry_run
        self.running = False
        self.scan_interval = 60  # seconds between scans
        self.monitor_interval = 5  # seconds between position checks
        
        # Initialize modules
        print("🚀 Initializing Trading Agent...")
        self.dm = get_data_manager()
        self.se = get_signal_engine()
        self.rm = get_risk_manager()
        self.ee = get_execution_engine()
        
        # Threads
        self.scanner_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Register signal handlers for graceful shutdown
        os_signal.signal(os_signal.SIGINT, self._signal_handler)
        os_signal.signal(os_signal.SIGTERM, self._signal_handler)
        
        print(f"   Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
        print(f"   Universe: {len(UNIVERSE)} symbols")
        print(f"   Risk per trade: {self.rm.capital * 0.005:,.0f}")
        print("✅ Agent initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n⚠️ Shutdown signal received...")
        self.stop()
    
    def is_market_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now().time()
        market_open = datetime.strptime(MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
        return market_open <= now <= market_close
    
    def start(self):
        """Start the trading agent"""
        if not self.dm.is_authenticated():
            print("❌ Not authenticated with Zerodha. Please login first.")
            return False
        
        # Check if market is open
        if not self.is_market_hours():
            print(f"⚠️ Market is closed. Opens at {MARKET_OPEN}")
            # Still allow to run for testing
        
        self.running = True
        print(f"\n{'='*60}")
        print(f"🚀 TRADING AGENT STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Sync positions with broker
        self.ee.sync_positions_with_broker()
        
        # Start WebSocket for real-time prices
        self.dm.register_tick_callback(self._on_tick)
        self.dm.start_websocket()
        
        # Start scanner thread
        self.scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self.scanner_thread.start()
        
        # Start position monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the trading agent gracefully"""
        print("\n🛑 Stopping Trading Agent...")
        self.running = False
        
        # Stop WebSocket
        self.dm.stop_websocket()
        
        # Wait for threads
        if self.scanner_thread and self.scanner_thread.is_alive():
            self.scanner_thread.join(timeout=5)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Print summary
        self._print_summary()
        
        print("✅ Agent stopped")
    
    def _on_tick(self, ticks):
        """Callback for real-time price updates"""
        for tick in ticks:
            symbol = tick.get('symbol')
            ltp = tick.get('last_price', 0)
            
            if symbol and ltp > 0:
                # Update position P&L
                self.rm.update_position_pnl(symbol, ltp)
    
    def _scanner_loop(self):
        """Background thread that scans for signals"""
        print("📡 Scanner started")
        last_scan = datetime.min
        
        while self.running:
            try:
                now = datetime.now()
                
                # Skip if not market hours
                if not self.is_market_hours():
                    time.sleep(30)
                    continue
                
                # Skip if kill switch is active
                if self.rm.kill_switch:
                    time.sleep(30)
                    continue
                
                # Check if time to scan
                if (now - last_scan).seconds < self.scan_interval:
                    time.sleep(5)
                    continue
                
                last_scan = now
                self._run_scan()
                
            except Exception as e:
                print(f"❌ Scanner error: {e}")
                time.sleep(10)
        
        print("📡 Scanner stopped")
    
    def _run_scan(self):
        """Run one scan cycle"""
        print(f"\n🔍 Scanning universe at {datetime.now().strftime('%H:%M:%S')}...")
        
        # Get historical data for all symbols
        ohlc_data = {}
        for symbol in UNIVERSE[:10]:  # Limit for testing
            df = self.dm.get_historical_data(symbol, CANDLE_INTERVAL, days=5)
            if not df.empty:
                ohlc_data[symbol] = df
        
        if not ohlc_data:
            print("   No data available")
            return
        
        # Scan for signals
        signals = self.se.scan_universe(ohlc_data)
        
        if not signals:
            print("   No signals found")
            return
        
        print(f"   Found {len(signals)} signals:")
        
        for signal in signals[:3]:  # Top 3 signals
            emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
            print(f"   {emoji} {signal.symbol}: {signal.signal_type.value} (strength: {signal.strength:.0f})")
            print(f"      Entry: ₹{signal.entry_price:,.2f} | SL: ₹{signal.stop_loss:,.2f} | Target: ₹{signal.target:,.2f}")
            print(f"      Reason: {signal.reason}")
            
            # Execute if not in dry run
            if signal.strength >= 70:  # Only high-confidence signals
                if self.rm.can_take_new_trade():
                    self.ee.execute_signal(signal, dry_run=self.dry_run)
                else:
                    print(f"      ⏸️ Skipped - No new trades after {NO_NEW_TRADES_AFTER}")
    
    def _monitor_loop(self):
        """Background thread that monitors open positions"""
        print("👀 Position monitor started")
        
        while self.running:
            try:
                positions = self.rm.get_open_positions()
                
                for pos in positions:
                    ltp = self.dm.get_cached_ltp(pos.symbol)
                    if ltp <= 0:
                        continue
                    
                    # Check SL/Target hit
                    hit = self.rm.check_sl_target(pos.symbol, ltp)
                    
                    if hit == "SL_HIT":
                        print(f"🛑 SL HIT: {pos.symbol} @ ₹{ltp:,.2f}")
                        if not self.dry_run:
                            self.ee.exit_position(pos.symbol, "SL_HIT")
                        else:
                            self.rm.close_position(pos.symbol, ltp, "SL_HIT")
                    
                    elif hit == "TARGET_HIT":
                        print(f"🎯 TARGET HIT: {pos.symbol} @ ₹{ltp:,.2f}")
                        if not self.dry_run:
                            self.ee.exit_position(pos.symbol, "TARGET_HIT")
                        else:
                            self.rm.close_position(pos.symbol, ltp, "TARGET_HIT")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(5)
        
        print("👀 Position monitor stopped")
    
    def _print_summary(self):
        """Print trading summary"""
        summary = self.rm.get_risk_summary()
        
        print(f"\n{'='*60}")
        print("📊 TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"   Trades Today: {summary['trades_today']}")
        print(f"   Winning: {self.rm.today_stats.winning_trades}")
        print(f"   Losing: {self.rm.today_stats.losing_trades}")
        print(f"   Win Rate: {summary['win_rate']}%")
        print(f"")
        print(f"   Realized P&L: ₹{summary['realized_pnl']:,.2f}")
        print(f"   Unrealized P&L: ₹{summary['unrealized_pnl']:,.2f}")
        print(f"   Total P&L: ₹{summary['total_pnl']:,.2f}")
        print(f"   Max Drawdown: ₹{summary['max_drawdown']:,.2f}")
        print(f"{'='*60}")
    
    def manual_scan(self):
        """Manually trigger a scan (for testing)"""
        self._run_scan()
    
    def get_status(self) -> dict:
        """Get current agent status"""
        return {
            'running': self.running,
            'dry_run': self.dry_run,
            'websocket_connected': self.dm.connected,
            'market_hours': self.is_market_hours(),
            'kill_switch': self.rm.kill_switch,
            'risk_summary': self.rm.get_risk_summary(),
            'open_positions': [p.to_dict() for p in self.rm.get_open_positions()]
        }
    
    def activate_kill_switch(self, reason: str = "Manual"):
        """Activate kill switch - stop all trading"""
        self.rm.activate_kill_switch(reason)
        
        # Exit all positions if desired
        # self.ee.exit_all_positions("KILL_SWITCH")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch - resume trading"""
        self.rm.deactivate_kill_switch()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Agent')
    parser.add_argument('--live', action='store_true', help='Run in LIVE mode (real orders)')
    parser.add_argument('--scan', action='store_true', help='Run single scan and exit')
    args = parser.parse_args()
    
    # Default to dry run for safety
    dry_run = not args.live
    
    if not dry_run:
        print("⚠️ WARNING: LIVE TRADING MODE")
        print("   Real orders will be placed!")
        confirm = input("   Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("Aborted.")
            return
    
    agent = TradingAgent(dry_run=dry_run)
    
    if args.scan:
        # Single scan mode
        agent.manual_scan()
    else:
        # Full agent mode
        agent.start()


if __name__ == "__main__":
    main()
