"""
MCX Trading Dashboard
======================
Main dashboard integrating all modules:
- Sync Analysis
- Real-Time Alerts
- Backtesting
- Options Greeks

Author: Options Trader Tool
Date: February 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcx_sync_analyzer import MCXSyncAnalyzer
from alerts_module import MCXRealTimeMonitor, default_alert_handler
from backtesting_engine import MCXBacktester
from options_greeks import MCXOptionsAnalyzer

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class MCXTradingDashboard:
    """
    Unified trading dashboard for MCX options trading.
    Combines sync analysis, alerts, backtesting, and options Greeks.
    """
    
    def __init__(self, initial_capital=100000):
        print("=" * 70)
        print("🚀 MCX TRADING DASHBOARD - Initializing...")
        print("=" * 70)
        
        self.initial_capital = initial_capital
        
        # Initialize all modules
        self.sync_analyzer = MCXSyncAnalyzer()
        self.alert_monitor = MCXRealTimeMonitor()
        self.backtester = MCXBacktester(initial_capital)
        self.options_analyzer = MCXOptionsAnalyzer()
        
        # State
        self.last_analysis = None
        self.current_signal = None
        self.active_positions = []
        
    def run_full_analysis(self, period='1y'):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("📊 RUNNING FULL ANALYSIS PIPELINE")
        print("=" * 70)
        
        results = {}
        
        # 1. Sync Analysis
        print("\n" + "-" * 50)
        print("STEP 1: MCX-Metal Sync Analysis")
        print("-" * 50)
        sync_results = self.sync_analyzer.run_full_analysis(period)
        results['sync'] = sync_results
        
        # 2. Current Alert Status
        print("\n" + "-" * 50)
        print("STEP 2: Alert Check")
        print("-" * 50)
        alerts = self.alert_monitor.check_alerts()
        results['alerts'] = alerts
        
        # 3. Options Analysis
        print("\n" + "-" * 50)
        print("STEP 3: Options Analysis")
        print("-" * 50)
        self.options_analyzer.fetch_underlying_data()
        
        # Get divergence for strategy recommendation
        if sync_results and sync_results.get('signals') is not None:
            signals_df = sync_results['signals']
            if not signals_df.empty and 'Divergence_Z' in signals_df.columns:
                current_z = signals_df['Divergence_Z'].iloc[-1]
                if not pd.isna(current_z):
                    option_rec = self.options_analyzer.sync_based_option_strategy(current_z, 30)
                    results['option_recommendation'] = option_rec
                    self.current_signal = current_z
        
        self.last_analysis = results
        return results
        
    def run_backtest(self, period='2y'):
        """Run backtesting on all strategies"""
        print("\n" + "=" * 70)
        print("📊 RUNNING BACKTESTS")
        print("=" * 70)
        
        self.backtester.fetch_historical_data(period)
        results = self.backtester.run_all_strategies()
        self.backtester.print_results(results)
        self.backtester.plot_results(results)
        
        return results
        
    def get_option_chain(self, days_to_expiry=30):
        """Generate and display option chain"""
        print("\n" + "=" * 70)
        print(f"📊 OPTION CHAIN ({days_to_expiry} days to expiry)")
        print("=" * 70)
        
        chain = self.options_analyzer.generate_option_chain(days_to_expiry)
        self.options_analyzer.print_option_chain(chain)
        
        return chain
        
    def start_monitoring(self, interval_minutes=15):
        """Start real-time monitoring"""
        print("\n🔔 Starting Real-Time Monitoring...")
        self.alert_monitor.add_alert_callback(default_alert_handler)
        self.alert_monitor.start_monitoring(interval_minutes)
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.alert_monitor.stop_monitoring()
        
    def get_trading_recommendation(self):
        """Get current trading recommendation"""
        if self.last_analysis is None:
            self.run_full_analysis()
            
        print("\n" + "=" * 70)
        print("🎯 TRADING RECOMMENDATION")
        print("=" * 70)
        
        recommendation = {
            'timestamp': datetime.now(),
            'sync_score': None,
            'divergence_z': self.current_signal,
            'direction': None,
            'confidence': None,
            'option_strategy': None,
            'risk_level': None
        }
        
        if self.last_analysis:
            # Sync score
            if 'sync' in self.last_analysis and self.last_analysis['sync']:
                recommendation['sync_score'] = self.last_analysis['sync'].get('sync_score')
                
            # Divergence signal
            z = self.current_signal
            if z is not None and not pd.isna(z):
                if z < -2.0:
                    recommendation['direction'] = 'STRONG BUY'
                    recommendation['confidence'] = 'HIGH'
                    recommendation['risk_level'] = 'MODERATE'
                elif z < -1.5:
                    recommendation['direction'] = 'BUY'
                    recommendation['confidence'] = 'MEDIUM'
                    recommendation['risk_level'] = 'LOW'
                elif z > 2.0:
                    recommendation['direction'] = 'STRONG SELL'
                    recommendation['confidence'] = 'HIGH'
                    recommendation['risk_level'] = 'MODERATE'
                elif z > 1.5:
                    recommendation['direction'] = 'SELL'
                    recommendation['confidence'] = 'MEDIUM'
                    recommendation['risk_level'] = 'LOW'
                else:
                    recommendation['direction'] = 'NEUTRAL'
                    recommendation['confidence'] = 'LOW'
                    recommendation['risk_level'] = 'LOW'
                    
            # Option strategy
            if 'option_recommendation' in self.last_analysis:
                recommendation['option_strategy'] = self.last_analysis['option_recommendation']
                
        # Print recommendation
        print(f"\n📅 Timestamp: {recommendation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Sync Score: {recommendation['sync_score']:.3f}" if recommendation['sync_score'] else "")
        print(f"📈 Divergence Z: {recommendation['divergence_z']:.2f}" if recommendation['divergence_z'] else "")
        print(f"\n🎯 DIRECTION: {recommendation['direction']}")
        print(f"💪 Confidence: {recommendation['confidence']}")
        print(f"⚠️ Risk Level: {recommendation['risk_level']}")
        
        if recommendation['option_strategy']:
            print(f"\n📋 Option Strategy: {recommendation['option_strategy']['strategy']}")
            if 'details' in recommendation['option_strategy']:
                for key, val in recommendation['option_strategy']['details'].items():
                    if isinstance(val, float):
                        print(f"   • {key}: ₹{val:.2f}")
                    else:
                        print(f"   • {key}: {val}")
                        
        return recommendation
        
    def generate_report(self, save_path='trading_report.txt'):
        """Generate comprehensive trading report"""
        if self.last_analysis is None:
            self.run_full_analysis()
            
        report = []
        report.append("=" * 70)
        report.append("MCX TRADING REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Sync Analysis
        report.append("\n1. SYNC ANALYSIS")
        report.append("-" * 50)
        if self.last_analysis and 'sync' in self.last_analysis:
            sync = self.last_analysis['sync']
            if sync:
                report.append(f"Sync Score: {sync.get('sync_score', 'N/A')}")
                if 'correlations' in sync:
                    report.append("\nCorrelations:")
                    for name, data in sync['correlations'].items():
                        report.append(f"  • {name}: r={data['correlation']:.3f}")
                        
        # Current Signal
        report.append("\n2. CURRENT SIGNAL")
        report.append("-" * 50)
        if self.current_signal:
            report.append(f"Divergence Z-Score: {self.current_signal:.2f}")
            if self.current_signal < -1.5:
                report.append("Signal: BUY (MCX underperforming)")
            elif self.current_signal > 1.5:
                report.append("Signal: SELL (MCX outperforming)")
            else:
                report.append("Signal: NEUTRAL")
                
        # Options
        report.append("\n3. OPTIONS RECOMMENDATION")
        report.append("-" * 50)
        if self.last_analysis and 'option_recommendation' in self.last_analysis:
            opt = self.last_analysis['option_recommendation']
            report.append(f"Strategy: {opt['strategy']}")
            report.append(f"Signal: {opt['signal']}")
            
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
            
        print(f"\n📄 Report saved to: {save_path}")
        return report_text


def main():
    """Main dashboard execution"""
    print("\n" + "=" * 70)
    print("🚀 MCX OPTIONS TRADING DASHBOARD")
    print("   Base Metal Sync Analysis | Alerts | Backtesting | Greeks")
    print("=" * 70)
    
    # Initialize dashboard
    dashboard = MCXTradingDashboard(initial_capital=100000)
    
    # Menu
    while True:
        print("\n" + "-" * 50)
        print("📋 MENU")
        print("-" * 50)
        print("1. Run Full Analysis")
        print("2. Run Backtests")
        print("3. View Option Chain")
        print("4. Get Trading Recommendation")
        print("5. Start Real-Time Monitoring")
        print("6. Generate Report")
        print("7. Quick Analysis (All)")
        print("0. Exit")
        print("-" * 50)
        
        try:
            choice = input("Enter choice (0-7): ").strip()
        except:
            choice = '7'  # Default to quick analysis for non-interactive
            
        if choice == '1':
            dashboard.run_full_analysis()
        elif choice == '2':
            dashboard.run_backtest()
        elif choice == '3':
            dashboard.get_option_chain()
        elif choice == '4':
            dashboard.get_trading_recommendation()
        elif choice == '5':
            dashboard.start_monitoring(15)
            input("Press Enter to stop monitoring...")
            dashboard.stop_monitoring()
        elif choice == '6':
            dashboard.generate_report()
        elif choice == '7':
            # Quick analysis - run everything
            print("\n🔄 Running Quick Analysis (All Components)...")
            dashboard.run_full_analysis('1y')
            dashboard.get_option_chain(30)
            dashboard.get_trading_recommendation()
            dashboard.generate_report()
            break
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            
    return dashboard


if __name__ == "__main__":
    dashboard = main()
