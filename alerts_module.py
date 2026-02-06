"""
MCX Real-Time Alerts Module
============================
Monitors MCX stock and base metal divergence in real-time
and triggers alerts when thresholds are exceeded.

Author: Options Trader Tool
Date: February 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import winsound
import threading
from dataclasses import dataclass
from typing import Callable, Optional, List
import json
import os


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    alert_type: str
    symbol: str
    message: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    value: float
    threshold: float
    action: str


class AlertManager:
    """Manages alert thresholds and notifications"""
    
    def __init__(self):
        self.alerts_history: List[Alert] = []
        self.callbacks: List[Callable] = []
        self.alert_file = "alerts_log.json"
        
    def add_callback(self, callback: Callable):
        """Add a callback function to be called when alert triggers"""
        self.callbacks.append(callback)
        
    def trigger_alert(self, alert: Alert):
        """Trigger an alert and notify all callbacks"""
        self.alerts_history.append(alert)
        self._save_alert(alert)
        self._notify(alert)
        
        # Sound alert for critical
        if alert.severity == 'CRITICAL':
            self._sound_alert()
            
    def _notify(self, alert: Alert):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
                
    def _sound_alert(self):
        """Play sound for critical alerts (Windows)"""
        try:
            winsound.Beep(1000, 500)  # 1000Hz for 500ms
        except:
            pass
            
    def _save_alert(self, alert: Alert):
        """Save alert to file"""
        alert_dict = {
            'timestamp': alert.timestamp.isoformat(),
            'alert_type': alert.alert_type,
            'symbol': alert.symbol,
            'message': alert.message,
            'severity': alert.severity,
            'value': alert.value,
            'threshold': alert.threshold,
            'action': alert.action
        }
        
        try:
            if os.path.exists(self.alert_file):
                with open(self.alert_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
                
            data.append(alert_dict)
            
            with open(self.alert_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving alert: {e}")


class MCXRealTimeMonitor:
    """
    Real-time monitor for MCX stock and base metal divergence.
    Triggers alerts when thresholds are exceeded.
    """
    
    def __init__(self):
        self.mcx_ticker = "MCX.NS"
        self.metal_etf_ticker = "DBB"  # Base Metals ETF
        
        # Alert thresholds
        self.thresholds = {
            'divergence_warning': 1.5,      # Z-score for warning
            'divergence_critical': 2.0,     # Z-score for critical alert
            'price_change_warning': 3.0,    # % change for warning
            'price_change_critical': 5.0,   # % change for critical
            'correlation_breakdown': 0.0,   # When correlation turns negative
            'volume_spike': 2.0,            # Volume > 2x average
        }
        
        self.alert_manager = AlertManager()
        self.is_monitoring = False
        self.monitor_thread = None
        self.historical_data = {}
        self.last_check = None
        
    def set_threshold(self, threshold_name: str, value: float):
        """Update an alert threshold"""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            print(f"✅ Threshold '{threshold_name}' set to {value}")
        else:
            print(f"❌ Unknown threshold: {threshold_name}")
            
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self.alert_manager.add_callback(callback)
        
    def fetch_current_data(self):
        """Fetch latest price data"""
        try:
            # MCX Stock
            mcx = yf.Ticker(self.mcx_ticker)
            mcx_data = mcx.history(period='5d', interval='1h')
            
            # Base Metals ETF
            metal = yf.Ticker(self.metal_etf_ticker)
            metal_data = metal.history(period='5d', interval='1h')
            
            return {
                'mcx': mcx_data,
                'metal': metal_data,
                'mcx_price': mcx_data['Close'].iloc[-1] if not mcx_data.empty else None,
                'metal_price': metal_data['Close'].iloc[-1] if not metal_data.empty else None,
                'mcx_volume': mcx_data['Volume'].iloc[-1] if not mcx_data.empty else None,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
            
    def calculate_divergence(self, lookback_days=50):
        """Calculate current divergence Z-score"""
        try:
            # Get historical data
            mcx = yf.Ticker(self.mcx_ticker)
            metal = yf.Ticker(self.metal_etf_ticker)
            
            mcx_hist = mcx.history(period='3mo')
            metal_hist = metal.history(period='3mo')
            
            # Remove timezone
            mcx_hist.index = mcx_hist.index.tz_localize(None)
            metal_hist.index = metal_hist.index.tz_localize(None)
            
            # Normalize to date
            mcx_hist.index = pd.to_datetime(mcx_hist.index.date)
            metal_hist.index = pd.to_datetime(metal_hist.index.date)
            
            # Align
            common = mcx_hist.index.intersection(metal_hist.index)
            mcx_aligned = mcx_hist.loc[common]
            metal_aligned = metal_hist.loc[common]
            
            # Calculate relative strength
            mcx_rs = mcx_aligned['Close'] / mcx_aligned['Close'].rolling(20).mean()
            metal_rs = metal_aligned['Close'] / metal_aligned['Close'].rolling(20).mean()
            
            # Divergence
            divergence = mcx_rs - metal_rs
            divergence_z = (divergence - divergence.rolling(lookback_days).mean()) / divergence.rolling(lookback_days).std()
            
            return {
                'current_divergence': divergence.iloc[-1],
                'divergence_z': divergence_z.iloc[-1],
                'mean_divergence': divergence.rolling(lookback_days).mean().iloc[-1],
                'std_divergence': divergence.rolling(lookback_days).std().iloc[-1]
            }
        except Exception as e:
            print(f"Error calculating divergence: {e}")
            return None
            
    def calculate_rolling_correlation(self, window=30):
        """Calculate rolling correlation"""
        try:
            mcx = yf.Ticker(self.mcx_ticker)
            metal = yf.Ticker(self.metal_etf_ticker)
            
            mcx_hist = mcx.history(period='3mo')
            metal_hist = metal.history(period='3mo')
            
            # Align data
            mcx_hist.index = mcx_hist.index.tz_localize(None)
            metal_hist.index = metal_hist.index.tz_localize(None)
            mcx_hist.index = pd.to_datetime(mcx_hist.index.date)
            metal_hist.index = pd.to_datetime(metal_hist.index.date)
            
            common = mcx_hist.index.intersection(metal_hist.index)
            
            mcx_returns = mcx_hist.loc[common]['Close'].pct_change()
            metal_returns = metal_hist.loc[common]['Close'].pct_change()
            
            rolling_corr = mcx_returns.rolling(window).corr(metal_returns)
            
            return {
                'current_correlation': rolling_corr.iloc[-1],
                'avg_correlation': rolling_corr.mean(),
                'correlation_series': rolling_corr
            }
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return None
            
    def check_alerts(self):
        """Check all alert conditions"""
        alerts_triggered = []
        
        print(f"\n⏰ Checking alerts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Check divergence
        div_data = self.calculate_divergence()
        if div_data and not np.isnan(div_data['divergence_z']):
            z_score = div_data['divergence_z']
            
            if abs(z_score) >= self.thresholds['divergence_critical']:
                action = "SELL MCX" if z_score > 0 else "BUY MCX"
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type='DIVERGENCE',
                    symbol='MCX.NS',
                    message=f"CRITICAL: Divergence Z-score at {z_score:.2f}",
                    severity='CRITICAL',
                    value=z_score,
                    threshold=self.thresholds['divergence_critical'],
                    action=action
                )
                self.alert_manager.trigger_alert(alert)
                alerts_triggered.append(alert)
                print(f"   🔴 CRITICAL: Divergence Z={z_score:.2f} | Action: {action}")
                
            elif abs(z_score) >= self.thresholds['divergence_warning']:
                action = "Consider SELL MCX" if z_score > 0 else "Consider BUY MCX"
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type='DIVERGENCE',
                    symbol='MCX.NS',
                    message=f"WARNING: Divergence Z-score at {z_score:.2f}",
                    severity='WARNING',
                    value=z_score,
                    threshold=self.thresholds['divergence_warning'],
                    action=action
                )
                self.alert_manager.trigger_alert(alert)
                alerts_triggered.append(alert)
                print(f"   🟡 WARNING: Divergence Z={z_score:.2f} | Action: {action}")
            else:
                print(f"   ✅ Divergence Z={z_score:.2f} (within normal range)")
                
        # 2. Check correlation breakdown
        corr_data = self.calculate_rolling_correlation()
        if corr_data and not np.isnan(corr_data['current_correlation']):
            current_corr = corr_data['current_correlation']
            
            if current_corr <= self.thresholds['correlation_breakdown']:
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type='CORRELATION_BREAKDOWN',
                    symbol='MCX.NS',
                    message=f"Correlation has turned negative: {current_corr:.3f}",
                    severity='WARNING',
                    value=current_corr,
                    threshold=self.thresholds['correlation_breakdown'],
                    action="Monitor closely - relationship breakdown"
                )
                self.alert_manager.trigger_alert(alert)
                alerts_triggered.append(alert)
                print(f"   🟡 WARNING: Correlation breakdown r={current_corr:.3f}")
            else:
                print(f"   ✅ Correlation r={current_corr:.3f}")
                
        # 3. Check price changes
        current_data = self.fetch_current_data()
        if current_data and current_data.get('mcx') is not None and not current_data['mcx'].empty:
            mcx_df = current_data['mcx']
            if len(mcx_df) >= 2:
                pct_change = ((mcx_df['Close'].iloc[-1] - mcx_df['Close'].iloc[0]) / mcx_df['Close'].iloc[0]) * 100
                
                if abs(pct_change) >= self.thresholds['price_change_critical']:
                    direction = "UP" if pct_change > 0 else "DOWN"
                    alert = Alert(
                        timestamp=datetime.now(),
                        alert_type='PRICE_MOVEMENT',
                        symbol='MCX.NS',
                        message=f"CRITICAL: MCX stock {direction} {abs(pct_change):.2f}% in 5 days",
                        severity='CRITICAL',
                        value=pct_change,
                        threshold=self.thresholds['price_change_critical'],
                        action=f"Large move {direction} - Review positions"
                    )
                    self.alert_manager.trigger_alert(alert)
                    alerts_triggered.append(alert)
                    print(f"   🔴 CRITICAL: Price change {pct_change:+.2f}%")
                elif abs(pct_change) >= self.thresholds['price_change_warning']:
                    print(f"   🟡 WARNING: Price change {pct_change:+.2f}%")
                else:
                    print(f"   ✅ Price change {pct_change:+.2f}%")
                    
        self.last_check = datetime.now()
        return alerts_triggered
        
    def start_monitoring(self, interval_minutes=15):
        """Start real-time monitoring in background thread"""
        if self.is_monitoring:
            print("Already monitoring!")
            return
            
        self.is_monitoring = True
        
        def monitor_loop():
            print(f"\n🚀 Starting real-time monitoring (interval: {interval_minutes} minutes)")
            print("=" * 50)
            
            while self.is_monitoring:
                try:
                    self.check_alerts()
                except Exception as e:
                    print(f"Monitor error: {e}")
                    
                # Wait for next check
                for _ in range(interval_minutes * 60):
                    if not self.is_monitoring:
                        break
                    time.sleep(1)
                    
            print("\n🛑 Monitoring stopped")
            
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        print("Stopping monitor...")
        
    def get_alert_summary(self):
        """Get summary of recent alerts"""
        if not self.alert_manager.alerts_history:
            return "No alerts triggered yet."
            
        summary = "\n📋 ALERT SUMMARY\n" + "=" * 40 + "\n"
        
        # Group by severity
        critical = [a for a in self.alert_manager.alerts_history if a.severity == 'CRITICAL']
        warning = [a for a in self.alert_manager.alerts_history if a.severity == 'WARNING']
        
        summary += f"\n🔴 Critical Alerts: {len(critical)}\n"
        for alert in critical[-5:]:  # Last 5
            summary += f"   • {alert.timestamp.strftime('%m/%d %H:%M')} - {alert.message}\n"
            
        summary += f"\n🟡 Warning Alerts: {len(warning)}\n"
        for alert in warning[-5:]:
            summary += f"   • {alert.timestamp.strftime('%m/%d %H:%M')} - {alert.message}\n"
            
        return summary


def default_alert_handler(alert: Alert):
    """Default handler to print alerts to console"""
    emoji = "🔴" if alert.severity == 'CRITICAL' else "🟡" if alert.severity == 'WARNING' else "ℹ️"
    print(f"\n{emoji} ALERT: {alert.message}")
    print(f"   Action: {alert.action}")
    print(f"   Value: {alert.value:.3f} (Threshold: {alert.threshold})")


def main():
    """Demo of the alert system"""
    print("=" * 60)
    print("🔔 MCX REAL-TIME ALERT SYSTEM")
    print("=" * 60)
    
    # Initialize monitor
    monitor = MCXRealTimeMonitor()
    
    # Add callback
    monitor.add_alert_callback(default_alert_handler)
    
    # Set custom thresholds (optional)
    monitor.set_threshold('divergence_warning', 1.5)
    monitor.set_threshold('divergence_critical', 2.0)
    
    # Run single check
    print("\n📊 Running alert check...")
    alerts = monitor.check_alerts()
    
    print("\n" + monitor.get_alert_summary())
    
    # For continuous monitoring, uncomment:
    # monitor.start_monitoring(interval_minutes=15)
    # input("Press Enter to stop monitoring...")
    # monitor.stop_monitoring()
    
    return monitor


if __name__ == "__main__":
    monitor = main()
