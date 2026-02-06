"""
MCX Backtesting Engine
=======================
Backtest trading strategies based on MCX stock and base metal sync.
Includes multiple strategies and performance analytics.

Author: Options Trader Tool
Date: February 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Single trade record"""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    direction: str = 'LONG'  # 'LONG' or 'SHORT'
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal_reason: str = ""
    

@dataclass
class BacktestResult:
    """Backtest performance results"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    

class MCXBacktester:
    """
    Backtesting engine for MCX sync-based trading strategies.
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.mcx_ticker = "MCX.NS"
        self.metal_etf_ticker = "DBB"
        self.data = None
        self.signals = None
        
    def fetch_historical_data(self, period='2y'):
        """Fetch historical data for backtesting"""
        print(f"📊 Fetching {period} historical data...")
        
        # MCX Stock
        mcx = yf.Ticker(self.mcx_ticker)
        mcx_data = mcx.history(period=period)
        
        # Base Metals ETF
        metal = yf.Ticker(self.metal_etf_ticker)
        metal_data = metal.history(period=period)
        
        # Clean and align
        mcx_data.index = mcx_data.index.tz_localize(None)
        metal_data.index = metal_data.index.tz_localize(None)
        mcx_data.index = pd.to_datetime(mcx_data.index.date)
        metal_data.index = pd.to_datetime(metal_data.index.date)
        
        common_dates = mcx_data.index.intersection(metal_data.index)
        
        self.data = pd.DataFrame({
            'mcx_close': mcx_data.loc[common_dates, 'Close'],
            'mcx_open': mcx_data.loc[common_dates, 'Open'],
            'mcx_high': mcx_data.loc[common_dates, 'High'],
            'mcx_low': mcx_data.loc[common_dates, 'Low'],
            'mcx_volume': mcx_data.loc[common_dates, 'Volume'],
            'metal_close': metal_data.loc[common_dates, 'Close'],
        })
        
        # Calculate indicators
        self._calculate_indicators()
        
        print(f"   ✅ Loaded {len(self.data)} days of data")
        return self.data
        
    def _calculate_indicators(self):
        """Calculate technical indicators for strategies"""
        df = self.data
        
        # Returns
        df['mcx_return'] = df['mcx_close'].pct_change()
        df['metal_return'] = df['metal_close'].pct_change()
        
        # Moving Averages
        df['mcx_sma20'] = df['mcx_close'].rolling(20).mean()
        df['mcx_sma50'] = df['mcx_close'].rolling(50).mean()
        df['metal_sma20'] = df['metal_close'].rolling(20).mean()
        
        # Relative Strength
        df['mcx_rs'] = df['mcx_close'] / df['mcx_sma20']
        df['metal_rs'] = df['metal_close'] / df['metal_sma20']
        
        # Divergence
        df['divergence'] = df['mcx_rs'] - df['metal_rs']
        df['divergence_ma'] = df['divergence'].rolling(50).mean()
        df['divergence_std'] = df['divergence'].rolling(50).std()
        df['divergence_z'] = (df['divergence'] - df['divergence_ma']) / df['divergence_std']
        
        # Rolling Correlation
        df['rolling_corr'] = df['mcx_return'].rolling(30).corr(df['metal_return'])
        
        # RSI
        delta = df['mcx_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['mcx_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['mcx_bb_mid'] = df['mcx_close'].rolling(20).mean()
        df['mcx_bb_std'] = df['mcx_close'].rolling(20).std()
        df['mcx_bb_upper'] = df['mcx_bb_mid'] + 2 * df['mcx_bb_std']
        df['mcx_bb_lower'] = df['mcx_bb_mid'] - 2 * df['mcx_bb_std']
        
        # Volume SMA
        df['volume_sma'] = df['mcx_volume'].rolling(20).mean()
        df['volume_ratio'] = df['mcx_volume'] / df['volume_sma']
        
        self.data = df
        
    def strategy_divergence_mean_reversion(self, z_entry=1.5, z_exit=0.5, max_holding=20):
        """
        Strategy 1: Divergence Mean Reversion
        - Buy when MCX underperforms metals (Z < -threshold)
        - Sell when MCX outperforms metals (Z > threshold)
        - Exit when Z reverts to exit threshold
        """
        df = self.data.copy()
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        entry_idx = None
        trades = []
        
        for i in range(50, len(df)):
            z = df['divergence_z'].iloc[i]
            
            if pd.isna(z):
                continue
                
            # Entry signals
            if position == 0:
                if z < -z_entry:  # MCX underperforming - BUY
                    position = 1
                    entry_idx = i
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    
                elif z > z_entry:  # MCX outperforming - SHORT
                    position = -1
                    entry_idx = i
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    
            # Exit signals
            elif position == 1:  # Long position
                days_held = i - entry_idx
                if z > -z_exit or days_held >= max_holding:
                    # Close long
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='LONG',
                        signal_reason='Divergence Mean Reversion'
                    )
                    trade.pnl = trade.exit_price - trade.entry_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    trades.append(trade)
                    position = 0
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    
            elif position == -1:  # Short position
                days_held = i - entry_idx
                if z < z_exit or days_held >= max_holding:
                    # Close short
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='SHORT',
                        signal_reason='Divergence Mean Reversion'
                    )
                    trade.pnl = trade.entry_price - trade.exit_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    trades.append(trade)
                    position = 0
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    
            df.iloc[i, df.columns.get_loc('position')] = position
            
        return trades, df
        
    def strategy_correlation_regime(self, corr_threshold=0.3, lookback=60):
        """
        Strategy 2: Correlation Regime Trading
        - Trade MCX only when correlation with metals is high
        - Use metal momentum as signal when correlated
        """
        df = self.data.copy()
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        entry_idx = None
        trades = []
        
        for i in range(lookback, len(df)):
            corr = df['rolling_corr'].iloc[i]
            metal_mom = df['metal_close'].iloc[i] / df['metal_close'].iloc[i-20] - 1
            
            if pd.isna(corr) or pd.isna(metal_mom):
                continue
                
            # Only trade when correlation is high
            if abs(corr) > corr_threshold:
                if position == 0:
                    if metal_mom > 0.03 and corr > 0:  # Metals up, positive corr -> BUY MCX
                        position = 1
                        entry_idx = i
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                        
                    elif metal_mom < -0.03 and corr > 0:  # Metals down, positive corr -> SHORT MCX
                        position = -1
                        entry_idx = i
                        df.iloc[i, df.columns.get_loc('signal')] = -1
                        
                elif position == 1:
                    if metal_mom < 0 or i - entry_idx >= 15:
                        trade = Trade(
                            entry_date=df.index[entry_idx],
                            entry_price=df['mcx_close'].iloc[entry_idx],
                            exit_date=df.index[i],
                            exit_price=df['mcx_close'].iloc[i],
                            direction='LONG',
                            signal_reason='Correlation Regime'
                        )
                        trade.pnl = trade.exit_price - trade.entry_price
                        trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                        trades.append(trade)
                        position = 0
                        
                elif position == -1:
                    if metal_mom > 0 or i - entry_idx >= 15:
                        trade = Trade(
                            entry_date=df.index[entry_idx],
                            entry_price=df['mcx_close'].iloc[entry_idx],
                            exit_date=df.index[i],
                            exit_price=df['mcx_close'].iloc[i],
                            direction='SHORT',
                            signal_reason='Correlation Regime'
                        )
                        trade.pnl = trade.entry_price - trade.exit_price
                        trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                        trades.append(trade)
                        position = 0
                        
            # Exit if correlation breaks down
            elif position != 0 and abs(corr) < 0.1:
                if position == 1:
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='LONG',
                        signal_reason='Correlation Regime'
                    )
                    trade.pnl = trade.exit_price - trade.entry_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                else:
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='SHORT',
                        signal_reason='Correlation Regime'
                    )
                    trade.pnl = trade.entry_price - trade.exit_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    
                trades.append(trade)
                position = 0
                
            df.iloc[i, df.columns.get_loc('position')] = position
            
        return trades, df
        
    def strategy_rsi_divergence_combo(self, rsi_oversold=30, rsi_overbought=70):
        """
        Strategy 3: RSI + Divergence Combo
        - Combine RSI signals with divergence for confirmation
        """
        df = self.data.copy()
        df['signal'] = 0
        df['position'] = 0
        
        position = 0
        entry_idx = None
        trades = []
        
        for i in range(50, len(df)):
            rsi = df['mcx_rsi'].iloc[i]
            z = df['divergence_z'].iloc[i]
            
            if pd.isna(rsi) or pd.isna(z):
                continue
                
            if position == 0:
                # BUY: RSI oversold AND MCX underperforming metals
                if rsi < rsi_oversold and z < -1.0:
                    position = 1
                    entry_idx = i
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    
                # SHORT: RSI overbought AND MCX outperforming metals
                elif rsi > rsi_overbought and z > 1.0:
                    position = -1
                    entry_idx = i
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    
            elif position == 1:
                # Exit long: RSI > 50 or max holding
                if rsi > 50 or i - entry_idx >= 20:
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='LONG',
                        signal_reason='RSI Divergence Combo'
                    )
                    trade.pnl = trade.exit_price - trade.entry_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    trades.append(trade)
                    position = 0
                    
            elif position == -1:
                if rsi < 50 or i - entry_idx >= 20:
                    trade = Trade(
                        entry_date=df.index[entry_idx],
                        entry_price=df['mcx_close'].iloc[entry_idx],
                        exit_date=df.index[i],
                        exit_price=df['mcx_close'].iloc[i],
                        direction='SHORT',
                        signal_reason='RSI Divergence Combo'
                    )
                    trade.pnl = trade.entry_price - trade.exit_price
                    trade.pnl_pct = (trade.pnl / trade.entry_price) * 100
                    trades.append(trade)
                    position = 0
                    
            df.iloc[i, df.columns.get_loc('position')] = position
            
        return trades, df
        
    def calculate_performance(self, trades: List[Trade], strategy_name: str) -> BacktestResult:
        """Calculate performance metrics from trades"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                max_drawdown=0,
                sharpe_ratio=0,
                avg_trade_pnl=0,
                avg_winner=0,
                avg_loser=0,
                profit_factor=0,
                trades=trades
            )
            
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pct = sum(t.pnl_pct for t in trades)
        
        # Equity curve
        equity = [self.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        equity_series = pd.Series(equity)
        
        # Max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized, assuming ~250 trading days)
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(250 / len(trades))
        else:
            sharpe = 0
            
        # Profit factor
        gross_profit = sum(t.pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl for t in losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(trades) * 100 if trades else 0,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            avg_trade_pnl=total_pnl / len(trades) if trades else 0,
            avg_winner=sum(t.pnl for t in winning) / len(winning) if winning else 0,
            avg_loser=sum(t.pnl for t in losing) / len(losing) if losing else 0,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_series
        )
        
    def run_all_strategies(self):
        """Run all strategies and compare results"""
        if self.data is None:
            self.fetch_historical_data()
            
        results = {}
        
        # Strategy 1: Divergence Mean Reversion
        print("\n🔄 Running Strategy 1: Divergence Mean Reversion...")
        trades1, df1 = self.strategy_divergence_mean_reversion()
        results['Divergence Mean Reversion'] = self.calculate_performance(trades1, 'Divergence Mean Reversion')
        
        # Strategy 2: Correlation Regime
        print("🔄 Running Strategy 2: Correlation Regime...")
        trades2, df2 = self.strategy_correlation_regime()
        results['Correlation Regime'] = self.calculate_performance(trades2, 'Correlation Regime')
        
        # Strategy 3: RSI Divergence Combo
        print("🔄 Running Strategy 3: RSI Divergence Combo...")
        trades3, df3 = self.strategy_rsi_divergence_combo()
        results['RSI Divergence Combo'] = self.calculate_performance(trades3, 'RSI Divergence Combo')
        
        return results
        
    def print_results(self, results: Dict[str, BacktestResult]):
        """Print backtest results comparison"""
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS COMPARISON")
        print("=" * 80)
        
        # Header
        print(f"\n{'Strategy':<30} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Max DD':>10} {'Sharpe':>8} {'PF':>8}")
        print("-" * 80)
        
        for name, result in results.items():
            print(f"{name:<30} {result.total_trades:>8} {result.win_rate:>7.1f}% "
                  f"₹{result.total_pnl:>10,.0f} {result.max_drawdown:>9.1f}% "
                  f"{result.sharpe_ratio:>8.2f} {result.profit_factor:>8.2f}")
                  
        print("-" * 80)
        
        # Best strategy
        best = max(results.values(), key=lambda x: x.total_pnl)
        print(f"\n🏆 Best Strategy: {best.strategy_name}")
        print(f"   Total P&L: ₹{best.total_pnl:,.0f}")
        print(f"   Win Rate: {best.win_rate:.1f}%")
        print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
        
    def plot_results(self, results: Dict[str, BacktestResult], save_path='backtest_results.png'):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MCX Sync Strategy Backtesting Results', fontsize=14, fontweight='bold')
        
        # 1. Equity Curves
        ax1 = axes[0, 0]
        for name, result in results.items():
            if result.equity_curve is not None:
                ax1.plot(result.equity_curve.values, label=name, linewidth=1.5)
        ax1.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curves')
        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Capital (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win Rate Comparison
        ax2 = axes[0, 1]
        names = list(results.keys())
        win_rates = [r.win_rate for r in results.values()]
        colors = ['green' if w > 50 else 'red' for w in win_rates]
        bars = ax2.bar(range(len(names)), win_rates, color=colors, alpha=0.7)
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Win Rate by Strategy')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
        
        # 3. P&L Comparison
        ax3 = axes[1, 0]
        pnls = [r.total_pnl for r in results.values()]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.bar(range(len(names)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_title('Total P&L by Strategy')
        ax3.set_ylabel('P&L (₹)')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=45, ha='right')
        
        # 4. Risk/Return Scatter
        ax4 = axes[1, 1]
        for name, result in results.items():
            ax4.scatter(abs(result.max_drawdown), result.total_pnl_pct, s=100, label=name)
        ax4.set_title('Risk vs Return')
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart saved to: {save_path}")
        plt.show()
        
        return fig


def main():
    """Run backtesting"""
    print("=" * 60)
    print("🔄 MCX SYNC STRATEGY BACKTESTER")
    print("=" * 60)
    
    # Initialize
    backtester = MCXBacktester(initial_capital=100000)
    
    # Fetch data
    backtester.fetch_historical_data(period='2y')
    
    # Run strategies
    results = backtester.run_all_strategies()
    
    # Print results
    backtester.print_results(results)
    
    # Plot results
    backtester.plot_results(results)
    
    return backtester, results


if __name__ == "__main__":
    backtester, results = main()
