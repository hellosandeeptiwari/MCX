"""
MCX Base Metal Prices vs MCX Ltd Stock Price Sync Analyzer
============================================================
This algorithm analyzes the correlation between MCX base metal prices
and MCX Ltd stock price on NSE.

Author: Options Trader Tool
Date: February 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MCXSyncAnalyzer:
    """
    Analyzes the relationship between MCX base metal prices and MCX Ltd stock.
    """
    
    def __init__(self):
        # MCX Ltd stock ticker on NSE
        self.mcx_stock_ticker = "MCX.NS"
        
        # Base metal futures tickers (using global proxies as MCX API is not freely available)
        # These are highly correlated with MCX prices
        self.base_metals = {
            'Copper': 'HG=F',      # COMEX Copper
            'Aluminum': 'ALI=F',   # Aluminum Futures
            'Zinc': 'ZNC=F',       # Zinc Futures (may use proxy)
            'Nickel': '^SPGSIKTR', # S&P GSCI Nickel Index
            'Lead': 'LEAD.L',      # Lead (London)
        }
        
        # Alternative: Use ETFs that track base metals
        self.metal_etfs = {
            'Base Metals ETF': 'DBB',      # Invesco DB Base Metals Fund
            'Copper ETF': 'CPER',          # United States Copper Index Fund
            'Industrial Metals': 'XME',    # SPDR S&P Metals & Mining ETF
        }
        
        self.data = {}
        self.correlation_results = {}
        
    def fetch_mcx_stock_data(self, period='1y'):
        """Fetch MCX Ltd stock data from NSE via yfinance"""
        print(f"\n📈 Fetching MCX Ltd stock data...")
        try:
            mcx = yf.Ticker(self.mcx_stock_ticker)
            df = mcx.history(period=period)
            if not df.empty:
                self.data['MCX_Stock'] = df
                print(f"   ✅ MCX Ltd: {len(df)} days of data fetched")
                print(f"   📊 Latest Price: ₹{df['Close'].iloc[-1]:.2f}")
                return df
            else:
                print("   ❌ No data available for MCX Ltd")
                return None
        except Exception as e:
            print(f"   ❌ Error fetching MCX stock: {e}")
            return None
    
    def fetch_base_metal_data(self, period='1y'):
        """Fetch base metal prices"""
        print(f"\n🔧 Fetching Base Metal data...")
        
        # Try primary metal futures
        for metal_name, ticker in self.base_metals.items():
            try:
                metal = yf.Ticker(ticker)
                df = metal.history(period=period)
                if not df.empty and len(df) > 10:
                    self.data[metal_name] = df
                    print(f"   ✅ {metal_name}: {len(df)} days | Latest: ${df['Close'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"   ⚠️ {metal_name}: Not available ({ticker})")
        
        # Fetch ETFs as reliable proxies
        print(f"\n📦 Fetching Metal ETFs (reliable proxies)...")
        for etf_name, ticker in self.metal_etfs.items():
            try:
                etf = yf.Ticker(ticker)
                df = etf.history(period=period)
                if not df.empty:
                    self.data[etf_name] = df
                    print(f"   ✅ {etf_name}: {len(df)} days | Latest: ${df['Close'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"   ⚠️ {etf_name}: Not available")
        
        return self.data
    
    def calculate_correlations(self):
        """Calculate correlation between MCX stock and base metals"""
        print(f"\n📊 Calculating Correlations...")
        
        if 'MCX_Stock' not in self.data:
            print("   ❌ MCX Stock data not available")
            return None
        
        # Remove timezone info for proper alignment
        mcx_df = self.data['MCX_Stock'].copy()
        mcx_df.index = mcx_df.index.tz_localize(None)
        mcx_returns = mcx_df['Close'].pct_change().dropna()
        
        correlations = {}
        
        for name, df in self.data.items():
            if name == 'MCX_Stock':
                continue
                
            try:
                # Remove timezone info
                metal_df = df.copy()
                metal_df.index = metal_df.index.tz_localize(None)
                metal_returns = metal_df['Close'].pct_change().dropna()
                
                # Align dates by date only (ignore time)
                mcx_returns.index = pd.to_datetime(mcx_returns.index.date)
                metal_returns.index = pd.to_datetime(metal_returns.index.date)
                
                common_dates = mcx_returns.index.intersection(metal_returns.index)
                if len(common_dates) < 20:
                    print(f"   ⚠️ {name}: Only {len(common_dates)} common dates, skipping...")
                    continue
                    
                mcx_aligned = mcx_returns.loc[common_dates]
                metal_aligned = metal_returns.loc[common_dates]
                
                # Calculate Pearson correlation
                corr, p_value = stats.pearsonr(mcx_aligned, metal_aligned)
                
                # Calculate rolling correlation (30-day)
                combined = pd.DataFrame({
                    'MCX': mcx_aligned,
                    'Metal': metal_aligned
                })
                rolling_corr = combined['MCX'].rolling(30).corr(combined['Metal'])
                
                correlations[name] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'rolling_corr': rolling_corr,
                    'data_points': len(common_dates)
                }
                
                significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
                print(f"   {name}: r = {corr:.3f} (p={p_value:.4f}) {significance}")
                
            except Exception as e:
                print(f"   ⚠️ {name}: Could not calculate correlation - {e}")
        
        self.correlation_results = correlations
        return correlations
    
    def calculate_sync_score(self):
        """Calculate overall sync score between MCX stock and base metals"""
        print(f"\n🎯 Calculating Sync Score...")
        
        if not self.correlation_results:
            print("   ❌ No correlation data available")
            return None
        
        # Weighted average of significant correlations
        weights = {
            'Base Metals ETF': 0.4,  # Highest weight - most representative
            'Copper': 0.2,
            'Copper ETF': 0.15,
            'Industrial Metals': 0.15,
            'Aluminum': 0.05,
            'Zinc': 0.025,
            'Nickel': 0.025,
        }
        
        total_weight = 0
        weighted_corr = 0
        
        for name, result in self.correlation_results.items():
            if name in weights and result['significant']:
                w = weights[name]
                weighted_corr += result['correlation'] * w
                total_weight += w
        
        if total_weight > 0:
            sync_score = weighted_corr / total_weight
        else:
            # Simple average if no weights match
            correlations = [r['correlation'] for r in self.correlation_results.values()]
            sync_score = np.mean(correlations) if correlations else 0
        
        print(f"\n   📈 Overall Sync Score: {sync_score:.3f}")
        
        if sync_score > 0.5:
            print(f"   💚 STRONG POSITIVE SYNC - Metal prices and MCX stock move together")
        elif sync_score > 0.2:
            print(f"   💛 MODERATE POSITIVE SYNC - Some relationship exists")
        elif sync_score > -0.2:
            print(f"   ⚪ WEAK/NO SYNC - Prices move independently")
        elif sync_score > -0.5:
            print(f"   🟠 MODERATE NEGATIVE SYNC - Inverse relationship")
        else:
            print(f"   🔴 STRONG NEGATIVE SYNC - Prices move opposite")
        
        return sync_score
    
    def generate_trading_signals(self):
        """Generate trading signals based on sync analysis"""
        print(f"\n🚦 Generating Trading Signals...")
        
        if 'MCX_Stock' not in self.data or 'Base Metals ETF' not in self.data:
            print("   ❌ Insufficient data for signals")
            return None
        
        mcx_df = self.data['MCX_Stock'].copy()
        metal_df = self.data['Base Metals ETF'].copy()
        
        # Remove timezone info
        mcx_df.index = mcx_df.index.tz_localize(None)
        metal_df.index = metal_df.index.tz_localize(None)
        
        # Convert to date only for alignment
        mcx_df.index = pd.to_datetime(mcx_df.index.date)
        metal_df.index = pd.to_datetime(metal_df.index.date)
        
        # Align data
        common_dates = mcx_df.index.intersection(metal_df.index)
        
        if len(common_dates) == 0:
            print("   ❌ No common dates found between MCX stock and metals data")
            return None
            
        mcx_aligned = mcx_df.loc[common_dates].copy()
        metal_aligned = metal_df.loc[common_dates].copy()
        
        # Calculate indicators
        signals = pd.DataFrame(index=common_dates)
        signals['MCX_Price'] = mcx_aligned['Close']
        signals['Metal_Price'] = metal_aligned['Close']
        
        # Returns
        signals['MCX_Return'] = signals['MCX_Price'].pct_change()
        signals['Metal_Return'] = signals['Metal_Price'].pct_change()
        
        # Moving averages
        signals['MCX_MA20'] = signals['MCX_Price'].rolling(20).mean()
        signals['Metal_MA20'] = signals['Metal_Price'].rolling(20).mean()
        
        # Relative Strength
        signals['MCX_RS'] = signals['MCX_Price'] / signals['MCX_MA20']
        signals['Metal_RS'] = signals['Metal_Price'] / signals['Metal_MA20']
        
        # Divergence (when they don't move together)
        signals['Divergence'] = signals['MCX_RS'] - signals['Metal_RS']
        signals['Divergence_Z'] = (signals['Divergence'] - signals['Divergence'].rolling(50).mean()) / signals['Divergence'].rolling(50).std()
        
        # Generate signals
        def get_signal(row):
            if pd.isna(row['Divergence_Z']):
                return 'HOLD'
            
            # If MCX stock is underperforming metals significantly
            if row['Divergence_Z'] < -1.5:
                return 'BUY MCX'  # Mean reversion expected
            # If MCX stock is outperforming metals significantly
            elif row['Divergence_Z'] > 1.5:
                return 'SELL MCX'  # Mean reversion expected
            else:
                return 'HOLD'
        
        signals['Signal'] = signals.apply(get_signal, axis=1)
        
        # Current signal
        latest = signals.iloc[-1]
        print(f"\n   📅 Latest Date: {signals.index[-1].strftime('%Y-%m-%d')}")
        print(f"   💰 MCX Stock Price: ₹{latest['MCX_Price']:.2f}")
        print(f"   🔧 Base Metal ETF: ${latest['Metal_Price']:.2f}")
        print(f"   📊 Divergence Z-Score: {latest['Divergence_Z']:.2f}")
        print(f"\n   🎯 CURRENT SIGNAL: {latest['Signal']}")
        
        return signals
    
    def plot_analysis(self, save_path='mcx_sync_analysis.png'):
        """Create visualization of the sync analysis"""
        print(f"\n📊 Creating Visualization...")
        
        if 'MCX_Stock' not in self.data:
            print("   ❌ No data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('MCX Base Metal vs MCX Ltd Stock Sync Analysis', fontsize=14, fontweight='bold')
        
        # 1. MCX Stock Price
        ax1 = axes[0, 0]
        mcx_df = self.data['MCX_Stock'].copy()
        mcx_df.index = mcx_df.index.tz_localize(None)
        ax1.plot(mcx_df.index, mcx_df['Close'], 'b-', linewidth=1.5)
        ax1.set_title('MCX Ltd Stock Price (NSE)')
        ax1.set_ylabel('Price (₹)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Base Metals ETF (DBB)
        ax2 = axes[0, 1]
        if 'Base Metals ETF' in self.data:
            metal_df = self.data['Base Metals ETF'].copy()
            metal_df.index = metal_df.index.tz_localize(None)
            ax2.plot(metal_df.index, metal_df['Close'], 'orange', linewidth=1.5)
            ax2.set_title('Base Metals ETF (DBB)')
            ax2.set_ylabel('Price ($)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Normalized comparison
        ax3 = axes[1, 0]
        if 'Base Metals ETF' in self.data:
            mcx_norm = mcx_df['Close'] / mcx_df['Close'].iloc[0] * 100
            metal_norm = metal_df['Close'] / metal_df['Close'].iloc[0] * 100
            
            mcx_norm.index = pd.to_datetime(mcx_norm.index.date)
            metal_norm.index = pd.to_datetime(metal_norm.index.date)
            
            common_dates = mcx_norm.index.intersection(metal_norm.index)
            ax3.plot(common_dates, mcx_norm.loc[common_dates], 'b-', label='MCX Stock', linewidth=1.5)
            ax3.plot(common_dates, metal_norm.loc[common_dates], 'orange', label='Base Metals', linewidth=1.5)
            ax3.set_title('Normalized Price Comparison (Base=100)')
            ax3.set_ylabel('Normalized Price')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Correlation
        ax4 = axes[1, 1]
        if 'Base Metals ETF' in self.correlation_results:
            rolling_corr = self.correlation_results['Base Metals ETF']['rolling_corr']
            ax4.plot(rolling_corr.index, rolling_corr.values, 'g-', linewidth=1.5)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
            ax4.axhline(y=-0.5, color='g', linestyle='--', alpha=0.3)
            ax4.set_title('30-Day Rolling Correlation')
            ax4.set_ylabel('Correlation')
            ax4.set_ylim(-1, 1)
            ax4.grid(True, alpha=0.3)
        
        # 5. Correlation Heatmap
        ax5 = axes[2, 0]
        if self.correlation_results:
            names = list(self.correlation_results.keys())
            corrs = [self.correlation_results[n]['correlation'] for n in names]
            colors = ['green' if c > 0 else 'red' for c in corrs]
            bars = ax5.barh(names, corrs, color=colors, alpha=0.7)
            ax5.axvline(x=0, color='black', linewidth=0.5)
            ax5.set_title('Correlation with MCX Stock')
            ax5.set_xlabel('Correlation Coefficient')
            ax5.set_xlim(-1, 1)
            
            # Add values on bars
            for bar, corr in zip(bars, corrs):
                ax5.text(corr + 0.05 if corr >= 0 else corr - 0.15, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{corr:.2f}', va='center', fontsize=9)
        
        # 6. Summary Stats
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create summary text
        summary_text = "📊 SYNC ANALYSIS SUMMARY\n"
        summary_text += "=" * 35 + "\n\n"
        
        if 'MCX_Stock' in self.data:
            mcx_latest = self.data['MCX_Stock']['Close'].iloc[-1]
            mcx_change = self.data['MCX_Stock']['Close'].pct_change().iloc[-1] * 100
            summary_text += f"MCX Ltd Stock: ₹{mcx_latest:.2f} ({mcx_change:+.2f}%)\n\n"
        
        if self.correlation_results:
            for name, result in list(self.correlation_results.items())[:5]:
                sig = "✓" if result['significant'] else "✗"
                summary_text += f"{name}:\n  r = {result['correlation']:.3f} {sig}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Chart saved to: {save_path}")
        plt.show()
        
        return fig
    
    def run_full_analysis(self, period='1y'):
        """Run complete sync analysis"""
        print("=" * 60)
        print("🔄 MCX BASE METAL vs MCX STOCK SYNC ANALYZER")
        print("=" * 60)
        
        # Fetch data
        self.fetch_mcx_stock_data(period)
        self.fetch_base_metal_data(period)
        
        # Analysis
        self.calculate_correlations()
        sync_score = self.calculate_sync_score()
        signals = self.generate_trading_signals()
        
        # Visualization
        self.plot_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'sync_score': sync_score,
            'correlations': self.correlation_results,
            'signals': signals,
            'data': self.data
        }


def main():
    """Main execution function"""
    analyzer = MCXSyncAnalyzer()
    results = analyzer.run_full_analysis(period='1y')
    
    print("\n💡 TRADING INSIGHTS:")
    print("-" * 40)
    print("• Watch for divergences between MCX stock and metal prices")
    print("• High metal volatility often leads to increased MCX volumes")
    print("• Use correlation shifts as early warning signals")
    print("• Consider MCX stock as a hedge for commodity positions")
    
    return results


if __name__ == "__main__":
    results = main()
