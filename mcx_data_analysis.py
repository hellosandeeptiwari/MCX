"""
MCX Stock Price Driver Analysis
================================
Proper data science approach - test hypotheses with real data.
Let the DATA tell us what drives MCX stock, not assumptions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_all_data(period='1y'):
    """Fetch MCX and all potential driver data"""
    print("=" * 60)
    print("MCX STOCK DRIVER ANALYSIS - Data Science Approach")
    print("=" * 60)
    print(f"\nFetching data for {period}...")
    
    tickers = {
        'MCX': 'MCX.NS',           # MCX Ltd on NSE
        'NIFTY': '^NSEI',          # Nifty 50 (market sentiment)
        'GOLD': 'GLD',             # Gold ETF
        'SILVER': 'SLV',           # Silver ETF
        'OIL': 'USO',              # Oil ETF
        'NATGAS': 'UNG',           # Natural Gas ETF
        'BASEMETAL': 'DBB',        # Base Metals ETF
        'USD_INR': 'INR=X',        # USD/INR (currency effect)
        'VIX': '^VIX',             # Volatility Index (global fear)
    }
    
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                data[name] = df['Close'].squeeze()
                print(f"  ✓ {name}: {len(df)} data points")
            else:
                print(f"  ✗ {name}: No data")
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
    
    return pd.DataFrame(data).dropna()

def calculate_returns(df):
    """Calculate daily returns"""
    return df.pct_change().dropna()

def calculate_volatility(series, window=20):
    """Calculate rolling volatility"""
    return series.pct_change().rolling(window).std() * np.sqrt(252) * 100

def analyze_correlations(df):
    """Analyze what correlates with MCX returns"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS - MCX Daily Returns vs Others")
    print("=" * 60)
    
    returns = calculate_returns(df)
    
    # Correlation with MCX returns
    mcx_returns = returns['MCX']
    
    results = []
    for col in returns.columns:
        if col != 'MCX':
            corr, p_value = stats.pearsonr(mcx_returns, returns[col])
            results.append({
                'Factor': col,
                'Correlation': corr,
                'P-Value': p_value,
                'Significant': '✓' if p_value < 0.05 else '✗',
                'Strength': 'STRONG' if abs(corr) > 0.5 else ('MODERATE' if abs(corr) > 0.3 else 'WEAK')
            })
    
    results_df = pd.DataFrame(results).sort_values('Correlation', key=abs, ascending=False)
    print("\nCorrelation with MCX Daily Returns:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        sig = "***" if row['P-Value'] < 0.01 else ("**" if row['P-Value'] < 0.05 else "")
        print(f"  {row['Factor']:12} : {row['Correlation']:+.4f} {sig:3} ({row['Strength']})")
    
    print("\n  *** = Highly significant (p < 0.01)")
    print("  **  = Significant (p < 0.05)")
    
    return results_df

def analyze_volatility_relationship(df):
    """Test if commodity volatility drives MCX"""
    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS - Does commodity vol drive MCX?")
    print("=" * 60)
    
    # Calculate volatility for commodities
    commodities = ['GOLD', 'SILVER', 'OIL', 'NATGAS', 'BASEMETAL']
    
    for comm in commodities:
        if comm in df.columns:
            df[f'{comm}_VOL'] = calculate_volatility(df[comm])
    
    # MCX returns
    mcx_returns = df['MCX'].pct_change()
    
    print("\nCorrelation: Commodity Volatility vs MCX Returns")
    print("-" * 60)
    
    for comm in commodities:
        vol_col = f'{comm}_VOL'
        if vol_col in df.columns:
            # Align data
            valid = df[[vol_col]].join(mcx_returns.rename('MCX_RET')).dropna()
            if len(valid) > 30:
                corr, p_val = stats.pearsonr(valid[vol_col], valid['MCX_RET'])
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
                print(f"  {comm:12} Volatility : {corr:+.4f} {sig}")

def analyze_lagged_effects(df):
    """Check if yesterday's commodity move predicts today's MCX"""
    print("\n" + "=" * 60)
    print("LAGGED ANALYSIS - Do commodity moves predict MCX next day?")
    print("=" * 60)
    
    returns = calculate_returns(df)
    mcx_returns = returns['MCX']
    
    print("\nCorrelation: Yesterday's Move vs Today's MCX")
    print("-" * 60)
    
    for col in ['GOLD', 'SILVER', 'OIL', 'NATGAS', 'BASEMETAL', 'NIFTY', 'VIX']:
        if col in returns.columns:
            lagged = returns[col].shift(1)
            valid = pd.DataFrame({'MCX': mcx_returns, 'LAGGED': lagged}).dropna()
            if len(valid) > 30:
                corr, p_val = stats.pearsonr(valid['MCX'], valid['LAGGED'])
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
                useful = "PREDICTIVE!" if p_val < 0.05 and abs(corr) > 0.1 else ""
                print(f"  {col:12} (t-1) : {corr:+.4f} {sig:3} {useful}")

def analyze_regime_correlation(df):
    """Check if correlation changes in different market regimes"""
    print("\n" + "=" * 60)
    print("REGIME ANALYSIS - Does correlation change in bull/bear markets?")
    print("=" * 60)
    
    returns = calculate_returns(df)
    
    # Define regimes based on Nifty
    if 'NIFTY' in returns.columns:
        nifty_ma = df['NIFTY'].rolling(50).mean()
        bull_regime = df['NIFTY'] > nifty_ma
        bear_regime = df['NIFTY'] <= nifty_ma
        
        print("\nMCX-Gold Correlation by Market Regime:")
        print("-" * 60)
        
        if 'GOLD' in returns.columns:
            bull_returns = returns[bull_regime.reindex(returns.index).fillna(False)]
            bear_returns = returns[bear_regime.reindex(returns.index).fillna(False)]
            
            if len(bull_returns) > 30:
                bull_corr, _ = stats.pearsonr(bull_returns['MCX'], bull_returns['GOLD'])
                print(f"  Bull Market (Nifty > 50MA): {bull_corr:+.4f}")
            
            if len(bear_returns) > 30:
                bear_corr, _ = stats.pearsonr(bear_returns['MCX'], bear_returns['GOLD'])
                print(f"  Bear Market (Nifty < 50MA): {bear_corr:+.4f}")

def analyze_what_really_matters(df):
    """Multiple regression to find what really drives MCX"""
    print("\n" + "=" * 60)
    print("REGRESSION ANALYSIS - What REALLY drives MCX?")
    print("=" * 60)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    returns = calculate_returns(df).dropna()
    
    # Features
    feature_cols = [c for c in ['NIFTY', 'GOLD', 'SILVER', 'OIL', 'USD_INR', 'VIX'] if c in returns.columns]
    
    if len(feature_cols) > 0:
        X = returns[feature_cols].values
        y = returns['MCX'].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Results
        print("\nStandardized Coefficients (impact on MCX):")
        print("-" * 60)
        
        coef_df = pd.DataFrame({
            'Factor': feature_cols,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        for _, row in coef_df.iterrows():
            bar = '█' * int(abs(row['Coefficient']) * 100)
            direction = '+' if row['Coefficient'] > 0 else '-'
            print(f"  {row['Factor']:12} : {direction} {bar} ({row['Coefficient']:+.4f})")
        
        print(f"\n  R-squared: {model.score(X_scaled, y):.4f}")
        print(f"  (How much of MCX movement is explained: {model.score(X_scaled, y)*100:.1f}%)")

def summarize_findings(corr_df):
    """Summarize what we found"""
    print("\n" + "=" * 60)
    print("SUMMARY - What Actually Drives MCX Stock?")
    print("=" * 60)
    
    print("\nBased on data analysis:")
    print("-" * 60)
    
    # Find strongest correlations
    significant = corr_df[corr_df['P-Value'] < 0.05]
    
    if len(significant) > 0:
        strongest = significant.iloc[0]
        print(f"\n1. STRONGEST DRIVER: {strongest['Factor']}")
        print(f"   Correlation: {strongest['Correlation']:+.4f}")
        
        if 'NIFTY' in significant['Factor'].values:
            print("\n2. MCX moves with NIFTY (overall market sentiment)")
            print("   → It's a stock, so market direction matters!")
    else:
        print("\n⚠️  No statistically significant correlations found!")
        print("   MCX may be driven by company-specific factors:")
        print("   - Quarterly earnings")
        print("   - Regulatory changes (SEBI)")
        print("   - New product launches")
        print("   - Competition news")
    
    print("\n" + "=" * 60)

def main():
    # Fetch data
    df = fetch_all_data(period='2y')
    
    if 'MCX' not in df.columns:
        print("\n❌ Could not fetch MCX data. Check internet connection.")
        return
    
    # Run all analyses
    corr_df = analyze_correlations(df)
    analyze_volatility_relationship(df)
    analyze_lagged_effects(df)
    analyze_regime_correlation(df)
    
    try:
        analyze_what_really_matters(df)
    except ImportError:
        print("\n(Install sklearn for regression analysis: pip install scikit-learn)")
    
    summarize_findings(corr_df)
    
    return df, corr_df

if __name__ == "__main__":
    df, corr_df = main()
