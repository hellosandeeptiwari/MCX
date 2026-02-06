"""
MCX Trading Web Dashboard
==========================
Flask-based web interface for MCX Options Trading Analysis

Author: Options Trader Tool
Date: February 2026
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============== Configuration ==============
MCX_LOT_SIZE = 625  # Correct lot size for MCX Ltd

# Commodity weights based on typical MCX trading volumes
COMMODITY_WEIGHTS = {
    'GLD': 0.35,   # Gold - 35%
    'SLV': 0.25,   # Silver - 25%
    'USO': 0.20,   # Crude Oil - 20%
    'UNG': 0.10,   # Natural Gas - 10%
    'DBB': 0.10    # Base Metals - 10%
}

# ============== Data Fetching Functions ==============

def fetch_mcx_data(period='1y'):
    """Fetch MCX stock data"""
    mcx = yf.Ticker("MCX.NS")
    df = mcx.history(period=period)
    if not df.empty:
        df.index = df.index.tz_localize(None)
    return df

def fetch_metal_data(period='1y'):
    """Fetch base metal ETF data"""
    metal = yf.Ticker("DBB")
    df = metal.history(period=period)
    if not df.empty:
        df.index = df.index.tz_localize(None)
    return df

def fetch_commodity_data(ticker, period='1y'):
    """Fetch individual commodity data"""
    try:
        data = yf.Ticker(ticker)
        df = data.history(period=period)
        if not df.empty:
            df.index = df.index.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

def fetch_all_commodities(period='1y'):
    """Fetch all commodities that affect MCX with volatility data"""
    commodities = {
        'Gold (GLD)': 'GLD',
        'Silver (SLV)': 'SLV', 
        'Crude Oil (USO)': 'USO',
        'Natural Gas (UNG)': 'UNG',
        'Base Metals (DBB)': 'DBB'
    }
    
    data = {}
    for name, ticker in commodities.items():
        df = fetch_commodity_data(ticker, period)
        if not df.empty and len(df) > 5:
            # Calculate daily volatility (5-day)
            returns = df['Close'].pct_change().dropna()
            volatility_5d = float(returns.tail(5).std() * np.sqrt(252) * 100)  # Annualized %
            volatility_20d = float(returns.tail(20).std() * np.sqrt(252) * 100) if len(returns) >= 20 else volatility_5d
            
            # Volatility change (is vol increasing?)
            vol_change = volatility_5d - volatility_20d
            
            data[name] = {
                'ticker': ticker,
                'price': float(df['Close'].iloc[-1]),
                'change': float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100) if len(df) > 1 else 0,
                'weight': COMMODITY_WEIGHTS.get(ticker, 0.1),
                'volatility': volatility_5d,
                'vol_trend': 'RISING' if vol_change > 2 else ('FALLING' if vol_change < -2 else 'STABLE')
            }
    return data

def calculate_commodity_volatility_index():
    """Calculate weighted volatility index of all commodities - THE KEY METRIC!"""
    commodities_data = fetch_all_commodities('3mo')
    
    if not commodities_data:
        return None
    
    weighted_vol = 0
    total_weight = 0
    individual_vols = {}
    
    for name, data in commodities_data.items():
        weight = data.get('weight', 0.1)
        vol = data.get('volatility', 0)
        weighted_vol += vol * weight
        total_weight += weight
        individual_vols[name] = {
            'volatility': vol,
            'vol_trend': data.get('vol_trend', 'STABLE'),
            'weight': weight
        }
    
    avg_vol = weighted_vol / total_weight if total_weight > 0 else 0
    
    # Determine if overall volatility is high (good for MCX)
    if avg_vol > 30:
        vol_signal = 'HIGH_VOL'
        mcx_outlook = 'BULLISH'
    elif avg_vol > 20:
        vol_signal = 'MODERATE_VOL'
        mcx_outlook = 'NEUTRAL'
    else:
        vol_signal = 'LOW_VOL'
        mcx_outlook = 'BEARISH'
    
    return {
        'composite_volatility': avg_vol,
        'vol_signal': vol_signal,
        'mcx_outlook': mcx_outlook,
        'individual': individual_vols
    }

def fetch_mcx_volume_data(period='3mo'):
    """Fetch MCX trading volume data"""
    mcx_df = fetch_mcx_data(period)
    
    if mcx_df.empty:
        return None
    
    # Volume analysis
    current_vol = float(mcx_df['Volume'].iloc[-1])
    avg_vol_20d = float(mcx_df['Volume'].tail(20).mean())
    avg_vol_5d = float(mcx_df['Volume'].tail(5).mean())
    
    vol_ratio = current_vol / avg_vol_20d if avg_vol_20d > 0 else 1
    vol_trend = 'INCREASING' if avg_vol_5d > avg_vol_20d * 1.2 else ('DECREASING' if avg_vol_5d < avg_vol_20d * 0.8 else 'STABLE')
    
    # Volume spike detection
    vol_std = float(mcx_df['Volume'].tail(20).std())
    is_spike = current_vol > avg_vol_20d + (2 * vol_std)
    
    return {
        'current_volume': current_vol,
        'avg_volume_20d': avg_vol_20d,
        'volume_ratio': vol_ratio,
        'volume_trend': vol_trend,
        'is_volume_spike': bool(is_spike),
        'volume_change_pct': float((current_vol / avg_vol_20d - 1) * 100)
    }

def calculate_composite_commodity_index(period='6mo'):
    """Calculate weighted composite index of all commodities"""
    commodities = {}
    for ticker, weight in COMMODITY_WEIGHTS.items():
        df = fetch_commodity_data(ticker, period)
        if not df.empty:
            # Normalize to percentage returns from start
            normalized = (df['Close'] / df['Close'].iloc[0]) * 100
            commodities[ticker] = {'data': normalized, 'weight': weight}
    
    if not commodities:
        return None
    
    # Create composite index
    first_ticker = list(commodities.keys())[0]
    composite = pd.Series(0, index=commodities[first_ticker]['data'].index)
    
    for ticker, info in commodities.items():
        aligned = info['data'].reindex(composite.index, method='ffill').fillna(method='bfill')
        composite = composite + (aligned * info['weight'])
    
    return composite

def calculate_sync_analysis():
    """Calculate sync analysis between MCX and ALL commodities (Gold, Silver, Oil, Gas, Base Metals)"""
    mcx_df = fetch_mcx_data()
    
    # Fetch all commodities for composite index
    composite = calculate_composite_commodity_index('1y')
    metal_df = fetch_metal_data()  # Keep for backward compatibility
    
    if mcx_df.empty or composite is None:
        return None
    
    # Align MCX data with composite
    mcx_df.index = pd.to_datetime(mcx_df.index.date)
    composite.index = pd.to_datetime(composite.index.date)
    
    common = mcx_df.index.intersection(composite.index)
    
    if len(common) < 20:
        # Fallback to metal only
        metal_df.index = pd.to_datetime(metal_df.index.date)
        common = mcx_df.index.intersection(metal_df.index)
        mcx_returns = mcx_df.loc[common, 'Close'].pct_change().dropna()
        commodity_returns = metal_df.loc[common, 'Close'].pct_change().dropna()
    else:
        mcx_returns = mcx_df.loc[common, 'Close'].pct_change().dropna()
        commodity_returns = composite.loc[common].pct_change().dropna()
    
    # Correlation with composite
    common_ret = mcx_returns.index.intersection(commodity_returns.index)
    corr, p_value = stats.pearsonr(mcx_returns.loc[common_ret], commodity_returns.loc[common_ret])
    
    # Divergence Z-score using composite
    mcx_norm = mcx_df.loc[common, 'Close'] / mcx_df.loc[common, 'Close'].iloc[0] * 100
    composite_aligned = composite.loc[common]
    
    divergence = mcx_norm - composite_aligned
    divergence_z = (divergence - divergence.rolling(50).mean()) / divergence.rolling(50).std()
    
    current_z = divergence_z.iloc[-1] if not divergence_z.empty and not pd.isna(divergence_z.iloc[-1]) else 0
    
    # Signal
    if current_z < -1.5:
        signal = "BUY MCX"
        signal_class = "success"
    elif current_z > 1.5:
        signal = "SELL MCX"
        signal_class = "danger"
    else:
        signal = "HOLD"
        signal_class = "warning"
    
    return {
        'mcx_price': float(mcx_df['Close'].iloc[-1]),
        'mcx_change': float(mcx_df['Close'].pct_change().iloc[-1] * 100),
        'metal_price': float(metal_df['Close'].iloc[-1]),
        'metal_change': float(metal_df['Close'].pct_change().iloc[-1] * 100),
        'correlation': float(corr),
        'p_value': float(p_value),
        'divergence_z': float(current_z) if not np.isnan(current_z) else 0,
        'signal': signal,
        'signal_class': signal_class,
        'sync_score': float(corr * 0.5 + 0.5) if not np.isnan(corr) else 0.5
    }

# ============== Options Greeks Functions ==============

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0: return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_delta(S, K, T, r, sigma, opt_type='CALL'):
    if T <= 0:
        return 1.0 if (opt_type == 'CALL' and S > K) else (-1.0 if opt_type == 'PUT' and S < K else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt_type == 'CALL' else norm.cdf(d1) - 1

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_theta(S, K, T, r, sigma, opt_type='CALL'):
    if T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if opt_type == 'CALL':
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

def calculate_vega(S, K, T, r, sigma):
    if T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def generate_option_chain(spot_price, days_to_expiry=30, r=0.065, sigma=0.30):
    """Generate option chain"""
    T = days_to_expiry / 365
    strikes = [round(spot_price + (i - 5) * spot_price * 0.025, -1) for i in range(11)]
    
    chain = []
    for K in strikes:
        call_price = black_scholes_call(spot_price, K, T, r, sigma)
        put_price = black_scholes_put(spot_price, K, T, r, sigma)
        
        chain.append({
            'strike': K,
            'call_price': round(call_price, 2),
            'call_delta': round(calculate_delta(spot_price, K, T, r, sigma, 'CALL'), 3),
            'call_gamma': round(calculate_gamma(spot_price, K, T, r, sigma), 4),
            'call_theta': round(calculate_theta(spot_price, K, T, r, sigma, 'CALL'), 2),
            'call_vega': round(calculate_vega(spot_price, K, T, r, sigma), 2),
            'put_price': round(put_price, 2),
            'put_delta': round(calculate_delta(spot_price, K, T, r, sigma, 'PUT'), 3),
            'put_gamma': round(calculate_gamma(spot_price, K, T, r, sigma), 4),
            'put_theta': round(calculate_theta(spot_price, K, T, r, sigma, 'PUT'), 2),
            'put_vega': round(calculate_vega(spot_price, K, T, r, sigma), 2),
            'is_atm': bool(abs(K - spot_price) < spot_price * 0.025)
        })
    
    return chain

# ============== Chart Generation ==============

def generate_price_chart():
    """Generate price comparison chart"""
    mcx_df = fetch_mcx_data('6mo')
    metal_df = fetch_metal_data('6mo')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')
    
    ax1.set_facecolor('#1a1a2e')
    ax1.plot(mcx_df.index, mcx_df['Close'], color='#00d4ff', linewidth=1.5)
    ax1.set_title('MCX Ltd Stock Price (NSE)', color='white', fontsize=12)
    ax1.set_ylabel('Price (₹)', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(metal_df.index, metal_df['Close'], color='#ffa500', linewidth=1.5)
    ax2.set_title('Base Metals ETF (DBB)', color='white', fontsize=12)
    ax2.set_ylabel('Price ($)', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', dpi=100)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_correlation_chart():
    """Generate rolling correlation chart"""
    mcx_df = fetch_mcx_data('6mo')
    metal_df = fetch_metal_data('6mo')
    
    mcx_df.index = pd.to_datetime(mcx_df.index.date)
    metal_df.index = pd.to_datetime(metal_df.index.date)
    
    common = mcx_df.index.intersection(metal_df.index)
    
    mcx_returns = mcx_df.loc[common, 'Close'].pct_change()
    metal_returns = metal_df.loc[common, 'Close'].pct_change()
    
    rolling_corr = mcx_returns.rolling(30).corr(metal_returns)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.plot(rolling_corr.index, rolling_corr.values, color='#00ff88', linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.5, color='green', linestyle='--', alpha=0.3)
    ax.fill_between(rolling_corr.index, rolling_corr.values, 0, 
                    where=(rolling_corr.values > 0), alpha=0.3, color='green')
    ax.fill_between(rolling_corr.index, rolling_corr.values, 0, 
                    where=(rolling_corr.values < 0), alpha=0.3, color='red')
    
    ax.set_title('30-Day Rolling Correlation', color='white', fontsize=12)
    ax.set_ylabel('Correlation', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', dpi=100)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============== Technical Analysis Functions ==============

def calculate_technical_indicators():
    """Calculate technical indicators for MCX"""
    df = fetch_mcx_data('6mo')
    if df.empty:
        return None
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # RSI (14-period)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # Support & Resistance (recent pivots)
    recent_high = high.rolling(20).max().iloc[-1]
    recent_low = low.rolling(20).min().iloc[-1]
    pivot = (recent_high + recent_low + close.iloc[-1]) / 3
    r1 = 2 * pivot - recent_low
    r2 = pivot + (recent_high - recent_low)
    s1 = 2 * pivot - recent_high
    s2 = pivot - (recent_high - recent_low)
    
    # Moving Averages
    sma20_val = sma20.iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    
    # Volume Analysis
    avg_volume = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # ATR (Average True Range) for volatility
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # Historical Volatility (annualized)
    returns = np.log(close / close.shift(1)).dropna()
    hist_vol = returns.std() * np.sqrt(252) * 100
    
    return {
        'rsi': float(rsi.iloc[-1]),
        'rsi_signal': 'OVERSOLD' if rsi.iloc[-1] < 30 else ('OVERBOUGHT' if rsi.iloc[-1] > 70 else 'NEUTRAL'),
        'macd': float(macd.iloc[-1]),
        'macd_signal': float(signal.iloc[-1]),
        'macd_histogram': float(histogram.iloc[-1]),
        'macd_trend': 'BULLISH' if histogram.iloc[-1] > 0 else 'BEARISH',
        'bb_upper': float(bb_upper.iloc[-1]),
        'bb_lower': float(bb_lower.iloc[-1]),
        'bb_mid': float(sma20.iloc[-1]),
        'bb_position': 'UPPER' if close.iloc[-1] > bb_upper.iloc[-1] else ('LOWER' if close.iloc[-1] < bb_lower.iloc[-1] else 'MIDDLE'),
        'pivot': float(pivot),
        'r1': float(r1),
        'r2': float(r2),
        's1': float(s1),
        's2': float(s2),
        'sma20': float(sma20_val),
        'sma50': float(sma50),
        'sma200': float(sma200) if sma200 else None,
        'trend_sma': 'BULLISH' if close.iloc[-1] > sma50 else 'BEARISH',
        'avg_volume': float(avg_volume),
        'volume_ratio': float(volume_ratio),
        'volume_signal': 'HIGH' if volume_ratio > 1.5 else ('LOW' if volume_ratio < 0.5 else 'NORMAL'),
        'atr': float(atr),
        'hist_volatility': float(hist_vol),
        'expected_move_1std': float(close.iloc[-1] * hist_vol / 100 / np.sqrt(252) * np.sqrt(30)),  # 30 day
    }

def calculate_position_sizing(capital, risk_percent, entry_price, stop_loss_price):
    """Calculate position size based on risk management"""
    risk_amount = capital * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share <= 0:
        return None
    
    shares = risk_amount / risk_per_share
    lots = int(shares / MCX_LOT_SIZE)
    
    return {
        'risk_amount': risk_amount,
        'risk_per_share': risk_per_share,
        'shares': int(shares),
        'lots': lots,
        'actual_shares': lots * MCX_LOT_SIZE,
        'actual_risk': lots * MCX_LOT_SIZE * risk_per_share,
        'capital_required': lots * MCX_LOT_SIZE * entry_price
    }

def calculate_option_pnl(option_type, strike, premium, spot_prices):
    """Calculate P&L for different spot prices"""
    pnl_data = []
    for spot in spot_prices:
        if option_type == 'CALL':
            intrinsic = max(spot - strike, 0)
        else:
            intrinsic = max(strike - spot, 0)
        
        pnl_per_share = intrinsic - premium
        pnl_per_lot = pnl_per_share * MCX_LOT_SIZE
        
        pnl_data.append({
            'spot': spot,
            'intrinsic': intrinsic,
            'pnl_per_share': pnl_per_share,
            'pnl_per_lot': pnl_per_lot
        })
    
    return pnl_data

def get_pre_market_summary():
    """Get pre-market analysis summary"""
    mcx_df = fetch_mcx_data('1mo')
    metal_df = fetch_metal_data('1mo')
    
    if mcx_df.empty:
        return None
    
    # Previous day data
    prev_close = mcx_df['Close'].iloc[-2] if len(mcx_df) > 1 else mcx_df['Close'].iloc[-1]
    prev_high = mcx_df['High'].iloc[-2] if len(mcx_df) > 1 else mcx_df['High'].iloc[-1]
    prev_low = mcx_df['Low'].iloc[-2] if len(mcx_df) > 1 else mcx_df['Low'].iloc[-1]
    
    # Current/Last close
    current_close = mcx_df['Close'].iloc[-1]
    
    # Gap analysis (would need pre-market data, using previous for demo)
    overnight_change = ((current_close - prev_close) / prev_close) * 100
    
    # Weekly change
    week_ago = mcx_df['Close'].iloc[-5] if len(mcx_df) >= 5 else mcx_df['Close'].iloc[0]
    weekly_change = ((current_close - week_ago) / week_ago) * 100
    
    # Monthly change
    month_start = mcx_df['Close'].iloc[0]
    monthly_change = ((current_close - month_start) / month_start) * 100
    
    # 52-week high/low (using available data)
    high_52w = mcx_df['High'].max()
    low_52w = mcx_df['Low'].min()
    from_52w_high = ((current_close - high_52w) / high_52w) * 100
    from_52w_low = ((current_close - low_52w) / low_52w) * 100
    
    # Metal overnight change
    metal_overnight = 0
    if not metal_df.empty and len(metal_df) > 1:
        metal_prev = metal_df['Close'].iloc[-2]
        metal_current = metal_df['Close'].iloc[-1]
        metal_overnight = ((metal_current - metal_prev) / metal_prev) * 100
    
    return {
        'prev_close': float(prev_close),
        'prev_high': float(prev_high),
        'prev_low': float(prev_low),
        'current_close': float(current_close),
        'overnight_change': float(overnight_change),
        'weekly_change': float(weekly_change),
        'monthly_change': float(monthly_change),
        'high_52w': float(high_52w),
        'low_52w': float(low_52w),
        'from_52w_high': float(from_52w_high),
        'from_52w_low': float(from_52w_low),
        'metal_overnight_change': float(metal_overnight),
        'divergence_building': bool(abs(overnight_change - metal_overnight) > 1)
    }

# ============== Routes ==============

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analysis')
def api_analysis():
    try:
        data = calculate_sync_analysis()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/commodities')
def api_commodities():
    """Get all commodities data with individual performance and volatility"""
    try:
        commodities = fetch_all_commodities('3mo')
        return jsonify({'success': True, 'data': commodities})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/volatility')
def api_volatility():
    """Get commodity volatility index - THE KEY METRIC for MCX"""
    try:
        vol_data = calculate_commodity_volatility_index()
        return jsonify({'success': True, 'data': vol_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/mcx-volume')
def api_mcx_volume():
    """Get MCX trading volume analysis"""
    try:
        vol_data = fetch_mcx_volume_data()
        return jsonify({'success': True, 'data': vol_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
        return jsonify({'success': True, 'data': commodities})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/option-chain')
def api_option_chain():
    try:
        days = int(request.args.get('days', 30))
        mcx_df = fetch_mcx_data('5d')
        spot = mcx_df['Close'].iloc[-1]
        
        # Calculate historical volatility
        returns = np.log(mcx_df['Close'] / mcx_df['Close'].shift(1)).dropna()
        hist_vol = returns.std() * np.sqrt(252)
        
        chain = generate_option_chain(spot, days, sigma=max(hist_vol, 0.25))
        return jsonify({'success': True, 'data': chain, 'spot': float(spot)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/charts/price')
def api_price_chart():
    try:
        chart = generate_price_chart()
        return jsonify({'success': True, 'chart': chart})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/charts/correlation')
def api_correlation_chart():
    try:
        chart = generate_correlation_chart()
        return jsonify({'success': True, 'chart': chart})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategy')
def api_strategy():
    """
    UPDATED STRATEGY: Now based on VOLATILITY, not price correlation!
    
    KEY INSIGHT: MCX is an EXCHANGE - it earns from TRADING VOLUME, not commodity prices!
    High commodity volatility → More trading → More fees → MCX stock UP!
    """
    try:
        data = calculate_sync_analysis()
        mcx_price = data['mcx_price']
        
        # Get volatility data - THIS IS THE KEY DRIVER NOW
        vol_data = calculate_commodity_volatility_index()
        vol_signal = vol_data['vol_signal']
        mcx_outlook = vol_data['mcx_outlook']
        composite_vol = vol_data['composite_volatility']
        
        # Get volume data
        volume_data = fetch_mcx_volume_data()
        volume_ratio = volume_data['volume_ratio']
        is_spike = volume_data['is_volume_spike']
        
        # NEW LOGIC: High volatility = BULLISH for MCX (more trading = more revenue)
        if mcx_outlook == 'BULLISH' and is_spike:
            strategy = {
                'name': 'BUY ATM CALL',
                'signal': 'STRONG BULLISH',
                'confidence': 'HIGH',
                'strike': round(mcx_price, -1),
                'action': 'Buy Call Option',
                'rationale': f'HIGH commodity volatility ({composite_vol:.1f}%) + Volume spike ({volume_ratio:.1f}x avg) = More trading fees for MCX!',
                'risk': 'Premium paid',
                'volatility_driven': True
            }
        elif mcx_outlook == 'BULLISH':
            strategy = {
                'name': 'BULL CALL SPREAD',
                'signal': 'BULLISH',
                'confidence': 'MEDIUM',
                'strike': f"{round(mcx_price, -1)} / {round(mcx_price * 1.05, -1)}",
                'action': 'Buy lower strike Call, Sell higher strike Call',
                'rationale': f'High commodity volatility ({composite_vol:.1f}%) driving MCX volumes up.',
                'risk': 'Net premium paid',
                'volatility_driven': True
            }
        elif mcx_outlook == 'BEARISH' and volume_ratio < 0.7:
            strategy = {
                'name': 'BUY ATM PUT',
                'signal': 'STRONG BEARISH',
                'confidence': 'HIGH',
                'strike': round(mcx_price, -1),
                'action': 'Buy Put Option',
                'rationale': f'LOW commodity volatility ({composite_vol:.1f}%) + Low volume ({volume_ratio:.1f}x avg) = Less trading activity for MCX.',
                'risk': 'Premium paid',
                'volatility_driven': True
            }
        elif mcx_outlook == 'BEARISH':
            strategy = {
                'name': 'BEAR PUT SPREAD',
                'signal': 'BEARISH',
                'confidence': 'MEDIUM',
                'strike': f"{round(mcx_price, -1)} / {round(mcx_price * 0.95, -1)}",
                'action': 'Buy higher strike Put, Sell lower strike Put',
                'rationale': f'Low commodity volatility ({composite_vol:.1f}%) suggests reduced MCX trading activity.',
                'risk': 'Net premium paid',
                'volatility_driven': True
            }
        else:
            # Neutral volatility - range-bound strategies
            strategy = {
                'name': 'SELL NAKED PUT (Collect Premium)',
                'signal': 'NEUTRAL-BULLISH',
                'confidence': 'MEDIUM',
                'strike': round(mcx_price * 0.95, -1),
                'action': 'Sell OTM Put - Collect theta decay',
                'rationale': f'Moderate volatility ({composite_vol:.1f}%). If assigned, own MCX at good price. You prefer selling puts anyway!',
                'risk': 'Unlimited downside if MCX crashes',
                'volatility_driven': True
            }
        
        # Add volatility context to response
        strategy['vol_context'] = {
            'composite_volatility': composite_vol,
            'vol_signal': vol_signal,
            'volume_ratio': volume_ratio,
            'is_volume_spike': is_spike
        }
        
        return jsonify({'success': True, 'data': strategy})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/technical')
def api_technical():
    """Get technical indicators"""
    try:
        data = calculate_technical_indicators()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pre-market')
def api_pre_market():
    """Get pre-market summary"""
    try:
        data = get_pre_market_summary()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/position-size', methods=['POST'])
def api_position_size():
    """Calculate position size"""
    try:
        req_data = request.json
        capital = float(req_data.get('capital', 500000))
        risk_percent = float(req_data.get('risk_percent', 2))
        entry_price = float(req_data.get('entry_price', 2320))
        stop_loss = float(req_data.get('stop_loss', 2200))
        
        result = calculate_position_sizing(capital, risk_percent, entry_price, stop_loss)
        return jsonify({'success': True, 'data': result, 'lot_size': MCX_LOT_SIZE})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pnl-simulator', methods=['POST'])
def api_pnl_simulator():
    """Simulate P&L for option trade"""
    try:
        req_data = request.json
        option_type = req_data.get('option_type', 'PUT')
        strike = float(req_data.get('strike', 2320))
        premium = float(req_data.get('premium', 80))
        num_lots = int(req_data.get('lots', 1))
        
        # Generate spot price range
        spot_prices = np.arange(strike * 0.85, strike * 1.15, 10).tolist()
        
        pnl_data = calculate_option_pnl(option_type, strike, premium, spot_prices)
        
        # Multiply by number of lots
        for item in pnl_data:
            item['pnl_per_lot'] = item['pnl_per_lot'] * num_lots
            item['total_shares'] = MCX_LOT_SIZE * num_lots
        
        # Find breakeven
        breakeven = strike - premium if option_type == 'PUT' else strike + premium
        max_loss = premium * MCX_LOT_SIZE * num_lots
        
        return jsonify({
            'success': True, 
            'data': pnl_data,
            'breakeven': breakeven,
            'max_loss': max_loss,
            'lot_size': MCX_LOT_SIZE,
            'total_shares': MCX_LOT_SIZE * num_lots
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create templates folder and HTML file
    import os
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 MCX Trading Web Dashboard")
    print("=" * 60)
    print("\n📊 Starting server at http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
