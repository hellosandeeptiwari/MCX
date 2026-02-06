"""
MCX Real-Time Trading Dashboard v4.0
=====================================
Fetches LIVE data from Screener.in + Yahoo Finance
Designed for frequent refresh during market hours
"""

from flask import Flask, render_template, jsonify, request
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

MCX_LOT_SIZE = 625
MCX_TICKER = 'MCX.NS'

# Cache to avoid too many requests
_cache = {}
_cache_time = {}
CACHE_DURATION = 60  # seconds

def get_cached(key, fetch_func, duration=CACHE_DURATION):
    """Simple cache to avoid hammering APIs"""
    now = datetime.now()
    if key in _cache and key in _cache_time:
        if (now - _cache_time[key]).seconds < duration:
            return _cache[key]
    
    data = fetch_func()
    _cache[key] = data
    _cache_time[key] = now
    return data

def fetch_screener_data():
    """Fetch REAL-TIME data from Screener.in"""
    try:
        url = 'https://www.screener.in/company/MCX/consolidated/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {
            'metrics': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # Parse key metrics
        items = soup.find_all('li', class_='flex')
        for item in items:
            name = item.find('span', class_='name')
            value = item.find('span', class_='number')
            if name and value:
                key = name.text.strip()
                val = value.text.strip().replace(',', '')
                # Try to convert to float
                try:
                    if '%' in val:
                        data['metrics'][key] = float(val.replace('%', ''))
                    else:
                        data['metrics'][key] = float(val)
                except:
                    data['metrics'][key] = val
        
        # Parse strengths
        pros = soup.find('div', class_='pros')
        if pros:
            for li in pros.find_all('li'):
                data['strengths'].append(li.text.strip())
        
        # Parse weaknesses
        cons = soup.find('div', class_='cons')
        if cons:
            for li in cons.find_all('li'):
                data['weaknesses'].append(li.text.strip())
        
        return data
    except Exception as e:
        return {'error': str(e)}

def fetch_live_price():
    """Fetch live price from Yahoo Finance"""
    try:
        mcx = yf.download(MCX_TICKER, period='5d', interval='1m', progress=False)
        if mcx.empty:
            # Try daily data
            mcx = yf.download(MCX_TICKER, period='5d', progress=False)
        
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        current = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else current
        change = current - prev_close
        change_pct = (change / prev_close) * 100
        
        high = float(mcx['High'].squeeze().iloc[-1])
        low = float(mcx['Low'].squeeze().iloc[-1])
        volume = int(mcx['Volume'].squeeze().iloc[-1])
        
        return {
            'price': current,
            'change': change,
            'change_pct': change_pct,
            'high': high,
            'low': low,
            'volume': volume,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return {'error': str(e)}

def fetch_technicals():
    """Fetch and calculate technical indicators"""
    try:
        mcx = yf.download(MCX_TICKER, period='6mo', progress=False)
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        high = mcx['High'].squeeze()
        low = mcx['Low'].squeeze()
        volume = mcx['Volume'].squeeze()
        
        current = float(close.iloc[-1])
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = float(rsi.iloc[-1])
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # Moving Averages
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        
        # Bollinger Bands
        bb_mid = sma_20
        bb_std = float(close.rolling(20).std().iloc[-1])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        # Volume analysis
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_current = float(volume.iloc[-1])
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Support/Resistance (20-day)
        support = float(low.tail(20).min())
        resistance = float(high.tail(20).max())
        
        # Signals
        rsi_signal = 'OVERSOLD' if rsi_value < 30 else ('OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL')
        macd_signal = 'BULLISH' if float(histogram.iloc[-1]) > 0 else 'BEARISH'
        
        # MA Signal
        if sma_200:
            if current > sma_50 > sma_200:
                ma_signal = 'STRONG_BULLISH'
            elif current > sma_50:
                ma_signal = 'BULLISH'
            elif current < sma_50 < sma_200:
                ma_signal = 'STRONG_BEARISH'
            elif current < sma_50:
                ma_signal = 'BEARISH'
            else:
                ma_signal = 'NEUTRAL'
        else:
            ma_signal = 'BULLISH' if current > sma_50 else 'BEARISH'
        
        # BB position
        if current > bb_upper:
            bb_signal = 'OVERBOUGHT'
        elif current < bb_lower:
            bb_signal = 'OVERSOLD'
        else:
            bb_position = (current - bb_lower) / (bb_upper - bb_lower) * 100
            bb_signal = f'{bb_position:.0f}%'
        
        return {
            'rsi': rsi_value,
            'rsi_signal': rsi_signal,
            'macd': float(macd.iloc[-1]),
            'macd_signal_line': float(signal.iloc[-1]),
            'macd_histogram': float(histogram.iloc[-1]),
            'macd_signal': macd_signal,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'ma_signal': ma_signal,
            'bb_upper': bb_upper,
            'bb_middle': bb_mid,
            'bb_lower': bb_lower,
            'bb_signal': bb_signal,
            'atr': atr,
            'atr_pct': (atr / current) * 100,
            'support': support,
            'resistance': resistance,
            'volume_ratio': vol_ratio,
            'volume_signal': 'HIGH' if vol_ratio > 1.5 else ('LOW' if vol_ratio < 0.7 else 'NORMAL')
        }
    except Exception as e:
        return {'error': str(e)}

def fetch_momentum():
    """Calculate momentum indicators"""
    try:
        mcx = yf.download(MCX_TICKER, period='1y', progress=False)
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        current = float(close.iloc[-1])
        
        # Returns over periods
        returns = {}
        periods = {'1D': 1, '1W': 5, '1M': 22, '3M': 66, '6M': 132, '1Y': 252}
        
        for name, days in periods.items():
            if len(close) > days:
                past = float(close.iloc[-days-1])
                returns[name] = ((current / past) - 1) * 100
            else:
                returns[name] = None
        
        # Relative strength vs Nifty
        try:
            nifty = yf.download('^NSEI', period='1y', progress=False)
            if not nifty.empty:
                nifty_close = nifty['Close'].squeeze()
                nifty_ret = (float(nifty_close.iloc[-1]) / float(nifty_close.iloc[0]) - 1) * 100
                mcx_ret = returns.get('1Y', 0) or 0
                relative_strength = mcx_ret - nifty_ret
            else:
                relative_strength = None
        except:
            relative_strength = None
        
        return {
            'returns': returns,
            'relative_strength': relative_strength,
            'trend': 'UP' if returns.get('1M', 0) and returns['1M'] > 0 else 'DOWN'
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_option_greeks(spot, strike, days, rate=0.07, vol=0.35):
    """Black-Scholes Greeks"""
    from scipy.stats import norm
    
    T = max(days, 1) / 365
    d1 = (np.log(spot/strike) + (rate + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    
    call_price = spot*norm.cdf(d1) - strike*np.exp(-rate*T)*norm.cdf(d2)
    put_price = strike*np.exp(-rate*T)*norm.cdf(-d2) - spot*norm.cdf(-d1)
    
    call_delta = norm.cdf(d1)
    put_delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (spot * vol * np.sqrt(T))
    theta = (-spot*norm.pdf(d1)*vol/(2*np.sqrt(T))) / 365
    vega = spot * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'call_price': round(call_price, 2),
        'put_price': round(put_price, 2),
        'call_delta': round(call_delta, 3),
        'put_delta': round(put_delta, 3),
        'gamma': round(gamma, 5),
        'theta': round(theta, 3),
        'vega': round(vega, 3)
    }

def generate_trading_signal(screener_data, technicals, momentum):
    """Generate trading signal based on all data"""
    score = 0
    reasons = []
    
    if screener_data and 'metrics' in screener_data:
        m = screener_data['metrics']
        
        # ROE > 20% is good
        roe = m.get('ROE', 0)
        if roe and roe > 25:
            score += 15
            reasons.append(f"Strong ROE: {roe}%")
        
        # ROCE > 30% is excellent
        roce = m.get('ROCE', 0)
        if roce and roce > 30:
            score += 15
            reasons.append(f"Excellent ROCE: {roce}%")
        
        # P/E assessment (high P/E is risky for puts)
        pe = m.get('Stock P/E', 0)
        if pe:
            if pe > 50:
                score -= 10
                reasons.append(f"High P/E ({pe}) - correction risk")
            elif pe < 25:
                score += 10
                reasons.append(f"Reasonable P/E: {pe}")
    
    if technicals:
        # RSI
        if technicals.get('rsi_signal') == 'OVERSOLD':
            score += 20
            reasons.append(f"RSI Oversold ({technicals['rsi']:.0f}) - bounce likely")
        elif technicals.get('rsi_signal') == 'OVERBOUGHT':
            score -= 15
            reasons.append(f"RSI Overbought ({technicals['rsi']:.0f}) - pullback risk")
        
        # MACD
        if technicals.get('macd_signal') == 'BULLISH':
            score += 10
            reasons.append("MACD Bullish")
        else:
            score -= 10
            reasons.append("MACD Bearish")
        
        # MA
        ma = technicals.get('ma_signal', '')
        if 'BULLISH' in ma:
            score += 15
            reasons.append(f"MA Signal: {ma}")
        elif 'BEARISH' in ma:
            score -= 15
            reasons.append(f"MA Signal: {ma}")
    
    if momentum and momentum.get('returns'):
        ret_1m = momentum['returns'].get('1M', 0) or 0
        if ret_1m > 10:
            score += 10
            reasons.append(f"Strong 1M momentum: +{ret_1m:.1f}%")
        elif ret_1m < -10:
            score -= 10
            reasons.append(f"Weak 1M momentum: {ret_1m:.1f}%")
    
    # Determine signal
    if score >= 30:
        signal = 'STRONG_BULLISH'
        strategy = 'SELL OTM PUT (collect premium)'
        strategy_detail = 'Sell Put 5-10% below current price'
    elif score >= 10:
        signal = 'BULLISH'
        strategy = 'SELL ATM/OTM PUT'
        strategy_detail = 'Sell Put at or slightly below current price'
    elif score <= -30:
        signal = 'STRONG_BEARISH'
        strategy = 'BUY PUT or AVOID'
        strategy_detail = 'Wait for better entry or buy protection'
    elif score <= -10:
        signal = 'BEARISH'
        strategy = 'WAIT or SMALL POSITION'
        strategy_detail = 'Reduce position size, sell far OTM put'
    else:
        signal = 'NEUTRAL'
        strategy = 'SELL FAR OTM PUT'
        strategy_detail = 'Sell Put 10-15% below current price for safety'
    
    return {
        'signal': signal,
        'score': score,
        'strategy': strategy,
        'strategy_detail': strategy_detail,
        'reasons': reasons,
        'confidence': min(abs(score), 80)
    }

# Flask Routes
@app.route('/')
def dashboard():
    return render_template('dashboard_v4.html')

@app.route('/api/live')
def api_live():
    """Get all live data - call this frequently"""
    try:
        # Fetch all data (with caching)
        screener = get_cached('screener', fetch_screener_data, duration=300)  # 5 min cache
        price = get_cached('price', fetch_live_price, duration=10)  # 10 sec cache
        technicals = get_cached('technicals', fetch_technicals, duration=60)  # 1 min cache
        momentum = get_cached('momentum', fetch_momentum, duration=300)  # 5 min cache
        
        # Generate signal
        signal = generate_trading_signal(screener, technicals, momentum)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': {
                'screener': screener,
                'price': price,
                'technicals': technicals,
                'momentum': momentum,
                'signal': signal
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/refresh')
def api_refresh():
    """Force refresh all data (clear cache)"""
    global _cache, _cache_time
    _cache = {}
    _cache_time = {}
    return api_live()

@app.route('/api/option-chain')
def api_option_chain():
    """Get option chain"""
    try:
        days = int(request.args.get('days', 30))
        price_data = get_cached('price', fetch_live_price, duration=10)
        spot = price_data['price'] if price_data and 'price' in price_data else 2300
        
        # Generate strikes
        base = int(spot / 50) * 50
        strikes = list(range(base - 300, base + 350, 50))
        
        chain = []
        for strike in strikes:
            greeks = calculate_option_greeks(spot, strike, days)
            chain.append({
                'strike': strike,
                'is_atm': abs(strike - spot) < 25,
                **greeks
            })
        
        return jsonify({
            'success': True,
            'spot': spot,
            'days': days,
            'data': chain
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/position-size', methods=['POST'])
def api_position_size():
    """Calculate position size"""
    try:
        data = request.json
        capital = float(data.get('capital', 500000))
        risk_pct = float(data.get('risk_percent', 2))
        entry = float(data.get('entry_price', 2300))
        stop = float(data.get('stop_loss', 2200))
        
        risk_amount = capital * (risk_pct / 100)
        risk_per_share = abs(entry - stop)
        
        if risk_per_share > 0:
            shares = int(risk_amount / risk_per_share)
            lots = shares // MCX_LOT_SIZE
            actual_shares = lots * MCX_LOT_SIZE
            actual_risk = actual_shares * risk_per_share
        else:
            lots = actual_shares = actual_risk = 0
        
        return jsonify({
            'success': True,
            'data': {
                'lots': lots,
                'shares': actual_shares,
                'risk_amount': actual_risk,
                'capital_required': actual_shares * entry
            },
            'lot_size': MCX_LOT_SIZE
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 MCX REAL-TIME Trading Dashboard v4.0")
    print("=" * 70)
    print()
    print("📊 Data Sources:")
    print("   • Screener.in - Fundamentals (ROE, ROCE, P/E, Strengths/Weaknesses)")
    print("   • Yahoo Finance - Live Price, Technicals, Momentum")
    print()
    print("⏱️  Refresh Intervals:")
    print("   • Price: 10 seconds")
    print("   • Technicals: 60 seconds")
    print("   • Fundamentals: 5 minutes")
    print()
    print("🌐 Dashboard: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)
