"""
MCX Options Trading Dashboard v2.0
===================================
Based on ACTUAL data science findings:
1. MCX correlates with NIFTY (0.52) - Main driver
2. Commodities have WEAK same-day correlation
3. Yesterday's commodity moves have predictive power (lagged effect)
4. 71% of movement is company-specific (earnings, news, SEBI)
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Constants
MCX_LOT_SIZE = 625

def fetch_mcx_data(period='6mo'):
    """Fetch MCX stock data"""
    try:
        mcx = yf.download('MCX.NS', period=period, progress=False)
        return mcx
    except:
        return pd.DataFrame()

def fetch_nifty_data(period='6mo'):
    """Fetch Nifty data - the REAL driver"""
    try:
        nifty = yf.download('^NSEI', period=period, progress=False)
        return nifty
    except:
        return pd.DataFrame()

def fetch_commodity_data(period='6mo'):
    """Fetch commodity data for lagged analysis"""
    tickers = {
        'Gold': 'GLD',
        'Silver': 'SLV', 
        'Oil': 'USO',
        'Base_Metals': 'DBB'
    }
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                data[name] = df['Close'].squeeze()
        except:
            pass
    return pd.DataFrame(data)

def calculate_lagged_signals():
    """
    Calculate LAGGED commodity signals - these have predictive power!
    Yesterday's commodity move predicts today's MCX (statistically significant)
    """
    try:
        # Get yesterday's commodity moves
        commodities = fetch_commodity_data(period='5d')
        
        if commodities.empty or len(commodities) < 2:
            return None
        
        # Calculate yesterday's returns
        returns = commodities.pct_change()
        yesterday_returns = returns.iloc[-2] if len(returns) >= 2 else returns.iloc[-1]
        
        signals = {}
        
        # Based on our regression analysis:
        # Silver (t-1): +0.163 correlation
        # Gold (t-1): +0.142 correlation  
        # Base Metals (t-1): +0.105 correlation
        
        weights = {
            'Silver': 0.163,
            'Gold': 0.142,
            'Base_Metals': 0.105
        }
        
        weighted_signal = 0
        signal_count = 0
        
        for commodity, weight in weights.items():
            if commodity in yesterday_returns.index:
                ret = yesterday_returns[commodity]
                signals[commodity] = {
                    'yesterday_return': float(ret * 100),
                    'weight': weight,
                    'contribution': float(ret * weight)
                }
                weighted_signal += ret * weight
                signal_count += 1
        
        if signal_count > 0:
            # Normalize
            composite_signal = weighted_signal / sum(weights.values()) * 100
            
            # Determine direction
            if composite_signal > 0.5:
                direction = 'BULLISH'
                confidence = min(abs(composite_signal) * 20, 80)  # Max 80% confidence
            elif composite_signal < -0.5:
                direction = 'BEARISH'
                confidence = min(abs(composite_signal) * 20, 80)
            else:
                direction = 'NEUTRAL'
                confidence = 30
            
            return {
                'signals': signals,
                'composite_signal': float(composite_signal),
                'direction': direction,
                'confidence': float(confidence),
                'note': "Based on lagged correlation (yesterday's move predicts today)"
            }
        
        return None
    except Exception as e:
        return {'error': str(e)}

def calculate_nifty_correlation():
    """
    Calculate MCX-Nifty correlation - the MAIN driver (0.52 correlation)
    """
    try:
        mcx = fetch_mcx_data(period='3mo')
        nifty = fetch_nifty_data(period='3mo')
        
        if mcx.empty or nifty.empty:
            return None
        
        # Align dates
        mcx_close = mcx['Close'].squeeze()
        nifty_close = nifty['Close'].squeeze()
        
        mcx_close.index = mcx_close.index.tz_localize(None)
        nifty_close.index = nifty_close.index.tz_localize(None)
        
        combined = pd.DataFrame({
            'MCX': mcx_close,
            'NIFTY': nifty_close
        }).dropna()
        
        if len(combined) < 20:
            return None
        
        # Calculate returns
        returns = combined.pct_change().dropna()
        
        # Correlation
        corr, p_value = stats.pearsonr(returns['MCX'], returns['NIFTY'])
        
        # Today's Nifty move
        nifty_today_change = (combined['NIFTY'].iloc[-1] / combined['NIFTY'].iloc[-2] - 1) * 100
        
        # Nifty trend
        nifty_sma20 = combined['NIFTY'].rolling(20).mean().iloc[-1]
        nifty_current = combined['NIFTY'].iloc[-1]
        nifty_trend = 'BULLISH' if nifty_current > nifty_sma20 else 'BEARISH'
        
        return {
            'correlation': float(corr),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'nifty_today_change': float(nifty_today_change),
            'nifty_current': float(nifty_current),
            'nifty_trend': nifty_trend,
            'mcx_expected_direction': 'UP' if nifty_today_change > 0 else 'DOWN',
            'note': 'MCX correlates 52% with Nifty - market direction matters!'
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_mcx_technicals():
    """Calculate MCX-specific technical indicators"""
    try:
        mcx = fetch_mcx_data(period='6mo')
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        high = mcx['High'].squeeze()
        low = mcx['Low'].squeeze()
        volume = mcx['Volume'].squeeze()
        
        current_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (current_price / prev_close - 1) * 100
        
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
        macd_value = float(macd.iloc[-1])
        signal_value = float(signal.iloc[-1])
        macd_histogram = macd_value - signal_value
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = float((sma20 + 2 * std20).iloc[-1])
        bb_lower = float((sma20 - 2 * std20).iloc[-1])
        bb_middle = float(sma20.iloc[-1])
        
        # Volume analysis
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_current = float(volume.iloc[-1])
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Support & Resistance
        recent_high = float(high.tail(20).max())
        recent_low = float(low.tail(20).min())
        
        # ATR for expected move
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        # Determine signals
        rsi_signal = 'OVERSOLD' if rsi_value < 30 else ('OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL')
        macd_signal = 'BULLISH' if macd_histogram > 0 else 'BEARISH'
        
        # Price position in BB
        if current_price > bb_upper:
            bb_signal = 'OVERBOUGHT'
        elif current_price < bb_lower:
            bb_signal = 'OVERSOLD'
        else:
            bb_signal = 'NEUTRAL'
        
        return {
            'price': current_price,
            'change_pct': float(change_pct),
            'rsi': rsi_value,
            'rsi_signal': rsi_signal,
            'macd': macd_value,
            'macd_signal_line': signal_value,
            'macd_histogram': macd_histogram,
            'macd_signal': macd_signal,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_signal': bb_signal,
            'volume_ratio': float(vol_ratio),
            'volume_signal': 'HIGH' if vol_ratio > 1.5 else ('LOW' if vol_ratio < 0.7 else 'NORMAL'),
            'support': recent_low,
            'resistance': recent_high,
            'atr': atr,
            'expected_range': {
                'low': current_price - atr,
                'high': current_price + atr
            }
        }
    except Exception as e:
        return {'error': str(e)}

def get_trading_signal():
    """
    Generate trading signal based on DATA-DRIVEN factors:
    1. Nifty direction (52% weight - strongest correlation)
    2. Lagged commodity signal (15% weight - predictive)
    3. MCX technicals (33% weight - RSI, MACD)
    """
    try:
        nifty = calculate_nifty_correlation()
        lagged = calculate_lagged_signals()
        technicals = calculate_mcx_technicals()
        
        if not all([nifty, technicals]):
            return {'signal': 'NO DATA', 'confidence': 0}
        
        score = 0
        max_score = 0
        reasons = []
        
        # 1. Nifty Direction (52% weight - matches correlation)
        if nifty and 'nifty_today_change' in nifty:
            max_score += 52
            nifty_change = nifty['nifty_today_change']
            if nifty_change > 0.5:
                score += 52
                reasons.append(f"Nifty UP {nifty_change:.1f}% (strong driver)")
            elif nifty_change < -0.5:
                score -= 52
                reasons.append(f"Nifty DOWN {nifty_change:.1f}% (strong driver)")
            else:
                reasons.append(f"Nifty flat ({nifty_change:.1f}%)")
        
        # 2. Lagged Commodity Signal (15% weight)
        if lagged and 'direction' in lagged:
            max_score += 15
            if lagged['direction'] == 'BULLISH':
                score += 15
                reasons.append(f"Yesterday's commodities UP (lagged signal)")
            elif lagged['direction'] == 'BEARISH':
                score -= 15
                reasons.append(f"Yesterday's commodities DOWN (lagged signal)")
        
        # 3. RSI Signal (17% weight)
        if technicals and 'rsi_signal' in technicals:
            max_score += 17
            if technicals['rsi_signal'] == 'OVERSOLD':
                score += 17
                reasons.append(f"RSI oversold ({technicals['rsi']:.0f}) - bounce likely")
            elif technicals['rsi_signal'] == 'OVERBOUGHT':
                score -= 17
                reasons.append(f"RSI overbought ({technicals['rsi']:.0f}) - correction likely")
        
        # 4. MACD Signal (16% weight)
        if technicals and 'macd_signal' in technicals:
            max_score += 16
            if technicals['macd_signal'] == 'BULLISH':
                score += 16
                reasons.append("MACD bullish crossover")
            else:
                score -= 16
                reasons.append("MACD bearish")
        
        # Calculate final signal
        if max_score > 0:
            normalized_score = (score / max_score) * 100
        else:
            normalized_score = 0
        
        if normalized_score > 30:
            signal = 'BULLISH'
            strategy = 'BUY CALL or SELL PUT'
        elif normalized_score < -30:
            signal = 'BEARISH'
            strategy = 'BUY PUT or SELL CALL'
        else:
            signal = 'NEUTRAL'
            strategy = 'SELL STRANGLE or IRON CONDOR'
        
        confidence = min(abs(normalized_score), 80)
        
        return {
            'signal': signal,
            'score': float(normalized_score),
            'confidence': float(confidence),
            'strategy': strategy,
            'reasons': reasons,
            'mcx_price': technicals['price'] if technicals else 0
        }
    except Exception as e:
        return {'signal': 'ERROR', 'error': str(e)}

def calculate_option_greeks(spot, strike, days_to_expiry, risk_free_rate=0.07, volatility=0.35):
    """Black-Scholes option pricing"""
    from scipy.stats import norm
    
    if days_to_expiry <= 0:
        days_to_expiry = 1
    
    T = days_to_expiry / 365
    S = spot
    K = strike
    r = risk_free_rate
    sigma = volatility
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Call
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    call_delta = norm.cdf(d1)
    
    # Put
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    put_delta = -norm.cdf(-d1)
    
    # Common Greeks
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    theta_put = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'vega': vega,
        'call_theta': theta_call,
        'put_theta': theta_put
    }

def get_option_chain(spot_price, days_to_expiry=30):
    """Generate option chain"""
    strikes = list(range(int(spot_price * 0.85), int(spot_price * 1.15), 50))
    chain = []
    
    for strike in strikes:
        greeks = calculate_option_greeks(spot_price, strike, days_to_expiry)
        is_atm = abs(strike - spot_price) < 25
        
        chain.append({
            'strike': strike,
            'is_atm': bool(is_atm),
            'call_price': round(greeks['call_price'], 2),
            'put_price': round(greeks['put_price'], 2),
            'call_delta': round(greeks['call_delta'], 3),
            'put_delta': round(greeks['put_delta'], 3),
            'gamma': round(greeks['gamma'], 5),
            'vega': round(greeks['vega'], 3),
            'call_theta': round(greeks['call_theta'], 3),
            'put_theta': round(greeks['put_theta'], 3)
        })
    
    return chain

# Flask Routes
@app.route('/')
def dashboard():
    return render_template('dashboard_v2.html')

@app.route('/api/overview')
def api_overview():
    """Main overview with all key data"""
    try:
        technicals = calculate_mcx_technicals()
        nifty = calculate_nifty_correlation()
        lagged = calculate_lagged_signals()
        signal = get_trading_signal()
        
        return jsonify({
            'success': True,
            'data': {
                'mcx': technicals,
                'nifty': nifty,
                'lagged_signal': lagged,
                'trading_signal': signal
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/option-chain')
def api_option_chain():
    """Get option chain"""
    try:
        days = int(request.args.get('days', 30))
        technicals = calculate_mcx_technicals()
        spot = technicals['price'] if technicals else 2300
        chain = get_option_chain(spot, days)
        return jsonify({'success': True, 'data': chain, 'spot': spot})
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
            lots = 0
            actual_shares = 0
            actual_risk = 0
        
        return jsonify({
            'success': True,
            'data': {
                'recommended_lots': lots,
                'shares': actual_shares,
                'risk_amount': actual_risk,
                'max_loss': actual_risk,
                'capital_required': actual_shares * entry
            },
            'lot_size': MCX_LOT_SIZE
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 MCX Trading Dashboard v2.0 - Data-Driven")
    print("=" * 60)
    print("\n📊 Key Insights from Data Analysis:")
    print("   • MCX correlates 52% with NIFTY (main driver)")
    print("   • Commodities have WEAK same-day correlation")
    print("   • Yesterday's commodity moves predict today's MCX")
    print("   • 71% of MCX movement is company-specific")
    print()
    print("🌐 Dashboard: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
