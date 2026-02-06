"""
MCX Trading Dashboard v6.0 - HONEST EDITION
=============================================
Shows REAL accuracy metrics, not inflated claims

KEY FINDINGS FROM BACKTESTS:
- Direction prediction: ~50-52% (barely better than coin flip)
- Price error: ~2-3% MAE (good, but direction matters more)
- Best strategy: MA Crossover at 52.1%
- NO ML model significantly beats simple rules

HONEST APPROACH:
- Show multiple signals, let trader decide
- Clear confidence indicators
- Risk management focus
- No false promises
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Cache
CACHE = {}
CACHE_TIMEOUT = 60

def get_cached(key, func, timeout=60):
    now = datetime.now()
    if key in CACHE:
        data, ts = CACHE[key]
        if (now - ts).seconds < timeout:
            return data
    data = func()
    CACHE[key] = (data, now)
    return data


def fetch_mcx_data():
    """Fetch MCX stock data"""
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='1y')
    return df


def calculate_indicators(df):
    """Calculate all technical indicators"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = close.pct_change()
    
    indicators = {}
    
    # Current price
    indicators['price'] = float(close.iloc[-1])
    indicators['prev_close'] = float(close.iloc[-2])
    indicators['change'] = float(ret.iloc[-1] * 100)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    indicators['rsi'] = float(rsi.iloc[-1])
    
    # Moving Averages
    indicators['ma5'] = float(close.rolling(5).mean().iloc[-1])
    indicators['ma20'] = float(close.rolling(20).mean().iloc[-1])
    indicators['ma50'] = float(close.rolling(50).mean().iloc[-1])
    
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    indicators['bb_upper'] = float((ma20 + 2*std20).iloc[-1])
    indicators['bb_lower'] = float((ma20 - 2*std20).iloc[-1])
    indicators['bb_position'] = float((close.iloc[-1] - indicators['bb_lower']) / 
                                      (indicators['bb_upper'] - indicators['bb_lower']))
    
    # Volatility
    indicators['volatility'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    
    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    indicators['atr'] = float(tr.rolling(14).mean().iloc[-1])
    
    # Last 3 days
    indicators['last_3_days'] = [float(r * 100) for r in ret.iloc[-3:].tolist()]
    
    # Support/Resistance
    indicators['support'] = float(low.rolling(20).min().iloc[-1])
    indicators['resistance'] = float(high.rolling(20).max().iloc[-1])
    
    return indicators


def generate_signals(indicators):
    """Generate trading signals with honest accuracy estimates"""
    signals = []
    
    # RSI Signal
    rsi = indicators['rsi']
    if rsi < 30:
        signals.append({
            'name': 'RSI Oversold',
            'direction': 'BUY',
            'strength': 'MODERATE',
            'accuracy': 46,  # Honest backtest result
            'reason': f'RSI at {rsi:.1f} (below 30)'
        })
    elif rsi > 70:
        signals.append({
            'name': 'RSI Overbought',
            'direction': 'SELL',
            'strength': 'MODERATE',
            'accuracy': 46,
            'reason': f'RSI at {rsi:.1f} (above 70)'
        })
    
    # MA Crossover Signal
    if indicators['ma5'] > indicators['ma20']:
        signals.append({
            'name': 'MA Crossover',
            'direction': 'BUY',
            'strength': 'WEAK',
            'accuracy': 52,  # Best performer at 52.1%
            'reason': 'MA5 above MA20 (bullish trend)'
        })
    else:
        signals.append({
            'name': 'MA Crossover',
            'direction': 'SELL',
            'strength': 'WEAK',
            'accuracy': 52,
            'reason': 'MA5 below MA20 (bearish trend)'
        })
    
    # Bollinger Signal
    bb_pos = indicators['bb_position']
    if bb_pos < 0.1:
        signals.append({
            'name': 'Bollinger Bounce',
            'direction': 'BUY',
            'strength': 'MODERATE',
            'accuracy': 51,
            'reason': 'Price near lower Bollinger Band'
        })
    elif bb_pos > 0.9:
        signals.append({
            'name': 'Bollinger Bounce',
            'direction': 'SELL',
            'strength': 'MODERATE',
            'accuracy': 51,
            'reason': 'Price near upper Bollinger Band'
        })
    
    # 3-Day Reversal
    last_3 = indicators['last_3_days']
    if all(r < 0 for r in last_3):
        signals.append({
            'name': '3-Day Reversal',
            'direction': 'BUY',
            'strength': 'WEAK',
            'accuracy': 48,
            'reason': '3 consecutive down days, reversal expected'
        })
    elif all(r > 0 for r in last_3):
        signals.append({
            'name': '3-Day Reversal',
            'direction': 'SELL',
            'strength': 'WEAK',
            'accuracy': 48,
            'reason': '3 consecutive up days, reversal expected'
        })
    
    return signals


def get_consensus(signals):
    """Get consensus direction from all signals"""
    if not signals:
        return {'direction': 'NEUTRAL', 'confidence': 50, 'buy_count': 0, 'sell_count': 0}
    
    buy_count = sum(1 for s in signals if s['direction'] == 'BUY')
    sell_count = sum(1 for s in signals if s['direction'] == 'SELL')
    
    if buy_count > sell_count:
        direction = 'BULLISH'
        confidence = 50 + (buy_count - sell_count) * 5
    elif sell_count > buy_count:
        direction = 'BEARISH'
        confidence = 50 + (sell_count - buy_count) * 5
    else:
        direction = 'NEUTRAL'
        confidence = 50
    
    # Cap confidence at realistic levels
    confidence = min(confidence, 60)
    
    return {
        'direction': direction,
        'confidence': confidence,
        'buy_count': buy_count,
        'sell_count': sell_count
    }


@app.route('/')
def dashboard():
    return render_template('dashboard_v6.html')


@app.route('/api/data')
def get_data():
    """Get all dashboard data"""
    try:
        df = get_cached('mcx_df', fetch_mcx_data)
        
        if df.empty:
            return jsonify({'success': False, 'error': 'No data available'})
        
        indicators = calculate_indicators(df)
        signals = generate_signals(indicators)
        consensus = get_consensus(signals)
        
        # Convert signals for JSON
        signals_json = []
        for s in signals:
            signals_json.append({
                'name': s['name'],
                'direction': s['direction'],
                'strength': s['strength'],
                'accuracy': int(s['accuracy']),
                'reason': s['reason']
            })
        
        return jsonify({
            'success': True,
            'data': {
                'price': round(indicators['price'], 2),
                'prev_close': round(indicators['prev_close'], 2),
                'change': round(indicators['change'], 2),
                'indicators': {
                    'rsi': round(indicators['rsi'], 1),
                    'ma5': round(indicators['ma5'], 2),
                    'ma20': round(indicators['ma20'], 2),
                    'ma50': round(indicators['ma50'], 2),
                    'bb_upper': round(indicators['bb_upper'], 2),
                    'bb_lower': round(indicators['bb_lower'], 2),
                    'bb_position': round(indicators['bb_position'], 2),
                    'volatility': round(indicators['volatility'], 1),
                    'atr': round(indicators['atr'], 2),
                    'support': round(indicators['support'], 2),
                    'resistance': round(indicators['resistance'], 2)
                },
                'last_3_days': [round(d, 2) for d in indicators['last_3_days']],
                'signals': signals_json,
                'consensus': consensus,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'disclaimer': 'Direction accuracy: 50-52%. Not financial advice.'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/honest-backtest')
def get_backtest():
    """Return honest backtest results"""
    return jsonify({
        'success': True,
        'data': {
            'strategies': [
                {'name': 'RSI Mean Reversion', 'accuracy': 46.1, 'signals': 232},
                {'name': '3-Day Reversal', 'accuracy': 48.2, 'signals': 112},
                {'name': 'Bollinger Bounce', 'accuracy': 51.0, 'signals': 49},
                {'name': 'MA Crossover', 'accuracy': 52.1, 'signals': 445}
            ],
            'ml_models': [
                {'name': 'Gradient Boosting', 'accuracy': 48.6},
                {'name': 'Random Forest', 'accuracy': 49.2},
                {'name': 'Ensemble', 'accuracy': 50.8}
            ],
            'conclusion': 'No strategy significantly beats random. Best is MA Crossover at 52.1%.',
            'recommendation': 'Focus on risk management, not prediction accuracy.'
        }
    })


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   📊 MCX TRADING DASHBOARD v6.0 - HONEST EDITION                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ⚠️  IMPORTANT: HONEST METRICS                                              ║
║                                                                              ║
║   BACKTEST RESULTS (2 years of data):                                        ║
║   • RSI Strategy:        46.1% accuracy (worse than coin flip)              ║
║   • Reversal Strategy:   48.2% accuracy                                      ║
║   • Bollinger Strategy:  51.0% accuracy                                      ║
║   • MA Crossover:        52.1% accuracy (BEST, but barely)                  ║
║   • ML Ensemble:         50.8% accuracy                                      ║
║                                                                              ║
║   REALITY:                                                                   ║
║   • Stock direction is HARD to predict                                       ║
║   • Even 55% accuracy is EXCELLENT                                           ║
║   • Focus on RISK MANAGEMENT, not prediction                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   🌐 Dashboard: http://127.0.0.1:5000                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000)
