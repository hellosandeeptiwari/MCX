"""
MCX Trading Dashboard v7.0 - Volatility + ChatGPT + News Edition
=================================================================
Focus: Volatility prediction (more reliable than direction)
Features: 
- ChatGPT analysis with weighted signals
- News sentiment from multiple sources
- Combined weighted scoring
"""

from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import news sentiment agent
try:
    from news_sentiment_agent import NewsSentimentAgent
    NEWS_AGENT_AVAILABLE = True
except ImportError:
    NEWS_AGENT_AVAILABLE = False

app = Flask(__name__)

# API Key - set via environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Cache
CACHE = {}
CACHE_TIMEOUT = 120  # 2 minutes for API calls

# Initialize news agent
NEWS_AGENT = None
if NEWS_AGENT_AVAILABLE and OPENAI_API_KEY:
    NEWS_AGENT = NewsSentimentAgent(openai_api_key=OPENAI_API_KEY)


def get_cached(key, func, timeout=120):
    now = datetime.now()
    if key in CACHE:
        data, ts = CACHE[key]
        if (now - ts).seconds < timeout:
            return data
    data = func()
    CACHE[key] = (data, now)
    return data


def fetch_mcx_data():
    mcx = yf.Ticker('MCX.NS')
    return mcx.history(period='1y')


def calculate_features(df):
    """Calculate all features for analysis"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = close.pct_change()
    
    f = {}
    
    # Price
    f['price'] = float(close.iloc[-1])
    f['prev_close'] = float(close.iloc[-2])
    f['change'] = float(ret.iloc[-1] * 100)
    f['today_range'] = float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] * 100)
    
    # Volatility
    f['vol_20d'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    f['vol_5d'] = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
    
    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f['atr'] = float(atr.iloc[-1])
    f['atr_pct'] = float(atr.iloc[-1] / close.iloc[-1] * 100)
    f['atr_vs_avg'] = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
    
    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f['bb_upper'] = float((ma20 + 2*std20).iloc[-1])
    f['bb_lower'] = float((ma20 - 2*std20).iloc[-1])
    bb_width = (4 * std20) / ma20 * 100
    f['bb_width'] = float(bb_width.iloc[-1])
    
    # Volume
    f['volume'] = int(volume.iloc[-1])
    f['vol_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    f['vol_trend'] = float(volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1])
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    f['rsi'] = float(rsi.iloc[-1])
    
    # MA
    f['ma5'] = float(close.rolling(5).mean().iloc[-1])
    f['ma20'] = float(close.rolling(20).mean().iloc[-1])
    f['ma50'] = float(close.rolling(50).mean().iloc[-1])
    
    # Regime
    if f['vol_5d'] > f['vol_20d'] * 1.2:
        f['regime'] = 'EXPANDING'
    elif f['vol_5d'] < f['vol_20d'] * 0.8:
        f['regime'] = 'CONTRACTING'
    else:
        f['regime'] = 'STABLE'
    
    # Support/Resistance
    f['support'] = float(low.rolling(20).min().iloc[-1])
    f['resistance'] = float(high.rolling(20).max().iloc[-1])
    
    # Last 5 days
    f['last_5_returns'] = [round(r*100, 2) for r in ret.iloc[-5:].tolist()]
    
    # Average move
    f['avg_move'] = float(ret.abs().rolling(20).mean().iloc[-1] * 100)
    
    return f


def get_technical_signals(f):
    """Technical volatility analysis"""
    score = 50
    reasons = []
    
    if f['atr_vs_avg'] > 1.2:
        score += 15
        reasons.append(f"ATR {f['atr_vs_avg']:.1f}x above average")
    elif f['atr_vs_avg'] < 0.8:
        score -= 10
        reasons.append("ATR below average - calm period")
    
    if f['bb_width'] < 8:
        score += 20
        reasons.append(f"Bollinger squeeze ({f['bb_width']:.1f}%)")
    
    if f['vol_ratio'] > 1.5:
        score += 15
        reasons.append(f"Volume surge {f['vol_ratio']:.1f}x")
    
    if f['today_range'] > f['avg_move'] * 1.5:
        score += 10
        reasons.append(f"High range today {f['today_range']:.1f}%")
    
    if f['rsi'] < 25 or f['rsi'] > 75:
        score += 10
        reasons.append(f"RSI extreme ({f['rsi']:.0f})")
    
    if f['regime'] == 'EXPANDING':
        score += 10
        reasons.append("Volatility expanding")
    
    score = max(20, min(90, score))
    
    return {
        'score': score,
        'expected_range': f['atr_pct'] * (score / 50),
        'reasons': reasons
    }


def get_chatgpt_analysis(f):
    """Get ChatGPT market analysis"""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return {'available': False, 'error': 'API not configured'}
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""Analyze MCX Ltd stock. Respond ONLY with JSON.

DATA:
- Price: ₹{f['price']:.2f}, Change: {f['change']:.2f}%
- RSI: {f['rsi']:.1f}, MA5/MA20: {f['ma5']:.0f}/{f['ma20']:.0f}
- Volatility: {f['vol_20d']:.0f}% (20d), {f['vol_5d']:.0f}% (5d)
- ATR: {f['atr_pct']:.1f}%, vs avg: {f['atr_vs_avg']:.2f}x
- Volume: {f['vol_ratio']:.2f}x average
- Regime: {f['regime']}
- Last 5 days: {f['last_5_returns']}

JSON format:
{{"direction":"BULLISH/BEARISH/NEUTRAL","dir_conf":50-65,"vol_forecast":"HIGH/MEDIUM/LOW","vol_conf":50-90,"reasoning":"1-2 sentences","risk":"brief warning"}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a quant analyst. JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        if '```' in content:
            content = content.split('```')[1].replace('json', '').strip()
        
        data = json.loads(content)
        data['available'] = True
        return data
        
    except Exception as e:
        return {'available': False, 'error': str(e)}


def get_news_sentiment():
    """Get news sentiment analysis"""
    if not NEWS_AGENT:
        return {'available': False, 'sentiment': 'NEUTRAL', 'score': 50, 'confidence': 0}
    
    try:
        analysis = NEWS_AGENT.get_sentiment_analysis(use_cache=True)
        sent = analysis.get('sentiment', {})
        
        return {
            'available': sent.get('available', False),
            'sentiment': sent.get('sentiment', 'NEUTRAL'),
            'score': sent.get('score', 50),
            'confidence': sent.get('confidence', 0),
            'key_themes': sent.get('key_themes', []),
            'bullish_factors': sent.get('bullish_factors', []),
            'bearish_factors': sent.get('bearish_factors', []),
            'reasoning': sent.get('reasoning', 'N/A'),
            'article_count': sent.get('article_count', 0),
            'news': analysis.get('news', [])[:5]  # Top 5 headlines
        }
    except Exception as e:
        return {'available': False, 'sentiment': 'NEUTRAL', 'score': 50, 'error': str(e)}


def combine_analysis(tech, gpt, features, news=None):
    """Combine technical + ChatGPT + News with weights"""
    # New weights: Technical 50%, ChatGPT 25%, News 25%
    TECH_WEIGHT = 0.50
    GPT_WEIGHT = 0.25
    NEWS_WEIGHT = 0.25
    
    # Volatility score (technical + GPT)
    tech_score = tech['score']
    gpt_score = 50
    if gpt.get('available'):
        vf = gpt.get('vol_forecast', 'MEDIUM')
        gpt_score = {'HIGH': 80, 'MEDIUM': 50, 'LOW': 30}.get(vf, 50)
    
    vol_score = int(tech_score * (TECH_WEIGHT + NEWS_WEIGHT) + gpt_score * GPT_WEIGHT)
    
    # Expected range
    tech_range = tech['expected_range']
    exp_range = round(tech_range, 2)
    
    # Direction (combine GPT + News sentiment)
    gpt_direction = gpt.get('direction', 'NEUTRAL') if gpt.get('available') else 'NEUTRAL'
    gpt_conf = gpt.get('dir_conf', 50) if gpt.get('available') else 50
    
    news_direction = 'NEUTRAL'
    news_conf = 50
    news_score = 50
    
    if news and news.get('available'):
        news_score = news.get('score', 50)
        news_conf = news.get('confidence', 50)
        if news_score > 55:
            news_direction = 'BULLISH'
        elif news_score < 45:
            news_direction = 'BEARISH'
    
    # Combine directions
    dir_scores = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
    
    if gpt_direction == 'BULLISH':
        dir_scores['BULLISH'] += GPT_WEIGHT * gpt_conf
    elif gpt_direction == 'BEARISH':
        dir_scores['BEARISH'] += GPT_WEIGHT * gpt_conf
    else:
        dir_scores['NEUTRAL'] += GPT_WEIGHT * 50
    
    if news_direction == 'BULLISH':
        dir_scores['BULLISH'] += NEWS_WEIGHT * news_conf
    elif news_direction == 'BEARISH':
        dir_scores['BEARISH'] += NEWS_WEIGHT * news_conf
    else:
        dir_scores['NEUTRAL'] += NEWS_WEIGHT * 50
    
    # Final direction
    direction = max(dir_scores, key=dir_scores.get)
    dir_conf = int(50 + (dir_scores[direction] - 50) * 0.5)  # Normalize
    dir_conf = max(45, min(70, dir_conf))  # Cap between 45-70
    
    # Strategy recommendation
    if vol_score > 70:
        strategy = 'LONG STRADDLE/STRANGLE'
        strategy_desc = 'High volatility expected - profit from big move'
    elif vol_score < 35:
        strategy = 'IRON CONDOR'
        strategy_desc = 'Low volatility - profit from range-bound'
    else:
        if direction == 'BULLISH' and dir_conf > 55:
            strategy = 'CALL SPREAD'
            strategy_desc = 'Bullish lean from news + technicals'
        elif direction == 'BEARISH' and dir_conf > 55:
            strategy = 'PUT SPREAD'
            strategy_desc = 'Bearish lean from news + technicals'
        else:
            strategy = 'WAIT'
            strategy_desc = 'No clear edge - preserve capital'
    
    # Risk level
    if exp_range > 5:
        risk = 'HIGH'
    elif exp_range > 3:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    
    return {
        'vol_score': vol_score,
        'expected_range': exp_range,
        'expected_move': round(features['price'] * exp_range / 100, 0),
        'direction': direction,
        'dir_conf': dir_conf,
        'strategy': strategy,
        'strategy_desc': strategy_desc,
        'risk': risk,
        'regime': features['regime'],
        'tech_reasons': tech['reasons'],
        'gpt_reasoning': gpt.get('reasoning', 'N/A'),
        'gpt_risk': gpt.get('risk', 'N/A'),
        'gpt_available': gpt.get('available', False),
        'gpt_direction': gpt_direction,
        'news_sentiment': news.get('sentiment', 'NEUTRAL') if news else 'N/A',
        'news_score': news_score,
        'news_conf': news_conf,
        'news_available': news.get('available', False) if news else False,
        'news_reasoning': news.get('reasoning', 'N/A') if news else 'N/A',
        'news_themes': news.get('key_themes', []) if news else [],
        'news_headlines': news.get('news', []) if news else [],
        'weights': {'tech': int(TECH_WEIGHT*100), 'gpt': int(GPT_WEIGHT*100), 'news': int(NEWS_WEIGHT*100)}
    }


@app.route('/')
def dashboard():
    return render_template('dashboard_v7.html')


@app.route('/api/analysis')
def get_analysis():
    try:
        df = get_cached('mcx_data', fetch_mcx_data, 60)
        if df.empty:
            return jsonify({'success': False, 'error': 'No data'})
        
        features = calculate_features(df)
        tech = get_technical_signals(features)
        gpt = get_cached('gpt_analysis', lambda: get_chatgpt_analysis(features), 300)  # Cache 5 min
        news = get_cached('news_sentiment', get_news_sentiment, 600)  # Cache 10 min
        combined = combine_analysis(tech, gpt, features, news)
        
        # ATM strike
        atm = round(features['price'] / 100) * 100
        
        return jsonify({
            'success': True,
            'data': {
                'price': round(features['price'], 2),
                'change': round(features['change'], 2),
                'rsi': round(features['rsi'], 1),
                'vol_20d': round(features['vol_20d'], 1),
                'vol_5d': round(features['vol_5d'], 1),
                'atr': round(features['atr'], 2),
                'atr_pct': round(features['atr_pct'], 2),
                'atr_vs_avg': round(features['atr_vs_avg'], 2),
                'vol_ratio': round(features['vol_ratio'], 2),
                'bb_width': round(features['bb_width'], 2),
                'ma5': round(features['ma5'], 2),
                'ma20': round(features['ma20'], 2),
                'support': round(features['support'], 2),
                'resistance': round(features['resistance'], 2),
                'regime': features['regime'],
                'last_5': features['last_5_returns'],
                'combined': combined,
                'options': {
                    'atm_strike': atm,
                    'upper_target': round(features['price'] + combined['expected_move']),
                    'lower_target': round(features['price'] - combined['expected_move']),
                    'lot_size': 625
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/news')
def get_news():
    """Dedicated news endpoint"""
    try:
        news = get_cached('news_sentiment', get_news_sentiment, 600)
        return jsonify({'success': True, 'data': news})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  📊 MCX TRADING DASHBOARD v7.0 - VOLATILITY + CHATGPT + NEWS                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✅ Volatility prediction (more reliable than direction)                     ║
║  ✅ ChatGPT analysis integrated with 25% weightage                          ║
║  ✅ News sentiment analysis with 25% weightage                              ║
║  ✅ Technical analysis with 50% weightage                                   ║
║  ✅ Options strategy recommendations                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌐 Dashboard: http://127.0.0.1:5000                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000)
