"""
MCX UNIFIED TRADING DASHBOARD
=============================
All data in one place - make your decision here
"""

from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import math
from scipy.stats import norm

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

app = Flask(__name__)

# Initialize OpenAI
API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY) if OPENAI_AVAILABLE and API_KEY else None

LOT_SIZE = 625


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate option price using Black-Scholes"""
    if T <= 0:
        T = 1/365
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price


def fetch_stock_data():
    """Fetch MCX stock data"""
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='1y')
    return df


def calculate_technicals(df):
    """Calculate all technical indicators"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = close.pct_change()
    
    data = {
        'price': float(close.iloc[-1]),
        'prev_close': float(close.iloc[-2]),
        'change_today': float(ret.iloc[-1] * 100),
        'change_5d': float((close.iloc[-1] / close.iloc[-5] - 1) * 100),
        'change_20d': float((close.iloc[-1] / close.iloc[-20] - 1) * 100),
        'high_52w': float(high.tail(252).max()),
        'low_52w': float(low.tail(252).min()),
    }
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['rsi'] = float(rsi.iloc[-1])
    
    # Moving Averages
    data['ma5'] = float(close.rolling(5).mean().iloc[-1])
    data['ma20'] = float(close.rolling(20).mean().iloc[-1])
    data['ma50'] = float(close.rolling(50).mean().iloc[-1])
    data['price_vs_ma20'] = float((close.iloc[-1] / data['ma20'] - 1) * 100)
    
    # Volatility
    data['volatility_5d'] = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
    data['volatility_20d'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    
    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    data['atr'] = float(atr.iloc[-1])
    data['atr_pct'] = float(atr.iloc[-1] / close.iloc[-1] * 100)
    data['atr_vs_avg'] = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
    
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_upper'] = float((ma20 + 2 * std20).iloc[-1])
    data['bb_lower'] = float((ma20 - 2 * std20).iloc[-1])
    data['bb_position'] = float((close.iloc[-1] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower']))
    
    # Volume
    data['volume'] = int(volume.iloc[-1])
    data['volume_avg'] = int(volume.rolling(20).mean().iloc[-1])
    data['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    
    # Recent pattern
    data['last_5_returns'] = [round(r*100, 2) for r in ret.iloc[-5:].tolist()]
    data['up_days_5'] = sum(1 for r in ret.iloc[-5:] if r > 0)
    data['down_days_5'] = sum(1 for r in ret.iloc[-5:] if r < 0)
    
    # Support/Resistance
    data['support'] = float(low.rolling(20).min().iloc[-1])
    data['resistance'] = float(high.rolling(20).max().iloc[-1])
    
    return data


def calculate_options(price, volatility):
    """Calculate option premiums using Black-Scholes"""
    S = price
    K = round(price / 100) * 100  # ATM strike
    T = 7 / 365  # 1 week to expiry
    r = 0.07  # Risk-free rate
    
    # Estimate IV based on realized vol (IV typically higher)
    iv_low = volatility / 100 * 0.8  # Conservative
    iv_mid = volatility / 100 * 1.2  # Normal premium
    iv_high = volatility / 100 * 1.8  # Post-crash spike
    
    options = {
        'atm_strike': K,
        'days_to_expiry': 7,
        'scenarios': []
    }
    
    for iv, label in [(iv_low, 'Low IV'), (iv_mid, 'Normal IV'), (iv_high, 'High IV (post-crash)')]:
        ce = black_scholes(S, K, T, r, iv, 'call')
        pe = black_scholes(S, K, T, r, iv, 'put')
        straddle = ce + pe
        
        options['scenarios'].append({
            'label': label,
            'iv': round(iv * 100, 1),
            'ce_premium': round(ce, 2),
            'pe_premium': round(pe, 2),
            'straddle_premium': round(straddle, 2),
            'straddle_cost': round(straddle * LOT_SIZE),
            'upper_breakeven': round(K + straddle),
            'lower_breakeven': round(K - straddle),
            'required_move_pct': round(straddle / S * 100, 1)
        })
    
    # Also calculate strangle (OTM options)
    K_ce = K + 100
    K_pe = K - 100
    
    for iv, label in [(iv_mid, 'Normal IV')]:
        ce_otm = black_scholes(S, K_ce, T, r, iv, 'call')
        pe_otm = black_scholes(S, K_pe, T, r, iv, 'put')
        strangle = ce_otm + pe_otm
        
        options['strangle'] = {
            'ce_strike': K_ce,
            'pe_strike': K_pe,
            'ce_premium': round(ce_otm, 2),
            'pe_premium': round(pe_otm, 2),
            'strangle_premium': round(strangle, 2),
            'strangle_cost': round(strangle * LOT_SIZE),
            'upper_breakeven': round(K_ce + strangle),
            'lower_breakeven': round(K_pe - strangle),
            'required_move_pct': round(max(K_ce - S, S - K_pe) / S * 100 + strangle / S * 100, 1)
        }
    
    return options


def get_gpt_analysis(technicals):
    """Get unified analysis from ChatGPT"""
    if not client:
        return None
    
    prompt = f"""You are a trading analyst. Analyze MCX Ltd stock and give ONE clear recommendation.

CURRENT DATA:
- Price: ₹{technicals['price']:.2f}
- Today: {technicals['change_today']:+.2f}%
- 5-Day: {technicals['change_5d']:+.2f}%
- 20-Day: {technicals['change_20d']:+.2f}%
- RSI: {technicals['rsi']:.1f}
- Price vs MA20: {technicals['price_vs_ma20']:+.1f}%
- Bollinger Position: {technicals['bb_position']:.2f} (0=lower, 1=upper)
- ATR vs Average: {technicals['atr_vs_avg']:.2f}x
- 5-Day Volatility: {technicals['volatility_5d']:.1f}%
- Volume: {technicals['volume_ratio']:.2f}x average
- Last 5 days: {technicals['last_5_returns']}

Respond in this EXACT JSON format:
{{
    "situation": "1 sentence summary",
    "direction": "BULLISH" or "BEARISH" or "NEUTRAL",
    "direction_confidence": 45-60,
    "volatility": "HIGH" or "MEDIUM" or "LOW",
    "volatility_confidence": 50-85,
    "expected_move_pct": 3.0-12.0,
    "recommendation": "STRADDLE" or "STRANGLE" or "CALL_SPREAD" or "PUT_SPREAD" or "WAIT",
    "reasoning": "Why this strategy",
    "risk": "Main risk"
}}

JSON only:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a quant analyst. Be honest about uncertainty. JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        content = response.choices[0].message.content.strip()
        if '```' in content:
            content = content.split('```')[1].replace('json', '').strip()
        
        return json.loads(content)
    except Exception as e:
        return {'error': str(e)}


def create_recommendation(technicals, gpt, options):
    """Create final unified recommendation"""
    rec = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Volatility assessment (more reliable)
    if technicals['atr_vs_avg'] > 1.5:
        rec['volatility_level'] = 'HIGH'
        rec['volatility_confidence'] = 80
    elif technicals['atr_vs_avg'] > 1.2:
        rec['volatility_level'] = 'ELEVATED'
        rec['volatility_confidence'] = 70
    elif technicals['atr_vs_avg'] < 0.7:
        rec['volatility_level'] = 'LOW'
        rec['volatility_confidence'] = 70
    else:
        rec['volatility_level'] = 'NORMAL'
        rec['volatility_confidence'] = 60
    
    # Direction (low confidence)
    direction_score = 50
    if technicals['rsi'] < 35:
        direction_score += 5
    elif technicals['rsi'] > 65:
        direction_score -= 5
    
    if technicals['bb_position'] < 0.2:
        direction_score += 5
    elif technicals['bb_position'] > 0.8:
        direction_score -= 5
    
    rec['direction_score'] = direction_score
    if direction_score > 55:
        rec['direction'] = 'BULLISH'
    elif direction_score < 45:
        rec['direction'] = 'BEARISH'
    else:
        rec['direction'] = 'NEUTRAL'
    rec['direction_confidence'] = min(55, 45 + abs(direction_score - 50))
    
    # Expected move
    rec['expected_move_pct'] = round(technicals['atr_pct'] * (1.5 if rec['volatility_level'] == 'HIGH' else 1.0), 1)
    rec['expected_move'] = round(technicals['price'] * rec['expected_move_pct'] / 100)
    
    # Check option pricing
    normal_scenario = options['scenarios'][1]  # Normal IV
    high_scenario = options['scenarios'][2]  # High IV
    
    # Use conservative (high IV) for post-crash
    if technicals['atr_vs_avg'] > 1.3:
        active_scenario = high_scenario
    else:
        active_scenario = normal_scenario
    
    rec['active_option_scenario'] = active_scenario
    
    # Trade decision
    required_move = active_scenario['required_move_pct']
    expected = rec['expected_move_pct']
    
    if expected > required_move * 1.3:
        rec['trade_quality'] = 'GOOD'
        rec['trade_quality_reason'] = f'Expected {expected}% > breakeven {required_move}% × 1.3'
    elif expected > required_move:
        rec['trade_quality'] = 'MARGINAL'
        rec['trade_quality_reason'] = f'Expected {expected}% slightly > breakeven {required_move}%'
    else:
        rec['trade_quality'] = 'AVOID'
        rec['trade_quality_reason'] = f'Expected {expected}% < breakeven {required_move}%'
    
    # Final recommendation
    if rec['volatility_level'] in ['HIGH', 'ELEVATED']:
        if rec['trade_quality'] in ['GOOD', 'MARGINAL']:
            rec['strategy'] = 'BUY STRADDLE'
            rec['strategy_detail'] = f"Buy {options['atm_strike']} CE + {options['atm_strike']} PE"
        else:
            rec['strategy'] = 'WAIT'
            rec['strategy_detail'] = 'Options too expensive, wait for IV to drop'
    elif rec['volatility_level'] == 'LOW':
        rec['strategy'] = 'SELL PREMIUM / IRON CONDOR'
        rec['strategy_detail'] = 'Low volatility - range bound expected'
    else:
        if rec['direction'] == 'BULLISH':
            rec['strategy'] = 'BULL CALL SPREAD (small)'
            rec['strategy_detail'] = f"Buy {options['atm_strike']} CE, Sell {options['atm_strike']+100} CE"
        elif rec['direction'] == 'BEARISH':
            rec['strategy'] = 'BEAR PUT SPREAD (small)'
            rec['strategy_detail'] = f"Buy {options['atm_strike']} PE, Sell {options['atm_strike']-100} PE"
        else:
            rec['strategy'] = 'WAIT'
            rec['strategy_detail'] = 'No clear edge'
    
    # Incorporate GPT
    if gpt and 'error' not in gpt:
        rec['gpt_agrees'] = gpt.get('recommendation', '').upper() in rec['strategy'].upper()
        rec['gpt_analysis'] = gpt
    
    return rec


@app.route('/')
def dashboard():
    return render_template('unified_dashboard.html')


@app.route('/api/all-data')
def get_all_data():
    """Single API endpoint with ALL data"""
    try:
        # Fetch stock data
        df = fetch_stock_data()
        
        # Calculate technicals
        technicals = calculate_technicals(df)
        
        # Get GPT analysis
        gpt = get_gpt_analysis(technicals)
        
        # Calculate options
        options = calculate_options(technicals['price'], technicals['volatility_20d'])
        
        # Create recommendation
        recommendation = create_recommendation(technicals, gpt, options)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock': technicals,
            'options': options,
            'gpt': gpt,
            'recommendation': recommendation,
            'lot_size': LOT_SIZE
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("📊 MCX UNIFIED TRADING DASHBOARD")
    print("="*60)
    print(f"✅ All data in ONE place")
    print(f"✅ Stock + Technical + Options + GPT")
    print(f"✅ Clear YES/NO recommendation")
    print(f"\n🌐 Dashboard: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, port=5000)
