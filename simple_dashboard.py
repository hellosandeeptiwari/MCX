"""
MCX SIMPLE TRADING DASHBOARD
============================
Direct trades: BUY CALL or BUY PUT
Works for 2-3% swings
"""

from flask import Flask, render_template_string, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
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

API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY) if OPENAI_AVAILABLE and API_KEY else None

LOT_SIZE = 625


def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        T = 1/365
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*norm.cdf(-d2) - S*norm.cdf(-d1)


def get_all_data():
    """Get stock data and calculate everything"""
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='6mo')
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = close.pct_change()
    
    price = float(close.iloc[-1])
    
    # Key metrics
    data = {
        'price': price,
        'change_today': float(ret.iloc[-1] * 100),
        'change_2d': float((close.iloc[-1] / close.iloc[-2] - 1) * 100),
        'change_5d': float((close.iloc[-1] / close.iloc[-5] - 1) * 100),
        'change_20d': float((close.iloc[-1] / close.iloc[-20] - 1) * 100),
    }
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['rsi'] = float(rsi.iloc[-1])
    
    # Short-term RSI (more sensitive)
    gain5 = delta.where(delta > 0, 0).rolling(5).mean()
    loss5 = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rsi5 = 100 - (100 / (1 + gain5 / (loss5 + 1e-10)))
    data['rsi_5'] = float(rsi5.iloc[-1])
    
    # Volatility
    data['volatility_5d'] = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
    data['volatility_20d'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    
    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    data['atr'] = float(atr.iloc[-1])
    data['atr_pct'] = float(atr.iloc[-1] / close.iloc[-1] * 100)
    
    # Moving averages
    data['ma5'] = float(close.rolling(5).mean().iloc[-1])
    data['ma20'] = float(close.rolling(20).mean().iloc[-1])
    data['price_vs_ma5'] = float((close.iloc[-1] / data['ma5'] - 1) * 100)
    data['price_vs_ma20'] = float((close.iloc[-1] / data['ma20'] - 1) * 100)
    
    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = (ma20 + 2 * std20).iloc[-1]
    bb_lower = (ma20 - 2 * std20).iloc[-1]
    data['bb_upper'] = float(bb_upper)
    data['bb_lower'] = float(bb_lower)
    data['bb_position'] = float((close.iloc[-1] - bb_lower) / (bb_upper - bb_lower))
    
    # Volume
    data['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    
    # Last 5 days pattern
    data['last_5_returns'] = [round(r*100, 2) for r in ret.iloc[-5:].tolist()]
    data['up_days'] = sum(1 for r in ret.iloc[-5:] if r > 0)
    data['down_days'] = sum(1 for r in ret.iloc[-5:] if r < 0)
    
    # Support/Resistance
    data['support'] = float(low.rolling(10).min().iloc[-1])
    data['resistance'] = float(high.rolling(10).max().iloc[-1])
    
    # Options pricing
    atm = round(price / 100) * 100
    T = 7/365
    r = 0.07
    iv = data['volatility_20d'] / 100 * 1.3
    
    # ATM options
    data['atm_strike'] = atm
    data['atm_ce'] = round(black_scholes(price, atm, T, r, iv, 'call'), 2)
    data['atm_pe'] = round(black_scholes(price, atm, T, r, iv, 'put'), 2)
    
    # OTM options (cheaper, higher leverage)
    otm_ce_strike = atm + 100
    otm_pe_strike = atm - 100
    data['otm_ce_strike'] = otm_ce_strike
    data['otm_pe_strike'] = otm_pe_strike
    data['otm_ce'] = round(black_scholes(price, otm_ce_strike, T, r, iv, 'call'), 2)
    data['otm_pe'] = round(black_scholes(price, otm_pe_strike, T, r, iv, 'put'), 2)
    
    return data


def get_simple_trade(data):
    """Use ChatGPT to give simple BUY CALL or BUY PUT recommendation"""
    if not client:
        return create_fallback_trade(data)
    
    prompt = f"""You are a trading advisor. Give ONE simple trade for MCX stock.

STOCK DATA:
- Price: ₹{data['price']:.2f}
- Today: {data['change_today']:+.2f}%
- Last 5 days: {data['change_5d']:+.2f}%
- Last 5 days pattern: {data['last_5_returns']} (up days: {data['up_days']}, down: {data['down_days']})

INDICATORS:
- RSI (14-day): {data['rsi']:.0f} (below 30=oversold, above 70=overbought)
- RSI (5-day): {data['rsi_5']:.0f} (short-term momentum)
- Price vs 5-day avg: {data['price_vs_ma5']:+.1f}%
- Price vs 20-day avg: {data['price_vs_ma20']:+.1f}%
- Bollinger position: {data['bb_position']:.0%} (0%=bottom, 100%=top)
- Support: ₹{data['support']:.0f}
- Resistance: ₹{data['resistance']:.0f}

OPTIONS AVAILABLE:
- {data['atm_strike']} CE (Call): ₹{data['atm_ce']} per share
- {data['atm_strike']} PE (Put): ₹{data['atm_pe']} per share
- {data['otm_ce_strike']} CE (cheaper Call): ₹{data['otm_ce']} per share
- {data['otm_pe_strike']} PE (cheaper Put): ₹{data['otm_pe']} per share

TRADER'S REQUIREMENT:
- Wants simple naked CALL or PUT (no complex strategies)
- Even 2-3% move is good enough for them
- Lot size: 625 shares

Based on the data, give ONE clear trade. Reply in this EXACT JSON format:

{{
    "market_view": "Stock is [going up/going down/unclear] because [simple reason]",
    "trade_type": "BUY CALL" or "BUY PUT" or "NO TRADE",
    "confidence": "HIGH" or "MEDIUM" or "LOW",
    "strike": {data['atm_strike']} or {data['otm_ce_strike']} or {data['otm_pe_strike']},
    "option_type": "CE" or "PE",
    "premium": XX.XX,
    "total_cost": XXXXX,
    "target_price": XXXX,
    "stop_loss_price": XXXX,
    "expected_profit": "₹XX,XXX if stock moves X%",
    "max_loss": "₹XX,XXX (premium paid)",
    "hold_for": "X days",
    "simple_reason": "One sentence why this trade",
    "warning": "One key risk"
}}

RULES:
1. If RSI < 35 and price near support → BUY CALL (expecting bounce)
2. If RSI > 65 and price near resistance → BUY PUT (expecting drop)
3. If 5 days mostly down and RSI low → BUY CALL (reversal expected)
4. If 5 days mostly up and RSI high → BUY PUT (reversal expected)
5. If unclear → NO TRADE

Be decisive. Even 50-55% confidence is okay for a trade. JSON only:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You give simple option trades. Be decisive. Even small edges are worth trading."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        
        content = response.choices[0].message.content.strip()
        if '```' in content:
            content = content.split('```')[1].replace('json', '').strip()
        
        return json.loads(content)
    except Exception as e:
        return create_fallback_trade(data, str(e))


def create_fallback_trade(data, error=None):
    """Fallback if ChatGPT fails"""
    price = data['price']
    rsi = data['rsi']
    rsi5 = data['rsi_5']
    bb = data['bb_position']
    change_5d = data['change_5d']
    
    # Simple decision logic
    bullish_signals = 0
    bearish_signals = 0
    
    if rsi < 40: bullish_signals += 1
    if rsi > 60: bearish_signals += 1
    if rsi5 < 35: bullish_signals += 1
    if rsi5 > 65: bearish_signals += 1
    if bb < 0.3: bullish_signals += 1
    if bb > 0.7: bearish_signals += 1
    if change_5d < -3: bullish_signals += 1  # Oversold bounce
    if change_5d > 3: bearish_signals += 1   # Overbought drop
    if data['down_days'] >= 4: bullish_signals += 1
    if data['up_days'] >= 4: bearish_signals += 1
    
    if bullish_signals > bearish_signals and bullish_signals >= 2:
        # BUY CALL
        strike = data['atm_strike']
        premium = data['atm_ce']
        return {
            "market_view": f"Stock dropped {change_5d:.1f}% in 5 days, RSI at {rsi:.0f} - expecting bounce",
            "trade_type": "BUY CALL",
            "confidence": "MEDIUM" if bullish_signals >= 3 else "LOW",
            "strike": strike,
            "option_type": "CE",
            "premium": premium,
            "total_cost": round(premium * LOT_SIZE),
            "target_price": round(price * 1.03),
            "stop_loss_price": round(price * 0.98),
            "expected_profit": f"₹{round((price * 0.03) * LOT_SIZE * 0.5):,} if stock moves up 3%",
            "max_loss": f"₹{round(premium * LOT_SIZE):,} (premium paid)",
            "hold_for": "3-5 days",
            "simple_reason": f"Stock oversold (RSI {rsi:.0f}), expecting 2-3% bounce",
            "warning": "Stock may continue falling if market weak"
        }
    
    elif bearish_signals > bullish_signals and bearish_signals >= 2:
        # BUY PUT
        strike = data['atm_strike']
        premium = data['atm_pe']
        return {
            "market_view": f"Stock up {change_5d:.1f}% in 5 days, RSI at {rsi:.0f} - expecting pullback",
            "trade_type": "BUY PUT",
            "confidence": "MEDIUM" if bearish_signals >= 3 else "LOW",
            "strike": strike,
            "option_type": "PE",
            "premium": premium,
            "total_cost": round(premium * LOT_SIZE),
            "target_price": round(price * 0.97),
            "stop_loss_price": round(price * 1.02),
            "expected_profit": f"₹{round((price * 0.03) * LOT_SIZE * 0.5):,} if stock drops 3%",
            "max_loss": f"₹{round(premium * LOT_SIZE):,} (premium paid)",
            "hold_for": "3-5 days",
            "simple_reason": f"Stock overbought (RSI {rsi:.0f}), expecting 2-3% drop",
            "warning": "Stock may continue rising if momentum strong"
        }
    
    else:
        return {
            "market_view": f"Stock at ₹{price:.0f}, no clear direction",
            "trade_type": "NO TRADE",
            "confidence": "LOW",
            "strike": data['atm_strike'],
            "option_type": "-",
            "premium": 0,
            "total_cost": 0,
            "target_price": round(price),
            "stop_loss_price": round(price),
            "expected_profit": "₹0",
            "max_loss": "₹0",
            "hold_for": "-",
            "simple_reason": "No clear signal, wait for better setup",
            "warning": "Forcing trades without edge leads to losses"
        }


SIMPLE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>MCX Simple Trade</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #0f0f1a; 
            color: white; 
            padding: 20px;
            min-height: 100vh;
        }
        
        .container { max-width: 700px; margin: 0 auto; }
        
        .header { 
            text-align: center; 
            padding: 20px;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 1.8em; color: #00d4ff; }
        
        /* Price Box */
        .price-box {
            text-align: center;
            padding: 25px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .price { font-size: 3.5em; font-weight: bold; }
        .change { font-size: 1.5em; margin-left: 15px; }
        .change.up { color: #00ff88; }
        .change.down { color: #ff4444; }
        
        /* BIG TRADE BOX */
        .trade-box {
            text-align: center;
            padding: 40px;
            border-radius: 20px;
            margin: 20px 0;
        }
        .trade-box.call {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #000;
        }
        .trade-box.put {
            background: linear-gradient(135deg, #ff4444 0%, #cc2222 100%);
            color: #fff;
        }
        .trade-box.wait {
            background: linear-gradient(135deg, #666 0%, #444 100%);
            color: #fff;
        }
        
        .trade-action {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .trade-detail {
            font-size: 2em;
            margin: 10px 0;
        }
        
        .trade-premium {
            font-size: 1.3em;
            opacity: 0.9;
        }
        
        /* Market View */
        .market-view {
            background: rgba(0,212,255,0.1);
            border: 1px solid #00d4ff;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2em;
            text-align: center;
        }
        
        /* Details Grid */
        .details-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .detail-box {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .detail-box .label { color: #888; font-size: 0.9em; }
        .detail-box .value { 
            font-size: 1.8em; 
            font-weight: bold; 
            margin-top: 8px;
        }
        .detail-box .value.green { color: #00ff88; }
        .detail-box .value.red { color: #ff4444; }
        .detail-box .value.blue { color: #00d4ff; }
        
        /* Profit/Loss */
        .outcome-box {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .outcome {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .outcome.profit {
            background: rgba(0,255,136,0.15);
            border: 2px solid #00ff88;
        }
        .outcome.loss {
            background: rgba(255,68,68,0.15);
            border: 2px solid #ff4444;
        }
        .outcome h3 { margin-bottom: 10px; font-size: 1.1em; }
        .outcome .amount { font-size: 1.5em; font-weight: bold; }
        .outcome.profit h3, .outcome.profit .amount { color: #00ff88; }
        .outcome.loss h3, .outcome.loss .amount { color: #ff4444; }
        
        /* Warning */
        .warning {
            background: rgba(255,170,0,0.15);
            border: 1px solid #ffaa00;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: #ffaa00;
            margin: 20px 0;
        }
        
        /* Simple Reason */
        .reason {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            margin: 20px 0;
        }
        
        /* Indicators */
        .indicators {
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .indicators h3 { color: #888; margin-bottom: 10px; font-size: 0.9em; }
        .ind-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .ind-row:last-child { border-bottom: none; }
        
        /* Refresh */
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #00d4ff;
            color: #000;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            font-size: 1em;
        }
        
        .loading { text-align: center; padding: 60px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 50px; height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        .confidence {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .confidence.high { background: #00ff88; color: #000; }
        .confidence.medium { background: #ffaa00; color: #000; }
        .confidence.low { background: #888; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 MCX SIMPLE TRADE</h1>
            <p style="color: #888; margin-top: 5px;">Just BUY CALL or BUY PUT</p>
        </div>
        
        <div id="content">
            <div class="loading">
                <div class="spinner"></div>
                <div>Analyzing MCX...</div>
            </div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
    
    <script>
        async function loadData() {
            document.getElementById('content').innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <div>Analyzing MCX stock...</div>
                </div>
            `;
            
            try {
                const response = await fetch('/api/trade');
                const data = await response.json();
                
                if (data.success) {
                    renderTrade(data);
                } else {
                    document.getElementById('content').innerHTML = `
                        <div class="loading" style="color: #ff4444;">Error: ${data.error}</div>
                    `;
                }
            } catch (error) {
                document.getElementById('content').innerHTML = `
                    <div class="loading" style="color: #ff4444;">Failed: ${error.message}</div>
                `;
            }
        }
        
        function renderTrade(data) {
            const s = data.stock;
            const t = data.trade;
            
            const changeClass = s.change_today >= 0 ? 'up' : 'down';
            const tradeClass = t.trade_type === 'BUY CALL' ? 'call' : t.trade_type === 'BUY PUT' ? 'put' : 'wait';
            const confClass = t.confidence.toLowerCase();
            
            let tradeBoxContent = '';
            if (t.trade_type === 'NO TRADE') {
                tradeBoxContent = `
                    <div class="trade-action">⏸️ WAIT</div>
                    <div class="trade-detail">No Trade Today</div>
                `;
            } else {
                tradeBoxContent = `
                    <div class="trade-action">${t.trade_type === 'BUY CALL' ? '📈' : '📉'} ${t.trade_type}</div>
                    <div class="trade-detail">${t.strike} ${t.option_type}</div>
                    <div class="trade-premium">Premium: ₹${t.premium} × 625 = ₹${t.total_cost.toLocaleString()}</div>
                    <div class="confidence ${confClass}">${t.confidence} Confidence</div>
                `;
            }
            
            const html = `
                <!-- Price -->
                <div class="price-box">
                    <span class="price">₹${s.price.toFixed(2)}</span>
                    <span class="change ${changeClass}">${s.change_today >= 0 ? '+' : ''}${s.change_today.toFixed(2)}%</span>
                    <div style="color: #666; margin-top: 10px;">5-day: ${s.change_5d >= 0 ? '+' : ''}${s.change_5d.toFixed(1)}%</div>
                </div>
                
                <!-- Market View -->
                <div class="market-view">
                    💡 ${t.market_view}
                </div>
                
                <!-- BIG TRADE -->
                <div class="trade-box ${tradeClass}">
                    ${tradeBoxContent}
                </div>
                
                ${t.trade_type !== 'NO TRADE' ? `
                    <!-- Details -->
                    <div class="details-grid">
                        <div class="detail-box">
                            <div class="label">Target Price</div>
                            <div class="value green">₹${t.target_price}</div>
                        </div>
                        <div class="detail-box">
                            <div class="label">Stop Loss</div>
                            <div class="value red">₹${t.stop_loss_price}</div>
                        </div>
                        <div class="detail-box">
                            <div class="label">Hold For</div>
                            <div class="value blue">${t.hold_for}</div>
                        </div>
                        <div class="detail-box">
                            <div class="label">Total Cost</div>
                            <div class="value">₹${t.total_cost.toLocaleString()}</div>
                        </div>
                    </div>
                    
                    <!-- Profit/Loss -->
                    <div class="outcome-box">
                        <div class="outcome profit">
                            <h3>✅ If Right</h3>
                            <div class="amount">${t.expected_profit}</div>
                        </div>
                        <div class="outcome loss">
                            <h3>❌ If Wrong</h3>
                            <div class="amount">${t.max_loss}</div>
                        </div>
                    </div>
                ` : ''}
                
                <!-- Reason -->
                <div class="reason">
                    📌 ${t.simple_reason}
                </div>
                
                <!-- Warning -->
                <div class="warning">
                    ⚠️ ${t.warning}
                </div>
                
                <!-- Indicators -->
                <div class="indicators">
                    <h3>📊 INDICATORS</h3>
                    <div class="ind-row">
                        <span>RSI (14-day)</span>
                        <span style="color: ${s.rsi < 35 ? '#00ff88' : s.rsi > 65 ? '#ff4444' : '#fff'}">${s.rsi.toFixed(0)}</span>
                    </div>
                    <div class="ind-row">
                        <span>RSI (5-day)</span>
                        <span style="color: ${s.rsi_5 < 35 ? '#00ff88' : s.rsi_5 > 65 ? '#ff4444' : '#fff'}">${s.rsi_5.toFixed(0)}</span>
                    </div>
                    <div class="ind-row">
                        <span>Price vs 5-day avg</span>
                        <span>${s.price_vs_ma5 >= 0 ? '+' : ''}${s.price_vs_ma5.toFixed(1)}%</span>
                    </div>
                    <div class="ind-row">
                        <span>Bollinger Position</span>
                        <span>${(s.bb_position * 100).toFixed(0)}%</span>
                    </div>
                    <div class="ind-row">
                        <span>Support / Resistance</span>
                        <span>₹${s.support.toFixed(0)} / ₹${s.resistance.toFixed(0)}</span>
                    </div>
                    <div class="ind-row">
                        <span>Last 5 days</span>
                        <span>${s.up_days} up, ${s.down_days} down</span>
                    </div>
                </div>
                
                <div style="text-align: center; color: #444; margin-top: 20px; font-size: 0.9em;">
                    Updated: ${data.timestamp}
                </div>
            `;
            
            document.getElementById('content').innerHTML = html;
        }
        
        loadData();
    </script>
</body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(SIMPLE_HTML)


@app.route('/api/trade')
def api_trade():
    try:
        data = get_all_data()
        trade = get_simple_trade(data)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock': data,
            'trade': trade
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("📊 MCX SIMPLE TRADE DASHBOARD")
    print("="*60)
    print("✅ Just BUY CALL or BUY PUT")
    print("✅ Works for 2-3% swings")
    print("✅ No complex strategies")
    print(f"\n🌐 Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, port=5000)
