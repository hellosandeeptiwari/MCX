"""
MCX Professional Analysis Dashboard v3.0
=========================================
Based on institutional-grade factor investing methodology:

QUANTITATIVE FACTORS (from Fama-French, Factor Investing):
1. Value: P/E, P/B, EV/EBITDA
2. Momentum: Price momentum (3m, 6m, 12m)
3. Quality: ROE, Debt/Equity, Earnings stability
4. Size: Market Cap relative to sector
5. Volatility: Historical vol, Beta

FUNDAMENTAL ANALYSIS:
- Revenue growth
- Earnings growth (EPS)
- Free Cash Flow
- Profit margins

TECHNICAL ANALYSIS:
- Moving averages (50/200 MA)
- RSI, MACD
- Support/Resistance
- Volume analysis

This is what hedge funds and research firms ACTUALLY use.
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

MCX_LOT_SIZE = 625
MCX_TICKER = 'MCX.NS'

# ============================================
# QUANTITATIVE FACTOR ANALYSIS
# ============================================

def get_mcx_fundamentals():
    """Get MCX fundamental data - what institutional investors analyze"""
    try:
        mcx = yf.Ticker(MCX_TICKER)
        info = mcx.info
        
        # Valuation Ratios
        pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
        pb_ratio = info.get('priceToBook', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        peg_ratio = info.get('pegRatio', None)
        
        # Enterprise Value metrics
        ev = info.get('enterpriseValue', None)
        ebitda = info.get('ebitda', None)
        ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else None
        
        # Profitability (Quality Factor)
        roe = info.get('returnOnEquity', None)
        roa = info.get('returnOnAssets', None)
        profit_margin = info.get('profitMargins', None)
        operating_margin = info.get('operatingMargins', None)
        
        # Growth
        revenue_growth = info.get('revenueGrowth', None)
        earnings_growth = info.get('earningsGrowth', None)
        
        # Balance Sheet (Quality Factor)
        debt_to_equity = info.get('debtToEquity', None)
        current_ratio = info.get('currentRatio', None)
        
        # Cash Flow
        free_cash_flow = info.get('freeCashflow', None)
        operating_cash_flow = info.get('operatingCashflow', None)
        
        # Size Factor
        market_cap = info.get('marketCap', None)
        
        # Dividend
        dividend_yield = info.get('dividendYield', None)
        
        # Current Price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        
        # 52-week range
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', None)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', None)
        
        # Beta (Volatility Factor)
        beta = info.get('beta', None)
        
        # EPS
        trailing_eps = info.get('trailingEps', None)
        forward_eps = info.get('forwardEps', None)
        
        return {
            'valuation': {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'peg_ratio': peg_ratio,
                'ev_ebitda': ev_ebitda
            },
            'profitability': {
                'roe': roe * 100 if roe else None,
                'roa': roa * 100 if roa else None,
                'profit_margin': profit_margin * 100 if profit_margin else None,
                'operating_margin': operating_margin * 100 if operating_margin else None
            },
            'growth': {
                'revenue_growth': revenue_growth * 100 if revenue_growth else None,
                'earnings_growth': earnings_growth * 100 if earnings_growth else None
            },
            'financial_health': {
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'free_cash_flow': free_cash_flow,
                'operating_cash_flow': operating_cash_flow
            },
            'market_data': {
                'current_price': current_price,
                'market_cap': market_cap,
                'beta': beta,
                'dividend_yield': dividend_yield * 100 if dividend_yield else None,
                'fifty_two_week_high': fifty_two_week_high,
                'fifty_two_week_low': fifty_two_week_low
            },
            'eps': {
                'trailing': trailing_eps,
                'forward': forward_eps
            }
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_momentum_factors():
    """Calculate price momentum - key factor in factor investing"""
    try:
        mcx = yf.download(MCX_TICKER, period='1y', progress=False)
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        current = float(close.iloc[-1])
        
        # Momentum over different periods
        mom_1m = (current / float(close.iloc[-22]) - 1) * 100 if len(close) > 22 else None
        mom_3m = (current / float(close.iloc[-66]) - 1) * 100 if len(close) > 66 else None
        mom_6m = (current / float(close.iloc[-132]) - 1) * 100 if len(close) > 132 else None
        mom_12m = (current / float(close.iloc[0]) - 1) * 100
        
        # Relative Strength vs Nifty
        nifty = yf.download('^NSEI', period='1y', progress=False)
        if not nifty.empty:
            nifty_close = nifty['Close'].squeeze()
            nifty_ret = (float(nifty_close.iloc[-1]) / float(nifty_close.iloc[0]) - 1) * 100
            relative_strength = mom_12m - nifty_ret if mom_12m else None
        else:
            relative_strength = None
        
        # Moving Average signals
        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        
        ma_signal = None
        if sma_50 and sma_200:
            if current > sma_50 > sma_200:
                ma_signal = 'STRONG_BULLISH'  # Golden cross territory
            elif current > sma_50 and sma_50 < sma_200:
                ma_signal = 'RECOVERING'
            elif current < sma_50 < sma_200:
                ma_signal = 'STRONG_BEARISH'  # Death cross territory
            elif current < sma_50 and sma_50 > sma_200:
                ma_signal = 'WEAKENING'
            else:
                ma_signal = 'NEUTRAL'
        
        return {
            'momentum_1m': mom_1m,
            'momentum_3m': mom_3m,
            'momentum_6m': mom_6m,
            'momentum_12m': mom_12m,
            'relative_strength_vs_nifty': relative_strength,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'current_price': current,
            'ma_signal': ma_signal,
            'above_50ma': current > sma_50 if sma_50 else None,
            'above_200ma': current > sma_200 if sma_200 else None
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_volatility_factor():
    """Calculate volatility metrics - important for options trading"""
    try:
        mcx = yf.download(MCX_TICKER, period='1y', progress=False)
        if mcx.empty:
            return None
        
        close = mcx['Close'].squeeze()
        returns = close.pct_change().dropna()
        
        # Historical Volatility (annualized)
        vol_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)
        vol_60d = float(returns.tail(60).std() * np.sqrt(252) * 100)
        vol_252d = float(returns.std() * np.sqrt(252) * 100)
        
        # Volatility trend
        vol_trend = 'INCREASING' if vol_20d > vol_60d else 'DECREASING'
        
        # ATR (Average True Range)
        high = mcx['High'].squeeze()
        low = mcx['Low'].squeeze()
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr_14 / float(close.iloc[-1]) * 100
        
        # Max Drawdown (last 1 year)
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max * 100
        max_drawdown = float(drawdown.min())
        
        return {
            'volatility_20d': vol_20d,
            'volatility_60d': vol_60d,
            'volatility_1y': vol_252d,
            'volatility_trend': vol_trend,
            'atr_14': atr_14,
            'atr_percent': atr_pct,
            'max_drawdown_1y': max_drawdown,
            'vol_regime': 'HIGH' if vol_20d > 35 else ('LOW' if vol_20d < 20 else 'NORMAL')
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_technical_indicators():
    """Standard technical indicators used by traders"""
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
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = float((sma20 + 2 * std20).iloc[-1])
        bb_lower = float((sma20 - 2 * std20).iloc[-1])
        bb_middle = float(sma20.iloc[-1])
        bb_width = (bb_upper - bb_lower) / bb_middle * 100
        
        # Support and Resistance (from pivot points)
        pivot = (float(high.iloc[-1]) + float(low.iloc[-1]) + current) / 3
        r1 = 2 * pivot - float(low.iloc[-1])
        s1 = 2 * pivot - float(high.iloc[-1])
        r2 = pivot + (float(high.iloc[-1]) - float(low.iloc[-1]))
        s2 = pivot - (float(high.iloc[-1]) - float(low.iloc[-1]))
        
        # Volume analysis
        vol_avg_20 = float(volume.rolling(20).mean().iloc[-1])
        vol_current = float(volume.iloc[-1])
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 1
        
        # 52-week high/low
        high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
        low_52w = float(low.tail(252).min()) if len(low) >= 252 else float(low.min())
        pct_from_high = (current - high_52w) / high_52w * 100
        pct_from_low = (current - low_52w) / low_52w * 100
        
        return {
            'rsi': {
                'value': rsi_value,
                'signal': 'OVERSOLD' if rsi_value < 30 else ('OVERBOUGHT' if rsi_value > 70 else 'NEUTRAL')
            },
            'macd': {
                'line': float(macd_line.iloc[-1]),
                'signal': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1]),
                'trend': 'BULLISH' if float(histogram.iloc[-1]) > 0 else 'BEARISH'
            },
            'bollinger': {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower,
                'width': bb_width,
                'position': 'UPPER' if current > bb_upper else ('LOWER' if current < bb_lower else 'MIDDLE')
            },
            'pivot_points': {
                'pivot': pivot,
                'r1': r1, 'r2': r2,
                's1': s1, 's2': s2
            },
            'volume': {
                'current': vol_current,
                'avg_20': vol_avg_20,
                'ratio': vol_ratio,
                'signal': 'HIGH' if vol_ratio > 1.5 else ('LOW' if vol_ratio < 0.5 else 'NORMAL')
            },
            '52_week': {
                'high': high_52w,
                'low': low_52w,
                'pct_from_high': pct_from_high,
                'pct_from_low': pct_from_low
            },
            'current_price': current
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================
# COMPOSITE SCORING SYSTEM
# ============================================

def calculate_factor_scores():
    """
    Calculate composite factor scores like professional quant funds do.
    Each factor gets a score from -2 to +2.
    """
    fundamentals = get_mcx_fundamentals()
    momentum = calculate_momentum_factors()
    volatility = calculate_volatility_factor()
    technicals = calculate_technical_indicators()
    
    scores = {
        'value': 0,
        'quality': 0,
        'momentum': 0,
        'technical': 0,
        'total': 0
    }
    
    reasons = []
    
    # VALUE SCORE (-2 to +2)
    if fundamentals and 'valuation' in fundamentals:
        val = fundamentals['valuation']
        value_score = 0
        
        # P/E Analysis (compare to typical exchange P/E of 25-30)
        if val.get('pe_ratio'):
            pe = val['pe_ratio']
            if pe < 20:
                value_score += 2
                reasons.append(f"P/E {pe:.1f} is attractive (below 20)")
            elif pe < 25:
                value_score += 1
                reasons.append(f"P/E {pe:.1f} is reasonable")
            elif pe > 35:
                value_score -= 1
                reasons.append(f"P/E {pe:.1f} is expensive")
            elif pe > 45:
                value_score -= 2
                reasons.append(f"P/E {pe:.1f} is very expensive")
        
        # P/B Analysis
        if val.get('pb_ratio'):
            pb = val['pb_ratio']
            if pb < 3:
                value_score += 1
                reasons.append(f"P/B {pb:.1f} is reasonable")
            elif pb > 5:
                value_score -= 1
                reasons.append(f"P/B {pb:.1f} is high")
        
        scores['value'] = max(-2, min(2, value_score))
    
    # QUALITY SCORE (-2 to +2)
    if fundamentals and 'profitability' in fundamentals:
        prof = fundamentals['profitability']
        fh = fundamentals.get('financial_health', {})
        quality_score = 0
        
        # ROE (>15% is good for exchanges)
        if prof.get('roe'):
            roe = prof['roe']
            if roe > 20:
                quality_score += 2
                reasons.append(f"ROE {roe:.1f}% is excellent")
            elif roe > 15:
                quality_score += 1
                reasons.append(f"ROE {roe:.1f}% is good")
            elif roe < 10:
                quality_score -= 1
                reasons.append(f"ROE {roe:.1f}% is weak")
        
        # Debt/Equity (lower is better)
        if fh.get('debt_to_equity'):
            de = fh['debt_to_equity']
            if de < 50:
                quality_score += 1
                reasons.append(f"Low debt (D/E: {de:.0f}%)")
            elif de > 100:
                quality_score -= 1
                reasons.append(f"High debt (D/E: {de:.0f}%)")
        
        # Profit Margin
        if prof.get('profit_margin'):
            pm = prof['profit_margin']
            if pm > 30:
                quality_score += 1
                reasons.append(f"High profit margin ({pm:.1f}%)")
        
        scores['quality'] = max(-2, min(2, quality_score))
    
    # MOMENTUM SCORE (-2 to +2)
    if momentum and 'momentum_3m' in momentum:
        mom_score = 0
        
        # 3-month momentum
        if momentum.get('momentum_3m'):
            m3 = momentum['momentum_3m']
            if m3 > 15:
                mom_score += 2
                reasons.append(f"Strong 3M momentum (+{m3:.1f}%)")
            elif m3 > 5:
                mom_score += 1
                reasons.append(f"Positive 3M momentum (+{m3:.1f}%)")
            elif m3 < -15:
                mom_score -= 2
                reasons.append(f"Weak 3M momentum ({m3:.1f}%)")
            elif m3 < -5:
                mom_score -= 1
                reasons.append(f"Negative 3M momentum ({m3:.1f}%)")
        
        # MA Signal
        if momentum.get('ma_signal'):
            ma = momentum['ma_signal']
            if ma == 'STRONG_BULLISH':
                mom_score += 1
                reasons.append("Price above 50MA and 200MA (bullish trend)")
            elif ma == 'STRONG_BEARISH':
                mom_score -= 1
                reasons.append("Price below 50MA and 200MA (bearish trend)")
        
        scores['momentum'] = max(-2, min(2, mom_score))
    
    # TECHNICAL SCORE (-2 to +2)
    if technicals:
        tech_score = 0
        
        # RSI
        if technicals.get('rsi'):
            rsi = technicals['rsi']
            if rsi['signal'] == 'OVERSOLD':
                tech_score += 1
                reasons.append(f"RSI oversold ({rsi['value']:.0f}) - potential bounce")
            elif rsi['signal'] == 'OVERBOUGHT':
                tech_score -= 1
                reasons.append(f"RSI overbought ({rsi['value']:.0f}) - potential pullback")
        
        # MACD
        if technicals.get('macd'):
            macd = technicals['macd']
            if macd['trend'] == 'BULLISH' and macd['histogram'] > 0:
                tech_score += 1
                reasons.append("MACD bullish crossover")
            elif macd['trend'] == 'BEARISH' and macd['histogram'] < 0:
                tech_score -= 1
                reasons.append("MACD bearish")
        
        scores['technical'] = max(-2, min(2, tech_score))
    
    # TOTAL SCORE
    scores['total'] = scores['value'] + scores['quality'] + scores['momentum'] + scores['technical']
    
    # Overall signal
    total = scores['total']
    if total >= 4:
        signal = 'STRONG BUY'
        strategy = 'Buy Calls / Sell Puts'
    elif total >= 2:
        signal = 'BUY'
        strategy = 'Bull Call Spread'
    elif total <= -4:
        signal = 'STRONG SELL'
        strategy = 'Buy Puts / Sell Calls'
    elif total <= -2:
        signal = 'SELL'
        strategy = 'Bear Put Spread'
    else:
        signal = 'HOLD'
        strategy = 'Iron Condor / Sell Premium'
    
    return {
        'scores': scores,
        'signal': signal,
        'strategy': strategy,
        'reasons': reasons,
        'max_possible': 8,
        'min_possible': -8
    }

# ============================================
# OPTION PRICING (Black-Scholes)
# ============================================

def calculate_option_greeks(spot, strike, days_to_expiry, volatility=None, risk_free_rate=0.07):
    """Black-Scholes option pricing"""
    from scipy.stats import norm
    
    # Use actual volatility if available
    if volatility is None:
        vol_data = calculate_volatility_factor()
        volatility = vol_data['volatility_20d'] / 100 if vol_data else 0.35
    else:
        volatility = volatility / 100
    
    if days_to_expiry <= 0:
        days_to_expiry = 1
    
    T = days_to_expiry / 365
    S = spot
    K = strike
    r = risk_free_rate
    sigma = volatility
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    call_delta = norm.cdf(d1)
    put_delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    call_theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    put_theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    
    return {
        'call_price': round(call_price, 2),
        'put_price': round(put_price, 2),
        'call_delta': round(call_delta, 3),
        'put_delta': round(put_delta, 3),
        'gamma': round(gamma, 5),
        'vega': round(vega, 3),
        'call_theta': round(call_theta, 3),
        'put_theta': round(put_theta, 3)
    }

def get_option_chain(spot_price, days_to_expiry=30):
    """Generate option chain with actual volatility"""
    vol_data = calculate_volatility_factor()
    volatility = vol_data['volatility_20d'] if vol_data else 35
    
    strikes = list(range(int(spot_price * 0.85), int(spot_price * 1.15), 50))
    chain = []
    
    for strike in strikes:
        greeks = calculate_option_greeks(spot_price, strike, days_to_expiry, volatility)
        is_atm = abs(strike - spot_price) < 25
        
        chain.append({
            'strike': strike,
            'is_atm': bool(is_atm),
            **greeks
        })
    
    return chain, volatility

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def dashboard():
    return render_template('dashboard_v3.html')

@app.route('/api/fundamentals')
def api_fundamentals():
    try:
        data = get_mcx_fundamentals()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/momentum')
def api_momentum():
    try:
        data = calculate_momentum_factors()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/volatility')
def api_volatility():
    try:
        data = calculate_volatility_factor()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/technicals')
def api_technicals():
    try:
        data = calculate_technical_indicators()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/factor-scores')
def api_factor_scores():
    try:
        data = calculate_factor_scores()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/option-chain')
def api_option_chain():
    try:
        days = int(request.args.get('days', 30))
        technicals = calculate_technical_indicators()
        spot = technicals['current_price'] if technicals else 2300
        chain, vol = get_option_chain(spot, days)
        return jsonify({
            'success': True, 
            'data': chain, 
            'spot': spot,
            'implied_vol': vol
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/all')
def api_all():
    """Get all data in one call"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'fundamentals': get_mcx_fundamentals(),
                'momentum': calculate_momentum_factors(),
                'volatility': calculate_volatility_factor(),
                'technicals': calculate_technical_indicators(),
                'factor_scores': calculate_factor_scores()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 70)
    print("🏦 MCX Professional Analysis Dashboard v3.0")
    print("=" * 70)
    print()
    print("📊 Institutional-Grade Analysis Methods:")
    print("   • Factor Investing (Fama-French style)")
    print("   • Value: P/E, P/B, EV/EBITDA")
    print("   • Quality: ROE, Profit Margins, Debt/Equity")
    print("   • Momentum: 3M/6M/12M returns, Moving Averages")
    print("   • Technical: RSI, MACD, Bollinger Bands, Volume")
    print()
    print("🌐 Dashboard: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)
