"""
MCX Advanced Price Predictor
============================
Multi-Source Ensemble Prediction System

Sources:
1. COMMODITY CORRELATION MODEL
   - Bullion (Gold, Silver)
   - Energy (Crude Oil, Natural Gas)
   - Base Metals (Copper, Aluminum, Zinc)
   - Weighted composite correlation with MCX Ltd

2. EQUITY RESEARCH TARGETS
   - Scrape from top firms (Screener, MoneyControl, TradingView)
   
3. TECHNICAL MODELS
   - Momentum, Mean Reversion, AR(1), Pattern, VWAP, RSI

4. AI PREDICTIONS
   - Context for GPT/Claude integration

5. ENSEMBLE
   - Permutation & combination of all sources
   - Weighted by historical accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMMODITY DATA FETCHER
# =============================================================================

COMMODITY_TICKERS = {
    # Bullion
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    
    # Energy
    'CRUDE_OIL': 'CL=F',
    'NATURAL_GAS': 'NG=F',
    'BRENT': 'BZ=F',
    
    # Base Metals (LME)
    'COPPER': 'HG=F',
    'ALUMINUM': 'ALI=F',
    
    # MCX India
    'MCX_GOLD': 'GOLD.NS',  # MCX Gold futures ETF proxy
    'MCX_SILVER': 'SILVER.NS',
}

# MCX Ltd revenue exposure weights (approximate)
COMMODITY_WEIGHTS = {
    'BULLION': {
        'weight': 0.35,  # 35% of MCX volume
        'components': {'GOLD': 0.6, 'SILVER': 0.4}
    },
    'ENERGY': {
        'weight': 0.30,  # 30% of MCX volume
        'components': {'CRUDE_OIL': 0.7, 'NATURAL_GAS': 0.3}
    },
    'BASE_METALS': {
        'weight': 0.25,  # 25% of MCX volume
        'components': {'COPPER': 0.5, 'ALUMINUM': 0.5}
    },
    'AGRI': {
        'weight': 0.10,  # 10% of MCX volume (hard to track)
        'components': {}
    }
}


def fetch_commodity_data(period='6mo'):
    """Fetch all commodity prices"""
    print("Fetching commodity data...")
    data = {}
    
    for name, ticker in COMMODITY_TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                data[name] = df['Close']
                print(f"  ✓ {name}: {len(df)} days")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    return pd.DataFrame(data)


def calculate_commodity_composite(commodity_df):
    """Calculate weighted commodity composite index"""
    
    # Normalize each commodity to % change
    returns = commodity_df.pct_change().dropna()
    
    # Calculate sector composites
    composites = {}
    
    # Bullion composite
    if 'GOLD' in returns.columns and 'SILVER' in returns.columns:
        composites['BULLION'] = (
            returns['GOLD'] * 0.6 + 
            returns['SILVER'] * 0.4
        )
    
    # Energy composite
    if 'CRUDE_OIL' in returns.columns:
        energy = returns['CRUDE_OIL'] * 0.7
        if 'NATURAL_GAS' in returns.columns:
            energy += returns['NATURAL_GAS'] * 0.3
        composites['ENERGY'] = energy
    
    # Base metals composite
    if 'COPPER' in returns.columns:
        composites['BASE_METALS'] = returns['COPPER']
    
    # Overall MCX commodity composite
    composite_df = pd.DataFrame(composites)
    
    # Weighted overall
    overall = pd.Series(0, index=composite_df.index)
    for sector, config in COMMODITY_WEIGHTS.items():
        if sector in composite_df.columns:
            overall += composite_df[sector] * config['weight']
    
    composite_df['MCX_COMMODITY_INDEX'] = overall
    
    return composite_df


def calculate_mcx_correlation(commodity_composite, mcx_returns):
    """Calculate correlation between commodity composite and MCX Ltd stock"""
    
    # Convert both to date only (no time) for proper alignment
    commodity_df = commodity_composite.copy()
    mcx_df = mcx_returns.copy()
    
    # Remove timezone and convert to date
    commodity_df.index = pd.to_datetime(commodity_df.index).tz_localize(None).date
    mcx_df.index = pd.to_datetime(mcx_df.index).tz_localize(None).date
    
    # Convert back to datetime for pandas operations
    commodity_df.index = pd.to_datetime(commodity_df.index)
    mcx_df.index = pd.to_datetime(mcx_df.index)
    
    # Align dates
    common_dates = commodity_df.index.intersection(mcx_df.index)
    
    if len(common_dates) < 10:
        print(f"   WARNING: Only {len(common_dates)} common dates found")
        return {}
    
    correlations = {}
    
    for col in commodity_df.columns:
        try:
            x = commodity_df.loc[common_dates, col].dropna()
            y = mcx_df.loc[common_dates].dropna()
            
            # Get common non-NaN dates
            valid_dates = x.index.intersection(y.index)
            if len(valid_dates) < 10:
                continue
                
            corr, pvalue = stats.pearsonr(
                x.loc[valid_dates],
                y.loc[valid_dates]
            )
            correlations[col] = {
                'correlation': corr,
                'p_value': pvalue,
                'significant': pvalue < 0.05,
                'n_samples': len(valid_dates)
            }
        except Exception as e:
            print(f"   Error calculating {col} correlation: {e}")
    
    return correlations


# =============================================================================
# EQUITY RESEARCH SCRAPER
# =============================================================================

def scrape_screener_targets():
    """Scrape analyst targets from Screener.in"""
    try:
        url = 'https://www.screener.in/company/MCX/consolidated/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract metrics
        metrics = {}
        items = soup.find_all('li', class_='flex')
        for item in items:
            name = item.find('span', class_='name')
            value = item.find('span', class_='number')
            if name and value:
                metrics[name.text.strip()] = value.text.strip()
        
        return {
            'source': 'Screener.in',
            'pe_ratio': metrics.get('Stock P/E', 'N/A'),
            'book_value': metrics.get('Book Value', 'N/A'),
            'roe': metrics.get('ROE', 'N/A'),
        }
    except Exception as e:
        return {'source': 'Screener.in', 'error': str(e)}


def scrape_moneycontrol_targets():
    """Scrape targets from MoneyControl"""
    try:
        url = 'https://www.moneycontrol.com/india/stockpricequote/finance-investments/multi-commodity-exchange-of-india/MCI'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # MoneyControl often has analyst ratings
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find target price
        target_section = soup.find('div', class_='analyst_rating')
        
        return {
            'source': 'MoneyControl',
            'status': 'parsed' if target_section else 'no_data'
        }
    except Exception as e:
        return {'source': 'MoneyControl', 'error': str(e)}


def get_nse_data():
    """Get real-time data from NSE"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/',
        }
        
        session = requests.Session()
        session.get('https://www.nseindia.com', headers=headers, timeout=5)
        
        r = session.get('https://www.nseindia.com/api/quote-equity?symbol=MCX', 
                       headers=headers, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            price_info = data.get('priceInfo', {})
            return {
                'source': 'NSE',
                'last_price': price_info.get('lastPrice'),
                'prev_close': price_info.get('previousClose'),
                'change_pct': price_info.get('pChange'),
                '52w_high': price_info.get('weekHighLow', {}).get('max'),
                '52w_low': price_info.get('weekHighLow', {}).get('min'),
            }
    except Exception as e:
        return {'source': 'NSE', 'error': str(e)}


# =============================================================================
# TECHNICAL MODELS
# =============================================================================

def calculate_technical_predictions(df, current_price):
    """Calculate all technical model predictions"""
    
    predictions = {}
    
    # Returns
    returns = df['Close'].pct_change().dropna()
    
    # 1. Momentum
    avg_return = returns.tail(20).mean()
    predictions['MOMENTUM'] = current_price * (1 + avg_return)
    
    # 2. Mean Reversion
    ma20 = df['Close'].tail(20).mean()
    predictions['MEAN_REVERSION'] = current_price + 0.3 * (ma20 - current_price)
    
    # 3. AR(1) Autoregressive
    returns_arr = returns.tail(60).values
    if len(returns_arr) > 2:
        ar1_coef = np.corrcoef(returns_arr[:-1], returns_arr[1:])[0, 1]
        predictions['AR1'] = current_price * (1 + ar1_coef * returns.iloc[-1])
    
    # 4. VWAP Target
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).tail(20).sum() / df['Volume'].tail(20).sum()
    predictions['VWAP'] = vwap
    
    # 5. RSI Based
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 70:
        predictions['RSI'] = current_price * 0.98  # Overbought pullback
    elif current_rsi < 30:
        predictions['RSI'] = current_price * 1.02  # Oversold bounce
    else:
        predictions['RSI'] = current_price
    
    # 6. Bollinger Band Reversion
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    std20 = df['Close'].rolling(20).std().iloc[-1]
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20
    
    if current_price > upper_band:
        predictions['BOLLINGER'] = ma20  # Revert to mean
    elif current_price < lower_band:
        predictions['BOLLINGER'] = ma20
    else:
        predictions['BOLLINGER'] = current_price
    
    # 7. Pattern (after big moves)
    big_moves = returns[abs(returns) > 0.03].index
    next_day_returns = []
    for date in big_moves:
        try:
            idx = df.index.get_loc(date)
            if idx + 1 < len(df):
                next_day_returns.append(returns.iloc[idx + 1])
        except:
            pass
    
    if next_day_returns:
        avg_after_big = np.mean(next_day_returns)
        predictions['PATTERN'] = current_price * (1 + avg_after_big)
    else:
        predictions['PATTERN'] = current_price
    
    return predictions, current_rsi


# =============================================================================
# COMMODITY-BASED PREDICTION
# =============================================================================

def predict_from_commodities(commodity_composite, mcx_returns, correlations, current_price):
    """Use commodity movements to predict MCX"""
    
    # Get today's commodity moves
    latest_commodity = commodity_composite.iloc[-1]
    
    predictions = {}
    
    for sector, corr_data in correlations.items():
        if sector in latest_commodity.index:
            corr = corr_data['correlation']
            commodity_move = latest_commodity[sector]
            
            # Predicted MCX move = correlation * commodity move
            predicted_move = corr * commodity_move
            predictions[f'COMMODITY_{sector}'] = current_price * (1 + predicted_move)
    
    return predictions


# =============================================================================
# AI CONTEXT GENERATOR
# =============================================================================

def generate_ai_context(current_price, technical_preds, commodity_preds, correlations, rsi):
    """Generate context for AI models (GPT/Claude)"""
    
    context = f"""
STOCK ANALYSIS REQUEST: MCX Ltd (Multi Commodity Exchange of India)
Current Price: ₹{current_price:.2f}
Date: {datetime.now().strftime('%Y-%m-%d')}

TASK: Predict tomorrow's closing price for MCX Ltd stock.

TECHNICAL INDICATORS:
- RSI(14): {rsi:.1f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}
- Technical Model Predictions:
{chr(10).join([f'  - {k}: ₹{v:.0f}' for k, v in technical_preds.items()])}

COMMODITY CORRELATIONS (MCX's business depends on commodity trading volumes):
{chr(10).join([f'  - {k}: {v["correlation"]:.2%} correlation' for k, v in correlations.items()])}

COMMODITY-BASED PREDICTIONS:
{chr(10).join([f'  - {k}: ₹{v:.0f}' for k, v in commodity_preds.items()])}

BUSINESS CONTEXT:
- MCX is India's largest commodity exchange
- Revenue depends on trading volumes of Gold, Silver, Crude Oil, Natural Gas, Base Metals
- Stock price is influenced by:
  1. Commodity price volatility (higher volatility = more trading = more revenue)
  2. Regulatory changes (SEBI)
  3. Market share vs competitors (NCDEX)
  4. Technology platform updates

Based on all the above data, provide:
1. Your predicted closing price for tomorrow
2. Confidence level (Low/Medium/High)
3. Key factors driving your prediction
4. Risk factors that could invalidate the prediction

IMPORTANT: Respond with a specific price target, not a range.
"""
    return context


# =============================================================================
# ENSEMBLE PREDICTOR
# =============================================================================

def create_ensemble(all_predictions, weights=None):
    """Create weighted ensemble from all predictions"""
    
    if weights is None:
        # Equal weights
        weights = {k: 1.0 / len(all_predictions) for k in all_predictions}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted average
    ensemble_price = sum(
        all_predictions[k] * weights.get(k, 0) 
        for k in all_predictions if k in weights
    )
    
    return ensemble_price, weights


def calculate_permutation_ensembles(all_predictions):
    """Calculate all possible combinations of models"""
    
    models = list(all_predictions.keys())
    results = []
    
    # Try all combinations of 2 to N models
    for r in range(2, len(models) + 1):
        for combo in combinations(models, r):
            subset = {k: all_predictions[k] for k in combo}
            avg_price = np.mean(list(subset.values()))
            std_price = np.std(list(subset.values()))
            
            results.append({
                'models': combo,
                'n_models': len(combo),
                'prediction': avg_price,
                'std': std_price,
                'confidence': 1 / (1 + std_price / avg_price)  # Lower variance = higher confidence
            })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results


# =============================================================================
# BACKTESTER
# =============================================================================

def backtest_models(df, lookback=30):
    """Backtest all models on historical data"""
    
    results = {
        'MOMENTUM': [],
        'MEAN_REVERSION': [],
        'AR1': [],
        'VWAP': [],
        'RSI': [],
        'PATTERN': [],
    }
    
    print(f"\nBacktesting {lookback} days...")
    
    for i in range(lookback, 0, -1):
        # Data available at prediction time
        train_df = df.iloc[:-i]
        if len(train_df) < 60:
            continue
            
        current_price = train_df['Close'].iloc[-1]
        actual_next = df['Close'].iloc[-i+1] if i > 1 else df['Close'].iloc[-1]
        
        # Get predictions
        preds, _ = calculate_technical_predictions(train_df, current_price)
        
        # Calculate errors
        for model, pred in preds.items():
            if model in results:
                error = (pred - actual_next) / actual_next
                direction_correct = (pred > current_price) == (actual_next > current_price)
                results[model].append({
                    'predicted': pred,
                    'actual': actual_next,
                    'error_pct': error * 100,
                    'direction_correct': direction_correct
                })
    
    # Summarize
    summary = {}
    for model, data in results.items():
        if data:
            errors = [d['error_pct'] for d in data]
            directions = [d['direction_correct'] for d in data]
            summary[model] = {
                'mae': np.mean(np.abs(errors)),
                'direction_accuracy': np.mean(directions) * 100,
                'n_samples': len(data)
            }
    
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_advanced_prediction():
    """Run the full advanced prediction pipeline"""
    
    print("="*70)
    print("MCX ADVANCED MULTI-SOURCE PREDICTION SYSTEM")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # 1. Fetch MCX stock data
    print("1. FETCHING MCX STOCK DATA")
    print("-"*70)
    mcx = yf.Ticker('MCX.NS')
    mcx_df = mcx.history(period='1y')
    
    if mcx_df.empty:
        print("   ERROR: Could not fetch MCX data from Yahoo Finance")
        # Try NSE
        nse_data = get_nse_data()
        if 'error' not in nse_data:
            current_price = nse_data['last_price']
            print(f"   Using NSE data: ₹{current_price}")
        else:
            print("   ERROR: Could not fetch from NSE either")
            return
    else:
        current_price = mcx_df['Close'].iloc[-1]
        print(f"   Current Price: ₹{current_price:.2f}")
        print(f"   Data Points: {len(mcx_df)}")
    
    mcx_returns = mcx_df['Close'].pct_change().dropna()
    
    # 2. Fetch commodity data
    print()
    print("2. FETCHING COMMODITY DATA")
    print("-"*70)
    commodity_df = fetch_commodity_data(period='6mo')
    
    if commodity_df.empty:
        print("   WARNING: Limited commodity data available")
    
    # 3. Calculate commodity composite
    print()
    print("3. CALCULATING COMMODITY COMPOSITE")
    print("-"*70)
    if not commodity_df.empty:
        commodity_composite = calculate_commodity_composite(commodity_df)
        print(f"   Sectors: {list(commodity_composite.columns)}")
        
        # Calculate correlations
        correlations = calculate_mcx_correlation(commodity_composite, mcx_returns)
        print()
        print("   CORRELATIONS WITH MCX Ltd:")
        for sector, data in correlations.items():
            sig = "***" if data['significant'] else ""
            print(f"   - {sector}: {data['correlation']:.2%} {sig}")
    else:
        commodity_composite = pd.DataFrame()
        correlations = {}
    
    # 4. Technical predictions
    print()
    print("4. TECHNICAL MODEL PREDICTIONS")
    print("-"*70)
    technical_preds, rsi = calculate_technical_predictions(mcx_df, current_price)
    for model, pred in technical_preds.items():
        change = (pred / current_price - 1) * 100
        print(f"   {model:20}: ₹{pred:,.0f} ({change:+.1f}%)")
    
    # 5. Commodity-based predictions
    print()
    print("5. COMMODITY-BASED PREDICTIONS")
    print("-"*70)
    if not commodity_composite.empty and correlations:
        commodity_preds = predict_from_commodities(
            commodity_composite, mcx_returns, correlations, current_price
        )
        for model, pred in commodity_preds.items():
            change = (pred / current_price - 1) * 100
            print(f"   {model:20}: ₹{pred:,.0f} ({change:+.1f}%)")
    else:
        commodity_preds = {}
        print("   No commodity predictions available")
    
    # 6. Equity research data
    print()
    print("6. EQUITY RESEARCH DATA")
    print("-"*70)
    screener_data = scrape_screener_targets()
    print(f"   Screener.in: P/E = {screener_data.get('pe_ratio', 'N/A')}")
    
    nse_data = get_nse_data()
    if 'error' not in nse_data:
        print(f"   NSE: 52W High = ₹{nse_data.get('52w_high')}, Low = ₹{nse_data.get('52w_low')}")
    
    # 7. Combine all predictions
    print()
    print("7. ENSEMBLE PREDICTION")
    print("-"*70)
    
    # Filter out NaN predictions
    all_predictions = {}
    for k, v in technical_preds.items():
        if not np.isnan(v):
            all_predictions[k] = v
    for k, v in commodity_preds.items():
        if not np.isnan(v):
            all_predictions[k] = v
    
    if not all_predictions:
        print("   ERROR: No valid predictions")
        return
    
    # Simple average ensemble
    simple_ensemble = np.mean(list(all_predictions.values()))
    print(f"   Simple Average: ₹{simple_ensemble:,.0f}")
    
    # Best combinations
    print()
    print("   TOP PERMUTATION COMBINATIONS:")
    combos = calculate_permutation_ensembles(all_predictions)
    for i, combo in enumerate(combos[:5]):
        print(f"   {i+1}. {combo['models']}")
        print(f"      Prediction: ₹{combo['prediction']:,.0f} | Confidence: {combo['confidence']:.2%}")
    
    # 8. Backtest
    print()
    print("8. BACKTEST RESULTS (Last 30 days)")
    print("-"*70)
    backtest_results = backtest_models(mcx_df, lookback=30)
    print()
    print(f"   {'Model':<20} {'MAE %':<12} {'Direction Acc':<15}")
    print("   " + "-"*47)
    for model, data in sorted(backtest_results.items(), key=lambda x: x[1]['mae']):
        print(f"   {model:<20} {data['mae']:.2f}%        {data['direction_accuracy']:.0f}%")
    
    # 9. Generate AI context
    print()
    print("9. AI CONTEXT (for GPT/Claude)")
    print("-"*70)
    ai_context = generate_ai_context(
        current_price, technical_preds, commodity_preds, correlations, rsi
    )
    print("   Context generated. Copy the following to ChatGPT or Claude:")
    print()
    
    # 10. Final prediction
    print()
    print("="*70)
    print("FINAL PREDICTION SUMMARY")
    print("="*70)
    
    # Weight models by backtest accuracy
    weighted_sum = 0
    total_weight = 0
    for model, pred in technical_preds.items():
        if model in backtest_results:
            accuracy = backtest_results[model]['direction_accuracy']
            weight = accuracy / 100
            weighted_sum += pred * weight
            total_weight += weight
    
    weighted_prediction = weighted_sum / total_weight if total_weight > 0 else simple_ensemble
    
    print(f"\n   Current Price:         ₹{current_price:,.2f}")
    print(f"   Simple Ensemble:       ₹{simple_ensemble:,.0f} ({(simple_ensemble/current_price-1)*100:+.2f}%)")
    print(f"   Accuracy-Weighted:     ₹{weighted_prediction:,.0f} ({(weighted_prediction/current_price-1)*100:+.2f}%)")
    print(f"   Best Combo:            ₹{combos[0]['prediction']:,.0f} ({(combos[0]['prediction']/current_price-1)*100:+.2f}%)")
    
    # Direction consensus
    bullish = sum(1 for p in all_predictions.values() if p > current_price)
    bearish = len(all_predictions) - bullish
    
    print()
    print(f"   Direction Consensus:   {bullish} BULLISH / {bearish} BEARISH")
    print(f"   Bias:                  {'BULLISH' if bullish > bearish else 'BEARISH' if bearish > bullish else 'NEUTRAL'}")
    
    # Save AI context to file
    with open('ai_prediction_context.txt', 'w', encoding='utf-8') as f:
        f.write(ai_context)
    print()
    print("   AI context saved to: ai_prediction_context.txt")
    
    return {
        'current_price': current_price,
        'simple_ensemble': simple_ensemble,
        'weighted_prediction': weighted_prediction,
        'best_combo': combos[0],
        'all_predictions': all_predictions,
        'backtest': backtest_results,
        'ai_context': ai_context
    }


if __name__ == '__main__':
    results = run_advanced_prediction()
