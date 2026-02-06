"""
MCX Volatility & Volume Predictor with ChatGPT Integration
============================================================
Instead of predicting DIRECTION (which is ~50%), we predict:
1. VOLATILITY - Will tomorrow be a big move day?
2. VOLUME - Will there be unusual activity?
3. ChatGPT Analysis - LLM sentiment with weightage

This is MORE PREDICTABLE than direction!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# OpenAI import - will handle if not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Run: pip install openai")


class VolatilityPredictor:
    def __init__(self, openai_api_key=None):
        """Initialize with optional OpenAI API key"""
        self.api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            print("✅ ChatGPT API initialized")
        else:
            print("⚠️ ChatGPT API not available - using technical analysis only")
    
    def fetch_data(self, period='1y'):
        """Fetch MCX data"""
        mcx = yf.Ticker('MCX.NS')
        df = mcx.history(period=period)
        return df
    
    def calculate_volatility_features(self, df):
        """Calculate volatility and volume features"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        ret = close.pct_change()
        
        features = {}
        
        # Current values
        features['price'] = float(close.iloc[-1])
        features['today_return'] = float(ret.iloc[-1] * 100)
        features['today_range'] = float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] * 100)
        
        # Volatility metrics
        features['volatility_20d'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        features['volatility_5d'] = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
        
        # ATR (Average True Range)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        features['atr'] = float(atr_14.iloc[-1])
        features['atr_percent'] = float(atr_14.iloc[-1] / close.iloc[-1] * 100)
        
        # ATR expansion/contraction
        features['atr_vs_avg'] = float(atr_14.iloc[-1] / atr_14.rolling(50).mean().iloc[-1])
        
        # Bollinger Band Width (volatility indicator)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = (4 * std20) / ma20 * 100
        features['bb_width'] = float(bb_width.iloc[-1])
        features['bb_width_percentile'] = float((bb_width.iloc[-1] > bb_width.rolling(100).quantile(0.5).iloc[-1]) * 100)
        
        # Volume analysis
        features['volume'] = float(volume.iloc[-1])
        features['volume_ma20'] = float(volume.rolling(20).mean().iloc[-1])
        features['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
        features['volume_trend'] = float(volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1])
        
        # Historical volatility pattern
        abs_returns = ret.abs() * 100
        features['avg_daily_move'] = float(abs_returns.rolling(20).mean().iloc[-1])
        features['max_daily_move_20d'] = float(abs_returns.rolling(20).max().iloc[-1])
        
        # Volatility regime
        if features['volatility_5d'] > features['volatility_20d'] * 1.2:
            features['vol_regime'] = 'EXPANDING'
        elif features['volatility_5d'] < features['volatility_20d'] * 0.8:
            features['vol_regime'] = 'CONTRACTING'
        else:
            features['vol_regime'] = 'STABLE'
        
        # RSI for context
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features['rsi'] = float(rsi.iloc[-1])
        
        # Recent price history for ChatGPT
        features['last_5_returns'] = [round(r * 100, 2) for r in ret.iloc[-5:].tolist()]
        features['last_5_volumes'] = [int(v) for v in volume.iloc[-5:].tolist()]
        
        return features
    
    def predict_volatility(self, features):
        """Predict tomorrow's volatility regime"""
        score = 50  # Base score
        reasons = []
        
        # ATR expansion suggests continued volatility
        if features['atr_vs_avg'] > 1.2:
            score += 15
            reasons.append(f"ATR {features['atr_vs_avg']:.2f}x above average - high volatility regime")
        elif features['atr_vs_avg'] < 0.8:
            score -= 10
            reasons.append(f"ATR {features['atr_vs_avg']:.2f}x below average - low volatility")
        
        # Bollinger squeeze often precedes big moves
        if features['bb_width'] < 8:  # Tight bands
            score += 20
            reasons.append(f"Bollinger squeeze (width {features['bb_width']:.1f}%) - big move likely")
        
        # Volume surge often precedes volatility
        if features['volume_ratio'] > 1.5:
            score += 15
            reasons.append(f"Volume {features['volume_ratio']:.1f}x average - institutional activity")
        
        # Recent big moves tend to cluster
        if features['today_range'] > features['avg_daily_move'] * 1.5:
            score += 10
            reasons.append(f"Today's range {features['today_range']:.2f}% above average")
        
        # Extreme RSI often leads to volatile reversals
        if features['rsi'] < 25 or features['rsi'] > 75:
            score += 10
            reasons.append(f"RSI at extreme ({features['rsi']:.1f}) - reversal volatility possible")
        
        # Volatility regime
        if features['vol_regime'] == 'EXPANDING':
            score += 10
            reasons.append("Short-term volatility expanding")
        elif features['vol_regime'] == 'CONTRACTING':
            score -= 5
            reasons.append("Volatility contracting - calm period")
        
        # Cap score
        score = max(20, min(90, score))
        
        return {
            'volatility_score': score,
            'expected_range': features['atr_percent'] * (score / 50),
            'vol_regime': features['vol_regime'],
            'reasons': reasons
        }
    
    def get_chatgpt_analysis(self, features):
        """Get ChatGPT's analysis of the market situation"""
        if not self.client:
            return {
                'available': False,
                'direction': 'NEUTRAL',
                'confidence': 50,
                'volatility_view': 'NEUTRAL',
                'reasoning': 'ChatGPT API not configured'
            }
        
        prompt = f"""You are a quantitative analyst. Analyze MCX Ltd (Multi Commodity Exchange of India) stock based on these technical indicators:

CURRENT DATA:
- Price: ₹{features['price']:.2f}
- Today's Return: {features['today_return']:.2f}%
- Today's Range: {features['today_range']:.2f}%

VOLATILITY METRICS:
- 20-day Volatility: {features['volatility_20d']:.1f}% annualized
- 5-day Volatility: {features['volatility_5d']:.1f}% annualized
- ATR: ₹{features['atr']:.2f} ({features['atr_percent']:.2f}%)
- ATR vs 50-day avg: {features['atr_vs_avg']:.2f}x
- Bollinger Width: {features['bb_width']:.2f}%
- Volatility Regime: {features['vol_regime']}

VOLUME:
- Today vs 20-day avg: {features['volume_ratio']:.2f}x
- 5-day vs 20-day avg: {features['volume_trend']:.2f}x

MOMENTUM:
- RSI (14): {features['rsi']:.1f}
- Last 5 days returns: {features['last_5_returns']}

IMPORTANT: Direction prediction is ~50% accurate historically. Focus on VOLATILITY and RISK.

Provide your analysis in EXACTLY this JSON format:
{{
    "direction": "BULLISH" or "BEARISH" or "NEUTRAL",
    "direction_confidence": 50-70 (be realistic, even 55% is good),
    "volatility_forecast": "HIGH" or "MEDIUM" or "LOW",
    "volatility_confidence": 50-90,
    "expected_range_percent": 1.0-5.0,
    "key_levels": {{"support": price, "resistance": price}},
    "risk_warning": "brief risk statement",
    "reasoning": "2-3 sentence analysis"
}}

Respond ONLY with the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a quantitative analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean up if needed
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            analysis = json.loads(content)
            analysis['available'] = True
            return analysis
            
        except Exception as e:
            return {
                'available': False,
                'direction': 'NEUTRAL',
                'direction_confidence': 50,
                'volatility_forecast': 'MEDIUM',
                'volatility_confidence': 50,
                'reasoning': f'API Error: {str(e)}'
            }
    
    def combine_signals(self, tech_analysis, chatgpt_analysis):
        """Combine technical and ChatGPT analysis with weightage"""
        
        # Weightage (technical gets more for volatility, ChatGPT adds color)
        TECH_WEIGHT = 0.7
        GPT_WEIGHT = 0.3
        
        combined = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Volatility Score (technical + GPT)
        tech_vol_score = tech_analysis['volatility_score']
        
        gpt_vol_score = 50
        if chatgpt_analysis.get('available'):
            vol_forecast = chatgpt_analysis.get('volatility_forecast', 'MEDIUM')
            if vol_forecast == 'HIGH':
                gpt_vol_score = 75
            elif vol_forecast == 'LOW':
                gpt_vol_score = 30
            else:
                gpt_vol_score = 50
        
        combined['volatility_score'] = int(tech_vol_score * TECH_WEIGHT + gpt_vol_score * GPT_WEIGHT)
        
        # Expected range
        tech_range = tech_analysis['expected_range']
        gpt_range = chatgpt_analysis.get('expected_range_percent', tech_range)
        combined['expected_range'] = round(tech_range * TECH_WEIGHT + gpt_range * GPT_WEIGHT, 2)
        
        # Direction (mostly GPT since technical is ~50%)
        if chatgpt_analysis.get('available'):
            combined['direction'] = chatgpt_analysis.get('direction', 'NEUTRAL')
            combined['direction_confidence'] = chatgpt_analysis.get('direction_confidence', 50)
        else:
            combined['direction'] = 'NEUTRAL'
            combined['direction_confidence'] = 50
        
        # Volatility regime
        combined['vol_regime'] = tech_analysis['vol_regime']
        
        # Trading recommendation
        if combined['volatility_score'] > 70:
            combined['vol_action'] = 'HIGH_VOL_EXPECTED'
            combined['strategy'] = 'Consider straddle/strangle - big move likely'
        elif combined['volatility_score'] < 35:
            combined['vol_action'] = 'LOW_VOL_EXPECTED'
            combined['strategy'] = 'Consider iron condor - range-bound expected'
        else:
            combined['vol_action'] = 'NORMAL_VOL'
            combined['strategy'] = 'Standard directional if conviction exists'
        
        # Risk level
        if combined['expected_range'] > 3:
            combined['risk_level'] = 'HIGH'
        elif combined['expected_range'] > 2:
            combined['risk_level'] = 'MEDIUM'
        else:
            combined['risk_level'] = 'LOW'
        
        # Sources
        combined['tech_reasons'] = tech_analysis['reasons']
        combined['gpt_reasoning'] = chatgpt_analysis.get('reasoning', 'N/A')
        combined['gpt_available'] = chatgpt_analysis.get('available', False)
        
        return combined
    
    def run_analysis(self):
        """Run full analysis"""
        print("\n" + "="*70)
        print("📊 MCX VOLATILITY & VOLUME ANALYSIS")
        print("="*70)
        
        # Fetch data
        print("\n⏳ Fetching MCX data...")
        df = self.fetch_data()
        
        # Calculate features
        print("📈 Calculating volatility features...")
        features = self.calculate_volatility_features(df)
        
        # Technical volatility prediction
        print("🔧 Running technical analysis...")
        tech_analysis = self.predict_volatility(features)
        
        # ChatGPT analysis
        print("🤖 Getting ChatGPT analysis...")
        gpt_analysis = self.get_chatgpt_analysis(features)
        
        # Combine signals
        print("⚖️ Combining signals with weightage...")
        combined = self.combine_signals(tech_analysis, gpt_analysis)
        
        # Display results
        self.display_results(features, tech_analysis, gpt_analysis, combined)
        
        return {
            'features': features,
            'technical': tech_analysis,
            'chatgpt': gpt_analysis,
            'combined': combined
        }
    
    def display_results(self, features, tech, gpt, combined):
        """Display analysis results"""
        print("\n" + "="*70)
        print("📊 ANALYSIS RESULTS")
        print("="*70)
        
        # Current state
        print(f"\n💰 CURRENT STATE:")
        print(f"   Price: ₹{features['price']:,.2f}")
        print(f"   Today: {features['today_return']:+.2f}% | Range: {features['today_range']:.2f}%")
        print(f"   RSI: {features['rsi']:.1f}")
        
        # Volatility metrics
        print(f"\n📈 VOLATILITY METRICS:")
        print(f"   20-day Volatility: {features['volatility_20d']:.1f}% annualized")
        print(f"   5-day Volatility: {features['volatility_5d']:.1f}% annualized")
        print(f"   ATR: ₹{features['atr']:.2f} ({features['atr_percent']:.2f}%)")
        print(f"   ATR vs Average: {features['atr_vs_avg']:.2f}x")
        print(f"   Regime: {features['vol_regime']}")
        
        # Volume
        print(f"\n📊 VOLUME:")
        print(f"   Today vs 20-day avg: {features['volume_ratio']:.2f}x")
        print(f"   5-day trend: {features['volume_trend']:.2f}x")
        
        # Technical analysis
        print(f"\n🔧 TECHNICAL VOLATILITY PREDICTION:")
        print(f"   Score: {tech['volatility_score']}/100")
        print(f"   Expected Range: {tech['expected_range']:.2f}%")
        for reason in tech['reasons']:
            print(f"   • {reason}")
        
        # ChatGPT analysis
        print(f"\n🤖 CHATGPT ANALYSIS:")
        if gpt.get('available'):
            print(f"   Direction: {gpt.get('direction')} ({gpt.get('direction_confidence')}% conf)")
            print(f"   Volatility: {gpt.get('volatility_forecast')} ({gpt.get('volatility_confidence', 'N/A')}% conf)")
            print(f"   Reasoning: {gpt.get('reasoning', 'N/A')}")
            if gpt.get('risk_warning'):
                print(f"   ⚠️ Risk: {gpt.get('risk_warning')}")
        else:
            print(f"   ❌ Not available - {gpt.get('reasoning', 'API not configured')}")
        
        # Combined signal
        print(f"\n" + "="*70)
        print(f"⚡ COMBINED SIGNAL (Tech 70% + GPT 30%)")
        print(f"="*70)
        print(f"\n   🎯 VOLATILITY SCORE: {combined['volatility_score']}/100")
        print(f"   📏 EXPECTED RANGE: {combined['expected_range']:.2f}%")
        print(f"   📊 DIRECTION: {combined['direction']} ({combined['direction_confidence']}% conf)")
        print(f"   ⚠️ RISK LEVEL: {combined['risk_level']}")
        print(f"\n   💡 STRATEGY: {combined['strategy']}")
        
        # Options trading guidance
        print(f"\n" + "="*70)
        print(f"📋 OPTIONS TRADING GUIDANCE (Lot Size: 625)")
        print(f"="*70)
        
        price = features['price']
        expected_move = price * combined['expected_range'] / 100
        
        if combined['volatility_score'] > 65:
            print(f"\n   🎯 HIGH VOLATILITY EXPECTED")
            print(f"   Expected Move: ±₹{expected_move:.0f} (±{combined['expected_range']:.1f}%)")
            print(f"\n   Consider: LONG STRADDLE/STRANGLE")
            print(f"   • ATM Strike: {round(price/100)*100}")
            print(f"   • Upper target: ₹{price + expected_move:.0f}")
            print(f"   • Lower target: ₹{price - expected_move:.0f}")
            print(f"   • Max loss: Premium paid")
            
        elif combined['volatility_score'] < 40:
            print(f"\n   😴 LOW VOLATILITY EXPECTED")
            print(f"   Expected Range: ±₹{expected_move:.0f}")
            print(f"\n   Consider: SHORT STRADDLE or IRON CONDOR")
            print(f"   • Sell strikes outside expected range")
            print(f"   • Upper sell: {round((price + expected_move*1.5)/100)*100}")
            print(f"   • Lower sell: {round((price - expected_move*1.5)/100)*100}")
            print(f"   ⚠️ Risk: Unlimited on naked, defined on iron condor")
            
        else:
            print(f"\n   📊 NORMAL VOLATILITY")
            if combined['direction'] == 'BULLISH' and combined['direction_confidence'] > 55:
                print(f"   Slight bullish lean - consider CALL spread")
            elif combined['direction'] == 'BEARISH' and combined['direction_confidence'] > 55:
                print(f"   Slight bearish lean - consider PUT spread")
            else:
                print(f"   No strong conviction - consider waiting or small position")
        
        print(f"\n" + "="*70)
        print(f"⚠️ DISCLAIMER: Not financial advice. Always manage risk!")
        print(f"="*70)


def main():
    """Main function"""
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("\n" + "="*70)
        print("🔑 OPENAI API KEY SETUP")
        print("="*70)
        print("\nTo enable ChatGPT analysis, set your API key:")
        print("  Option 1: Set environment variable OPENAI_API_KEY")
        print("  Option 2: Enter it below (or press Enter to skip)")
        
        user_key = input("\nEnter OpenAI API key (or Enter to skip): ").strip()
        if user_key:
            api_key = user_key
    
    # Run analysis
    predictor = VolatilityPredictor(openai_api_key=api_key)
    results = predictor.run_analysis()
    
    return results


if __name__ == '__main__':
    main()
