"""
MCX UNIFIED PREDICTION SYSTEM
==============================
Problem: Technical, ChatGPT, and News were giving disconnected signals
Solution: Single unified consensus with clear reasoning

HONEST TRUTH:
1. Direction prediction is ~50% (we accept this)
2. Volatility prediction is better (~65%)
3. News is REACTIVE (about past), not PREDICTIVE
4. The 15% crash ALREADY HAPPENED - question is: what next?
"""

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


class UnifiedPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key) if OPENAI_AVAILABLE and self.api_key else None
    
    def fetch_data(self):
        """Get MCX data"""
        mcx = yf.Ticker('MCX.NS')
        df = mcx.history(period='1y')
        return df
    
    def analyze_technicals(self, df):
        """Pure technical analysis - no interpretation"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        ret = close.pct_change()
        
        # Facts only - no predictions
        facts = {
            'price': float(close.iloc[-1]),
            'prev_close': float(close.iloc[-2]),
            'change_today': float(ret.iloc[-1] * 100),
            'change_5d': float((close.iloc[-1] / close.iloc[-5] - 1) * 100),
            'change_20d': float((close.iloc[-1] / close.iloc[-20] - 1) * 100),
        }
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        facts['rsi'] = float(rsi.iloc[-1])
        
        # Volatility
        facts['volatility_20d'] = float(ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        facts['volatility_5d'] = float(ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
        
        # ATR
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        facts['atr'] = float(atr.iloc[-1])
        facts['atr_pct'] = float(atr.iloc[-1] / close.iloc[-1] * 100)
        facts['atr_vs_avg'] = float(atr.iloc[-1] / atr.rolling(50).mean().iloc[-1])
        
        # Moving averages
        facts['ma5'] = float(close.rolling(5).mean().iloc[-1])
        facts['ma20'] = float(close.rolling(20).mean().iloc[-1])
        facts['ma50'] = float(close.rolling(50).mean().iloc[-1])
        facts['price_vs_ma20'] = float((close.iloc[-1] / facts['ma20'] - 1) * 100)
        
        # Volume
        facts['volume_ratio'] = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
        
        # Bollinger position
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        facts['bb_position'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
        
        # Support/Resistance
        facts['support_20d'] = float(low.rolling(20).min().iloc[-1])
        facts['resistance_20d'] = float(high.rolling(20).max().iloc[-1])
        
        # Recent pattern
        facts['last_5_returns'] = [round(r*100, 2) for r in ret.iloc[-5:].tolist()]
        facts['up_days_last_5'] = sum(1 for r in ret.iloc[-5:] if r > 0)
        facts['down_days_last_5'] = sum(1 for r in ret.iloc[-5:] if r < 0)
        
        return facts
    
    def get_unified_gpt_analysis(self, facts):
        """Get ONE unified analysis from ChatGPT"""
        if not self.client:
            return None
        
        prompt = f"""You are a trading analyst. Give ONE UNIFIED recommendation for MCX Ltd stock.

CURRENT FACTS (as of today):
- Price: ₹{facts['price']:.2f}
- Today's Change: {facts['change_today']:.2f}%
- 5-Day Change: {facts['change_5d']:.2f}%
- 20-Day Change: {facts['change_20d']:.2f}%

TECHNICAL INDICATORS:
- RSI (14): {facts['rsi']:.1f}
- Price vs MA20: {facts['price_vs_ma20']:.1f}%
- Bollinger Position: {facts['bb_position']:.2f} (0=lower band, 1=upper band)
- Volume: {facts['volume_ratio']:.2f}x average

VOLATILITY:
- 5-Day Volatility: {facts['volatility_5d']:.1f}% annualized
- 20-Day Volatility: {facts['volatility_20d']:.1f}% annualized
- ATR: {facts['atr_pct']:.2f}% of price
- ATR vs 50-day avg: {facts['atr_vs_avg']:.2f}x

RECENT PATTERN:
- Last 5 days returns: {facts['last_5_returns']}
- Up days: {facts['up_days_last_5']}, Down days: {facts['down_days_last_5']}

IMPORTANT CONTEXT:
1. If there was a recent crash (large negative %), the question is: bounce or continue?
2. High volatility after a crash often means more movement coming
3. Be HONEST about uncertainty - direction is hard to predict

Give your analysis in this EXACT JSON format:
{{
    "situation_summary": "1-2 sentence summary of current situation",
    "direction_view": "BULLISH" or "BEARISH" or "NEUTRAL",
    "direction_confidence": 45-60 (be realistic, >55 is high),
    "direction_reasoning": "Why this direction (or why uncertain)",
    "volatility_view": "HIGH" or "MEDIUM" or "LOW",
    "volatility_confidence": 50-85,
    "volatility_reasoning": "Why this volatility expectation",
    "expected_range_pct": 2.0-10.0,
    "recommendation": "STRADDLE" or "CALL_SPREAD" or "PUT_SPREAD" or "WAIT" or "IRON_CONDOR",
    "recommendation_reasoning": "Why this strategy",
    "key_risk": "Main risk to watch",
    "honest_disclaimer": "What we DON'T know"
}}

Respond with ONLY the JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a quantitative analyst. Be honest about uncertainty. JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            if '```' in content:
                content = content.split('```')[1].replace('json', '').strip()
            
            return json.loads(content)
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_unified_signal(self, facts, gpt_analysis):
        """Create ONE unified signal from all inputs"""
        
        signal = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price': facts['price'],
            'sources_agree': False,
        }
        
        # Technical signals (rule-based, no interpretation)
        tech_signals = []
        
        if facts['rsi'] < 30:
            tech_signals.append(('RSI', 'OVERSOLD', 'Potential bounce'))
        elif facts['rsi'] > 70:
            tech_signals.append(('RSI', 'OVERBOUGHT', 'Potential pullback'))
        else:
            tech_signals.append(('RSI', 'NEUTRAL', f"RSI at {facts['rsi']:.0f}"))
        
        if facts['price_vs_ma20'] < -5:
            tech_signals.append(('MA20', 'BELOW', f"{facts['price_vs_ma20']:.1f}% below MA20"))
        elif facts['price_vs_ma20'] > 5:
            tech_signals.append(('MA20', 'ABOVE', f"+{facts['price_vs_ma20']:.1f}% above MA20"))
        else:
            tech_signals.append(('MA20', 'NEAR', 'Price near MA20'))
        
        if facts['bb_position'] < 0.2:
            tech_signals.append(('BB', 'LOWER', 'Near lower Bollinger band'))
        elif facts['bb_position'] > 0.8:
            tech_signals.append(('BB', 'UPPER', 'Near upper Bollinger band'))
        else:
            tech_signals.append(('BB', 'MIDDLE', 'In middle of Bollinger bands'))
        
        # Volatility assessment (more reliable)
        if facts['atr_vs_avg'] > 1.5:
            vol_signal = ('VOLATILITY', 'HIGH', f"ATR {facts['atr_vs_avg']:.1f}x above normal")
        elif facts['atr_vs_avg'] < 0.7:
            vol_signal = ('VOLATILITY', 'LOW', f"ATR {facts['atr_vs_avg']:.1f}x below normal")
        else:
            vol_signal = ('VOLATILITY', 'NORMAL', 'Normal volatility range')
        tech_signals.append(vol_signal)
        
        signal['technical_signals'] = tech_signals
        
        # GPT analysis
        if gpt_analysis and 'error' not in gpt_analysis:
            signal['gpt_analysis'] = gpt_analysis
            signal['has_gpt'] = True
        else:
            signal['has_gpt'] = False
        
        # Create UNIFIED recommendation
        # Priority: Volatility > Direction (since vol is more predictable)
        
        # Determine volatility level
        if facts['atr_vs_avg'] > 1.3 or facts['volatility_5d'] > facts['volatility_20d'] * 1.3:
            signal['volatility_level'] = 'HIGH'
            signal['volatility_confidence'] = 75
        elif facts['atr_vs_avg'] < 0.7:
            signal['volatility_level'] = 'LOW'
            signal['volatility_confidence'] = 70
        else:
            signal['volatility_level'] = 'NORMAL'
            signal['volatility_confidence'] = 60
        
        # Expected range
        signal['expected_range_pct'] = facts['atr_pct'] * (1.5 if signal['volatility_level'] == 'HIGH' else 1.0)
        signal['expected_move'] = round(facts['price'] * signal['expected_range_pct'] / 100, 0)
        
        # Direction (LOW confidence by default)
        direction_score = 50  # Start neutral
        
        # Slight adjustments based on technicals (but acknowledge uncertainty)
        if facts['rsi'] < 35:
            direction_score += 5  # Slight bullish bias (oversold)
        elif facts['rsi'] > 65:
            direction_score -= 5  # Slight bearish bias (overbought)
        
        if facts['bb_position'] < 0.2:
            direction_score += 5  # Mean reversion bias
        elif facts['bb_position'] > 0.8:
            direction_score -= 5
        
        # Recent momentum (weak signal)
        if facts['up_days_last_5'] >= 4:
            direction_score -= 3  # Possible exhaustion
        elif facts['down_days_last_5'] >= 4:
            direction_score += 3  # Possible bounce
        
        signal['direction_score'] = direction_score
        if direction_score > 55:
            signal['direction'] = 'BULLISH'
        elif direction_score < 45:
            signal['direction'] = 'BEARISH'
        else:
            signal['direction'] = 'NEUTRAL'
        
        # Direction confidence is ALWAYS low (honest)
        signal['direction_confidence'] = min(55, 45 + abs(direction_score - 50))
        
        # FINAL UNIFIED RECOMMENDATION
        if signal['volatility_level'] == 'HIGH':
            signal['recommendation'] = 'STRADDLE/STRANGLE'
            signal['recommendation_reason'] = 'High volatility expected - big move likely but direction uncertain'
        elif signal['volatility_level'] == 'LOW':
            signal['recommendation'] = 'IRON CONDOR / SELL PREMIUM'
            signal['recommendation_reason'] = 'Low volatility - range-bound, sell premium'
        else:
            if signal['direction'] == 'BULLISH' and signal['direction_confidence'] >= 52:
                signal['recommendation'] = 'BULL CALL SPREAD (small size)'
                signal['recommendation_reason'] = 'Slight bullish lean, but keep position small'
            elif signal['direction'] == 'BEARISH' and signal['direction_confidence'] >= 52:
                signal['recommendation'] = 'BEAR PUT SPREAD (small size)'
                signal['recommendation_reason'] = 'Slight bearish lean, but keep position small'
            else:
                signal['recommendation'] = 'WAIT / NO TRADE'
                signal['recommendation_reason'] = 'No clear edge - preserve capital'
        
        # Key warnings
        signal['warnings'] = [
            f"Direction prediction accuracy: ~{signal['direction_confidence']}% (near random)",
            f"Volatility prediction more reliable: ~{signal['volatility_confidence']}%",
            "Past news (like crash) already reflected in price",
            "Use small position sizes - max 2% of capital at risk"
        ]
        
        return signal
    
    def run(self):
        """Run unified analysis"""
        print("\n" + "="*70)
        print("📊 MCX UNIFIED PREDICTION SYSTEM")
        print("="*70)
        print("⚠️  HONEST APPROACH: One clear signal, acknowledging uncertainty")
        print("="*70)
        
        # Get data
        print("\n⏳ Fetching data...")
        df = self.fetch_data()
        
        # Analyze
        print("📈 Analyzing technicals...")
        facts = self.analyze_technicals(df)
        
        print("🤖 Getting unified GPT analysis...")
        gpt = self.get_unified_gpt_analysis(facts)
        
        print("⚡ Creating unified signal...")
        signal = self.create_unified_signal(facts, gpt)
        
        # Display
        self.display(facts, signal)
        
        return signal
    
    def display(self, facts, signal):
        """Display unified results"""
        print("\n" + "="*70)
        print("💰 CURRENT STATE")
        print("="*70)
        print(f"   Price: ₹{facts['price']:,.2f}")
        print(f"   Today: {facts['change_today']:+.2f}%")
        print(f"   5-Day: {facts['change_5d']:+.2f}%")
        print(f"   20-Day: {facts['change_20d']:+.2f}%")
        print(f"   RSI: {facts['rsi']:.1f}")
        print(f"   vs MA20: {facts['price_vs_ma20']:+.1f}%")
        
        print("\n" + "="*70)
        print("📊 TECHNICAL SIGNALS")
        print("="*70)
        for name, status, detail in signal['technical_signals']:
            emoji = '🟢' if 'OVERSOLD' in status or 'BELOW' in status else '🔴' if 'OVERBOUGHT' in status or 'ABOVE' in status else '🟡'
            print(f"   {emoji} {name}: {status} - {detail}")
        
        if signal.get('has_gpt') and signal.get('gpt_analysis'):
            gpt = signal['gpt_analysis']
            print("\n" + "="*70)
            print("🤖 CHATGPT ANALYSIS")
            print("="*70)
            print(f"   📝 Situation: {gpt.get('situation_summary', 'N/A')}")
            print(f"   📈 Direction: {gpt.get('direction_view', 'N/A')} ({gpt.get('direction_confidence', 50)}% conf)")
            print(f"   📊 Volatility: {gpt.get('volatility_view', 'N/A')} ({gpt.get('volatility_confidence', 50)}% conf)")
            print(f"   💡 GPT Recommendation: {gpt.get('recommendation', 'N/A')}")
            print(f"   ⚠️ Risk: {gpt.get('key_risk', 'N/A')}")
        
        print("\n" + "="*70)
        print("⚡ UNIFIED RECOMMENDATION")
        print("="*70)
        
        # Big clear recommendation
        print(f"""
   ╔════════════════════════════════════════════════════════════╗
   ║                                                            ║
   ║   🎯 {signal['recommendation']:^48} ║
   ║                                                            ║
   ╚════════════════════════════════════════════════════════════╝
        """)
        
        print(f"   📝 Reason: {signal['recommendation_reason']}")
        print(f"\n   📊 Volatility: {signal['volatility_level']} ({signal['volatility_confidence']}% confident)")
        print(f"   📈 Direction: {signal['direction']} ({signal['direction_confidence']}% confident)")
        print(f"   📏 Expected Range: ±{signal['expected_range_pct']:.1f}% (±₹{signal['expected_move']:.0f})")
        
        print("\n" + "="*70)
        print("⚠️ HONEST WARNINGS")
        print("="*70)
        for w in signal['warnings']:
            print(f"   • {w}")
        
        print("\n" + "="*70)
        print("📋 OPTIONS TRADE SETUP (Lot Size: 625)")
        print("="*70)
        atm = round(facts['price'] / 100) * 100
        upper = round(facts['price'] + signal['expected_move'])
        lower = round(facts['price'] - signal['expected_move'])
        
        print(f"   ATM Strike: ₹{atm}")
        print(f"   Upper Target: ₹{upper}")
        print(f"   Lower Target: ₹{lower}")
        
        if 'STRADDLE' in signal['recommendation']:
            print(f"\n   🎯 STRADDLE SETUP:")
            print(f"   • Buy {atm} CE + Buy {atm} PE")
            print(f"   • Max Loss: Premium paid")
            print(f"   • Breakeven: ₹{lower} - ₹{upper}")
        elif 'IRON CONDOR' in signal['recommendation']:
            print(f"\n   🎯 IRON CONDOR SETUP:")
            print(f"   • Sell {atm-100} PE, Buy {atm-200} PE")
            print(f"   • Sell {atm+100} CE, Buy {atm+200} CE")
        elif 'CALL' in signal['recommendation']:
            print(f"\n   🎯 BULL CALL SPREAD:")
            print(f"   • Buy {atm} CE, Sell {atm+100} CE")
        elif 'PUT' in signal['recommendation']:
            print(f"\n   🎯 BEAR PUT SPREAD:")
            print(f"   • Buy {atm} PE, Sell {atm-100} PE")
        else:
            print(f"\n   ⏸️ NO TRADE - Wait for better setup")
        
        print("\n" + "="*70)


def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    predictor = UnifiedPredictor(api_key)
    signal = predictor.run()
    return signal


if __name__ == '__main__':
    main()
