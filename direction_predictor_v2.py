"""
MCX Direction Prediction Engine v2.0
=====================================
Honest Assessment + Improved Model

REALITY CHECK:
- Stock direction prediction is HARD
- Even the best quant funds get 53-55% accuracy
- The edge comes from SIZING and RISK MANAGEMENT

IMPROVEMENTS:
1. Better feature engineering
2. Multiple ensemble methods
3. Honest confidence calibration
4. Only trade when multiple signals align
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


def create_predictive_features(df):
    """Features with actual predictive power based on research"""
    
    data = pd.DataFrame(index=df.index)
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    ret = close.pct_change()
    
    # =========================================================================
    # MOMENTUM (short-term continuation)
    # =========================================================================
    
    for lag in [1, 2, 3, 5]:
        data[f'ret_{lag}d'] = close.pct_change(lag)
    
    # Momentum at different horizons
    data['mom_5d'] = close.pct_change(5)
    data['mom_10d'] = close.pct_change(10)
    data['mom_20d'] = close.pct_change(20)
    
    # Momentum change
    data['mom_accel'] = data['mom_5d'] - data['mom_5d'].shift(5)
    
    # =========================================================================
    # MEAN REVERSION (key for prediction)
    # =========================================================================
    
    # Price relative to moving averages
    for w in [5, 10, 20, 50]:
        ma = close.rolling(w).mean()
        data[f'ma{w}_ratio'] = close / ma - 1
    
    # Z-score (critical for mean reversion)
    for w in [10, 20, 50]:
        data[f'zscore_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-10)
    
    # Distance from Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_upper_dist'] = (close - (ma20 + 2*std20)) / close
    data['bb_lower_dist'] = (close - (ma20 - 2*std20)) / close
    
    # =========================================================================
    # RSI (proven predictor)
    # =========================================================================
    
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        data[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # RSI extreme zones (mean reversion signal)
    data['rsi_14_extreme'] = np.where(data['rsi_14'] > 70, 1, np.where(data['rsi_14'] < 30, -1, 0))
    
    # =========================================================================
    # VOLATILITY (affects prediction reliability)
    # =========================================================================
    
    for w in [5, 10, 20]:
        data[f'vol_{w}d'] = ret.rolling(w).std() * np.sqrt(252)
    
    # Volatility regime
    data['vol_ratio'] = data['vol_5d'] / (data['vol_20d'] + 1e-10)
    
    # ATR normalized
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    data['atr_pct'] = tr.rolling(14).mean() / close
    
    # =========================================================================
    # VOLUME (confirms moves)
    # =========================================================================
    
    data['vol_ratio_10'] = volume / (volume.rolling(10).mean() + 1e-10)
    data['vol_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    
    # Up volume vs down volume
    up_vol = volume.where(ret > 0, 0).rolling(10).sum()
    down_vol = volume.where(ret < 0, 0).rolling(10).sum()
    data['up_down_vol_ratio'] = up_vol / (down_vol + 1e-10)
    
    # =========================================================================
    # PRICE PATTERNS
    # =========================================================================
    
    # Price position in recent range
    data['price_pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
    
    # Higher highs / lower lows
    data['new_20d_high'] = (close >= high.rolling(20).max().shift(1)).astype(int)
    data['new_20d_low'] = (close <= low.rolling(20).min().shift(1)).astype(int)
    
    # =========================================================================
    # TREND STRENGTH
    # =========================================================================
    
    # ADX approximation
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_14 = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).sum() / (tr_14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    data['adx'] = dx.rolling(14).mean()
    
    # =========================================================================
    # CROSS SIGNALS
    # =========================================================================
    
    data['ma5_above_ma20'] = (close.rolling(5).mean() > close.rolling(20).mean()).astype(int)
    data['price_above_ma50'] = (close > close.rolling(50).mean()).astype(int)
    
    return data


class ImprovedDirectionPredictor:
    """Improved ensemble with multiple model types"""
    
    def __init__(self):
        self.models = {
            'gb': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            ),
            'et': ExtraTreesClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            ),
            'ada': AdaBoostClassifier(
                n_estimators=50, learning_rate=0.1, random_state=42
            ),
            'lr': LogisticRegression(C=0.1, max_iter=500, random_state=42)
        }
        
        self.scaler = RobustScaler()
        self.fitted_models = {}
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            # Use calibration for better probability estimates
            calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
            calibrated.fit(X_scaled, y)
            self.fitted_models[name] = calibrated
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        all_probas = []
        all_preds = []
        
        for name, model in self.fitted_models.items():
            proba = model.predict_proba(X_scaled)
            pred = model.predict(X_scaled)
            all_probas.append(proba)
            all_preds.append(pred)
        
        # Average probabilities across models
        avg_proba = np.mean(all_probas, axis=0)
        
        # Voting prediction
        vote_pred = (np.mean(all_preds, axis=0) > 0.5).astype(int)
        
        # Confidence: how many models agree
        agreement = np.mean([p == vote_pred for p in all_preds], axis=0)
        
        # Final confidence = probability * agreement
        confidence = np.max(avg_proba, axis=1) * agreement
        
        return vote_pred, confidence, avg_proba, agreement


def run_improved_backtest():
    """Run honest backtest with improved model"""
    
    print("="*70)
    print("MCX DIRECTION PREDICTION ENGINE v2.0")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("HONEST ASSESSMENT:")
    print("- Stock prediction is fundamentally difficult")
    print("- 55-60% accuracy is EXCELLENT in real trading")
    print("- Focus on high-confidence signals only")
    print()
    
    # Fetch data
    print("1. FETCHING DATA")
    print("-"*70)
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='2y')
    print(f"   Records: {len(df)} trading days")
    
    # Create features
    print()
    print("2. FEATURE ENGINEERING")
    print("-"*70)
    features = create_predictive_features(df)
    
    # Target
    next_ret = df['Close'].pct_change().shift(-1)
    features['target'] = (next_ret > 0).astype(int)
    
    # Drop NaN
    features_clean = features.dropna()
    print(f"   Features: {len([c for c in features_clean.columns if c != 'target'])}")
    print(f"   Samples: {len(features_clean)}")
    
    up_pct = features_clean['target'].mean() * 100
    print(f"   Base rate (UP days): {up_pct:.1f}%")
    print(f"   (Random guess accuracy: ~{max(up_pct, 100-up_pct):.1f}%)")
    
    # Prepare data
    feature_cols = [c for c in features_clean.columns if c != 'target']
    X = features_clean[feature_cols].values
    y = features_clean['target'].values
    dates = features_clean.index
    
    # Walk-forward validation
    print()
    print("3. WALK-FORWARD BACKTEST")
    print("-"*70)
    print("   Training on past 120 days, predicting day 121...")
    
    train_window = 120
    results = []
    predictor = ImprovedDirectionPredictor()
    
    for i in range(train_window, len(X) - 1):
        X_train = X[max(0, i-train_window):i]
        y_train = y[max(0, i-train_window):i]
        
        predictor.fit(X_train, y_train)
        
        X_test = X[i:i+1]
        pred, conf, proba, agreement = predictor.predict(X_test)
        actual = y[i]
        
        results.append({
            'date': dates[i],
            'predicted': int(pred[0]),
            'actual': int(actual),
            'confidence': float(conf[0]),
            'prob_up': float(proba[0][1]),
            'prob_down': float(proba[0][0]),
            'model_agreement': float(agreement[0]),
            'correct': pred[0] == actual
        })
        
        if len(results) % 50 == 0:
            recent = results[-50:]
            acc = np.mean([r['correct'] for r in recent]) * 100
            high_conf = [r for r in recent if r['confidence'] > 0.55]
            hc_acc = np.mean([r['correct'] for r in high_conf]) * 100 if high_conf else 0
            print(f"   Day {len(results):3d}: All={acc:.1f}% | HighConf={hc_acc:.1f}% ({len(high_conf)}/{len(recent)})")
    
    results_df = pd.DataFrame(results)
    
    # Performance Analysis
    print()
    print("="*70)
    print("4. PERFORMANCE RESULTS")
    print("="*70)
    
    overall_acc = results_df['correct'].mean() * 100
    print(f"\n   📊 OVERALL:")
    print(f"      Predictions: {len(results_df)}")
    print(f"      Accuracy: {overall_acc:.1f}%")
    print(f"      vs Random: {'+' if overall_acc > 50 else ''}{overall_acc - 50:.1f}% edge")
    
    # Confidence analysis
    print(f"\n   🎯 BY CONFIDENCE LEVEL:")
    print(f"      {'Threshold':<12} {'Accuracy':>10} {'Trades':>10} {'% Days':>10}")
    print(f"      {'-'*45}")
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        filtered = results_df[results_df['confidence'] > threshold]
        if len(filtered) >= 10:
            acc = filtered['correct'].mean() * 100
            pct = len(filtered) / len(results_df) * 100
            edge = acc - 50
            marker = "⭐" if acc >= 60 else "✓" if acc >= 55 else ""
            print(f"      >{threshold*100:4.0f}%      {acc:>9.1f}% {len(filtered):>10} {pct:>9.1f}% {marker}")
    
    # Model agreement analysis
    print(f"\n   🤝 BY MODEL AGREEMENT:")
    for agree_threshold in [0.6, 0.8, 1.0]:
        filtered = results_df[results_df['model_agreement'] >= agree_threshold]
        if len(filtered) >= 10:
            acc = filtered['correct'].mean() * 100
            print(f"      {agree_threshold*100:.0f}%+ models agree: {acc:.1f}% accuracy ({len(filtered)} trades)")
    
    # Combined filter (high confidence + high agreement)
    print(f"\n   🏆 BEST TRADING SIGNALS:")
    best_signals = results_df[(results_df['confidence'] > 0.60) & (results_df['model_agreement'] >= 0.80)]
    if len(best_signals) >= 5:
        best_acc = best_signals['correct'].mean() * 100
        print(f"      Conf>60% AND Agreement≥80%:")
        print(f"      Accuracy: {best_acc:.1f}%")
        print(f"      Trades: {len(best_signals)} ({len(best_signals)/len(results_df)*100:.1f}% of days)")
    
    # Last 30 days
    print(f"\n   📅 LAST 30 DAYS:")
    last30 = results_df.tail(30)
    print(f"      Overall: {last30['correct'].mean()*100:.1f}%")
    
    last30_hc = last30[last30['confidence'] > 0.60]
    if len(last30_hc) > 0:
        print(f"      High-Conf: {last30_hc['correct'].mean()*100:.1f}% ({len(last30_hc)} trades)")
    
    # Detailed last 10
    print()
    print("="*70)
    print("5. LAST 10 PREDICTIONS")
    print("="*70)
    print(f"{'Date':<12} {'Pred':>6} {'Actual':>6} {'Conf':>6} {'Agree':>6} {'Result':>7}")
    print("-"*50)
    
    for _, row in results_df.tail(10).iterrows():
        pred_str = '📈UP' if row['predicted'] == 1 else '📉DN'
        actual_str = 'UP' if row['actual'] == 1 else 'DOWN'
        result = '✓' if row['correct'] else '✗'
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {pred_str:>6} {actual_str:>6} {row['confidence']*100:>5.0f}% {row['model_agreement']*100:>5.0f}% {result:>7}")
    
    # Tomorrow's prediction
    print()
    print("="*70)
    print("6. TOMORROW'S PREDICTION")
    print("="*70)
    
    predictor.fit(X[-120:], y[-120:])
    pred, conf, proba, agreement = predictor.predict(X[-1:])
    
    current_price = df['Close'].iloc[-1]
    direction = 'UP 📈' if pred[0] == 1 else 'DOWN 📉'
    action = 'BUY CALL' if pred[0] == 1 else 'BUY PUT'
    
    print(f"\n   Current Price: ₹{current_price:.2f}")
    print(f"\n   🎯 PREDICTION: {direction}")
    print(f"   📊 Confidence: {conf[0]*100:.1f}%")
    print(f"   🤝 Model Agreement: {agreement[0]*100:.0f}%")
    print(f"   📈 Prob UP: {proba[0][1]*100:.1f}%")
    print(f"   📉 Prob DOWN: {proba[0][0]*100:.1f}%")
    
    # Signal quality
    if conf[0] > 0.60 and agreement[0] >= 0.80:
        print(f"\n   ✅ STRONG SIGNAL - Consider trading")
        print(f"   💡 Action: {action}")
    elif conf[0] > 0.55:
        print(f"\n   ⚠️ MODERATE SIGNAL - Use smaller position")
    else:
        print(f"\n   ❌ WEAK SIGNAL - Skip this trade")
    
    # Context
    rsi = features_clean['rsi_14'].iloc[-1]
    zscore = features_clean['zscore_20'].iloc[-1]
    vol = features_clean['vol_20d'].iloc[-1]
    
    print(f"\n   📋 MARKET CONTEXT:")
    print(f"      RSI(14): {rsi:.1f}", end="")
    if rsi > 70: print(" (OVERBOUGHT - caution on calls)")
    elif rsi < 30: print(" (OVERSOLD - caution on puts)")
    else: print(" (Neutral)")
    
    print(f"      Z-Score: {zscore:+.2f}", end="")
    if abs(zscore) > 2: print(" (EXTENDED - mean reversion likely)")
    else: print("")
    
    print(f"      Volatility: {vol*100:.1f}% annualized")
    
    # Risk warning
    print()
    print("="*70)
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("-"*70)
    print("• Direction prediction is ~55-60% accurate at best")
    print("• Only trade high-confidence signals (>60% conf, >80% agreement)")
    print("• Use proper position sizing (max 2% risk per trade)")
    print("• This is a decision SUPPORT tool, not financial advice")
    print("="*70)
    
    return results_df, predictor, features_clean


if __name__ == '__main__':
    results, predictor, features = run_improved_backtest()
