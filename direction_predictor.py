"""
MCX Direction Prediction Engine
================================
Focus: Predict UP or DOWN with high accuracy
Target: 65%+ accuracy on high-confidence trades

Key Insight: For options trading, DIRECTION is everything.
A 2% price error that's in the wrong direction = 100% loss on options.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


def create_direction_features(df):
    """Create features optimized for direction prediction"""
    
    data = pd.DataFrame(index=df.index)
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # =========================================================================
    # MOMENTUM FEATURES (key for direction)
    # =========================================================================
    
    # Returns at multiple horizons
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f'ret_{lag}d'] = close.pct_change(lag)
    
    # Consecutive up/down days
    daily_ret = close.pct_change()
    data['up_streak'] = (daily_ret > 0).astype(int).groupby((daily_ret > 0).astype(int).diff().ne(0).cumsum()).cumsum()
    data['down_streak'] = (daily_ret < 0).astype(int).groupby((daily_ret < 0).astype(int).diff().ne(0).cumsum()).cumsum()
    
    # Return acceleration
    data['ret_accel'] = data['ret_1d'] - data['ret_1d'].shift(1)
    
    # =========================================================================
    # MEAN REVERSION FEATURES
    # =========================================================================
    
    # MA ratios (price vs moving average)
    for w in [5, 10, 20, 50]:
        ma = close.rolling(w).mean()
        data[f'ma{w}_ratio'] = close / ma
        data[f'ma{w}_dist'] = (close - ma) / ma
    
    # MA crossover signals
    data['ma5_ma20_cross'] = (close.rolling(5).mean() > close.rolling(20).mean()).astype(int)
    data['ma10_ma50_cross'] = (close.rolling(10).mean() > close.rolling(50).mean()).astype(int)
    
    # Z-score (how many std devs from mean)
    for w in [10, 20]:
        data[f'zscore_{w}d'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-10)
    
    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================
    
    # Historical volatility
    for w in [5, 10, 20]:
        data[f'vol_{w}d'] = daily_ret.rolling(w).std()
    
    # Volatility regime
    data['vol_ratio'] = data['vol_5d'] / (data['vol_20d'] + 1e-10)
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(14).mean()
    data['atr_pct'] = data['atr_14'] / close
    
    # =========================================================================
    # OSCILLATORS (key for direction)
    # =========================================================================
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # RSI zones
    data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
    data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
    data['rsi_neutral'] = ((data['rsi'] >= 40) & (data['rsi'] <= 60)).astype(int)
    
    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    data['macd_cross'] = (data['macd'] > data['macd_signal']).astype(int)
    
    # Williams %R
    data['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
    
    # =========================================================================
    # PRICE PATTERN FEATURES
    # =========================================================================
    
    # Price position in range
    data['price_pos_20d'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
    
    # Distance from highs/lows
    data['dist_20d_high'] = close / high.rolling(20).max() - 1
    data['dist_20d_low'] = close / low.rolling(20).min() - 1
    
    # Bollinger Band position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    data['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Gap (overnight)
    if 'Open' in df.columns:
        data['gap'] = (df['Open'] - close.shift()) / close.shift()
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    
    data['vol_ma_ratio'] = volume / (volume.rolling(10).mean() + 1e-10)
    data['vol_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    
    # Volume-price relationship
    data['vol_price_corr'] = daily_ret.rolling(10).corr(volume.pct_change())
    
    # =========================================================================
    # TIME/CALENDAR FEATURES
    # =========================================================================
    
    data['day_of_week'] = df.index.dayofweek
    data['is_monday'] = (df.index.dayofweek == 0).astype(int)
    data['is_friday'] = (df.index.dayofweek == 4).astype(int)
    data['month'] = df.index.month
    
    # =========================================================================
    # LAG FEATURES
    # =========================================================================
    
    # Previous day's direction
    data['prev_direction'] = (daily_ret > 0).astype(int)
    data['prev_direction_2d'] = (close.pct_change(2) > 0).astype(int)
    
    return data


class DirectionPredictor:
    """Ensemble classifier for direction prediction with confidence filtering"""
    
    def __init__(self):
        # Multiple classifiers for ensemble
        self.classifiers = {
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'lr': LogisticRegression(
                C=0.1,
                max_iter=500,
                random_state=42
            )
        }
        
        # Voting ensemble
        self.ensemble = VotingClassifier(
            estimators=list(self.classifiers.items()),
            voting='soft'
        )
        
        self.scaler = RobustScaler()
        self.calibrator = None
        
    def fit(self, X, y):
        """Fit the ensemble with calibration"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit calibrated classifier for better probability estimates
        self.calibrator = CalibratedClassifierCV(self.ensemble, cv=3, method='isotonic')
        self.calibrator.fit(X_scaled, y)
        
    def predict(self, X):
        """Predict direction with confidence"""
        X_scaled = self.scaler.transform(X)
        
        # Get calibrated probabilities
        proba = self.calibrator.predict_proba(X_scaled)
        pred_class = self.calibrator.predict(X_scaled)
        
        # Confidence is the probability of the predicted class
        confidence = np.max(proba, axis=1)
        
        return pred_class, confidence, proba


def run_direction_backtest():
    """Run walk-forward backtest for direction prediction"""
    
    print("="*70)
    print("MCX DIRECTION PREDICTION ENGINE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Target: Predict UP/DOWN direction for options trading")
    print()
    
    # Fetch data
    print("1. FETCHING DATA")
    print("-"*70)
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='2y')
    print(f"   Records: {len(df)}")
    
    # Create features
    print()
    print("2. CREATING FEATURES")
    print("-"*70)
    features = create_direction_features(df)
    
    # Target: Next day direction (1 = UP, 0 = DOWN)
    next_return = df['Close'].pct_change().shift(-1)
    features['target'] = (next_return > 0).astype(int)
    
    # Drop NaN
    features_clean = features.dropna()
    print(f"   Features: {len([c for c in features_clean.columns if c != 'target'])}")
    print(f"   Samples: {len(features_clean)}")
    print(f"   Class balance: {features_clean['target'].mean()*100:.1f}% UP days")
    
    # Prepare data
    feature_cols = [c for c in features_clean.columns if c != 'target']
    X = features_clean[feature_cols].values
    y = features_clean['target'].values
    dates = features_clean.index
    
    # Walk-forward validation
    print()
    print("3. WALK-FORWARD BACKTEST")
    print("-"*70)
    
    train_window = 120
    results = []
    predictor = DirectionPredictor()
    
    for i in range(train_window, len(X) - 1):
        # Training data
        X_train = X[max(0, i-train_window):i]
        y_train = y[max(0, i-train_window):i]
        
        # Fit
        predictor.fit(X_train, y_train)
        
        # Predict
        X_test = X[i:i+1]
        pred_class, confidence, proba = predictor.predict(X_test)
        
        actual = y[i]
        
        results.append({
            'date': dates[i],
            'predicted': int(pred_class[0]),
            'actual': int(actual),
            'confidence': float(confidence[0]),
            'prob_up': float(proba[0][1]),
            'prob_down': float(proba[0][0]),
            'correct': pred_class[0] == actual
        })
        
        if len(results) % 50 == 0:
            recent = results[-50:]
            acc = np.mean([r['correct'] for r in recent]) * 100
            high_conf = [r for r in recent if r['confidence'] > 0.6]
            high_conf_acc = np.mean([r['correct'] for r in high_conf]) * 100 if high_conf else 0
            print(f"   Day {len(results):3d}: Overall={acc:.1f}% | High-Conf(>60%)={high_conf_acc:.1f}% ({len(high_conf)} trades)")
    
    results_df = pd.DataFrame(results)
    
    # Performance Analysis
    print()
    print("="*70)
    print("4. PERFORMANCE RESULTS")
    print("="*70)
    
    overall_acc = results_df['correct'].mean() * 100
    print(f"\n   📊 OVERALL METRICS:")
    print(f"      Total Predictions: {len(results_df)}")
    print(f"      Direction Accuracy: {overall_acc:.1f}%")
    
    # Confidence-filtered results
    print(f"\n   🎯 CONFIDENCE-FILTERED ACCURACY:")
    
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        filtered = results_df[results_df['confidence'] > threshold]
        if len(filtered) > 0:
            acc = filtered['correct'].mean() * 100
            pct_trades = len(filtered) / len(results_df) * 100
            print(f"      >{threshold*100:.0f}% confidence: {acc:.1f}% accuracy ({len(filtered)} trades, {pct_trades:.0f}% of days)")
    
    # Best threshold analysis
    print(f"\n   🏆 OPTIMAL TRADING STRATEGY:")
    best_threshold = 0.60
    best_acc = 0
    for t in np.arange(0.55, 0.75, 0.01):
        filtered = results_df[results_df['confidence'] > t]
        if len(filtered) >= 20:
            acc = filtered['correct'].mean() * 100
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
    
    filtered_best = results_df[results_df['confidence'] > best_threshold]
    print(f"      Best confidence threshold: {best_threshold*100:.0f}%")
    print(f"      Accuracy at this threshold: {best_acc:.1f}%")
    print(f"      Number of trades: {len(filtered_best)} ({len(filtered_best)/len(results_df)*100:.0f}% of days)")
    
    # Regime analysis
    print(f"\n   📈 BY MARKET REGIME:")
    
    # Trending vs Mean Reverting
    features_clean['ma_trend'] = (features_clean['ma5_ma20_cross'] == 1).astype(int)
    trending = results_df.merge(features_clean[['ma_trend']], left_on='date', right_index=True, how='left')
    
    trend_up = trending[trending['ma_trend'] == 1]
    trend_down = trending[trending['ma_trend'] == 0]
    
    if len(trend_up) > 0:
        print(f"      Uptrend (MA5>MA20): {trend_up['correct'].mean()*100:.1f}% ({len(trend_up)} days)")
    if len(trend_down) > 0:
        print(f"      Downtrend (MA5<MA20): {trend_down['correct'].mean()*100:.1f}% ({len(trend_down)} days)")
    
    # Last 30 days
    print(f"\n   📅 LAST 30 DAYS:")
    last30 = results_df.tail(30)
    print(f"      Overall Accuracy: {last30['correct'].mean()*100:.1f}%")
    last30_high = last30[last30['confidence'] > 0.60]
    if len(last30_high) > 0:
        print(f"      High-Conf Accuracy: {last30_high['correct'].mean()*100:.1f}% ({len(last30_high)} trades)")
    
    # Last 10 predictions
    print()
    print("="*70)
    print("5. LAST 10 PREDICTIONS")
    print("="*70)
    print(f"{'Date':<12} {'Pred':>6} {'Actual':>6} {'Conf':>7} {'Result':>8}")
    print("-"*70)
    
    for _, row in results_df.tail(10).iterrows():
        pred_str = 'UP' if row['predicted'] == 1 else 'DOWN'
        actual_str = 'UP' if row['actual'] == 1 else 'DOWN'
        result = '✓' if row['correct'] else '✗'
        conf_color = '' if row['confidence'] > 0.60 else ''
        trade_signal = '📈' if row['predicted'] == 1 else '📉'
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {trade_signal}{pred_str:>4} {actual_str:>6} {row['confidence']*100:>6.1f}% {result:>8}")
    
    # Tomorrow's Prediction
    print()
    print("="*70)
    print("6. TOMORROW'S PREDICTION")
    print("="*70)
    
    # Train on all data
    predictor.fit(X[-120:], y[-120:])
    
    # Predict using today's features
    X_today = X[-1:]
    pred_class, confidence, proba = predictor.predict(X_today)
    
    direction = 'UP 📈' if pred_class[0] == 1 else 'DOWN 📉'
    trade_action = 'BUY CALL' if pred_class[0] == 1 else 'BUY PUT'
    
    print(f"\n   Current Price: ₹{df['Close'].iloc[-1]:.2f}")
    print(f"\n   🎯 PREDICTION: {direction}")
    print(f"   📊 Confidence: {confidence[0]*100:.1f}%")
    print(f"   📈 Prob UP: {proba[0][1]*100:.1f}%")
    print(f"   📉 Prob DOWN: {proba[0][0]*100:.1f}%")
    
    if confidence[0] > 0.60:
        print(f"\n   ✅ HIGH CONFIDENCE SIGNAL!")
        print(f"   💡 Suggested Action: {trade_action}")
    else:
        print(f"\n   ⚠️ LOW CONFIDENCE - Consider waiting")
    
    # Risk assessment
    rsi = features_clean['rsi'].iloc[-1]
    zscore = features_clean['zscore_20d'].iloc[-1]
    
    print(f"\n   📋 INDICATORS:")
    print(f"      RSI: {rsi:.1f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}")
    print(f"      Z-Score: {zscore:+.2f} {'(Extended)' if abs(zscore) > 2 else ''}")
    
    print()
    print("="*70)
    print(f"RECOMMENDATION: Trade only when confidence > 60%")
    print(f"Historical accuracy at >60% confidence: ~{results_df[results_df['confidence'] > 0.60]['correct'].mean()*100:.0f}%")
    print("="*70)
    
    return results_df, predictor, features_clean


if __name__ == '__main__':
    results, predictor, features = run_direction_backtest()
