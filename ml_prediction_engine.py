"""
MCX ML Prediction Engine v2.0
==============================
Target: <5% Mean Absolute Error

Advanced Data Science Techniques:
1. Rich Feature Engineering (50+ features)
2. Multiple ML Models (Random Forest, Gradient Boosting, Ridge, ElasticNet)
3. Stacked Ensemble with Meta-Learner
4. Walk-Forward Validation (proper time series CV)
5. Regime Detection (volatility-based model switching)
6. Volatility-Adjusted Predictions
7. Feature Selection (recursive feature elimination)
8. Dynamic Model Weighting based on recent performance
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    """Create 50+ technical and statistical features"""
    
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    # =========================================================================
    # PRICE-BASED FEATURES
    # =========================================================================
    
    # Returns at multiple horizons
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f'return_{lag}d'] = close.pct_change(lag)
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        data[f'ma_{window}'] = close.rolling(window).mean()
        data[f'ma_ratio_{window}'] = close / data[f'ma_{window}']
    
    # Exponential Moving Averages
    for span in [5, 10, 20]:
        data[f'ema_{span}'] = close.ewm(span=span).mean()
        data[f'ema_ratio_{span}'] = close / data[f'ema_{span}']
    
    # Price position in range
    for window in [5, 10, 20]:
        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        data[f'price_position_{window}'] = (close - roll_low) / (roll_high - roll_low + 1e-10)
    
    # Distance from highs/lows
    data['dist_from_20d_high'] = close / high.rolling(20).max() - 1
    data['dist_from_20d_low'] = close / low.rolling(20).min() - 1
    data['dist_from_52w_high'] = close / high.rolling(252).max() - 1
    data['dist_from_52w_low'] = close / low.rolling(252).min() - 1
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    # RSI at multiple periods
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Rate of Change
    for period in [5, 10, 20]:
        data[f'roc_{period}'] = close.pct_change(period) * 100
    
    # Momentum
    for period in [10, 20]:
        data[f'momentum_{period}'] = close - close.shift(period)
    
    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================
    
    # Historical Volatility
    for window in [5, 10, 20]:
        data[f'volatility_{window}'] = close.pct_change().rolling(window).std() * np.sqrt(252)
    
    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    for period in [7, 14, 21]:
        data[f'atr_{period}'] = tr.rolling(period).mean()
        data[f'atr_ratio_{period}'] = data[f'atr_{period}'] / close
    
    # Bollinger Bands
    for window in [10, 20]:
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        data[f'bb_upper_{window}'] = ma + 2 * std
        data[f'bb_lower_{window}'] = ma - 2 * std
        data[f'bb_width_{window}'] = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']) / ma
        data[f'bb_position_{window}'] = (close - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'] + 1e-10)
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    
    # Volume Moving Averages
    for window in [5, 10, 20]:
        data[f'volume_ma_{window}'] = volume.rolling(window).mean()
        data[f'volume_ratio_{window}'] = volume / (data[f'volume_ma_{window}'] + 1e-10)
    
    # On-Balance Volume
    obv = (np.sign(close.diff()) * volume).cumsum()
    data['obv'] = obv
    data['obv_ma_10'] = obv.rolling(10).mean()
    data['obv_ratio'] = obv / (data['obv_ma_10'] + 1e-10)
    
    # Volume-Price Trend
    data['vpt'] = (close.pct_change() * volume).cumsum()
    
    # =========================================================================
    # PATTERN FEATURES
    # =========================================================================
    
    # Candlestick patterns (simplified)
    data['body_size'] = (close - data['Open']) / (high - low + 1e-10) if 'Open' in data.columns else 0
    data['upper_shadow'] = (high - close.combine(data['Open'] if 'Open' in data.columns else close, max)) / (high - low + 1e-10)
    data['lower_shadow'] = (close.combine(data['Open'] if 'Open' in data.columns else close, min) - low) / (high - low + 1e-10)
    
    # Gap features
    if 'Open' in data.columns:
        data['gap'] = (data['Open'] - close.shift()) / close.shift()
    
    # Consecutive up/down days
    up_days = (close.diff() > 0).astype(int)
    data['consecutive_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
    
    down_days = (close.diff() < 0).astype(int)
    data['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()
    
    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================
    
    # Skewness and Kurtosis
    for window in [10, 20]:
        data[f'skew_{window}'] = close.pct_change().rolling(window).skew()
        data[f'kurt_{window}'] = close.pct_change().rolling(window).kurt()
    
    # Z-Score
    for window in [10, 20]:
        data[f'zscore_{window}'] = (close - close.rolling(window).mean()) / (close.rolling(window).std() + 1e-10)
    
    # Autocorrelation
    returns = close.pct_change()
    for lag in [1, 2, 5]:
        data[f'autocorr_{lag}'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False)
    
    # =========================================================================
    # TIME FEATURES
    # =========================================================================
    
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month
    data['is_month_start'] = data.index.is_month_start.astype(int)
    data['is_month_end'] = data.index.is_month_end.astype(int)
    
    # =========================================================================
    # TARGET VARIABLE
    # =========================================================================
    
    data['target'] = close.shift(-1)  # Next day's close
    data['target_return'] = close.pct_change().shift(-1)  # Next day's return
    
    return data


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(df, window=20):
    """Detect market regime: trending, mean-reverting, or volatile"""
    
    close = df['Close']
    returns = close.pct_change()
    
    # Volatility regime
    vol = returns.rolling(window).std() * np.sqrt(252)
    vol_percentile = vol.rank(pct=True)
    
    # Trend regime (ADX-like)
    ma_short = close.rolling(10).mean()
    ma_long = close.rolling(50).mean()
    trend_strength = (ma_short - ma_long).abs() / close
    
    # Mean reversion strength (Hurst exponent proxy)
    def hurst_proxy(series):
        if len(series) < 20:
            return 0.5
        lags = range(2, min(20, len(series)//2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        if len(tau) < 2:
            return 0.5
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2
    
    hurst = returns.rolling(60).apply(hurst_proxy, raw=True)
    
    regimes = pd.DataFrame(index=df.index)
    regimes['volatility'] = np.where(vol_percentile > 0.75, 'HIGH_VOL', 
                                     np.where(vol_percentile < 0.25, 'LOW_VOL', 'NORMAL_VOL'))
    regimes['trend'] = np.where(hurst > 0.55, 'TRENDING', 
                                np.where(hurst < 0.45, 'MEAN_REVERTING', 'RANDOM'))
    
    return regimes


# =============================================================================
# ML MODELS
# =============================================================================

def create_stacked_model():
    """Create a stacked ensemble model"""
    
    # Base models with different strengths
    base_models = [
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )),
        ('ridge', Ridge(alpha=1.0)),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
    ]
    
    # Meta-learner
    meta_learner = Ridge(alpha=0.5)
    
    # Stacking
    stacked = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    return stacked


def create_regime_specific_models():
    """Create models optimized for different regimes"""
    
    models = {
        'HIGH_VOL': GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,  # Shallower for noisy data
            learning_rate=0.05,
            random_state=42
        ),
        'LOW_VOL': Ridge(alpha=0.1),  # Simple model for stable periods
        'NORMAL_VOL': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        'TRENDING': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'MEAN_REVERTING': ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42)
    }
    
    return models


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(df, features, target='target', train_window=120, test_window=1):
    """
    Walk-forward validation with retraining
    - Train on past 'train_window' days
    - Predict next 'test_window' day
    - Move forward and repeat
    """
    
    results = []
    predictions = []
    
    n = len(df)
    start_idx = train_window
    
    # Prepare data
    feature_cols = [c for c in features if c in df.columns]
    X = df[feature_cols].values
    y = df[target].values
    
    # Scaler
    scaler = RobustScaler()
    
    # Model
    model = create_stacked_model()
    
    print(f"Walk-forward validation: {n - start_idx} predictions")
    print(f"Training window: {train_window} days, Predicting: {test_window} day ahead")
    
    for i in range(start_idx, n - test_window):
        # Training data
        train_start = max(0, i - train_window)
        X_train = X[train_start:i]
        y_train = y[train_start:i]
        
        # Test data
        X_test = X[i:i+test_window]
        y_test = y[i:i+test_window]
        
        # Handle NaN
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train_clean = X_train[valid_train]
        y_train_clean = y_train[valid_train]
        
        if len(X_train_clean) < 30 or np.isnan(X_test).any():
            continue
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train_clean)
        y_pred = model.predict(X_test_scaled)
        
        # Store results
        actual = y_test[0]
        predicted = y_pred[0]
        current = df['Close'].iloc[i]
        
        error_pct = abs(predicted - actual) / actual * 100
        direction_correct = (predicted > current) == (actual > current)
        
        results.append({
            'date': df.index[i],
            'current': current,
            'predicted': predicted,
            'actual': actual,
            'error_pct': error_pct,
            'direction_correct': direction_correct
        })
        
        predictions.append({
            'date': df.index[i],
            'predicted': predicted,
            'actual': actual
        })
        
        if len(results) % 20 == 0:
            recent_mae = np.mean([r['error_pct'] for r in results[-20:]])
            recent_dir = np.mean([r['direction_correct'] for r in results[-20:]]) * 100
            print(f"  Day {len(results)}: Recent MAE={recent_mae:.2f}%, Direction={recent_dir:.1f}%")
    
    return pd.DataFrame(results), pd.DataFrame(predictions)


# =============================================================================
# DYNAMIC MODEL WEIGHTING
# =============================================================================

class DynamicEnsemble:
    """Ensemble that adjusts weights based on recent performance"""
    
    def __init__(self, lookback=10):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        self.weights = {k: 0.25 for k in self.models}
        self.recent_errors = {k: [] for k in self.models}
        self.lookback = lookback
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        for name, model in self.models.items():
            model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Weighted average
        final_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            final_pred += pred * self.weights[name]
        
        return final_pred, predictions
    
    def update_weights(self, actual, predictions):
        """Update weights based on prediction errors"""
        for name, pred in predictions.items():
            error = abs(pred - actual) / actual
            self.recent_errors[name].append(error)
            if len(self.recent_errors[name]) > self.lookback:
                self.recent_errors[name].pop(0)
        
        # Inverse error weighting
        mean_errors = {k: np.mean(v) if v else 1.0 for k, v in self.recent_errors.items()}
        inv_errors = {k: 1.0 / (v + 0.01) for k, v in mean_errors.items()}
        total = sum(inv_errors.values())
        self.weights = {k: v / total for k, v in inv_errors.items()}


# =============================================================================
# ADVANCED WALK-FORWARD WITH DYNAMIC ENSEMBLE
# =============================================================================

def advanced_walk_forward(df, feature_cols, train_window=120):
    """Walk-forward with dynamic model weighting"""
    
    results = []
    ensemble = DynamicEnsemble(lookback=10)
    
    X = df[feature_cols].values
    y = df['target'].values
    
    n = len(df)
    
    print(f"\nAdvanced Walk-Forward with Dynamic Ensemble")
    print(f"{'='*70}")
    
    for i in range(train_window, n - 1):
        # Training data
        train_start = max(0, i - train_window)
        X_train = X[train_start:i]
        y_train = y[train_start:i]
        
        # Clean NaN
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train_clean = X_train[valid]
        y_train_clean = y_train[valid]
        
        if len(X_train_clean) < 50:
            continue
        
        # Test data
        X_test = X[i:i+1]
        if np.isnan(X_test).any():
            continue
        
        # Fit and predict
        ensemble.fit(X_train_clean, y_train_clean)
        pred, individual_preds = ensemble.predict(X_test)
        
        actual = y[i]
        current = df['Close'].iloc[i]
        predicted = pred[0]
        
        # Update weights
        ensemble.update_weights(actual, {k: v[0] for k, v in individual_preds.items()})
        
        error_pct = abs(predicted - actual) / actual * 100
        direction_correct = (predicted > current) == (actual > current)
        
        results.append({
            'date': df.index[i],
            'current': current,
            'predicted': predicted,
            'actual': actual,
            'error_pct': error_pct,
            'direction_correct': direction_correct,
            'weights': ensemble.weights.copy()
        })
        
        if len(results) % 30 == 0:
            recent = results[-30:]
            mae = np.mean([r['error_pct'] for r in recent])
            dir_acc = np.mean([r['direction_correct'] for r in recent]) * 100
            print(f"  Predictions: {len(results)} | Recent MAE: {mae:.2f}% | Direction: {dir_acc:.1f}%")
    
    return pd.DataFrame(results)


# =============================================================================
# VOLATILITY-ADJUSTED PREDICTIONS
# =============================================================================

def volatility_adjusted_prediction(predicted, current, volatility, confidence_level=0.68):
    """
    Adjust prediction based on volatility
    - High volatility = shrink prediction toward current price
    - Low volatility = trust prediction more
    """
    
    # Expected daily move based on volatility
    expected_move = current * (volatility / np.sqrt(252))
    
    # Raw prediction change
    pred_change = predicted - current
    
    # If prediction is beyond 2x expected daily move, shrink it
    max_reasonable_change = 2 * expected_move
    
    if abs(pred_change) > max_reasonable_change:
        shrink_factor = max_reasonable_change / abs(pred_change)
        adjusted_change = pred_change * shrink_factor
    else:
        adjusted_change = pred_change
    
    return current + adjusted_change


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_best_features(X, y, n_features=30):
    """Select best features using RFE"""
    
    # Clean data
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid]
    y_clean = y[valid]
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # RFE with Random Forest
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X_scaled, y_clean)
    
    return rfe.support_, rfe.ranking_


# =============================================================================
# MAIN ENGINE
# =============================================================================

def run_ml_engine():
    """Run the complete ML prediction engine"""
    
    print("="*70)
    print("MCX ML PREDICTION ENGINE v2.0")
    print("Target: <5% Mean Absolute Error")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # 1. Fetch data
    print("1. FETCHING DATA")
    print("-"*70)
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='2y')
    print(f"   Data points: {len(df)}")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # 2. Feature Engineering
    print()
    print("2. FEATURE ENGINEERING")
    print("-"*70)
    df_features = create_features(df)
    
    # Get feature columns (exclude target and non-feature columns)
    exclude_cols = ['target', 'target_return', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    print(f"   Created {len(feature_cols)} features")
    
    # 3. Feature Selection
    print()
    print("3. FEATURE SELECTION")
    print("-"*70)
    X = df_features[feature_cols].values
    y = df_features['target'].values
    
    # Select top 30 features
    feature_mask, rankings = select_best_features(X, y, n_features=30)
    selected_features = [f for f, m in zip(feature_cols, feature_mask) if m]
    print(f"   Selected top {len(selected_features)} features")
    
    # Show top 10 features
    feature_importance = [(f, r) for f, r in zip(feature_cols, rankings)]
    feature_importance.sort(key=lambda x: x[1])
    print("   Top 10 features:")
    for f, r in feature_importance[:10]:
        print(f"      - {f}")
    
    # 4. Regime Detection
    print()
    print("4. REGIME DETECTION")
    print("-"*70)
    regimes = detect_regime(df)
    current_vol_regime = regimes['volatility'].iloc[-1]
    current_trend_regime = regimes['trend'].iloc[-1]
    print(f"   Current Volatility Regime: {current_vol_regime}")
    print(f"   Current Trend Regime: {current_trend_regime}")
    
    # 5. Walk-Forward Validation
    print()
    print("5. WALK-FORWARD VALIDATION")
    print("-"*70)
    
    results = advanced_walk_forward(df_features, selected_features, train_window=120)
    
    # 6. Performance Metrics
    print()
    print("6. PERFORMANCE METRICS")
    print("-"*70)
    
    if len(results) > 0:
        mae = results['error_pct'].mean()
        median_error = results['error_pct'].median()
        direction_acc = results['direction_correct'].mean() * 100
        
        # Percentiles
        p90_error = results['error_pct'].quantile(0.90)
        p95_error = results['error_pct'].quantile(0.95)
        
        # Under 5% error rate
        under_5pct = (results['error_pct'] < 5).mean() * 100
        
        print(f"   Total Predictions: {len(results)}")
        print()
        print(f"   Mean Absolute Error: {mae:.2f}%")
        print(f"   Median Error: {median_error:.2f}%")
        print(f"   90th Percentile Error: {p90_error:.2f}%")
        print(f"   95th Percentile Error: {p95_error:.2f}%")
        print()
        print(f"   Direction Accuracy: {direction_acc:.1f}%")
        print(f"   Predictions under 5% error: {under_5pct:.1f}%")
        
        # Last 30 days performance
        print()
        print("   LAST 30 DAYS PERFORMANCE:")
        last_30 = results.tail(30)
        mae_30 = last_30['error_pct'].mean()
        dir_30 = last_30['direction_correct'].mean() * 100
        under_5_30 = (last_30['error_pct'] < 5).mean() * 100
        print(f"   - MAE: {mae_30:.2f}%")
        print(f"   - Direction: {dir_30:.1f}%")
        print(f"   - Under 5% error: {under_5_30:.1f}%")
    
    # 7. Tomorrow's Prediction
    print()
    print("7. TOMORROW'S PREDICTION")
    print("-"*70)
    
    # Train on all data
    X_all = df_features[selected_features].values
    y_all = df_features['target'].values
    
    # Clean
    valid = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
    X_clean = X_all[valid]
    y_clean = y_all[valid]
    
    # Fit final model
    ensemble = DynamicEnsemble(lookback=20)
    ensemble.fit(X_clean[-120:], y_clean[-120:])
    
    # Predict tomorrow
    X_today = X_all[-1:] if not np.isnan(X_all[-1]).any() else X_all[-2:-1]
    pred, individual = ensemble.predict(X_today)
    
    current_price = df['Close'].iloc[-1]
    predicted_price = pred[0]
    
    # Volatility adjustment
    vol = df['Close'].pct_change().tail(20).std() * np.sqrt(252)
    adjusted_price = volatility_adjusted_prediction(predicted_price, current_price, vol)
    
    change_pct = (adjusted_price / current_price - 1) * 100
    direction = "BULLISH 📈" if change_pct > 0 else "BEARISH 📉" if change_pct < 0 else "NEUTRAL ➡️"
    
    print(f"   Current Price: ₹{current_price:.2f}")
    print(f"   Raw Prediction: ₹{predicted_price:.2f}")
    print(f"   Vol-Adjusted: ₹{adjusted_price:.2f}")
    print(f"   Expected Change: {change_pct:+.2f}%")
    print(f"   Direction: {direction}")
    print()
    print("   Individual Model Predictions:")
    for name, pred_val in individual.items():
        chg = (pred_val[0] / current_price - 1) * 100
        print(f"      {name:12}: ₹{pred_val[0]:.0f} ({chg:+.2f}%)")
    
    print()
    print("   Model Weights (dynamic):")
    for name, weight in ensemble.weights.items():
        print(f"      {name:12}: {weight*100:.1f}%")
    
    # Confidence based on model agreement
    preds_list = [v[0] for v in individual.values()]
    pred_std = np.std(preds_list)
    pred_cv = pred_std / np.mean(preds_list)
    confidence = max(0, min(100, (1 - pred_cv * 10) * 100))
    
    print()
    print(f"   Confidence: {confidence:.0f}%")
    
    # Price range
    expected_range = current_price * vol / np.sqrt(252)
    print(f"   Likely Range: ₹{adjusted_price - expected_range:.0f} - ₹{adjusted_price + expected_range:.0f}")
    
    # 8. Detailed daily results
    print()
    print("8. LAST 10 PREDICTIONS vs ACTUALS")
    print("-"*70)
    print(f"{'Date':<12} {'Current':>10} {'Predicted':>10} {'Actual':>10} {'Error':>8} {'Dir':>6}")
    print("-"*70)
    
    for _, row in results.tail(10).iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        dir_mark = "✓" if row['direction_correct'] else "✗"
        print(f"{date_str:<12} {row['current']:>10.0f} {row['predicted']:>10.0f} {row['actual']:>10.0f} {row['error_pct']:>7.2f}% {dir_mark:>6}")
    
    return results, df_features, selected_features, ensemble


if __name__ == '__main__':
    results, df_features, features, model = run_ml_engine()
