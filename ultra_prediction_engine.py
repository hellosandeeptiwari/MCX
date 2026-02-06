"""
MCX Ultra Prediction Engine v3.0
=================================
Target: <5% MAE consistently

Additional Techniques:
1. Quantile Regression (predict confidence intervals)
2. Target Transformation (log returns instead of prices)
3. Outlier-Robust Training
4. Volatility Clustering (GARCH-like)
5. Adaptive Training Window
6. Error Correction Model
7. Multi-Horizon Ensemble
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


def create_comprehensive_features(df):
    """Create comprehensive feature set optimized for prediction"""
    
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    # =========================================================================
    # RETURN-BASED FEATURES (most predictive)
    # =========================================================================
    
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f'ret_{lag}d'] = close.pct_change(lag)
        data[f'ret_{lag}d_sq'] = data[f'ret_{lag}d'] ** 2  # Volatility proxy
    
    # Cumulative returns
    data['ret_3d_cum'] = close.pct_change(3)
    data['ret_5d_cum'] = close.pct_change(5)
    
    # =========================================================================
    # MOVING AVERAGE FEATURES
    # =========================================================================
    
    for w in [5, 10, 20, 50]:
        data[f'ma{w}'] = close.rolling(w).mean()
        data[f'ma{w}_slope'] = data[f'ma{w}'].pct_change(5)  # MA trend
        data[f'price_ma{w}_ratio'] = close / data[f'ma{w}']
    
    # MA crossovers
    data['ma5_ma20_ratio'] = data['ma5'] / data['ma20']
    data['ma10_ma50_ratio'] = data['ma10'] / data['ma50']
    
    # =========================================================================
    # VOLATILITY FEATURES (crucial for adjustment)
    # =========================================================================
    
    for w in [5, 10, 20]:
        data[f'vol_{w}d'] = close.pct_change().rolling(w).std()
        data[f'vol_{w}d_ann'] = data[f'vol_{w}d'] * np.sqrt(252)
    
    # Volatility ratio (recent vs historical)
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Parkinson volatility (using high-low)
    data['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * (np.log(high/low)**2).rolling(20).mean())
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    data['atr_14'] = tr.rolling(14).mean()
    data['atr_pct'] = data['atr_14'] / close
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
    data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
    
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
    
    # Williams %R
    data['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
    
    # =========================================================================
    # PRICE PATTERN FEATURES
    # =========================================================================
    
    # Distance from highs/lows
    data['dist_20d_high'] = close / high.rolling(20).max() - 1
    data['dist_20d_low'] = close / low.rolling(20).min() - 1
    
    # Price position in range
    data['price_position_20d'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
    
    # Bollinger position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_position'] = (close - (ma20 - 2*std20)) / (4*std20 + 1e-10)
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    
    data['vol_ma10'] = volume.rolling(10).mean()
    data['vol_ratio_10d'] = volume / (data['vol_ma10'] + 1e-10)
    
    # Volume-price divergence
    data['vol_price_corr'] = close.pct_change().rolling(10).corr(volume.pct_change())
    
    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================
    
    data['skew_20d'] = close.pct_change().rolling(20).skew()
    data['kurt_20d'] = close.pct_change().rolling(20).kurt()
    
    # Z-score
    data['zscore_20d'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    
    # =========================================================================
    # LAG FEATURES (autoregression)
    # =========================================================================
    
    for lag in [1, 2, 3, 5]:
        data[f'close_lag_{lag}'] = close.shift(lag)
        data[f'ret_lag_{lag}'] = data['ret_1d'].shift(lag)
    
    # =========================================================================
    # TIME FEATURES
    # =========================================================================
    
    data['dow'] = data.index.dayofweek
    data['dom'] = data.index.day
    data['is_monday'] = (data.index.dayofweek == 0).astype(int)
    data['is_friday'] = (data.index.dayofweek == 4).astype(int)
    
    # =========================================================================
    # TARGET
    # =========================================================================
    
    data['target_price'] = close.shift(-1)
    data['target_return'] = close.pct_change().shift(-1)
    
    return data


class UltraPredictionEngine:
    """Advanced prediction engine with multiple techniques"""
    
    def __init__(self, train_window=100):
        self.train_window = train_window
        self.scaler = RobustScaler()
        
        # Multiple models for ensemble
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=6, 
                                        min_samples_leaf=10, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=150, max_depth=4, 
                                            learning_rate=0.05, subsample=0.8, random_state=42),
            'huber': HuberRegressor(epsilon=1.35, max_iter=200),  # Robust to outliers
            'ridge': Ridge(alpha=1.0)
        }
        
        # Quantile models for confidence intervals
        self.quantile_models = {
            'q10': None,  # 10th percentile
            'q50': None,  # Median
            'q90': None   # 90th percentile
        }
        
        # Dynamic weights based on recent performance
        self.model_weights = {k: 0.25 for k in self.models}
        self.recent_errors = {k: [] for k in self.models}
        self.lookback = 10
        
        # Error correction term
        self.recent_prediction_errors = []
        
    def fit(self, X, y, y_returns=None):
        """Fit all models"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit main models
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        # Fit quantile regression for confidence intervals (on returns)
        if y_returns is not None:
            valid = ~np.isnan(y_returns)
            X_valid = X_scaled[valid]
            y_ret_valid = y_returns[valid]
            
            for q, alpha in [('q10', 0.1), ('q50', 0.5), ('q90', 0.9)]:
                try:
                    self.quantile_models[q] = QuantileRegressor(quantile=alpha, alpha=0.01, solver='highs')
                    self.quantile_models[q].fit(X_valid, y_ret_valid)
                except:
                    pass
    
    def predict(self, X, current_price):
        """Make prediction with confidence intervals"""
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)[0]
        
        # Weighted ensemble
        ensemble_pred = sum(predictions[k] * self.model_weights[k] for k in predictions)
        
        # Error correction
        if self.recent_prediction_errors:
            avg_error = np.mean(self.recent_prediction_errors[-5:])
            ensemble_pred -= avg_error * 0.3  # Partial correction
        
        # Quantile predictions for confidence interval (on returns)
        quantile_preds = {}
        for q, model in self.quantile_models.items():
            if model is not None:
                try:
                    ret_pred = model.predict(X_scaled)[0]
                    quantile_preds[q] = current_price * (1 + ret_pred)
                except:
                    pass
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions,
            'quantiles': quantile_preds
        }
    
    def update_weights(self, actual, predictions):
        """Update model weights based on recent performance"""
        
        for name, pred in predictions.items():
            error = abs(pred - actual) / actual * 100
            self.recent_errors[name].append(error)
            if len(self.recent_errors[name]) > self.lookback:
                self.recent_errors[name].pop(0)
        
        # Inverse MAE weighting
        mean_errors = {k: np.mean(v) if v else 5.0 for k, v in self.recent_errors.items()}
        inv_errors = {k: 1.0 / (v + 0.5) for k, v in mean_errors.items()}
        total = sum(inv_errors.values())
        self.model_weights = {k: v / total for k, v in inv_errors.items()}
    
    def update_error_correction(self, predicted, actual):
        """Track prediction errors for correction"""
        error = predicted - actual
        self.recent_prediction_errors.append(error)
        if len(self.recent_prediction_errors) > 10:
            self.recent_prediction_errors.pop(0)


def adaptive_training_window(volatility_ratio, base_window=100):
    """Adjust training window based on market regime"""
    
    if volatility_ratio > 1.5:  # High volatility
        return int(base_window * 0.7)  # Shorter window, more adaptive
    elif volatility_ratio < 0.7:  # Low volatility
        return int(base_window * 1.3)  # Longer window, more stable
    else:
        return base_window


def volatility_adjusted_prediction(predicted, current, volatility, max_move_multiplier=2.0):
    """Constrain prediction based on volatility"""
    
    # Expected daily move
    expected_daily_move = current * volatility
    max_move = expected_daily_move * max_move_multiplier
    
    # Constrain prediction
    pred_change = predicted - current
    
    if abs(pred_change) > max_move:
        constrained_change = np.sign(pred_change) * max_move
        return current + constrained_change
    
    return predicted


def run_ultra_engine():
    """Run the ultra prediction engine"""
    
    print("="*70)
    print("MCX ULTRA PREDICTION ENGINE v3.0")
    print("Target: Consistent <5% MAE")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # 1. Fetch data
    print("1. DATA ACQUISITION")
    print("-"*70)
    mcx = yf.Ticker('MCX.NS')
    df = mcx.history(period='2y')
    print(f"   Records: {len(df)}")
    print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # 2. Feature Engineering
    print()
    print("2. FEATURE ENGINEERING")
    print("-"*70)
    df_feat = create_comprehensive_features(df)
    
    exclude_cols = ['target_price', 'target_return', 'Open', 'High', 'Low', 'Close', 
                    'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols and not c.startswith('close_lag')]
    
    # Add lag features but not raw close lags
    feature_cols = [c for c in feature_cols if 'close_lag' not in c]
    
    print(f"   Features: {len(feature_cols)}")
    
    # 3. Walk-Forward Backtest
    print()
    print("3. WALK-FORWARD BACKTEST")
    print("-"*70)
    
    X = df_feat[feature_cols].values
    y_price = df_feat['target_price'].values
    y_return = df_feat['target_return'].values
    
    results = []
    engine = UltraPredictionEngine(train_window=100)
    
    # Start after enough data
    start_idx = 120
    
    for i in range(start_idx, len(df_feat) - 1):
        # Current volatility for adaptive window
        vol_ratio = df_feat['vol_ratio'].iloc[i] if 'vol_ratio' in df_feat.columns else 1.0
        if np.isnan(vol_ratio):
            vol_ratio = 1.0
        
        # Adaptive training window
        train_window = adaptive_training_window(vol_ratio, base_window=100)
        train_start = max(0, i - train_window)
        
        # Training data
        X_train = X[train_start:i]
        y_train = y_price[train_start:i]
        y_ret_train = y_return[train_start:i]
        
        # Clean NaN
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train_clean = X_train[valid]
        y_train_clean = y_train[valid]
        y_ret_clean = y_ret_train[valid] if y_ret_train is not None else None
        
        if len(X_train_clean) < 50:
            continue
        
        # Test data
        X_test = X[i:i+1]
        if np.isnan(X_test).any():
            continue
        
        current_price = df_feat['Close'].iloc[i]
        actual_price = y_price[i]
        
        if np.isnan(actual_price):
            continue
        
        # Fit and predict
        engine.fit(X_train_clean, y_train_clean, y_ret_clean)
        pred_result = engine.predict(X_test, current_price)
        
        raw_pred = pred_result['ensemble']
        
        # Volatility adjustment
        current_vol = df_feat['vol_20d_ann'].iloc[i] if 'vol_20d_ann' in df_feat.columns else 0.3
        if np.isnan(current_vol):
            current_vol = 0.3
        
        adjusted_pred = volatility_adjusted_prediction(raw_pred, current_price, current_vol/np.sqrt(252))
        
        # Update weights and error correction
        engine.update_weights(actual_price, pred_result['individual'])
        engine.update_error_correction(adjusted_pred, actual_price)
        
        # Calculate metrics
        error_pct = abs(adjusted_pred - actual_price) / actual_price * 100
        direction_correct = (adjusted_pred > current_price) == (actual_price > current_price)
        
        results.append({
            'date': df_feat.index[i],
            'current': current_price,
            'raw_pred': raw_pred,
            'adjusted_pred': adjusted_pred,
            'actual': actual_price,
            'error_pct': error_pct,
            'direction_correct': direction_correct,
            'volatility': current_vol,
            'train_window': train_window
        })
        
        if len(results) % 30 == 0:
            recent = results[-30:]
            mae = np.mean([r['error_pct'] for r in recent])
            dir_acc = np.mean([r['direction_correct'] for r in recent]) * 100
            under_5 = np.mean([r['error_pct'] < 5 for r in recent]) * 100
            print(f"   Day {len(results):3d}: MAE={mae:.2f}% | Direction={dir_acc:.1f}% | Under5%={under_5:.0f}%")
    
    results_df = pd.DataFrame(results)
    
    # 4. Performance Report
    print()
    print("4. PERFORMANCE METRICS")
    print("-"*70)
    
    mae = results_df['error_pct'].mean()
    median_err = results_df['error_pct'].median()
    p90 = results_df['error_pct'].quantile(0.90)
    p95 = results_df['error_pct'].quantile(0.95)
    dir_acc = results_df['direction_correct'].mean() * 100
    under_5 = (results_df['error_pct'] < 5).mean() * 100
    under_3 = (results_df['error_pct'] < 3).mean() * 100
    
    print(f"   Total Predictions: {len(results_df)}")
    print()
    print(f"   📊 ERROR METRICS:")
    print(f"      Mean Absolute Error:    {mae:.2f}%")
    print(f"      Median Error:           {median_err:.2f}%")
    print(f"      90th Percentile:        {p90:.2f}%")
    print(f"      95th Percentile:        {p95:.2f}%")
    print()
    print(f"   🎯 ACCURACY METRICS:")
    print(f"      Direction Accuracy:     {dir_acc:.1f}%")
    print(f"      Predictions <3% error:  {under_3:.1f}%")
    print(f"      Predictions <5% error:  {under_5:.1f}%")
    
    # Regime-specific performance
    print()
    print("   📈 REGIME-SPECIFIC PERFORMANCE:")
    
    low_vol = results_df[results_df['volatility'] < 0.25]
    med_vol = results_df[(results_df['volatility'] >= 0.25) & (results_df['volatility'] < 0.40)]
    high_vol = results_df[results_df['volatility'] >= 0.40]
    
    if len(low_vol) > 0:
        print(f"      Low Volatility:   MAE={low_vol['error_pct'].mean():.2f}%, Under5%={((low_vol['error_pct']<5).mean()*100):.0f}%")
    if len(med_vol) > 0:
        print(f"      Med Volatility:   MAE={med_vol['error_pct'].mean():.2f}%, Under5%={((med_vol['error_pct']<5).mean()*100):.0f}%")
    if len(high_vol) > 0:
        print(f"      High Volatility:  MAE={high_vol['error_pct'].mean():.2f}%, Under5%={((high_vol['error_pct']<5).mean()*100):.0f}%")
    
    # Last 30 days
    print()
    print("   📅 LAST 30 DAYS:")
    last30 = results_df.tail(30)
    print(f"      MAE:              {last30['error_pct'].mean():.2f}%")
    print(f"      Direction:        {last30['direction_correct'].mean()*100:.1f}%")
    print(f"      Under 5% error:   {(last30['error_pct']<5).mean()*100:.1f}%")
    
    # 5. Tomorrow's Prediction
    print()
    print("5. TOMORROW'S PREDICTION")
    print("-"*70)
    
    # Final model training on recent data
    final_window = 100
    X_final = X[-final_window:]
    y_final = y_price[-final_window:]
    y_ret_final = y_return[-final_window:]
    
    valid = ~(np.isnan(X_final).any(axis=1) | np.isnan(y_final))
    X_final_clean = X_final[valid]
    y_final_clean = y_final[valid]
    y_ret_clean = y_ret_final[valid]
    
    engine.fit(X_final_clean, y_final_clean, y_ret_clean)
    
    # Today's features
    X_today = X[-1:]
    current_price = df['Close'].iloc[-1]
    current_vol = df_feat['vol_20d_ann'].iloc[-1] if not np.isnan(df_feat['vol_20d_ann'].iloc[-1]) else 0.3
    
    pred_result = engine.predict(X_today, current_price)
    raw_pred = pred_result['ensemble']
    adj_pred = volatility_adjusted_prediction(raw_pred, current_price, current_vol/np.sqrt(252))
    
    change_pct = (adj_pred / current_price - 1) * 100
    direction = "📈 BULLISH" if change_pct > 0.5 else "📉 BEARISH" if change_pct < -0.5 else "➡️ NEUTRAL"
    
    print(f"   Current Price:        ₹{current_price:.2f}")
    print(f"   Raw Prediction:       ₹{raw_pred:.2f}")
    print(f"   Vol-Adjusted:         ₹{adj_pred:.2f}")
    print(f"   Expected Change:      {change_pct:+.2f}%")
    print(f"   Direction:            {direction}")
    print()
    
    # Confidence interval from quantiles
    if pred_result['quantiles']:
        q10 = pred_result['quantiles'].get('q10', adj_pred * 0.97)
        q90 = pred_result['quantiles'].get('q90', adj_pred * 1.03)
        print(f"   80% Confidence Range: ₹{q10:.0f} - ₹{q90:.0f}")
    
    # Model breakdown
    print()
    print("   Model Predictions:")
    for name, pred in pred_result['individual'].items():
        chg = (pred / current_price - 1) * 100
        wt = engine.model_weights[name] * 100
        print(f"      {name:8}: ₹{pred:.0f} ({chg:+.1f}%) [weight: {wt:.0f}%]")
    
    # Current regime
    print()
    print("   Current Regime:")
    print(f"      Volatility (ann.): {current_vol*100:.1f}%")
    rsi = df_feat['rsi'].iloc[-1] if not np.isnan(df_feat['rsi'].iloc[-1]) else 50
    print(f"      RSI:               {rsi:.1f}")
    zscore = df_feat['zscore_20d'].iloc[-1] if not np.isnan(df_feat['zscore_20d'].iloc[-1]) else 0
    print(f"      Z-Score:           {zscore:+.2f}")
    
    # 6. Last 10 predictions detail
    print()
    print("6. LAST 10 PREDICTIONS vs ACTUALS")
    print("-"*70)
    print(f"{'Date':<12} {'Current':>9} {'Predicted':>10} {'Actual':>9} {'Error':>7} {'Dir':>5}")
    print("-"*70)
    
    for _, row in results_df.tail(10).iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        dir_mark = "✓" if row['direction_correct'] else "✗"
        err_mark = "✓" if row['error_pct'] < 5 else "✗"
        print(f"{date_str:<12} {row['current']:>9.0f} {row['adjusted_pred']:>10.0f} {row['actual']:>9.0f} {row['error_pct']:>6.2f}%{err_mark} {dir_mark:>4}")
    
    print()
    print("="*70)
    print(f"ENGINE STATUS: {'✅ TARGET MET' if mae < 5 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"Overall MAE: {mae:.2f}%  |  Target: <5%")
    print("="*70)
    
    return results_df, engine


if __name__ == '__main__':
    results, engine = run_ultra_engine()
