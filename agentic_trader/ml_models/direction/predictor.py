"""
DAILY DIRECTION PREDICTOR — Predict next-day UP/DOWN for Titan

Uses daily OHLCV candles to predict whether a stock will close UP or DOWN
the next trading day. Works independently from Model 1 (move_predictor).

Model 1: "Will this stock have a big intraday move?" (5-min data)
Model 2: "Will this stock close UP or DOWN tomorrow?" (daily data)

Both models provide independent signals to Titan.

Usage in Titan:
    from ml_models.direction.predictor import DirectionPredictor
    
    dp = DirectionPredictor()
    result = dp.predict(daily_candles_df)
    # result = {
    #   'ml_direction': 'UP',
    #   'ml_up_prob': 0.58,
    #   'ml_down_prob': 0.42,
    #   'ml_dir_confidence': 0.58,
    #   'ml_dir_signal': 'BULLISH',
    #   'ml_dir_score_boost': 3,
    # }
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

import xgboost as xgb

from .feature_engineering import compute_direction_features, get_direction_feature_names


MODELS_DIR = Path(__file__).parent / "saved_models"


class DirectionPredictor:
    """Next-day direction predictor from daily OHLCV data."""
    
    def __init__(self, model_path: Optional[str] = None, logger=None):
        self.logger = logger
        self.dir_model = None
        self.feature_names = None
        self.metadata = None
        self.ready = False
        
        # Load Direction model
        if model_path is None:
            model_path = str(MODELS_DIR / "direction_predictor_latest.json")
        
        meta_path = model_path.replace('.json', '_meta.json')
        
        if not os.path.exists(model_path):
            self._log(f"Direction model not found: {model_path}")
            self._log("Run: python -m ml_models.direction.trainer")
            return
        
        try:
            self.dir_model = xgb.XGBClassifier()
            self.dir_model.load_model(model_path)
            
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', get_direction_feature_names())
            else:
                self.feature_names = get_direction_feature_names()
            
            self.ready = True
            acc = self.metadata.get('accuracy', '?') if self.metadata else '?'
            gap = self.metadata.get('signal_gap', '?') if self.metadata else '?'
            self._log(f"DirectionPredictor loaded (accuracy={acc}, signal_gap={gap})")
        
        except Exception as e:
            self._log(f"Failed to load direction model: {e}")
    
    def predict(self, daily_df: pd.DataFrame) -> dict:
        """Predict next-day direction for a single stock.
        
        Args:
            daily_df: Daily OHLCV candles (need >= 55 rows for feature warmup)
        
        Returns:
            dict with direction prediction, or empty dict on failure
        """
        if not self.ready:
            return {}
        
        try:
            # Compute daily features
            featured = compute_direction_features(daily_df)
            if featured.empty:
                return {}
            
            # Take last row (most recent day)
            latest = featured.iloc[-1:].copy()
            
            for feat in self.feature_names:
                if feat not in latest.columns:
                    latest[feat] = 0.0
            
            X = latest[self.feature_names].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict direction
            dir_proba = self.dir_model.predict_proba(X)[0]
            down_prob = float(dir_proba[0])
            up_prob = float(dir_proba[1])
            
            direction = 'UP' if up_prob >= 0.50 else 'DOWN'
            dir_confidence = max(up_prob, down_prob)
            
            # Signal and score boost
            signal = _compute_signal(up_prob, direction, dir_confidence)
            score_boost = _compute_score_boost(up_prob, direction, dir_confidence)
            
            return {
                'ml_direction': direction,
                'ml_up_prob': round(up_prob, 4),
                'ml_down_prob': round(down_prob, 4),
                'ml_dir_confidence': round(dir_confidence, 4),
                'ml_dir_signal': signal,
                'ml_dir_score_boost': score_boost,
            }
        
        except Exception as e:
            self._log(f"Direction prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def predict_batch(self, stock_daily: dict) -> dict:
        """Predict direction for multiple stocks.
        
        Args:
            stock_daily: {symbol: daily_df} mapping
        
        Returns:
            {symbol: prediction_dict}
        """
        results = {}
        for symbol, daily_df in stock_daily.items():
            pred = self.predict(daily_df)
            if pred:
                results[symbol] = pred
        return results
    
    def get_directional_picks(self, stock_daily: dict,
                               min_confidence: float = 0.55) -> dict:
        """Get stocks with confident directional predictions.
        
        Returns:
            {'bullish': [(sym, pred), ...], 'bearish': [(sym, pred), ...]}
        """
        predictions = self.predict_batch(stock_daily)
        
        bullish = []
        bearish = []
        
        for sym, pred in predictions.items():
            if pred['ml_dir_confidence'] < min_confidence:
                continue
            
            if pred['ml_direction'] == 'UP':
                bullish.append((sym, pred))
            else:
                bearish.append((sym, pred))
        
        bullish.sort(key=lambda x: x[1]['ml_up_prob'], reverse=True)
        bearish.sort(key=lambda x: x[1]['ml_down_prob'], reverse=True)
        
        return {'bullish': bullish, 'bearish': bearish}
    
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def _compute_signal(up_prob: float, direction: str, confidence: float) -> str:
    """Convert direction probability into a named signal.
    
    Returns:
        STRONG_BULLISH, BULLISH, MILD_BULLISH,
        STRONG_BEARISH, BEARISH, MILD_BEARISH,
        NEUTRAL
    """
    strength = abs(up_prob - 0.50)  # 0 to 0.50
    
    if strength < 0.03:
        return 'NEUTRAL'
    
    if direction == 'UP':
        if strength >= 0.15:
            return 'STRONG_BULLISH'
        elif strength >= 0.08:
            return 'BULLISH'
        else:
            return 'MILD_BULLISH'
    else:
        if strength >= 0.15:
            return 'STRONG_BEARISH'
        elif strength >= 0.08:
            return 'BEARISH'
        else:
            return 'MILD_BEARISH'


def _compute_score_boost(up_prob: float, direction: str, confidence: float) -> int:
    """Compute Titan score boost from daily direction prediction.
    
    Boost range: -4 to +6
    
    Conservative: only boost when confidence is genuinely high.
    Direction alone is not enough for Titan to take trades — it should
    be combined with Model 1 (intraday move probability) and other signals.
    """
    if confidence >= 0.65:
        boost = 6 if direction == 'UP' else -4
    elif confidence >= 0.60:
        boost = 4 if direction == 'UP' else -3
    elif confidence >= 0.55:
        boost = 2 if direction == 'UP' else -1
    elif confidence >= 0.52:
        boost = 1 if direction == 'UP' else 0
    else:
        boost = 0
    
    return boost


# Singleton
_predictor_instance = None

def get_direction_predictor(logger=None) -> DirectionPredictor:
    """Get or create singleton DirectionPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DirectionPredictor(logger=logger)
    return _predictor_instance


if __name__ == '__main__':
    import sys
    
    predictor = DirectionPredictor()
    
    if predictor.ready:
        symbol = sys.argv[1] if len(sys.argv) > 1 else 'SBIN'
        
        from ml_models.data_fetcher import load_daily_candles
        daily_df = load_daily_candles(symbol)
        
        if len(daily_df) > 0:
            result = predictor.predict(daily_df)
            
            last_close = daily_df.sort_values('date').iloc[-1]['close']
            last_date = daily_df.sort_values('date').iloc[-1]['date']
            
            print(f"\n{'='*60}")
            print(f"  {symbol} NEXT-DAY DIRECTION PREDICTION")
            print(f"  Last close: Rs {last_close:.2f} ({last_date})")
            print(f"{'='*60}")
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"No daily data for {symbol}. Run: python -m ml_models.data_fetcher --daily")
    else:
        print("Direction model not ready. Train with:")
        print("  python -m ml_models.direction.trainer")
