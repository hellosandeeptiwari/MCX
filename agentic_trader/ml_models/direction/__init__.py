# Daily Direction Predictor — Model 2
# Predicts: "Will this stock close UP or DOWN tomorrow?"
#
# Architecture:
#   Model 1 (move_predictor) → "Will this stock MOVE ≥0.5% intraday?" (5-min data)
#   Model 2 (direction_predictor) → "Next-day UP or DOWN?" (daily data)
#
# Both models provide independent signals to Titan.
# Model 2 uses 35 daily technical features from OHLCV data.
# 5-min direction prediction was proven unpredictable (2 failed attempts at ~50%).
# Daily direction has real signal: 53.3% accuracy, 59% precision at high confidence.
#
# Pipeline:
#   python -m ml_models.direction.trainer           # Train direction model
#   python -m ml_models.direction.validator          # Validate
#   python -m ml_models.direction.predictor SBIN     # Test prediction
#
# Integration:
#   from ml_models.direction.predictor import DirectionPredictor
#   dp = DirectionPredictor()
#   result = dp.predict(daily_df)
#   # result = {'ml_direction': 'UP', 'ml_up_prob': 0.58, 'ml_dir_signal': 'BULLISH', ...}
