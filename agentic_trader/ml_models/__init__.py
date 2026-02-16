# Move Predictor ML Model
# Predicts: "Will this stock move >1% in next 30 min?"
# Uses: 100 days of Kite 5-min candle data × 50+ F&O stocks
#
# Pipeline:
#   python -m ml_models.run_pipeline --fetch    # Fetch candle data from Kite
#   python -m ml_models.run_pipeline --train    # Train XGBoost model
#   python -m ml_models.run_pipeline --info     # Show model metrics
#   python -m ml_models.run_pipeline --predict SBIN  # Test prediction
#
# Integration: Titan uses MovePredictor.predict() to boost/penalize stock scores
