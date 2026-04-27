import json, joblib, os
os.chdir('/home/ubuntu/titan/agentic_trader')
fn = json.load(open('ml_models/feature_names.json'))
print(f'Predictor feature_names: {len(fn)} features')
try:
    s = joblib.load('ml_models/down_risk_UP_scaler.pkl')
    print(f'DR UP scaler expects: {s.n_features_in_} features')
except Exception as e:
    print(f'DR UP scaler: {e}')
try:
    s = joblib.load('ml_models/down_risk_DOWN_scaler.pkl')
    print(f'DR DOWN scaler expects: {s.n_features_in_} features')
except Exception as e:
    print(f'DR DOWN scaler: {e}')
