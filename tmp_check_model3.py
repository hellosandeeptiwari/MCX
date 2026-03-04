import json

d = json.load(open('/home/ubuntu/titan/agentic_trader/ml_models/saved_models/meta_labeling_latest_meta.json'))

# dir_feature_importance is a list of names, we need the actual values
# Let's check if it's a dict or list
dfi = d.get('dir_feature_importance')
print('Type:', type(dfi))
if isinstance(dfi, dict):
    sorted_fi = sorted(dfi.items(), key=lambda x: x[1], reverse=True)[:20]
    for k, v in sorted_fi:
        print(f'  {k}: {v:.4f}')
elif isinstance(dfi, list):
    print('First 5:', dfi[:5])
    # Check if it's a list of names ordered by importance
    # Need to load the actual model to get importance values
    print('\nChecking pickled models for feature importance...')
    import pickle, numpy as np
    
    # Try loading the direction model
    import glob
    model_files = glob.glob('/home/ubuntu/titan/agentic_trader/ml_models/saved_models/*dir*.pkl')
    print('Direction model files:', model_files)
    
    model_files2 = glob.glob('/home/ubuntu/titan/agentic_trader/ml_models/saved_models/*latest*')
    print('Latest model files:', model_files2)

print('\n--- Direction model metadata ---')
dm = d.get('direction_model', {})
for k, v in dm.items():
    print(f'  {k}: {v}')

print('\n--- Direction feature names (53) ---')
for i, f in enumerate(d.get('direction_feature_names', [])):
    print(f'  {i}: {f}')

print('\n--- Samples ---')
print(f'  Train: {d.get("train_samples")}')
print(f'  Val: {d.get("val_samples")}')
print(f'  Test: {d.get("test_samples")}')

print('\n--- Combined metrics ---')
for k, v in d.get('combined', {}).items():
    print(f'  {k}: {v}')
