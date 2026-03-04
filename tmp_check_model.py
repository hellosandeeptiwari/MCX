import json

d = json.load(open('/home/ubuntu/titan/agentic_trader/ml_models/saved_models/meta_labeling_latest_meta.json'))
di = d.get('direction_model', {})

print('Direction features:', len(di.get('feature_names', [])))

fi = di.get('feature_importance', {})
sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20]
print('\nFeature importance (top 20):')
for k, v in sorted_fi:
    print(f'  {k}: {v:.4f}')

print('\nTraining info:')
for k in ['train_samples', 'class_counts', 'scale_pos_weight', 'accuracy', 'f1_macro']:
    print(f'  {k}: {di.get(k)}')

print('\nDirection feature list:')
for f in di.get('feature_names', []):
    print(f'  {f}')

# Also check NIFTY feature importance sum
nifty_imp = sum(v for k, v in fi.items() if 'nifty' in k.lower())
oi_imp = sum(v for k, v in fi.items() if 'oi' in k.lower() or 'fut_' in k.lower())
print(f'\nNIFTY features total importance: {nifty_imp:.4f}')
print(f'OI features total importance: {oi_imp:.4f}')
