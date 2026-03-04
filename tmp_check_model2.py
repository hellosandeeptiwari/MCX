import json

d = json.load(open('/home/ubuntu/titan/agentic_trader/ml_models/saved_models/meta_labeling_latest_meta.json'))
print('Top-level keys:', list(d.keys()))
for k in d:
    v = d[k]
    if isinstance(v, dict):
        print(f'\n{k}: {list(v.keys())[:20]}')
    elif isinstance(v, list):
        print(f'\n{k}: list of {len(v)} items, first={v[0] if v else None}')
    else:
        print(f'\n{k}: {v}')
