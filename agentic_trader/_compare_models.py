import json

old = json.load(open('ml_models/saved_models/meta_labeling_20260215_153507_meta.json'))
new = json.load(open('ml_models/saved_models/meta_labeling_20260216_210217_meta.json'))

print("METRIC                        OLD MODEL    NEW MODEL    DELTA")
print("=" * 68)
for sect, label in [('gate_model', 'Gate'), ('direction_model', 'Dir'), ('combined', 'Combined')]:
    for k in old[sect]:
        ov = old[sect][k]
        nv = new[sect].get(k, 'N/A')
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            d = nv - ov
            s = '+' if d >= 0 else ''
            name = f"{label}.{k}"
            print(f"  {name:<28} {ov:>10}    {nv:>10}    {s}{d:.4f}")
    print()

print("BOS/SWEEP FEATURE IMPORTANCES:")
for f in ['bos_signal', 'sweep_signal', 'sweep_x_wick']:
    gi = new.get('gate_feature_importance', {}).get(f, 0)
    di = new.get('dir_feature_importance', {}).get(f, 'PRUNED')
    print(f"  {f:<20} Gate={gi:.6f}  Dir={di}")

print()
print("OLD features:", old.get('n_features', old.get('gate', {}).get('best_iteration', '?')))
print("NEW features:", new.get('n_features', new.get('gate', {}).get('best_iteration', '?')))
