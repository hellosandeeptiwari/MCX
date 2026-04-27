import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
from ml_models.feature_engineering import get_feature_names
names = get_feature_names()
print(f"Total features: {len(names)}")
sectors = [n for n in names if 'sector' in n]
print(f"Sector features ({len(sectors)}): {sectors}")
v8 = [n for n in names if n in ['kaufman_er_10','vol_term_ratio','rsi_divergence','return_skew_10','return_kurtosis_10','mom_exhaustion','vwap_dist_accel','roc_48','eff_vol_interact','gap_fill_pct','close_vs_range_3d']]
print(f"v8 features ({len(v8)}): {v8}")
dupes = [n for n in names if names.count(n) > 1]
print(f"Duplicates: {set(dupes) if dupes else 'None'}")
print(f"has_oi_data: {'has_oi_data' in names}")
