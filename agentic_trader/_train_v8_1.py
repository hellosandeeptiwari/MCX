"""Train v8.1 "Clean Refinement" down risk detector model.

v8.1 vs v6 PROD (87 features → 98 features):
  +11 Refinement features (candle/volume only, NO OI contamination):
    - kaufman_er_10, vol_term_ratio, rsi_divergence
    - return_skew_10, return_kurtosis_10
    - mom_exhaustion, vwap_dist_accel, roc_48
    - eff_vol_interact, gap_fill_pct, close_vs_range_3d

  v8 post-mortem:
    - v8 trained with 107 features (9 toxic OI features from v7 leftovers)
    - Options OI: 43% symbol coverage, daily→5min mismatch → overfit
    - UP Test AUROC crashed 0.6445→0.5412
    - v8.1: ONLY 11 pure candle/volume features, zero OI contamination

  Conservative settings (match v6 PROD):
    - GBM: 100 trees, depth 2, no L2 (proven in v6)
    - VAE: 100 epochs, standard LR, no SWA
    - val_days=30 for stable threshold calibration
"""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
sys.argv = [
    '_train_v8_1.py', 'train',
    '--latent-dim', '16',
    '--vae-epochs', '100',
    '--vae-kl-weight', '0.5',
    '--val-days', '30',
    '--calibrator-trees', '100',
    '--calibrator-depth', '2',
    '--calibrator-l2', '0.0',
]
from ml_models.down_risk_detector import main
main()
