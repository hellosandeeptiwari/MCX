"""Train v8 "Signal Refinement" down risk detector model.

v8 improvements over v6 PROD:
  Feature Engineering (+11 features → ~99 total):
    - kaufman_er_10: Trend quality (efficiency ratio, 0=noise 1=perfect trend)
    - vol_term_ratio: Volatility term structure (BB_width_10/BB_width_20)
    - rsi_divergence: RSI vs price divergence (classic reversal signal)
    - return_skew_10, return_kurtosis_10: Multi-scale distribution stats
    - mom_exhaustion: RSI extreme × volume spike (Wyckoff exhaustion)
    - vwap_dist_accel: VWAP distance acceleration (institutional conviction)
    - roc_48: 4-hour momentum (intraday swing cycle capture)
    - eff_vol_interact: Efficiency × Volume (confirmed trend signal)
    - gap_fill_pct: Gap fill progress (mean-reversion vs continuation)
    - close_vs_range_3d: Close position in 3-day range (multi-day context)

  GBM Calibrator Enhancement (+8 dims → 38 total):
    - Per-component Mahalanobis distance (6 dims, accounts for covariance)
    - Reconstruction error entropy (1, global vs localized anomaly)
    - Cluster density (1, isolation vs between-cluster)

  Training Improvements:
    - Cosine annealing LR (finds flatter minima than ReduceLROnPlateau)
    - SWA weight averaging (averages last 30% of checkpoints)
    - Stronger weight decay (5e-5 vs 1e-5)
    - GBM: 200 trees, depth 3, L2=0.1 (richer feature space needs more capacity)
    - 120 epochs with patience 20 (longer training for cosine schedule)
    - 30-day validation window (more stable threshold calibration)

  Architecture (UNCHANGED from v6):
    - VAE: [64, 32] → 16 latent (proven optimal)
    - GMM: 6 tied-covariance (proven optimal)
    - β warmup: 0.01 → 0.5 over 25% of epochs
    - Dropout: 0.2

  Philosophy: v7 failed by adding noisy external data (estimated IV).
  v8 extracts MORE signal from EXISTING candle/volume data.
"""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
sys.argv = [
    '_train_v8.py', 'train',
    '--latent-dim', '16',
    '--vae-epochs', '120',
    '--vae-kl-weight', '0.5',
    '--val-days', '30',
    '--cosine-lr',
    '--swa',
    '--swa-start-frac', '0.7',
    '--weight-decay', '5e-5',
    '--calibrator-trees', '200',
    '--calibrator-depth', '3',
    '--calibrator-l2', '0.1',
]
from ml_models.down_risk_detector import main
main()
