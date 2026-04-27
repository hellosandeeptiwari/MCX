"""Train v7.1 down risk detector model.

v7.1 changes vs v7.0:
  - latent 20→16 (back to v6 proven value)
  - dropout 0.2→0.25
  - β (KL weight) 0.5→0.8 (stronger regularization)
  - hidden [80,40]→[64,32] (back to v6)
  - Dropped 5 noisy IV features + oi_pcr_change + oi_iv_skew + oi_atm_iv
  - Added oi_pcr_3d_ma, oi_buildup_3d_ma (OI velocity)
"""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
sys.argv = ['_train_v7.py', 'train', '--latent-dim', '16', '--vae-epochs', '100', '--vae-kl-weight', '0.8']
from ml_models.down_risk_detector import main
main()
