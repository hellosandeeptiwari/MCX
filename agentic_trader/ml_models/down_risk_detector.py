"""
DOWN-RISK DETECTOR — VAE + GMM anomaly detector for UP/FLAT regimes

Detects hidden DOWN risk when the XGBoost meta-labeling model predicts UP or FLAT.
Runs AFTER the main model, only on UP/FLAT predictions.

Architecture:
  1. Reuse existing feature pipeline (84 features from feature_engineering.py)
  2. Generate XGBoost regime predictions via cross-prediction (no leakage)
  3. Train separate VAE+GMM for UP-regime and FLAT-regime on "clean normal" samples
  4. Anomaly score = negative log-likelihood under GMM in VAE latent space
  5. Calibrate threshold on validation set → down_risk_flag

Evaluation label (NOT a training target):
  y_down8 = 1 if worst drawdown over next 8 candles > ATR × 3.0
  Used only for metric computation (AUROC, lift, precision@flag)

Training concept:
  - "Clean normal UP"  = samples where XGB predicts UP  AND y_down8=0
  - "Clean normal FLAT" = samples where XGB predicts FLAT AND y_down8=0
  - VAE learns to reconstruct these normal patterns
  - GMM in latent space measures density (low density = anomaly)
  - At inference: high anomaly score within UP/FLAT → hidden DOWN risk

Usage:
  python -m ml_models.down_risk_detector train
  python -m ml_models.down_risk_detector evaluate
  python -m ml_models.down_risk_detector infer --symbol SBIN
"""

from __future__ import annotations

import os
import json
import time
import logging
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# ── Paths ──
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "candles_5min"

# ── Logging ──
logger = logging.getLogger("down_risk_detector")

# ── Reproducibility ──
SEED = 42

def _set_seed(seed: int = SEED):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """Configuration for the down-risk detector."""
    # VAE architecture
    input_dim: int = 84           # number of features (auto-detected)
    latent_dim: int = 16          # latent space dimensionality
    hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    
    # VAE training
    vae_epochs: int = 80
    vae_batch_size: int = 512
    vae_lr: float = 1e-3
    vae_kl_weight: float = 0.5   # β in β-VAE (< 1 = more reconstruction focus)
    vae_patience: int = 12       # early stopping patience
    
    # GMM
    gmm_n_components: int = 8    # number of Gaussian components
    gmm_covariance_type: str = "full"
    gmm_max_iter: int = 300
    
    # Down event label (evaluation only)
    down_lookahead: int = 8      # candles to look ahead
    down_atr_factor: float = 3.0 # ATR multiplier for "down event" threshold (12% base rate)
    down_fixed_pct: float = 0.5  # fallback if ATR not available
    
    # Threshold calibration
    target_flag_rate: float = 0.15  # flag ~15% of UP/FLAT as risky
    min_lift: float = 1.05          # minimum lift to accept threshold
    
    # Data
    test_days: int = 20
    val_days: int = 10
    
    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
#  VAE MODEL (PyTorch)
# ══════════════════════════════════════════════════════════════════════

class VAE(nn.Module):
    """Variational Autoencoder for learning latent representations."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ── Encoder ──
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        
        # Latent space: mu and log_var
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # ── Decoder ──
        dec_layers = []
        dec_hidden = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in dec_hidden:
            dec_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> np.ndarray:
        """Get latent mu (deterministic) for GMM fitting."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu.cpu().numpy()


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 0.5) -> Tuple[torch.Tensor, float, float]:
    """VAE loss = Reconstruction + β × KL divergence."""
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + kl_weight * kl
    return total, recon.item(), kl.item()


# ══════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════

def _create_down_label(df: pd.DataFrame, lookahead: int = 8,
                       atr_factor: float = 3.0,
                       fixed_pct: float = 0.5) -> np.ndarray:
    """Create y_down8: 1 if worst drawdown in next `lookahead` candles breaches threshold.
    
    Uses ATR-normalized threshold if 'atr_pct' column exists, else fixed %.
    
    No leakage: uses only the label creation mechanism (future price for labels only).
    """
    n = len(df)
    y_down = np.zeros(n, dtype=np.int32)
    close = df['close'].values
    low = df['low'].values if 'low' in df.columns else close
    
    # ATR-based threshold (per-candle, adaptive)
    has_atr = 'atr_pct' in df.columns
    if has_atr:
        atr_pct = df['atr_pct'].values
        threshold_pct = atr_factor * atr_pct  # array
        threshold_pct = np.where(np.isnan(threshold_pct), fixed_pct, threshold_pct)
        threshold_pct = np.maximum(threshold_pct, 0.10)  # floor at 0.1%
    else:
        threshold_pct = np.full(n, fixed_pct)
    
    for i in range(n - lookahead):
        # Worst drawdown: (min low in next 8 candles - current close) / current close
        future_lows = low[i + 1: i + 1 + lookahead]
        if close[i] > 0:
            max_down = (future_lows.min() - close[i]) / close[i] * 100  # negative %
            if max_down <= -threshold_pct[i]:
                y_down[i] = 1
    
    # Last `lookahead` candles: mark as NaN (insufficient data) → we'll filter them
    y_down[n - lookahead:] = -1  # sentinel for "unknown"
    
    return y_down


def _generate_xgb_cross_predictions(
    X: np.ndarray, y_gate: np.ndarray, y_dir: np.ndarray,
    dates: np.ndarray, feature_names: list,
    n_folds: int = 5,
) -> np.ndarray:
    """Generate XGBoost regime predictions via time-series cross-prediction.
    
    Splits data into `n_folds` chronological folds. For each fold, trains on
    all prior folds and predicts the current fold. Returns predicted class
    for every sample (no leakage).
    
    Returns: array of predicted regime: 'UP', 'DOWN', 'FLAT' for each row.
    """
    import xgboost as xgb
    
    n = len(X)
    predictions = np.full(n, '', dtype=object)
    
    # Sort by date and create fold indices
    sorted_idx = np.argsort(dates)
    fold_size = n // n_folds
    
    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
        
        test_idx = sorted_idx[fold_start:fold_end]
        
        if fold == 0:
            # First fold: no training data before it — use a simple heuristic
            # or train on itself (won't be used for final eval anyway since it's
            # the earliest data and our eval is on the last test_days)
            # We'll assign 'FLAT' as default for the first fold
            predictions[test_idx] = 'FLAT'
            continue
        
        train_idx = sorted_idx[:fold_start]
        
        X_tr, X_te = X[train_idx], X[test_idx]
        
        # Gate model: MOVE vs FLAT
        y_gate_tr = y_gate[train_idx]
        gate = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=5, learning_rate=0.05,
            n_estimators=500, min_child_weight=15, subsample=0.75,
            colsample_bytree=0.8, gamma=1.5, reg_lambda=2.0,
            random_state=SEED, verbosity=0,
            scale_pos_weight=(y_gate_tr == 0).sum() / max((y_gate_tr == 1).sum(), 1),
        )
        gate.fit(X_tr, y_gate_tr, verbose=False)
        gate_pred = (gate.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
        
        # Direction model: UP vs DOWN (only on MOVE samples)
        move_mask_tr = y_gate_tr == 1
        y_dir_tr_move = y_dir[train_idx][move_mask_tr]
        X_tr_move = X_tr[move_mask_tr]
        
        if len(X_tr_move) > 50 and len(np.unique(y_dir_tr_move)) > 1:
            direction = xgb.XGBClassifier(
                objective='binary:logistic', max_depth=6, learning_rate=0.03,
                n_estimators=800, min_child_weight=18, subsample=0.72,
                colsample_bytree=0.75, gamma=1.0, reg_lambda=2.0,
                random_state=SEED, verbosity=0,
            )
            direction.fit(X_tr_move, y_dir_tr_move, verbose=False)
            dir_proba = direction.predict_proba(X_te)[:, 1]  # P(UP|MOVE)
        else:
            dir_proba = np.full(len(X_te), 0.5)
        
        # Combine: FLAT if gate=0, else UP/DOWN based on direction
        for i, idx in enumerate(test_idx):
            if gate_pred[i] == 0:
                predictions[idx] = 'FLAT'
            elif dir_proba[i] >= 0.5:
                predictions[idx] = 'UP'
            else:
                predictions[idx] = 'DOWN'
    
    return predictions


def load_detector_dataset(
    config: DetectorConfig,
    symbols: Optional[list] = None,
) -> Tuple[pd.DataFrame, list]:
    """Load and prepare the full dataset with features, labels, and XGB predictions.
    
    Returns:
        (df, feature_names) where df has columns:
        - All original features
        - 'y_down8': binary down-event label (for evaluation)
        - 'xgb_regime': cross-predicted regime (UP/DOWN/FLAT)
        - 'date', 'symbol'
    """
    from .trainer import load_and_prepare_data, LABEL_MAP
    from .feature_engineering import get_feature_names
    
    logger.info("Loading base dataset...")
    train_df, val_df, test_df, feature_names = load_and_prepare_data(
        symbols=symbols,
        atr_factor=1.5,  # match production gate labeling
        test_days=config.test_days,
        val_days=config.val_days,
        label_method='first_to_break',
    )
    
    # Recombine for cross-prediction (we'll re-split later)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined):,} rows, {len(feature_names)} features")
    
    # ── Create y_down8 label (evaluation only) ──
    logger.info("Creating y_down8 labels (per-symbol, ATR-normalized)...")
    y_down_all = np.zeros(len(combined), dtype=np.int32)
    
    for sym, grp in combined.groupby('symbol'):
        idx = grp.index
        y_down = _create_down_label(
            grp, lookahead=config.down_lookahead,
            atr_factor=config.down_atr_factor,
            fixed_pct=config.down_fixed_pct,
        )
        y_down_all[idx] = y_down
    
    combined['y_down8'] = y_down_all
    
    # Filter out boundary rows (y_down8 == -1)
    before = len(combined)
    combined = combined[combined['y_down8'] >= 0].reset_index(drop=True)
    logger.info(f"Removed {before - len(combined)} boundary rows → {len(combined):,} remaining")
    
    # ── Generate XGB cross-predictions (no leakage) ──
    logger.info("Generating XGB cross-predictions (5-fold chronological)...")
    
    X = combined[feature_names].values.astype(np.float32)
    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gate labels: MOVE(1) vs FLAT(0)
    y_gate = (combined['label_idx'] != 1).astype(int).values  # label_idx=1 is FLAT
    # Direction labels: UP(1) vs DOWN(0)
    y_dir = (combined['label_idx'] == 2).astype(int).values  # label_idx=2 is UP
    
    dates = combined['date'].values
    
    xgb_regime = _generate_xgb_cross_predictions(
        X, y_gate, y_dir, dates, feature_names, n_folds=5
    )
    combined['xgb_regime'] = xgb_regime
    
    # Stats
    regime_counts = combined['xgb_regime'].value_counts()
    down_rate = combined['y_down8'].mean()
    logger.info(f"XGB regime distribution: {regime_counts.to_dict()}")
    logger.info(f"Overall y_down8 rate: {down_rate:.3f}")
    
    for regime in ['UP', 'FLAT', 'DOWN']:
        mask = combined['xgb_regime'] == regime
        if mask.sum() > 0:
            rate = combined.loc[mask, 'y_down8'].mean()
            logger.info(f"  y_down8 rate within {regime}: {rate:.3f} ({mask.sum():,} samples)")
    
    return combined, feature_names


# ══════════════════════════════════════════════════════════════════════
#  REGIME DETECTOR (one per regime: UP or FLAT)
# ══════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """VAE + GMM anomaly detector for a single regime (UP or FLAT).
    
    Hybrid scoring: combines 3 signals via logistic regression:
      1. GMM negative log-likelihood in VAE latent space
      2. VAE reconstruction MSE
      3. VAE KL divergence per sample
    """
    
    def __init__(self, regime: str, config: DetectorConfig):
        assert regime in ('UP', 'FLAT'), f"Invalid regime: {regime}"
        self.regime = regime
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.vae: Optional[VAE] = None
        self.gmm: Optional[GaussianMixture] = None
        self.lr_calibrator: Optional[LogisticRegression] = None  # hybrid calibrator
        self.threshold: float = 0.0
        self.base_down_rate: float = 0.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_trained = False
    
    def train(self, X_normal: np.ndarray, X_val_all: np.ndarray,
              y_val_down: np.ndarray) -> dict:
        """Train VAE + GMM on "clean normal" samples for this regime.
        
        Args:
            X_normal: Feature matrix of clean normal samples (y_down8=0 within regime)
            X_val_all: ALL validation samples for this regime (for threshold calibration)
            y_val_down: y_down8 for all validation samples
            
        Returns:
            dict with training metrics
        """
        _set_seed()
        metrics = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training {self.regime} Regime Detector")
        logger.info(f"  Normal train samples: {len(X_normal):,}")
        logger.info(f"  Validation samples: {len(X_val_all):,} (down rate: {y_val_down.mean():.3f})")
        logger.info(f"{'='*60}")
        
        # ── 1. Fit scaler on normal training data ──
        self.scaler = StandardScaler()
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        X_val_scaled = self.scaler.transform(X_val_all)
        
        input_dim = X_normal_scaled.shape[1]
        actual_config = DetectorConfig(**{**self.config.to_dict(), 'input_dim': input_dim})
        
        # ── 2. Train VAE ──
        self.vae = VAE(
            input_dim=input_dim,
            latent_dim=actual_config.latent_dim,
            hidden_dims=actual_config.hidden_dims,
            dropout=actual_config.dropout,
        ).to(self.device)
        
        vae_metrics = self._train_vae(X_normal_scaled, X_val_scaled, actual_config)
        metrics['vae'] = vae_metrics
        
        # ── 3. Extract latent codes (from normal train set) ──
        X_normal_t = torch.tensor(X_normal_scaled, dtype=torch.float32).to(self.device)
        latent_train = self.vae.get_latent(X_normal_t)
        
        # ── 4. Fit GMM in latent space ──
        logger.info(f"  Fitting GMM ({actual_config.gmm_n_components} components) in latent space...")
        self.gmm = GaussianMixture(
            n_components=actual_config.gmm_n_components,
            covariance_type=actual_config.gmm_covariance_type,
            max_iter=actual_config.gmm_max_iter,
            random_state=SEED,
            n_init=3,
        )
        self.gmm.fit(latent_train)
        logger.info(f"  GMM converged: {self.gmm.converged_}")
        
        # ── 5. Compute raw anomaly features on validation set ──
        val_raw = self._raw_anomaly_features(X_val_all)
        
        # ── 6. Fit logistic calibrator on validation set (supervised hybrid) ──
        logger.info(f"  Fitting logistic calibrator on {len(y_val_down)} val samples...")
        self.lr_calibrator = LogisticRegression(
            C=1.0, max_iter=500, random_state=SEED, solver='lbfgs'
        )
        self.lr_calibrator.fit(val_raw, y_val_down)
        
        # Use LR probability as the hybrid anomaly score
        val_scores = self.score(X_val_all)
        
        # ── 7. Calibrate threshold ──
        self._calibrate_threshold(val_scores, y_val_down)
        metrics['threshold'] = self.threshold
        metrics['base_down_rate'] = self.base_down_rate
        
        # Eval on validation
        val_eval = self._evaluate(val_scores, y_val_down, prefix='val')
        metrics.update(val_eval)
        
        self._is_trained = True
        return metrics
    
    def _train_vae(self, X_train: np.ndarray, X_val: np.ndarray,
                   config: DetectorConfig) -> dict:
        """Train the VAE with early stopping."""
        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        X_va_t = torch.tensor(X_val, dtype=torch.float32)
        
        train_ds = TensorDataset(X_tr_t)
        train_loader = DataLoader(train_ds, batch_size=config.vae_batch_size,
                                  shuffle=True, drop_last=False)
        
        optimizer = optim.AdamW(self.vae.parameters(), lr=config.vae_lr,
                                weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"  Training VAE: {config.vae_epochs} epochs, batch={config.vae_batch_size}")
        
        for epoch in range(config.vae_epochs):
            # ── Train ──
            self.vae.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                x_hat, mu, logvar = self.vae(batch)
                loss, recon, kl = vae_loss(batch, x_hat, mu, logvar, config.vae_kl_weight)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train = epoch_loss / max(n_batches, 1)
            
            # ── Validate ──
            self.vae.eval()
            with torch.no_grad():
                X_va_dev = X_va_t.to(self.device)
                x_hat_v, mu_v, lv_v = self.vae(X_va_dev)
                val_loss, _, _ = vae_loss(X_va_dev, x_hat_v, mu_v, lv_v, config.vae_kl_weight)
                avg_val = val_loss.item()
            
            scheduler.step(avg_val)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            
            if epoch % 10 == 0 or epoch == config.vae_epochs - 1:
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(f"    Epoch {epoch:3d}: train={avg_train:.4f}  val={avg_val:.4f}  lr={lr_now:.2e}")
            
            if avg_val < best_val_loss - 1e-5:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.vae.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.vae_patience:
                    logger.info(f"    Early stopping at epoch {epoch} (patience={config.vae_patience})")
                    break
        
        # Restore best weights
        if best_state is not None:
            self.vae.load_state_dict(best_state)
            self.vae.to(self.device)
        
        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch,
            'train_loss_final': history['train_loss'][-1],
        }
    
    def _raw_anomaly_features(self, X: np.ndarray) -> np.ndarray:
        """Compute raw anomaly feature matrix (3 columns) for hybrid scoring.
        
        Features:
          [0] GMM negative log-likelihood in latent space
          [1] VAE per-sample reconstruction MSE
          [2] VAE per-sample KL divergence
        """
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(X_t)
            z = mu  # deterministic for scoring
            x_hat = self.vae.decode(z)
            
            # Per-sample reconstruction error (MSE)
            recon_mse = ((X_t - x_hat) ** 2).mean(dim=1).cpu().numpy()
            
            # Per-sample KL divergence
            kl_per_sample = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).cpu().numpy()
        
        latent = mu.cpu().numpy()
        
        # GMM negative log-likelihood
        gmm_nll = -self.gmm.score_samples(latent)
        
        # Stack into feature matrix [n_samples, 3]
        return np.column_stack([gmm_nll, recon_mse, kl_per_sample])
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute hybrid anomaly scores for raw (unscaled) feature matrix.
        
        Returns: array of anomaly scores (higher = more anomalous).
        Hybrid: LR calibrated probability from [GMM NLL, recon MSE, KL divergence].
        Falls back to GMM NLL if calibrator not fitted yet.
        """
        if not self._is_trained and self.scaler is None:
            raise RuntimeError("Detector not trained. Call train() first.")
        
        raw_feats = self._raw_anomaly_features(X)
        
        if self.lr_calibrator is not None:
            # Hybrid score: calibrated probability of DOWN event
            return self.lr_calibrator.predict_proba(raw_feats)[:, 1]
        else:
            # Fallback: just GMM NLL
            return raw_feats[:, 0]
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Full prediction: anomaly scores, flags, and confidence buckets.
        
        Returns dict with:
            'anomaly_score': float array (calibrated probability, 0-1)
            'anomaly_flag': bool array (True = flagged as risky)
            'down_risk_flag': bool array (alias for anomaly_flag)
            'confidence_bucket': str array ('LOW', 'MED', 'HIGH')
        """
        scores = self.score(X)
        flags = scores >= self.threshold
        
        # Confidence buckets based on score (probability) ranges
        buckets = np.full(len(scores), 'LOW', dtype=object)
        if self.threshold > 0:
            # Scores are probabilities (0-1). Use simple fixed thresholds.
            buckets[scores >= self.threshold] = 'MED'
            # HIGH: top quartile of flagged probability range
            high_thresh = self.threshold + 0.75 * (1.0 - self.threshold)
            buckets[scores >= high_thresh] = 'HIGH'
        
        return {
            'anomaly_score': scores,
            'anomaly_flag': flags,
            'down_risk_flag': flags,
            'confidence_bucket': buckets,
        }
    
    def _calibrate_threshold(self, val_scores: np.ndarray, y_val_down: np.ndarray):
        """Find threshold with best lift of down_rate in flagged vs unflagged."""
        self.base_down_rate = y_val_down.mean()
        
        # Try percentile-based thresholds (flag top X%)
        best_threshold = np.percentile(val_scores, 100 * (1 - self.config.target_flag_rate))
        best_lift = 0.0
        
        for pct in np.arange(0.05, 0.50, 0.02):
            thresh = np.percentile(val_scores, 100 * (1 - pct))
            flagged = val_scores >= thresh
            if flagged.sum() < 10:
                continue
            
            down_if_flagged = y_val_down[flagged].mean()
            down_if_not = y_val_down[~flagged].mean() if (~flagged).sum() > 0 else 0
            lift = down_if_flagged / max(self.base_down_rate, 0.001)
            
            # Prefer threshold that maximizes lift while flagging a reasonable %
            score = lift * min(pct / self.config.target_flag_rate, 1.0)
            if score > best_lift:
                best_lift = score
                best_threshold = thresh
        
        self.threshold = float(best_threshold)
        flagged = val_scores >= self.threshold
        flag_rate = flagged.mean()
        down_if_flagged = y_val_down[flagged].mean() if flagged.sum() > 0 else 0
        lift = down_if_flagged / max(self.base_down_rate, 0.001)
        
        logger.info(f"  Calibrated threshold: {self.threshold:.3f}")
        logger.info(f"  Flag rate: {flag_rate:.1%}, Down-if-flagged: {down_if_flagged:.3f}, Lift: {lift:.2f}x")
    
    def _evaluate(self, scores: np.ndarray, y_down: np.ndarray,
                  prefix: str = '') -> dict:
        """Compute evaluation metrics."""
        metrics = {}
        p = f"{prefix}_" if prefix else ""
        
        # AUROC
        if len(np.unique(y_down)) > 1:
            auroc = roc_auc_score(y_down, scores)
            ap = average_precision_score(y_down, scores)
            metrics[f'{p}auroc'] = round(auroc, 4)
            metrics[f'{p}avg_precision'] = round(ap, 4)
            logger.info(f"  {prefix.upper()} AUROC: {auroc:.4f}, Avg Precision: {ap:.4f}")
        else:
            logger.warning(f"  {prefix}: only one class in y_down, skipping AUROC")
            metrics[f'{p}auroc'] = None
        
        # Flag-based metrics
        flagged = scores >= self.threshold
        if flagged.sum() > 0:
            flag_rate = flagged.mean()
            down_if_flagged = y_down[flagged].mean()
            down_if_not = y_down[~flagged].mean() if (~flagged).sum() > 0 else 0
            lift = down_if_flagged / max(y_down.mean(), 0.001)
            risk_reduction = 1 - (down_if_not / max(y_down.mean(), 0.001))
            
            metrics[f'{p}flag_rate'] = round(flag_rate, 4)
            metrics[f'{p}down_if_flagged'] = round(down_if_flagged, 4)
            metrics[f'{p}down_if_not_flagged'] = round(down_if_not, 4)
            metrics[f'{p}lift'] = round(lift, 2)
            metrics[f'{p}risk_reduction_pct'] = round(risk_reduction * 100, 1)
            
            logger.info(f"  {prefix.upper()} Flag rate: {flag_rate:.1%}")
            logger.info(f"  {prefix.upper()} Down if flagged: {down_if_flagged:.3f} vs base {y_down.mean():.3f} → lift {lift:.2f}x")
            logger.info(f"  {prefix.upper()} Down if NOT flagged: {down_if_not:.3f} (risk reduction: {risk_reduction*100:.1f}%)")
        
        # Decile analysis
        if len(scores) >= 100:
            deciles = pd.qcut(scores, 10, labels=False, duplicates='drop')
            decile_df = pd.DataFrame({'decile': deciles, 'y_down': y_down})
            decile_stats = decile_df.groupby('decile')['y_down'].agg(['mean', 'count'])
            logger.info(f"  {prefix.upper()} Decile analysis (0=least anomalous, 9=most):")
            for d, row in decile_stats.iterrows():
                logger.info(f"    Decile {d}: down_rate={row['mean']:.3f} (n={int(row['count'])})")
            metrics[f'{p}decile_down_rates'] = decile_stats['mean'].tolist()
        
        return metrics
    
    def save(self, path: Path):
        """Save all artifacts (scaler, VAE weights, GMM, LR calibrator, threshold)."""
        path.mkdir(parents=True, exist_ok=True)
        prefix = f"down_risk_{self.regime.lower()}"
        
        # Scaler
        with open(path / f"{prefix}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # VAE weights
        torch.save(self.vae.state_dict(), path / f"{prefix}_vae.pt")
        
        # VAE config (for reconstruction)
        vae_config = {
            'input_dim': self.vae.input_dim,
            'latent_dim': self.vae.latent_dim,
            'hidden_dims': self.config.hidden_dims,
            'dropout': self.config.dropout,
        }
        with open(path / f"{prefix}_vae_config.json", 'w') as f:
            json.dump(vae_config, f, indent=2)
        
        # GMM
        with open(path / f"{prefix}_gmm.pkl", 'wb') as f:
            pickle.dump(self.gmm, f)
        
        # LR calibrator (hybrid scoring)
        if self.lr_calibrator is not None:
            with open(path / f"{prefix}_lr_calibrator.pkl", 'wb') as f:
                pickle.dump(self.lr_calibrator, f)
        
        # Threshold and metadata
        meta = {
            'regime': self.regime,
            'threshold': self.threshold,
            'base_down_rate': self.base_down_rate,
            'config': self.config.to_dict(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(path / f"{prefix}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"  Saved {self.regime} detector to {path}")
    
    def load(self, path: Path) -> bool:
        """Load all artifacts. Returns True if successful."""
        prefix = f"down_risk_{self.regime.lower()}"
        
        try:
            # Scaler
            with open(path / f"{prefix}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # VAE config
            with open(path / f"{prefix}_vae_config.json", 'r') as f:
                vae_config = json.load(f)
            
            # VAE model
            self.vae = VAE(
                input_dim=vae_config['input_dim'],
                latent_dim=vae_config['latent_dim'],
                hidden_dims=vae_config['hidden_dims'],
                dropout=vae_config.get('dropout', 0.1),
            ).to(self.device)
            self.vae.load_state_dict(torch.load(
                path / f"{prefix}_vae.pt", map_location=self.device, weights_only=True
            ))
            self.vae.eval()
            
            # GMM
            with open(path / f"{prefix}_gmm.pkl", 'rb') as f:
                self.gmm = pickle.load(f)
            
            # LR calibrator (optional, for hybrid scoring)
            lr_path = path / f"{prefix}_lr_calibrator.pkl"
            if lr_path.exists():
                with open(lr_path, 'rb') as f:
                    self.lr_calibrator = pickle.load(f)
            
            # Metadata
            with open(path / f"{prefix}_meta.json", 'r') as f:
                meta = json.load(f)
            self.threshold = meta['threshold']
            self.base_down_rate = meta.get('base_down_rate', 0.0)
            
            self._is_trained = True
            logger.info(f"  Loaded {self.regime} detector from {path}")
            return True
            
        except Exception as e:
            logger.error(f"  Failed to load {self.regime} detector: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════
#  UNIFIED DETECTOR (wraps UP + FLAT regime detectors)
# ══════════════════════════════════════════════════════════════════════

class DownRiskDetector:
    """Unified down-risk detector that dispatches to UP or FLAT regime models."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.up_detector = RegimeDetector('UP', self.config)
        self.flat_detector = RegimeDetector('FLAT', self.config)
        self._feature_names: list = []
    
    def train(self, symbols: Optional[list] = None) -> dict:
        """Full training pipeline: load data, train both regime detectors."""
        _set_seed()
        t0 = time.time()
        all_metrics = {'config': self.config.to_dict()}
        
        # ── Load dataset with cross-predicted XGB regimes ──
        combined, feature_names = load_detector_dataset(self.config, symbols)
        self._feature_names = feature_names
        all_metrics['n_total'] = len(combined)
        all_metrics['n_features'] = len(feature_names)
        
        # ── Time-based split ──
        combined['_date'] = combined['date'].dt.date
        all_dates = sorted(combined['_date'].unique())
        
        test_cutoff = all_dates[-self.config.test_days]
        val_cutoff = all_dates[-(self.config.test_days + self.config.val_days)]
        
        train_mask = combined['_date'] < val_cutoff
        val_mask = (combined['_date'] >= val_cutoff) & (combined['_date'] < test_cutoff)
        test_mask = combined['_date'] >= test_cutoff
        
        logger.info(f"\nSplit: train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,}")
        logger.info(f"  Train: up to {val_cutoff}")
        logger.info(f"  Val:   {val_cutoff} to {test_cutoff}")
        logger.info(f"  Test:  {test_cutoff} onwards")
        
        X_all = np.nan_to_num(combined[feature_names].values.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        y_down_all = combined['y_down8'].values
        regime_all = combined['xgb_regime'].values
        
        # ── Train each regime detector ──
        for regime, detector in [('UP', self.up_detector), ('FLAT', self.flat_detector)]:
            # Training: clean normal = regime predicted AND y_down8=0
            train_regime = train_mask & (combined['xgb_regime'] == regime)
            train_normal = train_regime & (combined['y_down8'] == 0)
            
            # Validation: all samples in regime (both normal and down)
            val_regime = val_mask & (combined['xgb_regime'] == regime)
            
            n_regime_train = train_regime.sum()
            n_normal = train_normal.sum()
            n_val = val_regime.sum()
            
            logger.info(f"\n{regime} regime: {n_regime_train:,} train total, {n_normal:,} clean normal, {n_val:,} val")
            
            if n_normal < 100 or n_val < 50:
                logger.warning(f"  Insufficient data for {regime} detector, skipping")
                all_metrics[f'{regime}_skipped'] = True
                continue
            
            X_normal = X_all[train_normal.values]
            X_val_regime = X_all[val_regime.values]
            y_val_down = y_down_all[val_regime.values]
            
            regime_metrics = detector.train(X_normal, X_val_regime, y_val_down)
            all_metrics[f'{regime}_metrics'] = regime_metrics
        
        # ── Evaluate on test set ──
        logger.info(f"\n{'='*60}")
        logger.info(f"  TEST SET EVALUATION")
        logger.info(f"{'='*60}")
        
        test_eval = self._evaluate_test(
            X_all[test_mask.values],
            y_down_all[test_mask.values],
            regime_all[test_mask.values],
        )
        all_metrics['test'] = test_eval
        
        # ── Save artifacts ──
        self.up_detector.save(MODELS_DIR)
        self.flat_detector.save(MODELS_DIR)
        
        # Save feature names
        with open(MODELS_DIR / "down_risk_feature_names.json", 'w') as f:
            json.dump(self._feature_names, f)
        
        # Save report
        elapsed = time.time() - t0
        all_metrics['training_time_sec'] = round(elapsed, 1)
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        report_path = REPORTS_DIR / f"down_risk_report_{ts}.json"
        with open(report_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        logger.info(f"\nTraining complete in {elapsed:.0f}s. Report: {report_path}")
        
        return all_metrics
    
    def _evaluate_test(self, X_test: np.ndarray, y_test_down: np.ndarray,
                       regime_test: np.ndarray) -> dict:
        """Evaluate both detectors on the test set."""
        results = {}
        
        for regime, detector in [('UP', self.up_detector), ('FLAT', self.flat_detector)]:
            mask = regime_test == regime
            if mask.sum() < 20 or not detector._is_trained:
                continue
            
            X_r = X_test[mask]
            y_r = y_test_down[mask]
            scores = detector.score(X_r)
            
            regime_eval = detector._evaluate(scores, y_r, prefix=f'test_{regime.lower()}')
            results[regime] = regime_eval
        
        # Combined: unified score for all UP+FLAT
        up_flat_mask = (regime_test == 'UP') | (regime_test == 'FLAT')
        if up_flat_mask.sum() > 50:
            X_uf = X_test[up_flat_mask]
            y_uf = y_test_down[up_flat_mask]
            regimes_uf = regime_test[up_flat_mask]
            
            scores_uf = np.zeros(len(X_uf))
            for i in range(len(X_uf)):
                if regimes_uf[i] == 'UP' and self.up_detector._is_trained:
                    scores_uf[i] = self.up_detector.score(X_uf[i:i+1])[0]
                elif regimes_uf[i] == 'FLAT' and self.flat_detector._is_trained:
                    scores_uf[i] = self.flat_detector.score(X_uf[i:i+1])[0]
            
            if len(np.unique(y_uf)) > 1:
                combined_auroc = roc_auc_score(y_uf, scores_uf)
                results['combined_auroc'] = round(combined_auroc, 4)
                logger.info(f"\n  COMBINED TEST AUROC (UP+FLAT): {combined_auroc:.4f}")
            
            # Flag stats
            flags = np.zeros(len(scores_uf), dtype=bool)
            for i in range(len(scores_uf)):
                if regimes_uf[i] == 'UP' and self.up_detector._is_trained:
                    flags[i] = scores_uf[i] >= self.up_detector.threshold
                elif regimes_uf[i] == 'FLAT' and self.flat_detector._is_trained:
                    flags[i] = scores_uf[i] >= self.flat_detector.threshold
            
            if flags.sum() > 0:
                base = y_uf.mean()
                flagged_rate = y_uf[flags].mean()
                clean_rate = y_uf[~flags].mean() if (~flags).sum() > 0 else 0
                
                results['combined_flag_rate'] = round(flags.mean(), 4)
                results['combined_down_if_flagged'] = round(flagged_rate, 4)
                results['combined_down_if_clean'] = round(clean_rate, 4)
                results['combined_lift'] = round(flagged_rate / max(base, 0.001), 2)
                
                logger.info(f"  COMBINED: flag={flags.mean():.1%}, down_if_flag={flagged_rate:.3f} "
                            f"vs base={base:.3f} → lift {flagged_rate/max(base,0.001):.2f}x")
                logger.info(f"  COMBINED: Avoiding flagged would reduce down-risk to {clean_rate:.3f} "
                            f"({(1 - clean_rate/max(base,0.001))*100:.1f}% reduction)")
        
        return results
    
    def predict_single(self, X: np.ndarray, regime: str) -> Dict[str, any]:
        """Predict for a single row (or batch) with known regime.
        
        Args:
            X: Feature array, shape (n_features,) or (n, n_features)
            regime: 'UP' or 'FLAT'
            
        Returns:
            dict with anomaly_score, anomaly_flag, down_risk_flag, confidence_bucket
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if regime == 'UP' and self.up_detector._is_trained:
            return self.up_detector.predict(X)
        elif regime == 'FLAT' and self.flat_detector._is_trained:
            return self.flat_detector.predict(X)
        else:
            # Fallback: no flag
            n = X.shape[0]
            return {
                'anomaly_score': np.zeros(n),
                'anomaly_flag': np.zeros(n, dtype=bool),
                'down_risk_flag': np.zeros(n, dtype=bool),
                'confidence_bucket': np.full(n, 'LOW', dtype=object),
            }
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load both regime detectors from saved artifacts."""
        path = path or MODELS_DIR
        
        # Load feature names
        fn_path = path / "down_risk_feature_names.json"
        if fn_path.exists():
            with open(fn_path, 'r') as f:
                self._feature_names = json.load(f)
        
        up_ok = self.up_detector.load(path)
        flat_ok = self.flat_detector.load(path)
        
        if up_ok or flat_ok:
            logger.info(f"DownRiskDetector loaded: UP={'OK' if up_ok else 'FAIL'}, FLAT={'OK' if flat_ok else 'FAIL'}")
            return True
        return False


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_detector(symbols: Optional[list] = None, config: Optional[DetectorConfig] = None):
    """Load trained detector and evaluate on test set (walk-forward)."""
    config = config or DetectorConfig()
    detector = DownRiskDetector(config)
    
    if not detector.load():
        logger.error("No saved detector found. Run 'train' first.")
        return
    
    combined, feature_names = load_detector_dataset(config, symbols)
    
    # Test split
    combined['_date'] = combined['date'].dt.date
    all_dates = sorted(combined['_date'].unique())
    test_cutoff = all_dates[-config.test_days]
    test_mask = combined['_date'] >= test_cutoff
    
    X_test = np.nan_to_num(combined.loc[test_mask, feature_names].values.astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    y_test = combined.loc[test_mask, 'y_down8'].values
    regimes = combined.loc[test_mask, 'xgb_regime'].values
    
    results = detector._evaluate_test(X_test, y_test, regimes)
    
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    return results


# ══════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════

def infer_from_features(X: np.ndarray, regime: str,
                        detector: Optional[DownRiskDetector] = None) -> Dict[str, any]:
    """One-shot inference from a feature array.
    
    Args:
        X: Feature array (n_features,) or (n, n_features)
        regime: 'UP' or 'FLAT' (from your XGBoost prediction)
        detector: Pre-loaded detector (if None, loads from disk)
    
    Returns:
        dict with anomaly_score, down_risk_flag, confidence_bucket
    """
    if detector is None:
        detector = DownRiskDetector()
        if not detector.load():
            return {
                'anomaly_score': 0.0,
                'down_risk_flag': False,
                'confidence_bucket': 'LOW',
                'error': 'No trained model found',
            }
    
    result = detector.predict_single(X, regime)
    
    # Flatten for single-row case
    if X.ndim == 1 or (X.ndim == 2 and X.shape[0] == 1):
        return {
            'anomaly_score': float(result['anomaly_score'][0]),
            'down_risk_flag': bool(result['down_risk_flag'][0]),
            'confidence_bucket': str(result['confidence_bucket'][0]),
        }
    
    return result


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    parser = argparse.ArgumentParser(description='Down-Risk Detector (VAE + GMM)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # ── train ──
    p_train = subparsers.add_parser('train', help='Train the down-risk detector')
    p_train.add_argument('--latent-dim', type=int, default=16)
    p_train.add_argument('--gmm-components', type=int, default=8)
    p_train.add_argument('--vae-epochs', type=int, default=80)
    p_train.add_argument('--vae-kl-weight', type=float, default=0.5)
    p_train.add_argument('--down-atr-factor', type=float, default=3.0)
    p_train.add_argument('--test-days', type=int, default=20)
    p_train.add_argument('--val-days', type=int, default=10)
    p_train.add_argument('--target-flag-rate', type=float, default=0.15)
    p_train.add_argument('--symbols', nargs='*', default=None)
    
    # ── evaluate ──
    p_eval = subparsers.add_parser('evaluate', help='Evaluate on test set')
    p_eval.add_argument('--test-days', type=int, default=20)
    p_eval.add_argument('--symbols', nargs='*', default=None)
    
    # ── infer ──
    p_infer = subparsers.add_parser('infer', help='Run inference on a symbol')
    p_infer.add_argument('--symbol', type=str, required=True)
    p_infer.add_argument('--tail', type=int, default=50, help='Show last N rows')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = DetectorConfig(
            latent_dim=args.latent_dim,
            gmm_n_components=args.gmm_components,
            vae_epochs=args.vae_epochs,
            vae_kl_weight=args.vae_kl_weight,
            down_atr_factor=args.down_atr_factor,
            test_days=args.test_days,
            val_days=args.val_days,
            target_flag_rate=args.target_flag_rate,
        )
        detector = DownRiskDetector(config)
        metrics = detector.train(symbols=args.symbols)
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        
        # Print summary
        for regime in ['UP', 'FLAT']:
            key = f'{regime}_metrics'
            if key in metrics and metrics[key]:
                m = metrics[key]
                print(f"\n  {regime} Regime:")
                print(f"    Val AUROC:        {m.get('val_auroc', 'N/A')}")
                print(f"    Val Lift:         {m.get('val_lift', 'N/A')}x")
                print(f"    Val Flag Rate:    {m.get('val_flag_rate', 'N/A')}")
                print(f"    Val Down|Flag:    {m.get('val_down_if_flagged', 'N/A')}")
                print(f"    Val Down|Clean:   {m.get('val_down_if_not_flagged', 'N/A')}")
        
        if 'test' in metrics:
            t = metrics['test']
            print(f"\n  Combined Test:")
            print(f"    AUROC:            {t.get('combined_auroc', 'N/A')}")
            print(f"    Lift:             {t.get('combined_lift', 'N/A')}x")
            print(f"    Down if flagged:  {t.get('combined_down_if_flagged', 'N/A')}")
            print(f"    Down if clean:    {t.get('combined_down_if_clean', 'N/A')}")
        
        print(f"\n  Time: {metrics.get('training_time_sec', '?')}s")
    
    elif args.command == 'evaluate':
        config = DetectorConfig(test_days=args.test_days)
        evaluate_detector(symbols=args.symbols, config=config)
    
    elif args.command == 'infer':
        # Load detector
        detector = DownRiskDetector()
        if not detector.load():
            print("ERROR: No trained model found. Run 'train' first.")
            return
        
        # Load the symbol's data and compute features
        from .feature_engineering import compute_features, get_feature_names, get_sector_for_symbol
        
        parquet_path = DATA_DIR / f"{args.symbol}.parquet"
        if not parquet_path.exists():
            print(f"ERROR: No parquet file for {args.symbol}")
            return
        
        df = pd.read_parquet(parquet_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Compute features (minimal — no daily/OI/nifty/sector for quick inference)
        df = compute_features(df, symbol=args.symbol)
        if df.empty:
            print(f"ERROR: Feature computation failed for {args.symbol}")
            return
        
        feature_names = get_feature_names()
        
        # Take last N rows
        df_tail = df.tail(args.tail).copy()
        X = np.nan_to_num(df_tail[feature_names].values.astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        
        # Score for both regimes
        print(f"\n{'='*70}")
        print(f"  Down-Risk Inference: {args.symbol} (last {len(df_tail)} candles)")
        print(f"{'='*70}")
        
        for regime in ['UP', 'FLAT']:
            result = detector.predict_single(X, regime)
            
            print(f"\n  If XGB predicts {regime}:")
            flagged = result['down_risk_flag']
            scores = result['anomaly_score']
            buckets = result['confidence_bucket']
            
            n_flagged = flagged.sum()
            print(f"    Flagged: {n_flagged}/{len(flagged)} ({n_flagged/len(flagged)*100:.0f}%)")
            print(f"    Score range: [{scores.min():.2f}, {scores.max():.2f}]")
            
            # Show flagged rows
            if n_flagged > 0:
                print(f"    Flagged candles:")
                for i in range(len(df_tail)):
                    if flagged[i]:
                        row = df_tail.iloc[i]
                        print(f"      {row['date']} close={row['close']:.2f} "
                              f"score={scores[i]:.2f} bucket={buckets[i]}")


if __name__ == '__main__':
    main()
