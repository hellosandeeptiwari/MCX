"""
PREDICTOR (INFERENCE) — Real-time prediction for Titan integration

Binary classification model: MOVE vs NO_MOVE (≥0.5% in 30 min)

Usage in Titan's scan pipeline:
    from ml_models.predictor import MovePredictor
    
    predictor = MovePredictor()  # loads latest model
    
    # During scan cycle, for each stock:
    result = predictor.predict(candles_df)
    # result = {
    #   'ml_move_prob': 0.72,          # probability stock moves ≥0.5%
    #   'ml_no_move_prob': 0.28,       # probability stock stays flat
    #   'ml_signal': 'MOVE',           # MOVE or NO_MOVE
    #   'ml_confidence': 0.72,         # confidence in predicted class
    #   'ml_score_boost': 4,           # boost for Titan scoring
    # }
    #
    # Score boost:  +6/+4/+2 for high MOVE probability
    #              -1/-3 for high NO_MOVE probability (flat stock = waste of capital)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import threading

import xgboost as xgb

from .feature_engineering import compute_features, get_feature_names


MODELS_DIR = Path(__file__).parent / "saved_models"


class MovePredictor:
    """Real-time prediction using trained XGBoost model.
    
    Supports two architectures:
    - Legacy 3-class (single model: DOWN/FLAT/UP)
    - Meta-labeling (2 models: Gate MOVE/FLAT + Direction UP/DOWN)
    """
    
    def __init__(self, model_path: Optional[str] = None, logger=None):
        """Load trained model(s).
        
        Automatically detects meta-labeling models if available,
        falls back to legacy 3-class model otherwise.
        
        Args:
            model_path: Path to model .json file. If None, loads latest.
            logger: Optional logger instance
        """
        self.logger = logger
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.ready = False
        self.model_type = None  # 'meta_labeling' or '3-class'
        
        # Meta-labeling specific
        self.gate_model = None
        self.dir_model = None
        self.gate_cal = None
        self.dir_cal = None
        
        # Legacy 3-class specific
        self.calibrators = None
        
        # Thread-safety: PyTorch VAE is not thread-safe for concurrent inference
        self._down_risk_lock = threading.Lock()
        # Thread-safety: XGBoost predict_proba is not thread-safe
        self._xgb_lock = threading.Lock()
        
        # Try meta-labeling first (preferred)
        meta_gate_path = str(MODELS_DIR / "meta_gate_latest.json")
        meta_dir_path = str(MODELS_DIR / "meta_direction_latest.json")
        meta_meta_path = str(MODELS_DIR / "meta_labeling_latest_meta.json")
        
        if os.path.exists(meta_gate_path) and os.path.exists(meta_dir_path):
            try:
                self._load_meta_labeling(meta_gate_path, meta_dir_path, meta_meta_path)
                return  # Successfully loaded meta-labeling
            except Exception as e:
                self._log(f"⚠ Meta-labeling load failed, falling back to 3-class: {e}")
        
        # Fall back to legacy 3-class model
        if model_path is None:
            model_path = str(MODELS_DIR / "move_predictor_latest.json")
        
        self._load_legacy_3class(model_path)
    
    def _load_meta_labeling(self, gate_path: str, dir_path: str, meta_path: str):
        """Load meta-labeling models (Gate + Direction)."""
        self.gate_model = xgb.XGBClassifier()
        self.gate_model.load_model(gate_path)
        
        self.dir_model = xgb.XGBClassifier()
        self.dir_model.load_model(dir_path)
        
        # Load calibrators
        gate_cal_path = gate_path.replace('.json', '_calibrator.pkl')
        dir_cal_path = dir_path.replace('.json', '_calibrator.pkl')
        
        if os.path.exists(gate_cal_path):
            self.gate_cal = joblib.load(gate_cal_path)
        if os.path.exists(dir_cal_path):
            self.dir_cal = joblib.load(dir_cal_path)
        
        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('feature_names', get_feature_names())
            # Direction model may use a pruned feature subset
            self.dir_feature_names = self.metadata.get('direction_feature_names', self.feature_names)
        else:
            self.feature_names = get_feature_names()
            self.dir_feature_names = self.feature_names
        
        self.model_type = 'meta_labeling'
        self.ready = True
        
        # Down-risk detector (lazy-loaded on first use)
        self._down_risk_detector = None
        self._down_risk_loaded = False
        
        self._log(f"✅ MovePredictor loaded: META-LABELING (Gate + Direction)")
        if self.metadata:
            gate_info = self.metadata.get('gate_model', {})
            dir_info = self.metadata.get('direction_model', {})
            combined = self.metadata.get('combined', {})
            self._log(f"   Gate accuracy: {gate_info.get('accuracy', '?')}")
            self._log(f"   Direction accuracy: {dir_info.get('accuracy', '?')}")
            self._log(f"   Combined macro F1: {combined.get('macro_f1', '?')}")
            self._log(f"   Gate calibrator: {'✓' if self.gate_cal else '✗'}")
            self._log(f"   Direction calibrator: {'✓' if self.dir_cal else '✗'}")
    
    def _load_legacy_3class(self, model_path: str):
        """Load legacy 3-class single model."""
        meta_path = model_path.replace('.json', '_meta.json')
        
        if not os.path.exists(model_path):
            self._log(f"⚠ Model file not found: {model_path}")
            self._log("  Run trainer.py or meta_trainer.py first.")
            return
        
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', get_feature_names())
            else:
                self.feature_names = get_feature_names()
            
            # Load isotonic calibrators if available
            cal_path = model_path.replace('.json', '_calibrators.pkl')
            if os.path.exists(cal_path):
                try:
                    self.calibrators = joblib.load(cal_path)
                    self._log(f"   ✓ Isotonic calibrators loaded ({len(self.calibrators)} classes)")
                except Exception as cal_e:
                    self._log(f"   ⚠ Calibrator load failed (using raw probs): {cal_e}")
            else:
                self._log(f"   ⚠ No calibrator file found (using raw probabilities)")
            
            self.model_type = '3-class'
            self.ready = True
            self._log(f"✅ MovePredictor loaded: 3-CLASS (legacy)")
            if self.metadata:
                acc = self.metadata.get('accuracy', '?')
                self._log(f"   Accuracy: {acc}")
                cal_flag = self.metadata.get('calibrated', False)
                self._log(f"   Calibrated: {cal_flag}")
                self._log(f"   Trained on: {self.metadata.get('train_samples', '?')} samples")
        
        except Exception as e:
            self._log(f"❌ Failed to load model: {e}")
    
    def predict(self, candles_df: pd.DataFrame, daily_df: pd.DataFrame = None,
                 oi_df: pd.DataFrame = None, futures_oi_df: pd.DataFrame = None,
                 nifty_5min_df: pd.DataFrame = None, nifty_daily_df: pd.DataFrame = None,
                 sector_5min_df: pd.DataFrame = None, sector_daily_df: pd.DataFrame = None) -> dict:
        """Predict move probability for a single stock.
        
        Args:
            candles_df: Recent 5-min OHLCV candles (need >=50 for feature warmup).
                        Columns: date, open, high, low, close, volume
            daily_df: Optional daily OHLCV candles for context features.
            oi_df: Optional OI snapshot DataFrame for options context features.
            futures_oi_df: Optional daily futures OI features DataFrame.
            nifty_5min_df: Optional NIFTY50 5-min candles for market context.
            nifty_daily_df: Optional NIFTY50 daily candles for market context.
            sector_5min_df: Optional sector index 5-min candles for sector context.
            sector_daily_df: Optional sector index daily candles for sector context.
        
        Returns:
            dict with prediction results, or empty dict if prediction fails.
        """
        if not self.ready:
            return {}
        
        try:
            # Compute features on the candles
            featured = compute_features(candles_df, daily_df=daily_df, oi_df=oi_df, futures_oi_df=futures_oi_df,
                                        nifty_5min_df=nifty_5min_df, nifty_daily_df=nifty_daily_df,
                                        sector_5min_df=sector_5min_df, sector_daily_df=sector_daily_df)
            if featured.empty or len(featured) < 2:
                return {}
            
            # ── SAFEGUARD: Detect zero-OI when OI was provided ──
            # If futures_oi_df was given but fut_oi_buildup is still 0 on the
            # latest candle, something went wrong with the date merge.
            if futures_oi_df is not None and len(futures_oi_df) > 0:
                _latest_oi_val = featured.iloc[-1].get('fut_oi_buildup', 0.0)
                if _latest_oi_val == 0.0:
                    # Check if OI has ANY non-zero values (if all zero, data itself is zero)
                    _any_nonzero = (featured['fut_oi_buildup'] != 0).any() if 'fut_oi_buildup' in featured.columns else False
                    if _any_nonzero and self.logger:
                        self.logger.warning(
                            'OI forward-fill may have failed: fut_oi_buildup=0 on latest candle '
                            f'but {(featured["fut_oi_buildup"] != 0).sum()} earlier candles have non-zero values. '
                            f'Candle date: {featured.iloc[-1]["date"]}, OI max date: {futures_oi_df["date"].max()}'
                        )
            
            # Take the LAST row (most recent candle = what we want to predict)
            latest = featured.iloc[-1:]
            
            # Ensure we have all required features (fill missing with 0)
            for feat in self.feature_names:
                if feat not in latest.columns:
                    latest = latest.copy()
                    latest[feat] = 0.0
            
            # Extract feature values
            X = latest[self.feature_names].values
            
            # Replace any remaining NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ── Route to appropriate model architecture ──
            if self.model_type == 'meta_labeling':
                result = self._predict_meta(X)
            else:
                result = self._predict_3class(X)
            
            # Stash feature array for downstream use (down-risk detector)
            result['_features_array'] = X
            return result
        
        except Exception as e:
            self._log(f"❌ Prediction error: {e}")
            return {}
    
    def _predict_meta(self, X: np.ndarray) -> dict:
        """Meta-labeling prediction: Gate + Direction models.
        
        Signal logic based on individual model outputs (NOT combined product):
        - P(MOVE) from gate → determines if stock will move
        - P(UP|MOVE) from direction → if moving, which direction?
        - Thresholds are applied to EACH model separately
        
        Hybrid model (v7+) uses calibrated probabilities:
        - Gate: calibrated P(MOVE) in 0-1 range, threshold = 0.50
        - Direction: calibrated P(UP|MOVE) in 0-1 range, dead zone 0.48-0.52
        """
        # Gate model: P(MOVE) — uses full feature set
        with self._xgb_lock:
            gate_raw = float(self.gate_model.predict_proba(X)[0, 1])  # P(MOVE)
        # RAW probabilities — no calibration overlay
        # Calibrators removed Mar-05: they were biasing predictions
        # (e.g. compressing all signals toward FLAT in volatile markets)
        p_move = gate_raw
        
        # Direction model: P(UP|MOVE) — may use pruned feature subset
        if self.dir_feature_names != self.feature_names:
            # Build index mapping from full features to pruned direction features
            dir_indices = [self.feature_names.index(f) for f in self.dir_feature_names if f in self.feature_names]
            X_dir = X[:, dir_indices]
        else:
            X_dir = X
        
        with self._xgb_lock:
            dir_raw = float(self.dir_model.predict_proba(X_dir)[0, 1])  # P(UP|MOVE)
        # RAW probabilities — no calibration overlay
        p_up_given_move = dir_raw
        
        # Market regime Bayesian adjustment REMOVED (Mar-05):
        # The regime adjustment was biasing 40/40 stocks to DOWN in bullish
        # markets by over-weighting NIFTY daily features. The model should
        # predict what it predicts — downstream sector/breadth gates handle
        # market context filtering without distorting raw probabilities.
        _regime_adj = 0.0
        
        p_down_given_move = 1 - p_up_given_move
        
        # Combined probabilities (for backward-compat output)
        prob_up = p_move * p_up_given_move
        prob_down = p_move * p_down_given_move
        prob_flat = 1 - p_move
        
        # ── Signal determination (calibrated probabilities, 0-1 range) ──
        # Gate:      calibrated P(MOVE), threshold 0.50
        # Direction: calibrated P(UP|MOVE), dead zone 0.46-0.54 → coin-flip
        #
        # High-confidence thresholds from training analysis:
        # Combined 50%+ → 52-53% precision (usable)
        # Combined 60%+ → 58-68% precision (high quality)
        #
        # Only assign UP/DOWN when BOTH gate says likely move AND direction
        # model shows genuine edge (outside dead zone).
        # Dead zone widened to 46-54% to avoid trades where direction is ambiguous.
        
        if p_move >= 0.50:
            # Gate says move likely
            if p_up_given_move >= 0.54:
                signal = 'UP'
                confidence = prob_up
            elif p_down_given_move >= 0.54:
                signal = 'DOWN'
                confidence = prob_down
            else:
                # Direction model is ~50/50 — no real conviction.
                # Still flag as MOVE but with neutral direction
                signal = 'FLAT'
                confidence = prob_flat
        else:
            signal = 'FLAT'
            confidence = prob_flat
        
        direction_bias = prob_up - prob_down
        
        # Score boost — calibrated probability thresholds
        # Only boost when BOTH gate + direction have conviction.
        # CRITICAL: FLAT signal → 0 or negative boost.
        score_boost = 0
        if signal != 'FLAT':
            combined_prob = max(prob_up, prob_down)  # the winning direction's combined prob
            if combined_prob >= 0.40:
                score_boost = 10  # High-conviction directional (~68% precision)
            elif combined_prob >= 0.35:
                score_boost = 8   # Strong signal (~58% precision)
            elif combined_prob >= 0.30:
                score_boost = 5   # Moderate signal (~52% precision)
            elif combined_prob >= 0.25:
                score_boost = 3   # Weak-but-directional
        # FLAT signal penalties
        if signal == 'FLAT':
            if p_move < 0.35:
                # Gate model says stock unlikely to move → penalize
                if prob_flat >= 0.75:
                    score_boost = -5  # Strong FLAT conviction
                elif prob_flat >= 0.65:
                    score_boost = -3  # Moderate FLAT
                elif prob_flat >= 0.55:
                    score_boost = -2  # Mild FLAT
                else:
                    score_boost = 0
            else:
                # Gate says move possible, but direction is coin-flip → neutral
                score_boost = 0
        elif p_move <= 0.30:
            score_boost = -8  # Very likely flat (bottom quartile)
        elif p_move <= 0.40:
            score_boost = -5  # Likely flat
        
        return {
            'ml_prob_up': round(prob_up, 4),
            'ml_prob_down': round(prob_down, 4),
            'ml_prob_flat': round(prob_flat, 4),
            'ml_move_prob': round(p_move, 4),
            'ml_direction_bias': round(direction_bias, 4),
            'ml_signal': signal,
            'ml_confidence': round(confidence, 4),
            'ml_score_boost': score_boost,
            'ml_p_move': round(p_move, 4),
            'ml_p_up_given_move': round(p_up_given_move, 4),
            'ml_p_down_given_move': round(p_down_given_move, 4),
            'ml_model_type': 'meta_labeling',
            'ml_regime_adj': round(_regime_adj, 4),
        }
    
    def _market_regime_adjust(self, X: np.ndarray, p_up_given_move: float) -> tuple:
        """Bayesian market-regime adjustment for P(UP|MOVE).
        
        The direction model structurally underweights market context:
        - All NIFTY features combined = ~10% importance
        - fut_oi_buildup alone = 46.5% importance
        - During bull-market training, bearish OI signals (buildup=-1)
          correlated with UP (mean-reversion / short squeeze). In a real
          bear market, these should be DOWN (continuation).
        
        This amplifies the effective NIFTY weight from ~10% to ~25-30%
        by shifting P(UP|MOVE) based on market regime derived from the
        same NIFTY features already in the feature vector.
        
        CRITICAL: Daily features are weighted 4x because they represent
        the TRUE regime (multi-day trend), while 5-min features are noisy
        and show false bullish signals during intraday bounces in crashes.
        
        Returns:
            (adjusted_p_up_given_move, regime_adjustment_amount)
        """
        # Extract NIFTY features from the FULL feature array (87 features)
        def _get(name, default=0.0):
            try:
                idx = self.feature_names.index(name)
                return float(X[0, idx])
            except (ValueError, IndexError):
                return default
        
        nifty_roc = _get('nifty_roc_6')
        nifty_rsi = _get('nifty_rsi_14')
        nifty_trend = _get('nifty_daily_trend')
        nifty_slope = _get('nifty_ema9_slope')
        nifty_bb = _get('nifty_bb_position')
        nifty_d_rsi = _get('nifty_daily_rsi')
        
        # Check if NIFTY data is actually present (not all zeros = missing)
        vals = [nifty_roc, nifty_rsi, nifty_trend, nifty_slope, nifty_bb, nifty_d_rsi]
        if all(abs(v) < 0.001 for v in vals):
            return p_up_given_move, 0.0  # NIFTY data missing → no adjustment
        
        # Compute directional regime score in [-1.0, +1.0]
        # CRITICAL: Daily features get 4x weight — they ARE the regime signal.
        # 5-min features are noisy intraday indicators that show false bullish
        # signals during bounces in crash days. Daily trend (-4.6% below EMA-50)
        # and daily RSI (23 = extreme oversold) are the truth.
        weighted_signals = []  # (signal, weight)
        
        # ── DAILY FEATURES (high weight = 4) — TRUE regime indicators ──
        
        # nifty_daily_trend: price vs EMA (%). Typical: -5% to +5%
        # -4.6% means NIFTY is deeply below its 50-day moving average
        if abs(nifty_trend) > 0.01:
            weighted_signals.append((max(-1.0, min(1.0, nifty_trend / 2.0)), 4.0))
        
        # nifty_daily_rsi: daily RSI — strongest regime indicator
        # RSI 23 = extreme bearish; RSI 70+ = extreme bullish
        if nifty_d_rsi > 0.1:
            d_rsi_signal = (nifty_d_rsi - 50) / 15  # More sensitive: 35=-1, 50=0, 65=+1
            weighted_signals.append((max(-1.0, min(1.0, d_rsi_signal)), 4.0))
        
        # ── 5-MIN FEATURES (low weight = 1) — noisy intraday indicators ──
        
        # nifty_roc_6: 6-period rate of change (5-min)
        if abs(nifty_roc) > 0.01:
            weighted_signals.append((max(-1.0, min(1.0, nifty_roc / 1.0)), 1.0))
        
        # nifty_rsi_14: intraday RSI
        if nifty_rsi > 0.1:
            rsi_signal = (nifty_rsi - 50) / 20
            weighted_signals.append((max(-1.0, min(1.0, rsi_signal)), 1.0))
        
        # nifty_ema9_slope: intraday trend direction
        if abs(nifty_slope) > 0.001:
            weighted_signals.append((max(-1.0, min(1.0, nifty_slope * 15)), 1.0))
        
        # nifty_bb_position: intraday Bollinger band position
        if abs(nifty_bb) > 0.01:
            bb_signal = (nifty_bb - 0.5) * 3.0
            weighted_signals.append((max(-1.0, min(1.0, bb_signal)), 1.0))
        
        if not weighted_signals:
            return p_up_given_move, 0.0
        
        total_weight = sum(w for _, w in weighted_signals)
        regime_score = sum(s * w for s, w in weighted_signals) / total_weight
        regime_score = max(-1.0, min(1.0, regime_score))
        
        # Apply when regime is even mildly directional (|score| > 0.10)
        # Lower threshold because daily features already dominate the score;
        # if score > 0.10, it means the daily regime IS directional.
        if abs(regime_score) < 0.10:
            return p_up_given_move, 0.0
        
        # ── STEP 1: Confidence compression (reduce overconfidence) ──
        # The model's raw P(UP|MOVE) can be 0.87 even in a crash because
        # the 46.5% OI-buildup feature was trained in a bull market where
        # bearish OI → mean-reversion UP. In bear markets, this is wrong.
        # 
        # Compress P(UP|MOVE) toward 0.50 proportional to |regime_score|.
        # This doesn't bias direction — it reduces CERTAINTY when the
        # market regime disagrees with the model's confidence level.
        #
        # squeeze = |score| capped at 0.75:
        #   score=-0.53 → squeeze=0.53, P(UP)=0.87 → 0.67
        #   score=-0.25 → squeeze=0.25, P(UP)=0.87 → 0.78
        #   score=-1.00 → squeeze=0.75, P(UP)=0.87 → 0.59
        squeeze = min(0.75, abs(regime_score))
        retention = 1.0 - squeeze
        p_squeezed = 0.50 + (p_up_given_move - 0.50) * retention
        
        # ── STEP 2: Directional shift (apply regime bias) ──
        # After compression, shift the probability in the regime direction.
        # MAX_SHIFT = 0.25 — needs to be strong since compression alone
        # only brings 0.867→0.673 (still above 0.54 UP threshold).
        # Combined effect at score=-0.53: 0.867→0.673→0.540 (UP→FLAT ✓)
        MAX_SHIFT = 0.25
        shift = regime_score * MAX_SHIFT

        adjusted = p_squeezed + shift
        adjusted = max(0.10, min(0.90, adjusted))  # clip to valid probability range
        
        total_shift = adjusted - p_up_given_move  # total effect for logging
        
        # Log to stdout (captured by titan.log) for visibility
        if abs(total_shift) > 0.02:
            print(
                f"      📊 Regime adj: score={regime_score:+.2f} "
                f"P(UP|MOVE) {p_up_given_move:.3f}→{p_squeezed:.3f}→{adjusted:.3f} "
                f"[trend={nifty_trend:+.1f}% d_rsi={nifty_d_rsi:.0f} "
                f"roc={nifty_roc:+.2f} rsi={nifty_rsi:.0f} slope={nifty_slope:+.3f} bb={nifty_bb:.2f}]"
            )
        
        return adjusted, total_shift
    
    def _predict_3class(self, X: np.ndarray) -> dict:
        """Legacy 3-class single model prediction."""
        # Predict probabilities — 3-class: [prob_down, prob_flat, prob_up]
        probas = self.model.predict_proba(X)[0]
        
        # Isotonic calibration REMOVED (Mar-05): using raw model probabilities
        # Calibrators were distorting raw predictions and contributing to
        # uniform signal bias (40/40 stocks getting same direction).
        
        # Handle both 3-class and legacy 2-class models gracefully
        if len(probas) == 3:
            prob_down = float(probas[0])   # class 0 = DOWN
            prob_flat = float(probas[1])   # class 1 = FLAT
            prob_up = float(probas[2])     # class 2 = UP
        else:
            # Legacy 2-class fallback
            prob_down = 0.0
            prob_flat = float(probas[0])
            prob_up = float(probas[1]) if len(probas) > 1 else 0.0
        
        prob_move = prob_up + prob_down  # Total directional probability
        
        # Signal determination (3-class)
        # Dead zone: 46-54% → treat as FLAT (direction too ambiguous)
        if prob_up >= 0.54:
            signal = 'UP'
            confidence = prob_up
        elif prob_down >= 0.54:
            signal = 'DOWN'
            confidence = prob_down
        else:
            signal = 'FLAT'
            confidence = prob_flat
        
        # Direction bias: positive = bullish, negative = bearish
        direction_bias = prob_up - prob_down
        
        # Score boost
        score_boost = 0
        max_dir = max(prob_up, prob_down)
        if max_dir >= 0.50:
            score_boost = 10
        elif max_dir >= 0.40:
            score_boost = 8
        elif max_dir >= 0.30:
            score_boost = 5
        elif prob_flat >= 0.70:
            score_boost = -8
        elif prob_flat >= 0.60:
            score_boost = -5
        elif prob_flat >= 0.55:
            score_boost = -3
        
        return {
            'ml_prob_up': round(prob_up, 4),
            'ml_prob_down': round(prob_down, 4),
            'ml_prob_flat': round(prob_flat, 4),
            'ml_move_prob': round(prob_move, 4),
            'ml_direction_bias': round(direction_bias, 4),
            'ml_signal': signal,
            'ml_confidence': round(confidence, 4),
            'ml_score_boost': score_boost,
            'ml_model_type': '3-class',
        }
    
    def get_titan_signals(self, candles_df, daily_df=None, oi_df=None, futures_oi_df=None,
                           nifty_5min_df=None, nifty_daily_df=None,
                           sector_5min_df=None, sector_daily_df=None) -> dict:
        """All-in-one Titan integration signals.
        
        FAIL-SAFE: Returns neutral defaults on ANY error.
        If this crashes, Titan continues exactly as if ML doesn't exist.
        
        Args:
            candles_df: Recent 5-min OHLCV candles.
            daily_df: Optional daily OHLCV candles for context features.
            oi_df: Optional OI snapshot DataFrame for options context features.
            futures_oi_df: Optional daily futures OI features DataFrame.
            nifty_5min_df: Optional NIFTY50 5-min candles for market context.
            nifty_daily_df: Optional NIFTY50 daily candles for market context.
            sector_5min_df: Optional sector index 5-min candles for sector context.
            sector_daily_df: Optional sector index daily candles for sector context.
        
        Returns:
            dict with all predict() fields PLUS:
            - ml_sizing_factor: position sizing multiplier (0.7 to 1.2)
            - ml_entry_caution: True if stock likely flat (soft warning, NEVER blocks)
            - ml_chop_hint: True if ML sees very low move probability
            - ml_elite_ok: True if ML supports elite auto-fire
            - ml_gpt_summary: one-line summary for GPT prompt
        """
        try:
            pred = self.predict(candles_df, daily_df, oi_df=oi_df, futures_oi_df=futures_oi_df,
                               nifty_5min_df=nifty_5min_df, nifty_daily_df=nifty_daily_df,
                               sector_5min_df=sector_5min_df, sector_daily_df=sector_daily_df)
            if not pred:
                return self._titan_defaults()
            
            # ── Futures OI staleness check ──
            # fut_oi_buildup + oi_price_confirm = ~30% of direction model importance.
            # ONLY penalize when OI data EXISTS but is outdated (>3 days).
            # Missing OI (None / not in universe) is NOT stale — just unavailable.
            oi_stale = False
            oi_available = False
            if futures_oi_df is not None and not futures_oi_df.empty:
                oi_available = True
                try:
                    last_oi_date = pd.Timestamp(futures_oi_df['date'].max())
                    now = pd.Timestamp.now()
                    # Allow for weekends: check if last date is > 3 calendar days ago
                    if (now - last_oi_date).days > 3:
                        oi_stale = True
                except Exception:
                    oi_stale = True
            
            if oi_stale:
                original_boost = pred.get('ml_score_boost', 0)
                # OI features = 57% of direction model importance.
                # When stale, direction prediction is unreliable → zero the boost.
                pred['ml_score_boost'] = 0
                pred['ml_signal'] = 'FLAT'
                self._log(f"⚠️ Futures OI stale → score_boost zeroed: {original_boost} → 0, signal→FLAT (OI=57% of direction model)")
            # Also zero boost when OI data is completely missing but model expects it
            if not oi_available and pred.get('ml_model_type') == 'meta_labeling':
                pred['ml_score_boost'] = 0
                pred['ml_signal'] = 'FLAT'
                pred['ml_oi_missing'] = True
            
            # ── OI data-quality gate ──
            # Even when OI data "exists" and is "fresh", the feature VALUES may be
            # garbage — e.g. DhanHQ returns futures data but equity spot merge fails
            # → basis=0, vol_ratio=0 for that date.  The model sees "no premium +
            # no volume" as a specific regime and may uniformly predict UP or DOWN.
            # Detect this by checking the LAST ROW of the OI file: if both basis
            # and vol_ratio are zero, the spot merge likely failed.
            oi_quality_ok = True
            if oi_available and not oi_stale and futures_oi_df is not None and len(futures_oi_df) > 0:
                _last_row = futures_oi_df.iloc[-1]
                _last_basis = abs(float(_last_row.get('fut_basis_pct', 0)))
                _last_vol = float(_last_row.get('fut_vol_ratio', 0))
                if _last_basis < 0.001 and _last_vol < 0.001:
                    oi_quality_ok = False
                    original_boost = pred.get('ml_score_boost', 0)
                    pred['ml_score_boost'] = 0
                    pred['ml_signal'] = 'FLAT'
                    pred['ml_oi_quality_bad'] = True
                    self._log(f"⚠️ OI quality gate: basis={_last_basis:.3f}% vol_ratio={_last_vol:.2f} "
                              f"→ spot merge failed → boost zeroed, signal→FLAT")
            pred['ml_oi_quality_ok'] = oi_quality_ok
            
            pred['ml_oi_stale'] = oi_stale
            pred['ml_oi_available'] = oi_available
            
            prob_up = pred.get('ml_prob_up', 0)
            prob_down = pred.get('ml_prob_down', 0)
            prob_flat = pred.get('ml_prob_flat', 0.5)
            move_prob = pred.get('ml_move_prob', 0.5)
            signal = pred.get('ml_signal', 'FLAT')
            direction_bias = pred.get('ml_direction_bias', 0)
            
            # Meta-labeling specific fields
            p_move = pred.get('ml_p_move', move_prob)
            p_up_given_move = pred.get('ml_p_up_given_move', 0.5)
            p_down_given_move = pred.get('ml_p_down_given_move', 0.5)
            is_meta = pred.get('ml_model_type') == 'meta_labeling'
            
            # === Position sizing factor (calibrated probabilities, 0-1 range) ===
            if is_meta:
                # Combined probability = max(prob_up, prob_down)
                combined_dir = max(prob_up, prob_down)
                if combined_dir >= 0.40:
                    sizing_factor = 2.0   # ELITE: ~68% precision from training
                elif combined_dir >= 0.35:
                    sizing_factor = 1.5   # HIGH: ~58% precision
                elif combined_dir >= 0.30:
                    sizing_factor = 1.3   # Good: ~52% precision
                elif combined_dir >= 0.25:
                    sizing_factor = 1.2   # Moderate: ~47% precision
                elif p_move >= 0.50:
                    sizing_factor = 1.0   # Move likely but direction unclear
                elif p_move <= 0.30:
                    sizing_factor = 0.7   # Flat likely
                elif p_move <= 0.40:
                    sizing_factor = 0.85  # Slightly flat-leaning
                else:
                    sizing_factor = 1.0
            else:
                max_dir = max(prob_up, prob_down)
                if max_dir >= 0.70:
                    sizing_factor = 2.0   # ELITE: 70%+ confidence → 2x size
                elif max_dir >= 0.65:
                    sizing_factor = 1.5   # HIGH: 65%+ confidence → 1.5x size
                elif max_dir >= 0.50:
                    sizing_factor = 1.3
                elif max_dir >= 0.40:
                    sizing_factor = 1.2
                elif max_dir >= 0.30:
                    sizing_factor = 1.0
                elif prob_flat >= 0.65:
                    sizing_factor = 0.7
                elif prob_flat >= 0.55:
                    sizing_factor = 0.85
                else:
                    sizing_factor = 1.0
            
            # === Entry caution (soft signal, NEVER blocks) ===
            entry_caution = p_move <= 0.35 if is_meta else prob_flat >= 0.70
            
            # === Chop hint (stock is extremely flat) ===
            chop_hint = p_move <= 0.30 if is_meta else prob_flat >= 0.75
            
            # === Elite auto-fire OK ===
            # Only allow elite auto-fire when combined directional prob >= 30%
            # (52%+ precision from training data)
            combined_dir_prob = max(prob_up, prob_down)
            elite_ok = (combined_dir_prob >= 0.30) if is_meta else prob_flat < 0.70
            
            # === Direction hint for strategy selection (calibrated thresholds) ===
            if is_meta:
                # Combined prob thresholds from training high-confidence analysis:
                # 35%+ → ~58% precision (strong), 25%+ → ~47% (lean)
                # CRITICAL: Compare which direction is stronger first to avoid
                # always hitting BULLISH when both exceed threshold.
                if prob_up >= 0.35 and prob_up > prob_down:
                    direction_hint = 'BULLISH'
                elif prob_down >= 0.35 and prob_down > prob_up:
                    direction_hint = 'BEARISH'
                elif prob_up >= 0.25 and prob_up >= prob_down:
                    direction_hint = 'BULLISH_LEAN'
                elif prob_down >= 0.25 and prob_down > prob_up:
                    direction_hint = 'BEARISH_LEAN'
                else:
                    direction_hint = 'NEUTRAL'
            else:
                if prob_up >= 0.50:
                    direction_hint = 'BULLISH'
                elif prob_down >= 0.50:
                    direction_hint = 'BEARISH'
                elif prob_up >= 0.40:
                    direction_hint = 'BULLISH_LEAN'
                elif prob_down >= 0.40:
                    direction_hint = 'BEARISH_LEAN'
                elif direction_bias > 0.10:
                    direction_hint = 'BULLISH_LEAN'
                elif direction_bias < -0.10:
                    direction_hint = 'BEARISH_LEAN'
                else:
                    direction_hint = 'NEUTRAL'
            
            # === GPT summary line (compact for prompt) ===
            if is_meta:
                if signal == 'UP':
                    gpt_line = f"\U0001f9e0ML:UP(move={p_move:.0%},dir={p_up_given_move:.0%})"
                elif signal == 'DOWN':
                    gpt_line = f"\U0001f9e0ML:DOWN(move={p_move:.0%},dir={p_down_given_move:.0%})"
                else:
                    gpt_line = f"\U0001f9e0ML:FLAT(move={p_move:.0%})"
            else:
                if signal == 'UP':
                    gpt_line = f"\U0001f9e0ML:UP({prob_up:.0%})bias={direction_bias:+.2f}"
                elif signal == 'DOWN':
                    gpt_line = f"\U0001f9e0ML:DOWN({prob_down:.0%})bias={direction_bias:+.2f}"
                else:
                    gpt_line = f"\U0001f9e0ML:FLAT({prob_flat:.0%})"
            
            result = {
                **pred,  # All existing fields (ml_score_boost, probabilities, etc.)
                'ml_sizing_factor': sizing_factor,
                'ml_entry_caution': entry_caution,
                'ml_chop_hint': chop_hint,
                'ml_elite_ok': elite_ok,
                'ml_direction_hint': direction_hint,
                'ml_gpt_summary': gpt_line,
            }
            
            # ── Down-Risk Detector overlay (VAE+GMM) ──
            # Runs for UP/FLAT (defensive) AND DOWN (confirmation) signals
            result.update(self._get_down_risk(signal, pred))
            
            return result
        except Exception:
            return self._titan_defaults()
    
    def _get_down_risk(self, signal: str, pred: dict) -> dict:
        """Run BOTH regime anomaly detectors and return independent flags.

        Runs UP + DOWN regime GMM models simultaneously for every prediction.
        This enables the full 18-case decision matrix (3 XGB × 2 Titan × 3 GMM).

        Flags:
          UP_Flag=True  → hidden DOWN risk (crash likely, 2x crash rate lift)
                          Predicts DOWN. Confirms SELL, opposes BUY.
          Down_Flag=True → hidden UP risk (bounce likely, 2x bounce rate lift)
                          Predicts UP. Confirms BUY, opposes SELL.
          Both False     → Clean (no anomaly edge detected)
          Both True      → Conflicting (extreme vol) → treated as BLOCK
        """
        defaults = {
            'ml_up_flag': False,
            'ml_down_flag': False,
            'ml_up_score': 0.0,
            'ml_down_score': 0.0,
            'ml_down_risk_flag': False,        # legacy compat — True if ANY flag fires
            'ml_down_risk_score': 0.0,
            'ml_down_risk_bucket': 'LOW',
            'ml_gmm_confirms_direction': False, # legacy — True only if clean (both False)
            'ml_gmm_regime_used': 'BOTH',
            'ml_down_risk_available': False,
        }

        if signal not in ('UP', 'FLAT', 'DOWN'):
            return defaults

        try:
            # Thread-safe lazy-load and inference (PyTorch VAE is NOT thread-safe)
            with self._down_risk_lock:
                # Lazy-load detector on first call
                if not self._down_risk_loaded:
                    self._down_risk_loaded = True
                    try:
                        from .down_risk_detector import DownRiskDetector
                        det = DownRiskDetector()
                        if det.load():
                            self._down_risk_detector = det
                            self._log("🛡️ Down-Risk Detector: LOADED (VAE+GMM) — DUAL REGIME")
                        else:
                            self._log("🛡️ Down-Risk Detector: Not trained yet (run down_risk_detector train)")
                    except Exception as e:
                        self._log(f"🛡️ Down-Risk Detector: Load failed ({e})")

                if self._down_risk_detector is None:
                    return defaults

                # Extract feature vector from the prediction's raw features
                features = pred.get('_features_array')  # set during predict()
                if features is None:
                    return defaults

                # Run BOTH regime models for full 18-case coverage
                up_result = self._down_risk_detector.predict_single(features, 'UP')
                down_result = self._down_risk_detector.predict_single(features, 'DOWN')

            # Post-processing outside lock (no shared state)
            up_flag = bool(up_result['down_risk_flag'][0])
            up_score = float(up_result['anomaly_score'][0])
            up_bucket = str(up_result['confidence_bucket'][0])

            down_flag = bool(down_result['down_risk_flag'][0])
            down_score = float(down_result['anomaly_score'][0])

            # Legacy compat: any flag fires → legacy dr_flag=True
            any_flag = up_flag or down_flag
            # Legacy score: max of both (used by downstream dr_score gates)
            legacy_score = max(up_score, down_score)
            # Legacy confirms: True only when clean (no anomaly from either model)
            gmm_confirms = not any_flag

            return {
                'ml_up_flag': up_flag,             # UP regime: True = crash risk (predicts DOWN)
                'ml_down_flag': down_flag,         # DOWN regime: True = bounce risk (predicts UP)
                'ml_up_score': up_score,           # UP regime anomaly score
                'ml_down_score': down_score,       # DOWN regime anomaly score
                'ml_down_risk_flag': any_flag,     # legacy compat — True if ANY flag fires
                'ml_down_risk_score': legacy_score, # legacy compat — max score
                'ml_down_risk_bucket': up_bucket,  # legacy compat
                'ml_gmm_confirms_direction': gmm_confirms, # legacy — True only if clean
                'ml_gmm_regime_used': 'BOTH',
                'ml_down_risk_available': True,
            }
        except Exception as e:
            self._log(f"⚠️ Down-Risk scoring failed: {e}")
            return defaults
    
    def _titan_defaults(self) -> dict:
        """Safe CONSERVATIVE defaults on total ML failure.
        
        These values BLOCK trading — Titan must never trade on crashed-ML
        with optimistic defaults. Every field is set to the most conservative
        value so no gate passes by accident.
        """
        return {
            'ml_score_boost': 0,
            'ml_prob_up': 0.33,
            'ml_prob_down': 0.33,
            'ml_prob_flat': 0.34,
            'ml_move_prob': 0.0,           # No movement detected → blocks gate
            'ml_direction_bias': 0.0,
            'ml_signal': 'UNKNOWN',
            'ml_confidence': 0.0,           # Zero confidence → blocks elite
            'ml_sizing_factor': 1.0,
            'ml_entry_caution': True,        # Caution ON when ML fails
            'ml_chop_hint': True,            # Assume chop when ML fails
            'ml_elite_ok': False,            # Block elite trades on ML crash
            'ml_direction_hint': 'NEUTRAL',
            'ml_gpt_summary': '',
            'ml_oi_stale': False,
            'ml_oi_available': False,
            'ml_down_risk_flag': True,       # Assume risk when ML fails (legacy compat)
            'ml_up_flag': True,              # Assume crash risk when ML fails
            'ml_down_flag': True,            # Assume bounce risk when ML fails
            'ml_down_risk_score': 0.5,       # Neutral risk, not zero
            'ml_down_risk_bucket': 'MEDIUM', # Neutral bucket, not LOW
            'ml_gmm_confirms_direction': False,
            'ml_gmm_regime_used': None,
            'ml_down_risk_available': False,  # Flag: ML didn't actually run
        }
    
    def predict_batch(self, stock_candles: dict, stock_daily: dict = None,
                       stock_oi: dict = None, stock_futures_oi: dict = None) -> dict:
        """Predict for multiple stocks.
        
        Args:
            stock_candles: {symbol: candles_df, ...}
            stock_daily: {symbol: daily_df, ...} optional daily data
            stock_oi: {symbol: oi_df, ...} optional OI snapshot data
            stock_futures_oi: {symbol: futures_oi_df, ...} optional futures OI data
            
        Returns:
            {symbol: prediction_dict, ...}
        """
        results = {}
        for symbol, candles in stock_candles.items():
            daily = stock_daily.get(symbol) if stock_daily else None
            oi = stock_oi.get(symbol) if stock_oi else None
            fut_oi = stock_futures_oi.get(symbol) if stock_futures_oi else None
            pred = self.predict(candles, daily_df=daily, oi_df=oi, futures_oi_df=fut_oi)
            if pred:
                results[symbol] = pred
        return results
    
    def get_top_movers(self, stock_candles: dict, min_prob: float = 0.5,
                       stock_daily: dict = None) -> list:
        """Get stocks with high MOVE probability, sorted by probability descending.
        
        Args:
            stock_candles: {symbol: candles_df, ...}
            min_prob: Minimum move probability threshold
            stock_daily: Optional daily data dict
            
        Returns:
            List of (symbol, prediction_dict) sorted by move probability descending
        """
        predictions = self.predict_batch(stock_candles, stock_daily)
        
        filtered = [
            (sym, pred) for sym, pred in predictions.items()
            if pred.get('ml_move_prob', 0) >= min_prob
        ]
        
        filtered.sort(key=lambda x: x[1]['ml_move_prob'], reverse=True)
        return filtered
    
    def apply_oi_overlay(self, ml_pred: dict, oi_data: dict) -> dict:
        """Post-ML overlay: adjust probabilities using live OI flow data.
        
        FAIL-SAFE: Returns ml_pred unchanged on ANY error.
        
        Logic (3-class aware):
        - OI BULLISH on ML UP → boost prob_up by up to 10%
        - OI BEARISH on ML DOWN → boost prob_down by up to 10%
        - OI contradicts direction → dampen directional prob by up to 8%
        - OI NEUTRAL + ML FLAT → strengthen flat conviction
        
        Also adds:
        - oi_adjusted: True/False
        - oi_flow_bias: BULLISH/BEARISH/NEUTRAL
        - oi_pcr: put-call ratio
        - oi_gpt_line: summary for GPT
        
        Args:
            ml_pred: dict from get_titan_signals()
            oi_data: dict from OptionsFlowAnalyzer.analyze()
            
        Returns:
            Updated ml_pred dict
        """
        try:
            if not oi_data or not ml_pred:
                ml_pred['oi_adjusted'] = False
                return ml_pred
            
            flow_bias = oi_data.get('flow_bias', 'NEUTRAL')
            flow_confidence = oi_data.get('flow_confidence', 0.0)
            pcr_oi = oi_data.get('pcr_oi', 1.0)
            oi_boost = oi_data.get('flow_score_boost', 0)
            oi_gpt_line = oi_data.get('flow_gpt_line', '')
            
            ml_signal = ml_pred.get('ml_signal', 'UNKNOWN')
            prob_up = ml_pred.get('ml_prob_up', 0.33)
            prob_down = ml_pred.get('ml_prob_down', 0.33)
            prob_flat = ml_pred.get('ml_prob_flat', 0.34)
            
            adjustment = 0.0
            reason = ''
            
            if ml_signal == 'UP' and flow_bias == 'BULLISH' and flow_confidence >= 0.55:
                # OI confirms UP → boost up probability
                adjustment = min(0.10, flow_confidence * 0.12)
                prob_up = min(1.0, prob_up + adjustment)
                prob_flat = max(0.0, prob_flat - adjustment * 0.6)
                prob_down = max(0.0, prob_down - adjustment * 0.4)
                reason = 'OI_CONFIRMS_UP'
            elif ml_signal == 'DOWN' and flow_bias == 'BEARISH' and flow_confidence >= 0.55:
                adjustment = min(0.10, flow_confidence * 0.12)
                prob_down = min(1.0, prob_down + adjustment)
                prob_flat = max(0.0, prob_flat - adjustment * 0.6)
                prob_up = max(0.0, prob_up - adjustment * 0.4)
                reason = 'OI_CONFIRMS_DOWN'
            elif ml_signal == 'UP' and flow_bias == 'BEARISH' and flow_confidence >= 0.60:
                adjustment = -0.08
                prob_up = max(0.0, prob_up + adjustment)
                prob_flat = min(1.0, prob_flat - adjustment)
                reason = 'OI_CONTRADICTS_UP'
            elif ml_signal == 'DOWN' and flow_bias == 'BULLISH' and flow_confidence >= 0.60:
                adjustment = -0.08
                prob_down = max(0.0, prob_down + adjustment)
                prob_flat = min(1.0, prob_flat - adjustment)
                reason = 'OI_CONTRADICTS_DOWN'
            elif ml_signal == 'FLAT' and flow_bias == 'NEUTRAL':
                adjustment = 0.05
                prob_flat = min(1.0, prob_flat + adjustment)
                prob_up = max(0.0, prob_up - adjustment * 0.5)
                prob_down = max(0.0, prob_down - adjustment * 0.5)
                reason = 'OI_CONFIRMS_FLAT'
            elif ml_signal == 'FLAT' and flow_bias in ('BULLISH', 'BEARISH') and flow_confidence >= 0.60:
                adjustment = 0.08
                if flow_bias == 'BULLISH':
                    prob_up = min(1.0, prob_up + adjustment)
                else:
                    prob_down = min(1.0, prob_down + adjustment)
                prob_flat = max(0.0, prob_flat - adjustment)
                reason = f'OI_BREAKOUT_HINT({flow_bias})'
            
            # Normalize probabilities
            total = prob_up + prob_down + prob_flat
            if total > 0:
                prob_up /= total
                prob_down /= total
                prob_flat /= total
            
            # Update prediction dict
            ml_pred['ml_prob_up'] = round(prob_up, 4)
            ml_pred['ml_prob_down'] = round(prob_down, 4)
            ml_pred['ml_prob_flat'] = round(prob_flat, 4)
            ml_pred['ml_move_prob'] = round(prob_up + prob_down, 4)
            ml_pred['ml_direction_bias'] = round(prob_up - prob_down, 4)
            ml_pred['oi_adjusted'] = abs(adjustment) > 0.001
            ml_pred['oi_adjustment'] = round(adjustment, 4)
            ml_pred['oi_adjustment_reason'] = reason
            ml_pred['oi_flow_bias'] = flow_bias
            ml_pred['oi_pcr'] = pcr_oi
            ml_pred['oi_gpt_line'] = oi_gpt_line
            
            # Re-derive signals from adjusted probabilities
            if ml_pred['oi_adjusted']:
                max_dir = max(prob_up, prob_down)
                if max_dir >= 0.50:
                    ml_pred['ml_sizing_factor'] = 1.3
                elif max_dir >= 0.40:
                    ml_pred['ml_sizing_factor'] = 1.2
                elif max_dir >= 0.30:
                    ml_pred['ml_sizing_factor'] = 1.0
                elif prob_flat >= 0.65:
                    ml_pred['ml_sizing_factor'] = 0.7
                else:
                    ml_pred['ml_sizing_factor'] = 1.0
                
                direction_bias = prob_up - prob_down
                if prob_up >= 0.40:
                    ml_pred['ml_gpt_summary'] = f"\U0001f9e0ML:UP({prob_up:.0%})+{oi_gpt_line}"
                elif prob_down >= 0.40:
                    ml_pred['ml_gpt_summary'] = f"\U0001f9e0ML:DOWN({prob_down:.0%})+{oi_gpt_line}"
                else:
                    ml_pred['ml_gpt_summary'] = f"\U0001f9e0ML:FLAT({prob_flat:.0%})+{oi_gpt_line}"
            
            return ml_pred
            
        except Exception:
            ml_pred['oi_adjusted'] = False
            return ml_pred
    
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# Singleton for Titan integration (lazy loaded)
_predictor_instance = None


def get_predictor(logger=None) -> MovePredictor:
    """Get or create singleton MovePredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MovePredictor(logger=logger)
    return _predictor_instance


if __name__ == '__main__':
    # Quick test
    predictor = MovePredictor()
    
    if predictor.ready:
        # Generate synthetic test candles
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2026-02-14 09:15', periods=n, freq='5min')
        price = 100 + np.cumsum(np.random.randn(n) * 0.3)
        
        test_candles = pd.DataFrame({
            'date': dates,
            'open': price + np.random.randn(n) * 0.1,
            'high': price + abs(np.random.randn(n) * 0.5),
            'low': price - abs(np.random.randn(n) * 0.5),
            'close': price,
            'volume': np.random.randint(10000, 500000, n)
        })
        
        result = predictor.predict(test_candles)
        print(f"\nPrediction result (3-class: UP/DOWN/FLAT):")
        for k, v in result.items():
            print(f"  {k}: {v}")
        
        titan = predictor.get_titan_signals(test_candles)
        print(f"\nTitan signals:")
        for k, v in titan.items():
            print(f"  {k}: {v}")
    else:
        print("Model not trained yet. Run: python -m ml_models.trainer")
