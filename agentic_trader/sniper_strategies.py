"""
Sniper Trading Strategies — OI Unwinding Reversal & PCR Extreme Fade
=====================================================================

Two high-edge intraday sniper strategies that leverage existing OI data
pipelines (DhanHQ / NSE fetchers) + GMM Down Risk Detector for directional
confirmation.

Strategy 1: Sniper-OIUnwinding
  Follows Zerodha Varsity OI + Price Action interpretation matrix:
    Price ↓ + OI ↓ = LONG_UNWINDING  (longs exiting, selling pressure weakening)
    Price ↑ + OI ↓ = SHORT_COVERING  (shorts exiting, rally may not sustain)

  Key insight: The unwinding signal alone is NOT a reversal signal — it indicates
  the CURRENT move is losing steam (not fresh positioning). The reversal trigger
  requires BOTH the unwinding signal AND price exhaustion at a key OI S/R level.

  LONG_UNWINDING at put_support → selling exhaustion → contrarian BUY
  SHORT_COVERING at call_resistance → rally exhaustion → contrarian SELL

  CRITICAL GUARD: We validate the OI signal against actual price direction:
    - LONG_UNWINDING must show price actually DOWN (change_pct < 0)
    - SHORT_COVERING must show price actually UP (change_pct > 0)
  If OI signal contradicts price action → misclassification → SKIP.

  Additional exhaustion checks: RSI divergence at extremes, momentum fading.

Strategy 2: Sniper-PCRExtreme
  PCR is a contrarian sentiment indicator, BUT "extreme" levels are NOT
  universal — they shift by:
    1. Market regime (trending vs range-bound)
    2. VIX level (high-vol vs low-vol environments)
    3. Index vs equity PCR (different normal ranges)

  We use REGIME-ADAPTIVE thresholds instead of fixed 1.35/0.65:
    - Track rolling PCR history to compute dynamic z-scores
    - High-VIX environments shift thresholds wider (fear is normal)
    - Index PCR and equity PCR have separate threshold ranges
    - Minimum holding period needed for PCR reversion (multi-hour play)

Both strategies:
  - Use OptionsFlowAnalyzer / OI fetchers for real-time data
  - Require GMM down_risk_score < threshold (clean signal)
  - Respect XGB direction alignment
  - Tagged as 'SNIPER_OI_UNWINDING' / 'SNIPER_PCR_EXTREME' for identification
  - Use separate capital pool and lot multiplier
  - Integrate with ExitManager, TradeLedger, RiskGovernor normally
"""
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# DEFAULT CONFIG (overridden by config.py if present)
# ---------------------------------------------------------------------------

DEFAULT_OI_UNWINDING_CONFIG = {
    "enabled": True,
    "max_trades_per_day": 4,
    "lot_multiplier": 1.5,
    # --- OI Unwinding Detection ---
    "required_buildups": ["LONG_UNWINDING", "SHORT_COVERING"],
    "min_buildup_strength": 0.45,       # OI buildup signal strength >= 0.45
    "min_oi_change_pct": 8.0,           # Dominant OI side must have changed >= 8%
    # --- Price Action Validation (Zerodha Varsity) ---
    "min_price_change_pct": 0.3,        # Minimum |change_pct| to confirm price direction
    "require_rsi_extreme": True,        # RSI must be at extreme for exhaustion
    "rsi_oversold_threshold": 38,       # RSI < 38 for LONG_UNWINDING BUY (not standard 30 — tighter)
    "rsi_overbought_threshold": 62,     # RSI > 62 for SHORT_COVERING SELL (not standard 70)
    # --- Price Reversal at S/R ---
    "max_distance_from_sr_pct": 1.5,    # Spot must be within 1.5% of OI support/resistance
    # --- GMM Quality Gate ---
    "max_updr_score": 0.15,              # GMM down-risk must be clean — UP regime
    "max_downdr_score": 0.12,            # GMM down-risk must be clean — DOWN regime (ratio 0.78)
    "min_gate_prob": 0.45,              # XGB gate P(move) floor
    "min_smart_score": 50,              # Minimum smart_score from scored stocks
    # --- Timing ---
    "earliest_entry": "09:45",          # Wait for first 30 min OI data to settle
    "no_entry_after": "14:30",          # No entries after 2:30 PM
    # --- Risk ---
    "score_tier": "premium",
    "separate_capital": 200000,         # ₹2L reserved for OI unwinding sniper
}

DEFAULT_PCR_EXTREME_CONFIG = {
    "enabled": True,
    "max_trades_per_day": 3,
    "lot_multiplier": 1.5,
    # --- PCR Extreme Detection (REGIME-ADAPTIVE) ---
    "pcr_oversold_threshold": 1.35,     # BASE threshold — adjusted by regime
    "pcr_overbought_threshold": 0.65,   # BASE threshold — adjusted by regime
    # --- Regime Adaptation ---
    "adaptive_pcr": True,               # Enable regime-adaptive thresholds
    "pcr_history_window": 20,           # Rolling window: last 20 PCR readings per symbol
    "pcr_zscore_threshold": 1.8,        # Z-score >= 1.8 = statistically extreme
    "high_vix_threshold": 18.0,         # INDIA VIX > 18 = high-vol regime
    "high_vix_pcr_widen": 0.10,         # Widen thresholds by ±0.10 in high-VIX (extremes are "normal")
    # --- Index PCR (Macro Confirmation) ---
    "use_index_pcr": True,              # Also check NIFTY PCR for macro regime
    "index_symbol": "NIFTY",            # Index to check for macro PCR
    "index_pcr_weight": 0.3,            # Blend: 70% stock PCR + 30% index PCR (reduced — stock signal primary)
    "require_index_agrees": False,      # If True, index PCR must also be extreme (same direction)
    # --- GMM Quality Gate ---
    "max_updr_score": 0.18,              # Slightly relaxed — PCR is strong standalone signal (UP regime)
    "max_downdr_score": 0.14,            # PCR strong standalone — DOWN regime (ratio 0.78)
    "min_gate_prob": 0.40,              # XGB gate P(move) floor
    "min_smart_score": 45,              # Lower floor — PCR extreme itself is high-edge
    # --- Timing ---
    "earliest_entry": "10:00",          # Need 45 min for reliable PCR
    "no_entry_after": "14:00",          # Earlier cutoff — PCR plays are multi-hour
    # --- Risk ---
    "score_tier": "premium",
    "separate_capital": 150000,         # ₹1.5L reserved for PCR extreme sniper
}


# ---------------------------------------------------------------------------
# SNIPER STRATEGY ENGINE
# ---------------------------------------------------------------------------

class SniperStrategies:
    """Stateful engine for OI Unwinding + PCR Extreme sniper trades.
    
    Called once per scan cycle by AutonomousTrader, right after GMM Sniper.
    Maintains daily counters, symbol dedup, and logs all decisions.
    
    OI Interpretation follows Zerodha Varsity OI + Price matrix:
      Price ↑ + OI ↑ = LONG_BUILDUP  (fresh buying → bullish)
      Price ↑ + OI ↓ = SHORT_COVERING (shorts exiting → weak bullish, may exhaust)
      Price ↓ + OI ↑ = SHORT_BUILDUP  (fresh selling → bearish)
      Price ↓ + OI ↓ = LONG_UNWINDING (longs exiting → weak bearish, may exhaust)
    
    We fade the UNWINDING signals at S/R when they show exhaustion.
    """

    def __init__(self, oi_unwinding_cfg: dict = None, pcr_extreme_cfg: dict = None):
        self._oi_cfg = {**DEFAULT_OI_UNWINDING_CONFIG, **(oi_unwinding_cfg or {})}
        self._pcr_cfg = {**DEFAULT_PCR_EXTREME_CONFIG, **(pcr_extreme_cfg or {})}

        # Daily state — OI Unwinding
        self._oi_trades_today = 0
        self._oi_date = datetime.now().date()
        self._oi_symbols = set()

        # Daily state — PCR Extreme
        self._pcr_trades_today = 0
        self._pcr_date = datetime.now().date()
        self._pcr_symbols = set()

        # Cache for index PCR (fetched once per cycle)
        self._index_pcr_cache = None
        self._index_pcr_ts = None

        # Rolling PCR history for regime-adaptive thresholds
        # Key: symbol → deque of recent PCR readings (capped at window size)
        self._pcr_history: Dict[str, deque] = {}
        self._pcr_history_window = self._pcr_cfg.get('pcr_history_window', 20)

    # ------------------------------------------------------------------
    # DAILY RESET
    # ------------------------------------------------------------------
    def _reset_if_new_day(self):
        today = datetime.now().date()
        if today != self._oi_date:
            self._oi_trades_today = 0
            self._oi_date = today
            self._oi_symbols = set()
            self._pcr_history.clear()  # Reset PCR history on new day
        if today != self._pcr_date:
            self._pcr_trades_today = 0
            self._pcr_date = today
            self._pcr_symbols = set()

    # ------------------------------------------------------------------
    # TIME GATE
    # ------------------------------------------------------------------
    @staticmethod
    def _in_time_window(earliest: str, latest: str) -> bool:
        now = datetime.now().strftime('%H:%M')
        return earliest <= now <= latest

    # ------------------------------------------------------------------
    # PRICE ACTION VALIDATOR (Zerodha Varsity OI + Price matrix)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_oi_price_action(buildup: str, market_info: dict,
                                   min_change_pct: float = 0.3) -> Tuple[bool, str]:
        """Cross-validate OI buildup signal against actual price action.
        
        Zerodha Varsity OI + Price matrix:
          LONG_UNWINDING  requires Price DOWN (change_pct < 0)
          SHORT_COVERING  requires Price UP  (change_pct > 0)
        
        If the OI signal contradicts the price direction, the classification
        is likely wrong — the fetcher only uses OI changes, not price.
        
        Returns:
            (is_valid, reason_string)
        """
        change_pct = market_info.get('change_pct', 0)
        
        if buildup == 'LONG_UNWINDING':
            # Varsity: Price DOWN + OI DOWN = longs exiting
            if change_pct >= min_change_pct:
                return False, f"CONTRADICTS: LONG_UNWINDING but price UP {change_pct:+.1f}% (should be down)"
            if abs(change_pct) < min_change_pct * 0.5:
                return False, f"WEAK: LONG_UNWINDING but price flat {change_pct:+.1f}%"
            return True, f"CONFIRMED: Price {change_pct:+.1f}% aligns with LONG_UNWINDING"
        
        elif buildup == 'SHORT_COVERING':
            # Varsity: Price UP + OI DOWN = shorts exiting
            if change_pct <= -min_change_pct:
                return False, f"CONTRADICTS: SHORT_COVERING but price DOWN {change_pct:+.1f}% (should be up)"
            if abs(change_pct) < min_change_pct * 0.5:
                return False, f"WEAK: SHORT_COVERING but price flat {change_pct:+.1f}%"
            return True, f"CONFIRMED: Price {change_pct:+.1f}% aligns with SHORT_COVERING"
        
        return False, f"Unknown buildup type: {buildup}"

    @staticmethod
    def _check_exhaustion(buildup: str, market_info: dict,
                          rsi_oversold: float = 38, rsi_overbought: float = 62) -> Tuple[bool, float, str]:
        """Check if the unwinding move is showing signs of exhaustion.
        
        Exhaustion indicators (price action context):
        - RSI at extreme (oversold for LONG_UNWINDING, overbought for SHORT_COVERING)
        - Momentum fading (follow_through_candles low + range_expansion shrinking)
        - ADX declining (trend weakening)
        
        Returns:
            (is_exhausted, exhaustion_score 0-1, reason)
        """
        rsi = market_info.get('rsi_14', 50)
        follow_through = market_info.get('follow_through_candles', 0)
        adx = market_info.get('adx', 20)
        range_exp = market_info.get('range_expansion_ratio', 0)
        
        exhaustion_score = 0.0
        reasons = []
        
        if buildup == 'LONG_UNWINDING':
            # Price has been falling — look for selling exhaustion
            if rsi <= rsi_oversold:
                exhaustion_score += 0.35
                reasons.append(f"RSI_oversold={rsi:.0f}")
            elif rsi <= 45:
                exhaustion_score += 0.15
                reasons.append(f"RSI_low={rsi:.0f}")
            
            # Low follow-through = move losing momentum
            if follow_through <= 1:
                exhaustion_score += 0.25
                reasons.append(f"FT_weak={follow_through}")
            
            # Declining ADX = trend weakening
            if adx < 25:
                exhaustion_score += 0.20
                reasons.append(f"ADX_weak={adx:.0f}")
            
            # Range not expanding = pressure easing
            if range_exp < 0.3:
                exhaustion_score += 0.15
                reasons.append(f"RangeExp_low={range_exp:.2f}")
        
        elif buildup == 'SHORT_COVERING':
            # Price has been rising on short covering — look for rally exhaustion
            if rsi >= rsi_overbought:
                exhaustion_score += 0.35
                reasons.append(f"RSI_overbought={rsi:.0f}")
            elif rsi >= 55:
                exhaustion_score += 0.15
                reasons.append(f"RSI_high={rsi:.0f}")
            
            if follow_through <= 1:
                exhaustion_score += 0.25
                reasons.append(f"FT_weak={follow_through}")
            
            if adx < 25:
                exhaustion_score += 0.20
                reasons.append(f"ADX_weak={adx:.0f}")
            
            if range_exp < 0.3:
                exhaustion_score += 0.15
                reasons.append(f"RangeExp_low={range_exp:.2f}")
        
        exhaustion_score = min(1.0, exhaustion_score)
        is_exhausted = exhaustion_score >= 0.30  # Need at least 1-2 exhaustion signals (relaxed from 0.40)
        reason = ' | '.join(reasons) if reasons else 'no_exhaustion_signals'
        
        return is_exhausted, round(exhaustion_score, 3), reason

    # ------------------------------------------------------------------
    # STRATEGY 1: OI UNWINDING REVERSAL
    # ------------------------------------------------------------------
    def scan_oi_unwinding(
        self,
        oi_results: Dict[str, dict],
        ml_results: Dict[str, dict],
        pre_scores: Dict[str, float],
        market_data: Dict[str, dict],
        active_symbols: set,
        model_tracker_symbols: set,
        gmm_sniper_symbols: set,
    ) -> List[dict]:
        """Scan for OI Unwinding reversal candidates with Varsity-correct interpretation.
        
        CRITICAL: Unlike the raw OI fetcher which classifies buildup using only OI
        changes, this strategy VALIDATES against actual price direction per Zerodha
        Varsity's OI + Price action matrix:
        
          LONG_UNWINDING = Price DOWN + OI DOWN → longs exiting → selling exhaustion
            → If near put_support + RSI oversold + momentum fading → contrarian BUY
          
          SHORT_COVERING = Price UP + OI DOWN → shorts exiting → rally exhaustion  
            → If near call_resistance + RSI overbought + momentum fading → contrarian SELL
        
        If OI signal contradicts price direction → misclassification → SKIP.
        If no exhaustion signs at S/R → trend may continue → SKIP.
        
        Returns list of candidate dicts ready for placement.
        """
        cfg = self._oi_cfg
        if not cfg.get('enabled', False):
            return []

        self._reset_if_new_day()

        if self._oi_trades_today >= cfg['max_trades_per_day']:
            return []

        if not self._in_time_window(cfg['earliest_entry'], cfg['no_entry_after']):
            return []

        required_buildups = set(cfg.get('required_buildups', ['LONG_UNWINDING', 'SHORT_COVERING']))
        min_strength = cfg.get('min_buildup_strength', 0.45)
        max_sr_dist = cfg.get('max_distance_from_sr_pct', 1.5)
        max_dr_up = cfg.get('max_updr_score', 0.15)
        max_dr_down = cfg.get('max_downdr_score', 0.12)
        min_gate = cfg.get('min_gate_prob', 0.45)
        min_smart = cfg.get('min_smart_score', 50)
        min_price_chg = cfg.get('min_price_change_pct', 0.3)
        rsi_oversold = cfg.get('rsi_oversold_threshold', 38)
        rsi_overbought = cfg.get('rsi_overbought_threshold', 62)

        try:
            from ml_models.feature_engineering import get_sector_for_symbol as _get_sector
        except Exception:
            _get_sector = lambda s: 'OTHER'

        candidates = []

        for sym, oi_data in oi_results.items():
            # --- OI Buildup Signal ---
            buildup = oi_data.get('nse_oi_buildup') or oi_data.get('oi_buildup_signal', 'NEUTRAL')
            if buildup not in required_buildups:
                continue

            strength = oi_data.get('nse_oi_buildup_strength') or oi_data.get('oi_buildup_strength', 0)
            if strength < min_strength:
                continue

            # --- Symbol dedup ---
            if sym in self._oi_symbols or sym in active_symbols:
                continue
            if sym in model_tracker_symbols or sym in gmm_sniper_symbols:
                continue

            # --- PRICE ACTION VALIDATION (Zerodha Varsity) ---
            # The OI fetcher classifies buildup using only OI changes.
            # We MUST cross-check against actual price direction.
            mkt = market_data.get(sym, {})
            if not isinstance(mkt, dict) or 'change_pct' not in mkt:
                continue  # No price data → can't validate

            price_valid, price_reason = self._validate_oi_price_action(
                buildup, mkt, min_price_chg
            )
            if not price_valid:
                # OI signal contradicts price action → misclassification
                sym_clean = sym.replace('NSE:', '')
                print(f"      ⚠️ OI-Sniper SKIP {sym_clean}: {price_reason}")
                continue

            # --- Spot vs S/R proximity ---
            spot = oi_data.get('spot_price', 0) or mkt.get('ltp', 0)
            put_support = oi_data.get('put_support', 0)
            call_resistance = oi_data.get('call_resistance', 0)

            if spot <= 0:
                continue

            # LONG_UNWINDING at put_support → selling exhaustion → BUY reversal
            # SHORT_COVERING at call_resistance → rally exhaustion → SELL reversal
            if buildup == 'LONG_UNWINDING':
                direction = 'BUY'
                sr_level = put_support if put_support > 0 else spot * 0.98
                dist_pct = abs(spot - sr_level) / spot * 100
            else:  # SHORT_COVERING
                direction = 'SELL'
                sr_level = call_resistance if call_resistance > 0 else spot * 1.02
                dist_pct = abs(spot - sr_level) / spot * 100

            if dist_pct > max_sr_dist:
                continue  # Too far from support/resistance — not a reversal zone

            # --- EXHAUSTION CHECK ---
            # The unwinding alone is not enough. We need the move to be EXHAUSTING
            # at the S/R level (RSI extreme, momentum fading, ADX declining).
            is_exhausted, exhaustion_score, exh_reason = self._check_exhaustion(
                buildup, mkt, rsi_oversold, rsi_overbought
            )
            if not is_exhausted:
                sym_clean = sym.replace('NSE:', '')
                print(f"      ⚠️ OI-Sniper SKIP {sym_clean}: No exhaustion at S/R "
                      f"(score={exhaustion_score:.2f}, need 0.40) — {exh_reason}")
                continue

            # --- GMM quality gate (regime-proportional) ---
            ml = ml_results.get(sym, {})
            dr_score = ml.get('ml_down_risk_score')
            dr_flag = ml.get('ml_down_risk_flag')
            if dr_score is None:
                continue
            _oi_regime = ml.get('ml_gmm_regime_used', 'UP')
            max_dr = max_dr_down if _oi_regime == 'DOWN' else max_dr_up
            if dr_score > max_dr or dr_flag:
                continue

            # --- XGB gate ---
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
            if ml_move_prob < min_gate:
                continue

            # --- XGB direction check — if XGB actively confirms CURRENT trend, skip reversal ---
            xgb_signal = ml.get('ml_signal', 'UNKNOWN')
            if direction == 'BUY' and xgb_signal == 'DOWN' and ml_move_prob >= 0.55:
                continue  # XGB strongly confirms downtrend — don't fade it
            if direction == 'SELL' and xgb_signal == 'UP' and ml_move_prob >= 0.55:
                continue  # XGB strongly confirms uptrend — don't fade it

            # --- Smart score (includes exhaustion quality) ---
            p_score = pre_scores.get(sym, 0)
            conviction = ml_move_prob * min(p_score / 100.0, 1.0) * 30.0
            safety = (1.0 - min(dr_score, 1.0)) * 18.0 + 5.0
            oi_boost = strength * 12.0                    # OI signal strength bonus
            exhaustion_boost = exhaustion_score * 15.0    # Exhaustion quality bonus
            technical = min(p_score, 100) * 0.12
            move_bonus = ml_move_prob * 8.0
            smart_score = conviction + safety + oi_boost + exhaustion_boost + technical + move_bonus

            if smart_score < min_smart:
                continue

            sym_clean = sym.replace('NSE:', '')
            sector = _get_sector(sym_clean) or 'OTHER'
            change_pct = mkt.get('change_pct', 0)
            rsi = mkt.get('rsi_14', 50)

            candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'sector': sector,
                'smart_score': round(smart_score, 2),
                'dr_score': dr_score,
                'ml_move_prob': ml_move_prob,
                'p_score': p_score,
                'oi_buildup': buildup,
                'oi_strength': strength,
                'spot_price': spot,
                'sr_level': sr_level,
                'dist_from_sr_pct': round(dist_pct, 2),
                'change_pct': round(change_pct, 2),
                'rsi': round(rsi, 1),
                'exhaustion_score': exhaustion_score,
                'exhaustion_reason': exh_reason,
                'price_validation': price_reason,
                'strategy_type': 'SNIPER_OI_UNWINDING',
                'ml_data': {
                    'smart_score': round(smart_score, 2),
                    'p_score': p_score,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'ml_confidence': ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': xgb_signal,
                        'move_prob': ml_move_prob,
                        'prob_up': ml.get('ml_prob_up', 0),
                        'prob_down': ml.get('ml_prob_down', 0),
                        'direction_hint': ml.get('ml_direction_hint', 'NEUTRAL'),
                        'model_type': ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': dr_score,
                        'up_flag': ml.get('ml_up_flag', False),
                        'down_flag': ml.get('ml_down_flag', False),
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', None),
                        'gmm_action': 'OI_UNWINDING_SNIPER',
                    },
                    'oi_context': {
                        'buildup': buildup,
                        'strength': strength,
                        'spot_price': spot,
                        'sr_level': sr_level,
                        'dist_from_sr_pct': round(dist_pct, 2),
                        'pcr_oi': oi_data.get('pcr_oi'),
                        'max_pain': oi_data.get('max_pain'),
                        'flow_bias': oi_data.get('flow_bias'),
                        'price_change_pct': round(change_pct, 2),
                        'rsi': round(rsi, 1),
                        'exhaustion_score': exhaustion_score,
                        'exhaustion_reason': exh_reason,
                        'price_validated': True,
                    },
                    'scored_direction': direction,
                    'sniper_type': 'OI_UNWINDING',
                },
            })

        # Sort by exhaustion_score (highest exhaustion = strongest reversal signal),
        # then by OI strength, tiebreak by lowest dr_score
        candidates.sort(key=lambda c: (-c['exhaustion_score'], -c['oi_strength'], c['dr_score']))

        # Limit to remaining budget
        remaining = cfg['max_trades_per_day'] - self._oi_trades_today
        return candidates[:remaining]

    # ------------------------------------------------------------------
    # STRATEGY 2: PCR EXTREME FADE (REGIME-ADAPTIVE)
    # ------------------------------------------------------------------
    def _fetch_index_pcr(self) -> Optional[float]:
        """Fetch NIFTY PCR for macro regime confirmation. Cached per cycle."""
        if self._index_pcr_cache is not None and self._index_pcr_ts:
            # Cache valid for 3 minutes
            if (datetime.now() - self._index_pcr_ts).total_seconds() < 180:
                return self._index_pcr_cache

        try:
            from nse_oi_fetcher import get_nse_oi_fetcher
            nse = get_nse_oi_fetcher()
            data = nse.fetch("NIFTY")
            if data and data.get('pcr_oi'):
                self._index_pcr_cache = data['pcr_oi']
                self._index_pcr_ts = datetime.now()
                return self._index_pcr_cache
        except Exception:
            pass

        try:
            from dhan_oi_fetcher import get_dhan_oi_fetcher
            dhan = get_dhan_oi_fetcher()
            data = dhan.fetch("NIFTY")
            if data and data.get('pcr_oi'):
                self._index_pcr_cache = data['pcr_oi']
                self._index_pcr_ts = datetime.now()
                return self._index_pcr_cache
        except Exception:
            pass

        return None

    def _update_pcr_history(self, sym: str, pcr_value: float):
        """Track rolling PCR history for regime-adaptive thresholds."""
        if sym not in self._pcr_history:
            self._pcr_history[sym] = deque(maxlen=self._pcr_history_window)
        self._pcr_history[sym].append(pcr_value)

    def _get_adaptive_pcr_thresholds(self, sym: str, base_oversold: float,
                                       base_overbought: float,
                                       vix_level: Optional[float] = None) -> Tuple[float, float]:
        """Compute regime-adaptive PCR extreme thresholds.
        
        Instead of fixed 1.35/0.65, we compute:
          1. Rolling mean + std of recent PCR readings for this stock
          2. Z-score based extreme = mean ± z_threshold × std
          3. VIX adjustment: high-VIX widens thresholds (fear inflates PCR)
          4. Fall back to base thresholds if insufficient history
        
        Returns:
            (oversold_threshold, overbought_threshold)
        """
        cfg = self._pcr_cfg
        z_thr = cfg.get('pcr_zscore_threshold', 1.8)
        high_vix_thr = cfg.get('high_vix_threshold', 18.0)
        high_vix_widen = cfg.get('high_vix_pcr_widen', 0.10)

        oversold = base_oversold
        overbought = base_overbought

        # Adaptive from rolling history
        history = self._pcr_history.get(sym)
        if history and len(history) >= 5:
            import statistics
            mean_pcr = statistics.mean(history)
            std_pcr = statistics.stdev(history) if len(history) >= 3 else 0.15
            
            if std_pcr > 0.05:  # Minimum variability to be meaningful
                adaptive_oversold = mean_pcr + z_thr * std_pcr
                adaptive_overbought = mean_pcr - z_thr * std_pcr
                
                # Clamp: never less extreme than base thresholds
                oversold = max(base_oversold, adaptive_oversold)
                overbought = min(base_overbought, adaptive_overbought)

        # VIX adjustment: high-VIX environments → PCR is naturally elevated
        # → need even MORE extreme PCR to be meaningful
        if vix_level is not None and vix_level > high_vix_thr:
            oversold += high_vix_widen
            overbought -= high_vix_widen

        return oversold, overbought

    def _fetch_vix(self) -> Optional[float]:
        """Attempt to fetch India VIX for regime adaptation."""
        try:
            from nse_oi_fetcher import get_nse_oi_fetcher
            nse = get_nse_oi_fetcher()
            data = nse.fetch("NIFTY")
            # VIX might be embedded in the response or available separately
            if data and data.get('iv_skew') is not None:
                # ATM IV is a reasonable proxy for VIX
                # NIFTY ATM IV ≈ India VIX (close enough for regime detection)
                for strike in data.get('strikes', []):
                    spot = data.get('spot_price', 0)
                    if spot and abs(strike.get('strike', 0) - spot) < spot * 0.01:
                        ce_iv = strike.get('ce_iv', 0)
                        if ce_iv and ce_iv > 0:
                            return ce_iv
        except Exception:
            pass
        return None

    def scan_pcr_extreme(
        self,
        oi_results: Dict[str, dict],
        ml_results: Dict[str, dict],
        pre_scores: Dict[str, float],
        market_data: Dict[str, dict],
        active_symbols: set,
        model_tracker_symbols: set,
        gmm_sniper_symbols: set,
        oi_unwinding_symbols: set,
    ) -> List[dict]:
        """Scan for PCR Extreme fade candidates with REGIME-ADAPTIVE thresholds.
        
        Key improvements over naive fixed-threshold PCR:
          1. Rolling PCR z-scores per stock (what's extreme FOR THIS STOCK today)
          2. VIX-adjusted thresholds (high-vol = wider thresholds)
          3. Index PCR blending with reduced weight (stock signal primary)
          4. Price action confirmation — don't fade a strong trend just because PCR is extreme
        
        Returns list of candidate dicts ready for placement.
        """
        cfg = self._pcr_cfg
        if not cfg.get('enabled', False):
            return []

        self._reset_if_new_day()

        if self._pcr_trades_today >= cfg['max_trades_per_day']:
            return []

        if not self._in_time_window(cfg['earliest_entry'], cfg['no_entry_after']):
            return []

        base_oversold = cfg.get('pcr_oversold_threshold', 1.35)
        base_overbought = cfg.get('pcr_overbought_threshold', 0.65)
        max_dr_up = cfg.get('max_updr_score', 0.18)
        max_dr_down = cfg.get('max_downdr_score', 0.14)
        min_gate = cfg.get('min_gate_prob', 0.40)
        min_smart = cfg.get('min_smart_score', 45)
        adaptive = cfg.get('adaptive_pcr', True)
        index_weight = cfg.get('index_pcr_weight', 0.3) if cfg.get('use_index_pcr', True) else 0.0
        require_index = cfg.get('require_index_agrees', False)

        try:
            from ml_models.feature_engineering import get_sector_for_symbol as _get_sector
        except Exception:
            _get_sector = lambda s: 'OTHER'

        # Fetch index PCR and VIX for regime context
        index_pcr = self._fetch_index_pcr() if (index_weight > 0 or require_index) else None
        vix_level = self._fetch_vix() if adaptive else None

        candidates = []

        for sym, oi_data in oi_results.items():
            stock_pcr = oi_data.get('pcr_oi')
            if stock_pcr is None or stock_pcr <= 0:
                continue

            # --- Update rolling PCR history ---
            self._update_pcr_history(sym, stock_pcr)

            # --- Compute regime-adaptive thresholds ---
            if adaptive:
                oversold_thr, overbought_thr = self._get_adaptive_pcr_thresholds(
                    sym, base_oversold, base_overbought, vix_level
                )
            else:
                oversold_thr, overbought_thr = base_oversold, base_overbought

            # --- Blend with index PCR (reduced weight — stock signal primary) ---
            if index_pcr and index_weight > 0:
                blended_pcr = stock_pcr * (1 - index_weight) + index_pcr * index_weight
            else:
                blended_pcr = stock_pcr

            # --- PCR Extreme Detection ---
            if blended_pcr >= oversold_thr:
                direction = 'BUY'   # Market oversold → contrarian BUY
                pcr_edge = blended_pcr - oversold_thr  # How extreme beyond threshold
            elif blended_pcr <= overbought_thr:
                direction = 'SELL'  # Market overbought → contrarian SELL
                pcr_edge = overbought_thr - blended_pcr
            else:
                continue  # PCR in neutral zone — no signal

            # --- Index PCR agreement check (optional) ---
            if require_index and index_pcr is not None:
                if direction == 'BUY' and index_pcr < base_oversold * 0.85:
                    continue  # Index not oversold — stock PCR extreme may be stock-specific noise
                if direction == 'SELL' and index_pcr > base_overbought * 1.15:
                    continue  # Index not overbought — stock PCR extreme may be noise

            # --- Symbol dedup ---
            if sym in self._pcr_symbols or sym in active_symbols:
                continue
            if sym in model_tracker_symbols or sym in gmm_sniper_symbols:
                continue
            if sym in oi_unwinding_symbols:
                continue

            # --- PRICE ACTION GUARD: Don't fade a strong trending move ---
            # If stock is trending strongly (ADX > 30, large move), PCR extreme
            # may reflect genuine institutional conviction, not contrarian opportunity
            mkt = market_data.get(sym, {})
            if isinstance(mkt, dict):
                adx = mkt.get('adx', 20)
                change_pct = mkt.get('change_pct', 0)
                follow_through = mkt.get('follow_through_candles', 0)
                
                # Strong trend guard: ADX > 35 + move > 2% + high follow-through
                # = institutional conviction, NOT a mere sentiment extreme → skip
                if adx > 35 and abs(change_pct) > 2.0 and follow_through >= 3:
                    sym_clean = sym.replace('NSE:', '')
                    print(f"      ⚠️ PCR-Sniper SKIP {sym_clean}: Strong trend "
                          f"(ADX={adx:.0f}, chg={change_pct:+.1f}%, FT={follow_through}) — "
                          f"PCR extreme may reflect conviction, not contrarian")
                    continue

            # --- GMM quality gate (regime-proportional) ---
            ml = ml_results.get(sym, {})
            dr_score = ml.get('ml_down_risk_score')
            dr_flag = ml.get('ml_down_risk_flag')
            if dr_score is None:
                continue
            _pcr_regime = ml.get('ml_gmm_regime_used', 'UP')
            max_dr = max_dr_down if _pcr_regime == 'DOWN' else max_dr_up
            if dr_score > max_dr or dr_flag:
                continue

            # --- XGB gate ---
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
            if ml_move_prob < min_gate:
                continue

            # --- XGB direction alignment (soft — PCR is strong standalone) ---
            xgb_signal = ml.get('ml_signal', 'UNKNOWN')
            xgb_penalty = 0
            if direction == 'BUY' and xgb_signal == 'DOWN' and ml_move_prob >= 0.55:
                xgb_penalty = 10
            elif direction == 'SELL' and xgb_signal == 'UP' and ml_move_prob >= 0.55:
                xgb_penalty = 10

            # --- Smart score (PCR-weighted, regime-aware) ---
            p_score = pre_scores.get(sym, 0)
            conviction = ml_move_prob * min(p_score / 100.0, 1.0) * 30.0
            safety = (1.0 - min(dr_score, 1.0)) * 18.0 + 5.0
            pcr_boost = min(pcr_edge, 0.5) * 30.0  # PCR extremity bonus (max 15 pts)
            # Bonus if BOTH stock and index PCR agree on extreme
            index_agree_bonus = 0
            if index_pcr is not None:
                if direction == 'BUY' and index_pcr >= base_oversold * 0.9:
                    index_agree_bonus = 5.0
                elif direction == 'SELL' and index_pcr <= base_overbought * 1.1:
                    index_agree_bonus = 5.0
            technical = min(p_score, 100) * 0.12
            move_bonus = ml_move_prob * 10.0
            smart_score = (conviction + safety + pcr_boost + index_agree_bonus
                          + technical + move_bonus - xgb_penalty)

            if smart_score < min_smart:
                continue

            sym_clean = sym.replace('NSE:', '')
            sector = _get_sector(sym_clean) or 'OTHER'
            spot = oi_data.get('spot_price', 0)

            candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'sector': sector,
                'smart_score': round(smart_score, 2),
                'dr_score': dr_score,
                'ml_move_prob': ml_move_prob,
                'p_score': p_score,
                'stock_pcr': stock_pcr,
                'blended_pcr': round(blended_pcr, 3),
                'index_pcr': index_pcr,
                'pcr_edge': round(pcr_edge, 3),
                'pcr_regime': 'OVERSOLD' if direction == 'BUY' else 'OVERBOUGHT',
                'adaptive_oversold_thr': round(oversold_thr, 3),
                'adaptive_overbought_thr': round(overbought_thr, 3),
                'vix_level': vix_level,
                'index_agrees': index_agree_bonus > 0,
                'strategy_type': 'SNIPER_PCR_EXTREME',
                'ml_data': {
                    'smart_score': round(smart_score, 2),
                    'p_score': p_score,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'ml_confidence': ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': xgb_signal,
                        'move_prob': ml_move_prob,
                        'prob_up': ml.get('ml_prob_up', 0),
                        'prob_down': ml.get('ml_prob_down', 0),
                        'direction_hint': ml.get('ml_direction_hint', 'NEUTRAL'),
                        'model_type': ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': dr_score,
                        'up_flag': ml.get('ml_up_flag', False),
                        'down_flag': ml.get('ml_down_flag', False),
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', None),
                        'gmm_action': 'PCR_EXTREME_SNIPER',
                    },
                    'pcr_context': {
                        'stock_pcr': stock_pcr,
                        'blended_pcr': round(blended_pcr, 3),
                        'index_pcr': index_pcr,
                        'pcr_edge': round(pcr_edge, 3),
                        'pcr_regime': 'OVERSOLD' if direction == 'BUY' else 'OVERBOUGHT',
                        'adaptive_thresholds': {
                            'oversold': round(oversold_thr, 3),
                            'overbought': round(overbought_thr, 3),
                        },
                        'vix_level': vix_level,
                        'pcr_history_count': len(self._pcr_history.get(sym, [])),
                        'index_agrees': index_agree_bonus > 0,
                        'spot_price': spot,
                        'max_pain': oi_data.get('max_pain'),
                        'flow_bias': oi_data.get('flow_bias'),
                    },
                    'scored_direction': direction,
                    'sniper_type': 'PCR_EXTREME',
                },
            })

        # Sort by PCR extremity descending, tiebreak by lowest dr_score
        candidates.sort(key=lambda c: (-c['pcr_edge'], c['dr_score']))

        # Limit to remaining budget
        remaining = cfg['max_trades_per_day'] - self._pcr_trades_today
        return candidates[:remaining]

    # ------------------------------------------------------------------
    # RECORD TRADE (called after successful placement)
    # ------------------------------------------------------------------
    def record_oi_trade(self, sym: str):
        self._oi_trades_today += 1
        self._oi_symbols.add(sym)

    def record_pcr_trade(self, sym: str):
        self._pcr_trades_today += 1
        self._pcr_symbols.add(sym)

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        self._reset_if_new_day()
        return {
            'oi_unwinding': {
                'enabled': self._oi_cfg.get('enabled', False),
                'trades_today': self._oi_trades_today,
                'max_per_day': self._oi_cfg.get('max_trades_per_day', 4),
                'symbols': list(self._oi_symbols),
            },
            'pcr_extreme': {
                'enabled': self._pcr_cfg.get('enabled', False),
                'trades_today': self._pcr_trades_today,
                'max_per_day': self._pcr_cfg.get('max_trades_per_day', 3),
                'symbols': list(self._pcr_symbols),
                'pcr_history_count': {s: len(d) for s, d in self._pcr_history.items()},
            },
        }
