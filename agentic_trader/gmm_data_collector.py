"""
GMM Data Collector — Background data collection for UPDR/DownDR sweet-spot calibration.

Records ML metrics for every stock evaluated in each 5-min scan cycle,
then tracks forward price outcomes over the next 8 candles (40 minutes).

Data collected for ~30 days → analyze to find profitable UPDR v6 / DownDR v8 ranges.

Output: /home/ubuntu/titan/logs/gmm_calibration_data.jsonl  (one JSON object per line)
"""

import json
import os
import threading
from datetime import datetime, timedelta
from collections import defaultdict


class GMMDataCollector:
    """Collects GMM model metrics and forward price outcomes for calibration."""

    FORWARD_CANDLES = 8  # Track next 8 five-minute candles (40 min)

    def __init__(self, log_dir: str | None = None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        self._log_dir = log_dir
        self._data_file = os.path.join(log_dir, 'gmm_calibration_data.jsonl')
        self._pending_file = os.path.join(log_dir, 'gmm_pending_tracks.json')
        self._lock = threading.Lock()

        # pending_tracks: {track_id -> record_dict}
        # Each record has 'entry_price' and 'forward_prices' list to fill
        self._pending_tracks = {}
        self._load_pending()

    # ── PUBLIC API ──────────────────────────────────────────────────

    def record_scan(self, ml_results: dict, pre_scores: dict, market_data: dict,
                    cycle_time: str, cycle_decisions: dict | None = None):
        """
        Called once per scan cycle. Records GMM metrics for ALL stocks that had ML predictions.

        Args:
            ml_results: {symbol: {ml_up_score, ml_down_score, ml_move_prob, ...}}
            pre_scores: {symbol: smart_score}
            market_data: {symbol: {ltp, open, high, low, close, volume, ...}}
            cycle_time: HH:MM:SS string
            cycle_decisions: {symbol: {decision obj}} — optional, for direction info
        """
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        ts = now.strftime('%Y-%m-%d %H:%M:%S')

        with self._lock:
            for sym, ml in ml_results.items():
                # Skip stocks without ML predictions
                if ml.get('ml_up_score') is None and ml.get('ml_down_score') is None:
                    continue

                mkt = market_data.get(sym, {})
                ltp = mkt.get('ltp') or mkt.get('last_price') or mkt.get('close', 0)
                if not ltp or ltp <= 0:
                    continue

                # Direction from cycle decisions
                direction = None
                dir_confidence = 0
                if cycle_decisions:
                    cd = cycle_decisions.get(sym, {})
                    dec = cd.get('decision') if isinstance(cd, dict) else cd
                    if dec and hasattr(dec, 'recommended_direction'):
                        direction = dec.recommended_direction
                        dir_confidence = getattr(dec, 'direction_confidence', 0)

                track_id = f"{date_str}_{cycle_time}_{sym}"

                record = {
                    'track_id': track_id,
                    'date': date_str,
                    'scan_time': ts,
                    'cycle_time': cycle_time,
                    'symbol': sym,

                    # === GMM Metrics (the core data we want to calibrate) ===
                    'updr_score': round(ml.get('ml_up_score', -1), 6),
                    'downdr_score': round(ml.get('ml_down_score', -1), 6),
                    'up_flag': ml.get('ml_up_flag', False),
                    'down_flag': ml.get('ml_down_flag', False),
                    'dr_legacy': round(ml.get('ml_down_risk_score', -1), 6),

                    # === XGB / Direction Metrics ===
                    'xgb_signal': ml.get('ml_signal', 'UNKNOWN'),
                    'move_prob': round(ml.get('ml_move_prob', 0), 4),
                    'prob_up': round(ml.get('ml_prob_up', 0), 4),
                    'prob_down': round(ml.get('ml_prob_down', 0), 4),
                    'prob_flat': round(ml.get('ml_prob_flat', 0), 4),
                    'direction_hint': ml.get('ml_direction_hint', ''),
                    'gmm_confirms_dir': ml.get('ml_gmm_confirms_direction', False),

                    # === Scoring ===
                    'smart_score': pre_scores.get(sym, 0),
                    'ml_score_boost': ml.get('ml_score_boost', 0),
                    'ml_sizing_factor': round(ml.get('ml_sizing_factor', 1.0), 3),
                    'ml_elite_ok': ml.get('ml_elite_ok', False),
                    'ml_chop_hint': ml.get('ml_chop_hint', False),

                    # === Direction from Scorer ===
                    'scorer_direction': direction,
                    'scorer_dir_confidence': dir_confidence,

                    # === Entry Price ===
                    'entry_price': round(ltp, 2),

                    # === Forward Tracking (to be filled) ===
                    'forward_prices': [],     # [{candle_n, time, price, pct_change}]
                    'max_favorable': 0.0,     # Best % move in any direction
                    'max_adverse': 0.0,       # Worst % move against both directions
                    'max_up_pct': 0.0,        # Best upward move %
                    'max_down_pct': 0.0,      # Best downward move %  (negative)
                    'candles_remaining': self.FORWARD_CANDLES,
                    'outcome': 'PENDING',     # PENDING -> PROFIT_UP / PROFIT_DOWN / FLAT / LOSS_UP / LOSS_DOWN
                    'completed': False,
                }

                self._pending_tracks[track_id] = record

            self._save_pending()

    def update_forward_prices(self, market_data: dict):
        """
        Called every scan cycle. Updates forward prices for all pending tracks.

        Args:
            market_data: {symbol: {ltp, ...}} — current market quotes
        """
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        completed_ids = []

        with self._lock:
            for track_id, rec in list(self._pending_tracks.items()):
                if rec['completed']:
                    completed_ids.append(track_id)
                    continue

                sym = rec['symbol']
                mkt = market_data.get(sym, {})
                ltp = mkt.get('ltp') or mkt.get('last_price') or mkt.get('close', 0)

                if not ltp or ltp <= 0:
                    continue

                entry = rec['entry_price']
                if entry <= 0:
                    continue

                candle_n = len(rec['forward_prices']) + 1
                pct_change = round(((ltp - entry) / entry) * 100, 4)

                rec['forward_prices'].append({
                    'candle_n': candle_n,
                    'time': ts,
                    'price': round(ltp, 2),
                    'pct_change': pct_change,
                })

                # Update extremes
                if pct_change > rec['max_up_pct']:
                    rec['max_up_pct'] = round(pct_change, 4)
                if pct_change < rec['max_down_pct']:
                    rec['max_down_pct'] = round(pct_change, 4)

                rec['max_favorable'] = round(max(abs(rec['max_up_pct']), abs(rec['max_down_pct'])), 4)
                rec['max_adverse'] = round(min(rec['max_up_pct'], rec['max_down_pct']), 4)

                rec['candles_remaining'] = max(0, self.FORWARD_CANDLES - candle_n)

                # Check completion
                if candle_n >= self.FORWARD_CANDLES:
                    rec['completed'] = True
                    rec['outcome'] = self._classify_outcome(rec)
                    completed_ids.append(track_id)

                # Also check time-based expiry (>50 min from scan = definitely done)
                try:
                    scan_dt = datetime.strptime(rec['scan_time'], '%Y-%m-%d %H:%M:%S')
                    if (now - scan_dt) > timedelta(minutes=50):
                        rec['completed'] = True
                        if rec['outcome'] == 'PENDING':
                            rec['outcome'] = self._classify_outcome(rec)
                        if track_id not in completed_ids:
                            completed_ids.append(track_id)
                except Exception:
                    pass

            # Flush completed records to JSONL
            for tid in completed_ids:
                rec = self._pending_tracks.get(tid)
                if rec and rec['completed']:
                    self._write_record(rec)
                    del self._pending_tracks[tid]

            self._save_pending()

    def get_stats(self) -> dict:
        """Returns current collection stats."""
        with self._lock:
            total_pending = len(self._pending_tracks)
            total_completed = 0
            try:
                if os.path.exists(self._data_file):
                    with open(self._data_file, 'r') as f:
                        total_completed = sum(1 for _ in f)
            except Exception:
                pass
            return {
                'pending_tracks': total_pending,
                'completed_records': total_completed,
                'data_file': self._data_file,
            }

    # ── INTERNAL ────────────────────────────────────────────────────

    def _classify_outcome(self, rec: dict) -> str:
        """Classify outcome based on forward price action."""
        up = rec.get('max_up_pct', 0)
        down = rec.get('max_down_pct', 0)

        # Thresholds for "meaningful" move
        profit_thresh = 0.5   # 0.5% qualifies as profitable
        loss_thresh = -0.5    # -0.5% qualifies as adverse

        had_up = up >= profit_thresh
        had_down = down <= loss_thresh

        if had_up and not had_down:
            return 'PROFIT_UP'
        elif had_down and not had_up:
            return 'LOSS_DOWN'
        elif had_up and had_down:
            # Both sides hit — classify by which was bigger
            if abs(up) > abs(down):
                return 'VOLATILE_UP'
            else:
                return 'VOLATILE_DOWN'
        else:
            return 'FLAT'

    def _write_record(self, rec: dict):
        """Append a completed record to the JSONL file."""
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(self._data_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, default=str) + '\n')
        except Exception as e:
            print(f"   ⚠️ GMMCollector write error: {e}")

    def _save_pending(self):
        """Persist pending tracks to disk (crash recovery)."""
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(self._pending_file, 'w', encoding='utf-8') as f:
                json.dump(self._pending_tracks, f, default=str)
        except Exception:
            pass

    def _load_pending(self):
        """Load pending tracks from disk on restart."""
        try:
            if os.path.exists(self._pending_file):
                with open(self._pending_file, 'r', encoding='utf-8') as f:
                    self._pending_tracks = json.load(f)
                n = len(self._pending_tracks)
                if n > 0:
                    print(f"   📊 GMMCollector: Restored {n} pending tracks from disk")
        except Exception:
            self._pending_tracks = {}
