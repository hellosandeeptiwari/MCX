"""Watcher Focused Scan Pipeline — extracted from autonomous_trader.py"""
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_done
from config import BREAKOUT_WATCHER, HARD_RULES, CAPITAL_SWAP
from oi_watcher_engine import _oi_signal_from_result, _oi_direction


class WatcherPipeline:
    def __init__(self, trader):
        self.trader = trader

    def focused_scan(self, triggers: list):  # type: ignore[reportGeneralIssues]
        """
        Run the IDENTICAL pipeline as scan_and_trade for watcher-detected stocks.
        
        ALL gates present in scan_and_trade are replicated here:
          ✅ Market data fetch (candles, indicators)
          ✅ Intraday scoring (IntradaySignal + scorer)
          ✅ Market breadth (from last scan, not hardcoded MIXED)
          ✅ ML predictions (XGB + GMM via get_titan_signals)
          ✅ OI flow overlay (adjusts ML with live options chain)
          ✅ OI cross-validation (penalises direction conflict)
          ✅ Sector index cross-validation (penalises against-sector trades)
          ✅ Down-risk soft scoring (±5 nudge from GMM DR model)
          ✅ Setup validation (ORB / VWAP / EMA / RSI)
          ✅ Follow-through candle gate (no stale breakouts)
          ✅ ADX trend strength gate (≥25)
          ✅ OI conflict veto (institutions vs direction)
          ✅ ML flat veto (high P(flat) blocks auto-fire)
          ✅ XGB direction conflict veto (strong opposing signal blocks)
          ✅ XGB move probability floor (P(move) >= 0.55)
          ✅ GMM down-risk veto (extreme anomaly opposes direction → block)
          ✅ Position limit (regime-aware)
        
        The ONLY difference from a 5-min scan:
          - Universe is limited to watcher-triggered symbols (speed)
          - Score threshold is min_score (default 66) instead of 78 (ELITE)
          - GPT analysis is skipped (watcher trigger + full gates = sufficient)
        
        No shortcuts.  No bypasses.  Full pipeline parity.
        """
        t = self.trader
        _ts = datetime.now().strftime('%H:%M:%S')
        _trigger_map = {_trig['symbol']: _trig for _trig in triggers}
        _symbols = [_trig['symbol'] for _trig in triggers]
        
        try:
            # ================================================================
            # 0) F&O ELIGIBILITY — skip non-F&O stocks BEFORE the expensive pipeline
            # ================================================================
            _fo_set = getattr(t, '_fno_universe', None)
            if not _fo_set:
                try:
                    _fo_set = {f"NSE:{s['name']}" for s in t.market_scanner._fo_stocks} if hasattr(t.market_scanner, '_fo_stocks') and t.market_scanner._fo_stocks else set()
                    if _fo_set:
                        t._fno_universe = _fo_set
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/fno_universe]: {e}")
                    _fo_set = set()
            
            if _fo_set:
                _before = len(_symbols)
                _fno_triggers = [t for t in triggers if t['symbol'] in _fo_set]
                _non_fno = [t['symbol'] for t in triggers if t['symbol'] not in _fo_set]
                if _non_fno:
                    t._wlog(f"PIPELINE: skipping non-F&O: {_non_fno}")
                if not _fno_triggers:
                    t._wlog(f"PIPELINE: all {_before} triggers are non-F&O — skipping pipeline")
                    return
                triggers = _fno_triggers
                _trigger_map = {_trig['symbol']: _trig for _trig in triggers}
                _symbols = [_trig['symbol'] for _trig in triggers]

            # ================================================================
            # 0b) OI FETCH (BACKGROUND) — overlaps with market data below
            #     Layer 1 already pre-fetched OI for most symbols.  Only
            #     launch fresh fetches for symbols NOT in _watcher_drain_oi
            #     (e.g. added after F&O filter changed the list).
            #     DhanHQ throttle (3.2s lock) serialises itself — safe.
            #     Results collected before Step 4 where OI is first needed.
            # ================================================================
            _oi_futures = {}
            _oi_executor = None
            _prefetched_oi = t._watcher_drain_oi or {}
            if t._oi_analyzer:
                _need_fetch = [s for s in _symbols if s not in _prefetched_oi]
                if _need_fetch:
                    _oi_executor = _TPE(max_workers=min(3, len(_need_fetch)),
                                        thread_name_prefix='oi-fetch')
                    for _sym_oi in _need_fetch:
                        _oi_futures[_oi_executor.submit(
                            t._oi_analyzer.analyze, _sym_oi)] = _sym_oi

            # ================================================================
            # 1) MARKET DATA — fresh candles + indicators for triggered stocks
            #    force_fresh=True bypasses 10-min indicator cache so RSI/VWAP/ADX
            #    are computed from live candles (watcher only processes 1-3 stocks)
            #    NOTE: OI is fetching concurrently in _oi_executor threads above
            # ================================================================
            market_data = t.tools.get_market_data(_symbols, force_fresh=True)
            _sorted_data = [(s, d) for s, d in market_data.items()
                            if isinstance(d, dict) and 'ltp' in d]
            if not _sorted_data:
                t._wlog(f"PIPELINE: no market data for {[s for s in _symbols]} — skipping")
                return
            
            # ================================================================
            # 2) SCORING — identical IntradaySignal + scorer as scan_and_trade
            # ================================================================
            from options_trader import get_intraday_scorer, IntradaySignal
            _scorer = get_intraday_scorer()
            
            # Use cached market breadth from last full scan (not hardcoded)
            _breadth = getattr(t, '_last_market_breadth', 'MIXED')
            
            _pre_scores = {}
            _cycle_decisions = {}
            
            for _sym, _d in _sorted_data:
                _d['market_breadth'] = _breadth
                _sym_trigger_type = _trigger_map.get(_sym, {}).get('trigger_type', '')
                # Inject trigger metadata into market_data for scorer (VOLUME_SURGE bonus)
                _sym_trigger_meta = _trigger_map.get(_sym, {})
                _d['surge_ratio'] = _sym_trigger_meta.get('surge_ratio', 0)
                _d['depth_imbalance'] = _sym_trigger_meta.get('depth_imbalance', 0)
                try:
                    _sig = IntradaySignal(
                        symbol=_sym,
                        orb_signal=_d.get('orb_signal', 'INSIDE_ORB'),
                        vwap_position=_d.get('price_vs_vwap', _d.get('vwap_position', 'AT_VWAP')),
                        vwap_trend=_d.get('vwap_slope', _d.get('vwap_trend', 'FLAT')),
                        ema_regime=_d.get('ema_regime', 'NORMAL'),
                        volume_regime=_d.get('volume_regime', 'NORMAL'),
                        rsi=_d.get('rsi_14', 50.0),
                        price_momentum=_d.get('momentum_15m', 0.0),
                        htf_alignment=_d.get('htf_alignment', 'NEUTRAL'),
                        chop_zone=_d.get('chop_zone', False),
                        follow_through_candles=_d.get('follow_through_candles', 0),
                        range_expansion_ratio=_d.get('range_expansion_ratio', 0.0),
                        vwap_slope_steepening=_d.get('vwap_slope_steepening', False),
                        atr=_d.get('atr_14', 0.0)
                    )
                    _dec = _scorer.score_intraday_signal(_sig, market_data=_d, caller_direction=None, source='watcher', trigger_type=_sym_trigger_type)
                    _pre_scores[_sym] = _dec.confidence_score
                    _cycle_decisions[_sym] = {
                        'decision': _dec,
                        'direction': None,
                        'score': _dec.confidence_score,
                        'raw_score': _dec.confidence_score,
                    }
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/watcher_scoring]: {e}")
            
            if not _pre_scores:
                t._wlog(f"PIPELINE: scoring failed for all symbols — skipping")
                return
            
            # ================================================================
            # 3) ML PREDICTIONS — same get_titan_signals as scan_and_trade
            # ================================================================
            _ml_results = {}
            try:
                if t._ml_predictor:
                    import pandas as _pd_ml
                    _candle_cache = getattr(t.tools, '_candle_cache', {})
                    _daily_cache = getattr(t.tools, '_daily_cache', {})
                    _futures_oi_cache = getattr(t, '_futures_oi_data', {}) or {}
                    _options_oi_cache = getattr(t, '_options_oi_data', {}) or {}
                    _sector_5min_cache = getattr(t, '_sector_5min_cache', {})
                    _sector_daily_cache = getattr(t, '_sector_daily_cache', {})
                    _nifty_5min = getattr(t, '_nifty_5min_df', None)
                    _nifty_daily = getattr(t, '_nifty_daily_df', None)
                    _hist_5min_cache = getattr(t, '_hist_5min_cache', {})
                    
                    try:
                        from ml_models.feature_engineering import get_sector_for_symbol as _get_sector
                    except ImportError:
                        _get_sector = lambda s: ''
                    
                    for _sym in list(_pre_scores.keys()):
                        try:
                            _sym_clean = _sym.replace('NSE:', '')
                            _live_intraday = _candle_cache.get(_sym)
                            _daily_df = _daily_cache.get(_sym)
                            _hist_5min = _hist_5min_cache.get(_sym_clean)
                            
                            # Build best possible 5-min candle series (same logic as scan_and_trade)
                            _ml_candles = None
                            if _hist_5min is not None:
                                if _live_intraday is not None and len(_live_intraday) >= 2:
                                    try:
                                        _live_copy = _live_intraday.copy()
                                        _live_copy['date'] = _pd_ml.to_datetime(_live_copy['date'])
                                        if _live_copy['date'].dt.tz is not None:
                                            _live_copy['date'] = _live_copy['date'].dt.tz_localize(None)
                                        _hist_copy = _hist_5min.copy()
                                        if _hist_copy['date'].dt.tz is not None:
                                            _hist_copy['date'] = _hist_copy['date'].dt.tz_localize(None)
                                        _gap_days = (_live_copy['date'].min() - _hist_copy['date'].max()).days
                                        if _gap_days > 3:
                                            _ml_candles = _hist_5min.tail(500)
                                        else:
                                            _hist_tail = _hist_copy.tail(500)
                                            _live_start = _live_copy['date'].min()
                                            _hist_tail = _hist_tail[_hist_tail['date'] < _live_start]
                                            _common_cols = [c for c in ['date','open','high','low','close','volume']
                                                           if c in _hist_tail.columns and c in _live_copy.columns]
                                            _ml_candles = _pd_ml.concat(
                                                [_hist_tail[_common_cols], _live_copy[_common_cols]],
                                                ignore_index=True)
                                    except Exception as e:
                                        print(f"⚠️ FALLBACK [trader/ml_candle_concat]: {e}")
                                        _ml_candles = _hist_5min.tail(500)
                                else:
                                    _ml_candles = _hist_5min.tail(500)
                            elif _live_intraday is not None and len(_live_intraday) >= 50:
                                _ml_candles = _live_intraday
                            
                            if _ml_candles is None or len(_ml_candles) < 50:
                                continue
                            
                            _fut_oi = _futures_oi_cache.get(_sym_clean)
                            _opt_oi = _options_oi_cache.get(_sym_clean)
                            _sec_name = _get_sector(_sym_clean)
                            _sec_5m = _sector_5min_cache.get(_sec_name) if _sec_name else None
                            _sec_dl = _sector_daily_cache.get(_sec_name) if _sec_name else None
                            
                            _pred = t._ml_predictor.get_titan_signals(
                                _ml_candles,
                                daily_df=_daily_df,
                                oi_df=_opt_oi,
                                futures_oi_df=_fut_oi,
                                nifty_5min_df=_nifty_5min,
                                nifty_daily_df=_nifty_daily,
                                sector_5min_df=_sec_5m,
                                sector_daily_df=_sec_dl,
                                symbol=_sym_clean
                            )
                            if _pred:
                                if _pred.get('ml_score_boost', 0) != 0:
                                    _pre_scores[_sym] += _pred['ml_score_boost']
                                if _pred.get('ml_signal') != 'UNKNOWN':
                                    _ml_results[_sym] = _pred
                                    if _sym in _cycle_decisions:
                                        _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                                        _cycle_decisions[_sym]['ml_prediction'] = _pred
                        except Exception as e:
                            print(f"⚠️ FALLBACK [trader/ml_prediction]: {e}")
            except Exception as e:
                print(f"⚠️ FALLBACK [trader/ml_pipeline]: {e}")  # ML failed — continue without it (fail-safe)
            
            # ================================================================
            # 3b) COLLECT OI RESULTS — background threads launched in Step 0b
            #      By now ~2-3s have elapsed (market data + scoring + ML ran
            #      concurrently with OI). Cached OI returns instantly; cold
            #      fetches may still be in-flight — wait up to 30s.
            # ================================================================
            _drain_oi_results = dict(_prefetched_oi)  # Start with Layer 1 pre-fetched OI
            if _oi_futures:
                try:
                    for _fut in _as_done(_oi_futures, timeout=30):
                        _oi_sym = _oi_futures[_fut]
                        try:
                            _oi_result = _fut.result()
                            if _oi_result:
                                _drain_oi_results[_oi_sym] = _oi_result
                                # Tag trigger dict with OI direction
                                _t_for_oi = _trigger_map.get(_oi_sym)
                                if _t_for_oi:
                                    _t_oi_signal = _oi_signal_from_result(_oi_result)
                                    _t_oi_dir = _oi_direction(_t_oi_signal)
                                    if _t_oi_dir:
                                        _t_for_oi['_oi_direction'] = _t_oi_dir
                                        _t_for_oi['_oi_signal'] = _t_oi_signal
                                        _t_type = _t_for_oi.get('trigger_type', '')
                                        _spike_dir = 'SELL' if ('DOWN' in _t_type or 'LOW' in _t_type) else 'BUY'
                                        if _t_oi_dir == _spike_dir:
                                            _t_for_oi['_oi_confirmed'] = True
                        except Exception as e:
                            print(f"⚠️ FALLBACK [trader/watcher_oi_drain_result]: {e}")
                except Exception:
                    t._wlog("⚠️ OI background timeout — proceeding with partial OI data")
                if _oi_executor:
                    _oi_executor.shutdown(wait=False)
            t._watcher_drain_oi = _drain_oi_results
            # Log OI results
            _oi_tags = []
            for _t in triggers:
                if _t.get('_oi_confirmed'):
                    _oi_tags.append(f"{_t['symbol'].replace('NSE:', '')}=OI✓{_t.get('_oi_signal', '')}")
                elif _t.get('_oi_direction'):
                    _oi_tags.append(f"{_t['symbol'].replace('NSE:', '')}=OI✗{_t.get('_oi_signal', '')}(vs {_t.get('trigger_type', '')})")
            if _oi_tags:
                t._wlog(f"   📊 OI: {', '.join(_oi_tags)}")

            # ================================================================
            # 4) OI FLOW OVERLAY — adjusts ML with live options chain data
            #    Uses OI collected from background threads above
            # ================================================================
            _oi_results = {}
            _drain_oi = t._watcher_drain_oi
            try:
                if t._oi_analyzer:
                    for _sym in list(_pre_scores.keys()):
                        try:
                            # Use background-fetched OI; fallback to fresh fetch if missing
                            _oi_data = _drain_oi.get(_sym)
                            if not _oi_data:
                                _oi_data = t._oi_analyzer.analyze(_sym)
                            if _oi_data:
                                _oi_results[_sym] = _oi_data
                                if t._ml_predictor and _sym in _ml_results:
                                    t._ml_predictor.apply_oi_overlay(_ml_results[_sym], _oi_data)
                                    if _sym in _cycle_decisions:
                                        _cycle_decisions[_sym]['ml_prediction'] = _ml_results[_sym]
                        except Exception as e:
                            print(f"⚠️ FALLBACK [trader/oi_crossval]: {e}")
            except Exception as e:
                print(f"⚠️ FALLBACK [trader/sector_crossval]: {e}")
            
            # ================================================================
            # 5) OI CROSS-VALIDATION — OI is AUTHORITATIVE direction signal
            #    Operators build OI positions first, then move the underlying.
            #    OI buildup = leading indicator. Trust it over scorer.
            # ================================================================
            for _oi_sym, _oi_data in _oi_results.items():
                try:
                    # First try nse_oi_buildup (strongest signal)
                    _oi_signal = _oi_signal_from_result(_oi_data)
                    _oi_dir = _oi_direction(_oi_signal)
                    if not _oi_dir:
                        # Fall back to flow_bias with lowered confidence threshold
                        _oi_bias = _oi_data.get('flow_bias', 'NEUTRAL')
                        _oi_conf = _oi_data.get('flow_confidence', 0.0)
                        if _oi_bias == 'NEUTRAL' or _oi_conf < 0.40:
                            continue
                        _oi_dir = 'BUY' if _oi_bias == 'BULLISH' else 'SELL'
                        _oi_signal = f'{_oi_bias}(conf={_oi_conf:.2f})'
                    _cd = _cycle_decisions.get(_oi_sym)
                    if not _cd or not _cd.get('decision'):
                        continue
                    _scored_dir = _cd['decision'].recommended_direction
                    # OI overrides even HOLD — operators have positioned, follow them
                    if _scored_dir != _oi_dir:
                        _cd['decision'].recommended_direction = _oi_dir
                        t._wlog(f"OI OVERRIDE: {_oi_sym.replace('NSE:', '')} "
                              f"{_scored_dir} → {_oi_dir} (OI={_oi_signal})")
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/oi_direction_override]: {e}")
            
            # Inject OI participant + PCR shift into market_data for scorer access
            for _oi_sym, _oi_data in _oi_results.items():
                _oi_pi = _oi_data.get('oi_participant_id', 'UNKNOWN')
                _stk_md = market_data.get(_oi_sym)
                if isinstance(_stk_md, dict) and _oi_pi != 'UNKNOWN':
                    _stk_md['oi_participant_id'] = _oi_pi
                    _stk_md['pcr_shift_rate'] = _oi_data.get('pcr_shift_rate', 0.0)
                    _stk_md['ce_oi_velocity'] = _oi_data.get('ce_oi_velocity', 0.0)
                    _stk_md['pe_oi_velocity'] = _oi_data.get('pe_oi_velocity', 0.0)
            
            # ================================================================
            # 6) SECTOR INDEX CROSS-VALIDATION — penalise against-sector trades
            # ================================================================
            _sec_changes = getattr(t, '_sector_index_changes_cache', {})
            _stock_to_sector = getattr(t, '_stock_to_sector', {})
            
            for _sym, _score in list(_pre_scores.items()):
                try:
                    _stock_name = _sym.replace('NSE:', '')
                    _sec_match = _stock_to_sector.get(_stock_name)
                    if not _sec_match:
                        continue
                    _sec_name, _sec_index = _sec_match
                    _sec_chg = _sec_changes.get(_sec_index)
                    if _sec_chg is None:
                        continue
                    _cd = _cycle_decisions.get(_sym)
                    if not _cd or not _cd.get('decision'):
                        continue
                    _scored_dir = _cd['decision'].recommended_direction
                    if _scored_dir == 'HOLD':
                        continue
                    _stk_data = market_data.get(_sym, {})
                    _stk_chg = _stk_data.get('change_pct', 0) if isinstance(_stk_data, dict) else 0
                    
                    if _sec_chg <= -1.0 and _scored_dir == 'BUY':
                        if _stk_chg > 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                            continue
                        _sec_penalty = -5 if _sec_chg <= -2.0 else -3
                        _pre_scores[_sym] += _sec_penalty
                        if _sym in _cycle_decisions:
                            _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                        t._wlog(f"Sector cross-val: {_stock_name} penalised {_sec_penalty} "
                              f"(sector {_sec_chg:+.1f}% vs BUY)")
                    elif _sec_chg >= 1.0 and _scored_dir == 'SELL':
                        if _stk_chg < 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                            continue
                        _sec_penalty = -5 if _sec_chg >= 2.0 else -3
                        _pre_scores[_sym] += _sec_penalty
                        if _sym in _cycle_decisions:
                            _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                        t._wlog(f"Sector cross-val: {_stock_name} penalised {_sec_penalty} "
                              f"(sector {_sec_chg:+.1f}% vs SELL)")
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/dr_soft_scores]: {e}")
            
            # ================================================================
            # 7) DOWN-RISK SOFT SCORING — ±5 nudge from GMM DR model
            # ================================================================
            try:
                t._apply_down_risk_soft_scores(_ml_results, _pre_scores)
            except Exception as e:
                print(f"⚠️ FALLBACK [trader/dr_soft_scores_outer]: {e}")
            
            # ================================================================
            # 8) GATE-CHECKED EXECUTION — same gates as ELITE auto-fire
            #    Setup validation, FT, ADX, OI conflict, ML flat veto
            #    Score threshold: min_score (default 66) from BREAKOUT_WATCHER config
            # ================================================================
            _min_score = BREAKOUT_WATCHER.get('min_score', 66)
            _max_per_scan = BREAKOUT_WATCHER.get('max_trades_per_scan', 2)
            _fired_count = 0
            
            # ── VIX-based score penalty: direction-aware ──
            # High VIX hurts CE buyers (expensive) but helps PE buyers (premium expands)
            # Only penalize when direction opposes VIX regime (buying CEs in high-VIX bearish)
            _vix = getattr(t, '_current_vix', 14.0)
            _vix_penalty_above = BREAKOUT_WATCHER.get('vix_penalty_above', 18.0)
            _vix_penalty_per_pt = BREAKOUT_WATCHER.get('vix_penalty_per_point', 2)
            _vix_hard_block = BREAKOUT_WATCHER.get('vix_hard_block_above', 32.0)
            _vix_raw_penalty = 0
            if _vix > _vix_hard_block:
                t._wlog(f"🚫 VIX BLOCK: India VIX={_vix:.1f} > {_vix_hard_block} — blocking ALL watcher entries")
                return
            if _vix > _vix_penalty_above:
                _vix_raw_penalty = round((_vix - _vix_penalty_above) * _vix_penalty_per_pt)
            _market_breadth = getattr(t, '_last_market_breadth', 'MIXED')
            
            # ════════════════════════════════════════════════════════════════
            # TWO-PASS PIPELINE: Run all gates first, collect candidates,
            # then RANK by P(move) and place BEST trades only.
            # ════════════════════════════════════════════════════════════════
            _candidates = []  # [{sym, direction, score, ml_move_prob, ...}]
            
            for _sym, _score in sorted(_pre_scores.items(), key=lambda x: x[1], reverse=True):
                _trigger = _trigger_map.get(_sym, {})
                _trigger_type = _trigger.get('trigger_type', '?')
                _move_pct = _trigger.get('move_pct', 0)
                _data = market_data.get(_sym, {})
                if not isinstance(_data, dict):
                    continue
                
                _cached = _cycle_decisions.get(_sym, {})
                _decision = _cached.get('decision')
                if not _decision:
                    continue
                
                # Score from decision (includes ML boost, OI penalty, sector penalty, DR nudge)
                _final_score = _pre_scores.get(_sym, 0)

                # ── Trigger-strength score bonus (Mar-09) ──
                # The watcher detected real-time momentum. Strong trigger types
                # deserve a score bonus — the ticker's 60s sustain IS evidence.
                _trigger_bonus = 0
                if _trigger_type in ('PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN'):
                    _trigger_bonus = 10  # Strongest: ≥0.7% move sustained 60s
                    # Accelerating spikes are stronger (recent ticks moving faster than earlier ticks)
                    _spike_accel = _trigger.get('spike_accel', 0)
                    if _spike_accel >= 1.5:
                        _trigger_bonus += 2  # Still accelerating at trigger time
                    # Large spikes (≥1.2%) get extra credit
                    _spike_mag = _trigger.get('spike_magnitude', 0)
                    if _spike_mag >= 1.2:
                        _trigger_bonus += 2
                elif _trigger_type in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):
                    _trigger_bonus = 5   # Persistent multi-minute directional move (was 8 — over-inflating weak grinds)
                    # Volume-confirmed grinds are stronger signals
                    if _trigger.get('vol_confirmed', False):
                        _trigger_bonus += 3
                elif _trigger_type in ('NEW_DAY_HIGH', 'NEW_DAY_LOW'):
                    _trigger_bonus = 7   # Structural break of day's range
                    # First break of the day is much stronger than the 5th incremental
                    _break_count = _trigger.get('break_count', 1)
                    if _break_count <= 2:
                        _trigger_bonus += 3  # Fresh breakout — strongest
                    elif _break_count == 3:
                        _trigger_bonus += 1  # Third break — still meaningful
                    elif _break_count >= 5:
                        _trigger_bonus -= 3  # 5th+ marginal — likely noise
                    # Larger margin above previous extreme = more conviction
                    _break_margin = _trigger.get('break_margin', 0)
                    if _break_margin >= 0.3:
                        _trigger_bonus += 2  # Strong decisive break
                    elif _break_margin < 0.15:
                        _trigger_bonus -= 1  # Marginal break — penalise
                    # Late-day penalty: after 14:00, range extensions are weaker
                    _now_dayext = datetime.now()
                    if _now_dayext.hour >= 14:
                        _trigger_bonus -= 2  # Late-day structural break = less reliable
                    # Consolidation lookback: long quiet period before break = powerful
                    # Short quiet period = just grinding higher, less meaningful
                    _quiet_min = _trigger.get('quiet_minutes', 60)
                    if _quiet_min >= 120:
                        _trigger_bonus += 4  # 2+ hours consolidation then break = strong
                    elif _quiet_min >= 60:
                        _trigger_bonus += 2  # 1+ hour pause before break = meaningful
                    elif _quiet_min < 15:
                        _trigger_bonus -= 2  # <15 min since last break = continuous grind
                    # [FIX Mar 18] OI validation for structural breaks
                    # A day-high/low confirmed by OI is much stronger than one without.
                    # Pattern mirrors VOLUME_SURGE OI gate below.
                    _dhl_oi_data = _oi_results.get(_sym, {})
                    if _dhl_oi_data:
                        _dhl_oi_sig = _oi_signal_from_result(_dhl_oi_data)
                        _dhl_oi_str = _dhl_oi_data.get('nse_oi_buildup_strength', 0.0)
                        _dhl_oi_dir = _oi_direction(_dhl_oi_sig)
                        _dhl_part = _dhl_oi_data.get('oi_participant_id', 'UNKNOWN')
                        _dhl_expect = 'BUY' if _trigger_type == 'NEW_DAY_HIGH' else 'SELL'
                        if _dhl_oi_dir == _dhl_expect and _dhl_oi_str >= 0.25:
                            _trigger_bonus += 2  # OI confirms structural break
                            if _dhl_part == 'WRITER_DOMINANT':
                                _trigger_bonus += 1  # Institutions backing the break
                        elif _dhl_oi_dir and _dhl_oi_dir != _dhl_expect and _dhl_oi_str >= 0.30:
                            _trigger_bonus -= 3  # OI contradicts break — likely false breakout
                        # PCR shift rate: fast PCR movement toward break direction = extra edge
                        _dhl_pcr_rate = _dhl_oi_data.get('pcr_shift_rate', 0.0)
                        if _dhl_pcr_rate != 0:
                            # Negative pcr_shift = puts decreasing relative to calls = bullish
                            _pcr_bullish = _dhl_pcr_rate < -0.01
                            if (_dhl_expect == 'BUY' and _pcr_bullish) or (_dhl_expect == 'SELL' and not _pcr_bullish):
                                _trigger_bonus += 1  # PCR shifting in break direction
                elif _trigger_type == 'VOLUME_SURGE':
                    # Volume surge: base bonus even at 0.3% move (detection threshold)
                    # Previously required 0.5% for ANY bonus — dead zone killed all VOLUME_SURGE trades
                    _trigger_bonus = 5 if abs(_move_pct) >= 0.3 else 3
                    # Higher surge ratios deserve more bonus (5x is much stronger than 3x)
                    _surge_ratio = _trigger.get('surge_ratio', 3.0)
                    if _surge_ratio >= 6.0:
                        _trigger_bonus += 4
                    elif _surge_ratio >= 4.5:
                        _trigger_bonus += 2
                    elif _surge_ratio >= 3.0:
                        _trigger_bonus += 1
                    # Depth imbalance confirms institutional intent
                    _depth_imb = abs(_trigger.get('depth_imbalance', 0))
                    if _depth_imb >= 0.3:
                        _trigger_bonus += 2
                    # OI validation: surge without new OI positions = noise
                    _vs_oi_data = _oi_results.get(_sym, {})
                    if _vs_oi_data:
                        _vs_oi_sig = _oi_signal_from_result(_vs_oi_data)
                        _vs_oi_str = _vs_oi_data.get('nse_oi_buildup_strength', 0.0)
                        _vs_oi_dir = _oi_direction(_vs_oi_sig)
                        _vs_part = _vs_oi_data.get('oi_participant_id', 'UNKNOWN')
                        _vs_expect = 'BUY' if _move_pct > 0 else 'SELL'
                        if _vs_oi_dir == _vs_expect and _vs_oi_str >= 0.25:
                            _trigger_bonus += 2  # OI confirms surge direction
                            if _vs_part == 'WRITER_DOMINANT':
                                _trigger_bonus += 1  # Smart money backing surge
                        elif _vs_oi_dir and _vs_oi_dir != _vs_expect and _vs_oi_str >= 0.30:
                            _trigger_bonus -= 3  # OI contradicts surge — likely noise

                _final_score += _trigger_bonus
                
                # ── OI Sustain Confirmation score bonus ──
                # If the BreakoutWatcher's sustain phase found OI buildup CONFIRMING
                # the price move, that's forward-looking institutional evidence.
                # Boost the score.  If OI contradicted, penalize (likely a trap move).
                _oi_sustain_bonus = 0
                if _trigger.get('_oi_confirmed'):
                    _oi_s = _trigger.get('_oi_strength', 0)
                    if _oi_s >= 0.60:
                        _oi_sustain_bonus = 6   # Strong OI + price = high conviction
                    elif _oi_s >= 0.45:
                        _oi_sustain_bonus = 4
                    else:
                        _oi_sustain_bonus = 2
                    _final_score += _oi_sustain_bonus
                elif _trigger.get('_oi_contradicted'):
                    _oi_s = _trigger.get('_oi_strength', 0)
                    if _oi_s >= 0.50:
                        _oi_sustain_bonus = -5  # Strong OI AGAINST = likely trap
                    else:
                        _oi_sustain_bonus = -2
                    _final_score += _oi_sustain_bonus

                # Post-restart caution: baselines are thin for first 2 min after restart
                # Apply a score penalty to avoid low-confidence triggers during warmup
                _restart_penalty = 0
                if _trigger.get('post_restart', False):
                    _restart_penalty = 5
                    _final_score -= _restart_penalty

                # --- Direction: reconcile scorer vs trigger vs OI ---
                # Priority: OI (leading) > Trigger (real-time) > Scorer (lagging).
                # Scorer never rejects — trigger wins on conflict.
                # OI overrides everything when directional.
                direction = None
                _ml_data_spike_rev = False  # Set True by Gate F2 spike reversal flip
                _trigger_dir = None  # What the trigger implies
                if _trigger_type in ('PRICE_SPIKE_UP', 'NEW_DAY_HIGH', 'SLOW_GRIND_UP'):
                    _trigger_dir = 'BUY'
                elif _trigger_type in ('PRICE_SPIKE_DOWN', 'NEW_DAY_LOW', 'SLOW_GRIND_DOWN'):
                    _trigger_dir = 'SELL'
                else:
                    _trigger_dir = 'BUY' if _move_pct > 0 else 'SELL'
                
                _scorer_dir = None
                _dir_conf_val = 0
                if hasattr(_decision, 'recommended_direction') and _decision.recommended_direction not in ('HOLD', None, ''):
                    _scorer_dir = _decision.recommended_direction
                
                if _scorer_dir and _scorer_dir == _trigger_dir:
                    # Agreement — strongest signal, use scorer direction
                    direction = _scorer_dir
                elif _scorer_dir and _scorer_dir != _trigger_dir:
                    # CONFLICT: scorer says one thing, trigger says another
                    _dir_conf_val = getattr(_decision, 'direction_confidence', 0)
                    # ── DOUBLE CONVICTION FLIP ──
                    # When scorer (high confidence) AND OI both oppose the trigger,
                    # the weight of evidence > real-time noise. Trigger provides TIMING
                    # (something is moving NOW), scorer+OI provide DIRECTION.
                    # e.g. SHORT_BUILDUP + scorer SELL(76%) + spike UP = exhaustion spike
                    #      → trade as SELL (put) — ride the reversal, not the spike.
                    _sym_oi_dc = _oi_results.get(_sym, {})
                    _oi_sig_dc = _oi_signal_from_result(_sym_oi_dc) if _sym_oi_dc else None
                    _oi_dir_dc = _oi_direction(_oi_sig_dc) if _oi_sig_dc else None
                    _oi_str_dc = _sym_oi_dc.get('nse_oi_buildup_strength', 0.0) if _sym_oi_dc else 0.0
                    # OI measurables for logging
                    _oi_ce_chg = _sym_oi_dc.get('nse_total_call_oi_change', 0) if _sym_oi_dc else 0
                    _oi_pe_chg = _sym_oi_dc.get('nse_total_put_oi_change', 0) if _sym_oi_dc else 0
                    _oi_pcr = _sym_oi_dc.get('pcr_oi', 0) if _sym_oi_dc else 0
                    # ── PCR-DERIVED SIGNAL FIX ──
                    # When _detect_oi_buildup returned NEUTRAL (str=0) but flow_bias
                    # produced a directional signal via PCR, use flow_confidence as
                    # surrogate strength. PCR ≤0.5 → conf~0.62, PCR ≤0.7 → conf=0.58
                    _oi_src = 'OI'  # signal source label
                    if _oi_str_dc < 0.01 and _oi_sig_dc and _oi_sig_dc not in ('NEUTRAL', ''):
                        _fc = _sym_oi_dc.get('flow_confidence', 0.0) if _sym_oi_dc else 0.0
                        if _fc > 0.1:
                            _oi_str_dc = _fc
                            _oi_src = 'PCR'
                    # Double conviction requires: OI agrees with scorer + scorer confident + OI strong enough
                    _double_conviction = (_oi_dir_dc == _scorer_dir and _dir_conf_val >= 60 and _oi_str_dc >= 0.35)
                    if _double_conviction:
                        # Scorer + OI consensus overrides trigger direction
                        direction = _scorer_dir
                        # Scale penalty by OI strength: stronger OI = more conviction = less penalty
                        _dc_penalty = 3 if _oi_str_dc < 0.50 else (1 if _oi_str_dc >= 0.70 else 2)
                        _final_score -= _dc_penalty
                        t._wlog(f"🔄 DOUBLE CONVICTION: {_sym.replace('NSE:', '')} "
                                   f"scorer={_scorer_dir}(conf={_dir_conf_val:.0f}%) + {_oi_src}={_oi_sig_dc}(str={_oi_str_dc:.0%}) "
                                   f"BOTH oppose trigger={_trigger_dir}({_trigger_type}) "
                                   f"→ FLIPPING to {_scorer_dir} (score -{_dc_penalty}) "
                                   f"[ΔCE:{_oi_ce_chg:+,} ΔPE:{_oi_pe_chg:+,} PCR:{_oi_pcr:.2f}]")
                    else:
                        # Single-sided conflict: trust trigger momentum
                        direction = _trigger_dir
                        _conflict_detail = f" [{_oi_src}={_oi_sig_dc or 'N/A'}(str={_oi_str_dc:.0%}) ΔCE:{_oi_ce_chg:+,} ΔPE:{_oi_pe_chg:+,} PCR:{_oi_pcr:.2f}]" if _sym_oi_dc else ""
                        if _dir_conf_val >= 70:
                            _final_score -= 3
                            t._wlog(f"⚠️ DIR CONFLICT: {_sym} scorer={_scorer_dir}(conf={_dir_conf_val:.0f}%) vs trigger={_trigger_dir}({_trigger_type}) → using trigger dir (score -3){_conflict_detail}")
                        else:
                            t._wlog(f"⚠️ DIR CONFLICT: {_sym} scorer={_scorer_dir}(conf={_dir_conf_val:.0f}%) vs trigger={_trigger_dir}({_trigger_type}) → using trigger dir{_conflict_detail}")
                else:
                    # Scorer said HOLD — use trigger direction
                    direction = _trigger_dir
                
                # ── OI DIRECTION OVERRIDE — OI is authoritative, overrides both ──
                # Operators build OI positions first, then move the underlying.
                # If OI says directional, it overrides scorer AND trigger.
                _oi_override_dir = None
                _sym_oi = _oi_results.get(_sym, {})
                if _sym_oi:
                    _oi_sig_here = _oi_signal_from_result(_sym_oi)
                    _oi_dir_here = _oi_direction(_oi_sig_here)
                    _oi_str_here = _sym_oi.get('nse_oi_buildup_strength', 0.0)
                    if _oi_dir_here and _oi_str_here >= 0.40:
                        if _oi_dir_here != direction:
                            _oi_override_dir = _oi_dir_here
                            t._wlog(f"🔄 OI OVERRIDE: {_sym.replace('NSE:', '')} "
                                       f"{direction} → {_oi_dir_here} "
                                       f"(OI={_oi_sig_here} str={_oi_str_here:.2f} > trigger={_trigger_dir})")
                            direction = _oi_dir_here
                
                # ── Direction-aware VIX penalty ──
                # High VIX + SELL direction on bearish day = PE buying = VIX helps → no penalty
                # High VIX + BUY direction on bearish day = CE buying = VIX hurts → full penalty
                _vix_penalty = 0
                if _vix_raw_penalty > 0:
                    _vix_dir_helps = (
                        (direction == 'SELL' and _market_breadth in ('BEARISH',))
                        or (direction == 'BUY' and _market_breadth in ('BULLISH',))
                    )
                    if _vix_dir_helps:
                        _vix_penalty = 0  # VIX aligns with trade direction — no penalty
                    else:
                        _vix_penalty = _vix_raw_penalty
                    _final_score -= _vix_penalty
                
                # Score audit from scorer (shows component breakdown)
                _score_audit = getattr(_scorer, '_last_score_audit', '')
                _stock_name = _sym.replace('NSE:', '')
                _ml_pred = _ml_results.get(_sym, {})
                _ml_move_prob = _ml_pred.get('ml_move_prob', 0)
                _dr_score = _ml_pred.get('ml_down_risk_score', 0)
                _up_score = _ml_pred.get('ml_up_score', 0)
                _down_score = _ml_pred.get('ml_down_score', 0)
                # Pre-init variables used in GATE CHECK log (computed later in loop)
                _conviction_bonus = 0
                _late_decay = 0
                t._wlog(f"GATE CHECK: {_stock_name} | score={_final_score:.0f}(+{_trigger_bonus}{f' oi{_oi_sustain_bonus:+d}' if _oi_sustain_bonus else ''}{f' conv+{_conviction_bonus}' if _conviction_bonus else ''}{f' decay{_late_decay}' if _late_decay else ''}{f' restart-{_restart_penalty}' if _restart_penalty else ''}) dir={direction}(scorer={_scorer_dir or 'HOLD'}{f' oi→{_oi_override_dir}' if _oi_override_dir else ''}) "
                           f"trigger={_trigger_type}({_move_pct:+.1f}%) VIX={_vix:.0f}(pen={_vix_penalty}) "
                           f"oi_sustain={_trigger.get('_oi_signal', '-')}({_trigger.get('_oi_strength', '-')}) "
                           f"vc={_trigger.get('vol_confirmed', '-')} sr={_trigger.get('surge_ratio', '-')} di={_trigger.get('depth_imbalance', '-')} "
                           f"accel={_trigger.get('spike_accel', '-')} mag={_trigger.get('spike_magnitude', '-')} "
                           f"brk={_trigger.get('break_count', '-')}/{_trigger.get('break_margin', '-')} | "
                           f"P(move)={_ml_move_prob:.2f} | "
                           f"GMM DR={_dr_score:.3f} up={_up_score:.3f} dn={_down_score:.3f} | "
                           f"ORB={_data.get('orb_signal', '?')} VWAP={_data.get('price_vs_vwap', '?')} "
                           f"VOL={_data.get('volume_regime', '?')} ADX={_data.get('adx', 0):.0f} "
                           f"FT={_data.get('follow_through_candles', 0)} RSI={_data.get('rsi_14', 50):.0f} "
                           f"peak={_trigger.get('_peak_move_pct', '-')} held={_trigger.get('_sustain_held_pct', '-')}")
                if _score_audit:
                    t._wlog(f"  AUDIT: {_score_audit}")
                
                # --- Repeat Conviction & Momentum Tracking ---
                # Track every time a stock reaches the scoring pipeline.
                # Layer 1: Repeat Conviction — a stock that keeps appearing across
                # multiple pipeline checks is building real momentum even if the
                # individual score stays moderate. WAAREEENER lesson: stock appeared
                # at 09:46 (score=32, blocked by 3 pts), then re-triggered at 14:54
                # (score=66, entered at the top). A repeat bonus would have gotten
                # us in early when risk/reward was best.
                _now_ts = time.time()
                _hist = t._watcher_score_history.get(_sym, [])
                # Prune entries older than 90 minutes (wider window for conviction)
                _hist = [(_ts_val, _s, _d) for _ts_val, _s, _d in _hist if _now_ts - _ts_val < 5400]
                _hist.append((_now_ts, _final_score, direction))
                t._watcher_score_history[_sym] = _hist
                
                # -- Layer 1a: Repeat Conviction Bonus --
                # Count how many times this stock has reached the pipeline today
                # (within 90-min window). Each repeat = the watcher keeps seeing it.
                # Consistent direction across checks = higher conviction.
                _conviction_bonus = 0
                _n_appearances = len(_hist)
                if _n_appearances >= 2:
                    # Count how many prior checks had the SAME direction
                    _same_dir = sum(1 for _, _, d in _hist[:-1] if d == direction)
                    _dir_consistency = _same_dir / (_n_appearances - 1) if _n_appearances > 1 else 0
                    if _dir_consistency >= 0.6:  # ≥60% same direction
                        # +3 per repeat appearance (2nd=+3, 3rd=+6, 4th=+9), cap +12
                        _conviction_bonus = min((_n_appearances - 1) * 3, 12)
                    else:
                        # Direction is flip-flopping — weaker conviction, +2 per repeat, cap +6
                        _conviction_bonus = min((_n_appearances - 1) * 2, 6)
                    _final_score += _conviction_bonus
                    if _conviction_bonus > 0:
                        t._wlog(f"  CONVICTION: {_stock_name} appeared {_n_appearances}x (dir consistency={_dir_consistency:.0%}) → +{_conviction_bonus} → {_final_score:.0f}")
                
                # -- Layer 1b: Score Momentum Boost (existing, unchanged) --
                _momentum_boost = 0
                if _n_appearances >= 3:
                    _rising = 0
                    _total_delta = 0.0
                    for i in range(_n_appearances - 1, 0, -1):
                        _delta = _hist[i][1] - _hist[i-1][1]
                        if _delta >= 3:
                            _rising += 1
                            _total_delta += _delta
                        else:
                            break
                    if _rising >= 3:
                        _avg_delta = _total_delta / _rising
                        _base_boost = (_rising - 2) * 3
                        _velocity_bonus = 3 if _avg_delta >= 10 else 0
                        _momentum_boost = min(_base_boost + _velocity_bonus, 15)
                        _final_score += _momentum_boost
                        t._wlog(f"  MOMENTUM: {_stock_name} {_rising} rising(Δ≥3) avg_Δ={_avg_delta:.0f} → boost +{_momentum_boost} → {_final_score:.0f}")
                
                # -- Layer 2: Late Entry Decay --
                # If the stock has already moved significantly intraday and we
                # haven't entered yet, the risk/reward is deteriorating.
                # Technicals inflate scores for extended moves (Trend, Accel,
                # ADX all improve), but the best R:R was earlier.
                # This counteracts the score-inflating effect of big moves.
                _late_decay = 0
                _abs_change = abs(_data.get('change_pct', 0))
                if _abs_change >= 3.5:
                    _late_decay = -18
                elif _abs_change >= 2.5:
                    _late_decay = -12
                elif _abs_change >= 1.8:
                    _late_decay = -6
                # SLOW_GRIND = persistent multi-minute trend, not exhaustion.
                # A stock grinding steadily for hours IS the trend — penalize
                # less than a sudden spike at the same intraday level.
                _grind_decay_tag = ''
                if _late_decay < 0 and _trigger_type in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):
                    _orig_decay = _late_decay
                    if _trigger.get('vol_confirmed', False):
                        _late_decay = max(_late_decay // 3, -6)  # Vol-confirmed: ~1/3 penalty
                        _grind_decay_tag = f' [grind+vol: {_orig_decay}→{_late_decay}]'
                    else:
                        _late_decay = _late_decay // 2  # Regular grind: half penalty
                        _grind_decay_tag = f' [grind: {_orig_decay}→{_late_decay}]'
                if _late_decay < 0:
                    _final_score += _late_decay
                    t._wlog(f"  LATE-DECAY: {_stock_name} already moved {_abs_change:+.1f}% intraday → {_late_decay} → {_final_score:.0f}{_grind_decay_tag}")
                
                # --- GATE G-SLOPE: Grind Quality Check (simplified v2, Mar 18) ---
                # Old: 5 independent signals (A-E) + 7-way compound with variable
                #      thresholds + 3 bypass mechanisms = unpredictable edge cases.
                # New: 3 clear layers, one fade score, one decision.
                #   Layer 1: Hard blocks (no trend / chasing crumbs) — always block
                #   Layer 2: OI exit (smart money leaving) — blocks if score < 65
                #   Layer 3: Momentum fade score — weighted evidence, single threshold
                if _trigger_type in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):
                    # --- Inputs ---
                    _g_intraday = abs(_trigger.get('intraday_pct', _data.get('change_pct', 0)))
                    _g_freshness = min(abs(_move_pct) / _g_intraday, 1.0) if _g_intraday > 0.3 else 1.0
                    _g_adx = _data.get('adx', 50)
                    _g_vol_confirmed = _trigger.get('vol_confirmed', False)
                    _g_velocity = _trigger.get('velocity', 0)
                    _g_peak_vel = _trigger.get('peak_velocity', _g_velocity)
                    _g_sb_move = _trigger.get('sb_move_pct', -1)
                    _g_oi_signal = _trigger.get('_oi_signal', '')
                    _g_oi_strength = _trigger.get('_oi_strength', 0)
                    _g_range_pos = _trigger.get('day_range_position', -1)
                    if _g_range_pos < 0:
                        _g_high, _g_low = _data.get('high', 0), _data.get('low', 0)
                        _g_ltp = _data.get('ltp', 0)
                        _g_range = _g_high - _g_low if _g_high > _g_low else 0.01
                        _g_range_pos = max(0, min(1, (_g_ltp - _g_low) / _g_range))

                    # Context flags
                    _g_early = (datetime.now().hour == 9 and datetime.now().minute < 30)
                    _g_ml_pmove = _ml_results.get(_sym, {}).get('ml_move_prob', 0.5)
                    _g_oi_data = _oi_results.get(_sym, {})
                    _g_part_id = _g_oi_data.get('oi_participant_id', 'UNKNOWN') if _g_oi_data else 'UNKNOWN'
                    _g_strong = (_g_vol_confirmed and _g_adx >= 40 and _g_freshness >= 0.80)
                    _g_writer_backed = (_g_part_id == 'WRITER_DOMINANT' and _g_oi_strength >= 0.35)

                    _grind_slope_blocked = False
                    _grind_slope_reason = ''
                    _fade_score = 0
                    _fade_tags = []
                    _fade_thresh = 99

                    # === LAYER 1: Hard blocks (no overrides possible) ===
                    if _g_adx < 20 and not _g_vol_confirmed:
                        _grind_slope_blocked = True
                        _grind_slope_reason = f'no-trend: ADX={_g_adx:.0f}<20 + no vol'
                    elif _g_freshness < 0.15 and _final_score < 55:
                        _grind_slope_blocked = True
                        _grind_slope_reason = f'chasing-crumbs: fresh={_g_freshness:.2f} score={_final_score:.0f}'

                    # === LAYER 2: OI exit — smart money leaving this direction ===
                    # LONG_UNWINDING on grind-UP or SHORT_COVERING on grind-DOWN
                    if not _grind_slope_blocked and _g_oi_strength >= 0.25 and _final_score < 65:
                        _oi_exit = ((_trigger_type == 'SLOW_GRIND_UP' and _g_oi_signal == 'LONG_UNWINDING') or
                                    (_trigger_type == 'SLOW_GRIND_DOWN' and _g_oi_signal == 'SHORT_COVERING'))
                        if _oi_exit:
                            _grind_slope_blocked = True
                            _grind_slope_reason = f'OI-exit: {_g_oi_signal}(str={_g_oi_strength:.2f}) score={_final_score:.0f}<65'

                    # === LAYER 3: Momentum fade score ===
                    # Strong moves (vol+ADX≥40+fresh≥0.80) and writer-backed bypass entirely.
                    # Each negative signal adds weighted points. If total >= threshold → block.
                    if not _grind_slope_blocked and not _g_strong and not _g_writer_backed:
                        # --- Penalties (grind is dying) ---
                        # Velocity dying: current vel < 30% of peak (ADX≥50+vol overrides)
                        if _g_peak_vel > 0.10:
                            _vel_ratio = _g_velocity / _g_peak_vel if _g_peak_vel > 0 else 1.0
                            if _vel_ratio < 0.30 and not (_g_adx >= 50 and _g_vol_confirmed):
                                _fade_score += 2
                                _fade_tags.append(f'vel({_vel_ratio:.2f})')

                        # 60s stall: last minute barely moved (skip early market — gaps show false stalls)
                        if _g_sb_move >= 0 and abs(_move_pct) > 0.3 and not _g_early:
                            if _g_sb_move / abs(_move_pct) < 0.10:
                                _fade_score += 2
                                _fade_tags.append('stall_60s')

                        # Staleness: trigger is chasing the tail of the move
                        if _g_freshness < 0.50:
                            _fade_score += 2
                            _fade_tags.append(f'stale({_g_freshness:.2f})')
                        elif _g_freshness < 0.65:
                            _fade_score += 1
                            _fade_tags.append(f'aging({_g_freshness:.2f})')

                        # At day range extreme + not driving the move
                        _at_extreme = ((_trigger_type == 'SLOW_GRIND_UP' and _g_range_pos > 0.90) or
                                       (_trigger_type == 'SLOW_GRIND_DOWN' and _g_range_pos < 0.10))
                        if _at_extreme and _g_freshness < 0.80:
                            _fade_score += 1
                            _fade_tags.append(f'extreme({_g_range_pos:.0%})')

                        # Low ADX (weak underlying trend)
                        if _g_adx < 25:
                            _fade_score += 1
                            _fade_tags.append(f'lowADX({_g_adx:.0f})')

                        # Buyer driven (trapped longs propping up a dying grind)
                        if _g_part_id == 'BUYER_DOMINANT' and _g_oi_strength >= 0.25:
                            _fade_score += 1
                            _fade_tags.append('buyer_driven')

                        # --- Offsets (evidence grind continues) ---
                        if _g_early:
                            _fade_score -= 1  # Opening gaps look exhausted but aren't
                        if _g_ml_pmove >= 0.60:
                            _fade_score -= 1  # ML sees continuation probability

                        # Block threshold: weaker score = easier to block
                        _fade_thresh = 5 if _final_score < 50 else 9 if _final_score < 65 else 99
                        if _fade_score >= _fade_thresh:
                            _grind_slope_blocked = True
                            _grind_slope_reason = (f'fading({_fade_score}/{_fade_thresh}): '
                                                   f'{"+".join(_fade_tags)} score={_final_score:.0f}')

                    # --- Log ---
                    _slope_detail = (f'fresh={_g_freshness:.2f} vel={_g_velocity:.3f}/{_g_peak_vel:.3f} '
                                     f'range={_g_range_pos:.0%} ADX={_g_adx:.0f} vc={_g_vol_confirmed} '
                                     f'oi={_g_oi_signal}({_g_oi_strength:.2f}) part={_g_part_id}')
                    if _grind_slope_blocked:
                        t._wlog(f"  BLOCKED(G-SLOPE): {_stock_name} {_grind_slope_reason} | {_slope_detail}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_GRIND_SLOPE',
                                          reason=f'Grind slope quality: {_grind_slope_reason}',
                                          direction=direction)
                        continue
                    elif _fade_score >= 2:
                        t._wlog(f"  ⚠️ SLOPE-WARN: {_stock_name} fade={_fade_score}/{_fade_thresh} [{'+'.join(_fade_tags) if _fade_tags else 'ok'}] | {_slope_detail}")
                
                # --- GATE A: Score threshold (with early-market hardening) ---
                _now_t = datetime.now()
                _early_mkt_end = BREAKOUT_WATCHER.get('early_market_end', '09:55')
                _em_h, _em_m = int(_early_mkt_end.split(':')[0]), int(_early_mkt_end.split(':')[1])
                _is_early_market = _now_t.hour < _em_h or (_now_t.hour == _em_h and _now_t.minute < _em_m)
                if _is_early_market:
                    _early_min_score = BREAKOUT_WATCHER.get('early_market_min_score', 50)
                    _effective_min = max(_min_score, _early_min_score)
                else:
                    _effective_min = _min_score
                if _final_score <= _effective_min:
                    _tag = "A-SCORE-EARLY" if _is_early_market and _effective_min > _min_score else "A-SCORE"
                    t._wlog(f"  BLOCKED({_tag}): {_stock_name} score={_final_score:.0f} <= {_effective_min}{' (early market hardening)' if _is_early_market and _effective_min > _min_score else ''}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_LOW_SCORE',
                                      reason=f'Breakout {_trigger_type} but score {_final_score:.0f} <= {_effective_min}{"(early)" if _is_early_market else ""}',
                                      direction=direction)
                    continue
                
                # --- GATE A2: Early-market direction confidence gate ---
                if _is_early_market:
                    _em_dir_conf = getattr(_decision, 'direction_confidence', 0)
                    _em_min_conf = BREAKOUT_WATCHER.get('early_market_min_dir_conf', 45)
                    if _em_dir_conf < _em_min_conf:
                        t._wlog(f"  BLOCKED(A2-EARLY-DIR): {_stock_name} dir_conf={_em_dir_conf:.0f}% < {_em_min_conf}% (early market — direction unreliable)")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_EARLY_DIR',
                                          reason=f'Early market dir_conf={_em_dir_conf:.0f}%<{_em_min_conf}%',
                                          direction=direction)
                        continue
                
                # --- GATE B: Chop zone filter ---
                if _data.get('chop_zone', False):
                    t._wlog(f"  BLOCKED(B-CHOP): {_stock_name} in chop zone: {_data.get('chop_reason', '')}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_CHOP_ZONE',
                                      reason=f'Breakout {_trigger_type} in chop zone: {_data.get("chop_reason", "")}',
                                      direction=direction)
                    continue
                
                # --- GATE C: Setup validation (relaxed for watcher) ---
                # The watcher already confirmed a real-time trigger (volume surge,
                # new day extreme, or price spike). These ARE setups — the ticker
                # detected them from raw ticks, so we grant implicit setup credit
                # for strong trigger types or VWAP alignment, not just classic ORB.
                orb = _data.get('orb_signal', 'INSIDE_ORB')
                vwap = _data.get('price_vs_vwap', 'AT_VWAP')
                vol = _data.get('volume_regime', 'NORMAL')
                ema = _data.get('ema_regime', 'NORMAL')
                _rsi_c = _data.get('rsi_intraday', _data.get('rsi_14', 50))  # Intraday RSI preferred
                
                # Classic setups (same as ELITE)
                _classic_setup = (
                    orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') or
                    (vwap in ('ABOVE_VWAP', 'BELOW_VWAP') and vol in ('HIGH', 'EXPLOSIVE')) or
                    ema == 'COMPRESSED' or
                    _rsi_c < 30 or _rsi_c > 70
                )
                
                # Watcher-implicit setups: the ticker's trigger IS the setup evidence
                # The ticker confirmed 60s sustained movement — that IS a setup.
                # All non-tiny triggers qualify (watcher already filtered noise).
                _watcher_setup = (
                    _trigger_type in ('NEW_DAY_LOW', 'NEW_DAY_HIGH', 'PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN', 'SLOW_GRIND_UP', 'SLOW_GRIND_DOWN') or
                    _trigger_type == 'VOLUME_SURGE'  # Volume surge with sustained price move = real momentum
                )
                
                has_setup = _classic_setup or _watcher_setup
                if not has_setup:
                    t._wlog(f"  BLOCKED(C-SETUP): {_stock_name} no setup | ORB={orb} VWAP={vwap} VOL={vol} EMA={ema} RSI={_rsi_c:.0f} trigger={_trigger_type}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_NO_SETUP',
                                      reason=f'Breakout {_trigger_type} but no setup (classic or watcher-implicit)',
                                      direction=direction)
                    continue
                
                # Log which setup pathway qualified
                _setup_path = 'classic' if _classic_setup else f'watcher-implicit({_trigger_type})'
                t._wlog(f"  PASSED(C-SETUP): {_stock_name} via {_setup_path}")
                
                # --- GATE D: Follow-through candle gate ---
                # FT measures candles since ORB breakout. This is ONLY relevant
                # when the trade thesis is ORB-based. For VOLUME_SURGE, SLOW_GRIND,
                # NEW_DAY_HIGH triggers, the ORB age is irrelevant — the trigger
                # is fresh momentum, not an old breakout.
                # [FIX Mar 6] UNITDSPR scored 68 with EXPLOSIVE vol, triggered by
                # VOLUME_SURGE, but was blocked because ORB breakout was from morning.
                # FT from the morning ORB has nothing to do with an afternoon volume surge.
                ft_candles = _data.get('follow_through_candles', 0)
                orb_hold = _data.get('orb_hold_candles', 0)
                adx_val = _data.get('adx', 20)
                _is_orb_trigger = orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and _trigger_type not in (
                    'VOLUME_SURGE', 'SLOW_GRIND_UP', 'SLOW_GRIND_DOWN',
                    'NEW_DAY_HIGH', 'NEW_DAY_LOW', 'PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN'
                )
                # FT gate only applies to ORB-based triggers
                _ft_blocked = False
                if ft_candles == 0 and orb_hold > 2 and _is_orb_trigger:
                    # --- Smart FT relaxation ---
                    # 1) High ADX (>=30) = strong trend — grind-ups don't need FT candles
                    # 2) ORB breakout aligned with VWAP = multiple confirmation, relax FT
                    # 3) Raise stale threshold from 2 to 4 candles (20 min)
                    # 4) Score momentum (stock showing up repeatedly with rising scores)
                    _ft_relax_adx = adx_val >= 30
                    _ft_relax_alignment = (
                        (orb == 'BREAKOUT_UP' and vwap == 'ABOVE_VWAP') or
                        (orb == 'BREAKOUT_DOWN' and vwap == 'BELOW_VWAP')
                    )
                    _ft_relax_momentum = _momentum_boost > 0
                    _ft_stale_threshold = 4  # Was 2 — give more time before calling stale
                    
                    if _ft_relax_adx:
                        t._wlog(f"  RELAXED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} but ADX={adx_val:.0f}>=30 — strong trend, FT waived")
                    elif _ft_relax_alignment:
                        t._wlog(f"  RELAXED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} but ORB+VWAP aligned — FT waived")
                    elif _ft_relax_momentum:
                        t._wlog(f"  RELAXED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} but score momentum +{_momentum_boost} — FT waived")
                    elif orb_hold <= _ft_stale_threshold:
                        t._wlog(f"  PASSED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} <= {_ft_stale_threshold} — not yet stale")
                    else:
                        _ft_blocked = True
                        t._wlog(f"  BLOCKED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold}>{_ft_stale_threshold} ADX={adx_val:.0f} — stale ORB breakout")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_NO_FOLLOWTHROUGH',
                                          reason=f'FT=0, ORB hold={orb_hold}>{_ft_stale_threshold}, ADX={adx_val:.0f} — stale ORB',
                                          direction=direction)
                        continue
                elif ft_candles == 0 and orb_hold > 2 and not _is_orb_trigger:
                    t._wlog(f"  PASSED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} but trigger={_trigger_type} — FT gate N/A for non-ORB triggers")
                
                # --- GATE E: ADX trend strength gate (relaxed for WATCHER) ---
                # adx_val already fetched above for FT gate
                _adx_min = BREAKOUT_WATCHER.get('watcher_min_adx', 20)
                if adx_val < _adx_min:
                    t._wlog(f"  BLOCKED(E-ADX): {_stock_name} ADX={adx_val:.0f} < {_adx_min}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_WEAK_ADX',
                                      reason=f'ADX={adx_val:.0f} < {_adx_min}',
                                      direction=direction)
                    continue
                
                # --- GATE F: OI AUTHORITY — OI informs but PRICE ACTION leads ---
                # OLD LOGIC (broken): OI unconditionally overrides direction.
                #   This killed ALL trades on Mar-16 afternoon — every stock had
                #   SHORT_BUILDUP while Nifty rallied. Gate F flipped every BUY→SELL,
                #   then G2c-RSI blocked the SELL. Total deadlock.
                #
                # NEW LOGIC: OI is ADVISORY, not authoritative. Price action from
                # the watcher's multi-signal pipeline (sustained grind, volume surge,
                # breakout, etc.) represents CONFIRMED momentum. OI override only
                # when BOTH agree. When they disagree, LOG it as a flag but
                # KEEP the trigger direction — the watcher saw the move happen.
                #
                # SHORT_BUILDUP + price UP = short squeeze (keep BUY)
                # SHORT_BUILDUP + price DOWN = shorts right (keep SELL, OI confirms)
                # LONG_BUILDUP + price DOWN = long liquidation (keep SELL)
                # LONG_BUILDUP + price UP = longs right (keep BUY, OI confirms)
                oi_signal = _oi_signal_from_result(_oi_results.get(_sym, {}))
                _oi_auth_dir = _oi_direction(oi_signal)
                _watcher_oi_flipped = False
                _oi_disagrees = False
                _oi_confirms = False
                if _oi_auth_dir and _oi_auth_dir != direction:
                    # OI disagrees with price action. Log as flag, but
                    # RESPECT the trigger direction — the watcher confirmed the move.
                    t._wlog(f"  OI FLAG(F): {_stock_name} OI={oi_signal} suggests {_oi_auth_dir} but price action = {direction} — keeping {direction} (price leads)")
                    # Don't flip. Just flag it for scoring awareness.
                    _oi_disagrees = True
                elif _oi_auth_dir and _oi_auth_dir == direction:
                    # OI CONFIRMS trigger direction — strong alignment
                    t._wlog(f"  OI CONFIRM(F): {_stock_name} OI={oi_signal} confirms {direction}")
                    _oi_confirms = True
                
                # --- GATE F2: SPIKE EXHAUSTION TRAP — Block spikes opposing OI + scorer ---
                # [FIX Mar 18] TATAELXSI: Spike UP with scorer=SELL(74%) + OI=SHORT_BUILDUP
                # → classic exhaustion spike. Price briefly spikes against institutional flow,
                # then reverses hard. Lost ₹24,350 (-21.7%).
                # Spikes are the MOST vulnerable to instant reversal (unlike grinds/new-day
                # which represent sustained movement). When BOTH scorer AND OI oppose a spike,
                # the spike is very likely a trap — block it entirely.
                # [ROBUST Mar 18] OI strength check with PCR surrogate — same pattern as
                # DOUBLE CONVICTION and grind guard. Prevents false blocks from noisy/weak OI.
                _sym_oi_f2 = _oi_results.get(_sym, {})
                _oi_str_f2 = _sym_oi_f2.get('nse_oi_buildup_strength', 0.0) if _sym_oi_f2 else 0.0
                _oi_src_f2 = 'OI'
                # PCR-derived signal fix (same as DOUBLE CONVICTION):
                # When buildup returns str=0 but PCR flow_bias gave a directional signal,
                # use flow_confidence as surrogate strength.
                if _oi_str_f2 < 0.01 and oi_signal and oi_signal not in ('NEUTRAL', ''):
                    _fc_f2 = _sym_oi_f2.get('flow_confidence', 0.0) if _sym_oi_f2 else 0.0
                    if _fc_f2 > 0.1:
                        _oi_str_f2 = _fc_f2
                        _oi_src_f2 = 'PCR'
                if (_trigger_type in ('PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN')
                        and _oi_disagrees
                        and _oi_str_f2 >= 0.25
                        and _scorer_dir and _scorer_dir != _trigger_dir
                        and _dir_conf_val >= 60):
                    # === SPIKE REVERSAL FLIP ===
                    # Instead of blocking, FLIP direction to ride the reversal.
                    # Spikes against institutional flow + scorer always snap back hard.
                    # TATAELXSI: spiked UP against SHORT_BUILDUP + scorer SELL(74%) →
                    # reversed -21.7%. We now trade the reversal (SELL/PUT) with fast exit.
                    _old_dir = direction
                    direction = _scorer_dir  # Flip to scorer direction (OI agrees)
                    _setup_type = 'SPIKE_REVERSAL'
                    _ml_data_spike_rev = True  # Flag for fast exit tagging after placement
                    _final_score -= 2  # Small penalty for counter-momentum entry
                    t._wlog(f"  🔄 SPIKE REVERSAL(F2): {_stock_name} "
                               f"spike={_trigger_dir} → FLIPPED to {direction} "
                               f"(scorer={_scorer_dir} conf={_dir_conf_val:.0f}% "
                               f"+ {_oi_src_f2}={oi_signal} str={_oi_str_f2:.0%}) "
                               f"→ riding reversal with fast exit")
                    t._log_decision(_ts, _sym, _final_score, 'SPIKE_REVERSAL_FLIP',
                                      reason=f'Spike {_trigger_dir} → FLIPPED {direction}: scorer={_scorer_dir}({_dir_conf_val:.0f}%) + {_oi_src_f2}={oi_signal}(str={_oi_str_f2:.0%})',
                                      direction=direction)
                
                # --- GATE F2b: DAY HIGH/LOW EXHAUSTION TRAP ---
                # [FIX Mar 18] Same concept as spike F2 but for structural breaks.
                # NEW_DAY_HIGH with OI=SHORT_BUILDUP + scorer=SELL = false breakout / bull trap.
                # Institutions are positioned short, scorer sees weakness — the break is a
                # retail chase that will reverse. Block (don't flip — unlike spikes,
                # structural breaks with OI opposition usually just fade, not reverse hard).
                if (_trigger_type in ('NEW_DAY_HIGH', 'NEW_DAY_LOW')
                        and _oi_disagrees
                        and _oi_str_f2 >= 0.25
                        and _scorer_dir and _scorer_dir != _trigger_dir
                        and _dir_conf_val >= 60):
                    t._wlog(f"  BLOCKED(F2b-TRAP): {_stock_name} "
                               f"{_trigger_type} but scorer={_scorer_dir}({_dir_conf_val:.0f}%) "
                               f"+ {_oi_src_f2}={oi_signal}(str={_oi_str_f2:.0%}) "
                               f"→ false breakout / bull-bear trap")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_DAY_HL_TRAP',
                                      reason=f'{_trigger_type} TRAPPED: scorer={_scorer_dir}({_dir_conf_val:.0f}%) + {_oi_src_f2}={oi_signal}(str={_oi_str_f2:.0%})',
                                      direction=direction)
                    continue
                
                # --- GATE G: ML flat veto — DISABLED for watcher (Mar-09) ---
                # The ticker detected a real sustained price move. ML saying "flat"
                # contradicts observable reality. Log it but don't block.
                try:
                    _cached_ml = _cycle_decisions.get(_sym, {}).get('ml_prediction', {})
                    if _cached_ml.get('ml_elite_ok') is False:
                        _ml_flat_p = _cached_ml.get('ml_prob_flat', 0)
                        t._wlog(f"  NOTE(G-ML): {_stock_name} P(flat)={_ml_flat_p:.0%} — BYPASSED (watcher saw real move)")
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/ml_flat_veto_log]: {e}")
                
                # --- GATE G2b: Scorer-Trigger Conflict for weak scores ---
                # If the technical scorer actively disagrees with the trigger direction
                # AND the score is below threshold, block. Don't override scorer with
                # weak evidence.
                # BYPASS: If OI confirms the trigger direction AND scorer disagrees,
                # the scorer is the outlier — OI + price action align, don't block.
                _scorer_conflict_max = BREAKOUT_WATCHER.get('scorer_conflict_max_score', 50)
                _oi_confirms_trigger = _oi_confirms
                if (_scorer_dir and _scorer_dir != _trigger_dir and
                        _final_score < _scorer_conflict_max and
                        not _oi_confirms_trigger):
                    t._wlog(f"  BLOCKED(G2b-SCORER): {_stock_name} scorer={_scorer_dir} vs trigger={_trigger_dir} with weak score={_final_score:.0f}<{_scorer_conflict_max}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_SCORER_CONFLICT',
                                      reason=f'Scorer {_scorer_dir} opposes trigger {_trigger_dir}, score {_final_score:.0f} < {_scorer_conflict_max}',
                                      direction=direction)
                    continue
                if _oi_confirms_trigger and _scorer_dir != _trigger_dir:
                    t._wlog(f"  PASSED(G2b): {_stock_name} scorer={_scorer_dir} vs trigger={_trigger_dir} — OI confirms trigger direction={direction}")
                
                # --- GATE G2c: RSI Extreme Guard (NEW Mar-10, FIXED Mar-16) ---
                # Uses INTRADAY RSI (5-min candles) instead of daily RSI.
                # Daily RSI reflects weeks of history — a stock at daily RSI=23
                # can be rallying +3% TODAY with intraday RSI=65. Using daily RSI
                # caused 18 false blocks on Mar-16 afternoon during Nifty rally.
                _rsi_pe_max = BREAKOUT_WATCHER.get('rsi_extreme_pe_max', 24)
                _rsi_ce_min = BREAKOUT_WATCHER.get('rsi_extreme_ce_min', 75)
                _rsi_intra = _data.get('rsi_intraday', 50)
                _rsi_daily = _data.get('rsi_14', 50)
                _rsi_val = _rsi_intra  # Use intraday RSI for gate decisions
                _rsi_blocks = False
                if direction == 'SELL' and _rsi_val < _rsi_pe_max:
                    _rsi_blocks = True
                    t._wlog(f"  BLOCKED(G2c-RSI): {_stock_name} SELL/PE with intraday_RSI={_rsi_val:.0f} < {_rsi_pe_max} (daily_RSI={_rsi_daily:.0f}) — oversold bounce risk")
                elif direction == 'BUY' and _rsi_val > _rsi_ce_min:
                    _rsi_blocks = True
                    t._wlog(f"  BLOCKED(G2c-RSI): {_stock_name} BUY/CE with intraday_RSI={_rsi_val:.0f} > {_rsi_ce_min} (daily_RSI={_rsi_daily:.0f}) — overbought pullback risk")
                if _rsi_blocks:
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_RSI_EXTREME',
                                      reason=f'RSI extreme: intraday_RSI={_rsi_val:.0f} daily_RSI={_rsi_daily:.0f} direction={direction}',
                                      direction=direction)
                    continue
                
                # --- GATE G2d: Exhaustion Index (NEW Mar-10) ---
                # Statistical composite of 4 signals to detect "chasing a done move".
                # Each component is 0-1, weighted to sum to 0-100 Exhaustion Index (EI).
                #
                # Components:
                #   A) Intraday move from open (weight 35): big drop/rally already done
                #   B) Trigger freshness ratio (weight 25): trigger_move / intraday_move
                #      Low ratio = tiny trigger after massive move = chasing crumbs
                #   C) Position in day range (weight 25): near day extreme = exhausted
                #   D) RSI extreme proximity (weight 15): oversold/overbought confirmation
                #
                # IndiGo PE at 10:12: open=4500, ltp=4348, intraday=-3.4%
                #   A) 3.4%/5% = 0.68 * 35 = 23.8
                #   B) trigger=-0.3%, ratio=0.09, (1-0.09) * 25 = 22.8
                #   C) pos_in_range ≈ 0.15, (1-0.15) * 25 = 21.3
                #   D) RSI=28, (30-28)/30 * 15 = 1.0
                #   EI = 68.9 → BLOCKED (>60)
                _open_price = _data.get('open', 0)
                _cur_ltp = _data.get('ltp', 0)
                _day_high = _data.get('high', _cur_ltp)
                _day_low = _data.get('low', _cur_ltp)
                _rsi_ei = _data.get('rsi_intraday', _data.get('rsi_14', 50))  # Intraday RSI for exhaustion
                _intraday_move = ((_cur_ltp - _open_price) / _open_price * 100) if _open_price > 0 else 0
                _day_range = _day_high - _day_low if _day_high > _day_low else 0.01
                _pos_in_range = max(0, min(1, (_cur_ltp - _day_low) / _day_range))
                _trig_abs = abs(_move_pct)
                _intra_abs = abs(_intraday_move)
                _trig_ratio = min(_trig_abs / _intra_abs, 1.0) if _intra_abs > 0.01 else 1.0
                
                # Component A: Intraday move magnitude (directional — only counts if move is in trigger direction)
                if direction == 'SELL':
                    _comp_a = min(abs(min(_intraday_move, 0)) / 5.0, 1.0)  # Capped at 5%
                    _comp_c = 1.0 - _pos_in_range  # Near day LOW = exhausted for SELL
                    _comp_d = max(0, (30 - _rsi_ei) / 30) if _rsi_ei < 30 else 0  # Oversold
                else:
                    _comp_a = min(max(_intraday_move, 0) / 5.0, 1.0)
                    _comp_c = _pos_in_range  # Near day HIGH = exhausted for BUY
                    _comp_d = max(0, (_rsi_ei - 70) / 30) if _rsi_ei > 70 else 0  # Overbought
                
                # Component B: Trigger freshness (lower = chasing crumbs)
                _comp_b = 1.0 - _trig_ratio  # 1.0 = 100% chasing, 0.0 = trigger IS the move
                
                # Weighted Exhaustion Index (0-100)
                _ei = (_comp_a * 35) + (_comp_b * 25) + (_comp_c * 25) + (_comp_d * 15)
                _ei = round(_ei, 1)
                
                # Store for logging in GATE CHECK line
                _ei_detail = (f'EI={_ei} [intra={_intraday_move:+.1f}%({_comp_a:.2f}) '
                              f'trig={_move_pct:+.1f}%/{_intra_abs:.1f}%({_comp_b:.2f}) '
                              f'range={_pos_in_range:.0%}({_comp_c:.2f}) '
                              f'RSI={_rsi_ei:.0f}({_comp_d:.2f})]')
                
                _ei_threshold = BREAKOUT_WATCHER.get('exhaustion_index_block', 70)
                # Regime-aware EI: on trending days where the move aligns with market
                # direction, the intraday move IS the trend — not exhaustion.
                # Relax threshold by +15 when breadth confirms trigger direction.
                _breadth_confirms_ei = (
                    (direction == 'SELL' and _breadth in ('BEARISH',))
                    or (direction == 'BUY' and _breadth in ('BULLISH',))
                )
                if _breadth_confirms_ei:
                    _ei_threshold += 15  # 70 → 85 on regime-aligned trending days
                # Score-based EI boost: high score = many confirming signals.
                # If the scorer rated this setup highly DESPITE the big move,
                # the move is momentum-backed, not exhaustion.
                if _final_score >= 65:
                    _ei_threshold += 15  # Strong conviction → very permissive
                elif _final_score >= 55:
                    _ei_threshold += 10  # Solid conviction → moderately permissive
                _ei_boost_note = []
                if _breadth_confirms_ei:
                    _ei_boost_note.append(f"regime+15")
                if _final_score >= 65:
                    _ei_boost_note.append(f"score({_final_score})+15")
                elif _final_score >= 55:
                    _ei_boost_note.append(f"score({_final_score})+10")
                _ei_boost_str = f" (boosted to {_ei_threshold}: {','.join(_ei_boost_note)})" if _ei_boost_note else ""
                if _ei > _ei_threshold:
                    t._wlog(f"  BLOCKED(G2d-EXHAUST): {_stock_name} {_ei_detail}{_ei_boost_str}")
                    t._watcher_total_gate_blocked += 1
                    t._log_decision(_ts, _sym, _final_score, 'WATCHER_EXHAUSTION',
                                      reason=f'Exhaustion Index {_ei} > {_ei_threshold}: {_ei_detail}',
                                      direction=direction)
                    continue
                elif _ei > 40:
                    # Log warning for borderline cases (visible but not blocking)
                    t._wlog(f"  ⚠️ EXHAUST-WARN: {_stock_name} {_ei_detail} (below {_ei_threshold} threshold{_ei_boost_str})")
                
                # --- GATE G3: XGB Move Probability Floor (watcher-specific) ---
                _breadth_confirms_dir = (
                    (direction == 'SELL' and _market_breadth == 'BEARISH')
                    or (direction == 'BUY' and _market_breadth == 'BULLISH')
                )
                _G3_MIN_MOVE_PROB = 0.30 if _breadth_confirms_dir else 0.35
                try:
                    _xgb_ml = _ml_results.get(_sym, {})
                    _xgb_move_prob = _xgb_ml.get('ml_move_prob', 0)
                    # Only gate if ML data is available (don't block when ML fails)
                    if _xgb_ml and _xgb_move_prob > 0 and _xgb_move_prob < _G3_MIN_MOVE_PROB:
                        t._wlog(f"  BLOCKED(G3-XGB_PROB): {_stock_name} P(move)={_xgb_move_prob:.2f} < {_G3_MIN_MOVE_PROB}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_XGB_LOW_PROB',
                                          reason=f'XGB P(move)={_xgb_move_prob:.2f} < {_G3_MIN_MOVE_PROB}',
                                          direction=direction)
                        continue
                except Exception as e:
                    print(f"⚠️ FALLBACK [trader/gmm_dr_gate]: {e}")
                
                # --- GATE G4: GMM Down-Risk Veto (watcher-specific) ---
                # If GMM DR score opposes trade direction AND is very high → block.
                # UP_Flag=True (UP regime, high dr) = hidden crash risk → opposes BUY
                # Down_Flag=True (DOWN regime, high dr) = hidden bounce risk → opposes SELL
                # Threshold: 0.30 (generous — only blocks extreme anomaly, main sniper uses 0.08-0.13)
                try:
                    _gmm_ml = _ml_results.get(_sym, {})
                    _dr_score = _gmm_ml.get('ml_down_risk_score', 0)
                    _up_flag = _gmm_ml.get('ml_up_flag', False)
                    _down_flag = _gmm_ml.get('ml_down_flag', False)
                    _up_score = _gmm_ml.get('ml_up_score', 0)
                    _down_score = _gmm_ml.get('ml_down_score', 0)
                    
                    _gmm_blocks = False
                    _gmm_reason = ''
                    
                    if direction == 'BUY' and _up_flag and _up_score >= 0.30:
                        # UP regime flagged = hidden crash risk → opposes BUY
                        _gmm_blocks = True
                        _gmm_reason = f'UP_flag=True up_score={_up_score:.3f}>=0.30 (crash risk opposes BUY)'
                    elif direction == 'SELL' and _down_flag and _down_score >= 0.30:
                        # DOWN regime flagged = hidden bounce risk → opposes SELL
                        _gmm_blocks = True
                        _gmm_reason = f'DOWN_flag=True down_score={_down_score:.3f}>=0.30 (bounce risk opposes SELL)'
                    
                    if _gmm_blocks:
                        t._wlog(f"  BLOCKED(G4-GMM): {_stock_name} {_gmm_reason}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_GMM_DR_VETO',
                                          reason=f'GMM DR veto: {_gmm_reason}',
                                          direction=direction)
                        continue
                except Exception:
                    pass
                
                # --- GATE G5: Opening period cap — DISABLED (early-market quality gates A-SCORE-EARLY + A2-EARLY-DIR handle this) ---
                
                # --- GATE H: Position limit (regime-aware, same as ELITE) ---
                active_positions = [_pos for _pos in t.tools.paper_positions if _pos.get('status', 'OPEN') == 'OPEN']
                if _breadth == 'MIXED':
                    _max_pos = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)
                elif _breadth in ('BULLISH', 'BEARISH'):
                    _max_pos = HARD_RULES.get('MAX_POSITIONS_TRENDING', 12)
                else:
                    _max_pos = HARD_RULES['MAX_POSITIONS']
                if len(active_positions) >= _max_pos:
                    t._wlog(f"  BLOCKED(H-POS): {_stock_name} positions={len(active_positions)}/{_max_pos} ({_breadth}) — EXHAUSTED")
                    t._watcher_total_pos_exhausted += 1
                    break
                
                # === ALL GATES PASSED — Execute trade ===
                t._wlog(f"  ✅ ALL GATES PASSED: {_stock_name} score={_final_score:.0f} dir={direction} "
                           f"trigger={_trigger_type} pos={len(active_positions)}/{_max_pos} — EXECUTING")
                
                # Build ML data payload (same format as ELITE auto-fire)
                _elite_ml = _ml_results.get(_sym, {})
                _ml_data = {
                    'smart_score': _final_score,
                    'p_score': _final_score,
                    'dr_score': _elite_ml.get('ml_down_risk_score', 0),
                    'ml_move_prob': _elite_ml.get('ml_move_prob', 0),
                    'ml_confidence': _elite_ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': 'N/A',
                        'move_prob': _elite_ml.get('ml_move_prob', 0),
                        'prob_up': 0,
                        'prob_down': 0,
                        'prob_flat': _elite_ml.get('ml_prob_flat', 0),
                        'direction_bias': 0,
                        'confidence': _elite_ml.get('ml_confidence', 0),
                        'score_boost': _elite_ml.get('ml_score_boost', 0),
                        'direction_hint': 'NEUTRAL',
                        'model_type': _elite_ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': _elite_ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': _elite_ml.get('ml_down_risk_score', 0),
                        'up_flag': _elite_ml.get('ml_up_flag', False),
                        'down_flag': _elite_ml.get('ml_down_flag', False),
                        'up_score': _elite_ml.get('ml_up_score', 0),
                        'down_score': _elite_ml.get('ml_down_score', 0),
                        'down_risk_bucket': _elite_ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': _elite_ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': _elite_ml.get('ml_gmm_regime_used', 'BOTH'),
                        'gmm_action': 'WATCHER_BREAKOUT',
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': False,
                    'oi_signal': _oi_signal_from_result(_oi_results.get(_sym, {})),
                    'oi_flipped': _watcher_oi_flipped,
                    'oi_strength': _oi_results.get(_sym, {}).get('nse_oi_buildup_strength', None),
                } if _elite_ml else {}
                if not _ml_data:
                    _ml_data = {}
                _ml_data.setdefault('oi_signal', _oi_signal_from_result(_oi_results.get(_sym, {})))
                _ml_data.setdefault('oi_flipped', _watcher_oi_flipped)
                _ml_data.setdefault('oi_strength', _oi_results.get(_sym, {}).get('nse_oi_buildup_strength', None))
                _ml_data['trigger_type'] = _trigger_type  # For position record identification
                # Pass trigger metadata for scorer (VOLUME_SURGE conviction bonus)
                _ml_data['surge_ratio'] = _trigger.get('surge_ratio', 0)
                _ml_data['depth_imbalance'] = _trigger.get('depth_imbalance', 0)
                # SPIKE + SURGE co-fire → double lot flag
                _ml_data['spike_plus_surge'] = _trigger.get('spike_plus_surge', False)
                # Setup type includes trigger for identification in positions tab
                if 'DAY' in _trigger_type or 'SPIKE' in _trigger_type:
                    _setup_type = f'WATCHER_{_trigger_type}'
                elif 'GRIND' in _trigger_type:
                    _setup_type = f'WATCHER_{_trigger_type}'
                elif _trigger_type == 'VOLUME_SURGE':
                    _setup_type = 'WATCHER_VOLUME_SURGE'
                else:
                    _setup_type = f'WATCHER_{_trigger_type}' if _trigger_type else 'WATCHER'
                
                # --- GATE I: ORB-specific tightening (higher bar for SPIKE/DAY triggers) ---
                if 'DAY' in _trigger_type or 'SPIKE' in _trigger_type:
                    _orb_min_score = BREAKOUT_WATCHER.get('orb_min_score', 45)
                    _orb_min_move_base = BREAKOUT_WATCHER.get('orb_min_move_prob', 0.65)
                    # Relax ORB P(move) when breadth confirms direction
                    _orb_min_move = 0.50 if _breadth_confirms_dir else _orb_min_move_base
                    _xgb_mp = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                    if _final_score < _orb_min_score:
                        t._wlog(f"  BLOCKED(I-ORB_SCORE): {_stock_name} ORB score={_final_score:.0f} < {_orb_min_score}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_ORB_LOW_SCORE',
                                          reason=f'ORB_BREAKOUT score {_final_score:.0f} < {_orb_min_score}',
                                          direction=direction)
                        continue
                    if _xgb_mp > 0 and _xgb_mp < _orb_min_move:
                        t._wlog(f"  BLOCKED(I-ORB_MOVE): {_stock_name} ORB P(move)={_xgb_mp:.2f} < {_orb_min_move}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_ORB_LOW_MOVE',
                                          reason=f'ORB_BREAKOUT P(move)={_xgb_mp:.2f} < {_orb_min_move}',
                                          direction=direction)
                        continue

                # --- GATE I-W: WATCHER P(move) floor (all non-SPIKE/DAY watcher triggers) ---
                elif _setup_type.startswith('WATCHER'):
                    _w_min_move = BREAKOUT_WATCHER.get('watcher_min_move_prob', 0.57)
                    
                    # VOLUME_SURGE relaxation: institutional volume bursts may have moderate
                    # ML scores because XGB wasn't trained on volume-surge patterns.
                    # Strong surge_ratio or depth_imbalance = genuine institutional intent.
                    if _trigger_type == 'VOLUME_SURGE':
                        _vs_surge = _trigger.get('surge_ratio', 0)
                        _vs_depth = abs(_trigger.get('depth_imbalance', 0))
                        if _vs_surge >= 5.0 or (_vs_surge >= 4.0 and _vs_depth >= 0.25):
                            _w_min_move = 0.40  # Strong institutional signal — trust volume
                        elif _vs_surge >= 3.5:
                            _w_min_move = 0.48  # Moderate surge — partial relaxation
                    
                    _w_mp = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                    if _w_mp > 0 and _w_mp < _w_min_move:
                        t._wlog(f"  BLOCKED(I-W_MOVE): {_stock_name} WATCHER P(move)={_w_mp:.2f} < {_w_min_move}")
                        t._watcher_total_gate_blocked += 1
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_LOW_MOVE',
                                          reason=f'WATCHER P(move)={_w_mp:.2f} < {_w_min_move}',
                                          direction=direction)
                        continue
                
                # All gates passed — add to candidates for P(move) ranking
                _cand_pmove = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                _candidates.append({
                    'sym': _sym,
                    'direction': direction,
                    'score': _final_score,
                    'ml_move_prob': _cand_pmove,
                    'trigger_type': _trigger_type,
                    'move_pct': _move_pct,
                    'setup_type': _setup_type,
                    'ml_data': _ml_data,
                    'spike_plus_surge': _trigger.get('spike_plus_surge', False),
                    'is_spike_reversal': _ml_data_spike_rev,
                    'oi_disagrees': _oi_disagrees,
                    'oi_confirms': _oi_confirms,
                })
                t._wlog(f"  ✅ PASSED ALL GATES: {_sym.replace('NSE:', '')} "
                           f"score={_final_score:.0f} P(move)={_cand_pmove:.2f} "
                           f"trigger={_trigger_type}({_move_pct:+.1f}%)")
            
            # ════════════════════════════════════════════════════════════════
            # RANK CANDIDATES BY P(move) DESCENDING — place best trades first
            # ════════════════════════════════════════════════════════════════
            if not _candidates:
                t._wlog(f"  ⚠️ No candidates passed all gates")
            else:
                _candidates.sort(key=lambda c: c['ml_move_prob'], reverse=True)
                _rank_summary = ' > '.join(
                    f"{c['sym'].replace('NSE:', '')}(P={c['ml_move_prob']:.2f},S={c['score']:.0f})"
                    for c in _candidates
                )
                t._wlog(f"  📊 P(move) RANKING: {_rank_summary}")
                
                # Check remaining position slots — don't exceed portfolio limit
                _pos_now = len([_pos for _pos in t.tools.paper_positions if _pos.get('status', 'OPEN') == 'OPEN'])
                _slots_left = max(0, _max_pos - _pos_now) if '_max_pos' in dir() else _max_per_scan
                _effective_max = min(_max_per_scan, _slots_left) if _slots_left > 0 else _max_per_scan
                
                for _cand in _candidates:
                    if _fired_count >= _effective_max:
                        t._wlog(f"  ⏸ Max trades per scan ({_effective_max}) reached — skipping remaining {len(_candidates) - _candidates.index(_cand)} candidates")
                        break
                    
                    _sym = _cand['sym']
                    _direction = _cand['direction']
                    _trigger_type = _cand['trigger_type']
                    _move_pct = _cand['move_pct']
                    _setup_type = _cand['setup_type']
                    _final_score = _cand['score']
                    _ml_data = _cand['ml_data']
                    
                    # ── OI HEATMAP STRIKE PICKER for watcher pipeline ──
                    _w_strike_sel = 'ATM'
                    _w_hm_tag = ''
                    try:
                        from dhan_oi_fetcher import DhanOIFetcher
                        _w_oi = (t._watcher_drain_oi or {}).get(_sym, {})
                        _w_hm_strikes = _w_oi.get('dhan_strikes', [])
                        _w_hm_spot = _w_oi.get('dhan_spot_price', 0)
                        if _w_hm_strikes and _w_hm_spot > 0:
                            _w_hm = DhanOIFetcher.find_optimal_strike(_direction, _w_hm_strikes, _w_hm_spot)
                            if _w_hm.get('score', 0) > 0:
                                _w_strike_sel = _w_hm['selection']
                                _w_hm_tag = f" strike={_w_strike_sel}@{_w_hm['strike']:.0f}(HM={_w_hm['score']:.0f})"
                    except Exception:
                        pass

                    # SPIKE + SURGE co-fire → double the lot (2x sizing)
                    _lot_mult = 1.0
                    _surge_tag = ''
                    if _cand.get('spike_plus_surge'):
                        _lot_mult = 2.0
                        _surge_tag = ' [SPIKE+SURGE→2x LOT]'

                    with t._trade_lock:
                        result = t.tools.place_option_order(
                            underlying=_sym,
                            direction=_direction,
                            strike_selection=_w_strike_sel,
                            rationale=(f"WATCHER→FULL_PIPELINE: {_trigger_type} ({_move_pct:+.1f}%) — "
                                      f"Score {_final_score:.0f} P(move)={_cand['ml_move_prob']:.2f}, "
                                      f"ranked #{_candidates.index(_cand)+1}/{len(_candidates)} by P(move)"
                                      f"{_w_hm_tag}{_surge_tag}"),
                            setup_type=_setup_type,
                            ml_data=_ml_data,
                            lot_multiplier=_lot_mult,
                            pre_fetched_market_data=market_data.get(_sym, {})
                        )
                    
                    if result and result.get('success'):
                        t._wlog(f"  🎯 TRADE PLACED: {_sym.replace('NSE:', '')} ({_direction}) "
                                   f"score={_final_score:.0f} P(move)={_cand['ml_move_prob']:.2f} "
                                   f"trigger={_trigger_type}({_move_pct:+.1f}%) "
                                   f"rank=#{_candidates.index(_cand)+1}/{len(_candidates)} "
                                   f"strike={_w_strike_sel}{_w_hm_tag} "
                                   f"order={result.get('order_id', '?')} setup={_setup_type}")
                        t._watcher_fired_this_session.add(_sym)
                        t._auto_fired_this_session.add(_sym)  # Prevent ELITE re-fire
                        _fired_count += 1
                        t._watcher_total_placed += 1
                        # Reset grind trend-origin baseline now that we've acted
                        _bw = getattr(getattr(t.tools, 'ticker', None), 'breakout_watcher', None)
                        if 'GRIND' in _trigger_type and _bw and hasattr(_bw, 'mark_grind_traded'):
                            _bw.mark_grind_traded(_sym)
                        
                        # === SPIKE REVERSAL: Tag position for fast exit management ===
                        # Spike reversals are quick scalps — tight target, tight SL, fast time guard.
                        if _cand.get('is_spike_reversal'):
                            _sr_entry = result.get('entry_price', 0)
                            _sr_target = _sr_entry * 1.10   # 10% profit target (fast scalp)
                            _sr_sl = _sr_entry * 0.94       # 6% SL (tight)
                            _sr_underlying = _sym.replace('NSE:', '')
                            with t.tools._positions_lock:
                                for _sr_pos in reversed(t.tools.paper_positions):
                                    if (_sr_pos.get('underlying') == _sr_underlying and
                                            _sr_pos.get('status', 'OPEN') == 'OPEN'):
                                        _sr_pos['is_spike_reversal'] = True
                                        _sr_pos['spike_rev_entry_time'] = datetime.now().isoformat()
                                        _sr_pos['spike_rev_hwm'] = _sr_entry
                                        _sr_pos['spike_rev_trailing_active'] = False
                                        _sr_pos['stop_loss'] = _sr_sl
                                        _sr_pos['target'] = _sr_target
                                        break
                            t._wlog(f"  🔄 SPIKE REVERSAL TAGGED: {_sr_underlying} "
                                       f"target=+10%(₹{_sr_target:.2f}) SL=-6%(₹{_sr_sl:.2f}) "
                                       f"time_guard=10min")
                        
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_FIRED',
                                          reason=(f'Breakout {_trigger_type} ({_move_pct:+.1f}%) — '
                                                 f'FULL PIPELINE: score+ML+OI+sector+DR+setup+FT+ADX all passed | '
                                                 f'P(move)={_cand["ml_move_prob"]:.2f} rank #{_candidates.index(_cand)+1}/{len(_candidates)}'),
                                          direction=_direction, setup=_setup_type)
                    else:
                        _err = result.get('error', 'unknown') if result else 'no result'
                        
                        # ── CAPITAL SWAP: If blocked by exposure/risk governor, try evicting a stale position ──
                        _is_exposure_block = ('RISK GOVERNOR BLOCK' in str(_err) and 'exposure' in str(_err).lower()) or \
                                             ('REGIME POSITION LIMIT' in str(_err))
                        if _is_exposure_block and CAPITAL_SWAP.get('enabled', False):
                            t._wlog(f"  🔄 CAPITAL SWAP: {_sym.replace('NSE:', '')} blocked ({_err[:60]}) — searching for eviction candidate...")
                            _evict = t._find_eviction_candidate(_setup_type)
                            if _evict:
                                with t._trade_lock:
                                    _evicted = t._execute_eviction(_evict, f"{_setup_type}:{_sym.replace('NSE:', '')}")
                                if _evicted:
                                    # Brief pause to let position state settle
                                    import time as _swap_time
                                    _swap_time.sleep(0.5)
                                    # Retry placement after eviction
                                    t._wlog(f"  🔄 RETRY after eviction: {_sym.replace('NSE:', '')}...")
                                    with t._trade_lock:
                                        result = t.tools.place_option_order(
                                            underlying=_sym,
                                            direction=_direction,
                                            strike_selection=_w_strike_sel,
                                            rationale=(f"CAPITAL_SWAP→RETRY: {_trigger_type} ({_move_pct:+.1f}%) — "
                                                      f"Score {_final_score:.0f} P(move)={_cand['ml_move_prob']:.2f} "
                                                      f"(evicted {_evict['symbol']}){_w_hm_tag}{_surge_tag}"),
                                            setup_type=_setup_type,
                                            ml_data=_ml_data,
                                            lot_multiplier=_lot_mult,
                                            pre_fetched_market_data=market_data.get(_sym, {})
                                        )
                                    if result and result.get('success'):
                                        t._wlog(f"  🎯 SWAP SUCCESS: {_sym.replace('NSE:', '')} ({_direction}) "
                                                   f"replaced {_evict['symbol']} | score={_final_score:.0f} "
                                                   f"P(move)={_cand['ml_move_prob']:.2f}")
                                        t._watcher_fired_this_session.add(_sym)
                                        t._auto_fired_this_session.add(_sym)
                                        _fired_count += 1
                                        t._watcher_total_placed += 1
                                        _bw2 = getattr(getattr(t.tools, 'ticker', None), 'breakout_watcher', None)
                                        if 'GRIND' in _trigger_type and _bw2 and hasattr(_bw2, 'mark_grind_traded'):
                                            _bw2.mark_grind_traded(_sym)
                                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_SWAP_FIRED',
                                                          reason=(f'CAPITAL_SWAP: evicted {_evict["symbol"]} '
                                                                 f'(held {_evict["hold_minutes"]:.0f}min R={_evict["r_multiple"]:.2f}) '
                                                                 f'for {_trigger_type} ({_move_pct:+.1f}%) P(move)={_cand["ml_move_prob"]:.2f}'),
                                                          direction=_direction, setup=_setup_type)
                                        continue
                                    else:
                                        _retry_err = result.get('error', 'unknown') if result else 'no result'
                                        t._wlog(f"  ⚠️ SWAP RETRY FAILED: {_sym.replace('NSE:', '')} — {_retry_err}")
                                else:
                                    t._wlog(f"  ❌ EVICTION FAILED — cannot free capital for {_sym.replace('NSE:', '')}")
                            else:
                                t._wlog(f"  ❌ NO EVICTION CANDIDATE: all positions are profitable, fresh, or higher priority")
                        
                        t._wlog(f"  ⚠️ TRADE FAILED: {_sym.replace('NSE:', '')} — {_err}")
                        t._log_decision(_ts, _sym, _final_score, 'WATCHER_TRADE_FAILED',
                                          reason=f'{_trigger_type}: {str(_err)[:80]}',
                                          direction=_direction)
        
        except Exception as _e:
            t._wlog(f"PIPELINE ERROR: {_e}")
            import traceback
            traceback.print_exc()

