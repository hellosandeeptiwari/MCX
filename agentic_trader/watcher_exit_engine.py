"""Watcher Momentum Exit Engine — extracted from autonomous_trader.py"""
from datetime import datetime
from config import calc_brokerage
from oi_watcher_engine import _oi_signal_from_result, _oi_direction


class WatcherExitEngine:
    def __init__(self, trader):
        self.trader = trader

    def check_momentum_exits(self):
        """Check open WATCHER positions for momentum reversal and exit.
        
        Detects when a spike has peaked (for CE/BUY trades) or a crater has
        bottomed (for PE/SELL trades) using multi-signal confirmation:
        
        Signals: price reversal, volume dry-up, momentum decay, pressure shift,
                 reversal acceleration, option premium reversal.
        
        Adaptive thresholds:
        - Profit-tiered: bigger gains = tighter stops (protect large profits)
        - CE/PE asymmetric: CE dies faster on reversal, PE bounces are sharp
        - Multi-signal: more confirmations = lower threshold needed
        
        After exit, cooldown is BYPASSED — the watcher can immediately re-enter
        if another spike/crater forms on the same symbol.
        """
        t = self.trader
        from config import BREAKOUT_WATCHER
        _cfg = BREAKOUT_WATCHER.get('momentum_exit', {})
        if not _cfg.get('enabled', False):
            return
        
        import time as _wme_time
        
        # --- Auto-discover new WATCHER trades not yet tracked ---
        with t.tools._positions_lock:
            _active = [_pos.copy() for _pos in t.tools.paper_positions if _pos.get('status', 'OPEN') == 'OPEN']
        
        _watcher_trades = [_t for _t in _active
                           if _t.get('setup_type') in ('WATCHER', 'ORB_BREAKOUT', 'OI_WATCHER')
                           and _t.get('is_option', False)
                           and not _t.get('is_debit_spread', False)
                           and not _t.get('is_credit_spread', False)]
        
        # Register any new watcher trades
        for _t in _watcher_trades:
            _opt_sym = _t['symbol']
            if _opt_sym in t._watcher_momentum_tracker:
                continue
            _ul = _t.get('underlying', '')
            if not _ul:
                continue
            # Parse entry time
            try:
                _entry_epoch = datetime.fromisoformat(_t.get('timestamp', '')).timestamp()
            except Exception:
                _entry_epoch = _wme_time.time()
            # Skip positions that existed before this bot session (stale UL price on restart)
            if _entry_epoch < t._wme_session_start:
                continue
            # Get underlying LTP from WebSocket cache
            _ul_ltp = 0
            if t.tools.ticker and t.tools.ticker.connected:
                _ul_prices = t.tools.ticker.get_ltp_batch([_ul])
                _ul_ltp = _ul_prices.get(_ul, 0)
            if _ul_ltp <= 0:
                continue
            
            _direction = _t.get('direction', 'BUY')
            _entry_price = _t.get('avg_price', 0)
            _opt_type = 'CE' if 'CE' in _opt_sym.upper() else 'PE' if 'PE' in _opt_sym.upper() else '?'
            _setup_type = _t.get('setup_type', 'WATCHER')
            t._watcher_momentum_tracker[_opt_sym] = {
                'underlying': _ul,
                'direction': _direction,
                'opt_type': _opt_type,
                'setup_type': _setup_type,
                'entry_ul_price': _ul_ltp,
                'entry_opt_price': _entry_price,
                'entry_time': _entry_epoch,
                'peak_price': _ul_ltp,
                'trough_price': _ul_ltp,
                'peak_opt_price': _entry_price,  # Track option premium peak
                # Multi-signal tracking
                'price_samples': [(_wme_time.time(), _ul_ltp)],
                'opt_price_samples': [(_wme_time.time(), _entry_price)],  # Option premium history
                'vol_samples': [],       # (timestamp, cumulative_volume)
                'peak_momentum': 0.0,    # Max price velocity seen (%/s, direction-adjusted)
                'reversal_samples': [],  # Track reversal speed over time for acceleration detection
            }
            # OI_WATCHER: attach entry OI snapshot for thesis-based exit
            if _setup_type == 'OI_WATCHER':
                _oi_snap = t._oi_watcher_entry_snapshots.get(_ul, {})
                t._watcher_momentum_tracker[_opt_sym]['entry_oi'] = {
                    'signal': _oi_snap.get('signal', _t.get('oi_signal', '')),
                    'strength': _oi_snap.get('strength', 0.0),
                    'direction': _oi_snap.get('direction', _direction),
                    'participant': _oi_snap.get('participant', 'UNKNOWN'),
                }
                t._watcher_momentum_tracker[_opt_sym]['_last_oi_exit_check'] = 0
            else:
                # ALL trades: capture entry OI for shield/accelerator
                # Fetch OI at registration time to establish baseline
                try:
                    if t._oi_analyzer:
                        _reg_oi = t._oi_analyzer.analyze(_ul)
                        if _reg_oi:
                            _reg_sig = _oi_signal_from_result(_reg_oi)
                            t._watcher_momentum_tracker[_opt_sym]['entry_oi'] = {
                                'signal': _reg_sig,
                                'strength': _reg_oi.get('nse_oi_buildup_strength', 0.0),
                                'direction': _oi_direction(_reg_sig) or _direction,
                                'participant': _reg_oi.get('oi_participant_id', 'UNKNOWN'),
                            }
                except Exception:
                    pass  # OI not available at registration — no baseline
            _ul_short = _ul.replace('NSE:', '')
            t._wlog(f"   🌊 WME: Tracking {_ul_short} ({_direction}/{_opt_type}) UL=₹{_ul_ltp:.2f} opt={_opt_sym.split(':')[-1]} entry=₹{_entry_price:.2f}{' [OI_WATCHER]' if _setup_type == 'OI_WATCHER' else ''}")
        
        # Clean tracker: remove entries whose positions are no longer open
        _active_opt_syms = {_t['symbol'] for _t in _watcher_trades}
        _stale = [k for k in t._watcher_momentum_tracker if k not in _active_opt_syms]
        for _s in _stale:
            del t._watcher_momentum_tracker[_s]
        
        if not t._watcher_momentum_tracker:
            return
        
        # --- Get underlying + option QUOTES via WebSocket cache ---
        _all_ul = list({v['underlying'] for v in t._watcher_momentum_tracker.values()})
        _all_opts = list(t._watcher_momentum_tracker.keys())
        if not t.tools.ticker or not t.tools.ticker.connected:
            return
        _ul_quotes = t.tools.ticker.get_quote_batch(_all_ul)
        _ul_prices = {sym: q.get('last_price', 0) for sym, q in _ul_quotes.items()}
        # Get option prices for premium tracking
        _opt_prices_batch = t.tools.ticker.get_ltp_batch(_all_opts)
        # Fallback for missing UL prices
        for _sym in _all_ul:
            if _sym not in _ul_prices or _ul_prices[_sym] <= 0:
                _ltp_batch = t.tools.ticker.get_ltp_batch([_sym])
                _ul_prices[_sym] = _ltp_batch.get(_sym, 0)
        
        # --- Load config parameters ---
        _reversal_pct = _cfg.get('reversal_pct', 0.5) / 100
        _confirmed_rev_pct = _cfg.get('confirmed_reversal_pct', 0.25) / 100
        _partial_rev_pct = _cfg.get('partial_confirm_reversal_pct', 0.35) / 100
        _min_hold_s = _cfg.get('min_hold_seconds', 60)
        _min_move_pct = _cfg.get('min_favorable_move_pct', 0.5) / 100
        _bypass_cooldown = _cfg.get('bypass_cooldown', True)
        _only_profit = _cfg.get('only_in_profit', True)
        _skip_trailing = _cfg.get('skip_trailing_active', True)
        _vol_dryup_ratio = _cfg.get('volume_dryup_ratio', 0.30)
        _mom_decay_thresh = _cfg.get('momentum_decay_threshold', 0.70)
        _pressure_enabled = _cfg.get('pressure_shift_enabled', True)
        _pressure_ratio = _cfg.get('pressure_shift_ratio', 1.5)
        _sample_window = _cfg.get('sample_window_seconds', 180)
        _premium_rev_pct = _cfg.get('premium_reversal_pct', 8.0) / 100
        _premium_rev_confirmed = _cfg.get('premium_reversal_confirmed_pct', 5.0) / 100
        _profit_tiers = _cfg.get('profit_tiers', [])
        _ce_mult = _cfg.get('ce_reversal_multiplier', 0.85)
        _pe_mult = _cfg.get('pe_reversal_multiplier', 0.90)
        _accel_enabled = _cfg.get('reversal_accel_enabled', True)
        _accel_threshold = _cfg.get('reversal_accel_threshold', 1.5)
        # --- Trend Shield params ---
        _retrace_enabled = _cfg.get('retracement_floor_enabled', True)
        _ul_retrace_factor = _cfg.get('ul_retracement_factor', 0.382)
        _prem_retrace_factor = _cfg.get('premium_retracement_factor', 0.382)
        _min_prem_gain = _cfg.get('min_premium_gain_pct', 5.0)
        _grace_seconds = _cfg.get('armed_grace_seconds', 30)
        _deepen_mult = _cfg.get('armed_deepen_multiplier', 1.5)
        _now = _wme_time.time()
        _exits_to_process = []
        
        # --- Check each tracked position ---
        for _opt_sym, _track in list(t._watcher_momentum_tracker.items()):
            _ul = _track['underlying']
            _ul_ltp = _ul_prices.get(_ul, 0)
            if _ul_ltp <= 0:
                continue
            
            _direction = _track['direction']
            _opt_type = _track.get('opt_type', '?')
            _entry_ul = _track['entry_ul_price']
            _entry_opt = _track.get('entry_opt_price', 0)
            _ul_quote = _ul_quotes.get(_ul, {})
            _opt_ltp = _opt_prices_batch.get(_opt_sym, 0)
            
            # ── ALWAYS collect samples (even before min_hold) to build baselines ──
            _track['price_samples'].append((_now, _ul_ltp))
            _cum_vol = _ul_quote.get('volume', 0)
            if _cum_vol > 0:
                _track['vol_samples'].append((_now, _cum_vol))
            if _opt_ltp > 0:
                _track['opt_price_samples'].append((_now, _opt_ltp))
            
            # Trim samples to rolling window
            _cutoff = _now - _sample_window
            _track['price_samples'] = [(ts, p) for ts, p in _track['price_samples'] if ts >= _cutoff]
            _track['vol_samples'] = [(ts, v) for ts, v in _track['vol_samples'] if ts >= _cutoff]
            _track['opt_price_samples'] = [(ts, p) for ts, p in _track.get('opt_price_samples', []) if ts >= _cutoff]
            # Keep reversal samples shorter (60s window)
            _track['reversal_samples'] = [(ts, r) for ts, r in _track.get('reversal_samples', []) if ts >= _now - 60]
            
            # ── Update peak/trough price (underlying) ──
            if _direction == 'BUY':
                if _ul_ltp > _track['peak_price']:
                    _track['peak_price'] = _ul_ltp
            elif _direction == 'SELL':
                if _ul_ltp < _track['trough_price']:
                    _track['trough_price'] = _ul_ltp
            
            # ── Update option premium peak ──
            if _opt_ltp > 0 and _opt_ltp > _track.get('peak_opt_price', 0):
                _track['peak_opt_price'] = _opt_ltp
            
            # ── Update peak momentum (max velocity seen over any ~15s window) ──
            _ps = _track['price_samples']
            if len(_ps) >= 4:
                _p_now = _ps[-1]
                _p_prev_idx = max(0, len(_ps) - 4)  # ~15-20s ago
                _p_prev = _ps[_p_prev_idx]
                _pdt = _p_now[0] - _p_prev[0]
                if _pdt > 0 and _p_prev[1] > 0:
                    _vel = (_p_now[1] - _p_prev[1]) / _p_prev[1] / _pdt  # %/s
                    # Track peak velocity in favorable direction
                    if _direction == 'BUY' and _vel > _track['peak_momentum']:
                        _track['peak_momentum'] = _vel
                    elif _direction == 'SELL' and _vel < _track['peak_momentum']:
                        _track['peak_momentum'] = _vel  # Negative for SELL
            
            # ── Min hold time: let the trade breathe ──
            if (_now - _track['entry_time']) < _min_hold_s:
                continue
            
            # ════════════════════════════════════════════════════════════════
            # OI_WATCHER THESIS EXIT — pure OI trade, exit when OI thesis breaks
            # Runs BEFORE standard WME gates (trailing, favorable move, premium floor)
            # because OI thesis break = immediate exit regardless of price state.
            # ════════════════════════════════════════════════════════════════
            if _track.get('setup_type') == 'OI_WATCHER':
                _oiw_cfg = BREAKOUT_WATCHER.get('oi_watcher_exit', {})
                if _oiw_cfg.get('enabled', True):
                    _oiw_min_hold = _oiw_cfg.get('min_hold_seconds', 60)
                    _oiw_check_iv = _oiw_cfg.get('oi_check_interval_seconds', 45)
                    _oiw_held = _now - _track['entry_time']
                    # [FIX Mar 24] PROFIT GATE: If trade is in profit ≥1%, leave it alone.
                    # OI exit engine only activates when profit < 1% or in loss.
                    # Let trailing stop manage winners — OI thesis exit is for cutting losers.
                    _oiw_profit_skip_pct = _oiw_cfg.get('profit_skip_pct', 1.0)
                    if _entry_opt > 0 and _opt_ltp > 0:
                        _oiw_cur_gain_pct = ((_opt_ltp - _entry_opt) / _entry_opt) * 100
                        if _oiw_cur_gain_pct >= _oiw_profit_skip_pct:
                            # Trade is profitable — skip OI exit engine entirely
                            continue
                    if _oiw_held >= _oiw_min_hold:
                        _last_oiw_check = _track.get('_last_oi_exit_check', 0)
                        if (_now - _last_oiw_check) >= _oiw_check_iv:
                            _track['_last_oi_exit_check'] = _now
                            _oiw_exit_reason = None
                            _oiw_detail = ''
                            _entry_oi = _track.get('entry_oi', {})
                            _entry_oi_dir = _entry_oi.get('direction', _direction)
                            _entry_oi_str = _entry_oi.get('strength', 0.0)
                            _entry_oi_part = _entry_oi.get('participant', 'UNKNOWN')
                            _entry_oi_sig = _entry_oi.get('signal', '')
                            _oiw_fetch_ok = False
                            try:
                                _oiw_data = t._oi_analyzer.analyze(_track['underlying']) if t._oi_analyzer else None
                                if _oiw_data:
                                    _oiw_fetch_ok = True
                                    _cur_oi_sig = _oi_signal_from_result(_oiw_data)
                                    _cur_oi_dir = _oi_direction(_cur_oi_sig)
                                    _cur_oi_str = _oiw_data.get('nse_oi_buildup_strength', 0.0)
                                    _cur_oi_part = _oiw_data.get('oi_participant_id', 'UNKNOWN')
                                    _track['_cur_oi_signal'] = _cur_oi_sig
                                    _track['_cur_oi_strength'] = _cur_oi_str
                                    _track['_cur_oi_participant'] = _cur_oi_part
                                    # [FIX Mar 19] Require consecutive confirmations before thesis exit
                                    # OI data is inherently noisy — one weak reading doesn't mean thesis is dead.
                                    # Track consecutive bad readings and only exit when confirmed N times.
                                    _confirms_needed = _oiw_cfg.get('consecutive_confirms_required', 2)
                                    _oiw_tentative_reason = None
                                    _oiw_tentative_detail = ''
                                    # Condition 1: OI Direction Flip
                                    if _oiw_cfg.get('direction_flip_exit', True):
                                        if _cur_oi_dir and _cur_oi_dir != _entry_oi_dir:
                                            _oiw_tentative_reason = 'OI_DIRECTION_FLIP'
                                            _oiw_tentative_detail = f'entry={_entry_oi_sig}→now={_cur_oi_sig} dir {_entry_oi_dir}→{_cur_oi_dir}'
                                    # Condition 2: OI Signal went NEUTRAL (buildup dissolved)
                                    if not _oiw_tentative_reason and _oiw_cfg.get('signal_neutral_exit', True):
                                        if _cur_oi_sig in ('NEUTRAL', '') and _entry_oi_sig not in ('NEUTRAL', ''):
                                            _oiw_tentative_reason = 'OI_SIGNAL_NEUTRAL'
                                            _oiw_tentative_detail = f'entry={_entry_oi_sig} str={_entry_oi_str:.2f}→now NEUTRAL str={_cur_oi_str:.2f}'
                                    # Condition 3: Strength Collapse
                                    if not _oiw_tentative_reason and _oiw_cfg.get('strength_collapse_exit', True):
                                        _collapse_ratio = _oiw_cfg.get('strength_collapse_ratio', 0.25)
                                        if _entry_oi_str > 0 and _cur_oi_str < _entry_oi_str * _collapse_ratio:
                                            _only_losing = _oiw_cfg.get('strength_collapse_only_losing', True)
                                            _oiw_pnl_check = (_opt_ltp - _entry_opt) if _opt_ltp > 0 and _entry_opt > 0 else 0
                                            if not _only_losing or _oiw_pnl_check <= 0:
                                                _oiw_tentative_reason = 'OI_STRENGTH_COLLAPSE'
                                                _oiw_tentative_detail = f'str {_entry_oi_str:.2f}→{_cur_oi_str:.2f} ({_cur_oi_str/_entry_oi_str*100:.0f}% of entry)'
                                    # Condition 4: Participant Flip (writers left)
                                    if not _oiw_tentative_reason and _oiw_cfg.get('participant_flip_exit', True):
                                        if _entry_oi_part == 'WRITER_DOMINANT' and _cur_oi_part == 'BUYER_DOMINANT':
                                            _oiw_tentative_reason = 'OI_WRITER_UNWINDING'
                                            _oiw_tentative_detail = f'WRITER_DOMINANT→BUYER_DOMINANT (writers left)'
                                    # [FIX Mar 19] Consecutive confirmation logic
                                    # [SMART Mar 19] Price-aware exit: if trade is profitable, OI thesis break
                                    # is less relevant — the thesis WORKED, price responded. Hand off to trailing stop.
                                    _oiw_prem_gain_pct = 0.0
                                    if _entry_opt > 0 and _opt_ltp > 0:
                                        _oiw_prem_gain_pct = ((_opt_ltp - _entry_opt) / _entry_opt) * 100
                                    if _oiw_tentative_reason:
                                        # If premium is not losing, the thesis is alive — don't kill on OI noise.
                                        # OI is a LEADING indicator; once price has responded, OI fading is expected.
                                        # Let the SL/trailing stop manage the trade instead.
                                        _prem_bypass = _oiw_cfg.get('premium_bypass_pct', 0.0)
                                        if _oiw_prem_gain_pct >= _prem_bypass:
                                            _oiw_ul_warn = _track['underlying'].replace('NSE:', '')
                                            t._wlog(f"      💰 OI_WATCHER {_oiw_ul_warn}: {_oiw_tentative_reason} but premium UP {_oiw_prem_gain_pct:+.1f}% — thesis worked, handing to trailing stop")
                                            _track['_oiw_bad_readings'] = 0
                                            _track['_oiw_bad_reason'] = ''
                                            # Don't set exit — let standard WME trailing manage
                                        else:
                                            _prev_bad = _track.get('_oiw_bad_readings', 0)
                                            _prev_reason = _track.get('_oiw_bad_reason', '')
                                            # Same reason as last time → increment; different → reset to 1
                                            if _oiw_tentative_reason == _prev_reason:
                                                _track['_oiw_bad_readings'] = _prev_bad + 1
                                            else:
                                                _track['_oiw_bad_readings'] = 1
                                            _track['_oiw_bad_reason'] = _oiw_tentative_reason
                                            if _track['_oiw_bad_readings'] >= _confirms_needed:
                                                _oiw_exit_reason = _oiw_tentative_reason
                                                _oiw_detail = _oiw_tentative_detail + f' (confirmed {_track["_oiw_bad_readings"]}x, prem={_oiw_prem_gain_pct:+.1f}%)'
                                            else:
                                                # Not yet confirmed — log warning but don't exit
                                                _oiw_ul_warn = _track['underlying'].replace('NSE:', '')
                                                t._wlog(f"      ⚠️ OI_WATCHER {_oiw_ul_warn}: {_oiw_tentative_reason} reading {_track['_oiw_bad_readings']}/{_confirms_needed} (not yet confirmed, prem={_oiw_prem_gain_pct:+.1f}%)")
                                    else:
                                        # Good reading — reset bad counter
                                        _track['_oiw_bad_readings'] = 0
                                        _track['_oiw_bad_reason'] = ''
                            except Exception as _oiw_exc:
                                pass  # OI fetch failed — skip this cycle
                            
                            # === OI THESIS WEAKENING → TIGHTEN TRAILING STOP ===
                            # When OI strength drops >40% from entry but NOT at collapse level,
                            # tighten the trailing stop so a dying trade gets cut faster.
                            # Don't exit — just reduce giveback room. If trade recovers, tight stop still gives room.
                            # Only run if OI fetch succeeded this cycle — stale 0.0 would falsely trigger.
                            if not _oiw_exit_reason and _oiw_fetch_ok and _oiw_cfg.get('thesis_weakening_enabled', True):
                                _tw_ratio = _oiw_cfg.get('thesis_weakening_ratio', 0.60)
                                _tw_confirms = _oiw_cfg.get('thesis_weakening_confirms', 2)
                                _cur_str_tw = _track.get('_cur_oi_strength', 0.0)
                                if _entry_oi_str > 0 and _cur_str_tw < _entry_oi_str * _tw_ratio:
                                    _prev_weak = _track.get('_oiw_weak_readings', 0)
                                    _track['_oiw_weak_readings'] = _prev_weak + 1
                                    if _track['_oiw_weak_readings'] >= _tw_confirms:
                                        _em_tw = t.exit_manager.get_trade_state(_opt_sym) if hasattr(t, 'exit_manager') else None
                                        if _em_tw and not _em_tw.oi_thesis_weakened:
                                            _em_tw.oi_thesis_weakened = True
                                            _tw_ul = _track['underlying'].replace('NSE:', '')
                                            _tw_pct = (_cur_str_tw / _entry_oi_str * 100) if _entry_oi_str > 0 else 0
                                            t._wlog(f"      🔧 OI_WATCHER {_tw_ul}: Thesis WEAKENED — str {_entry_oi_str:.2f}→{_cur_str_tw:.2f} ({_tw_pct:.0f}% of entry) | Tightening trailing stop")
                                else:
                                    # OI strength recovered — reset weak counter & relax trail
                                    if _track.get('_oiw_weak_readings', 0) > 0:
                                        _track['_oiw_weak_readings'] = 0
                                        _em_tw = t.exit_manager.get_trade_state(_opt_sym) if hasattr(t, 'exit_manager') else None
                                        if _em_tw and _em_tw.oi_thesis_weakened:
                                            _em_tw.oi_thesis_weakened = False
                                            _tw_ul = _track['underlying'].replace('NSE:', '')
                                            t._wlog(f"      ✅ OI_WATCHER {_tw_ul}: Thesis RECOVERED — str {_cur_str_tw:.2f} | Relaxing trailing stop")
                            
                            if _oiw_exit_reason:
                                # Confirmed thesis break — exit now
                                _oiw_entry_opt = _track.get('entry_opt_price', 0)
                                _oiw_pnl_est = (_opt_ltp - _oiw_entry_opt) if _opt_ltp > 0 and _oiw_entry_opt > 0 else 0
                                _oiw_ul_short = _track['underlying'].replace('NSE:', '')
                                _exits_to_process.append((
                                    _opt_sym, _track, _ul_ltp, _opt_ltp,
                                    0.0,  # rev_pct (not relevant — this is OI-driven)
                                    0.0,  # prem_rev_pct
                                    0.0,  # opt_gain_pct
                                    3,    # confirmations (high — this is thesis break)
                                    [_oiw_exit_reason],  # signal reasons
                                    0.0,  # eff_rev
                                    0.0,  # eff_prem_rev
                                    f'OI_WATCHER_{_oiw_exit_reason}',  # exit_reason
                                ))
                                t._wlog(f"\n   🔴 OI_WATCHER THESIS BROKEN: {_oiw_ul_short} ({_track.get('opt_type','?')})")
                                t._wlog(f"      Reason: {_oiw_exit_reason} | {_oiw_detail}")
                                t._wlog(f"      Held: {_oiw_held:.0f}s | Premium: ₹{_oiw_entry_opt:.2f}→₹{_opt_ltp:.2f} ({_oiw_pnl_est:+.2f})")
                                continue  # Skip standard WME checks — OI exit takes priority
            
            # Skip trades already managed by trailing stop (let winners run)
            if _skip_trailing:
                _em_state = t.exit_manager.get_trade_state(_opt_sym)
                if _em_state and _em_state.trailing_active:
                    continue
            
            # Check minimum favorable move from entry
            if _direction == 'BUY':
                _favorable = (_track['peak_price'] - _entry_ul) / _entry_ul if _entry_ul > 0 else 0
            else:
                _favorable = (_entry_ul - _track['trough_price']) / _entry_ul if _entry_ul > 0 else 0
            if _favorable < _min_move_pct:
                continue
            
            # ── TREND SHIELD Gate 1: Min premium gain floor ──
            # Don't WME-exit trades with tiny premium gains — nothing worth protecting.
            # Let exit_manager handle normally, giving the trade room to develop.
            _cur_prem_gain = 0.0
            if _entry_opt > 0 and _opt_ltp > 0:
                _cur_prem_gain = ((_opt_ltp - _entry_opt) / _entry_opt) * 100
            _peak_prem_gain = 0.0
            if _entry_opt > 0:
                _peak_opt_for_gate = _track.get('peak_opt_price', _entry_opt)
                _peak_prem_gain = ((_peak_opt_for_gate - _entry_opt) / _entry_opt) * 100
            if _peak_prem_gain < _min_prem_gain:
                # Tiny gain — clear any armed state and skip
                _track.pop('wme_armed_at', None)
                continue
            
            # ── Compute current reversal from peak/trough (underlying) ──
            if _direction == 'BUY':
                _rev = (_track['peak_price'] - _ul_ltp) / _track['peak_price'] if _track['peak_price'] > 0 else 0
            else:
                _rev = (_ul_ltp - _track['trough_price']) / _track['trough_price'] if _track['trough_price'] > 0 else 0
            
            # Record reversal for acceleration detection
            _track.setdefault('reversal_samples', []).append((_now, _rev))
            
            # ── Compute option premium reversal from peak ──
            _peak_opt = _track.get('peak_opt_price', _entry_opt)
            _opt_rev = 0.0
            if _opt_ltp > 0 and _peak_opt > 0:
                _opt_rev = (_peak_opt - _opt_ltp) / _peak_opt  # How much premium dropped from peak
            
            # ── Compute option premium gain from entry ──
            _opt_gain_pct = 0.0
            if _entry_opt > 0 and _peak_opt > 0:
                _opt_gain_pct = ((_peak_opt - _entry_opt) / _entry_opt) * 100  # % gain at peak
            
            # ════════════════════════════════════════════════
            # MULTI-SIGNAL REVERSAL CONFIRMATION
            # ════════════════════════════════════════════════
            _confirmations = 0
            _signal_reasons = []
            
            # ── Signal 1: VOLUME DRY-UP ──
            _vs = _track['vol_samples']
            if len(_vs) >= 4:
                _total_dt = _vs[-1][0] - _vs[0][0]
                if _total_dt > 10:
                    _avg_vol_rate = (_vs[-1][1] - _vs[0][1]) / _total_dt
                    _rv_idx = max(0, len(_vs) - 4)
                    _recent_dt = _vs[-1][0] - _vs[_rv_idx][0]
                    if _recent_dt > 0 and _avg_vol_rate > 0:
                        _recent_vol_rate = (_vs[-1][1] - _vs[_rv_idx][1]) / _recent_dt
                        if _recent_vol_rate < _avg_vol_rate * _vol_dryup_ratio:
                            _confirmations += 1
                            _signal_reasons.append('VOL_DRYUP')
            
            # ── Signal 2: MOMENTUM DECAY ──
            if len(_ps) >= 4 and abs(_track['peak_momentum']) > 1e-8:
                _p_now = _ps[-1]
                _p_prev_idx = max(0, len(_ps) - 4)
                _p_prev = _ps[_p_prev_idx]
                _pdt = _p_now[0] - _p_prev[0]
                if _pdt > 0 and _p_prev[1] > 0:
                    _cur_vel = (_p_now[1] - _p_prev[1]) / _p_prev[1] / _pdt
                    _peak_vel = _track['peak_momentum']
                    if _direction == 'BUY':
                        if _cur_vel <= 0:
                            _confirmations += 1
                            _signal_reasons.append('MOM_REVERSED')
                        elif _peak_vel > 0 and _cur_vel < _peak_vel * (1 - _mom_decay_thresh):
                            _confirmations += 1
                            _signal_reasons.append('MOM_DECAY')
                    elif _direction == 'SELL':
                        if _cur_vel >= 0:
                            _confirmations += 1
                            _signal_reasons.append('MOM_REVERSED')
                        elif _peak_vel < 0 and abs(_cur_vel) < abs(_peak_vel) * (1 - _mom_decay_thresh):
                            _confirmations += 1
                            _signal_reasons.append('MOM_DECAY')
            
            # ── Signal 3: PRESSURE SHIFT (buy/sell quantity imbalance) ──
            if _pressure_enabled:
                _buy_q = _ul_quote.get('buy_quantity', 0)
                _sell_q = _ul_quote.get('sell_quantity', 0)
                if _buy_q > 0 and _sell_q > 0:
                    if _direction == 'BUY' and _sell_q > _buy_q * _pressure_ratio:
                        _confirmations += 1
                        _signal_reasons.append('SELL_PRESSURE')
                    elif _direction == 'SELL' and _buy_q > _sell_q * _pressure_ratio:
                        _confirmations += 1
                        _signal_reasons.append('BUY_PRESSURE')
            
            # ── Signal 4: REVERSAL ACCELERATION (NEW) ──
            # If the reversal is getting faster over consecutive checks, it's accelerating
            if _accel_enabled:
                _rsamps = _track.get('reversal_samples', [])
                if len(_rsamps) >= 4:
                    # Compare reversal speed in last half vs first half of samples
                    _mid = len(_rsamps) // 2
                    _first_half = _rsamps[:_mid]
                    _second_half = _rsamps[_mid:]
                    if len(_first_half) >= 2 and len(_second_half) >= 2:
                        _fh_dt = _first_half[-1][0] - _first_half[0][0]
                        _sh_dt = _second_half[-1][0] - _second_half[0][0]
                        if _fh_dt > 0 and _sh_dt > 0:
                            _fh_speed = abs(_first_half[-1][1] - _first_half[0][1]) / _fh_dt
                            _sh_speed = abs(_second_half[-1][1] - _second_half[0][1]) / _sh_dt
                            if _fh_speed > 0 and _sh_speed > _fh_speed * _accel_threshold:
                                _confirmations += 1
                                _signal_reasons.append('REV_ACCEL')
            
            # ── Signal 5: OI REVERSAL (operators reversing position = move is done) ──
            # Fetch OI periodically (every 60s, not every 5s check) and detect when
            # operators flip: SHORT_COVERING after SELL trade, LONG_UNWINDING after BUY trade.
            # Counts as 2 confirmations — this is the STRONGEST exit signal.
            # Also caches OI strength + participant for Shield/Accelerator threshold adjustment.
            _oi_exit_cfg = _cfg.get('oi_reversal_exit_enabled', True)
            _oi_check_iv = _cfg.get('oi_check_interval_seconds', 60)
            _oi_min_hold = _cfg.get('oi_reversal_min_hold_seconds', 90)
            if _oi_exit_cfg and t._oi_analyzer and (_now - _track['entry_time']) >= _oi_min_hold:
                _last_oi_check = _track.get('_last_oi_check_ts', 0)
                if (_now - _last_oi_check) >= _oi_check_iv:
                    _track['_last_oi_check_ts'] = _now
                    try:
                        _oi_exit_data = t._oi_analyzer.analyze(_track['underlying'])
                        if _oi_exit_data:
                            _oi_exit_signal = _oi_signal_from_result(_oi_exit_data)
                            _oi_exit_dir = _oi_direction(_oi_exit_signal)
                            _track['_last_oi_signal'] = _oi_exit_signal
                            _track['_last_oi_dir'] = _oi_exit_dir
                            _track['_last_oi_strength'] = _oi_exit_data.get('nse_oi_buildup_strength', 0.0)
                            _track['_last_oi_participant'] = _oi_exit_data.get('oi_participant_id', 'UNKNOWN')
                    except Exception:
                        pass
                # Use cached OI signal for confirmation check
                _cached_oi_dir = _track.get('_last_oi_dir', '')
                _cached_oi_sig = _track.get('_last_oi_signal', '')
                if _cached_oi_dir and _cached_oi_dir != _direction:
                    # OI flipped AGAINST our trade direction — operators are reversing!
                    # This counts as 2 confirmations (strongest signal)
                    _confirmations += 2
                    _signal_reasons.append(f'OI_REVERSAL({_cached_oi_sig})')
            
            # ── Determine adaptive reversal threshold ──
            # Start with base thresholds
            if _confirmations >= 2:
                _effective_rev = _confirmed_rev_pct      # 0.25%
                _effective_prem_rev = _premium_rev_confirmed  # 5%
            elif _confirmations == 1:
                _effective_rev = _partial_rev_pct         # 0.35%
                _effective_prem_rev = (_premium_rev_pct + _premium_rev_confirmed) / 2  # 6.5%
            else:
                _effective_rev = _reversal_pct            # 0.50%
                _effective_prem_rev = _premium_rev_pct    # 8%
            
            # ── TREND SHIELD Gate 2: Fibonacci retracement floor ──
            # Scale reversal threshold with move size: big moves get proportionally more room.
            # A stock up +5% gets 1.91% pullback room (5% × 0.382) instead of fixed 0.5%.
            if _retrace_enabled and _favorable > 0:
                _fib_ul_floor = _favorable * _ul_retrace_factor     # e.g. 5% move × 0.382 = 1.91%
                _effective_rev = max(_effective_rev, _fib_ul_floor)
            if _retrace_enabled and _opt_gain_pct > 0:
                _fib_prem_floor = (_opt_gain_pct / 100) * _prem_retrace_factor  # e.g. 20% gain × 0.382 = 7.64%
                _effective_prem_rev = max(_effective_prem_rev, _fib_prem_floor)
            
            # ── Profit-tiered override: bigger profits = tighter stops ──
            for _tier in sorted(_profit_tiers, key=lambda x: -x.get('min_gain_pct', 0)):
                if _opt_gain_pct >= _tier.get('min_gain_pct', 999):
                    _effective_rev = min(_effective_rev, _tier.get('reversal_pct', 99) / 100)
                    _effective_prem_rev = min(_effective_prem_rev, _tier.get('premium_rev_pct', 99) / 100)
                    break
            
            # ── CE/PE asymmetric multiplier ──
            if _opt_type == 'CE':
                _effective_rev *= _ce_mult    # CE: tighter (0.85x)
                _effective_prem_rev *= _ce_mult
            elif _opt_type == 'PE':
                _effective_rev *= _pe_mult    # PE: tighter (0.90x)
                _effective_prem_rev *= _pe_mult
            
            # ════════════════════════════════════════════════════════════════
            # OI CONFIRMATION SHIELD & ACCELERATOR
            # Smart money awareness: let OI state adjust exit thresholds.
            #
            # OI CONFIRMS direction → SHIELD (widen thresholds, avoid premature exit)
            #   "Writers are still building — this pullback is just noise, hold."
            # OI OPPOSES direction → ACCELERATOR (tighten thresholds, exit faster)
            #   "Writers are unwinding — the move is done, get out."
            # OI NEUTRAL / weakened → slightly tighter (conviction fading)
            #   "Buildup dissolved — trend may be ending, less room."
            # ════════════════════════════════════════════════════════════════
            _oi_shield_mult = 1.0  # Default: no adjustment
            _oi_shield_tag = ''
            if _cfg.get('oi_shield_enabled', True):
                _cached_oi_dir_sh = _track.get('_last_oi_dir', '')
                _cached_oi_sig_sh = _track.get('_last_oi_signal', '')
                _cached_oi_str_sh = _track.get('_last_oi_strength', 0.0)
                _cached_oi_part_sh = _track.get('_last_oi_participant', '')
                _entry_oi_sh = _track.get('entry_oi', {})
                _entry_oi_str_sh = _entry_oi_sh.get('strength', 0.0)

                if _cached_oi_dir_sh:  # Only adjust if we have OI data
                    if _cached_oi_dir_sh == _direction:
                        # ── OI SHIELD: Confirms our direction → give trade more room ──
                        _oi_shield_mult = _cfg.get('oi_shield_confirm_multiplier', 1.40)
                        _oi_shield_tag = 'SHIELD'
                        # Extra bonus if writers are dominant (strongest hold signal)
                        if _cached_oi_part_sh == 'WRITER_DOMINANT':
                            _oi_shield_mult *= _cfg.get('oi_shield_writer_confirm_bonus', 1.15)
                            _oi_shield_tag = 'SHIELD_WRITER'
                    elif _cached_oi_dir_sh != _direction and _cached_oi_dir_sh != '':
                        # ── OI ACCELERATOR: Opposes our direction → tighten exit ──
                        _oi_shield_mult = _cfg.get('oi_shield_oppose_multiplier', 0.70)
                        _oi_shield_tag = 'ACCEL'
                    # Detect strength collapse (OI buildup fading even if direction same)
                    if _entry_oi_str_sh > 0 and _cached_oi_str_sh > 0:
                        _str_ratio = _cached_oi_str_sh / _entry_oi_str_sh
                        _collapse_thresh = _cfg.get('oi_shield_strength_collapse_ratio', 0.50)
                        if _str_ratio < _collapse_thresh and _oi_shield_mult >= 1.0:
                            # Strength collapsed but direction still same → override shield with mild tightening
                            _oi_shield_mult = _cfg.get('oi_shield_strength_collapse_mult', 0.80)
                            _oi_shield_tag = 'FADE'
                    # Detect neutral (OI signal dissolved entirely)
                    if _cached_oi_sig_sh in ('NEUTRAL', '') and not _oi_shield_tag.startswith('ACCEL'):
                        _oi_shield_mult = min(_oi_shield_mult, _cfg.get('oi_shield_neutral_multiplier', 0.90))
                        _oi_shield_tag = _oi_shield_tag or 'NEUTRAL'

                if _oi_shield_mult != 1.0:
                    _effective_rev *= _oi_shield_mult
                    _effective_prem_rev *= _oi_shield_mult
                    _track['_oi_shield_tag'] = _oi_shield_tag
                    _track['_oi_shield_mult'] = _oi_shield_mult
                else:
                    _track['_oi_shield_tag'] = ''
                    _track['_oi_shield_mult'] = 1.0
            
            # ── Check exit conditions ──
            _exit_reason = None
            _oi_reversal_active = any('OI_REVERSAL' in s for s in _signal_reasons)
            
            # Check 1: Underlying reversal from peak/trough
            if _rev >= _effective_rev:
                _exit_reason = 'UL_REVERSAL'
            
            # Check 2: Option premium reversal from peak (independent of underlying)
            # This catches IV crush: underlying holds but premium drops
            if not _exit_reason and _opt_rev >= _effective_prem_rev:
                _exit_reason = 'PREMIUM_REVERSAL'
            
            # Check 3: OI reversal alone can trigger exit (operators reversed = move is done)
            # Even if price hasn't reversed enough yet, OI flip is a leading indicator.
            if not _exit_reason and _oi_reversal_active and _rev > 0:
                _exit_reason = 'OI_REVERSAL'
            
            # ── TREND SHIELD Gate 3: Armed grace period (V-shape filter) ──
            # Instead of instant exit, enter ARMED state. Wait for grace period.
            # If price recovers → DISARM (it was just a healthy pullback).
            # If reversal deepens beyond threshold → exit immediately.
            # Exception: OI reversal with immediate_exit config skips grace entirely.
            _oi_immediate = _cfg.get('oi_reversal_immediate_exit', True)
            if _exit_reason and _oi_reversal_active and _oi_immediate:
                # OI reversal = operators reversed — skip grace, exit NOW
                _exit_reason = _exit_reason + '_OI'
                _exits_to_process.append((_opt_sym, _track, _ul_ltp, _opt_ltp, _rev * 100, _opt_rev * 100,
                                          _opt_gain_pct, _confirmations, _signal_reasons,
                                          _effective_rev * 100, _effective_prem_rev * 100, _exit_reason))
                continue
            if _exit_reason:
                _armed_at = _track.get('wme_armed_at', None)
                if _armed_at is None:
                    # First trigger → ARM the exit, don't fire yet
                    _track['wme_armed_at'] = _now
                    _track['wme_armed_reason'] = _exit_reason
                    _track['wme_armed_rev'] = _effective_rev
                    _track['wme_armed_prem_rev'] = _effective_prem_rev
                    _ul_short = _track['underlying'].replace('NSE:', '')
                    _armed_shield = f" OI:{_track.get('_oi_shield_tag','')}" if _track.get('_oi_shield_tag') else ''
                    t._wlog(f"   ⏳ WME ARMED: {_ul_short} ({_track.get('opt_type','?')}) — waiting {_grace_seconds}s grace for recovery{_armed_shield}")
                    t._wlog(f"      UL rev: {_rev*100:.2f}% (thresh {_effective_rev*100:.2f}%) | Prem rev: {_opt_rev*100:.1f}% (thresh {_effective_prem_rev*100:.1f}%)")
                    continue  # Don't exit yet — wait for next check
                else:
                    _elapsed_armed = _now - _armed_at
                    # Check if reversal deepened dangerously (1.5× threshold → exit NOW)
                    _armed_rev_thresh = _track.get('wme_armed_rev', _effective_rev)
                    _armed_prem_thresh = _track.get('wme_armed_prem_rev', _effective_prem_rev)
                    _deepened = (_rev >= _armed_rev_thresh * _deepen_mult or 
                                 _opt_rev >= _armed_prem_thresh * _deepen_mult)
                    if _deepened:
                        _exit_reason = _track.get('wme_armed_reason', _exit_reason) + '_DEEP'
                        # Fall through to exit
                    elif _elapsed_armed < _grace_seconds:
                        continue  # Still within grace period, keep waiting
                    # else: grace expired, reversal held → proceed with exit
                _exits_to_process.append((_opt_sym, _track, _ul_ltp, _opt_ltp, _rev * 100, _opt_rev * 100,
                                          _opt_gain_pct, _confirmations, _signal_reasons,
                                          _effective_rev * 100, _effective_prem_rev * 100, _exit_reason))
            else:
                # Exit conditions NOT met → DISARM if previously armed (price recovered!)
                if _track.get('wme_armed_at') is not None:
                    _ul_short = _track['underlying'].replace('NSE:', '')
                    t._wlog(f"   ✅ WME DISARMED: {_ul_short} ({_track.get('opt_type','?')}) — price recovered during grace! Continuing to hold.")
                    _track.pop('wme_armed_at', None)
                    _track.pop('wme_armed_reason', None)
                    _track.pop('wme_armed_rev', None)
                    _track.pop('wme_armed_prem_rev', None)
        
        # --- Process momentum exits ---
        for (_opt_sym, _track, _ul_ltp, _opt_ltp_cached, _rev_pct, _prem_rev_pct,
             _opt_gain_pct, _n_confirms, _sig_reasons, _eff_rev, _eff_prem_rev, _exit_reason) in _exits_to_process:
            _direction = _track['direction']
            _opt_type = _track.get('opt_type', '?')
            _ul = _track['underlying']
            _ul_short = _ul.replace('NSE:', '')
            
            # Find the active trade
            with t.tools._positions_lock:
                _trade = next((_pos for _pos in t.tools.paper_positions
                               if _pos['symbol'] == _opt_sym and _pos.get('status', 'OPEN') == 'OPEN'), None)
            
            if not _trade:
                t._watcher_momentum_tracker.pop(_opt_sym, None)
                continue
            
            # Get current option premium (fresh, not cached)
            try:
                if t.tools.ticker and t.tools.ticker.connected:
                    _opt_prices = t.tools.ticker.get_ltp_batch([_opt_sym])
                    _opt_ltp = _opt_prices.get(_opt_sym, 0)
                else:
                    _q = t.tools.kite.ltp([_opt_sym])
                    _opt_ltp = _q.get(_opt_sym, {}).get('last_price', 0)
            except Exception:
                continue
            
            if _opt_ltp <= 0:
                continue
            
            # Calculate P&L
            _entry_price = _trade['avg_price']
            _qty = _trade['quantity']
            _pnl = (_opt_ltp - _entry_price) * _qty
            _pnl -= calc_brokerage(_entry_price, _opt_ltp, _qty)
            
            # Only exit if option is in profit (don't cut losers — let exit manager handle)
            # Exception: OI_WATCHER thesis exits — thesis broken = exit regardless of P&L
            _is_oi_watcher_thesis_exit = _exit_reason.startswith('OI_WATCHER_')
            if _only_profit and _pnl <= 0 and not _is_oi_watcher_thesis_exit:
                continue
            
            _extreme_label = 'peak' if _direction == 'BUY' else 'trough'
            _extreme_val = _track['peak_price'] if _direction == 'BUY' else _track['trough_price']
            _trigger_label = 'spike peaked' if _direction == 'BUY' else 'crater bottomed'
            _signals_str = '+'.join(_sig_reasons) if _sig_reasons else 'PRICE_ONLY'
            _peak_opt = _track.get('peak_opt_price', _entry_price)
            
            # Differentiate logging for OI_WATCHER thesis exits  
            if _is_oi_watcher_thesis_exit:
                t._wlog(f"\n🔴 OI_WATCHER EXIT: {_ul_short} ({_opt_type})")
                t._wlog(f"   Reason: {_exit_reason} | Direction: {_direction}")
                t._wlog(f"   Premium: entry ₹{_entry_price:.2f} → now ₹{_opt_ltp:.2f} | P&L: ₹{_pnl:+,.0f}")
                _entry_oi = _track.get('entry_oi', {})
                t._wlog(f"   Entry OI: {_entry_oi.get('signal','')} str={_entry_oi.get('strength',0):.2f} part={_entry_oi.get('participant','?')}")
                t._wlog(f"   Current OI: {_track.get('_cur_oi_signal','')} str={_track.get('_cur_oi_strength',0):.2f} part={_track.get('_cur_oi_participant','?')}")
            else:
                _shield_info = ''
                _sh_tag = _track.get('_oi_shield_tag', '')
                _sh_mult = _track.get('_oi_shield_mult', 1.0)
                if _sh_tag:
                    _shield_info = f" | OI:{_sh_tag}(×{_sh_mult:.2f})"
                t._wlog(f"\n🌊 WATCHER MOMENTUM EXIT: {_ul_short} ({_opt_type})")
                t._wlog(f"   Reason: {_exit_reason} | Direction: {_direction} | Trigger: {_trigger_label}{_shield_info}")
                t._wlog(f"   Signals: {_signals_str} ({_n_confirms}/5 confirmed)")
                t._wlog(f"   UL: entry ₹{_track['entry_ul_price']:.2f} → {_extreme_label} ₹{_extreme_val:.2f} → now ₹{_ul_ltp:.2f} ({_rev_pct:.2f}% rev, thresh {_eff_rev:.2f}%)")
                t._wlog(f"   Premium: entry ₹{_entry_price:.2f} → peak ₹{_peak_opt:.2f} → now ₹{_opt_ltp:.2f} ({_prem_rev_pct:.1f}% drop, thresh {_eff_prem_rev:.1f}%)")
                t._wlog(f"   Peak gain: {_opt_gain_pct:.0f}% | P&L: ₹{_pnl:+,.0f}")
            
            # Execute exit
            _exit_label = _exit_reason if _is_oi_watcher_thesis_exit else 'WATCHER_MOMENTUM_EXIT'
            t.tools.update_trade_status(_opt_sym, _exit_label, _opt_ltp, _pnl)
            if _is_oi_watcher_thesis_exit:
                t._oi_watcher_thesis_exits += 1
            with t._pnl_lock:
                t.daily_pnl += _pnl
                t.capital += _pnl
            
            # Record with Risk Governor
            _remaining = [t for t in t.tools.paper_positions
                          if t.get('status') == 'OPEN' and t['symbol'] != _opt_sym]
            _unrealized = t.risk_governor._calc_unrealized_pnl(_remaining)
            t.risk_governor.record_trade_result(_opt_sym, _pnl, _pnl > 0, unrealized_pnl=_unrealized)
            t.risk_governor.update_capital(t.capital)
            
            # Remove from exit manager
            t.exit_manager.remove_trade(_opt_sym)
            
            # Remove from tracker
            t._watcher_momentum_tracker.pop(_opt_sym, None)
            
            # === COOLDOWN BYPASS: Allow watcher to re-enter this symbol ===
            if _bypass_cooldown:
                _nse_sym = f"NSE:{_ul_short}"
                t._watcher_fired_this_session.discard(_nse_sym)
                # Also clear the ticker-level cooldown for this symbol
                if t.tools.ticker and hasattr(t.tools.ticker, 'breakout_watcher'):
                    _bw = t.tools.ticker.breakout_watcher
                    if _bw and hasattr(_bw, '_cooldowns'):
                        _bw._cooldowns.pop(_nse_sym, None)
                t._wlog(f"   🔄 COOLDOWN BYPASSED: {_ul_short} — watcher can re-enter on next spike/crater")
            
            # Notify scorer
            try:
                from options_trader import get_intraday_scorer
                _scorer = get_intraday_scorer()
                if _pnl > 0:
                    _scorer.record_symbol_win(_opt_sym)
                else:
                    _scorer.record_symbol_loss(_opt_sym)
            except Exception:
                pass
