"""OI Watcher Engine — extracted from autonomous_trader.py"""
import threading
from config import CAPITAL_SWAP


# OI signals that indicate BEARISH direction (operators positioned for down move)
_OI_BEARISH_SIGNALS = {'SHORT_BUILDUP', 'LONG_UNWINDING'}
# OI signals that indicate BULLISH direction (operators positioned for up move)
_OI_BULLISH_SIGNALS = {'LONG_BUILDUP', 'SHORT_COVERING'}

def _oi_signal_from_result(oi_data: dict) -> str:
    """Extract best OI signal from OI analyzer result dict.
    Prefers nse_oi_buildup (DhanHQ/NSE) → falls back to flow_bias mapping."""
    if not oi_data:
        return ''
    sig = oi_data.get('nse_oi_buildup', '')
    if sig and sig != 'NEUTRAL':
        return sig
    fb = oi_data.get('flow_bias', 'NEUTRAL')
    if fb == 'BULLISH':
        return 'LONG_BUILDUP'
    if fb == 'BEARISH':
        return 'SHORT_BUILDUP'
    return sig or 'NEUTRAL'

def _oi_direction(oi_signal: str) -> str:
    """Get the authoritative direction from OI signal.
    Operators build OI positions first, then move the underlying.
    OI buildup IS the leading indicator — trust it over scorer."""
    if oi_signal in _OI_BEARISH_SIGNALS:
        return 'SELL'
    if oi_signal in _OI_BULLISH_SIGNALS:
        return 'BUY'
    return ''


class OIWatcherEngine:
    def __init__(self, trader):
        self.trader = trader

    def run_watcher_scan(self, layer1_oi):
        """OI_WATCHER — Pure OI-based trade, no model, no scoring.
        Picks the top-1 symbol by OI buildup strength from Layer 1 data
        and fires immediately before the expensive pipeline runs."""
        t = self.trader
        import time as _wt
        if not layer1_oi or t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
            return
        _oi_candidates = []
        for _oi_sym, _oi_res in layer1_oi.items():
            _oi_sig = _oi_signal_from_result(_oi_res)
            _oi_dir = _oi_direction(_oi_sig)
            _oi_str = _oi_res.get('nse_oi_buildup_strength', 0.0)
            # PCR surrogate: when raw OI strength=0 but PCR gave a directional
            # signal, use flow_confidence as strength (same pattern as Gate F2).
            if _oi_str < 0.01 and _oi_sig and _oi_sig not in ('NEUTRAL', ''):
                _fc_ow = _oi_res.get('flow_confidence', 0.0)
                if _fc_ow > 0.1:
                    _oi_str = _fc_ow
            if not _oi_dir:
                continue  # NEUTRAL — skip
            if _oi_str < t._oi_watcher_min_strength:
                continue  # Below base strength threshold
            # [FIX Mar 24 v2] SHORT_COVERING / LONG_UNWINDING need higher base (weaker signals)
            if _oi_sig in ('SHORT_COVERING', 'LONG_UNWINDING') and _oi_str < 0.40:
                continue
            if _oi_sym in t._oi_watcher_fired_this_session:
                continue  # Already fired OI_WATCHER on this symbol today
            if t.tools.is_symbol_in_active_trades(_oi_sym):
                continue  # Already holding this symbol
            _oi_candidates.append((_oi_sym, _oi_dir, _oi_sig, _oi_str, _oi_res))

        if _oi_candidates:
            # ── SMART QUALITY SCORING ──
            # Instead of just raw strength, compute a composite quality score
            # that rewards confluence: participant + cross-validation + price confirmation + OI trend.
            # No harsh gates — everything is a boost/discount to effective strength.
            _ticker = getattr(t.tools, 'ticker', None)
            for _oci in range(len(_oi_candidates)):
                _oc_sym, _oc_dir, _oc_sig, _oc_str, _oc_res = _oi_candidates[_oci]
                _oc_eff_str = _oc_str
                _oc_boosts = []
                _oc_confirms = 0  # Independent confirming factor count

                # (A) Participant quality: GRANULAR writer/buyer ratio
                # Raw writer_oi/buyer_oi numbers reveal conviction depth.
                # 90%+ writer = margin-locked, will defend level = near-guaranteed.
                # 50-65% writer = fragile, can flip = low conviction.
                # Buyer-dominated = hedging noise, NOT directional.
                _oc_part = _oc_res.get('oi_participant_id', 'UNKNOWN')
                _oc_pid_detail = _oc_res.get('oi_participant_detail', {})
                _oc_writer_ratio = None
                if _oc_pid_detail:
                    if _oc_dir == 'BUY':  # LONG_BUILDUP → PE writers are conviction
                        _oc_w_oi = _oc_pid_detail.get('pe_writer_oi', 0)
                        _oc_b_oi = _oc_pid_detail.get('pe_buyer_oi', 0)
                    else:  # SHORT_BUILDUP → CE writers are conviction
                        _oc_w_oi = _oc_pid_detail.get('ce_writer_oi', 0)
                        _oc_b_oi = _oc_pid_detail.get('ce_buyer_oi', 0)
                    _oc_total_classified = _oc_w_oi + _oc_b_oi
                    if _oc_total_classified > 0:
                        _oc_writer_ratio = _oc_w_oi / _oc_total_classified
                if _oc_writer_ratio is not None:
                    if _oc_writer_ratio >= 0.85:
                        _oc_eff_str *= 1.25  # 85%+ writer = rock solid institutional conviction
                        _oc_boosts.append(f'W✓✓{_oc_writer_ratio:.0%}')
                        _oc_confirms += 1
                    elif _oc_writer_ratio >= 0.65:
                        _oc_eff_str *= 1.12  # Strong writer majority
                        _oc_boosts.append(f'W✓{_oc_writer_ratio:.0%}')
                        _oc_confirms += 1
                    elif _oc_writer_ratio >= 0.50:
                        _oc_eff_str *= 1.0   # Neutral — no boost
                        _oc_boosts.append(f'W~{_oc_writer_ratio:.0%}')
                    else:
                        _oc_eff_str *= 0.75  # Buyer-dominated = hedging, NOT directional
                        _oc_boosts.append(f'B✗{_oc_writer_ratio:.0%}')
                else:
                    # Fallback to label when detail not available
                    if _oc_part == 'WRITER_DOMINANT':
                        _oc_eff_str *= 1.15
                        _oc_boosts.append('W+')
                    elif _oc_part == 'BUYER_DOMINANT':
                        _oc_eff_str *= 0.85
                        _oc_boosts.append('B-')

                # (B) Cross-validation: Kite PCR + DhanHQ both agree = high conviction
                if _oc_res.get('oi_cross_validated'):
                    _oc_eff_str *= 1.10  # Both sources agree → 10% boost
                    _oc_boosts.append('XV✓')
                    _oc_confirms += 1

                # (C) Price confirmation: check if stock move aligns with OI direction
                # [MANDATORY FACTOR] — price must move ≥0.20% in OI direction
                # OI says BUY + stock is actually rising = confluence
                # OI says BUY + stock is falling = divergence (could still work, but less confident)
                _oc_C_confirmed = False
                _oc_C_price_chg = 0.0
                _oc_C_min_delta = getattr(t, '_oi_watcher_min_price_delta', 0.20)
                if _ticker:
                    try:
                        _oc_clean = _oc_sym.replace('NSE:', '')
                        _oc_tok = None
                        with _ticker._lock:
                            for _tk, _tsym in _ticker._token_to_symbol.items():
                                if _tsym == _oc_sym:
                                    _oc_tok = _tk
                                    break
                            if _oc_tok:
                                _oc_q = _ticker._quote_cache.get(_oc_tok, {})
                                _oc_ltp = _oc_q.get('last_price', 0)
                                _oc_close = (_oc_q.get('ohlc', {}) or {}).get('close', 0)
                                if _oc_ltp > 0 and _oc_close > 0:
                                    _oc_chg = ((_oc_ltp - _oc_close) / _oc_close) * 100
                                    _oc_C_price_chg = _oc_chg
                                    _price_agrees = (
                                        (_oc_dir == 'BUY' and _oc_chg >= _oc_C_min_delta) or
                                        (_oc_dir == 'SELL' and _oc_chg <= -_oc_C_min_delta)
                                    )
                                    _price_diverges = (
                                        (_oc_dir == 'BUY' and _oc_chg < -0.5) or
                                        (_oc_dir == 'SELL' and _oc_chg > 0.5)
                                    )
                                    if _price_agrees:
                                        _oc_C_confirmed = True
                                        _oc_eff_str *= 1.08  # Price confirms OI → 8% boost
                                        _oc_boosts.append(f'P✓{_oc_chg:+.1f}%')
                                        _oc_confirms += 1
                                    elif _price_diverges:
                                        _oc_eff_str *= 0.90  # Price diverges → 10% discount
                                        _oc_boosts.append(f'P✗{_oc_chg:+.1f}%')
                    except Exception as e:
                        t._wlog(f"⚠️ FALLBACK [oi_watcher/price_check]: {e}")

                # (D) OI trend: check if strength is BUILDING (from aggr history)
                # If we've seen this symbol before with lower strength, it's getting stronger = good
                _oc_hist = t._oi_aggr_strength_history.get(_oc_sym, [])
                if len(_oc_hist) >= 2:
                    _oc_prev_str = _oc_hist[-1][1]  # Most recent prior reading
                    if _oc_eff_str > _oc_prev_str * 1.05:  # 5% stronger than last
                        _oc_eff_str *= 1.05  # Building → 5% boost
                        _oc_boosts.append('OI↑')
                        _oc_confirms += 1
                    elif _oc_eff_str < _oc_prev_str * 0.80:  # 20% weaker than last
                        _oc_eff_str *= 0.90  # Fading → 10% discount
                        _oc_boosts.append('OI↓')

                # (E) Sector alignment: if sector index is moving same direction = strong confluence
                _oc_s2s = getattr(t, '_stock_to_sector', {})
                _oc_sec_chgs = getattr(t, '_sector_index_changes_cache', {})
                _oc_sec_info = _oc_s2s.get(_oc_sym.replace('NSE:', ''))
                if _oc_sec_info and _oc_sec_chgs:
                    _oc_sec_name, _oc_sec_idx = _oc_sec_info
                    _oc_sec_chg = _oc_sec_chgs.get(_oc_sec_idx, 0)
                    _oc_sec_agrees = (
                        (_oc_dir == 'BUY' and _oc_sec_chg > 0.3) or
                        (_oc_dir == 'SELL' and _oc_sec_chg < -0.3)
                    )
                    _oc_sec_oppose = (
                        (_oc_dir == 'BUY' and _oc_sec_chg < -0.5) or
                        (_oc_dir == 'SELL' and _oc_sec_chg > 0.5)
                    )
                    if _oc_sec_agrees:
                        _oc_eff_str *= 1.08  # Sector confirms → 8% boost
                        _oc_boosts.append('SEC✓')
                        _oc_confirms += 1
                    elif _oc_sec_oppose:
                        _oc_eff_str *= 0.88  # Swimming against sector → 12% discount
                        _oc_boosts.append('SEC✗')

                # (F) Futures OI buildup cross-check: ML feature #1 (46.5% importance)
                # If futures show LONG_BUILDUP and OI says BUY = triple confluence
                # [MANDATORY FACTOR] — tracked for conviction gate
                _oc_F_confirmed = False
                _oc_ml = getattr(t, '_cycle_ml_results', {})
                _oc_ml_data = _oc_ml.get(_oc_sym, {})
                _oc_fut_buildup = _oc_ml_data.get('fut_oi_buildup', 0) if isinstance(_oc_ml_data, dict) else 0
                if _oc_fut_buildup:
                    _oc_fut_agrees = (
                        (_oc_dir == 'BUY' and _oc_fut_buildup > 0) or
                        (_oc_dir == 'SELL' and _oc_fut_buildup < 0)
                    )
                    _oc_fut_strong = abs(_oc_fut_buildup) >= 0.75  # LB/SB not SC/LU
                    if _oc_fut_agrees and _oc_fut_strong:
                        _oc_eff_str *= 1.12  # Futures + Options agree strongly → 12% boost
                        _oc_boosts.append('FUT✓✓')
                        _oc_confirms += 1
                        _oc_F_confirmed = True
                    elif _oc_fut_agrees:
                        _oc_eff_str *= 1.05  # Mild agreement → 5% boost
                        _oc_boosts.append('FUT✓')
                        _oc_confirms += 1
                        _oc_F_confirmed = True
                    elif not _oc_fut_agrees and _oc_fut_strong:
                        _oc_eff_str *= 0.85  # Futures strongly disagree → 15% discount
                        _oc_boosts.append('FUT✗✗')

                # (G) Live Futures Basis: premium/discount from ticker cache (0 API calls)
                # Futures at premium + BUY = smart money paying up (urgency) → boost
                # Futures at discount + BUY = no urgency → slight discount
                if _ticker:
                    try:
                        _oc_fut_data = _ticker.get_futures_oi(_oc_sym)
                        if _oc_fut_data and _oc_fut_data.get('ltp', 0) > 0:
                            _oc_fut_ltp = _oc_fut_data['ltp']
                            # Get equity spot from ticker cache
                            _oc_eq_ltp = 0
                            with _ticker._lock:
                                for _tk2, _tsym2 in _ticker._token_to_symbol.items():
                                    if _tsym2 == _oc_sym:
                                        _oc_eq_ltp = _ticker._quote_cache.get(_tk2, {}).get('last_price', 0)
                                        break
                            if _oc_eq_ltp > 0:
                                _oc_basis_pct = ((_oc_fut_ltp - _oc_eq_ltp) / _oc_eq_ltp) * 100
                                _oc_basis_agrees = (
                                    (_oc_dir == 'BUY' and _oc_basis_pct > 0.05) or
                                    (_oc_dir == 'SELL' and _oc_basis_pct < -0.05)
                                )
                                _oc_basis_disagrees = (
                                    (_oc_dir == 'BUY' and _oc_basis_pct < -0.10) or
                                    (_oc_dir == 'SELL' and _oc_basis_pct > 0.10)
                                )
                                if _oc_basis_agrees:
                                    _oc_eff_str *= 1.10  # Smart money paying premium in your direction
                                    _oc_boosts.append(f'BASIS✓{_oc_basis_pct:+.2f}%')
                                    _oc_confirms += 1
                                elif _oc_basis_disagrees:
                                    _oc_eff_str *= 0.90  # Futures pricing against you
                                    _oc_boosts.append(f'BASIS✗{_oc_basis_pct:+.2f}%')
                    except Exception as e:
                        t._wlog(f"⚠️ FALLBACK [oi_watcher/basis_check]: {e}")

                # (H) OI Concentration vs Spot: WHERE is buildup happening?
                # Buildup at/near ATM = institutional conviction (skin in the game)
                # Buildup far OTM = hedging/premium collection, NOT directional conviction
                # [MANDATORY FACTOR] — tracked for conviction gate
                _oc_H_confirmed = False
                _oc_spot = _oc_res.get('spot_price', 0) or _oc_res.get('dhan_spot_price', 0)
                if _oc_spot > 0:
                    # For BUY direction, check put OI buildup (support building)
                    # For SELL direction, check call OI buildup (resistance building)
                    _oc_relevant_strikes = (
                        _oc_res.get('nse_top_put_oi_change', []) if _oc_dir == 'BUY'
                        else _oc_res.get('nse_top_call_oi_change', [])
                    )
                    if _oc_relevant_strikes and len(_oc_relevant_strikes) > 0:
                        # Each entry is (strike, oi_change) tuple
                        _oc_top_strike = _oc_relevant_strikes[0][0] if isinstance(_oc_relevant_strikes[0], (list, tuple)) else 0
                        if _oc_top_strike > 0:
                            _oc_strike_dist = abs(_oc_top_strike - _oc_spot) / _oc_spot * 100
                            if _oc_strike_dist <= 2.0:
                                _oc_eff_str *= 1.10  # Near-money buildup = institutional conviction
                                _oc_boosts.append(f'ATM✓{_oc_strike_dist:.1f}%')
                                _oc_confirms += 1
                                _oc_H_confirmed = True
                            elif _oc_strike_dist >= 5.0:
                                _oc_eff_str *= 0.92  # Far OTM = hedging, not conviction
                                _oc_boosts.append(f'OTM✗{_oc_strike_dist:.1f}%')

                # (I) PCR Shift Rate: rate of PCR change (already computed, not used in scoring)
                # Fast-rising PCR = aggressive put writing = building support NOW
                # Fast-falling PCR = aggressive call writing = building resistance NOW
                _oc_pcr_rate = _oc_res.get('pcr_shift_rate', 0)
                if abs(_oc_pcr_rate) > 0.005:  # Meaningful rate of change
                    _oc_pcr_rate_confirms = (
                        (_oc_dir == 'BUY' and _oc_pcr_rate > 0.01) or   # Rising PCR = bullish (put support)
                        (_oc_dir == 'SELL' and _oc_pcr_rate < -0.01)     # Falling PCR = bearish
                    )
                    _oc_pcr_rate_opposes = (
                        (_oc_dir == 'BUY' and _oc_pcr_rate < -0.01) or
                        (_oc_dir == 'SELL' and _oc_pcr_rate > 0.01)
                    )
                    if _oc_pcr_rate_confirms:
                        _oc_eff_str *= 1.08  # PCR shifting your way NOW
                        _oc_boosts.append(f'PCR↗{_oc_pcr_rate:+.3f}')
                        _oc_confirms += 1
                    elif _oc_pcr_rate_opposes:
                        _oc_eff_str *= 0.92  # PCR shifting against you
                        _oc_boosts.append(f'PCR↘{_oc_pcr_rate:+.3f}')

                # (J) Volume PCR: today's trading intent vs stale OI
                # Volume PCR captures what traders are DOING today. OI PCR includes
                # stale overnight positions. When volume PCR strongly confirms = fresh conviction.
                _oc_vol_pcr = _oc_res.get('nse_pcr_volume', 0)
                _oc_oi_pcr = _oc_res.get('pcr_oi', 1.0)
                if _oc_vol_pcr and _oc_vol_pcr > 0:
                    _oc_vol_confirms = (
                        (_oc_dir == 'BUY' and _oc_vol_pcr > 1.3) or    # Heavy put volume = support
                        (_oc_dir == 'SELL' and _oc_vol_pcr < 0.7)      # Heavy call volume = pressure
                    )
                    _oc_vol_opposes = (
                        (_oc_dir == 'BUY' and _oc_vol_pcr < 0.6) or
                        (_oc_dir == 'SELL' and _oc_vol_pcr > 1.5)
                    )
                    if _oc_vol_confirms:
                        _oc_eff_str *= 1.08  # Today's volume confirms direction
                        _oc_boosts.append(f'VP✓{_oc_vol_pcr:.2f}')
                        _oc_confirms += 1
                    elif _oc_vol_opposes:
                        _oc_eff_str *= 0.90  # Today's volume against direction
                        _oc_boosts.append(f'VP✗{_oc_vol_pcr:.2f}')

                # (K) Futures Conviction Boost: OI Day-High + Order Book Imbalance
                # Kite WebSocket streams futures OI + buy/sell qty in real-time (0 API calls).
                # BOOST-ONLY: helps the BEST signals rise to the top for entry.
                # buy_quantity = total pending BUY orders in futures order book (bullish demand)
                # sell_quantity = total pending SELL orders in futures order book (bearish supply)
                # [MANDATORY FACTOR] — tracked for conviction gate
                _oc_K_confirmed = False
                if _ticker:
                    try:
                        _oc_fk = _ticker.get_futures_oi(_oc_sym)
                        if _oc_fk and _oc_fk.get('oi', 0) > 0:
                            _oc_fk_oi = _oc_fk['oi']
                            _oc_fk_high = _oc_fk.get('oi_day_high', 0)
                            _oc_fk_low = _oc_fk.get('oi_day_low', 0)

                            # K1: OI at Day-High = institutions actively adding RIGHT NOW
                            if _oc_fk_high > _oc_fk_low > 0:
                                _oc_fk_range = _oc_fk_high - _oc_fk_low
                                _oc_fk_pos = (_oc_fk_oi - _oc_fk_low) / _oc_fk_range  # 0=low, 1=high
                                if _oc_fk_pos >= 0.85:  # OI at/near day high = fresh positions
                                    _oc_eff_str *= 1.18
                                    _oc_boosts.append(f'FOIDH✓{_oc_fk_pos:.0%}')
                                    _oc_confirms += 1
                                    _oc_K_confirmed = True
                                elif _oc_fk_pos >= 0.65:  # OI trending up = steady buildup
                                    _oc_eff_str *= 1.08
                                    _oc_boosts.append(f'FOIDH~{_oc_fk_pos:.0%}')

                            # K2: Futures Order Book Imbalance — who's lining up?
                            # buy_qty > sell_qty = bullish demand, sell_qty > buy_qty = bearish pressure
                            _oc_fk_buy = _oc_fk.get('buy_quantity', 0)
                            _oc_fk_sell = _oc_fk.get('sell_quantity', 0)
                            if _oc_fk_buy > 0 and _oc_fk_sell > 0:
                                _oc_fk_imb = _oc_fk_buy / (_oc_fk_buy + _oc_fk_sell)  # 0.5 = balanced
                                _oc_fk_imb_confirms = (
                                    (_oc_dir == 'BUY' and _oc_fk_imb > 0.55) or
                                    (_oc_dir == 'SELL' and _oc_fk_imb < 0.45)
                                )
                                if _oc_fk_imb_confirms:
                                    _oc_boost_mult = 1.08 + min(0.10, abs(_oc_fk_imb - 0.50) * 0.5)  # Scale: 55%→1.105, 65%→1.155
                                    _oc_eff_str *= _oc_boost_mult
                                    _oc_boosts.append(f'FOBI✓{_oc_fk_imb:.0%}')
                                    _oc_confirms += 1
                                    _oc_K_confirmed = True
                    except Exception as e:
                        t._wlog(f"⚠️ FALLBACK [oi_watcher/fobi_check]: {e}")

                # (L) OI Velocity — is strength ACCELERATING vs recent readings?
                # Real institutional flow shows rising OI over multiple scan cycles.
                # Noise OI is flat or random. Check if current > 1.15x avg of last 3.
                _oc_vel_hist = t._oi_aggr_strength_history.get(_oc_sym, [])
                if len(_oc_vel_hist) >= 2:
                    _oc_vel_avg = sum(h[1] for h in _oc_vel_hist[-3:]) / min(3, len(_oc_vel_hist))
                    if _oc_vel_avg > 0 and _oc_str > _oc_vel_avg * 1.15:
                        _oc_eff_str *= 1.10  # Accelerating OI = fresh institutional entry
                        _oc_boosts.append(f'VEL✓{_oc_str/_oc_vel_avg:.2f}x')
                        _oc_confirms += 1

                # (M) Volume Surge Alignment — is tick volume surging for this stock?
                # [FIX Mar 23 v2] OI watcher must see volume confirming the OI signal.
                # If volume is surging AND price move direction matches OI direction → strong.
                # If volume is flat/absent → OI buildup may be stale/noise.
                _oc_vol_surge_aligned = False
                _oc_vol_surge_ratio = 0.0
                if _ticker:
                    try:
                        _oc_vdh_store = getattr(_ticker, '_vol_delta_history', None)
                        _oc_vdh = _oc_vdh_store.get(_oc_sym) if _oc_vdh_store is not None else None
                        if _oc_vdh and len(_oc_vdh) >= 5:
                            _oc_vavg = sum(_oc_vdh) / len(_oc_vdh)
                            _oc_vlast = list(_oc_vdh)[-1] if _oc_vdh else 0
                            if _oc_vavg > 0 and _oc_vlast > 0:
                                _oc_vol_surge_ratio = _oc_vlast / _oc_vavg
                                # Check recent 3 ticks for sustained elevation
                                _oc_v_recent = list(_oc_vdh)[-3:]
                                _oc_v_elevated = sum(1 for _vd in _oc_v_recent if _vd >= _oc_vavg * 1.4)
                                if _oc_vol_surge_ratio >= 1.79 and _oc_v_elevated >= 2:
                                    # Volume IS surging — check price alignment
                                    with _ticker._lock:
                                        _oc_vtok = None
                                        for _tk_v, _tsym_v in _ticker._token_to_symbol.items():
                                            if _tsym_v == _oc_sym:
                                                _oc_vtok = _tk_v
                                                break
                                        if _oc_vtok:
                                            _oc_vq = _ticker._quote_cache.get(_oc_vtok, {})
                                            _oc_vltp = _oc_vq.get('last_price', 0)
                                            _oc_vclose = (_oc_vq.get('ohlc', {}) or {}).get('close', 0)
                                            if _oc_vltp > 0 and _oc_vclose > 0:
                                                _oc_vchg = ((_oc_vltp - _oc_vclose) / _oc_vclose) * 100
                                                # Volume surging + price moving in OI direction = aligned
                                                if (_oc_dir == 'BUY' and _oc_vchg > 0) or \
                                                   (_oc_dir == 'SELL' and _oc_vchg < 0):
                                                    _oc_vol_surge_aligned = True
                                                    _oc_eff_str *= 1.15  # Vol surge confirms OI
                                                    _oc_boosts.append(f'VSRG✓{_oc_vol_surge_ratio:.1f}x')
                                                    _oc_confirms += 1
                                                else:
                                                    # Volume surging but price opposite to OI = divergence
                                                    _oc_eff_str *= 0.80
                                                    _oc_boosts.append(f'VSRG✗{_oc_vol_surge_ratio:.1f}x')
                                elif _oc_vol_surge_ratio >= 1.46 and _oc_v_elevated >= 1:
                                    # [FIX Mar 24 v3] Intermediate volume — require price alignment too
                                    with _ticker._lock:
                                        _oc_vtok_int = None
                                        for _tk_vi, _tsym_vi in _ticker._token_to_symbol.items():
                                            if _tsym_vi == _oc_sym:
                                                _oc_vtok_int = _tk_vi
                                                break
                                        if _oc_vtok_int:
                                            _oc_vq_int = _ticker._quote_cache.get(_oc_vtok_int, {})
                                            _oc_vltp_int = _oc_vq_int.get('last_price', 0)
                                            _oc_vclose_int = (_oc_vq_int.get('ohlc', {}) or {}).get('close', 0)
                                            if _oc_vltp_int > 0 and _oc_vclose_int > 0:
                                                _oc_vchg_int = ((_oc_vltp_int - _oc_vclose_int) / _oc_vclose_int) * 100
                                                if (_oc_dir == 'BUY' and _oc_vchg_int > 0) or \
                                                   (_oc_dir == 'SELL' and _oc_vchg_int < 0):
                                                    _oc_vol_surge_aligned = True
                                                    _oc_boosts.append(f'VOL~✓{_oc_vol_surge_ratio:.1f}x')
                                                else:
                                                    _oc_boosts.append(f'VOL~✗{_oc_vol_surge_ratio:.1f}x')
                                            else:
                                                _oc_boosts.append(f'VOL~?{_oc_vol_surge_ratio:.1f}x')
                                        else:
                                            _oc_boosts.append(f'VOL~?{_oc_vol_surge_ratio:.1f}x')
                                else:
                                    # Volume is flat — OI without volume = stale/noise
                                    _oc_boosts.append(f'VOL↓{_oc_vol_surge_ratio:.1f}x')
                        else:
                            # [FIX Mar 24 v3] Not enough volume history — fail-closed
                            t._wlog(f"⚠️ VOL_SURGE BLOCKED [oi_watcher]: {_oc_sym.replace('NSE:', '')} "
                                       f"— insufficient tick history ({len(_oc_vdh) if _oc_vdh else 0}<5), M=False")
                    except Exception as _ve:
                        # [FIX Mar 24 v3] Exception — fail-closed
                        t._wlog(f"⛔ VOL_SURGE HALTED [oi_watcher]: {_oc_sym.replace('NSE:', '')} — {_ve}")
                else:
                    # [FIX Mar 24 v3] No ticker — fail-closed, halt trades
                    t._wlog(f"⛔ VOL_SURGE HALTED [oi_watcher]: ticker DOWN — M=False for all, no trades without volume data")

                # Store mandatory factor results for conviction gate
                _oc_res['_vol_surge_aligned'] = _oc_vol_surge_aligned
                _oc_res['_vol_surge_ratio'] = _oc_vol_surge_ratio
                _oc_res['_mf_C'] = _oc_C_confirmed
                _oc_res['_mf_C_price_chg'] = _oc_C_price_chg
                _oc_res['_mf_F'] = _oc_F_confirmed
                _oc_res['_mf_H'] = _oc_H_confirmed
                _oc_res['_mf_K'] = _oc_K_confirmed
                _oc_res['_mf_M'] = _oc_vol_surge_aligned

                # [FIX Mar 24 v2] Cap multiplicative inflation — multiplier product capped at 2.0x base
                _oc_max_boosted = _oc_str * 2.0  # base strength * 2.0 = max allowed
                _oc_eff_str = min(1.0, min(_oc_max_boosted, _oc_eff_str))
                # [FIX Mar 24 v2] SHORT_COVERING / LONG_UNWINDING need ≥0.65 effective to fire
                if _oc_sig in ('SHORT_COVERING', 'LONG_UNWINDING') and _oc_eff_str < 0.65:
                    _oc_eff_str = 0.0  # Will be filtered by min_strength check downstream
                _oi_candidates[_oci] = (_oc_sym, _oc_dir, _oc_sig, _oc_eff_str, _oc_res)
                # Store boost tags + confirm count for logging
                _oc_res['_quality_boosts'] = ' '.join(_oc_boosts) if _oc_boosts else ''
                _oc_res['_confirm_count'] = _oc_confirms
            # Sort by (confirm_count DESC, effective_strength DESC) — conviction first, strength second
            _oi_candidates.sort(key=lambda x: (x[4].get('_confirm_count', 0), x[3]), reverse=True)
            _oi_max_fire = 1  # Fire ONLY the single best-convicted candidate per cycle
            _oi_placed_count = 0
            _oi_top_confirms = _oi_candidates[0][4].get('_confirm_count', 0) if _oi_candidates else 0
            t._wlog(f"🔬 OI_WATCHER: {len(_oi_candidates)} candidates, "
                       f"top conviction={_oi_top_confirms}/{t._oi_min_confirmations} factors")
            for _oi_rank, _oi_top in enumerate(_oi_candidates[:_oi_max_fire]):
                if t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
                    break
                _oi_sym, _oi_dir, _oi_sig, _oi_str, _oi_res = _oi_top
                _oi_pcr = _oi_res.get('pcr_oi', 1.0)
                _oi_bias = _oi_res.get('flow_bias', 'NEUTRAL')

                # ── OI HEATMAP STRIKE PICKER ──
                _oi_strike_sel = 'ATM'
                _oi_hm_tag = ''
                try:
                    from dhan_oi_fetcher import DhanOIFetcher
                    _hm_strikes = _oi_res.get('dhan_strikes', [])
                    _hm_spot = _oi_res.get('dhan_spot_price', 0)
                    if _hm_strikes and _hm_spot > 0:
                        _hm = DhanOIFetcher.find_optimal_strike(_oi_dir, _hm_strikes, _hm_spot)
                        if _hm.get('score', 0) > 0:
                            _oi_strike_sel = _hm['selection']
                            _oi_hm_tag = (f" | HEATMAP: {_hm['selection']}@{_hm['strike']:.0f} "
                                          f"score={_hm['score']:.0f} ({_hm['reason']})")
                except Exception as e:
                    t._wlog(f"⚠️ FALLBACK [oi_watcher/heatmap_strike]: {e}")

                _oi_part_id = _oi_res.get('oi_participant_id', 'UNKNOWN')

                _oi_quality_tags = _oi_res.get('_quality_boosts', '')
                _oi_confirm_ct = _oi_res.get('_confirm_count', 0)
                t._wlog(f"🔬 OI_WATCHER: #{_oi_rank+1} pick {_oi_sym.replace('NSE:', '')} "
                           f"signal={_oi_sig} strength={_oi_str:.3f} dir={_oi_dir} "
                           f"bias={_oi_bias} PCR={_oi_pcr:.2f} part={_oi_part_id} "
                           f"conviction={_oi_confirm_ct}/{t._oi_min_confirmations} "
                           f"factors=[{_oi_quality_tags}] "
                           f"strike={_oi_strike_sel}{_oi_hm_tag}")

                _oi_ml_data = {
                    'oi_signal': _oi_sig,
                    'oi_strength': _oi_str,
                    'oi_pcr': _oi_pcr,
                    'oi_bias': _oi_bias,
                    'oi_participant_id': _oi_part_id,
                    'trade_type': 'OI_WATCHER',
                    'oi_heatmap_strike': _oi_strike_sel,
                }

                # ── OI CONVICTION GATE (Instant — No Time Delay) ──
                # Instead of waiting 60s (by which time the move is over or you're chasing),
                # use CONFLUENCE COUNT: how many independent market microstructure factors
                # confirm the signal RIGHT NOW. ≥ N factors = enter immediately.
                _oi_confirms = _oi_res.get('_confirm_count', 0)
                if _oi_confirms < t._oi_min_confirmations:
                    t._wlog(f"  ⛔ OI_WATCHER LOW CONVICTION: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"only {_oi_confirms}/{t._oi_min_confirmations} factors confirm "
                               f"[{_oi_res.get('_quality_boosts', '')}] — need more confluence")
                    continue

                # ── MANDATORY FACTOR GATE ──
                # C (Price confirmation) is ABSOLUTELY mandatory — price must move ≥0.20% in OI direction.
                # M (Vol surge) is ABSOLUTELY mandatory — no trade without volume confirmation.
                # Of the remaining 3 (F, H, K), require at least 2 to confirm.
                _oi_mf_C = _oi_res.get('_mf_C', False)
                _oi_mf_C_chg = _oi_res.get('_mf_C_price_chg', 0.0)
                _oi_mf_F = _oi_res.get('_mf_F', False)
                _oi_mf_H = _oi_res.get('_mf_H', False)
                _oi_mf_K = _oi_res.get('_mf_K', False)
                _oi_mf_M = _oi_res.get('_mf_M', False)
                _oi_mf_count = sum([_oi_mf_C, _oi_mf_F, _oi_mf_H, _oi_mf_K, _oi_mf_M])
                _oi_mf_labels = []
                if _oi_mf_C: _oi_mf_labels.append('C')
                if _oi_mf_F: _oi_mf_labels.append('F')
                if _oi_mf_H: _oi_mf_labels.append('H')
                if _oi_mf_K: _oi_mf_labels.append('K')
                if _oi_mf_M: _oi_mf_labels.append('M')
                _oi_mf_fhk_count = sum([_oi_mf_F, _oi_mf_H, _oi_mf_K])
                # Gate 0: Price confirmation (C) is absolutely mandatory
                _oi_C_min = getattr(t, '_oi_watcher_min_price_delta', 0.20)
                if not _oi_mf_C:
                    t._wlog(f"  ⛔ OI_WATCHER PRICE CONFIRM MANDATORY: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"price {_oi_mf_C_chg:+.2f}% not confirming (need ≥{_oi_C_min}% in {_oi_dir} dir) "
                               f"— C is mandatory, no trade without price confirmation "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
                               f"[{_oi_res.get('_quality_boosts', '')}]")
                    continue
                # Gate 1: Vol surge (M) is absolutely mandatory
                if not _oi_mf_M:
                    t._wlog(f"  ⛔ OI_WATCHER VOL SURGE MANDATORY: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"volume not surging — M is mandatory, no trade without vol confirmation "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
                               f"[{_oi_res.get('_quality_boosts', '')}]")
                    continue
                # Gate 2: Of F, H, K — require at least 2 of 3
                if _oi_mf_fhk_count < 2:
                    _oi_mf_missing = []
                    if not _oi_mf_F: _oi_mf_missing.append('F:FutOI')
                    if not _oi_mf_H: _oi_mf_missing.append('H:OIconc')
                    if not _oi_mf_K: _oi_mf_missing.append('K:FutBook')
                    t._wlog(f"  ⛔ OI_WATCHER MANDATORY FACTORS: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"only {_oi_mf_fhk_count}/3 of F,H,K confirmed (need ≥2) "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
                               f"[missing={','.join(_oi_mf_missing)}] "
                               f"[{_oi_res.get('_quality_boosts', '')}]")
                    continue

                # [FIX Mar 23 v3] Vol surge gate REMOVED — vol surge is a technical
                # confirmation factor (Factor M boost/discount), NOT a hard gate.
                # Trade already passes all OI watcher quality factors.

                # ── HARD PRICE DIRECTION GATE ── (REMOVED — now handled by mandatory Gate C above)
                # Gate C already enforces ≥0.20% price move in OI direction as a mandatory factor.
                # No need for a separate redundant price gate here.

                _oi_price_delta_str = f'{_oi_mf_C_chg:+.2f}%'
                _oi_quality_tags_raw = _oi_res.get('_quality_boosts', '')

                t._wlog(f"  ✅ OI_WATCHER HIGH CONVICTION: {_oi_sym.replace('NSE:', '')} "
                           f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                           f"{_oi_confirms}/{t._oi_min_confirmations} factors confirm "
                           f"price={_oi_price_delta_str} "
                           f"[{_oi_quality_tags_raw}] → FIRING")

                try:
                    with t._trade_lock:
                        _oi_result = t.tools.place_option_order(
                            underlying=_oi_sym,
                            direction=_oi_dir,
                            strike_selection=_oi_strike_sel,
                            rationale=(f"OI_WATCHER: {_oi_confirms}-factor conviction — {_oi_sig} "
                                       f"strength={_oi_str:.2f} PCR={_oi_pcr:.2f} "
                                       f"bias={_oi_bias} strike={_oi_strike_sel}"
                                       f"{_oi_hm_tag}"
                                       f" | factors=[{_oi_res.get('_quality_boosts', '')}]"),
                            setup_type='OI_WATCHER',
                            ml_data=_oi_ml_data,
                            pre_fetched_market_data={}
                        )

                    if _oi_result and _oi_result.get('success'):
                        t._wlog(f"  🎯 OI_WATCHER FIRED: {_oi_sym.replace('NSE:', '')} "
                                   f"({_oi_dir}) signal={_oi_sig} strength={_oi_str:.3f} "
                                   f"strike={_oi_strike_sel} order={_oi_result.get('order_id', '?')}")
                        t._oi_watcher_fired_this_session.add(_oi_sym)
                        t._oi_watcher_total_placed += 1
                        t._watcher_fired_this_session.add(_oi_sym)  # Prevent pipeline re-fire
                        _oi_placed_count += 1
                        # Store entry OI snapshot for exit intelligence
                        t._oi_watcher_entry_snapshots[_oi_sym] = {
                            'signal': _oi_sig,
                            'strength': _oi_str,
                            'direction': _oi_dir,
                            'participant': _oi_res.get('oi_participant_id', 'UNKNOWN'),
                            'pcr': _oi_pcr,
                            'bias': _oi_bias,
                        }
                        t._log_decision(
                            _wt.strftime('%Y-%m-%d %H:%M:%S'), _oi_sym, _oi_str * 100,
                            'OI_WATCHER_FIRED',
                            reason=(f'{_oi_confirms}-factor conviction: {_oi_sig} strength={_oi_str:.3f} '
                                    f'PCR={_oi_pcr:.2f} bias={_oi_bias} '
                                    f'factors=[{_oi_res.get("_quality_boosts", "")}]'),
                            direction=_oi_dir, setup='OI_WATCHER')
                    else:
                        _oi_err = _oi_result.get('error', 'unknown') if _oi_result else 'no result'
                        # ── CAPITAL SWAP for OI_WATCHER (highest priority) ──
                        _oi_exp_block = ('RISK GOVERNOR BLOCK' in str(_oi_err) and 'exposure' in str(_oi_err).lower()) or \
                                        ('REGIME POSITION LIMIT' in str(_oi_err))
                        if _oi_exp_block and CAPITAL_SWAP.get('enabled', False):
                            t._wlog(f"  🔄 OI_WATCHER SWAP: {_oi_sym.replace('NSE:', '')} blocked — searching for eviction candidate...")
                            _oi_evict = t._find_eviction_candidate('OI_WATCHER')
                            if _oi_evict:
                                with t._trade_lock:
                                    _oi_evicted = t._execute_eviction(_oi_evict, f"OI_WATCHER:{_oi_sym.replace('NSE:', '')}")
                                if _oi_evicted:
                                    _wt.sleep(0.5)
                                    with t._trade_lock:
                                        _oi_result = t.tools.place_option_order(
                                            underlying=_oi_sym, direction=_oi_dir,
                                            strike_selection=_oi_strike_sel,
                                            rationale=(f"CAPITAL_SWAP→OI_WATCHER: {_oi_sig} str={_oi_str:.2f} "
                                                       f"(evicted {_oi_evict['symbol']})"),
                                            setup_type='OI_WATCHER', ml_data=_oi_ml_data, pre_fetched_market_data={}
                                        )
                                    if _oi_result and _oi_result.get('success'):
                                        t._wlog(f"  🎯 OI_WATCHER SWAP FIRED: {_oi_sym.replace('NSE:', '')} "
                                                   f"({_oi_dir}) — replaced {_oi_evict['symbol']}")
                                        t._oi_watcher_fired_this_session.add(_oi_sym)
                                        t._oi_watcher_total_placed += 1
                                        t._watcher_fired_this_session.add(_oi_sym)
                                        _oi_placed_count += 1
                                        t._oi_watcher_entry_snapshots[_oi_sym] = {
                                            'signal': _oi_sig, 'strength': _oi_str,
                                            'direction': _oi_dir,
                                            'participant': _oi_res.get('oi_participant_id', 'UNKNOWN'),
                                            'pcr': _oi_pcr, 'bias': _oi_bias,
                                        }
                                        t._log_decision(
                                            _wt.strftime('%Y-%m-%d %H:%M:%S'), _oi_sym, _oi_str * 100,
                                            'OI_WATCHER_SWAP_FIRED',
                                            reason=(f'CAPITAL_SWAP: evicted {_oi_evict["symbol"]} for OI trade: '
                                                    f'{_oi_sig} str={_oi_str:.3f}'),
                                            direction=_oi_dir, setup='OI_WATCHER')
                                    else:
                                        _retry_err = _oi_result.get('error', 'unknown') if _oi_result else 'no result'
                                        t._wlog(f"  ⚠️ OI_WATCHER SWAP RETRY FAILED: {_retry_err}")
                            else:
                                t._wlog(f"  ❌ OI_WATCHER: No eviction candidate available")
                        else:
                            t._wlog(f"  ⚠️ OI_WATCHER FAILED: {_oi_sym.replace('NSE:', '')} — {_oi_err}")
                except Exception as _oi_exc:
                    t._wlog(f"  ❌ OI_WATCHER ERROR: {_oi_exc}")
            if _oi_placed_count > 0:
                t._wlog(f"🔬 OI_WATCHER: Fired {_oi_placed_count}/{_oi_max_fire} candidates this cycle")
        else:
            t._wlog(f"🔬 OI_WATCHER: No candidates with strength >= {t._oi_watcher_min_strength}")

    def aggressive_buildup_scan(self):
        """Independent OI buildup scanner — finds LB/SB before price triggers."""
        t = self.trader
        import time as _oiag_t
        from datetime import datetime as _oiag_dt

        # Pre-checks
        if not t._oi_analyzer:
            return
        if t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
            return
        _now = _oiag_dt.now()
        _hm = _now.strftime('%H:%M')
        if _hm < '09:35' or _hm > '14:45':
            return  # Only during active trading window

        # Get ticker reference
        _ticker = getattr(t.tools, 'ticker', None)
        if not _ticker:
            return

        # Build list of equity symbols from ticker's token map, compute change%
        _syms_with_change = []
        with _ticker._lock:
            for _tok, _sym in _ticker._token_to_symbol.items():
                if not _sym.startswith('NSE:') or ':NIFTY' in _sym or 'NFO:' in _sym:
                    continue
                _q = _ticker._quote_cache.get(_tok)
                if not _q:
                    continue
                _ltp = _q.get('last_price', 0)
                _ohlc = _q.get('ohlc', {})
                _close = _ohlc.get('close', 0)
                if _ltp <= 0 or _close <= 0:
                    continue
                _chg_pct = ((_ltp - _close) / _close) * 100
                _syms_with_change.append((_sym, abs(_chg_pct), _chg_pct))

        if not _syms_with_change:
            return

        # Sort by absolute change%, pick top N movers for OI analysis
        _syms_with_change.sort(key=lambda x: x[1], reverse=True)
        _scan_syms = []
        for _s, _abs_chg, _chg in _syms_with_change:
            if _abs_chg < 0.3:
                break  # Below 0.3% change — not worth scanning
            if _s in t._oi_watcher_fired_this_session:
                continue
            if t.tools.is_symbol_in_active_trades(_s):
                continue
            _scan_syms.append(_s)
            if len(_scan_syms) >= t._oi_aggr_max_symbols:
                break

        if not _scan_syms:
            return

        # Parallel OI fetch (5 workers for faster completion)
        from concurrent.futures import ThreadPoolExecutor as _OITP, as_completed as _oi_done
        _oi_raw = {}
        _fetch_start = _oiag_t.time()
        with _OITP(max_workers=5, thread_name_prefix='oi-aggr') as _ex:
            _futs = {_ex.submit(t._oi_analyzer.analyze, _s): _s for _s in _scan_syms}
            try:
                for _f in _oi_done(_futs, timeout=max(18, len(_scan_syms) * 2)):
                    _sym = _futs[_f]
                    try:
                        _res = _f.result()
                        if _res:
                            _oi_raw[_sym] = _res
                    except Exception as e:
                        t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_result]: {e}")
            except Exception as e:
                t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_timeout]: {e}")  # Timeout — proceed with partial data
        _fetch_dur = _oiag_t.time() - _fetch_start

        # Score candidates: extract signal, strength, track history, detect acceleration
        _candidates = []
        _now_ts = _oiag_t.time()
        for _sym, _res in _oi_raw.items():
            _sig = _oi_signal_from_result(_res)
            _dir = _oi_direction(_sig)
            if not _dir:
                continue  # NEUTRAL — skip
            _str = _res.get('nse_oi_buildup_strength', 0.0)
            # PCR surrogate (same as existing OI_WATCHER)
            if _str < 0.01 and _sig and _sig not in ('NEUTRAL', ''):
                _fc = _res.get('flow_confidence', 0.0)
                if _fc > 0.1:
                    _str = _fc

            # [FIX Mar 24 v2] Base strength floor — reject noise before multipliers
            if _str < t._oi_watcher_min_strength:
                continue
            # [FIX Mar 24 v2] SHORT_COVERING / LONG_UNWINDING need higher base
            if _sig in ('SHORT_COVERING', 'LONG_UNWINDING') and _str < 0.40:
                continue

            # ── SMART QUALITY SCORING (same as primary OI_WATCHER) ──
            _part = _res.get('oi_participant_id', 'UNKNOWN')
            _eff_str = _str
            # [FIX Mar 24 v2] Track mandatory factors for AGGR path gate
            _ag_mf_C = False   # Price confirmation
            _ag_mf_F = False   # Futures OI buildup
            _ag_mf_H = False   # ATM OI concentration
            _ag_mf_K = False   # Futures conviction (OI day-high or order book)
            _ag_mf_M = False   # Volume surge alignment
            _ag_boosts = []
            # (A) Participant quality: GRANULAR writer/buyer ratio
            _ag_pid_detail = _res.get('oi_participant_detail', {})
            _ag_writer_ratio = None
            if _ag_pid_detail:
                if _dir == 'BUY':  # LONG_BUILDUP → PE writers are conviction
                    _ag_w_oi = _ag_pid_detail.get('pe_writer_oi', 0)
                    _ag_b_oi = _ag_pid_detail.get('pe_buyer_oi', 0)
                else:  # SHORT_BUILDUP → CE writers are conviction
                    _ag_w_oi = _ag_pid_detail.get('ce_writer_oi', 0)
                    _ag_b_oi = _ag_pid_detail.get('ce_buyer_oi', 0)
                _ag_total_classified = _ag_w_oi + _ag_b_oi
                if _ag_total_classified > 0:
                    _ag_writer_ratio = _ag_w_oi / _ag_total_classified
            if _ag_writer_ratio is not None:
                if _ag_writer_ratio >= 0.85:
                    _eff_str *= 1.25  # 85%+ writer = rock solid institutional conviction
                    _ag_boosts.append(f'W✓✓{_ag_writer_ratio:.0%}')
                elif _ag_writer_ratio >= 0.65:
                    _eff_str *= 1.12  # Strong writer majority
                    _ag_boosts.append(f'W✓{_ag_writer_ratio:.0%}')
                elif _ag_writer_ratio >= 0.50:
                    _eff_str *= 1.0   # Neutral — no boost
                    _ag_boosts.append(f'W~{_ag_writer_ratio:.0%}')
                else:
                    _eff_str *= 0.75  # Buyer-dominated = hedging, NOT directional
                    _ag_boosts.append(f'B✗{_ag_writer_ratio:.0%}')
            else:
                # Fallback to label when detail not available
                if _part == 'WRITER_DOMINANT':
                    _eff_str *= 1.15
                    _ag_boosts.append('W+')
                elif _part == 'BUYER_DOMINANT':
                    _eff_str *= 0.85
                    _ag_boosts.append('B-')
            # (B) Cross-validation boost
            if _res.get('oi_cross_validated'):
                _eff_str *= 1.10
                _ag_boosts.append('XV✓')
            # (C) Price confirmation (we already have change% from _syms_with_change)
            _ag_chg = next((_c for _s, _, _c in _syms_with_change if _s == _sym), None)
            if _ag_chg is not None:
                _ag_price_agrees = (
                    (_dir == 'BUY' and _ag_chg > 0.3) or
                    (_dir == 'SELL' and _ag_chg < -0.3)
                )
                _ag_price_diverges = (
                    (_dir == 'BUY' and _ag_chg < -0.5) or
                    (_dir == 'SELL' and _ag_chg > 0.5)
                )
                if _ag_price_agrees:
                    _eff_str *= 1.08
                    _ag_boosts.append(f'P✓{_ag_chg:+.1f}%')
                    _ag_mf_C = True
                elif _ag_price_diverges:
                    _eff_str *= 0.90
                    _ag_boosts.append(f'P✗{_ag_chg:+.1f}%')

            # (E) Sector alignment (same as primary OI_WATCHER)
            _ag_s2s = getattr(t, '_stock_to_sector', {})
            _ag_sec_chgs = getattr(t, '_sector_index_changes_cache', {})
            _ag_sec_info = _ag_s2s.get(_sym.replace('NSE:', ''))
            if _ag_sec_info and _ag_sec_chgs:
                _ag_sec_name, _ag_sec_idx = _ag_sec_info
                _ag_sec_chg = _ag_sec_chgs.get(_ag_sec_idx, 0)
                _ag_sec_agrees = (
                    (_dir == 'BUY' and _ag_sec_chg > 0.3) or
                    (_dir == 'SELL' and _ag_sec_chg < -0.3)
                )
                _ag_sec_oppose = (
                    (_dir == 'BUY' and _ag_sec_chg < -0.5) or
                    (_dir == 'SELL' and _ag_sec_chg > 0.5)
                )
                if _ag_sec_agrees:
                    _eff_str *= 1.08
                    _ag_boosts.append('SEC✓')
                elif _ag_sec_oppose:
                    _eff_str *= 0.88
                    _ag_boosts.append('SEC✗')

            # (F) Futures OI buildup cross-check
            _ag_ml = getattr(t, '_cycle_ml_results', {})
            _ag_ml_data = _ag_ml.get(_sym, {})
            _ag_fut_buildup = _ag_ml_data.get('fut_oi_buildup', 0) if isinstance(_ag_ml_data, dict) else 0
            if _ag_fut_buildup:
                _ag_fut_agrees = (
                    (_dir == 'BUY' and _ag_fut_buildup > 0) or
                    (_dir == 'SELL' and _ag_fut_buildup < 0)
                )
                _ag_fut_strong = abs(_ag_fut_buildup) >= 0.75
                if _ag_fut_agrees and _ag_fut_strong:
                    _eff_str *= 1.12
                    _ag_boosts.append('FUT✓✓')
                    _ag_mf_F = True
                elif _ag_fut_agrees:
                    _eff_str *= 1.05
                    _ag_boosts.append('FUT✓')
                    _ag_mf_F = True
                elif not _ag_fut_agrees and _ag_fut_strong:
                    _eff_str *= 0.85
                    _ag_boosts.append('FUT✗✗')

            # (G) Live Futures Basis: premium/discount from ticker cache (0 API calls)
            if _ticker:
                try:
                    _ag_fut_data = _ticker.get_futures_oi(_sym)
                    if _ag_fut_data and _ag_fut_data.get('ltp', 0) > 0:
                        _ag_fut_ltp = _ag_fut_data['ltp']
                        _ag_eq_ltp = 0
                        with _ticker._lock:
                            for _tk2, _tsym2 in _ticker._token_to_symbol.items():
                                if _tsym2 == _sym:
                                    _ag_eq_ltp = _ticker._quote_cache.get(_tk2, {}).get('last_price', 0)
                                    break
                        if _ag_eq_ltp > 0:
                            _ag_basis_pct = ((_ag_fut_ltp - _ag_eq_ltp) / _ag_eq_ltp) * 100
                            _ag_basis_agrees = (
                                (_dir == 'BUY' and _ag_basis_pct > 0.05) or
                                (_dir == 'SELL' and _ag_basis_pct < -0.05)
                            )
                            _ag_basis_disagrees = (
                                (_dir == 'BUY' and _ag_basis_pct < -0.10) or
                                (_dir == 'SELL' and _ag_basis_pct > 0.10)
                            )
                            if _ag_basis_agrees:
                                _eff_str *= 1.10
                                _ag_boosts.append(f'BASIS✓{_ag_basis_pct:+.2f}%')
                            elif _ag_basis_disagrees:
                                _eff_str *= 0.90
                                _ag_boosts.append(f'BASIS✗{_ag_basis_pct:+.2f}%')
                except Exception as e:
                    t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_basis]: {e}")

            # (H) OI Concentration vs Spot: near-ATM buildup = conviction, far OTM = hedging
            _ag_spot = _res.get('spot_price', 0) or _res.get('dhan_spot_price', 0)
            if _ag_spot > 0:
                _ag_rel_strikes = (
                    _res.get('nse_top_put_oi_change', []) if _dir == 'BUY'
                    else _res.get('nse_top_call_oi_change', [])
                )
                if _ag_rel_strikes and len(_ag_rel_strikes) > 0:
                    _ag_top_stk = _ag_rel_strikes[0][0] if isinstance(_ag_rel_strikes[0], (list, tuple)) else 0
                    if _ag_top_stk > 0:
                        _ag_stk_dist = abs(_ag_top_stk - _ag_spot) / _ag_spot * 100
                        if _ag_stk_dist <= 2.0:
                            _eff_str *= 1.10
                            _ag_boosts.append(f'ATM✓{_ag_stk_dist:.1f}%')
                            _ag_mf_H = True
                        elif _ag_stk_dist >= 5.0:
                            _eff_str *= 0.92
                            _ag_boosts.append(f'OTM✗{_ag_stk_dist:.1f}%')

            # (I) PCR Shift Rate: rate of PCR change
            _ag_pcr_rate = _res.get('pcr_shift_rate', 0)
            if abs(_ag_pcr_rate) > 0.005:
                _ag_pcr_confirms = (
                    (_dir == 'BUY' and _ag_pcr_rate > 0.01) or
                    (_dir == 'SELL' and _ag_pcr_rate < -0.01)
                )
                _ag_pcr_opposes = (
                    (_dir == 'BUY' and _ag_pcr_rate < -0.01) or
                    (_dir == 'SELL' and _ag_pcr_rate > 0.01)
                )
                if _ag_pcr_confirms:
                    _eff_str *= 1.08
                    _ag_boosts.append(f'PCR↗{_ag_pcr_rate:+.3f}')
                elif _ag_pcr_opposes:
                    _eff_str *= 0.92
                    _ag_boosts.append(f'PCR↘{_ag_pcr_rate:+.3f}')

            # (J) Volume PCR: today's volume intent vs stale OI
            _ag_vol_pcr = _res.get('nse_pcr_volume', 0)
            if _ag_vol_pcr and _ag_vol_pcr > 0:
                _ag_vp_confirms = (
                    (_dir == 'BUY' and _ag_vol_pcr > 1.3) or
                    (_dir == 'SELL' and _ag_vol_pcr < 0.7)
                )
                _ag_vp_opposes = (
                    (_dir == 'BUY' and _ag_vol_pcr < 0.6) or
                    (_dir == 'SELL' and _ag_vol_pcr > 1.5)
                )
                if _ag_vp_confirms:
                    _eff_str *= 1.08
                    _ag_boosts.append(f'VP✓{_ag_vol_pcr:.2f}')
                elif _ag_vp_opposes:
                    _eff_str *= 0.90
                    _ag_boosts.append(f'VP✗{_ag_vol_pcr:.2f}')

            # (K) Futures Conviction Boost: OI Day-High + Order Book Imbalance (BOOST-ONLY)
            # Same as OI_WATCHER Factor K — uses Kite WebSocket futures data (0 API calls).
            if _ticker:
                try:
                    _ag_fk = _ticker.get_futures_oi(_sym)
                    if _ag_fk and _ag_fk.get('oi', 0) > 0:
                        _ag_fk_oi = _ag_fk['oi']
                        _ag_fk_high = _ag_fk.get('oi_day_high', 0)
                        _ag_fk_low = _ag_fk.get('oi_day_low', 0)

                        # K1: OI at Day-High = institutions actively adding
                        if _ag_fk_high > _ag_fk_low > 0:
                            _ag_fk_range = _ag_fk_high - _ag_fk_low
                            _ag_fk_pos = (_ag_fk_oi - _ag_fk_low) / _ag_fk_range
                            if _ag_fk_pos >= 0.85:
                                _eff_str *= 1.18
                                _ag_boosts.append(f'FOIDH✓{_ag_fk_pos:.0%}')
                                _ag_mf_K = True
                            elif _ag_fk_pos >= 0.65:
                                _eff_str *= 1.08
                                _ag_boosts.append(f'FOIDH~{_ag_fk_pos:.0%}')

                        # K2: Futures Order Book Imbalance — confirms direction
                        _ag_fk_buy = _ag_fk.get('buy_quantity', 0)
                        _ag_fk_sell = _ag_fk.get('sell_quantity', 0)
                        if _ag_fk_buy > 0 and _ag_fk_sell > 0:
                            _ag_fk_imb = _ag_fk_buy / (_ag_fk_buy + _ag_fk_sell)
                            _ag_fk_imb_confirms = (
                                (_dir == 'BUY' and _ag_fk_imb > 0.55) or
                                (_dir == 'SELL' and _ag_fk_imb < 0.45)
                            )
                            if _ag_fk_imb_confirms:
                                _ag_boost_mult = 1.08 + min(0.10, abs(_ag_fk_imb - 0.50) * 0.5)
                                _eff_str *= _ag_boost_mult
                                _ag_boosts.append(f'FOBI✓{_ag_fk_imb:.0%}')
                                _ag_mf_K = True
                except Exception as e:
                    t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_fobi]: {e}")

            # (M) Volume Surge Alignment for AGGR path — same logic as Path 1
            if _ticker:
                try:
                    _ag_vdh_store = getattr(_ticker, '_vol_delta_history', None)
                    _ag_vdh = _ag_vdh_store.get(_sym) if _ag_vdh_store is not None else None
                    if _ag_vdh and len(_ag_vdh) >= 5:
                        _ag_vavg = sum(_ag_vdh) / len(_ag_vdh)
                        _ag_vlast = list(_ag_vdh)[-1] if _ag_vdh else 0
                        if _ag_vavg > 0 and _ag_vlast > 0:
                            _ag_vol_surge_ratio = _ag_vlast / _ag_vavg
                            _ag_v_recent = list(_ag_vdh)[-3:]
                            _ag_v_elevated = sum(1 for _vd in _ag_v_recent if _vd >= _ag_vavg * 1.4)
                            if _ag_vol_surge_ratio >= 1.79 and _ag_v_elevated >= 2:
                                with _ticker._lock:
                                    _ag_vtok = None
                                    for _tk_v, _tsym_v in _ticker._token_to_symbol.items():
                                        if _tsym_v == _sym:
                                            _ag_vtok = _tk_v
                                            break
                                    if _ag_vtok:
                                        _ag_vq = _ticker._quote_cache.get(_ag_vtok, {})
                                        _ag_vltp = _ag_vq.get('last_price', 0)
                                        _ag_vclose = (_ag_vq.get('ohlc', {}) or {}).get('close', 0)
                                        if _ag_vltp > 0 and _ag_vclose > 0:
                                            _ag_vchg = ((_ag_vltp - _ag_vclose) / _ag_vclose) * 100
                                            if (_dir == 'BUY' and _ag_vchg > 0) or \
                                               (_dir == 'SELL' and _ag_vchg < 0):
                                                _ag_mf_M = True
                                                _eff_str *= 1.15
                                                _ag_boosts.append(f'VSRG✓{_ag_vol_surge_ratio:.1f}x')
                                            else:
                                                _eff_str *= 0.80
                                                _ag_boosts.append(f'VSRG✗{_ag_vol_surge_ratio:.1f}x')
                            elif _ag_vol_surge_ratio >= 1.46 and _ag_v_elevated >= 1:
                                # [FIX Mar 24 v3] Intermediate volume — require price alignment too
                                with _ticker._lock:
                                    _ag_vtok_int = None
                                    for _tk_vi, _tsym_vi in _ticker._token_to_symbol.items():
                                        if _tsym_vi == _sym:
                                            _ag_vtok_int = _tk_vi
                                            break
                                    if _ag_vtok_int:
                                        _ag_vq_int = _ticker._quote_cache.get(_ag_vtok_int, {})
                                        _ag_vltp_int = _ag_vq_int.get('last_price', 0)
                                        _ag_vclose_int = (_ag_vq_int.get('ohlc', {}) or {}).get('close', 0)
                                        if _ag_vltp_int > 0 and _ag_vclose_int > 0:
                                            _ag_vchg_int = ((_ag_vltp_int - _ag_vclose_int) / _ag_vclose_int) * 100
                                            if (_dir == 'BUY' and _ag_vchg_int > 0) or \
                                               (_dir == 'SELL' and _ag_vchg_int < 0):
                                                _ag_mf_M = True
                                                _ag_boosts.append(f'VOL~✓{_ag_vol_surge_ratio:.1f}x')
                                            else:
                                                _ag_boosts.append(f'VOL~✗{_ag_vol_surge_ratio:.1f}x')
                                        else:
                                            _ag_boosts.append(f'VOL~?{_ag_vol_surge_ratio:.1f}x')
                                    else:
                                        _ag_boosts.append(f'VOL~?{_ag_vol_surge_ratio:.1f}x')
                            else:
                                _ag_boosts.append(f'VOL↓{_ag_vol_surge_ratio:.1f}x')
                    else:
                        # [FIX Mar 24 v3] Not enough volume history — fail-closed
                        t._wlog(f"⚠️ VOL_SURGE BLOCKED [aggr]: {_sym.replace('NSE:', '')} "
                                   f"— insufficient tick history ({len(_ag_vdh) if _ag_vdh else 0}<5), M=False")
                except Exception as _ve:
                    # [FIX Mar 24 v3] Exception — fail-closed
                    t._wlog(f"⛔ VOL_SURGE HALTED [aggr]: {_sym.replace('NSE:', '')} — {_ve}")
            else:
                # [FIX Mar 24 v3] No ticker — fail-closed, halt trades
                t._wlog(f"⛔ VOL_SURGE HALTED [aggr]: ticker DOWN — M=False for all, no trades without volume data")

            # [FIX Mar 24 v2] Cap multiplicative inflation — multiplier product capped at 2.0x base
            _ag_max_boosted = _str * 2.0
            _eff_str = min(1.0, min(_ag_max_boosted, _eff_str))
            # [FIX Mar 24 v2] SHORT_COVERING / LONG_UNWINDING need ≥0.65 effective to fire
            if _sig in ('SHORT_COVERING', 'LONG_UNWINDING') and _eff_str < 0.65:
                continue
            # [FIX Mar 24 v2] Store mandatory factor results for AGGR gate
            _res['_ag_mf_C'] = _ag_mf_C
            _res['_ag_mf_F'] = _ag_mf_F
            _res['_ag_mf_H'] = _ag_mf_H
            _res['_ag_mf_K'] = _ag_mf_K
            _res['_ag_mf_M'] = _ag_mf_M
            _res['_quality_boosts'] = ' '.join(_ag_boosts) if _ag_boosts else ''

            # Track history for acceleration detection (keep last 5 reads)
            if _sym not in t._oi_aggr_strength_history:
                t._oi_aggr_strength_history[_sym] = []
            _hist = t._oi_aggr_strength_history[_sym]
            _hist.append((_now_ts, _eff_str, _sig))
            if len(_hist) > 5:
                t._oi_aggr_strength_history[_sym] = _hist[-5:]

            # Detect acceleration: strength increasing across last 2 reads
            _is_accel = False
            _accel_delta = 0.0
            if len(_hist) >= 2:
                _prev_str = _hist[-2][1]
                _accel_delta = _eff_str - _prev_str
                _prev_sig_dir = _oi_direction(_hist[-2][2])
                # Acceleration = same direction + strength increasing
                if _accel_delta >= t._oi_aggr_accel_threshold and _prev_sig_dir == _dir:
                    _is_accel = True

            # Decision logic:
            # 1. Strong buildup (≥0.45) → fire immediately, no acceleration needed
            # 2. Moderate with acceleration (≥0.25 + accel) → fire immediately
            # 3. Below thresholds → skip
            _fire = False
            _reason = ''
            if _eff_str >= t._oi_aggr_strong_str:
                _fire = True
                _reason = f'STRONG str={_eff_str:.3f}≥{t._oi_aggr_strong_str}'
            elif _is_accel and _eff_str >= t._oi_aggr_accel_min_str:
                _fire = True
                _reason = f'ACCEL str={_eff_str:.3f} Δ={_accel_delta:+.3f}'
            elif _eff_str >= t._oi_watcher_min_strength:
                # Standard threshold met but no acceleration — still add as candidate
                # but lower priority than accelerating signals
                _fire = True
                _reason = f'STANDARD str={_eff_str:.3f}≥{t._oi_watcher_min_strength}'

            if _fire:
                _candidates.append((_sym, _dir, _sig, _eff_str, _res, _reason, _is_accel, _accel_delta))

        if not _candidates:
            t._wlog(f"🔬 OI_AGGR: Scanned {len(_scan_syms)} movers, fetched {len(_oi_raw)} OI "
                       f"({_fetch_dur:.1f}s) — no LB/SB candidates")
            return

        # Sort: accelerating first, then by strength
        _candidates.sort(key=lambda x: (x[6], x[3]), reverse=True)

        t._wlog(f"🔬 OI_AGGR: Scanned {len(_scan_syms)} movers → {len(_candidates)} candidates "
                   f"({_fetch_dur:.1f}s)")
        for _c in _candidates[:5]:
            _tag = '🚀ACCEL' if _c[6] else '⚡'
            _ag_qt = _c[4].get('_quality_boosts', '')
            t._wlog(f"  {_tag} {_c[0].replace('NSE:', '')} {_c[2]} str={_c[3]:.3f} "
                       f"dir={_c[1]} part={_c[4].get('oi_participant_id', '?')} "
                       f"quality=[{_ag_qt}] [{_c[5]}]")

        # Fire top candidates — up to 3 if accelerating, 2 otherwise
        _placed_this_scan = 0
        _max_fire = 3 if _candidates[0][6] else 2  # [FIX Mar 19] Fire top-3 if accelerating, top-2 otherwise

        for _c in _candidates:
            if _placed_this_scan >= _max_fire:
                break
            if t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
                break

            _sym, _dir, _sig, _str, _res, _reason, _is_accel, _accel_delta = _c
            _pcr = _res.get('pcr_oi', 1.0)
            _bias = _res.get('flow_bias', 'NEUTRAL')
            _part_id = _res.get('oi_participant_id', 'UNKNOWN')

            # [FIX Mar 24 v2] MANDATORY FACTOR GATE for AGGR path
            # Require price confirmation (C) + at least 1 of F/H/K
            _ag_mf_C = _res.get('_ag_mf_C', False)
            _ag_mf_F = _res.get('_ag_mf_F', False)
            _ag_mf_H = _res.get('_ag_mf_H', False)
            _ag_mf_K = _res.get('_ag_mf_K', False)
            _ag_mf_fhk = sum([_ag_mf_F, _ag_mf_H, _ag_mf_K])
            _ag_mf_M = _res.get('_ag_mf_M', False)
            if not _ag_mf_C:
                t._wlog(f"  ⛔ OI_AGGR PRICE MANDATORY: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} — "
                           f"price not confirming — skipping")
                continue
            if not _ag_mf_M:
                t._wlog(f"  ⛔ OI_AGGR VOL SURGE MANDATORY: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} — "
                           f"volume not surging — skipping")
                continue
            if _ag_mf_fhk < 1:
                t._wlog(f"  ⛔ OI_AGGR LOW CONVICTION: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} — "
                           f"0/{3} of F/H/K confirmed — need ≥1")
                continue

            # Heatmap strike picker
            _strike_sel = 'ATM'
            _hm_tag = ''
            try:
                from dhan_oi_fetcher import DhanOIFetcher
                _hm_strikes = _res.get('dhan_strikes', [])
                _hm_spot = _res.get('dhan_spot_price', 0)
                if _hm_strikes and _hm_spot > 0:
                    _hm = DhanOIFetcher.find_optimal_strike(_dir, _hm_strikes, _hm_spot)
                    if _hm.get('score', 0) > 0:
                        _strike_sel = _hm['selection']
                        _hm_tag = (f" | HEATMAP: {_hm['selection']}@{_hm['strike']:.0f} "
                                   f"score={_hm['score']:.0f} ({_hm['reason']})")
            except Exception as e:
                t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_heatmap]: {e}")

            _accel_tag = f' ACCEL(Δ={_accel_delta:+.3f})' if _is_accel else ''
            t._wlog(f"🎯 OI_AGGR FIRE: {_sym.replace('NSE:', '')} {_sig} "
                       f"str={_str:.3f} dir={_dir} PCR={_pcr:.2f} part={_part_id} "
                       f"[{_reason}]{_accel_tag} strike={_strike_sel}{_hm_tag}")

            _ml_data = {
                'oi_signal': _sig,
                'oi_strength': _str,
                'oi_pcr': _pcr,
                'oi_bias': _bias,
                'oi_participant_id': _part_id,
                'trade_type': 'OI_WATCHER',
                'oi_heatmap_strike': _strike_sel,
                'oi_aggressive': True,
                'oi_acceleration': _is_accel,
                'oi_accel_delta': round(_accel_delta, 4),
            }

            # ── 60-SECOND OI CONFIRMATION GATE + PRICE DELTA (OI_AGGR) ──
            _ag_now_ts = _oiag_t.time()
            _ag_spot_now = _res.get('spot_price', 0) or _res.get('dhan_spot_price', 0)
            if not _ag_spot_now and _ticker:
                try:
                    with _ticker._lock:
                        for _tk, _tsym in _ticker._token_to_symbol.items():
                            if _tsym == _sym:
                                _ag_spot_now = _ticker._quote_cache.get(_tk, {}).get('last_price', 0)
                                break
                except Exception as e:
                    t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_spot]: {e}")
            _ag_pending = t._oi_pending_confirm.get(_sym)
            if _ag_pending is None:
                t._oi_pending_confirm[_sym] = {
                    'ts': _ag_now_ts, 'direction': _dir,
                    'strength': _str, 'signal': _sig, 'source': 'OI_AGGR',
                    'spot_price': _ag_spot_now,
                }
                t._wlog(f"  ⏳ OI_AGGR PENDING: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} "
                           f"spot={_ag_spot_now:.2f} — "
                           f"waiting {t._oi_confirm_seconds}s confirmation")
                continue
            _ag_elapsed = _ag_now_ts - _ag_pending['ts']
            if _ag_pending['direction'] != _dir:
                t._oi_pending_confirm[_sym] = {
                    'ts': _ag_now_ts, 'direction': _dir,
                    'strength': _str, 'signal': _sig, 'source': 'OI_AGGR',
                    'spot_price': _ag_spot_now,
                }
                t._wlog(f"  🔄 OI_AGGR RESET: {_sym.replace('NSE:', '')} "
                           f"direction flipped {_ag_pending['direction']}→{_dir} — "
                           f"restarting {t._oi_confirm_seconds}s wait")
                continue
            if _ag_elapsed < t._oi_confirm_seconds:
                t._wlog(f"  ⏳ OI_AGGR WAITING: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} — {_ag_elapsed:.0f}s / "
                           f"{t._oi_confirm_seconds}s elapsed")
                continue
            # Time elapsed — check PRICE DELTA
            _ag_spot_entry = _ag_pending.get('spot_price', 0)
            _ag_price_ok = True
            _ag_price_delta_pct = 0.0
            if _ag_spot_entry > 0 and _ag_spot_now > 0:
                _ag_price_delta_pct = ((_ag_spot_now - _ag_spot_entry) / _ag_spot_entry) * 100
                if _dir == 'BUY' and _ag_price_delta_pct < t._oi_confirm_min_price_delta:
                    _ag_price_ok = False
                elif _dir == 'SELL' and _ag_price_delta_pct > -t._oi_confirm_min_price_delta:
                    _ag_price_ok = False
            if not _ag_price_ok:
                t._wlog(f"  ❌ OI_AGGR PRICE REJECT: {_sym.replace('NSE:', '')} "
                           f"{_sig} dir={_dir} — OI held {_ag_elapsed:.0f}s but "
                           f"price Δ={_ag_price_delta_pct:+.2f}% — OI trap, skipping")
                del t._oi_pending_confirm[_sym]
                continue
            # ✅ CONFIRMED: OI persisted + price moved in OI direction
            t._wlog(f"  ✅ OI_AGGR CONFIRMED: {_sym.replace('NSE:', '')} "
                       f"{_sig} str={_str:.3f} dir={_dir} — "
                       f"held {_ag_elapsed:.0f}s, price Δ={_ag_price_delta_pct:+.2f}% → FIRING")
            del t._oi_pending_confirm[_sym]

            try:
                with t._trade_lock:
                    _result = t.tools.place_option_order(
                        underlying=_sym,
                        direction=_dir,
                        strike_selection=_strike_sel,
                        rationale=(f"OI_WATCHER_AGGR: {_sig} str={_str:.2f} "
                                   f"PCR={_pcr:.2f} bias={_bias} "
                                   f"[{_reason}]{_accel_tag} strike={_strike_sel}"
                                   f"{_hm_tag}"),
                        setup_type='OI_WATCHER',
                        ml_data=_ml_data,
                        pre_fetched_market_data={}
                    )

                if _result and _result.get('success'):
                    t._wlog(f"  ✅ OI_AGGR PLACED: {_sym.replace('NSE:', '')} "
                               f"({_dir}) {_sig} str={_str:.3f} order={_result.get('order_id', '?')}")
                    t._oi_watcher_fired_this_session.add(_sym)
                    t._oi_watcher_total_placed += 1
                    t._watcher_fired_this_session.add(_sym)
                    t._oi_watcher_entry_snapshots[_sym] = {
                        'signal': _sig,
                        'strength': _str,
                        'direction': _dir,
                        'participant': _part_id,
                        'pcr': _pcr,
                        'bias': _bias,
                    }
                    _cycle_ts = _oiag_dt.now().strftime('%Y-%m-%d %H:%M:%S')
                    t._log_decision(
                        _cycle_ts, _sym, _str * 100,
                        'OI_WATCHER_AGGR_FIRED',
                        reason=(f'Aggressive OI: {_sig} str={_str:.3f} '
                                f'PCR={_pcr:.2f} [{_reason}]{_accel_tag}'),
                        direction=_dir, setup='OI_WATCHER')
                    _placed_this_scan += 1
                else:
                    _err = _result.get('error', 'unknown') if _result else 'no result'
                    t._wlog(f"  ⚠️ OI_AGGR FAILED: {_sym.replace('NSE:', '')} — {_err}")
            except Exception as _exc:
                t._wlog(f"  ❌ OI_AGGR ERROR: {_sym.replace('NSE:', '')} — {_exc}")

        # Prune stale history entries (older than 15 min or symbols not in scan)
        _cutoff = _now_ts - 900
        _stale_keys = [k for k, v in t._oi_aggr_strength_history.items()
                       if v and v[-1][0] < _cutoff]
        for _k in _stale_keys:
            del t._oi_aggr_strength_history[_k]

        # Prune stale OI pending confirmations (older than 5 min = signal expired)
        _confirm_cutoff = _now_ts - t._oi_confirm_expiry
        _stale_pending = [k for k, v in t._oi_pending_confirm.items()
                          if v['ts'] < _confirm_cutoff]
        for _k in _stale_pending:
            t._wlog(f"  🗑️ OI PENDING EXPIRED: {_k.replace('NSE:', '')} — "
                       f"signal did not reconfirm within {t._oi_confirm_expiry}s")
            del t._oi_pending_confirm[_k]
