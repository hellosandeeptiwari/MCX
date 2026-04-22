"""OI Watcher Engine — extracted from autonomous_trader.py

CRITICAL: This module MUST be imported and used by autonomous_trader.py.
If autonomous_trader.py has inline OI_WATCHER scoring instead of calling
OIWatcherEngine, the 13-factor + anchor gate structure is lost.
See commit 1f51144 for the original extraction.
"""
import threading
from config import CAPITAL_SWAP

# ── GUARD: Fail loudly if this module isn't being imported ──
_ENGINE_LOADED = True  # autonomous_trader checks this at startup


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
            # Apr 15 RCA: Don't filter out already-fired/active-trade symbols HERE.
            # The _mf_* enrichment must run for ALL directional symbols above threshold,
            # because the pipeline's B2-OI-ANCHOR gate reads these flags later.
            # Mark them as skip-for-fire instead, and filter after enrichment.
            _oi_skip_fire = False
            if _oi_sym in t._oi_watcher_fired_this_session:
                _oi_skip_fire = True  # Already fired — enrich but don't fire again
            if t.tools.is_symbol_in_active_trades(_oi_sym):
                _oi_skip_fire = True  # Already holding — enrich but don't fire again
            _oi_candidates.append((_oi_sym, _oi_dir, _oi_sig, _oi_str, _oi_res, _oi_skip_fire))

        if _oi_candidates:
            # ── SMART QUALITY SCORING ──
            # Instead of just raw strength, compute a composite quality score
            # that rewards confluence: participant + cross-validation + price confirmation + OI trend.
            # No harsh gates — everything is a boost/discount to effective strength.
            _ticker = getattr(t.tools, 'ticker', None)

            # (N) FII/DII Cash Flow — macro institutional direction (fetched once, cached 5 min)
            _oc_fii_data = {}
            try:
                from nse_oi_fetcher import get_nse_oi_fetcher
                _oc_fii_data = get_nse_oi_fetcher().fetch_fii_dii()
            except Exception:
                pass

            for _oci in range(len(_oi_candidates)):
                _oc_sym, _oc_dir, _oc_sig, _oc_str, _oc_res, _oc_skip_fire = _oi_candidates[_oci]
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
                    elif _oc_writer_ratio >= 0.70:
                        _oc_eff_str *= 1.12  # Strong writer majority (tightened 0.65→0.70)
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
                _oc_C_min_delta = getattr(t, '_oi_watcher_min_price_delta', 0.20)  # Apr 7: tightened 0.15→0.20%
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
                    if _oc_eff_str > _oc_prev_str * 1.10:  # 10% stronger than last (tightened 1.05→1.10)
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
                        (_oc_dir == 'BUY' and _oc_sec_chg > 0.40) or
                        (_oc_dir == 'SELL' and _oc_sec_chg < -0.40)
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
                _oc_F_strong_confirmed = False  # True only for FUT✓✓ (|buildup| ≥ 0.75)
                _oc_F_evaluable = False  # Apr 15 RCA: True when futures data exists (vs no data at all)
                # [FIX Apr 6] Read from _futures_oi_data (raw parquet), NOT _cycle_ml_results
                # _cycle_ml_results contains ML predictions (ml_signal, ml_prob_up etc.)
                # _futures_oi_data contains raw features (fut_oi_buildup, fut_basis_pct etc.)
                _oc_fut_buildup = 0
                try:
                    _oc_foi = getattr(t, '_futures_oi_data', None) or {}
                    # Apr 21 FIX: load_all_futures_oi_daily() keys are bare symbols
                    # ("MANAPPURAM"), but _oc_sym is "NSE:MANAPPURAM". Try both.
                    _oc_foi_df = _oc_foi.get(_oc_sym) or _oc_foi.get(_oc_sym.replace('NSE:', ''))
                    if _oc_foi_df is not None and len(_oc_foi_df) > 0:
                        _oc_fut_buildup = float(_oc_foi_df.iloc[-1].get('fut_oi_buildup', 0))
                        _oc_F_evaluable = True  # Futures data exists
                except Exception:
                    _oc_fut_buildup = 0
                if _oc_fut_buildup:
                    _oc_fut_agrees = (
                        (_oc_dir == 'BUY' and _oc_fut_buildup > 0) or
                        (_oc_dir == 'SELL' and _oc_fut_buildup < 0)
                    )
                    _oc_fut_strong = abs(_oc_fut_buildup) >= 0.75  # LB/SB not SC/LU
                    if _oc_fut_agrees and _oc_fut_strong:
                        _oc_eff_str *= 1.15  # Futures + Options agree strongly → 15% boost
                        _oc_boosts.append('FUT✓✓')
                        _oc_confirms += 1
                        _oc_F_confirmed = True
                        _oc_F_strong_confirmed = True
                    elif _oc_fut_agrees:
                        _oc_eff_str *= 1.08  # Mild agreement → 8% boost
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
                            _oc_F_evaluable = True  # Basis data exists
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
                                    (_oc_dir == 'BUY' and _oc_basis_pct > 0.10) or
                                    (_oc_dir == 'SELL' and _oc_basis_pct < -0.10)
                                )
                                _oc_basis_disagrees = (
                                    (_oc_dir == 'BUY' and _oc_basis_pct < -0.10) or
                                    (_oc_dir == 'SELL' and _oc_basis_pct > 0.10)
                                )
                                if _oc_basis_agrees:
                                    _oc_eff_str *= 1.10  # Smart money paying premium in your direction
                                    _oc_boosts.append(f'BASIS✓{_oc_basis_pct:+.2f}%')
                                    _oc_confirms += 1
                                    # [FIX Apr 6] Basis agreement = futures anchor fallback
                                    # If fut_oi_buildup was missing, basis confirms F
                                    if not _oc_F_confirmed:
                                        _oc_F_confirmed = True
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
                _oc_H_evaluable = False  # Apr 15 RCA: True when strike-level OI data exists
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
                            _oc_H_evaluable = True  # Strike data available
                            _oc_strike_dist = abs(_oc_top_strike - _oc_spot) / _oc_spot * 100
                            if _oc_strike_dist <= 2.5:  # Apr 7: tightened 3.0→2.5%
                                _oc_eff_str *= 1.14  # Near-money buildup = institutional conviction
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
                if abs(_oc_pcr_rate) > 0.010:  # Meaningful rate of change (tightened 0.005→0.010)
                    _oc_pcr_rate_confirms = (
                        (_oc_dir == 'BUY' and _oc_pcr_rate >= 0.015) or   # Rising PCR = bullish (tightened 0.01→0.015)
                        (_oc_dir == 'SELL' and _oc_pcr_rate <= -0.015)     # Falling PCR = bearish
                    )
                    _oc_pcr_rate_opposes = (
                        (_oc_dir == 'BUY' and _oc_pcr_rate <= -0.015) or
                        (_oc_dir == 'SELL' and _oc_pcr_rate >= 0.015)
                    )
                    if _oc_pcr_rate_confirms:
                        _oc_eff_str *= 1.08  # PCR shifting your way NOW
                        _oc_boosts.append(f'PCR↗{_oc_pcr_rate:+.3f}')
                        _oc_confirms += 1
                    elif _oc_pcr_rate_opposes:
                        _oc_eff_str *= 0.92  # PCR shifting against you
                        _oc_boosts.append(f'PCR↘{_oc_pcr_rate:+.3f}')

                # (J) Volume PCR: REMOVED — PCR extremes gate eliminated

                # (K) Futures Conviction Boost: OI Day-High + Order Book Imbalance
                # Kite WebSocket streams futures OI + buy/sell qty in real-time (0 API calls).
                # BOOST-ONLY: helps the BEST signals rise to the top for entry.
                # buy_quantity = total pending BUY orders in futures order book (bullish demand)
                # sell_quantity = total pending SELL orders in futures order book (bearish supply)
                # [MANDATORY FACTOR] — tracked for conviction gate
                _oc_K_confirmed = False
                _oc_K_strong_confirmed = False  # True only for FOIDH✓ (OI ≥85% of day range)
                _oc_K_evaluable = False  # Apr 15 RCA: True when ticker has futures data for this stock
                if _ticker:
                    try:
                        _oc_fk = _ticker.get_futures_oi(_oc_sym)
                        if _oc_fk and _oc_fk.get('oi', 0) > 0:
                            _oc_K_evaluable = True  # Futures OI data available
                            _oc_fk_oi = _oc_fk['oi']
                            _oc_fk_high = _oc_fk.get('oi_day_high', 0)
                            _oc_fk_low = _oc_fk.get('oi_day_low', 0)

                            # K1: OI at Day-High = institutions actively adding RIGHT NOW
                            if _oc_fk_high > _oc_fk_low > 0:
                                _oc_fk_range = _oc_fk_high - _oc_fk_low
                                _oc_fk_pos = (_oc_fk_oi - _oc_fk_low) / _oc_fk_range  # 0=low, 1=high
                                if _oc_fk_pos >= 0.75:  # Apr 7: tightened 0.70→0.75
                                    _oc_eff_str *= 1.18
                                    _oc_boosts.append(f'FOIDH✓{_oc_fk_pos:.0%}')
                                    _oc_confirms += 1
                                    _oc_K_confirmed = True
                                    _oc_K_strong_confirmed = True
                                elif _oc_fk_pos >= 0.60:  # Apr 7: tightened 0.55→0.60
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
                    if _oc_vel_avg > 0 and _oc_str > _oc_vel_avg * 1.20:  # tightened 1.15→1.20x
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
                        _oc_bw = getattr(_ticker, '_breakout_watcher', None)
                        _oc_vdh_store = getattr(_oc_bw, '_vol_delta_history', None) if _oc_bw else None
                        _oc_vdh = _oc_vdh_store.get(_oc_sym) if _oc_vdh_store is not None else None
                        if _oc_vdh and len(_oc_vdh) >= 5:
                            _oc_vavg = sum(_oc_vdh) / len(_oc_vdh)
                            _oc_vlast = list(_oc_vdh)[-1] if _oc_vdh else 0
                            if _oc_vavg > 0 and _oc_vlast > 0:
                                _oc_vol_surge_ratio = _oc_vlast / _oc_vavg
                                # Check recent 3 ticks for sustained elevation
                                _oc_v_recent = list(_oc_vdh)[-3:]
                                _oc_v_elevated = sum(1 for _vd in _oc_v_recent if _vd >= _oc_vavg * 1.4)
                                if _oc_vol_surge_ratio >= 2.0 and _oc_v_elevated >= 2:  # tightened 1.79→2.0x
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
                _oc_res['_mf_F_strong'] = _oc_F_strong_confirmed
                _oc_res['_mf_H'] = _oc_H_confirmed
                _oc_res['_mf_K'] = _oc_K_confirmed
                _oc_res['_mf_K_strong'] = _oc_K_strong_confirmed
                _oc_res['_mf_M'] = _oc_vol_surge_aligned
                # (N) FII/DII Cash Flow: macro institutional direction
                # FII net buying confirms BUY, FII net selling confirms SELL.
                # This is the single strongest macro institutional signal in Indian markets.
                _oc_N_confirmed = False
                _oc_N_evaluable = bool(_oc_fii_data and _oc_fii_data.get('fii_direction'))  # Apr 15 RCA
                if _oc_fii_data and _oc_fii_data.get('fii_direction'):
                    _oc_fii_dir = _oc_fii_data['fii_direction']
                    _oc_fii_net = _oc_fii_data.get('fii_net', 0)
                    _oc_fii_confirms = (
                        (_oc_dir == 'BUY' and _oc_fii_dir == 'BULLISH') or
                        (_oc_dir == 'SELL' and _oc_fii_dir == 'BEARISH')
                    )
                    _oc_fii_opposes = (
                        (_oc_dir == 'BUY' and _oc_fii_dir == 'BEARISH') or
                        (_oc_dir == 'SELL' and _oc_fii_dir == 'BULLISH')
                    )
                    if _oc_fii_confirms:
                        _oc_eff_str *= 1.10  # FII backing your direction
                        _oc_boosts.append(f'FII✓{_oc_fii_net:+,.0f}Cr')
                        _oc_confirms += 1
                        _oc_N_confirmed = True
                    elif _oc_fii_opposes:
                        _oc_eff_str *= 0.88  # Swimming against FII flow
                        _oc_boosts.append(f'FII✗{_oc_fii_net:+,.0f}Cr')
                    else:
                        # FII NEUTRAL = not opposing but not confirming either
                        _oc_boosts.append(f'FII~{_oc_fii_net:+,.0f}Cr')
                _oc_res['_mf_N'] = _oc_N_confirmed
                # Apr 17 FIX: Store evaluability flags AFTER all factors computed
                # (was crashing with NameError: _oc_N_evaluable used before assignment)
                _oc_res['_mf_F_eval'] = _oc_F_evaluable
                _oc_res['_mf_H_eval'] = _oc_H_evaluable
                _oc_res['_mf_K_eval'] = _oc_K_evaluable
                _oc_res['_mf_N_eval'] = _oc_N_evaluable

                # (B2) Momentum Confluence — 4 signals, ≥2/4 to confirm (anchor factor)
                # Computed natively from OI watcher data. Joins F,H,K in anchor pool.
                _oc_B2_signals = 0
                _oc_B2_tags = []
                # B2-1: Fresh price move (≥0.25% from prev close in either direction)
                if abs(_oc_C_price_chg) >= 0.25:
                    _oc_B2_signals += 1
                    _oc_B2_tags.append('FreshPx')
                # B2-2: Strong OI signal (base strength ≥ 0.45)
                if _oc_str >= 0.45:
                    _oc_B2_signals += 1
                    _oc_B2_tags.append('OIStr')
                # B2-3: Volume surge aligned (Factor M)
                if _oc_vol_surge_aligned:
                    _oc_B2_signals += 1
                    _oc_B2_tags.append('VolSrg')
                # B2-4: Price above open (BUY) / below open (SELL) — VWAP proxy
                _oc_B2_px_vs_open = False
                if _ticker:
                    try:
                        with _ticker._lock:
                            _oc_b2_tok = None
                            for _tk_b2, _tsym_b2 in _ticker._token_to_symbol.items():
                                if _tsym_b2 == _oc_sym:
                                    _oc_b2_tok = _tk_b2
                                    break
                            if _oc_b2_tok:
                                _oc_b2_q = _ticker._quote_cache.get(_oc_b2_tok, {})
                                _oc_b2_ltp = _oc_b2_q.get('last_price', 0)
                                _oc_b2_open = ((_oc_b2_q.get('ohlc', {}) or {}).get('open', 0))
                                if _oc_b2_ltp > 0 and _oc_b2_open > 0:
                                    _oc_B2_px_vs_open = (
                                        (_oc_dir == 'BUY' and _oc_b2_ltp > _oc_b2_open) or
                                        (_oc_dir == 'SELL' and _oc_b2_ltp < _oc_b2_open)
                                    )
                    except Exception:
                        pass
                if _oc_B2_px_vs_open:
                    _oc_B2_signals += 1
                    _oc_B2_tags.append('PxVsOpen')
                _oc_B2_confirmed = _oc_B2_signals >= 2
                _oc_res['_mf_B2'] = _oc_B2_confirmed
                _oc_res['_mf_B2_signals'] = _oc_B2_signals
                _oc_res['_mf_B2_tags'] = ','.join(_oc_B2_tags)
                if _oc_B2_confirmed:
                    _oc_boosts.append(f'B2✓{_oc_B2_signals}/4[{",".join(_oc_B2_tags)}]')
                else:
                    _oc_boosts.append(f'B2✗{_oc_B2_signals}/4')

                # [FIX Mar 24 v2] Cap multiplicative inflation — multiplier product capped at 2.0x base
                _oc_max_boosted = _oc_str * 2.0  # base strength * 2.0 = max allowed
                _oc_eff_str = min(1.0, min(_oc_max_boosted, _oc_eff_str))
                # [FIX Mar 24 v2] SHORT_COVERING / LONG_UNWINDING need ≥0.65 effective to fire
                if _oc_sig in ('SHORT_COVERING', 'LONG_UNWINDING') and _oc_eff_str < 0.65:
                    _oc_eff_str = 0.0  # Will be filtered by min_strength check downstream
                # [FIX Mar 29] LONG_BUILDUP / SHORT_BUILDUP need ≥0.55 effective — prevent weak noise
                if _oc_sig in ('LONG_BUILDUP', 'SHORT_BUILDUP') and _oc_eff_str < 0.55:
                    _oc_eff_str = 0.0
                _oi_candidates[_oci] = (_oc_sym, _oc_dir, _oc_sig, _oc_eff_str, _oc_res, _oc_skip_fire)
                # Store boost tags + confirm count for logging
                _oc_res['_quality_boosts'] = ' '.join(_oc_boosts) if _oc_boosts else ''
                _oc_res['_confirm_count'] = _oc_confirms
            # Apr 15 RCA: Filter out skip-for-fire candidates AFTER enrichment
            # All candidates now have _mf_* flags set on their _oc_res dicts (shared with layer1_oi).
            # Only fire-eligible candidates go through selection.
            _oi_fireable = [c for c in _oi_candidates if not c[5]]
            # Sort by (confirm_count DESC, effective_strength DESC) — conviction first, strength second
            _oi_fireable.sort(key=lambda x: (x[4].get('_confirm_count', 0), x[3]), reverse=True)
            _oi_max_fire = 3  # Fire up to 3 best-convicted candidates per cycle
            _oi_placed_count = 0
            _oi_top_confirms = _oi_fireable[0][4].get('_confirm_count', 0) if _oi_fireable else 0
            t._wlog(f"🔬 OI_WATCHER: {len(_oi_fireable)} candidates ({len(_oi_candidates)} enriched), "
                       f"top conviction={_oi_top_confirms}/{t._oi_min_confirmations} factors")
            for _oi_rank, _oi_top in enumerate(_oi_fireable[:_oi_max_fire]):
                if t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
                    break
                _oi_sym, _oi_dir, _oi_sig, _oi_str, _oi_res, _ = _oi_top
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

                # ── CONVICTION FACTOR GATE ──
                # Anchor pool: F (Futures OI), H (OI concentration), K (Futures conviction), N (FII/DII)
                #   → at least 1 of 4 must confirm (institutional/futures validation)
                # Rest pool: A, B2, C, D, E, G, I, L, M (9 factors)
                #   → at least 4 of 9 must confirm (breadth confirmation)
                _oi_mf_C = _oi_res.get('_mf_C', False)
                _oi_mf_C_chg = _oi_res.get('_mf_C_price_chg', 0.0)
                _oi_mf_F = _oi_res.get('_mf_F', False)
                _oi_mf_H = _oi_res.get('_mf_H', False)
                _oi_mf_K = _oi_res.get('_mf_K', False)
                _oi_mf_M = _oi_res.get('_mf_M', False)
                _oi_mf_N = _oi_res.get('_mf_N', False)
                _oi_mf_B2 = _oi_res.get('_mf_B2', False)
                # Anchor strength tiers: strong vs normal confirmation
                _oi_mf_F_strong = _oi_res.get('_mf_F_strong', False)  # FUT✓✓ (|buildup|≥0.75)
                _oi_mf_K_strong = _oi_res.get('_mf_K_strong', False)  # FOIDH✓ (OI≥85% day range)
                # H has single tier — always STRONG when confirmed (ATM ≤2%)
                _oi_mf_labels = []
                if _oi_mf_C: _oi_mf_labels.append('C')
                if _oi_mf_F: _oi_mf_labels.append('F✦' if _oi_mf_F_strong else 'F')
                if _oi_mf_H: _oi_mf_labels.append('H✦')  # always strong when confirmed
                if _oi_mf_K: _oi_mf_labels.append('K✦' if _oi_mf_K_strong else 'K')
                if _oi_mf_M: _oi_mf_labels.append('M')
                if _oi_mf_N: _oi_mf_labels.append('N')
                if _oi_mf_B2: _oi_mf_labels.append('B2')
                _oi_anchor_count = sum([_oi_mf_F, _oi_mf_H, _oi_mf_K, _oi_mf_N])
                _oi_anchor_strong = sum([_oi_mf_F_strong, _oi_mf_H, _oi_mf_K_strong])  # H=always strong
                # Apr 15 RCA: Track evaluable anchor count — don't block on missing data
                _oi_eval_F = _oi_res.get('_mf_F_eval', False)
                _oi_eval_H = _oi_res.get('_mf_H_eval', False)
                _oi_eval_K = _oi_res.get('_mf_K_eval', False)
                _oi_eval_N = _oi_res.get('_mf_N_eval', False)
                _oi_evaluable = sum([_oi_eval_F, _oi_eval_H, _oi_eval_K, _oi_eval_N])
                _oi_rest_count = _oi_confirms - _oi_anchor_count
                # B2 confirmed counts toward rest pool (breadth), not anchor
                if _oi_mf_B2 and not any([_oi_mf_F, _oi_mf_H, _oi_mf_K, _oi_mf_N]):
                    pass  # B2 alone cannot satisfy anchor — must have futures validation
                # Gate 1: At least 1 of {F, H, K, N} must confirm (anchor — institutional/futures)
                # Apr 15 RCA: Skip if no anchor factors were evaluable (no data ≠ data says no)
                if _oi_evaluable == 0:
                    t._wlog(f"  ⚠️ OI_WATCHER ANCHOR SKIP: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"0/4 anchor factors evaluable (no institutional data), bypassing "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
                               f"[{_oi_res.get('_quality_boosts', '')}]")
                elif _oi_anchor_count < 1:
                    t._wlog(f"  ⛔ OI_WATCHER ANCHOR GATE: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"{_oi_anchor_count}/{_oi_evaluable} of F,H,K,N confirmed (need ≥1 anchor) "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
                               f"[{_oi_res.get('_quality_boosts', '')}]")
                    continue
                # Gate 2: At least 4 from the remaining 9 factors incl B2 (breadth)
                if _oi_rest_count < 4:
                    t._wlog(f"  ⛔ OI_WATCHER BREADTH GATE: {_oi_sym.replace('NSE:', '')} "
                               f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                               f"only {_oi_rest_count}/9 rest-pool factors confirm (need ≥4) "
                               f"anchor={_oi_anchor_count}/4 "
                               f"[confirmed={','.join(_oi_mf_labels)}] "
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

                # Anchor-based lot sizing: more anchors = higher conviction = bigger size
                _oi_anchor_lot_mult = 1.0
                if _oi_anchor_count >= 3:
                    _oi_anchor_lot_mult = 3.0
                elif _oi_anchor_count >= 2:
                    _oi_anchor_lot_mult = 1.5

                t._wlog(f"  ✅ OI_WATCHER HIGH CONVICTION: {_oi_sym.replace('NSE:', '')} "
                           f"{_oi_sig} str={_oi_str:.3f} dir={_oi_dir} — "
                           f"{_oi_confirms}/{t._oi_min_confirmations} factors confirm "
                           f"anchor={_oi_anchor_count}/4({_oi_anchor_strong}✦) "
                           f"price={_oi_price_delta_str} "
                           f"lots={_oi_anchor_lot_mult}x "
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
                            lot_multiplier=_oi_anchor_lot_mult,
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
        # Apr 21: collect up to (kite_cap + dhan_cap) — first N go to Kite pool,
        # next M go to Dhan pool (separate rate bucket, runs concurrently).
        _n_kite = t._oi_aggr_max_symbols
        _n_dhan = getattr(t, '_oi_aggr_dhan_symbols', 0)
        _total_cap = _n_kite + _n_dhan
        for _s, _abs_chg, _chg in _syms_with_change:
            if _abs_chg < 0.3:
                break  # Below 0.3% change — not worth scanning
            if _s in t._oi_watcher_fired_this_session:
                continue
            if t.tools.is_symbol_in_active_trades(_s):
                continue
            _scan_syms.append(_s)
            if len(_scan_syms) >= _total_cap:
                break

        if not _scan_syms:
            return

        # Split universe across two independent rate buckets
        _kite_syms = _scan_syms[:_n_kite]
        _dhan_syms = _scan_syms[_n_kite:_n_kite + _n_dhan] if _n_dhan > 0 else []

        # Parallel OI fetch — DUAL-SOURCE SHARDING (Apr 21):
        #   • Kite pool: 8 workers, shares Kite's ~10 req/s bucket w/ ticker+exit_manager+watcher
        #   • Dhan pool: 3 workers, DhanHQ enforces 3s global serial throttle (more workers just queue)
        # Two pools run concurrently — Kite handles top-33 movers, Dhan handles next-15,
        # so a single cycle now covers 48 syms without increasing Kite load.
        from concurrent.futures import ThreadPoolExecutor as _OITP, as_completed as _oi_done
        _oi_raw = {}
        _fetch_start = _oiag_t.time()
        # Apr 21 FIX: do NOT use `with` ThreadPoolExecutor — its __exit__ blocks
        # on shutdown(wait=True) and waits for ALL running analyze() calls to
        # finish even after as_completed() timed out, which can stretch a 80s
        # timeout into a 280s stall and starve the next scan cycle.
        _ex_kite = _OITP(max_workers=8, thread_name_prefix='oi-aggr-kite')
        _ex_dhan = _OITP(max_workers=3, thread_name_prefix='oi-aggr-dhan') if _dhan_syms else None
        _timed_out = False
        _futs = {}
        try:
            for _s in _kite_syms:
                _futs[_ex_kite.submit(t._oi_analyzer.analyze, _s)] = _s
            if _ex_dhan is not None and hasattr(t._oi_analyzer, 'analyze_dhan_only'):
                for _s in _dhan_syms:
                    _futs[_ex_dhan.submit(t._oi_analyzer.analyze_dhan_only, _s)] = _s
            elif _dhan_syms:
                # Fallback: if analyze_dhan_only missing, route Dhan shard through Kite pool
                for _s in _dhan_syms:
                    _futs[_ex_kite.submit(t._oi_analyzer.analyze, _s)] = _s
            # Apr 21: floor 120s, scale with n above that — Dhan 15 syms × 3s serial
            # ≈ 45s worst-case, runs fully within the Kite window.
            _timeout_s = max(120, len(_kite_syms) * 3)
            try:
                for _f in _oi_done(_futs, timeout=_timeout_s):
                    _sym = _futs[_f]
                    try:
                        _res = _f.result()
                        if _res:
                            _oi_raw[_sym] = _res
                    except Exception as e:
                        t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_result]: {e}")
            except Exception as e:
                _timed_out = True
                _pending = sum(1 for _f in _futs if not _f.done())
                t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_timeout]: {e} — returning {len(_oi_raw)}/{len(_futs)} after {_timeout_s}s, cancelling {_pending} pending")
        finally:
            # cancel_futures=True cancels queued tasks; in-flight ones continue
            # but we do NOT wait for them — the cycle must move on.
            for _ex_ in (_ex_kite, _ex_dhan):
                if _ex_ is None:
                    continue
                try:
                    _ex_.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python <3.9 fallback — at least don't block
                    for _f in list(_futs.keys()):
                        _f.cancel()
                    _ex_.shutdown(wait=False)
        _fetch_dur = _oiag_t.time() - _fetch_start

        # Score candidates: extract signal, strength, track history, detect acceleration
        _candidates = []
        _now_ts = _oiag_t.time()

        # (N) FII/DII Cash Flow — macro institutional direction (fetched once, cached 5 min)
        _ag_fii_data = {}
        try:
            from nse_oi_fetcher import get_nse_oi_fetcher
            _ag_fii_data = get_nse_oi_fetcher().fetch_fii_dii()
        except Exception:
            pass

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
            # Apr 15 RCA: Track evaluability — True when data exists to evaluate the factor
            _ag_mf_F_eval = False
            _ag_mf_H_eval = False
            _ag_mf_K_eval = False
            _ag_mf_N_eval = False
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
                _ag_C_min = getattr(t, '_oi_aggr_min_price_delta', 0.15)  # Apr 9: tightened 0.10→0.15%
                _ag_price_agrees = (
                    (_dir == 'BUY' and _ag_chg >= _ag_C_min) or
                    (_dir == 'SELL' and _ag_chg <= -_ag_C_min)
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
                    (_dir == 'BUY' and _ag_sec_chg > 0.40) or
                    (_dir == 'SELL' and _ag_sec_chg < -0.40)
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
            # [FIX Apr 6] Read from _futures_oi_data (raw parquet), NOT _cycle_ml_results
            _ag_fut_buildup = 0
            try:
                _ag_foi = getattr(t, '_futures_oi_data', None) or {}
                _ag_foi_df = _ag_foi.get(_sym)
                if _ag_foi_df is not None and len(_ag_foi_df) > 0:
                    _ag_fut_buildup = float(_ag_foi_df.iloc[-1].get('fut_oi_buildup', 0))
                    _ag_mf_F_eval = True  # Apr 15 RCA: futures data exists
            except Exception:
                _ag_fut_buildup = 0
            if _ag_fut_buildup:
                _ag_fut_agrees = (
                    (_dir == 'BUY' and _ag_fut_buildup > 0) or
                    (_dir == 'SELL' and _ag_fut_buildup < 0)
                )
                _ag_fut_strong = abs(_ag_fut_buildup) >= 0.75
                _ag_fut_moderate = abs(_ag_fut_buildup) >= 0.30  # Apr 9: tightened — need ≥0.30 to confirm F
                if _ag_fut_agrees and _ag_fut_strong:
                    _eff_str *= 1.15
                    _ag_boosts.append('FUT✓✓')
                    _ag_mf_F = True
                elif _ag_fut_agrees and _ag_fut_moderate:
                    _eff_str *= 1.08
                    _ag_boosts.append('FUT✓')
                    _ag_mf_F = True
                elif _ag_fut_agrees:
                    _eff_str *= 1.04  # Apr 9: weak futures — small boost, no F confirm
                    _ag_boosts.append(f'FUT~{abs(_ag_fut_buildup):.2f}')
                elif not _ag_fut_agrees and _ag_fut_strong:
                    _eff_str *= 0.85
                    _ag_boosts.append('FUT✗✗')

            # (G) Live Futures Basis: premium/discount from ticker cache (0 API calls)
            if _ticker:
                try:
                    _ag_fut_data = _ticker.get_futures_oi(_sym)
                    if _ag_fut_data and _ag_fut_data.get('ltp', 0) > 0:
                        _ag_mf_F_eval = True  # Apr 15 RCA: basis data exists
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
                                (_dir == 'BUY' and _ag_basis_pct > 0.10) or   # Apr 7: tightened 0.05→0.10
                                (_dir == 'SELL' and _ag_basis_pct < -0.10)      # Apr 7: tightened
                            )
                            _ag_basis_disagrees = (
                                (_dir == 'BUY' and _ag_basis_pct < -0.10) or
                                (_dir == 'SELL' and _ag_basis_pct > 0.10)
                            )
                            if _ag_basis_agrees:
                                _eff_str *= 1.10
                                _ag_boosts.append(f'BASIS✓{_ag_basis_pct:+.2f}%')
                                # [FIX Apr 6] Basis agreement = futures anchor fallback
                                if not _ag_mf_F:
                                    _ag_mf_F = True
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
                        _ag_mf_H_eval = True  # Apr 15 RCA: strike data available
                        _ag_stk_dist = abs(_ag_top_stk - _ag_spot) / _ag_spot * 100
                        if _ag_stk_dist <= 2.5:  # Apr 7: tightened 3.0→2.5%
                            _eff_str *= 1.14
                            _ag_boosts.append(f'ATM✓{_ag_stk_dist:.1f}%')
                            _ag_mf_H = True
                        elif _ag_stk_dist >= 5.0:
                            _eff_str *= 0.92
                            _ag_boosts.append(f'OTM✗{_ag_stk_dist:.1f}%')

            # (I) PCR Shift Rate: rate of PCR change
            _ag_pcr_rate = _res.get('pcr_shift_rate', 0)
            if abs(_ag_pcr_rate) > 0.010:  # tightened 0.005→0.010
                _ag_pcr_confirms = (
                    (_dir == 'BUY' and _ag_pcr_rate >= 0.015) or   # tightened 0.01→0.015
                    (_dir == 'SELL' and _ag_pcr_rate <= -0.015)
                )
                _ag_pcr_opposes = (
                    (_dir == 'BUY' and _ag_pcr_rate <= -0.015) or
                    (_dir == 'SELL' and _ag_pcr_rate >= 0.015)
                )
                if _ag_pcr_confirms:
                    _eff_str *= 1.08
                    _ag_boosts.append(f'PCR↗{_ag_pcr_rate:+.3f}')
                elif _ag_pcr_opposes:
                    _eff_str *= 0.92
                    _ag_boosts.append(f'PCR↘{_ag_pcr_rate:+.3f}')

            # (J) Volume PCR: REMOVED — PCR extremes gate eliminated

            # (K) Futures Conviction Boost: OI Day-High + Order Book Imbalance (BOOST-ONLY)
            # Same as OI_WATCHER Factor K — uses Kite WebSocket futures data (0 API calls).
            if _ticker:
                try:
                    _ag_fk = _ticker.get_futures_oi(_sym)
                    if _ag_fk and _ag_fk.get('oi', 0) > 0:
                        _ag_mf_K_eval = True  # Apr 15 RCA: futures conviction data available
                        _ag_fk_oi = _ag_fk['oi']
                        _ag_fk_high = _ag_fk.get('oi_day_high', 0)
                        _ag_fk_low = _ag_fk.get('oi_day_low', 0)

                        # K1: OI at Day-High = institutions actively adding
                        if _ag_fk_high > _ag_fk_low > 0:
                            _ag_fk_range = _ag_fk_high - _ag_fk_low
                            _ag_fk_pos = (_ag_fk_oi - _ag_fk_low) / _ag_fk_range
                            if _ag_fk_pos >= 0.75:  # Apr 7: tightened 0.70→0.75
                                _eff_str *= 1.18
                                _ag_boosts.append(f'FOIDH✓{_ag_fk_pos:.0%}')
                                _ag_mf_K = True
                            elif _ag_fk_pos >= 0.60:  # Apr 7: tightened 0.55→0.60
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
                    _ag_bw = getattr(_ticker, '_breakout_watcher', None)
                    _ag_vdh_store = getattr(_ag_bw, '_vol_delta_history', None) if _ag_bw else None
                    _ag_vdh = _ag_vdh_store.get(_sym) if _ag_vdh_store is not None else None
                    if _ag_vdh and len(_ag_vdh) >= 5:
                        _ag_vavg = sum(_ag_vdh) / len(_ag_vdh)
                        _ag_vlast = list(_ag_vdh)[-1] if _ag_vdh else 0
                        if _ag_vavg > 0 and _ag_vlast > 0:
                            _ag_vol_surge_ratio = _ag_vlast / _ag_vavg
                            _ag_v_recent = list(_ag_vdh)[-3:]
                            _ag_v_elevated = sum(1 for _vd in _ag_v_recent if _vd >= _ag_vavg * 1.4)
                            if _ag_vol_surge_ratio >= 2.2 and _ag_v_elevated >= 2:  # Apr 9: tightened 2.0→2.2x
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
                            elif _ag_vol_surge_ratio >= 1.60 and _ag_v_elevated >= 1:  # Apr 9: tightened 1.46→1.60x
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
            # [FIX Mar 29] LONG_BUILDUP / SHORT_BUILDUP need ≥0.55 effective — prevent weak noise
            if _sig in ('LONG_BUILDUP', 'SHORT_BUILDUP') and _eff_str < 0.55:
                continue
            # [FIX Mar 24 v2] Store mandatory factor results for AGGR gate
            _res['_ag_mf_C'] = _ag_mf_C
            _res['_ag_mf_F'] = _ag_mf_F
            _res['_ag_mf_H'] = _ag_mf_H
            _res['_ag_mf_K'] = _ag_mf_K
            _res['_ag_mf_M'] = _ag_mf_M
            # Apr 15 RCA: Store evaluability flags
            _res['_ag_mf_F_eval'] = _ag_mf_F_eval
            _res['_ag_mf_H_eval'] = _ag_mf_H_eval
            _res['_ag_mf_K_eval'] = _ag_mf_K_eval

            # (N) FII/DII Cash Flow: macro institutional direction
            _ag_mf_N = False
            _ag_mf_N_eval = bool(_ag_fii_data and _ag_fii_data.get('fii_direction'))  # Apr 15 RCA
            if _ag_fii_data and _ag_fii_data.get('fii_direction'):
                _ag_fii_dir = _ag_fii_data['fii_direction']
                _ag_fii_net = _ag_fii_data.get('fii_net', 0)
                _ag_fii_confirms = (
                    (_dir == 'BUY' and _ag_fii_dir == 'BULLISH') or
                    (_dir == 'SELL' and _ag_fii_dir == 'BEARISH')
                )
                _ag_fii_opposes = (
                    (_dir == 'BUY' and _ag_fii_dir == 'BEARISH') or
                    (_dir == 'SELL' and _ag_fii_dir == 'BULLISH')
                )
                if _ag_fii_confirms:
                    _eff_str *= 1.10
                    _ag_boosts.append(f'FII✓{_ag_fii_net:+,.0f}Cr')
                    _ag_mf_N = True
                elif _ag_fii_opposes:
                    _eff_str *= 0.88
                    _ag_boosts.append(f'FII✗{_ag_fii_net:+,.0f}Cr')
                else:
                    # FII NEUTRAL = not opposing but not confirming either
                    _ag_boosts.append(f'FII~{_ag_fii_net:+,.0f}Cr')
            _res['_ag_mf_N'] = _ag_mf_N
            _res['_ag_mf_N_eval'] = _ag_mf_N_eval  # Apr 15 RCA

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
            t._wlog(f"🔬 OI_AGGR: Scanned {len(_kite_syms)}K+{len(_dhan_syms)}D movers, fetched {len(_oi_raw)} OI "
                       f"({_fetch_dur:.1f}s) — no LB/SB candidates")
            return

        # Sort: accelerating first, then by strength
        _candidates.sort(key=lambda x: (x[6], x[3]), reverse=True)

        t._wlog(f"🔬 OI_AGGR: Scanned {len(_kite_syms)}K+{len(_dhan_syms)}D movers → {len(_candidates)} candidates "
                   f"({_fetch_dur:.1f}s)")
        for _c in _candidates[:5]:
            _tag = '🚀ACCEL' if _c[6] else '⚡'
            _ag_qt = _c[4].get('_quality_boosts', '')
            t._wlog(f"  {_tag} {_c[0].replace('NSE:', '')} {_c[2]} str={_c[3]:.3f} "
                       f"dir={_c[1]} part={_c[4].get('oi_participant_id', '?')} "
                       f"quality=[{_ag_qt}] [{_c[5]}]")

        # Fire top candidate — 1 per cycle for quality (same as Path 1)
        _placed_this_scan = 0
        _max_fire = 3  # [FIX Apr 2] Up to 3 per cycle — conviction gate ensures quality

        for _c in _candidates:
            if _placed_this_scan >= _max_fire:
                break
            if t._oi_watcher_total_placed >= t._oi_watcher_max_per_day:
                break

            _sym, _dir, _sig, _str, _res, _reason, _is_accel, _accel_delta = _c
            _pcr = _res.get('pcr_oi', 1.0)
            _bias = _res.get('flow_bias', 'NEUTRAL')
            _part_id = _res.get('oi_participant_id', 'UNKNOWN')

            # [FIX Apr 9] CONVICTION GATE for AGGR path
            # Breadth technicals are tighter so ≥2/6 is sufficient:
            #   1. At least 1/4 of F,H,K,N (anchor — institutional/futures validation)
            #   2. At least 2/6 total mandatory factors (C,F,H,K,M,N)
            # This eliminates the #1 source of OI_WATCHER losses: AGGR firing
            # on weak OI signals with no futures/volume confirmation.
            _ag_mf_C = _res.get('_ag_mf_C', False)
            _ag_mf_F = _res.get('_ag_mf_F', False)
            _ag_mf_H = _res.get('_ag_mf_H', False)
            _ag_mf_K = _res.get('_ag_mf_K', False)
            _ag_mf_M = _res.get('_ag_mf_M', False)
            _ag_mf_N = _res.get('_ag_mf_N', False)
            _ag_mf_anchor = sum([_ag_mf_F, _ag_mf_H, _ag_mf_K, _ag_mf_N])
            _ag_mf_total = sum([_ag_mf_C, _ag_mf_F, _ag_mf_H, _ag_mf_K, _ag_mf_M, _ag_mf_N])
            # Apr 15 RCA: Track evaluable anchor count — don't block on missing data
            _ag_eval_F = _res.get('_ag_mf_F_eval', False)
            _ag_eval_H = _res.get('_ag_mf_H_eval', False)
            _ag_eval_K = _res.get('_ag_mf_K_eval', False)
            _ag_eval_N = _res.get('_ag_mf_N_eval', False)
            _ag_evaluable = sum([_ag_eval_F, _ag_eval_H, _ag_eval_K, _ag_eval_N])
            _ag_mf_labels = []
            if _ag_mf_C: _ag_mf_labels.append('C')
            if _ag_mf_F: _ag_mf_labels.append('F')
            if _ag_mf_H: _ag_mf_labels.append('H')
            if _ag_mf_K: _ag_mf_labels.append('K')
            if _ag_mf_M: _ag_mf_labels.append('M')
            if _ag_mf_N: _ag_mf_labels.append('N')
            # Gate 1: Anchor — at least 1 of F/H/K/N (institutional/futures validation)
            # Apr 15 RCA: Skip if no anchor factors were evaluable (no data ≠ data says no)
            if _ag_evaluable == 0:
                t._wlog(f"  ⚠️ OI_AGGR ANCHOR SKIP: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} — "
                           f"0/4 anchor factors evaluable (no institutional data), bypassing")
            elif _ag_mf_anchor < 1:
                t._wlog(f"  ⛔ OI_AGGR ANCHOR BLOCK: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} — "
                           f"{_ag_mf_anchor}/{_ag_evaluable} of F,H,K,N confirmed (need ≥1 anchor) "
                           f"[confirmed={','.join(_ag_mf_labels)}]")
                continue
            # Gate 2: Breadth — 2/6 for all entries (Apr 15, 2026)
            _ag_min_breadth = 2
            if _ag_mf_total < _ag_min_breadth:
                t._wlog(f"  ⛔ OI_AGGR BREADTH BLOCK: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} accel={_is_accel} — "
                           f"only {_ag_mf_total}/6 factors confirm (need ≥{_ag_min_breadth}) "
                           f"anchor={_ag_mf_anchor}/4 "
                           f"[confirmed={','.join(_ag_mf_labels)}]")
                continue
            # Anchor-based lot sizing: more anchors = higher conviction = bigger size
            _ag_anchor_lot_mult = 1.0
            if _ag_mf_anchor >= 3:
                _ag_anchor_lot_mult = 3.0
            elif _ag_mf_anchor >= 2:
                _ag_anchor_lot_mult = 1.5

            t._wlog(f"  ✅ OI_AGGR CONVICTION OK: {_sym.replace('NSE:', '')} "
                       f"{_sig} str={_str:.3f} dir={_dir} — "
                       f"{_ag_mf_total}/6 factors, anchor={_ag_mf_anchor}/4 "
                       f"lots={_ag_anchor_lot_mult}x "
                       f"[confirmed={','.join(_ag_mf_labels)}]")

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
            # Adaptive confirmation gate: strong/accelerating setups confirm faster
            # with lower required price delta; weak setups stay stricter.
            # Apr 15 v2: relaxed again — base 35s/0.15%, lower floor 0.08%
            _ag_confirm_secs = float(getattr(t, '_oi_confirm_seconds', 35))
            _ag_min_delta = float(getattr(t, '_oi_confirm_min_price_delta', 0.15))
            if _is_accel:
                _ag_confirm_secs *= 0.70
                _ag_min_delta *= 0.70
            if _ag_mf_M:  # Volume confirmation
                _ag_confirm_secs *= 0.85
                _ag_min_delta *= 0.80
            if _ag_mf_anchor >= 2:
                _ag_confirm_secs *= 0.85
                _ag_min_delta *= 0.85
            if _str >= 0.90:
                _ag_confirm_secs *= 0.70
                _ag_min_delta *= 0.75
            elif _str >= 0.75:
                _ag_confirm_secs *= 0.85
                _ag_min_delta *= 0.85
            elif _str < 0.60:
                _ag_confirm_secs *= 1.20
                _ag_min_delta *= 1.15
            _ag_confirm_secs = max(20.0, min(90.0, _ag_confirm_secs))
            _ag_min_delta = max(0.08, min(0.30, _ag_min_delta))
            if _ag_pending is None:
                t._oi_pending_confirm[_sym] = {
                    'ts': _ag_now_ts, 'direction': _dir,
                    'strength': _str, 'signal': _sig, 'source': 'OI_AGGR',
                    'spot_price': _ag_spot_now,
                    'confirm_seconds': _ag_confirm_secs,
                    'min_price_delta': _ag_min_delta,
                }
                t._wlog(f"  ⏳ OI_AGGR PENDING: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} dir={_dir} "
                           f"spot={_ag_spot_now:.2f} — "
                           f"waiting {_ag_confirm_secs:.0f}s, "
                           f"need Δ{_ag_min_delta:.2f}% confirmation")
                continue
            _ag_elapsed = _ag_now_ts - _ag_pending['ts']
            if _ag_pending['direction'] != _dir:
                t._oi_pending_confirm[_sym] = {
                    'ts': _ag_now_ts, 'direction': _dir,
                    'strength': _str, 'signal': _sig, 'source': 'OI_AGGR',
                    'spot_price': _ag_spot_now,
                    'confirm_seconds': _ag_confirm_secs,
                    'min_price_delta': _ag_min_delta,
                }
                t._wlog(f"  🔄 OI_AGGR RESET: {_sym.replace('NSE:', '')} "
                           f"direction flipped {_ag_pending['direction']}→{_dir} — "
                           f"restarting {_ag_confirm_secs:.0f}s / Δ{_ag_min_delta:.2f}% wait")
                continue
            _req_secs = float(_ag_pending.get('confirm_seconds', t._oi_confirm_seconds))
            _req_delta = float(_ag_pending.get('min_price_delta', t._oi_confirm_min_price_delta))
            if _ag_elapsed < _req_secs:
                t._wlog(f"  ⏳ OI_AGGR WAITING: {_sym.replace('NSE:', '')} "
                           f"{_sig} str={_str:.3f} — {_ag_elapsed:.0f}s / "
                           f"{_req_secs:.0f}s elapsed (Δ need {_req_delta:.2f}%)")
                continue
            # Time elapsed — check PRICE DELTA
            _ag_spot_entry = _ag_pending.get('spot_price', 0)
            _ag_price_ok = True
            _ag_price_delta_pct = 0.0
            if _ag_spot_entry > 0 and _ag_spot_now > 0:
                _ag_price_delta_pct = ((_ag_spot_now - _ag_spot_entry) / _ag_spot_entry) * 100
                if _dir == 'BUY' and _ag_price_delta_pct < _req_delta:
                    _ag_price_ok = False
                elif _dir == 'SELL' and _ag_price_delta_pct > -_req_delta:
                    _ag_price_ok = False
            if not _ag_price_ok:
                t._wlog(f"  ❌ OI_AGGR PRICE REJECT: {_sym.replace('NSE:', '')} "
                           f"{_sig} dir={_dir} — OI held {_ag_elapsed:.0f}s but "
                           f"price Δ={_ag_price_delta_pct:+.2f}% (need {'+' if _dir == 'BUY' else '-'}{_req_delta:.2f}%) "
                           f"— OI trap, skipping")
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
                        setup_type='OI_WATCHER_AGGR',
                        lot_multiplier=_ag_anchor_lot_mult,
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
