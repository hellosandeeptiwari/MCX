"""Capital Swap Engine — extracted from autonomous_trader.py"""
from config import CAPITAL_SWAP


class CapitalSwapEngine:
    def __init__(self, trader):
        self.trader = trader

    def find_eviction_candidate(self, incoming_setup_type: str) -> dict | None:
        """Find the worst stale position eligible for eviction when a higher-priority trade arrives.
        
        Returns a dict with eviction candidate info, or None if no suitable candidate.
        Priority hierarchy (from CAPITAL_SWAP config):
          OI_WATCHER(9) > SPIKE(8) > GRIND(7) > VOLUME_SURGE(6) > NEW_DAY(5) > SNIPER(4) > ELITE(3) > GMM(2)
        
        Eviction criteria:
          1. Position held >= min_stale_minutes (45 min default)
          2. Position is NOT profitable (R-multiple <= 0 or P&L <= 0)
          3. Unrealized loss < max_evict_loss_pct (don't evict deep losers — let SL handle)
          4. Incoming trade priority > position's setup priority
        """
        from datetime import datetime

        t = self.trader
        cfg = CAPITAL_SWAP
        if not cfg.get('enabled', False):
            return None
        
        # Check exposure threshold
        active_positions = [p for p in t.tools.paper_positions if p.get('status', 'OPEN') == 'OPEN']
        if not active_positions:
            return None
        
        total_exposure = sum(
            p.get('max_risk', p.get('avg_price', 0) * p.get('quantity', 0))
            if p.get('is_iron_condor', False)
            else p.get('avg_price', 0) * p.get('quantity', 0)
            for p in active_positions
            if not p.get('is_sniper', False)
        )
        exposure_pct = (total_exposure / t.capital) * 100 if t.capital > 0 else 0
        if exposure_pct < cfg.get('exposure_threshold_pct', 85.0):
            return None  # Enough capital available, no need to evict
        
        # Get incoming trade priority
        priority_tiers = cfg.get('priority_tiers', {})
        incoming_priority = priority_tiers.get(incoming_setup_type, cfg.get('default_priority', 1))
        
        min_stale_min = cfg.get('min_stale_minutes', 45)
        min_hold_min = cfg.get('min_hold_minutes', 30)
        max_loss_pct = cfg.get('max_evict_loss_pct', 15.0)
        
        now = datetime.now()
        candidates = []
        
        for pos in active_positions:
            sym = pos.get('symbol', '')
            setup = pos.get('setup_type', '')
            pos_priority = priority_tiers.get(setup, cfg.get('default_priority', 1))
            
            # Incoming must be strictly higher priority
            if incoming_priority <= pos_priority:
                continue
            
            # Check hold time
            ts_str = pos.get('timestamp', '')
            if not ts_str:
                continue
            try:
                entry_dt = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            hold_minutes = (now - entry_dt).total_seconds() / 60
            
            if hold_minutes < min_hold_min:
                continue  # Too fresh to evict
            
            # Check if stale (held long enough with no progress)
            # Use exit_manager state for R-multiple / candles
            em_state = t.exit_manager.get_trade_state(sym)
            if not em_state:
                continue
            
            candles_held = em_state.candles_since_entry
            r_multiple = 0.0
            if em_state.entry_price > 0 and em_state.initial_sl > 0:
                risk = abs(em_state.entry_price - em_state.initial_sl)
                if risk > 0:
                    if em_state.side == 'BUY':
                        current_ltp = em_state.highest_price if em_state.highest_price > 0 else em_state.entry_price
                        # Use latest price from premium history if available
                        if em_state.premium_history:
                            current_ltp = em_state.premium_history[-1]
                        r_multiple = (current_ltp - em_state.entry_price) / risk
                    else:
                        current_ltp = em_state.entry_price  # Simplified for SELL
                        if em_state.premium_history:
                            current_ltp = em_state.premium_history[-1]
                        r_multiple = (em_state.entry_price - current_ltp) / risk
            
            # Must be stale: held >= min_stale_minutes AND not profitable
            if hold_minutes < min_stale_min:
                continue
            if r_multiple > 0.10:
                continue  # Position is profitable — don't evict
            
            # Check unrealized loss isn't too deep
            entry_price = pos.get('avg_price', 0)
            if entry_price > 0 and em_state.premium_history:
                current_prem = em_state.premium_history[-1]
                loss_pct = abs(entry_price - current_prem) / entry_price * 100
                if loss_pct > max_loss_pct:
                    continue  # Deep loser — let SL handle
            
            # Compute eviction score: lower = worse position = better eviction target
            # Staler + lower R + lower priority = evict first
            evict_score = (r_multiple * 100) - (hold_minutes * 0.1) + (pos_priority * 10)
            
            candidates.append({
                'symbol': sym,
                'underlying': pos.get('underlying', ''),
                'setup_type': setup,
                'priority': pos_priority,
                'hold_minutes': hold_minutes,
                'r_multiple': r_multiple,
                'candles_held': candles_held,
                'evict_score': evict_score,
                'position': pos,
            })
        
        if not candidates:
            return None
        
        # Pick the worst candidate (lowest evict_score)
        candidates.sort(key=lambda c: c['evict_score'])
        best = candidates[0]
        
        t._wlog(f"\n🔄 CAPITAL SWAP: Found eviction candidate")
        t._wlog(f"   Evict: {best['symbol']} ({best['setup_type']}, priority={best['priority']})")
        t._wlog(f"   Held: {best['hold_minutes']:.0f}min, R={best['r_multiple']:.2f}, candles={best['candles_held']}")
        t._wlog(f"   For: {incoming_setup_type} (priority={incoming_priority})")
        t._wlog(f"   Exposure: {exposure_pct:.1f}% (threshold={cfg.get('exposure_threshold_pct', 85.0)}%)")
        
        return best

    def execute_eviction(self, candidate: dict, incoming_reason: str) -> bool:
        """Force-exit an eviction candidate to free capital for a higher-priority trade.
        
        Returns True if eviction succeeded.
        """
        from config import calc_brokerage

        t = self.trader
        sym = candidate['symbol']
        pos = candidate['position']
        
        try:
            # Get current LTP from ticker
            quotes = {}
            ticker = getattr(t.tools, 'ticker', None)
            if ticker:
                quotes = ticker.get_ws_quotes() or {}
            
            ltp = quotes.get(sym, {}).get('last_price', 0)
            if ltp <= 0:
                # Fallback: try REST quote
                try:
                    _q = t.tools.kite.quote([sym])
                    ltp = _q.get(sym, {}).get('last_price', 0)
                except Exception as e:
                    t._wlog(f"⚠️ FALLBACK [capital_swap/execute_eviction]: {e}")
            
            if ltp <= 0:
                t._wlog(f"   ❌ EVICTION FAILED: No LTP for {sym}")
                return False
            
            # Calculate P&L
            entry_price = pos.get('avg_price', 0)
            quantity = pos.get('quantity', 0)
            side = pos.get('side', 'BUY')
            if side == 'BUY':
                pnl = (ltp - entry_price) * quantity
            else:
                pnl = (entry_price - ltp) * quantity
            pnl -= calc_brokerage(entry_price, ltp, quantity)
            
            # Update trade status (this handles live exit order + ledger)
            exit_detail = {
                'candles_held': candidate.get('candles_held', 0),
                'r_multiple_achieved': candidate.get('r_multiple', 0),
                'max_favorable_excursion': 0,
                'exit_reason': f"CAPITAL_SWAP: evicted for {incoming_reason}",
            }
            t.tools.update_trade_status(
                sym, 'CAPITAL_SWAP_EVICT', ltp, pnl, exit_detail=exit_detail
            )
            
            # Remove from exit manager
            t.exit_manager.remove_trade(sym)
            
            # Record with risk governor
            remaining = [p for p in t.tools.paper_positions if p.get('status', 'OPEN') == 'OPEN']
            unrealized = t.risk_governor._calc_unrealized_pnl(remaining)
            t.risk_governor.record_trade_result(sym, pnl, pnl > 0, unrealized_pnl=unrealized)
            
            # Update capital
            with t._pnl_lock:
                t.daily_pnl += pnl
                t.capital += pnl
            t.risk_governor.update_capital(t.capital)
            
            t._wlog(f"   ✅ EVICTED: {sym} @ ₹{ltp:.2f} | P&L: ₹{pnl:+,.0f} | Reason: swap for {incoming_reason}")
            
            return True
            
        except Exception as e:
            t._wlog(f"   ❌ EVICTION ERROR: {sym} — {e}")
            import traceback
            traceback.print_exc()
            return False
