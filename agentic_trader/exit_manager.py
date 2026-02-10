"""
EXIT MANAGER MODULE
First-class exit logic for consistent edge preservation

Exit Types:
1. Hard Stop Loss (structure-based)
2. Time Stop (no follow-through)
3. Break-even Rule (+0.8R)
4. Session Cutoff (15:15)
5. Trailing Stop (after +1R)
"""

from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os
import logging

# Structured logger for speed gate evaluations
_speed_gate_logger = logging.getLogger('speed_gate')
if not _speed_gate_logger.handlers:
    _sg_handler = logging.FileHandler(
        os.path.join(os.path.dirname(__file__), 'speed_gate_log.jsonl'), mode='a'
    )
    _sg_handler.setFormatter(logging.Formatter('%(message)s'))
    _speed_gate_logger.addHandler(_sg_handler)
    _speed_gate_logger.setLevel(logging.INFO)
    _speed_gate_logger.propagate = False


@dataclass
class ExitSignal:
    """Exit signal with reason and urgency"""
    symbol: str
    should_exit: bool
    exit_type: str  # SL_HIT, TIME_STOP, SESSION_CUTOFF, TARGET_HIT, TRAILING_SL
    exit_price: float
    reason: str
    urgency: str  # IMMEDIATE, NORMAL, LOW


@dataclass
class TradeState:
    """Track state of a trade for exit management"""
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    initial_sl: float
    current_sl: float
    target: float
    quantity: int
    
    # Dynamic tracking
    highest_price: float = 0  # For trailing SL (longs)
    lowest_price: float = 0   # For trailing SL (shorts)
    candles_since_entry: int = 0
    max_favorable_move: float = 0  # Greatest R multiple achieved
    breakeven_applied: bool = False
    trailing_active: bool = False
    is_option: bool = False          # True for option positions
    max_premium_gain_pct: float = 0  # Max % gain on premium since entry
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeState':
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)


class ExitManager:
    """
    Manages all exit logic for open positions
    
    Rules:
    - Hard SL: Structure-based (ORB opposite, swing low/high, VWAP)
    - Time Stop: Exit if no +0.5R move in 7 candles
    - Break-even: Move SL to entry after +0.8R
    - Session Cutoff: Exit all by 15:15
    - Trailing: After +1R, trail at 50% of max profit
    """
    
    def __init__(self):
        # Configuration
        self.session_cutoff = time(15, 15)  # 3:15 PM — aligned with TRADING_HOURS['no_new_after'] and EOD exit
        self.time_stop_candles = 7  # Exit if no progress in 7 candles
        self.time_stop_min_r = 0.5  # Must make at least 0.5R to stay
        self.breakeven_trigger_r = 0.8  # Move SL to entry at 0.8R
        self.trailing_start_r = 1.0  # Start trailing at 1R
        self.trailing_pct = 0.5  # Trail at 50% of max profit
        
        # Early speed gate (options only)
        self.option_speed_gate_candles = 4   # Check after 4 candles (20 min)
        self.option_speed_gate_pct = 12.0    # Need +12% premium gain
        self.option_speed_gate_max_r = 0.3   # Only exit if R < 0.3 too
        
        # Track trades
        self.trade_states: Dict[str, TradeState] = {}
        
        # State persistence path
        self._state_file = os.path.join(os.path.dirname(__file__), 'exit_manager_state.json')
        self._load_persisted_state()
    
    def _load_persisted_state(self):
        """Restore trade states from disk (crash recovery)"""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                for sym, state_dict in data.items():
                    self.trade_states[sym] = TradeState.from_dict(state_dict)
                if self.trade_states:
                    print(f"📊 Exit Manager: Restored {len(self.trade_states)} trade states from disk")
        except Exception as e:
            print(f"⚠️ Exit Manager: Could not restore state: {e}")
    
    def _persist_state(self):
        """Save trade states to disk for crash recovery"""
        try:
            data = {}
            for sym, state in self.trade_states.items():
                d = state.to_dict()
                d['entry_time'] = state.entry_time.isoformat()
                data[sym] = d
            with open(self._state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Exit Manager: Could not persist state: {e}")
    
    def register_trade(
        self, 
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        quantity: int
    ) -> TradeState:
        """Register a new trade for exit management"""
        state = TradeState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            initial_sl=stop_loss,
            current_sl=stop_loss,
            target=target,
            quantity=quantity,
            highest_price=entry_price if side == "BUY" else 0,
            lowest_price=entry_price if side == "SELL" else float('inf'),
            candles_since_entry=0,
            max_favorable_move=0,
            breakeven_applied=False,
            trailing_active=False,
            is_option='NFO' in symbol or 'CE' in symbol[-5:] or 'PE' in symbol[-5:],
            max_premium_gain_pct=0,
        )
        self.trade_states[symbol] = state
        self._persist_state()
        print(f"📊 Exit Manager: Registered {symbol} {side} @ {entry_price}")
        return state
    
    def update_trade(self, symbol: str, current_price: float) -> Optional[ExitSignal]:
        """
        Update trade state and check for exit signals
        Call this on every price update
        """
        if symbol not in self.trade_states:
            return None
        
        state = self.trade_states[symbol]
        
        # Calculate R-multiple (risk units moved)
        risk = abs(state.entry_price - state.initial_sl)
        if risk == 0:
            risk = state.entry_price * 0.01  # Fallback to 1%
        
        if state.side == "BUY":
            pnl = current_price - state.entry_price
            state.highest_price = max(state.highest_price, current_price)
            r_multiple = pnl / risk
        else:  # SELL
            pnl = state.entry_price - current_price
            state.lowest_price = min(state.lowest_price, current_price)
            r_multiple = pnl / risk
        
        state.max_favorable_move = max(state.max_favorable_move, r_multiple)
        
        # Track premium % gain for options
        if state.is_option and state.entry_price > 0:
            premium_pct = (current_price - state.entry_price) / state.entry_price * 100
            state.max_premium_gain_pct = max(state.max_premium_gain_pct, premium_pct)
        
        # Check all exit conditions in priority order
        
        # 1. HARD STOP LOSS CHECK (IMMEDIATE)
        exit_signal = self._check_hard_sl(state, current_price)
        if exit_signal:
            return exit_signal
        
        # 2. SESSION CUTOFF CHECK (IMMEDIATE)
        exit_signal = self._check_session_cutoff(state, current_price)
        if exit_signal:
            return exit_signal
        
        # 3. TARGET HIT CHECK
        exit_signal = self._check_target(state, current_price)
        if exit_signal:
            return exit_signal
        
        # 4. BREAK-EVEN RULE (modify SL, don't exit)
        self._apply_breakeven(state, r_multiple)
        
        # 5. TRAILING STOP (modify SL, don't exit)
        self._apply_trailing_stop(state, current_price, r_multiple)
        
        # 6. OPTION SPEED GATE (options only, before general time stop)
        exit_signal = self._check_option_speed_gate(state, r_multiple)
        if exit_signal:
            return exit_signal
        
        # 7. TIME STOP CHECK (after updates)
        exit_signal = self._check_time_stop(state, r_multiple)
        if exit_signal:
            return exit_signal
        
        return None
    
    def increment_candles(self, symbol: str):
        """Call this when a new candle closes"""
        if symbol in self.trade_states:
            self.trade_states[symbol].candles_since_entry += 1
    
    def _check_hard_sl(self, state: TradeState, current_price: float) -> Optional[ExitSignal]:
        """Check if hard stop loss is hit"""
        # Guard: skip SL check if price is 0 or negative (data gap / quote failure)
        if current_price <= 0:
            return None
        if state.side == "BUY":
            if current_price <= state.current_sl:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="SL_HIT",
                    exit_price=current_price,
                    reason=f"Stop loss hit at {current_price} (SL: {state.current_sl})",
                    urgency="IMMEDIATE"
                )
        else:  # SELL
            if current_price >= state.current_sl:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="SL_HIT",
                    exit_price=current_price,
                    reason=f"Stop loss hit at {current_price} (SL: {state.current_sl})",
                    urgency="IMMEDIATE"
                )
        return None
    
    def _check_session_cutoff(self, state: TradeState, current_price: float) -> Optional[ExitSignal]:
        """Check if session cutoff time is reached"""
        now = datetime.now().time()
        if now >= self.session_cutoff:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="SESSION_CUTOFF",
                exit_price=current_price,
                reason=f"Session cutoff at {self.session_cutoff.strftime('%H:%M')} - forced exit",
                urgency="IMMEDIATE"
            )
        return None
    
    def _check_target(self, state: TradeState, current_price: float) -> Optional[ExitSignal]:
        """Check if target is hit"""
        if state.side == "BUY":
            if current_price >= state.target:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="TARGET_HIT",
                    exit_price=current_price,
                    reason=f"Target hit at {current_price} (Target: {state.target})",
                    urgency="NORMAL"
                )
        else:  # SELL
            if current_price <= state.target:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="TARGET_HIT",
                    exit_price=current_price,
                    reason=f"Target hit at {current_price} (Target: {state.target})",
                    urgency="NORMAL"
                )
        return None
    
    def _check_option_speed_gate(self, state: TradeState, r_multiple: float) -> Optional[ExitSignal]:
        """Early exit for options that aren't moving fast enough.
        
        Options are theta-bleeding instruments. If the premium hasn't gained
        +12% within 4 candles (20 min), and R-multiple is also < +0.3R,
        the trade thesis is failing — exit before theta eats the position.
        
        Only applies to option positions (is_option=True).
        Logs every evaluation (EXIT and PASS) for evidence-based tuning.
        """
        if not state.is_option:
            return None
        
        if state.candles_since_entry >= self.option_speed_gate_candles:
            t_minutes = state.candles_since_entry * 5
            should_exit = (state.max_premium_gain_pct < self.option_speed_gate_pct 
                          and state.max_favorable_move < self.option_speed_gate_max_r)
            
            # Structured log for every evaluation at the gate candle
            log_entry = json.dumps({
                "ts": datetime.now().isoformat(),
                "symbol": state.symbol,
                "candles": state.candles_since_entry,
                "t_minutes": t_minutes,
                "ltp_entry": round(state.entry_price, 2),
                "max_premium_pct": round(state.max_premium_gain_pct, 2),
                "r_progress": round(state.max_favorable_move, 3),
                "r_current": round(r_multiple, 3),
                "reason": "OPTION_SPEED_GATE_EXIT" if should_exit else "OPTION_SPEED_GATE_PASS",
            })
            _speed_gate_logger.info(log_entry)
            
            if should_exit:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="OPTION_SPEED_GATE",
                    exit_price=0,  # Will use current market price
                    reason=f"Option speed gate: {state.candles_since_entry} candles ({t_minutes}min), "
                           f"max premium +{state.max_premium_gain_pct:.1f}% (need +{self.option_speed_gate_pct}%), "
                           f"max R: {state.max_favorable_move:.2f} (need +{self.option_speed_gate_max_r}R)",
                    urgency="NORMAL"
                )
        return None
    
    def _check_time_stop(self, state: TradeState, r_multiple: float) -> Optional[ExitSignal]:
        """Exit if no follow-through after X candles"""
        if state.candles_since_entry >= self.time_stop_candles:
            if state.max_favorable_move < self.time_stop_min_r:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="TIME_STOP",
                    exit_price=0,  # Will use current market price
                    reason=f"No follow-through: {state.candles_since_entry} candles, max R: {state.max_favorable_move:.2f}",
                    urgency="NORMAL"
                )
        return None
    
    def _apply_breakeven(self, state: TradeState, r_multiple: float):
        """Move SL to entry after +0.8R"""
        if state.breakeven_applied:
            return
        
        if r_multiple >= self.breakeven_trigger_r:
            state.current_sl = state.entry_price
            state.breakeven_applied = True
            self._persist_state()
            print(f"🔒 {state.symbol}: Break-even applied (SL moved to {state.entry_price})")
    
    def _apply_trailing_stop(self, state: TradeState, current_price: float, r_multiple: float):
        """Apply trailing stop after +1R"""
        if r_multiple < self.trailing_start_r:
            return
        
        risk = abs(state.entry_price - state.initial_sl)
        
        if state.side == "BUY":
            # Trail at 50% of max profit from entry
            profit = state.highest_price - state.entry_price
            trail_distance = profit * (1 - self.trailing_pct)
            new_sl = state.highest_price - trail_distance
            
            if new_sl > state.current_sl:
                old_sl = state.current_sl
                state.current_sl = round(new_sl, 2)
                state.trailing_active = True
                self._persist_state()
                print(f"📈 {state.symbol}: Trailing SL updated {old_sl} → {state.current_sl}")
        else:  # SELL
            # Trail at 50% of max profit from entry
            profit = state.entry_price - state.lowest_price
            trail_distance = profit * (1 - self.trailing_pct)
            new_sl = state.lowest_price + trail_distance
            
            if new_sl < state.current_sl:
                old_sl = state.current_sl
                state.current_sl = round(new_sl, 2)
                state.trailing_active = True
                self._persist_state()
                print(f"📉 {state.symbol}: Trailing SL updated {old_sl} → {state.current_sl}")
    
    def remove_trade(self, symbol: str):
        """Remove a trade from tracking (after exit) and reset hysteresis"""
        if symbol in self.trade_states:
            del self.trade_states[symbol]
            self._persist_state()
            print(f"📊 Exit Manager: Removed {symbol} from tracking")
            
            # === RESET HYSTERESIS FOR TREND FOLLOWING + REGIME SCORER ===
            # This allows fresh trend detection on next entry
            try:
                from trend_following import get_trend_engine
                get_trend_engine().reset_hysteresis(symbol, "POSITION_CLOSED")
            except ImportError:
                pass
            try:
                from regime_score import get_regime_scorer
                get_regime_scorer().reset_hysteresis(symbol, "POSITION_CLOSED")
            except ImportError:
                pass
    
    def get_trade_state(self, symbol: str) -> Optional[TradeState]:
        """Get current state of a trade"""
        return self.trade_states.get(symbol)
    
    def get_all_states(self) -> List[TradeState]:
        """Get all tracked trades"""
        return list(self.trade_states.values())
    
    def check_all_exits(self, prices: Dict[str, float]) -> List[ExitSignal]:
        """
        Check all trades for exit signals
        prices: Dict of symbol -> current_price
        """
        signals = []
        for symbol, price in prices.items():
            signal = self.update_trade(symbol, price)
            if signal:
                signals.append(signal)
        return signals
    
    def force_exit_all(self, reason: str = "Manual exit") -> List[str]:
        """Force exit all positions"""
        symbols = list(self.trade_states.keys())
        for symbol in symbols:
            self.remove_trade(symbol)
        return symbols
    
    def get_status_summary(self) -> str:
        """Get summary of all tracked trades"""
        if not self.trade_states:
            return "No active trades being managed"
        
        lines = ["📊 EXIT MANAGER STATUS:"]
        for symbol, state in self.trade_states.items():
            be_status = "🔒BE" if state.breakeven_applied else ""
            trail_status = "📈TRAIL" if state.trailing_active else ""
            lines.append(
                f"  {symbol}: {state.side} @ {state.entry_price} | "
                f"SL: {state.current_sl} | Target: {state.target} | "
                f"Candles: {state.candles_since_entry} | "
                f"Max R: {state.max_favorable_move:.2f} {be_status} {trail_status}"
            )
        return "\n".join(lines)


# Singleton instance
_exit_manager: Optional[ExitManager] = None


def get_exit_manager() -> ExitManager:
    """Get singleton exit manager instance"""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ExitManager()
    return _exit_manager


def reset_exit_manager():
    """Reset exit manager (for testing)"""
    global _exit_manager
    _exit_manager = None


# Structure-based stop loss helpers
def calculate_structure_sl(
    side: str,
    entry_price: float,
    orb_high: float,
    orb_low: float,
    vwap: float,
    swing_low: float,
    swing_high: float,
    atr: float
) -> Tuple[float, str]:
    """
    Calculate structure-based stop loss
    Returns (stop_loss_price, reason)
    
    Priority:
    1. ORB opposite (if entered on ORB breakout)
    2. Recent swing point
    3. VWAP
    4. ATR-based fallback
    """
    buffer = atr * 0.1  # Small buffer
    
    if side == "BUY":
        candidates = []
        
        # ORB low (if breakout trade)
        if orb_low > 0:
            candidates.append((orb_low - buffer, "ORB_LOW"))
        
        # Swing low
        if swing_low > 0:
            candidates.append((swing_low - buffer, "SWING_LOW"))
        
        # VWAP
        if vwap > 0 and vwap < entry_price:
            candidates.append((vwap - buffer, "VWAP"))
        
        # ATR fallback
        candidates.append((entry_price - atr, "ATR_1X"))
        
        # Pick the tightest SL that makes sense (highest for BUY)
        valid_candidates = [(sl, reason) for sl, reason in candidates if sl < entry_price]
        if valid_candidates:
            # Sort by SL price descending (tightest = highest for BUY)
            valid_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Don't use too tight SL (at least 0.5% from entry)
            min_sl = entry_price * 0.995
            for sl, reason in valid_candidates:
                if sl <= min_sl:
                    return round(sl, 2), reason
        
        # Fallback
        return round(entry_price - atr, 2), "ATR_1X"
    
    else:  # SELL
        candidates = []
        
        # ORB high
        if orb_high > 0:
            candidates.append((orb_high + buffer, "ORB_HIGH"))
        
        # Swing high
        if swing_high > 0:
            candidates.append((swing_high + buffer, "SWING_HIGH"))
        
        # VWAP
        if vwap > 0 and vwap > entry_price:
            candidates.append((vwap + buffer, "VWAP"))
        
        # ATR fallback
        candidates.append((entry_price + atr, "ATR_1X"))
        
        # Pick tightest SL (lowest for SELL)
        valid_candidates = [(sl, reason) for sl, reason in candidates if sl > entry_price]
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[0])
            
            # At least 0.5% from entry
            min_sl = entry_price * 1.005
            for sl, reason in valid_candidates:
                if sl >= min_sl:
                    return round(sl, 2), reason
        
        # Fallback
        return round(entry_price + atr, 2), "ATR_1X"


if __name__ == "__main__":
    # Test the exit manager
    em = ExitManager()
    
    # Register a test trade
    state = em.register_trade(
        symbol="NSE:INFY",
        side="BUY",
        entry_price=1500,
        stop_loss=1485,
        target=1522.5,
        quantity=100
    )
    
    print(f"\nInitial state:")
    print(em.get_status_summary())
    
    # Simulate price movements
    prices = [1505, 1510, 1512, 1508, 1515, 1520]
    for i, price in enumerate(prices):
        em.increment_candles("NSE:INFY")
        signal = em.update_trade("NSE:INFY", price)
        print(f"\nCandle {i+1}: Price={price}")
        print(em.get_status_summary())
        if signal:
            print(f"EXIT SIGNAL: {signal}")
            break
