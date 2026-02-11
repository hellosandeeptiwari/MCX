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
    exit_type: str  # SL_HIT, TIME_STOP, SESSION_CUTOFF, TARGET_HIT, TRAILING_SL, PARTIAL_PROFIT
    exit_price: float
    reason: str
    urgency: str  # IMMEDIATE, NORMAL, LOW
    partial_pct: float = 0.0  # 0 = full exit, 0.5 = exit 50% of qty


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
    partial_booked: bool = False     # True after 50% profit booking at +15%
    is_credit_spread: bool = False   # True for credit spread positions
    net_credit: float = 0.0         # Net credit received per share (spreads)
    spread_width: float = 0.0       # Strike distance between legs (spreads)
    is_debit_spread: bool = False    # True for debit spread positions
    net_debit: float = 0.0          # Net debit paid per share (debit spreads)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeState':
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        # Backward compat: add fields that may not exist in old state files
        data.setdefault('partial_booked', False)
        data.setdefault('is_credit_spread', False)
        data.setdefault('net_credit', 0.0)
        data.setdefault('spread_width', 0.0)
        data.setdefault('is_debit_spread', False)
        data.setdefault('net_debit', 0.0)
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
        self.time_stop_candles = 10  # Exit if no progress in 10 candles (50 min) — was 7
        self.time_stop_min_r = 0.3  # Must make at least 0.3R to stay — was 0.5R (too aggressive)
        self.breakeven_trigger_r = 0.8  # Move SL to entry at 0.8R
        self.trailing_start_r = 0.5  # Start trailing at 0.5R (was 1.0R)
        self.trailing_pct = 0.6  # Trail retaining 60% of max profit (was 50%)
        
        # Scaled profit booking (options only)
        self.partial_profit_pct = 15.0       # Book 50% at +15% premium gain
        self.partial_exit_fraction = 0.5     # Exit 50% of position
        
        # Early speed gate (options only) — RELAXED: was killing 91% of trades
        self.option_speed_gate_candles = 12  # Check after 12 candles (60 min) — was 6
        self.option_speed_gate_pct = 3.0     # Need +3% premium gain — was 5%
        self.option_speed_gate_max_r = 0.10  # OR need +0.10R — was 0.15
        
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
        Call this on every price update.
        
        For credit spreads, current_price = net debit to close (sold_ltp - hedge_ltp).
        Profit = net_credit - current_debit. If current_debit rises, we're losing.
        """
        if symbol not in self.trade_states:
            return None
        
        state = self.trade_states[symbol]
        
        # === CREDIT SPREAD EXIT LOGIC (different from directional) ===
        if state.is_credit_spread and state.net_credit > 0:
            return self._update_credit_spread(state, current_price)
        
        # === DEBIT SPREAD EXIT LOGIC (intraday momentum) ===
        if state.is_debit_spread and state.net_debit > 0:
            return self._update_debit_spread(state, current_price)
        
        # === STANDARD DIRECTIONAL EXIT LOGIC ===
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
        
        # 3.5 PARTIAL PROFIT BOOKING (options only, before trailing/breakeven)
        exit_signal = self._check_partial_profit(state, current_price)
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
    
    def _check_partial_profit(self, state: TradeState, current_price: float) -> Optional[ExitSignal]:
        """Book 50% of position when premium gain hits +15%.
        
        Only for option positions with >= 2 lots. Fires once (partial_booked flag).
        For 1-lot positions: can't split, so applies aggressive breakeven + tight trail.
        After partial booking, also moves SL to entry (breakeven on remaining).
        """
        if not state.is_option:
            return None
        if state.partial_booked:
            return None
        
        if state.entry_price <= 0:
            return None
        
        premium_pct = (current_price - state.entry_price) / state.entry_price * 100
        
        if state.side == "BUY" and premium_pct >= self.partial_profit_pct:
            # Check if we have enough lots to split
            lot_size = self._get_lot_size_for_symbol(state.symbol)
            total_lots = state.quantity // lot_size if lot_size > 0 else 1
            
            if total_lots >= 2:
                # === MULTI-LOT: Partial exit 50% ===
                state.partial_booked = True
                state.current_sl = state.entry_price
                state.breakeven_applied = True
                self._persist_state()
                exit_lots = total_lots // 2
                print(f"💰 {state.symbol}: PARTIAL PROFIT — booking {exit_lots} of {total_lots} lots at +{premium_pct:.1f}% | SL→entry on remainder")
                
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="PARTIAL_PROFIT",
                    exit_price=current_price,
                    reason=f"Partial profit: +{premium_pct:.1f}% premium gain, booking {exit_lots}/{total_lots} lots",
                    urgency="NORMAL",
                    partial_pct=self.partial_exit_fraction,
                )
            else:
                # === SINGLE LOT: Can't split, protect with tight trail ===
                state.partial_booked = True  # Don't re-check
                state.current_sl = state.entry_price  # Breakeven
                state.breakeven_applied = True
                state.trailing_active = True
                # Tight trail: retain 70% of current profit
                profit = current_price - state.entry_price
                tight_sl = current_price - profit * 0.30  # Keep 70%
                if tight_sl > state.current_sl:
                    state.current_sl = round(tight_sl, 2)
                self._persist_state()
                print(f"🔒 {state.symbol}: +{premium_pct:.1f}% gain (1 lot) — can't split, tight trail SL→₹{state.current_sl:.2f} (lock 70%)")
                return None  # No exit signal, just tightened SL
        
        return None
    
    def _get_lot_size_for_symbol(self, symbol: str) -> int:
        """Get lot size for a symbol from FNO_LOT_SIZES"""
        try:
            from options_trader import FNO_LOT_SIZES
            # symbol like NFO:TATASTEEL26FEB207CE — extract underlying
            clean = symbol.replace('NFO:', '')
            for name, size in FNO_LOT_SIZES.items():
                if clean.startswith(name):
                    return size
        except ImportError:
            pass
        return 1  # Fallback
    
    def _check_option_speed_gate(self, state: TradeState, r_multiple: float) -> Optional[ExitSignal]:
        """Early exit for options that aren't moving fast enough.
        
        Options are theta-bleeding instruments. If the premium hasn't gained
        +5% within 6 candles (30 min), AND R-multiple is also < +0.15R,
        the trade thesis is failing — exit before theta eats the position.
        
        Uses OR-pass logic: survives if EITHER premium% OR R-multiple is good enough.
        Only applies to option positions (is_option=True).
        Logs every evaluation (EXIT and PASS) for evidence-based tuning.
        """
        if not state.is_option:
            return None
        # Credit spreads don't need speed gate - theta works FOR us
        if state.is_credit_spread:
            return None
        # Debit spreads have their own exit logic
        if state.is_debit_spread:
            return None
        
        if state.candles_since_entry >= self.option_speed_gate_candles:
            t_minutes = state.candles_since_entry * 5
            # OR-pass: trade survives if EITHER condition is met
            premium_ok = state.max_premium_gain_pct >= self.option_speed_gate_pct
            r_ok = state.max_favorable_move >= self.option_speed_gate_max_r
            should_exit = not premium_ok and not r_ok  # Only exit if BOTH fail
            
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
    
    def _update_credit_spread(self, state: TradeState, current_debit: float) -> Optional[ExitSignal]:
        """
        Exit logic for credit spread positions.
        
        Credit spread P&L:
        - We collected net_credit upfront
        - To close, we pay current_debit (= sold_ltp - hedge_ltp now)
        - Profit = net_credit - current_debit (per share)
        - If current_debit drops (options decay), we profit
        - If current_debit rises (move against us), we lose
        
        Exit conditions:
        1. TARGET: current_debit drops to target_credit (captured 65% of credit)
        2. STOP LOSS: current_debit rises to stop_loss_debit (loss = 2× credit)
        3. SESSION CUTOFF: exit at 15:15 regardless
        4. TIME DECAY WIN: If > 80% of credit captured, take profit early
        """
        # Guard against bad price data
        if current_debit < 0:
            current_debit = 0
        
        profit_per_share = state.net_credit - current_debit
        profit_pct = (profit_per_share / state.net_credit) * 100 if state.net_credit > 0 else 0
        
        # Track best profit seen
        state.max_favorable_move = max(state.max_favorable_move, profit_pct / 100)
        
        # 1. SESSION CUTOFF (IMMEDIATE)
        now = datetime.now().time()
        if now >= self.session_cutoff:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="SESSION_CUTOFF",
                exit_price=current_debit,
                reason=f"Credit spread session cutoff | P&L/share: ₹{profit_per_share:+.2f} ({profit_pct:+.1f}%)",
                urgency="IMMEDIATE"
            )
        
        # 2. STOP LOSS: loss exceeds SL threshold
        # stop_loss stored as max debit we're willing to pay to close
        if current_debit >= state.initial_sl:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="SPREAD_SL_HIT",
                exit_price=current_debit,
                reason=f"Credit spread SL hit | Debit ₹{current_debit:.2f} >= SL ₹{state.initial_sl:.2f} | Loss/share: ₹{profit_per_share:.2f}",
                urgency="IMMEDIATE"
            )
        
        # 3. TARGET: captured enough of the credit
        if current_debit <= state.target:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="SPREAD_TARGET_HIT",
                exit_price=current_debit,
                reason=f"Credit spread target hit | Debit ₹{current_debit:.2f} <= Target ₹{state.target:.2f} | Profit: ₹{profit_per_share:+.2f}/share ({profit_pct:+.1f}%)",
                urgency="NORMAL"
            )
        
        # 4. EARLY PROFIT: if > 80% captured, take it (don't wait for gamma risk)
        if profit_pct >= 80:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="SPREAD_EARLY_PROFIT",
                exit_price=current_debit,
                reason=f"Credit spread 80%+ captured | Profit: ₹{profit_per_share:+.2f}/share ({profit_pct:+.1f}%)",
                urgency="NORMAL"
            )
        
        # 5. TRAILING SL: once >40% profit captured, trail the SL tighter
        if profit_pct >= 40 and not state.trailing_active:
            # Activate trailing — move SL to breakeven (net_credit = entry cost)
            state.trailing_active = True
            state.current_sl = state.net_credit  # SL at breakeven (debit = credit = no loss)
            state.breakeven_applied = True
        
        if state.trailing_active:
            # Trail: allow max giveback of 30% of best profit seen
            best_debit = state.net_credit * (1 - state.max_favorable_move)  # Best (lowest) debit
            # New SL = best_debit + 30% of (entry_credit - best_debit) as cushion
            trail_cushion = (state.net_credit - best_debit) * 0.30
            new_sl = best_debit + trail_cushion
            
            # Only tighten, never loosen
            if new_sl < state.current_sl:
                state.current_sl = new_sl
            
            # Check if trailed SL hit
            if current_debit >= state.current_sl and state.current_sl < state.initial_sl:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="SPREAD_TRAIL_SL",
                    exit_price=current_debit,
                    reason=f"Credit spread trailing SL hit | Debit ₹{current_debit:.2f} >= Trail SL ₹{state.current_sl:.2f} | Profit: ₹{profit_per_share:+.2f}/share ({profit_pct:+.1f}%)",
                    urgency="NORMAL"
                )
        
        return None
    
    def _update_debit_spread(self, state: TradeState, current_value: float) -> Optional[ExitSignal]:
        """
        Exit logic for debit spread positions (intraday momentum).
        
        Debit spread P&L:
        - We paid net_debit to enter
        - current_value = buy_ltp - sell_ltp (current spread value)
        - Profit = current_value - net_debit (per share)
        - If current_value rises (move in our direction), we profit
        - If current_value drops (move against us), we lose
        
        Exit conditions:
        1. TARGET: current_value reaches target (50% gain on debit)
        2. STOP LOSS: current_value drops to SL (40% loss of debit)
        3. TIME CUTOFF: auto-exit at 15:05 (no overnight for debit spreads)
        4. TRAILING: after 30% profit, trail with 40% giveback
        """
        if current_value < 0:
            current_value = 0
        
        profit_per_share = current_value - state.net_debit
        profit_pct = (profit_per_share / state.net_debit) * 100 if state.net_debit > 0 else 0
        
        # Track best profit seen (as ratio)
        state.max_favorable_move = max(state.max_favorable_move, profit_pct / 100)
        
        # Track highest spread value for trailing
        state.highest_price = max(state.highest_price, current_value)
        
        # 1. TIME CUTOFF — debit spreads exit 10 min earlier than credit spreads
        now = datetime.now().time()
        from config import DEBIT_SPREAD_CONFIG
        auto_exit_str = DEBIT_SPREAD_CONFIG.get('auto_exit_time', '15:05')
        auto_exit_time = datetime.strptime(auto_exit_str, '%H:%M').time()
        if now >= auto_exit_time:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="DEBIT_SPREAD_TIME_EXIT",
                exit_price=current_value,
                reason=f"Debit spread auto-exit at {auto_exit_str} | P&L/share: ₹{profit_per_share:+.2f} ({profit_pct:+.1f}%)",
                urgency="IMMEDIATE"
            )
        
        # 2. STOP LOSS: spread value dropped too much
        if current_value <= state.current_sl:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="DEBIT_SPREAD_SL",
                exit_price=current_value,
                reason=f"Debit spread SL hit | Value ₹{current_value:.2f} <= SL ₹{state.current_sl:.2f} | Loss: ₹{profit_per_share:.2f}/share ({profit_pct:.1f}%)",
                urgency="IMMEDIATE"
            )
        
        # 3. TARGET: spread value rose enough
        if current_value >= state.target:
            return ExitSignal(
                symbol=state.symbol,
                should_exit=True,
                exit_type="DEBIT_SPREAD_TARGET",
                exit_price=current_value,
                reason=f"Debit spread target hit | Value ₹{current_value:.2f} >= Target ₹{state.target:.2f} | Profit: ₹{profit_per_share:+.2f}/share ({profit_pct:+.1f}%)",
                urgency="NORMAL"
            )
        
        # 4. MAX PROFIT CAP: if near max profit (spread_width), take it
        if state.spread_width > 0:
            max_possible = state.spread_width
            if current_value >= max_possible * 0.80:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="DEBIT_SPREAD_MAX_PROFIT",
                    exit_price=current_value,
                    reason=f"Debit spread 80%+ of max profit | Value ₹{current_value:.2f} / Max ₹{max_possible:.2f} | Profit: ₹{profit_per_share:+.2f}/share",
                    urgency="NORMAL"
                )
        
        # 5. TRAILING STOP: after 30% profit, trail with 40% giveback
        trail_activation = DEBIT_SPREAD_CONFIG.get('trail_activation_pct', 30)
        trail_giveback = DEBIT_SPREAD_CONFIG.get('trail_giveback_pct', 40)
        
        if profit_pct >= trail_activation and not state.trailing_active:
            state.trailing_active = True
            # Move SL to breakeven (entry debit)
            state.current_sl = state.net_debit
            state.breakeven_applied = True
        
        if state.trailing_active:
            # Trail: SL = highest_value - giveback% of (highest - entry)
            peak_profit = state.highest_price - state.net_debit
            if peak_profit > 0:
                trail_sl = state.highest_price - (peak_profit * trail_giveback / 100)
                # Only tighten, never loosen
                if trail_sl > state.current_sl:
                    state.current_sl = trail_sl
            
            # Check trailed SL
            if current_value <= state.current_sl and state.current_sl > state.initial_sl:
                return ExitSignal(
                    symbol=state.symbol,
                    should_exit=True,
                    exit_type="DEBIT_SPREAD_TRAIL_SL",
                    exit_price=current_value,
                    reason=f"Debit spread trailing SL | Value ₹{current_value:.2f} <= Trail SL ₹{state.current_sl:.2f} | Profit: ₹{profit_per_share:+.2f}/share ({profit_pct:+.1f}%)",
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
        """Apply trailing stop after +0.5R, retaining 60% of max profit"""
        if r_multiple < self.trailing_start_r:
            return
        
        risk = abs(state.entry_price - state.initial_sl)
        
        if state.side == "BUY":
            # Trail retaining 60% of max profit from entry
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
            # Trail retaining 60% of max profit from entry
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
                f"  {symbol}: {state.side} @ {round(state.entry_price, 2)} | "
                f"SL: {round(state.current_sl, 2)} | Target: {round(state.target, 2)} | "
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
