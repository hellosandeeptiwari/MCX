"""
RISK MANAGER MODULE
Enforces hard risk rules - THE MOST IMPORTANT MODULE

Rules:
1. Risk per trade: 0.25-1% of capital
2. Max daily loss: 1-2% (stops trading if hit)
3. Max open positions: 3-5
4. Always place stop-loss
5. Position sizing based on risk
"""

import json
import os
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from config import (
    CAPITAL, RISK_PER_TRADE, MAX_DAILY_LOSS,
    MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY,
    POSITIONS_FILE, TRADES_LOG, DAILY_PNL_FILE,
    MARKET_OPEN, MARKET_CLOSE, NO_NEW_TRADES_AFTER
)


class RiskViolation(Enum):
    NONE = "NONE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_TRADES = "MAX_TRADES"
    POSITION_SIZE = "POSITION_SIZE"
    TRADING_HOURS = "TRADING_HOURS"
    KILL_SWITCH = "KILL_SWITCH"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    order_id: str = ""
    sl_order_id: str = ""
    target_order_id: str = ""
    pnl: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, SL_HIT, TARGET_HIT
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'entry_time': self.entry_time.isoformat(),
            'order_id': self.order_id,
            'sl_order_id': self.sl_order_id,
            'target_order_id': self.target_order_id,
            'pnl': self.pnl,
            'status': self.status
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Position':
        return Position(
            symbol=data['symbol'],
            side=data['side'],
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            target=data['target'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            order_id=data.get('order_id', ''),
            sl_order_id=data.get('sl_order_id', ''),
            target_order_id=data.get('target_order_id', ''),
            pnl=data.get('pnl', 0.0),
            status=data.get('status', 'OPEN')
        )
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.side == "BUY":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
        return self.pnl
    
    def risk_amount(self) -> float:
        """Calculate risk amount (distance to SL)"""
        if self.side == "BUY":
            return (self.entry_price - self.stop_loss) * self.quantity
        else:
            return (self.stop_loss - self.entry_price) * self.quantity


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_taken: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'trades_taken': self.trades_taken,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown,
            'peak_pnl': self.peak_pnl
        }


class RiskManager:
    """
    Central risk management system
    ENFORCES hard rules on all trades
    """
    
    def __init__(self):
        self.capital = CAPITAL
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.today_stats = DailyStats(date=str(date.today()))
        self.trades_log: List[dict] = []
        self.kill_switch = False  # Emergency stop
        
        # Create directory if needed
        self._ensure_directories()
        
        # Load saved state
        self._load_state()
    
    def _ensure_directories(self):
        """Create directories for data files"""
        dir_path = os.path.dirname(os.path.join(os.path.dirname(__file__), POSITIONS_FILE))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_state(self):
        """Load positions and daily stats from files"""
        try:
            # Load positions
            pos_path = os.path.join(os.path.dirname(__file__), '..', POSITIONS_FILE)
            if os.path.exists(pos_path):
                with open(pos_path, 'r') as f:
                    data = json.load(f)
                for sym, pos_data in data.items():
                    self.positions[sym] = Position.from_dict(pos_data)
            
            # Load daily stats
            pnl_path = os.path.join(os.path.dirname(__file__), '..', DAILY_PNL_FILE)
            if os.path.exists(pnl_path):
                with open(pnl_path, 'r') as f:
                    data = json.load(f)
                if data.get('date') == str(date.today()):
                    self.today_stats = DailyStats(**data)
                else:
                    # New day - reset stats
                    self.today_stats = DailyStats(date=str(date.today()))
            
            print(f"✅ Loaded {len(self.positions)} positions")
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")
    
    def _save_state(self):
        """Save positions and daily stats to files"""
        try:
            # Save positions
            pos_path = os.path.join(os.path.dirname(__file__), '..', POSITIONS_FILE)
            with open(pos_path, 'w') as f:
                json.dump({sym: pos.to_dict() for sym, pos in self.positions.items()}, f, indent=2)
            
            # Save daily stats
            pnl_path = os.path.join(os.path.dirname(__file__), '..', DAILY_PNL_FILE)
            with open(pnl_path, 'w') as f:
                json.dump(self.today_stats.to_dict(), f, indent=2)
        except Exception as e:
            print(f"❌ Could not save state: {e}")
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """EMERGENCY STOP - No more trading"""
        self.kill_switch = True
        print(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        self._log_event("KILL_SWITCH", reason)
    
    def deactivate_kill_switch(self):
        """Resume trading"""
        self.kill_switch = False
        print("✅ Kill switch deactivated - Trading resumed")
    
    def _log_event(self, event_type: str, details: str):
        """Log important events"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details
        }
        self.trades_log.append(log_entry)
        
        # Append to log file
        try:
            log_path = os.path.join(os.path.dirname(__file__), '..', TRADES_LOG)
            logs = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            logs.append(log_entry)
            with open(log_path, 'w') as f:
                json.dump(logs[-1000:], f, indent=2)  # Keep last 1000 entries
        except:
            pass
    
    def is_trading_hours(self) -> bool:
        """Check if within trading hours"""
        now = datetime.now().time()
        market_open = datetime.strptime(MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
        return market_open <= now <= market_close
    
    def can_take_new_trade(self) -> bool:
        """Check if new entry is allowed after cutoff"""
        now = datetime.now().time()
        cutoff = datetime.strptime(NO_NEW_TRADES_AFTER, "%H:%M").time()
        return now <= cutoff
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float,
        lot_size: int = 1
    ) -> int:
        """
        Calculate position size based on risk per trade
        
        Risk = (Entry - SL) * Quantity
        Quantity = Risk Amount / (Entry - SL)
        
        Returns quantity (rounded to lot size)
        """
        risk_amount = self.capital * RISK_PER_TRADE
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        quantity = risk_amount / risk_per_share
        
        # Round to lot size
        if lot_size > 1:
            quantity = int(quantity / lot_size) * lot_size
        else:
            quantity = int(quantity)
        
        return max(0, quantity)
    
    def validate_trade(
        self, 
        symbol: str, 
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float
    ) -> tuple[bool, RiskViolation, str]:
        """
        Validate if a trade can be taken
        
        Returns:
            (can_trade, violation_type, reason)
        """
        # 1. Kill switch check
        if self.kill_switch:
            return False, RiskViolation.KILL_SWITCH, "Kill switch is active"
        
        # 2. Trading hours check
        if not self.is_trading_hours():
            return False, RiskViolation.TRADING_HOURS, "Outside trading hours"
        
        # 3. New trade cutoff
        if not self.can_take_new_trade():
            return False, RiskViolation.TRADING_HOURS, f"No new trades after {NO_NEW_TRADES_AFTER}"
        
        # 4. Daily loss limit
        total_pnl = self.today_stats.realized_pnl + self.today_stats.unrealized_pnl
        max_loss = self.capital * MAX_DAILY_LOSS
        if total_pnl <= -max_loss:
            self.activate_kill_switch(f"Daily loss limit hit: ₹{total_pnl:,.2f}")
            return False, RiskViolation.DAILY_LOSS_LIMIT, f"Daily loss limit (₹{max_loss:,.2f}) exceeded"
        
        # 5. Max positions
        open_positions = len([p for p in self.positions.values() if p.status == "OPEN"])
        if open_positions >= MAX_OPEN_POSITIONS:
            return False, RiskViolation.MAX_POSITIONS, f"Max {MAX_OPEN_POSITIONS} positions allowed"
        
        # 6. Max trades per day
        if self.today_stats.trades_taken >= MAX_TRADES_PER_DAY:
            return False, RiskViolation.MAX_TRADES, f"Max {MAX_TRADES_PER_DAY} trades/day reached"
        
        # 7. Duplicate position
        if symbol in self.positions and self.positions[symbol].status == "OPEN":
            return False, RiskViolation.DUPLICATE_POSITION, f"Already have open position in {symbol}"
        
        # 8. Position size validation
        risk_per_share = abs(entry_price - stop_loss)
        trade_risk = risk_per_share * quantity
        max_risk = self.capital * RISK_PER_TRADE * 1.5  # Allow 50% buffer
        if trade_risk > max_risk:
            return False, RiskViolation.POSITION_SIZE, f"Trade risk (₹{trade_risk:,.2f}) exceeds limit (₹{max_risk:,.2f})"
        
        # 9. Stop loss validation
        if stop_loss <= 0:
            return False, RiskViolation.POSITION_SIZE, "Stop loss must be positive"
        
        if side == "BUY" and stop_loss >= entry_price:
            return False, RiskViolation.POSITION_SIZE, "Stop loss must be below entry for BUY"
        
        if side == "SELL" and stop_loss <= entry_price:
            return False, RiskViolation.POSITION_SIZE, "Stop loss must be above entry for SELL"
        
        return True, RiskViolation.NONE, "Trade approved"
    
    def add_position(self, position: Position) -> bool:
        """Add a new position after validation"""
        can_trade, violation, reason = self.validate_trade(
            position.symbol,
            position.side,
            position.quantity,
            position.entry_price,
            position.stop_loss
        )
        
        if not can_trade:
            print(f"❌ Trade rejected: {reason}")
            self._log_event("TRADE_REJECTED", f"{position.symbol}: {reason}")
            return False
        
        self.positions[position.symbol] = position
        self.today_stats.trades_taken += 1
        
        self._save_state()
        self._log_event("POSITION_OPENED", f"{position.side} {position.quantity} {position.symbol} @ ₹{position.entry_price}")
        
        print(f"✅ Position opened: {position.side} {position.quantity} {position.symbol}")
        return True
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for a position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.status == "OPEN":
                pos.calculate_pnl(current_price)
                self._update_unrealized_pnl()
    
    def _update_unrealized_pnl(self):
        """Update total unrealized P&L"""
        self.today_stats.unrealized_pnl = sum(
            pos.pnl for pos in self.positions.values() if pos.status == "OPEN"
        )
        
        # Update peak and drawdown
        total_pnl = self.today_stats.realized_pnl + self.today_stats.unrealized_pnl
        if total_pnl > self.today_stats.peak_pnl:
            self.today_stats.peak_pnl = total_pnl
        
        drawdown = self.today_stats.peak_pnl - total_pnl
        if drawdown > self.today_stats.max_drawdown:
            self.today_stats.max_drawdown = drawdown
        
        # Check daily loss limit
        max_loss = self.capital * MAX_DAILY_LOSS
        if total_pnl <= -max_loss:
            self.activate_kill_switch(f"Daily loss limit hit: ₹{total_pnl:,.2f}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "MANUAL"):
        """Close a position and book P&L"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        if pos.status != "OPEN":
            return
        
        # Calculate final P&L
        pos.calculate_pnl(exit_price)
        realized_pnl = pos.pnl
        
        # Update stats
        self.today_stats.realized_pnl += realized_pnl
        if realized_pnl > 0:
            self.today_stats.winning_trades += 1
        else:
            self.today_stats.losing_trades += 1
        
        # Update position status
        pos.status = reason
        pos.pnl = realized_pnl
        
        self._update_unrealized_pnl()
        self._save_state()
        
        emoji = "🟢" if realized_pnl > 0 else "🔴"
        print(f"{emoji} Position closed: {symbol} | P&L: ₹{realized_pnl:,.2f} | Reason: {reason}")
        self._log_event("POSITION_CLOSED", f"{symbol}: ₹{realized_pnl:,.2f} ({reason})")
    
    def check_sl_target(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if price hit SL or Target
        Returns: 'SL_HIT', 'TARGET_HIT', or None
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        if pos.status != "OPEN":
            return None
        
        if pos.side == "BUY":
            if current_price <= pos.stop_loss:
                return "SL_HIT"
            if current_price >= pos.target:
                return "TARGET_HIT"
        else:  # SELL
            if current_price >= pos.stop_loss:
                return "SL_HIT"
            if current_price <= pos.target:
                return "TARGET_HIT"
        
        return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == "OPEN"]
    
    def get_risk_summary(self) -> dict:
        """Get current risk summary"""
        open_positions = self.get_open_positions()
        total_risk = sum(pos.risk_amount() for pos in open_positions)
        
        max_loss = self.capital * MAX_DAILY_LOSS
        total_pnl = self.today_stats.realized_pnl + self.today_stats.unrealized_pnl
        loss_limit_used = abs(min(0, total_pnl)) / max_loss * 100 if max_loss > 0 else 0
        
        return {
            'capital': self.capital,
            'open_positions': len(open_positions),
            'max_positions': MAX_OPEN_POSITIONS,
            'trades_today': self.today_stats.trades_taken,
            'max_trades': MAX_TRADES_PER_DAY,
            'realized_pnl': round(self.today_stats.realized_pnl, 2),
            'unrealized_pnl': round(self.today_stats.unrealized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'total_risk_exposure': round(total_risk, 2),
            'max_daily_loss': round(max_loss, 2),
            'loss_limit_used_pct': round(loss_limit_used, 1),
            'max_drawdown': round(self.today_stats.max_drawdown, 2),
            'win_rate': round(
                self.today_stats.winning_trades / max(1, self.today_stats.winning_trades + self.today_stats.losing_trades) * 100, 1
            ),
            'kill_switch': self.kill_switch,
            'can_trade': not self.kill_switch and self.is_trading_hours() and self.can_take_new_trade()
        }
    
    def reset_daily_stats(self):
        """Reset stats for new day"""
        self.today_stats = DailyStats(date=str(date.today()))
        self.kill_switch = False
        self._save_state()
        print("✅ Daily stats reset")


# Singleton instance
_risk_manager = None

def get_risk_manager() -> RiskManager:
    """Get singleton RiskManager instance"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


if __name__ == "__main__":
    # Test risk manager
    rm = get_risk_manager()
    
    print("\n📊 Risk Manager Test")
    print("=" * 50)
    
    # Get risk summary
    summary = rm.get_risk_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Test position sizing
    print("\n📐 Position Sizing Test:")
    entry = 2350
    sl = 2320
    qty = rm.calculate_position_size(entry, sl, lot_size=625)
    print(f"  Entry: ₹{entry}, SL: ₹{sl}")
    print(f"  Risk per share: ₹{entry - sl}")
    print(f"  Recommended qty: {qty} shares")
    print(f"  Total risk: ₹{(entry - sl) * qty:,.2f}")
    
    # Test trade validation
    print("\n✅ Trade Validation Test:")
    can_trade, violation, reason = rm.validate_trade(
        symbol="NSE:RELIANCE",
        side="BUY",
        quantity=100,
        entry_price=2500,
        stop_loss=2450
    )
    print(f"  Can trade: {can_trade}")
    print(f"  Violation: {violation.value}")
    print(f"  Reason: {reason}")
