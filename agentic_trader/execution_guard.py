"""
EXECUTION GUARD MODULE
Polices order execution to prevent leaking edge through poor fills

Features:
1. Bid-ask spread ceiling check
2. Dynamic slippage allowance by regime
3. Order type policy by volume regime
4. Price impact guard (thin liquidity)
5. Realized slippage logging
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import json
import os


@dataclass
class SpreadCheck:
    """Result of spread analysis"""
    symbol: str
    bid: float
    ask: float
    spread_abs: float  # Absolute spread in ₹
    spread_bps: float  # Spread in basis points
    is_acceptable: bool
    reason: str
    recommended_order_type: str  # MARKET, LIMIT, BLOCK


@dataclass
class ExecutionPolicy:
    """Policy for a specific order"""
    can_execute: bool
    order_type: str  # MARKET, LIMIT, IOC
    limit_price: float  # For limit orders
    max_slippage_pct: float
    timeout_seconds: int  # For limit orders
    reason: str
    warnings: list = field(default_factory=list)


@dataclass
class SlippageRecord:
    """Record of actual vs expected execution"""
    symbol: str
    side: str
    expected_price: float
    actual_price: float
    slippage_abs: float
    slippage_pct: float
    slippage_bps: float
    volume_regime: str
    timestamp: str
    order_type: str


class ExecutionGuard:
    """
    Guards order execution quality
    
    Rules:
    - Spread > threshold → BLOCK or use LIMIT
    - Volume LOW → BLOCK trading
    - EXPLOSIVE → Allow MARKET with higher slippage tolerance
    - Thin liquidity → Avoid MARKET orders
    """
    
    def __init__(self):
        # Spread thresholds
        self.max_spread_bps = 50  # 0.5% max spread
        self.max_spread_abs = 5.0  # ₹5 max absolute spread
        
        # Slippage allowances by regime
        self.slippage_by_regime = {
            "EXPLOSIVE": 0.5,  # 0.5% allowed
            "HIGH": 0.3,       # 0.3% allowed
            "NORMAL": 0.15,    # 0.15% allowed
            "LOW": 0.0         # No trading allowed
        }
        
        # Order type policy by regime
        self.order_type_by_regime = {
            "EXPLOSIVE": "IOC",    # Immediate or cancel
            "HIGH": "MARKET",      # Market OK
            "NORMAL": "LIMIT",     # Limit with timeout
            "LOW": "BLOCK"         # Block trading
        }
        
        # Limit order timeout
        self.limit_timeout_seconds = 30
        
        # Min depth for market orders (quantity at best bid/ask)
        self.min_depth_qty = 100
        
        # Slippage log
        self.slippage_log: list[SlippageRecord] = []
        self.slippage_log_file = "slippage_log.json"
        self._load_slippage_log()
    
    def _load_slippage_log(self):
        """Load historical slippage data"""
        if os.path.exists(self.slippage_log_file):
            try:
                with open(self.slippage_log_file, 'r') as f:
                    data = json.load(f)
                    self.slippage_log = [SlippageRecord(**r) for r in data]
            except:
                self.slippage_log = []
    
    def _save_slippage_log(self):
        """Save slippage data"""
        with open(self.slippage_log_file, 'w') as f:
            json.dump([asdict(r) for r in self.slippage_log[-1000:]], f, indent=2)
    
    def check_spread(
        self, 
        symbol: str, 
        bid: float, 
        ask: float, 
        ltp: float
    ) -> SpreadCheck:
        """
        Check if bid-ask spread is acceptable for trading
        
        Returns SpreadCheck with recommendation
        """
        if bid <= 0 or ask <= 0:
            return SpreadCheck(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread_abs=0,
                spread_bps=0,
                is_acceptable=False,
                reason="Invalid bid/ask prices",
                recommended_order_type="BLOCK"
            )
        
        spread_abs = ask - bid
        mid_price = (bid + ask) / 2
        spread_bps = (spread_abs / mid_price) * 10000  # Basis points
        
        # Check thresholds
        if spread_bps > self.max_spread_bps:
            return SpreadCheck(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread_abs=spread_abs,
                spread_bps=spread_bps,
                is_acceptable=False,
                reason=f"Spread {spread_bps:.1f} bps exceeds max {self.max_spread_bps} bps",
                recommended_order_type="BLOCK"
            )
        
        if spread_abs > self.max_spread_abs:
            return SpreadCheck(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread_abs=spread_abs,
                spread_bps=spread_bps,
                is_acceptable=False,
                reason=f"Spread ₹{spread_abs:.2f} exceeds max ₹{self.max_spread_abs}",
                recommended_order_type="LIMIT"
            )
        
        # Spread acceptable
        if spread_bps < 10:  # Very tight spread
            order_type = "MARKET"
        elif spread_bps < 30:  # Moderate spread
            order_type = "LIMIT"
        else:  # Wide but acceptable
            order_type = "LIMIT"
        
        return SpreadCheck(
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread_abs=spread_abs,
            spread_bps=spread_bps,
            is_acceptable=True,
            reason=f"Spread OK: {spread_bps:.1f} bps (₹{spread_abs:.2f})",
            recommended_order_type=order_type
        )
    
    def get_execution_policy(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        quantity: int,
        ltp: float,
        bid: float,
        ask: float,
        bid_qty: int,
        ask_qty: int,
        volume_regime: str
    ) -> ExecutionPolicy:
        """
        Determine execution policy for an order
        
        Returns ExecutionPolicy with order type, limits, and warnings
        """
        warnings = []
        
        # 1. Check volume regime
        if volume_regime == "LOW":
            return ExecutionPolicy(
                can_execute=False,
                order_type="BLOCK",
                limit_price=0,
                max_slippage_pct=0,
                timeout_seconds=0,
                reason="Volume regime LOW - trading blocked",
                warnings=["Low volume = high slippage risk"]
            )
        
        # 2. Check spread
        spread_check = self.check_spread(symbol, bid, ask, ltp)
        if not spread_check.is_acceptable:
            return ExecutionPolicy(
                can_execute=False,
                order_type="BLOCK",
                limit_price=0,
                max_slippage_pct=0,
                timeout_seconds=0,
                reason=spread_check.reason,
                warnings=[f"Wide spread: {spread_check.spread_bps:.1f} bps"]
            )
        
        # 3. Check price impact (liquidity at top of book)
        if side == "BUY":
            available_qty = ask_qty
            ref_price = ask
        else:
            available_qty = bid_qty
            ref_price = bid
        
        if available_qty < quantity * 0.5:
            warnings.append(f"Thin liquidity: only {available_qty} at best price vs order {quantity}")
            # Force limit order for thin liquidity
            if volume_regime in ["EXPLOSIVE", "HIGH"]:
                volume_regime = "NORMAL"  # Downgrade to limit order
        
        # 4. Determine order type by regime
        max_slippage = self.slippage_by_regime.get(volume_regime, 0.15)
        order_type = self.order_type_by_regime.get(volume_regime, "LIMIT")
        
        # 5. Calculate limit price
        if order_type in ["LIMIT", "BLOCK"]:
            if side == "BUY":
                # Limit at ask or slightly above for fills
                limit_price = round(ask * (1 + max_slippage / 100), 2)
            else:
                # Limit at bid or slightly below
                limit_price = round(bid * (1 - max_slippage / 100), 2)
        else:
            limit_price = 0  # Market order
        
        # 6. Set timeout for limit orders
        timeout = self.limit_timeout_seconds if order_type == "LIMIT" else 0
        
        return ExecutionPolicy(
            can_execute=True,
            order_type=order_type,
            limit_price=limit_price,
            max_slippage_pct=max_slippage,
            timeout_seconds=timeout,
            reason=f"OK: {order_type} order, max slip {max_slippage}%, regime={volume_regime}",
            warnings=warnings
        )
    
    def record_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        volume_regime: str,
        order_type: str
    ) -> SlippageRecord:
        """
        Record actual vs expected price for slippage analysis
        """
        if expected_price <= 0:
            expected_price = actual_price
        
        slippage_abs = actual_price - expected_price if side == "BUY" else expected_price - actual_price
        slippage_pct = (slippage_abs / expected_price) * 100 if expected_price > 0 else 0
        slippage_bps = slippage_pct * 100
        
        record = SlippageRecord(
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_abs=slippage_abs,
            slippage_pct=slippage_pct,
            slippage_bps=slippage_bps,
            volume_regime=volume_regime,
            timestamp=datetime.now().isoformat(),
            order_type=order_type
        )
        
        self.slippage_log.append(record)
        self._save_slippage_log()
        
        # Print warning if slippage is high
        if abs(slippage_pct) > 0.2:
            print(f"⚠️ SLIPPAGE ALERT: {symbol} {side}")
            print(f"   Expected: ₹{expected_price:.2f} → Actual: ₹{actual_price:.2f}")
            print(f"   Slippage: {slippage_bps:+.1f} bps (₹{slippage_abs:+.2f})")
        
        return record
    
    def get_slippage_stats(self, days: int = 7) -> Dict:
        """Get slippage statistics"""
        if not self.slippage_log:
            return {"avg_slippage_bps": 0, "total_slippage": 0, "count": 0}
        
        # Filter recent
        cutoff = datetime.now().isoformat()[:10]
        recent = [r for r in self.slippage_log if r.timestamp[:10] >= cutoff]
        
        if not recent:
            recent = self.slippage_log[-100:]
        
        total_slippage = sum(r.slippage_abs for r in recent)
        avg_bps = sum(r.slippage_bps for r in recent) / len(recent) if recent else 0
        
        by_regime = {}
        for r in recent:
            regime = r.volume_regime
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(r.slippage_bps)
        
        regime_avg = {k: sum(v)/len(v) for k, v in by_regime.items()}
        
        return {
            "avg_slippage_bps": round(avg_bps, 2),
            "total_slippage": round(total_slippage, 2),
            "count": len(recent),
            "by_regime": regime_avg
        }
    
    def get_status(self) -> str:
        """Get execution guard status"""
        stats = self.get_slippage_stats()
        return (
            f"📊 EXECUTION GUARD:\n"
            f"   Avg slippage: {stats['avg_slippage_bps']:.1f} bps\n"
            f"   Total slippage cost: ₹{stats['total_slippage']:.2f}\n"
            f"   Orders tracked: {stats['count']}"
        )


# Singleton instance
_execution_guard: Optional[ExecutionGuard] = None


def get_execution_guard() -> ExecutionGuard:
    """Get singleton execution guard instance"""
    global _execution_guard
    if _execution_guard is None:
        _execution_guard = ExecutionGuard()
    return _execution_guard


def reset_execution_guard():
    """Reset execution guard"""
    global _execution_guard
    _execution_guard = None


if __name__ == "__main__":
    # Test
    guard = ExecutionGuard()
    
    # Test spread check
    spread = guard.check_spread("NSE:INFY", 1495.5, 1496.0, 1495.75)
    print(f"Spread check: {spread}")
    
    # Test execution policy
    policy = guard.get_execution_policy(
        symbol="NSE:INFY",
        side="BUY",
        quantity=100,
        ltp=1495.75,
        bid=1495.5,
        ask=1496.0,
        bid_qty=500,
        ask_qty=200,
        volume_regime="HIGH"
    )
    print(f"Policy: {policy}")
    
    # Test slippage recording
    record = guard.record_slippage(
        symbol="NSE:INFY",
        side="BUY",
        expected_price=1496.0,
        actual_price=1496.5,
        volume_regime="HIGH",
        order_type="MARKET"
    )
    print(f"Slippage: {record}")
    print(guard.get_status())
