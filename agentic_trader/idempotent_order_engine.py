"""
IDEMPOTENT ORDER ENGINE
Guarantees "same intent = same order" - prevents duplicate entries

Features:
1. Deterministic client_order_id generation
2. Check broker open/completed orders before placing
3. Local order tracking with persistence
4. Reconnect/retry safety
"""

from datetime import datetime, date
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, asdict, field
import hashlib
import json
import os


@dataclass
class OrderIntent:
    """Represents the intent to place an order"""
    symbol: str
    direction: str  # BUY or SELL
    strategy: str   # ORB, VWAP, EMA_SQUEEZE, RSI, EOD, etc.
    setup_id: str   # Optional: specific setup identifier (e.g., "BREAKOUT_UP")
    trade_date: str  # YYYY-MM-DD
    sequence: int = 0  # For multiple orders with same intent on same day
    
    def generate_client_order_id(self) -> str:
        """
        Generate deterministic client_order_id
        Format: SYMBOL|DATE|STRATEGY|DIRECTION|SETUP|SEQ
        Hashed for broker compatibility (max 25 chars usually)
        """
        raw_id = f"{self.symbol}|{self.trade_date}|{self.strategy}|{self.direction}|{self.setup_id}|{self.sequence}"
        
        # Create a short hash for broker compatibility
        hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        
        # Clean symbol for ID (remove exchange prefix)
        clean_symbol = self.symbol.replace("NSE:", "").replace("BSE:", "")[:8]
        
        # Format: SYM_YYYYMMDD_STR_DIR_HASH
        date_short = self.trade_date.replace("-", "")
        
        client_id = f"{clean_symbol}_{date_short}_{self.strategy[:3]}_{self.direction[0]}_{hash_suffix}"
        
        return client_id.upper()


@dataclass
class OrderRecord:
    """Record of a placed order"""
    client_order_id: str
    broker_order_id: str  # Actual order ID from broker
    symbol: str
    direction: str
    strategy: str
    setup_id: str
    quantity: int
    price: float
    status: str  # PENDING, OPEN, COMPLETE, REJECTED, CANCELLED
    created_at: str
    updated_at: str
    raw_intent: str  # Original intent string for debugging


class IdempotentOrderEngine:
    """
    Ensures orders are idempotent - same intent produces same order
    
    Usage:
    1. Create intent: intent = engine.create_intent(symbol, direction, strategy, setup_id)
    2. Check if can place: can_place, reason = engine.can_place_order(intent, broker_orders)
    3. If yes, place order and record: engine.record_order(intent, broker_order_id, ...)
    """
    
    def __init__(self, persistence_file: str = "order_idempotency.json"):
        self.persistence_file = persistence_file
        self.today = date.today().isoformat()
        
        # Track all order IDs placed today
        self.placed_order_ids: Set[str] = set()
        
        # Full order records
        self.order_records: Dict[str, OrderRecord] = {}
        
        # Load persisted state
        self._load_state()
        
        # Reset if new day
        self._check_new_day()
    
    def _load_state(self):
        """Load persisted order state"""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    
                    # Only load if same day
                    if data.get('date') == self.today:
                        self.placed_order_ids = set(data.get('placed_order_ids', []))
                        
                        # Reconstruct order records
                        for oid, rec_data in data.get('order_records', {}).items():
                            self.order_records[oid] = OrderRecord(**rec_data)
                        
                        print(f"📋 Idempotency Engine: Loaded {len(self.placed_order_ids)} order IDs from today")
                    else:
                        print(f"📋 Idempotency Engine: New day - starting fresh")
            except Exception as e:
                print(f"⚠️ Error loading idempotency state: {e}")
    
    def _save_state(self):
        """Persist order state"""
        data = {
            'date': self.today,
            'placed_order_ids': list(self.placed_order_ids),
            'order_records': {oid: asdict(rec) for oid, rec in self.order_records.items()}
        }
        
        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _check_new_day(self):
        """Reset state if new trading day"""
        today = date.today().isoformat()
        if today != self.today:
            print(f"📅 Idempotency Engine: New day detected - resetting order tracking")
            self.today = today
            self.placed_order_ids = set()
            self.order_records = {}
            self._save_state()
    
    def create_intent(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        setup_id: str = "",
        sequence: int = 0
    ) -> OrderIntent:
        """
        Create an order intent with deterministic ID
        
        Args:
            symbol: Trading symbol (e.g., NSE:INFY)
            direction: BUY or SELL
            strategy: Strategy name (ORB, VWAP, EMA_SQUEEZE, RSI, EOD, etc.)
            setup_id: Optional specific setup (e.g., BREAKOUT_UP, OVERSOLD)
            sequence: Sequence number for multiple same-intent orders
        
        Returns:
            OrderIntent with deterministic client_order_id
        """
        self._check_new_day()
        
        return OrderIntent(
            symbol=symbol,
            direction=direction.upper(),
            strategy=strategy.upper(),
            setup_id=setup_id.upper() if setup_id else "DEFAULT",
            trade_date=self.today,
            sequence=sequence
        )
    
    def can_place_order(
        self,
        intent: OrderIntent,
        broker_open_orders: Optional[List[Dict]] = None,
        broker_completed_orders: Optional[List[Dict]] = None
    ) -> tuple[bool, str]:
        """
        Check if an order with this intent can be placed
        
        Returns:
            (can_place: bool, reason: str)
        """
        self._check_new_day()
        
        client_order_id = intent.generate_client_order_id()
        
        # 1. Check local tracking first (fastest)
        if client_order_id in self.placed_order_ids:
            record = self.order_records.get(client_order_id)
            status = record.status if record else "UNKNOWN"
            return False, f"DUPLICATE: Order {client_order_id} already placed (status: {status})"
        
        # 2. Check broker open orders
        if broker_open_orders:
            for order in broker_open_orders:
                # Check by client_order_id tag if available
                if order.get('tag') == client_order_id:
                    return False, f"DUPLICATE: Order {client_order_id} found in broker open orders"
                
                # Also check by symbol+direction if recently placed
                if (order.get('tradingsymbol') == intent.symbol.replace("NSE:", "").replace("BSE:", "") and
                    order.get('transaction_type') == intent.direction and
                    order.get('status') in ['OPEN', 'PENDING', 'TRIGGER PENDING']):
                    # Same symbol, same direction, still open - likely duplicate
                    order_time = order.get('order_timestamp', '')
                    if order_time and self._is_same_session(order_time):
                        return False, f"DUPLICATE: Open order exists for {intent.symbol} {intent.direction}"
        
        # 3. Check broker completed orders (today only)
        if broker_completed_orders:
            for order in broker_completed_orders:
                if order.get('tag') == client_order_id:
                    return False, f"DUPLICATE: Order {client_order_id} already completed"
        
        return True, f"OK: Order {client_order_id} can be placed"
    
    def _is_same_session(self, order_time_str: str) -> bool:
        """Check if order time is from today's session"""
        try:
            if isinstance(order_time_str, datetime):
                order_date = order_time_str.date()
            else:
                order_date = datetime.fromisoformat(order_time_str.replace('Z', '')).date()
            return order_date == date.today()
        except:
            return False
    
    def record_order(
        self,
        intent: OrderIntent,
        broker_order_id: str,
        quantity: int,
        price: float,
        status: str = "PENDING"
    ) -> str:
        """
        Record a placed order to prevent duplicates
        
        Returns:
            client_order_id
        """
        self._check_new_day()
        
        client_order_id = intent.generate_client_order_id()
        now = datetime.now().isoformat()
        
        # Create record
        record = OrderRecord(
            client_order_id=client_order_id,
            broker_order_id=str(broker_order_id),
            symbol=intent.symbol,
            direction=intent.direction,
            strategy=intent.strategy,
            setup_id=intent.setup_id,
            quantity=quantity,
            price=price,
            status=status,
            created_at=now,
            updated_at=now,
            raw_intent=f"{intent.symbol}|{intent.trade_date}|{intent.strategy}|{intent.direction}|{intent.setup_id}|{intent.sequence}"
        )
        
        # Store
        self.placed_order_ids.add(client_order_id)
        self.order_records[client_order_id] = record
        
        # Persist
        self._save_state()
        
        print(f"📋 Idempotency: Recorded order {client_order_id} -> {broker_order_id}")
        
        return client_order_id
    
    def update_order_status(self, client_order_id: str, new_status: str):
        """Update status of a recorded order"""
        if client_order_id in self.order_records:
            self.order_records[client_order_id].status = new_status
            self.order_records[client_order_id].updated_at = datetime.now().isoformat()
            self._save_state()
    
    def get_order_record(self, client_order_id: str) -> Optional[OrderRecord]:
        """Get order record by client_order_id"""
        return self.order_records.get(client_order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderRecord]:
        """Get all orders for a symbol today"""
        return [
            rec for rec in self.order_records.values()
            if rec.symbol == symbol
        ]
    
    def get_orders_by_strategy(self, strategy: str) -> List[OrderRecord]:
        """Get all orders for a strategy today"""
        return [
            rec for rec in self.order_records.values()
            if rec.strategy == strategy.upper()
        ]
    
    def get_today_stats(self) -> Dict:
        """Get statistics for today's orders"""
        total = len(self.order_records)
        by_status = {}
        by_strategy = {}
        by_direction = {"BUY": 0, "SELL": 0}
        
        for rec in self.order_records.values():
            by_status[rec.status] = by_status.get(rec.status, 0) + 1
            by_strategy[rec.strategy] = by_strategy.get(rec.strategy, 0) + 1
            by_direction[rec.direction] = by_direction.get(rec.direction, 0) + 1
        
        return {
            "date": self.today,
            "total_orders": total,
            "by_status": by_status,
            "by_strategy": by_strategy,
            "by_direction": by_direction,
            "order_ids": list(self.placed_order_ids)
        }
    
    def is_order_placed(self, symbol: str, direction: str, strategy: str, setup_id: str = "") -> bool:
        """Quick check if an order with this intent was already placed"""
        intent = self.create_intent(symbol, direction, strategy, setup_id)
        return intent.generate_client_order_id() in self.placed_order_ids
    
    def get_next_sequence(self, symbol: str, direction: str, strategy: str, setup_id: str = "") -> int:
        """Get next available sequence number for retry scenarios"""
        seq = 0
        while True:
            intent = self.create_intent(symbol, direction, strategy, setup_id, sequence=seq)
            if intent.generate_client_order_id() not in self.placed_order_ids:
                return seq
            seq += 1
            if seq > 10:  # Safety limit
                break
        return seq


# Singleton instance
_idempotent_engine: Optional[IdempotentOrderEngine] = None


def get_idempotent_engine() -> IdempotentOrderEngine:
    """Get singleton instance of IdempotentOrderEngine"""
    global _idempotent_engine
    if _idempotent_engine is None:
        _idempotent_engine = IdempotentOrderEngine()
    return _idempotent_engine
