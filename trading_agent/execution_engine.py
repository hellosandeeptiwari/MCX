"""
EXECUTION ENGINE MODULE
Converts signals to orders and handles order management

Features:
- Order placement (market, limit, SL)
- Bracket orders (entry + SL + target)
- Order status tracking
- Rejection handling with retries
- Position exit logic
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from kiteconnect import KiteConnect

from config import API_KEY
from data_manager import get_data_manager
from risk_manager import get_risk_manager, Position
from signal_engine import Signal, SignalType


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER PENDING"


class ProductType(Enum):
    MIS = "MIS"  # Intraday
    CNC = "CNC"  # Delivery (equity)
    NRML = "NRML"  # Normal (F&O)


@dataclass
class OrderResult:
    """Result of an order placement"""
    success: bool
    order_id: str
    message: str
    status: OrderStatus = OrderStatus.PENDING


class ExecutionEngine:
    """
    Handles all order execution
    Converts signals to actual orders via Zerodha API
    """
    
    def __init__(self):
        self.dm = get_data_manager()
        self.rm = get_risk_manager()
        self.kite = self.dm.kite
        self.pending_orders: Dict[str, dict] = {}  # order_id -> order details
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def _get_exchange_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        Parse symbol like NSE:RELIANCE into exchange and tradingsymbol
        """
        parts = symbol.split(":")
        if len(parts) == 2:
            return parts[0], parts[1]
        return "NSE", symbol
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for F&O instruments"""
        exchange, tradingsymbol = self._get_exchange_symbol(symbol)
        
        # Known lot sizes (cache this in production)
        lot_sizes = {
            "MCX": 625,
            "RELIANCE": 250,
            "TCS": 150,
            "HDFCBANK": 550,
            "INFY": 300,
            "SBIN": 1500,
            "NIFTY 50": 50,
            "NIFTY BANK": 25,
        }
        
        return lot_sizes.get(tradingsymbol, 1)
    
    def place_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0,
        trigger_price: float = 0,
        product: ProductType = ProductType.MIS,
        tag: str = ""
    ) -> OrderResult:
        """
        Place a single order
        
        Args:
            symbol: Trading symbol (e.g., NSE:RELIANCE)
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, SL, SL-M
            price: Limit price (for LIMIT/SL orders)
            trigger_price: Trigger price (for SL orders)
            product: MIS (intraday), CNC (delivery), NRML (F&O)
            tag: Custom tag for identification
        
        Returns:
            OrderResult with success status and order_id
        """
        exchange, tradingsymbol = self._get_exchange_symbol(symbol)
        
        try:
            order_params = {
                'exchange': exchange,
                'tradingsymbol': tradingsymbol,
                'transaction_type': self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
                'quantity': quantity,
                'product': product.value,
                'order_type': order_type.value,
                'validity': self.kite.VALIDITY_DAY,
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.SL]:
                order_params['price'] = price
            
            # Add trigger price for SL orders
            if order_type in [OrderType.SL, OrderType.SL_M]:
                order_params['trigger_price'] = trigger_price
            
            # Add tag
            if tag:
                order_params['tag'] = tag[:20]  # Max 20 chars
            
            order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **order_params)
            
            # Track order
            self.pending_orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type.value,
                'price': price,
                'status': 'PENDING',
                'time': datetime.now().isoformat()
            }
            
            print(f"✅ Order placed: {order_id} | {side} {quantity} {symbol}")
            return OrderResult(success=True, order_id=order_id, message="Order placed successfully")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Order failed: {error_msg}")
            return OrderResult(success=False, order_id="", message=error_msg, status=OrderStatus.REJECTED)
    
    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        trailing_sl: float = 0
    ) -> Tuple[OrderResult, OrderResult, OrderResult]:
        """
        Place a bracket order (entry + SL + target)
        
        Note: Zerodha BO is discontinued for most segments.
        This implements it using 3 separate orders.
        
        Returns:
            Tuple of (entry_result, sl_result, target_result)
        """
        # 1. Place entry order
        entry_result = self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=entry_price,
            product=ProductType.MIS,
            tag="ENTRY"
        )
        
        if not entry_result.success:
            return entry_result, OrderResult(False, "", "Entry failed"), OrderResult(False, "", "Entry failed")
        
        # Wait for entry to be filled
        time.sleep(1)
        status = self.get_order_status(entry_result.order_id)
        
        if status != OrderStatus.COMPLETE:
            # Entry not filled - cancel and return
            self.cancel_order(entry_result.order_id)
            return entry_result, OrderResult(False, "", "Entry not filled"), OrderResult(False, "", "Entry not filled")
        
        # 2. Place SL order (opposite side)
        sl_side = "SELL" if side == "BUY" else "BUY"
        sl_trigger = stop_loss * 1.001 if side == "BUY" else stop_loss * 0.999  # Small buffer
        
        sl_result = self.place_order(
            symbol=symbol,
            side=sl_side,
            quantity=quantity,
            order_type=OrderType.SL_M,
            trigger_price=sl_trigger,
            product=ProductType.MIS,
            tag="SL"
        )
        
        # 3. Place target order (opposite side)
        target_result = self.place_order(
            symbol=symbol,
            side=sl_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=target,
            product=ProductType.MIS,
            tag="TARGET"
        )
        
        return entry_result, sl_result, target_result
    
    def execute_signal(self, signal: Signal, dry_run: bool = False) -> bool:
        """
        Execute a trading signal
        
        Args:
            signal: Signal object from SignalEngine
            dry_run: If True, only simulate (don't place real orders)
        
        Returns:
            True if executed successfully
        """
        # Get lot size
        lot_size = self._get_lot_size(signal.symbol)
        
        # Calculate position size
        quantity = self.rm.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            lot_size=lot_size
        )
        
        if quantity == 0:
            print(f"⚠️ Position size is 0 for {signal.symbol}")
            return False
        
        # Validate with risk manager
        side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
        can_trade, violation, reason = self.rm.validate_trade(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss
        )
        
        if not can_trade:
            print(f"❌ Signal rejected: {reason}")
            return False
        
        print(f"\n🎯 Executing Signal: {signal.signal_type.value} {signal.symbol}")
        print(f"   Entry: ₹{signal.entry_price:,.2f}")
        print(f"   SL: ₹{signal.stop_loss:,.2f}")
        print(f"   Target: ₹{signal.target:,.2f}")
        print(f"   Quantity: {quantity}")
        print(f"   Strategy: {signal.strategy.value}")
        
        if dry_run:
            print("   [DRY RUN - No actual order placed]")
            return True
        
        # Place market order for entry
        entry_result = self.place_order(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
            tag=signal.strategy.value[:8]
        )
        
        if not entry_result.success:
            return False
        
        # Wait and check fill
        time.sleep(0.5)
        status = self.get_order_status(entry_result.order_id)
        
        if status != OrderStatus.COMPLETE:
            print(f"⚠️ Entry order not filled: {status}")
            return False
        
        # Get actual fill price
        order_history = self.kite.order_history(entry_result.order_id)
        fill_price = order_history[-1].get('average_price', signal.entry_price)
        
        # Create position in risk manager
        position = Position(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            entry_time=datetime.now(),
            order_id=entry_result.order_id
        )
        
        self.rm.add_position(position)
        
        # Place SL order
        sl_side = "SELL" if side == "BUY" else "BUY"
        sl_trigger = signal.stop_loss * (1.001 if side == "BUY" else 0.999)
        
        sl_result = self.place_order(
            symbol=signal.symbol,
            side=sl_side,
            quantity=quantity,
            order_type=OrderType.SL_M,
            trigger_price=sl_trigger,
            product=ProductType.MIS,
            tag="SL"
        )
        
        if sl_result.success:
            position.sl_order_id = sl_result.order_id
        
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order"""
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] == order_id:
                    status_str = order['status'].upper()
                    for status in OrderStatus:
                        if status.value in status_str:
                            return status
                    return OrderStatus.PENDING
            return OrderStatus.PENDING
        except Exception as e:
            print(f"❌ Error getting order status: {e}")
            return OrderStatus.PENDING
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
            print(f"✅ Order cancelled: {order_id}")
            return True
        except Exception as e:
            print(f"❌ Cancel failed: {e}")
            return False
    
    def modify_order(
        self,
        order_id: str,
        quantity: int = None,
        price: float = None,
        trigger_price: float = None,
        order_type: OrderType = None
    ) -> bool:
        """Modify an existing order"""
        try:
            params = {}
            if quantity:
                params['quantity'] = quantity
            if price:
                params['price'] = price
            if trigger_price:
                params['trigger_price'] = trigger_price
            if order_type:
                params['order_type'] = order_type.value
            
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id,
                **params
            )
            print(f"✅ Order modified: {order_id}")
            return True
        except Exception as e:
            print(f"❌ Modify failed: {e}")
            return False
    
    def exit_position(self, symbol: str, reason: str = "MANUAL") -> bool:
        """
        Exit an open position at market
        """
        positions = self.rm.get_open_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if not position:
            print(f"⚠️ No open position for {symbol}")
            return False
        
        # Place exit order (opposite side)
        exit_side = "SELL" if position.side == "BUY" else "BUY"
        
        result = self.place_order(
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            product=ProductType.MIS,
            tag="EXIT"
        )
        
        if result.success:
            # Cancel pending SL/target orders
            if position.sl_order_id:
                self.cancel_order(position.sl_order_id)
            if position.target_order_id:
                self.cancel_order(position.target_order_id)
            
            # Wait for fill and get exit price
            time.sleep(0.5)
            order_history = self.kite.order_history(result.order_id)
            exit_price = order_history[-1].get('average_price', 0)
            
            # Close in risk manager
            self.rm.close_position(symbol, exit_price, reason)
            return True
        
        return False
    
    def exit_all_positions(self, reason: str = "EXIT_ALL") -> int:
        """Exit all open positions"""
        positions = self.rm.get_open_positions()
        closed = 0
        
        for pos in positions:
            if self.exit_position(pos.symbol, reason):
                closed += 1
        
        print(f"🛑 Exited {closed}/{len(positions)} positions")
        return closed
    
    def get_order_book(self) -> List[dict]:
        """Get today's order book"""
        try:
            return self.kite.orders()
        except Exception as e:
            print(f"❌ Error fetching orders: {e}")
            return []
    
    def get_positions(self) -> dict:
        """Get Zerodha positions"""
        try:
            return self.kite.positions()
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return {'net': [], 'day': []}
    
    def sync_positions_with_broker(self):
        """
        Sync risk manager positions with actual broker positions
        Important to call at startup and periodically
        """
        try:
            broker_positions = self.kite.positions()
            day_positions = broker_positions.get('day', [])
            
            for bp in day_positions:
                symbol = f"{bp['exchange']}:{bp['tradingsymbol']}"
                quantity = bp['quantity']
                
                if quantity != 0:
                    # Position exists at broker
                    if symbol not in [p.symbol for p in self.rm.get_open_positions()]:
                        print(f"⚠️ Found untracked position: {symbol} qty={quantity}")
                        # Could add logic to auto-add to risk manager
            
            print("✅ Positions synced with broker")
        except Exception as e:
            print(f"❌ Sync error: {e}")


# Singleton instance
_execution_engine = None

def get_execution_engine() -> ExecutionEngine:
    """Get singleton ExecutionEngine instance"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = ExecutionEngine()
    return _execution_engine


if __name__ == "__main__":
    # Test execution engine
    ee = get_execution_engine()
    dm = get_data_manager()
    
    if dm.is_authenticated():
        print("\n📊 Execution Engine Test")
        print("=" * 50)
        
        # Get order book
        orders = ee.get_order_book()
        print(f"Orders today: {len(orders)}")
        
        # Get positions
        positions = ee.get_positions()
        print(f"Day positions: {len(positions.get('day', []))}")
        
        # Test signal execution (dry run)
        from signal_engine import Signal, SignalType, StrategyType
        
        test_signal = Signal(
            symbol="NSE:RELIANCE",
            signal_type=SignalType.BUY,
            strategy=StrategyType.MA_CROSSOVER,
            entry_price=2500,
            stop_loss=2450,
            target=2600,
            strength=75,
            reason="Test signal",
            timestamp=datetime.now()
        )
        
        print("\n🧪 Testing signal execution (DRY RUN):")
        ee.execute_signal(test_signal, dry_run=True)
    else:
        print("❌ Not authenticated")
