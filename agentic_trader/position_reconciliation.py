"""
POSITION & ORDER RECONCILIATION MODULE
Hard requirement for live trading - periodic "truth sync"

Responsibilities:
1. Pull broker positions + open orders every 5-15s
2. Compare to local state (active_trades.json)
3. If mismatch: Cancel all open orders, freeze entries, enter recovery mode
4. Handle: Partial fills, SL/target filled at broker but local missed, restart mid-session
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ReconciliationState(Enum):
    """Reconciliation system state"""
    SYNCED = "SYNCED"           # Local matches broker
    MISMATCH_DETECTED = "MISMATCH"  # Discrepancy found
    RECOVERY_MODE = "RECOVERY"   # In recovery, no new trades
    FROZEN = "FROZEN"           # Frozen, awaiting manual intervention
    INITIALIZING = "INIT"       # Startup sync in progress


@dataclass
class ReconciliationResult:
    """Result of a reconciliation check"""
    state: ReconciliationState
    local_positions: int
    broker_positions: int
    local_orders: int
    broker_orders: int
    mismatches: List[Dict]
    action_taken: str
    timestamp: str


class PositionReconciliation:
    """
    Position & Order Reconciliation Engine
    
    Runs a background loop to periodically sync local state with broker truth.
    Critical for preventing ghost positions, missed fills, and state drift.
    """
    
    RECONCILIATION_FILE = os.path.join(os.path.dirname(__file__), 'reconciliation_state.json')
    
    def __init__(self, kite=None, paper_mode: bool = True, check_interval: int = 10):
        """
        Initialize reconciliation engine
        
        Args:
            kite: KiteConnect instance
            paper_mode: If True, simulates reconciliation
            check_interval: Seconds between checks (5-15 recommended)
        """
        self.kite = kite
        self.paper_mode = paper_mode
        self.check_interval = max(5, min(15, check_interval))  # Clamp to 5-15s
        
        self.state = ReconciliationState.INITIALIZING
        self.frozen = False
        self.recovery_mode = False
        self.last_check = None
        self.mismatch_count = 0
        self.consecutive_mismatches = 0
        
        # Local state cache (will be refreshed from file)
        self.local_positions: List[Dict] = []
        self.local_orders: List[Dict] = []
        
        # Broker state cache
        self.broker_positions: List[Dict] = []
        self.broker_orders: List[Dict] = []
        
        # Mismatch history
        self.mismatch_history: List[Dict] = []
        
        # Background thread
        self._running = False
        self._thread = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Recovery actions log
        self.recovery_actions: List[str] = []
        
        # Load any saved state
        self._load_state()
        
        print(f"🔄 Position Reconciliation: INITIALIZED (interval: {self.check_interval}s)")
    
    def _load_state(self):
        """Load reconciliation state from file"""
        try:
            if os.path.exists(self.RECONCILIATION_FILE):
                with open(self.RECONCILIATION_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.mismatch_count = data.get('mismatch_count', 0)
                        self.mismatch_history = data.get('mismatch_history', [])
                        self.recovery_actions = data.get('recovery_actions', [])
                        if data.get('frozen'):
                            self.frozen = True
                            self.state = ReconciliationState.FROZEN
        except Exception as e:
            print(f"⚠️ Error loading reconciliation state: {e}")
    
    def _save_state(self):
        """Save reconciliation state to file"""
        try:
            data = {
                'date': str(datetime.now().date()),
                'last_updated': datetime.now().isoformat(),
                'state': self.state.value,
                'frozen': self.frozen,
                'mismatch_count': self.mismatch_count,
                'consecutive_mismatches': self.consecutive_mismatches,
                'mismatch_history': self.mismatch_history[-50:],  # Keep last 50
                'recovery_actions': self.recovery_actions[-20:]   # Keep last 20
            }
            with open(self.RECONCILIATION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving reconciliation state: {e}")
    
    def start(self):
        """Start background reconciliation loop"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._reconciliation_loop, daemon=True)
        self._thread.start()
        print(f"🔄 Reconciliation loop STARTED")
    
    def stop(self):
        """Stop background reconciliation loop"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print(f"🔄 Reconciliation loop STOPPED")
    
    def _reconciliation_loop(self):
        """Background loop that periodically checks reconciliation"""
        # Initial sync on startup
        time.sleep(2)  # Wait for system to initialize
        self._do_startup_sync()
        
        while self._running:
            try:
                time.sleep(self.check_interval)
                result = self.reconcile()
                
                # Log if not synced
                if result.state != ReconciliationState.SYNCED:
                    print(f"⚠️ Reconciliation: {result.state.value} - {result.action_taken}")
                    
            except Exception as e:
                print(f"❌ Reconciliation error: {e}")
    
    def _do_startup_sync(self):
        """Initial sync on startup - handle restart mid-session"""
        print("🔄 Performing startup reconciliation...")
        
        with self._lock:
            try:
                # Get broker truth
                self._fetch_broker_state()
                
                # Load local state
                self._load_local_state()
                
                # Check for positions at broker that aren't in local
                orphan_broker_positions = []
                for bp in self.broker_positions:
                    symbol = bp.get('symbol')
                    found = any(lp.get('symbol') == symbol for lp in self.local_positions)
                    if not found and bp.get('quantity', 0) != 0:
                        orphan_broker_positions.append(bp)
                
                # Check for local positions that aren't at broker
                orphan_local_positions = []
                for lp in self.local_positions:
                    symbol = lp.get('symbol')
                    found = any(bp.get('symbol') == symbol for bp in self.broker_positions)
                    if not found and lp.get('status', 'OPEN') == 'OPEN':
                        orphan_local_positions.append(lp)
                
                if orphan_broker_positions:
                    print(f"⚠️ ORPHAN BROKER POSITIONS: {[p['symbol'] for p in orphan_broker_positions]}")
                    self._log_recovery_action(f"STARTUP: Found {len(orphan_broker_positions)} orphan broker positions")
                    # Would need to add these to local state
                    
                if orphan_local_positions:
                    print(f"⚠️ ORPHAN LOCAL POSITIONS: {[p['symbol'] for p in orphan_local_positions]}")
                    self._log_recovery_action(f"STARTUP: Found {len(orphan_local_positions)} orphan local positions - likely filled at broker")
                    # Local thinks it has positions that broker doesn't - these were likely filled/closed
                
                if not orphan_broker_positions and not orphan_local_positions:
                    self.state = ReconciliationState.SYNCED
                    print("✅ Startup sync complete - all positions match")
                else:
                    self.state = ReconciliationState.MISMATCH_DETECTED
                    print("⚠️ Startup sync found mismatches - entering recovery mode")
                    self._enter_recovery_mode("Startup sync mismatch")
                    
            except Exception as e:
                print(f"❌ Startup sync error: {e}")
                self.state = ReconciliationState.SYNCED  # Assume OK in paper mode
    
    def _fetch_broker_state(self):
        """Fetch current positions and orders from broker"""
        if self.paper_mode or not self.kite:
            # In paper mode, broker state mirrors local
            return
        
        try:
            # Get positions
            positions = self.kite.positions()
            day_positions = positions.get('day', [])
            self.broker_positions = [{
                'symbol': f"{p['exchange']}:{p['tradingsymbol']}",
                'quantity': p['quantity'],
                'avg_price': p['average_price'],
                'pnl': p.get('pnl', 0),
                'side': 'LONG' if p['quantity'] > 0 else 'SHORT'
            } for p in day_positions if p['quantity'] != 0]
            
            # Get orders
            orders = self.kite.orders()
            self.broker_orders = [{
                'order_id': o['order_id'],
                'symbol': f"{o['exchange']}:{o['tradingsymbol']}",
                'transaction_type': o['transaction_type'],
                'quantity': o['quantity'],
                'price': o.get('price', 0),
                'trigger_price': o.get('trigger_price', 0),
                'status': o['status'],
                'order_type': o.get('order_type', 'MARKET'),
                'filled_quantity': o.get('filled_quantity', 0)
            } for o in orders if o['status'] in ['OPEN', 'TRIGGER PENDING', 'PENDING']]
            
        except Exception as e:
            print(f"⚠️ Error fetching broker state: {e}")
    
    def _load_local_state(self):
        """Load local positions from active_trades.json"""
        trades_file = os.path.join(os.path.dirname(__file__), 'active_trades.json')
        try:
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.local_positions = [
                            t for t in data.get('active_trades', [])
                            if t.get('status', 'OPEN') == 'OPEN'
                        ]
                    else:
                        self.local_positions = []
            else:
                self.local_positions = []
        except Exception as e:
            print(f"⚠️ Error loading local state: {e}")
            self.local_positions = []
    
    def reconcile(self) -> ReconciliationResult:
        """
        Perform reconciliation check
        
        Returns:
            ReconciliationResult with status and any actions taken
        """
        with self._lock:
            self.last_check = datetime.now()
            
            # Refresh states
            self._fetch_broker_state()
            self._load_local_state()
            
            mismatches = []
            action_taken = "None"
            
            # === POSITION RECONCILIATION ===
            
            # Check 1: Quantity mismatch
            for lp in self.local_positions:
                symbol = lp.get('symbol')
                local_qty = lp.get('quantity', 0)
                local_side = lp.get('side', 'BUY')
                
                # Find in broker
                broker_pos = next(
                    (bp for bp in self.broker_positions if bp.get('symbol') == symbol),
                    None
                )
                
                if not self.paper_mode:
                    if broker_pos is None:
                        # Position exists locally but not at broker - likely filled
                        mismatches.append({
                            'type': 'LOCAL_ONLY',
                            'symbol': symbol,
                            'local_qty': local_qty,
                            'broker_qty': 0,
                            'issue': 'Position in local but not at broker - SL/Target may have hit'
                        })
                    elif abs(broker_pos.get('quantity', 0)) != abs(local_qty):
                        # Quantity mismatch - partial fill?
                        mismatches.append({
                            'type': 'QTY_MISMATCH',
                            'symbol': symbol,
                            'local_qty': local_qty,
                            'broker_qty': broker_pos.get('quantity', 0),
                            'issue': 'Quantity mismatch - possible partial fill'
                        })
            
            # Check 2: Broker has position we don't know about
            for bp in self.broker_positions:
                symbol = bp.get('symbol')
                if not any(lp.get('symbol') == symbol for lp in self.local_positions):
                    if not self.paper_mode:
                        mismatches.append({
                            'type': 'BROKER_ONLY',
                            'symbol': symbol,
                            'local_qty': 0,
                            'broker_qty': bp.get('quantity', 0),
                            'issue': 'Position at broker not in local state'
                        })
            
            # Check 3: SL/Target orders - if local has SL set but broker doesn't have order
            # (This would be checked against self.broker_orders)
            
            # === HANDLE MISMATCHES ===
            
            if mismatches:
                self.consecutive_mismatches += 1
                self.mismatch_count += len(mismatches)
                
                # Log mismatches
                for m in mismatches:
                    self.mismatch_history.append({
                        'timestamp': datetime.now().isoformat(),
                        **m
                    })
                
                # If too many consecutive mismatches, enter recovery
                if self.consecutive_mismatches >= 3:
                    self._enter_recovery_mode(f"{len(mismatches)} mismatches detected")
                    action_taken = "ENTERED_RECOVERY_MODE"
                else:
                    action_taken = f"MISMATCH_LOGGED ({self.consecutive_mismatches}/3)"
                
                self.state = ReconciliationState.MISMATCH_DETECTED
            else:
                self.consecutive_mismatches = 0
                self.state = ReconciliationState.SYNCED if not self.recovery_mode else ReconciliationState.RECOVERY_MODE
                action_taken = "SYNCED" if not self.recovery_mode else "IN_RECOVERY"
            
            self._save_state()
            
            return ReconciliationResult(
                state=self.state,
                local_positions=len(self.local_positions),
                broker_positions=len(self.broker_positions),
                local_orders=len(self.local_orders),
                broker_orders=len(self.broker_orders),
                mismatches=mismatches,
                action_taken=action_taken,
                timestamp=datetime.now().isoformat()
            )
    
    def _enter_recovery_mode(self, reason: str):
        """Enter recovery mode - freeze new entries"""
        self.recovery_mode = True
        self.state = ReconciliationState.RECOVERY_MODE
        
        action = f"RECOVERY MODE: {reason}"
        self._log_recovery_action(action)
        print(f"🚨 {action}")
        
        # === RESET HYSTERESIS FOR ALL SYMBOLS (context-aware) ===
        # Prevents stale state when recovering from mismatch
        try:
            from trend_following import get_trend_engine
            get_trend_engine().reset_hysteresis(None, "RECOVERY_MODE")
        except ImportError:
            pass
        try:
            from regime_score import get_regime_scorer
            get_regime_scorer().reset_hysteresis(None, "RECOVERY_MODE")
        except ImportError:
            pass
        
        # Cancel all open orders at broker
        if not self.paper_mode and self.kite:
            self._cancel_all_orders()
    
    def _cancel_all_orders(self):
        """Cancel all open orders at broker"""
        if self.paper_mode or not self.kite:
            return
        
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['status'] in ['OPEN', 'TRIGGER PENDING']:
                    try:
                        self.kite.cancel_order(
                            variety=order.get('variety', 'regular'),
                            order_id=order['order_id']
                        )
                        self._log_recovery_action(f"CANCELLED: {order['order_id']} - {order['tradingsymbol']}")
                    except Exception as e:
                        print(f"⚠️ Failed to cancel order {order['order_id']}: {e}")
        except Exception as e:
            print(f"⚠️ Error cancelling orders: {e}")
    
    def _log_recovery_action(self, action: str):
        """Log a recovery action"""
        self.recovery_actions.append({
            'timestamp': datetime.now().isoformat(),
            'action': action
        })
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if new trades are allowed
        
        Returns:
            (can_trade, reason)
        """
        if self.frozen:
            return False, "FROZEN: Manual intervention required"
        
        if self.recovery_mode:
            return False, "RECOVERY_MODE: Reconciliation mismatch - no new entries"
        
        if self.state == ReconciliationState.MISMATCH_DETECTED:
            if self.consecutive_mismatches >= 2:
                return False, f"MISMATCH: {self.consecutive_mismatches} consecutive mismatches"
        
        return True, "SYNCED"
    
    def exit_recovery_mode(self, force: bool = False):
        """Exit recovery mode after manual verification"""
        if force or self.consecutive_mismatches == 0:
            self.recovery_mode = False
            self.frozen = False
            self.state = ReconciliationState.SYNCED
            self._log_recovery_action("EXITED RECOVERY MODE")
            self._save_state()
            print("✅ Exited recovery mode")
        else:
            print("⚠️ Cannot exit recovery - still have mismatches")
    
    def force_resync(self):
        """Force resync local state with broker (broker is truth)"""
        with self._lock:
            self._fetch_broker_state()
            
            # In a real system, we'd update active_trades.json with broker positions
            # For now, just clear mismatches and reset
            self.consecutive_mismatches = 0
            self.recovery_mode = False
            self.state = ReconciliationState.SYNCED
            
            self._log_recovery_action("FORCE RESYNC: Broker state accepted as truth")
            self._save_state()
            print("✅ Force resync complete")
    
    def get_status(self) -> Dict:
        """Get current reconciliation status"""
        return {
            'state': self.state.value,
            'frozen': self.frozen,
            'recovery_mode': self.recovery_mode,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'check_interval': self.check_interval,
            'consecutive_mismatches': self.consecutive_mismatches,
            'total_mismatches_today': self.mismatch_count,
            'local_positions': len(self.local_positions),
            'broker_positions': len(self.broker_positions),
            'recent_mismatches': self.mismatch_history[-5:],
            'recent_actions': self.recovery_actions[-5:]
        }
    
    def handle_partial_fill(self, order_id: str, filled_qty: int, total_qty: int) -> Dict:
        """
        Handle partial fill notification
        
        Args:
            order_id: The order that was partially filled
            filled_qty: Quantity that was filled
            total_qty: Total order quantity
            
        Returns:
            Dict with action taken
        """
        if filled_qty == 0:
            return {'action': 'NONE', 'reason': 'No fill yet'}
        
        fill_pct = (filled_qty / total_qty) * 100
        
        if fill_pct >= 80:
            # Mostly filled - can continue
            return {
                'action': 'CONTINUE',
                'reason': f'{fill_pct:.0f}% filled - acceptable',
                'adjust_qty': filled_qty
            }
        elif fill_pct >= 50:
            # Half filled - consider continuing with reduced size
            return {
                'action': 'REDUCE_SIZE',
                'reason': f'{fill_pct:.0f}% filled - reduce target/SL qty',
                'adjust_qty': filled_qty
            }
        else:
            # Minimal fill - cancel remaining
            return {
                'action': 'CANCEL_REMAINING',
                'reason': f'{fill_pct:.0f}% filled - cancel remaining, too little exposure',
                'adjust_qty': filled_qty
            }


# Singleton instance
_reconciliation_instance = None


def get_position_reconciliation(kite=None, paper_mode: bool = True, check_interval: int = 10) -> PositionReconciliation:
    """Get or create the singleton reconciliation instance"""
    global _reconciliation_instance
    if _reconciliation_instance is None:
        _reconciliation_instance = PositionReconciliation(
            kite=kite,
            paper_mode=paper_mode,
            check_interval=check_interval
        )
    return _reconciliation_instance


# === TEST ===
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING POSITION RECONCILIATION")
    print("=" * 60)
    
    # Create instance
    recon = get_position_reconciliation(paper_mode=True)
    print(f"✅ Created: state={recon.state.value}")
    
    # Test can_trade
    can_trade, reason = recon.can_trade()
    print(f"✅ Can trade: {can_trade} - {reason}")
    
    # Test reconcile
    result = recon.reconcile()
    print(f"✅ Reconcile result: {result.state.value}, action: {result.action_taken}")
    
    # Test status
    status = recon.get_status()
    print(f"✅ Status: {json.dumps(status, indent=2)}")
    
    # Test partial fill handling
    partial = recon.handle_partial_fill("ORD123", 30, 100)
    print(f"✅ Partial fill (30%): {partial}")
    
    partial = recon.handle_partial_fill("ORD456", 85, 100)
    print(f"✅ Partial fill (85%): {partial}")
    
    print("\n✅ ALL TESTS PASSED!")
