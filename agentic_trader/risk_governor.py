"""
RISK GOVERNOR MODULE
Account-level risk management with kill-switch capability

Features:
1. Max daily loss limit
2. Max consecutive losses
3. Max trades per day
4. Max symbol exposure (avoid correlated bets)
5. Cooldown after loss
6. Circuit breaker (data/order issues)
"""

from datetime import datetime, timedelta, time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import os
import logging
from state_db import get_state_db

logger = logging.getLogger('risk_governor')


class SystemState(Enum):
    """Trading system state"""
    ACTIVE = "ACTIVE"              # Normal trading
    COOLDOWN = "COOLDOWN"          # Temporary pause after loss
    HALT_TRADING = "HALT_TRADING"  # Stopped for the day
    CIRCUIT_BREAK = "CIRCUIT_BREAK"  # Emergency stop


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_daily_loss_pct: float = 20.0     # Max 20% daily loss
    max_consecutive_losses: int = 8       # Max 8 losses in a row
    max_trades_per_day: int = 80          # Max 80 trades per day
    max_symbol_exposure: int = 2          # Max 2 positions in same sector/correlated
    cooldown_minutes: int = 0             # No cooldown after loss (disabled per user request)
    max_position_pct: float = 25.0        # Max 25% of capital in one position
    max_total_exposure_pct: float = 80.0  # Max 80% of capital deployed


@dataclass
class RiskState:
    """Current risk state"""
    system_state: str = "ACTIVE"
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0
    last_trade_time: str = ""
    last_loss_time: str = ""
    cooldown_until: str = ""
    halt_reason: str = ""
    positions_by_sector: Dict = field(default_factory=dict)
    order_rejections: int = 0
    data_stale_count: int = 0


@dataclass
class TradePermission:
    """Result of trade permission check"""
    allowed: bool
    reason: str
    warnings: List[str] = field(default_factory=list)
    suggested_size_multiplier: float = 1.0  # Reduce size if near limits


class RiskGovernor:
    """
    Account-level risk governor with kill-switch
    
    Flow:
    1. Check before each trade → can_trade()
    2. Update after each trade → record_trade_result()
    3. Monitor for circuit breakers → check_circuit_breakers()
    """
    
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # Configuration
        self.limits = RiskLimits()
        
        # State
        self.state = RiskState()
        self.state_file = "risk_state.json"
        self.trade_date = datetime.now().date()
        self._cached_unrealized_pnl = 0.0  # Updated by is_trading_allowed() with live ticker data
        
        # Sector/correlation mapping - MUST cover ALL APPROVED_UNIVERSE symbols
        self.sector_map = {
            # IT
            "NSE:INFY": "IT", "NSE:TCS": "IT", "NSE:WIPRO": "IT", 
            "NSE:HCLTECH": "IT", "NSE:TECHM": "IT", "NSE:LTIM": "IT",
            # Banking
            "NSE:HDFCBANK": "BANK", "NSE:ICICIBANK": "BANK", "NSE:AXISBANK": "BANK",
            "NSE:KOTAKBANK": "BANK", "NSE:SBIN": "BANK", "NSE:INDUSINDBK": "BANK",
            "NSE:UNIONBANK": "BANK", "NSE:BANKBARODA": "BANK", "NSE:PNB": "BANK",
            # Telecom
            "NSE:BHARTIARTL": "TELECOM", "NSE:IDEA": "TELECOM",
            # Auto
            "NSE:M&M": "AUTO", "NSE:MARUTI": "AUTO",
            "NSE:BAJAJ-AUTO": "AUTO", "NSE:HEROMOTOCO": "AUTO", "NSE:EICHERMOT": "AUTO",
            # Pharma
            "NSE:SUNPHARMA": "PHARMA", "NSE:DRREDDY": "PHARMA", "NSE:CIPLA": "PHARMA",
            # Metal
            "NSE:TATASTEEL": "METAL", "NSE:HINDALCO": "METAL", "NSE:JSWSTEEL": "METAL",
            "NSE:VEDL": "METAL", "NSE:JINDALSTEL": "METAL", "NSE:NMDC": "METAL",
            "NSE:NATIONALUM": "METAL", "NSE:SAIL": "METAL", "NSE:HINDCOPPER": "METAL",
            # Oil & Gas
            "NSE:RELIANCE": "OIL", "NSE:ONGC": "OIL", "NSE:BPCL": "OIL",
            # Consumer
            "NSE:ASIANPAINT": "CONSUMER", "NSE:TITAN": "CONSUMER",
            "NSE:ITC": "CONSUMER", "NSE:HINDUNILVR": "CONSUMER",
            "NSE:ETERNAL": "CONSUMER",
            # NBFC
            "NSE:BAJFINANCE": "NBFC", "NSE:BAJAJFINSV": "NBFC",
            # Power / Infra
            "NSE:NHPC": "POWER", "NSE:POWERGRID": "POWER", "NSE:NTPC": "POWER",
            "NSE:ADANIPOWER": "POWER", "NSE:TATAPOWER": "POWER",
            # Misc cash
            "NSE:MCX": "EXCHANGE", "NSE:IRFC": "FINANCE",
            "NSE:YESBANK": "BANK", "NSE:CANBK": "BANK",
            # ETFs (separate sector to allow multiple)
            "NSE:NIFTYBEES": "ETF", "NSE:BANKBEES": "ETF", "NSE:GOLDBEES": "ETF",
            "NSE:SILVERBEES": "ETF", "NSE:ITBEES": "ETF", "NSE:JUNIORBEES": "ETF",
            "NSE:CPSEETF": "ETF", "NSE:PSUBNKBEES": "ETF", "NSE:NEXT50": "ETF",
            "NSE:PHARMABEES": "ETF", "NSE:INFRABEES": "ETF",
        }
        
        # Sector limits (how many positions allowed per sector)
        self.sector_limits = {
            "BANK": 3, "IT": 2, "METAL": 3, "ETF": 4,
            "OIL": 2, "PHARMA": 2, "AUTO": 2, "CONSUMER": 2,
            "TELECOM": 2, "NBFC": 2, "POWER": 2, "OTHER": 2,
        }
        
        self._load_state()
        self._check_new_day()
        self._reconcile_pnl_from_history()
        self._reeval_halt()
    
    def _reeval_halt(self, active_positions: List[Dict] = None, unrealized_pnl: float = 0.0):
        """Re-evaluate HALT_TRADING after state load + reconciliation.
        If NET P&L (realized + unrealized) is within limits, clear the halt.
        This prevents the asymmetry where SL cuts realize losses fast while
        open winners (target extension / trailing) stay unrealized.
        
        Args:
            active_positions: Open positions (legacy, used for fallback calc)
            unrealized_pnl: Pre-computed unrealized P&L from ticker quotes (preferred)
        """
        if self.state.system_state != SystemState.HALT_TRADING.value:
            return
        # Only re-evaluate daily-loss halts (not consecutive-loss or circuit-break halts)
        _is_loss_halt = 'daily loss' in (self.state.halt_reason or '').lower() or 'Max daily loss' in (self.state.halt_reason or '')
        if not _is_loss_halt:
            return
        # Use pre-computed unrealized if provided, otherwise try from positions
        unrealized = unrealized_pnl if unrealized_pnl != 0.0 else (
            self._calc_unrealized_pnl(active_positions) if active_positions else 0.0
        )
        net_pnl = self.state.daily_pnl + unrealized
        net_pnl_pct = (net_pnl / self.starting_capital) * 100 if self.starting_capital else 0.0
        if abs(net_pnl_pct) < self.limits.max_daily_loss_pct:
            # print(f"✅ Risk Governor: Net P&L ({net_pnl_pct:+.2f}%) is within {self.limits.max_daily_loss_pct}% limit "
            #       f"(realized {self.state.daily_pnl_pct:+.2f}% + unrealized ₹{unrealized:+,.0f}) — HALT CLEARED")
            self.state.system_state = SystemState.ACTIVE.value
            self.state.halt_reason = ""
            self._save_state()
        else:
            # Also check realized-only in case unrealized is 0 (positions not loaded yet)
            if abs(self.state.daily_pnl_pct) < self.limits.max_daily_loss_pct:
                # print(f"✅ Risk Governor: Realized P&L ({self.state.daily_pnl_pct:+.2f}%) is within {self.limits.max_daily_loss_pct}% limit — HALT CLEARED")
                self.state.system_state = SystemState.ACTIVE.value
                self.state.halt_reason = ""
                self._save_state()
    
    def _load_state(self):
        """Load state from SQLite (falls back to JSON for migration)"""
        try:
            data = get_state_db().load_risk_state()
            if data:
                self.state = RiskState(**{k: v for k, v in data.items() if k != 'date'})
                return
        except Exception as e:
            print(f"⚠️ Error loading risk state from SQLite: {e}")
        # Fallback: legacy JSON (pre-migration)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.state = RiskState(**{k: v for k, v in data.items() if k != 'date'})
            except Exception as e:
                print(f"⚠️ Error loading risk state from JSON: {e}")
    
    def _reconcile_pnl_from_history(self):
        """Reconcile daily_pnl using active_trades realized_pnl as single source of truth.
        
        SQLite daily_state realized_pnl is updated by zerodha_tools.paper_pnl on EVERY
        trade close (before archival), so it always reflects the true cumulative P&L.
        
        Falls back to Trade Ledger exits if SQLite is unavailable.
        """
        today_str = str(datetime.now().date())
        
        # --- PRIMARY SOURCE: SQLite daily_state realized_pnl ---
        try:
            db = get_state_db()
            _positions, active_pnl, _cap = db.load_active_trades(today_str)
            if active_pnl is not None:
                old_pnl = self.state.daily_pnl
                if abs(old_pnl - active_pnl) > 1.0:
                    self.state.daily_pnl = active_pnl
                    self.state.daily_pnl_pct = (active_pnl / self.starting_capital) * 100
                    self.current_capital = self.starting_capital + active_pnl
                    # Re-evaluate halt: if reconciled P&L is within limits, clear the halt
                    if self.state.system_state == SystemState.HALT_TRADING.value:
                        reconciled_pct = abs(self.state.daily_pnl_pct)
                        if reconciled_pct < self.limits.max_daily_loss_pct:
                            self.state.system_state = SystemState.ACTIVE.value
                            self.state.halt_reason = ""
                    self._save_state()
                return
        except Exception as e:
            print(f"⚠️ Risk Governor: active_trades reconciliation failed: {e}")
        
        # --- FALLBACK: Trade Ledger today's exits ---
        try:
            from trade_ledger import get_trade_ledger
            exits = get_trade_ledger().get_exits()
            if exits:
                today_pnl = sum(e.get('pnl', 0) for e in exits)
                old_pnl = self.state.daily_pnl
                if abs(old_pnl - today_pnl) > 1.0:
                    self.state.daily_pnl = today_pnl
                    self.state.daily_pnl_pct = (today_pnl / self.starting_capital) * 100
                    self.current_capital = self.starting_capital + today_pnl
                    # Re-evaluate halt: if reconciled P&L is within limits, clear the halt
                    if self.state.system_state == SystemState.HALT_TRADING.value:
                        reconciled_pct = abs(self.state.daily_pnl_pct)
                        if reconciled_pct < self.limits.max_daily_loss_pct:
                            self.state.system_state = SystemState.ACTIVE.value
                            self.state.halt_reason = ""
                            # print(f"✅ Risk Governor: Reconciled P&L ({self.state.daily_pnl_pct:+.2f}%) is within {self.limits.max_daily_loss_pct}% limit — HALT CLEARED")
                    self._save_state()
                    # print(f"📊 Risk Governor: Reconciled daily P&L from Trade Ledger (fallback): ₹{old_pnl:+,.0f} → ₹{today_pnl:+,.0f}")
        except Exception as e:
            print(f"⚠️ Risk Governor: P&L reconciliation failed: {e}")
    
    def _save_state(self):
        """Save state to SQLite (atomic, crash-safe)"""
        try:
            data = asdict(self.state)
            data['date'] = str(datetime.now().date())
            get_state_db().save_risk_state(data)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
    
    def _check_new_day(self):
        """Reset state if new trading day"""
        today = datetime.now().date()
        if today != self.trade_date:
            print(f"📅 New trading day - resetting risk state")
            self.state = RiskState()
            self.trade_date = today
            self._save_state()
    
    def update_capital(self, new_capital: float):
        """Update current capital tracking (does NOT overwrite daily_pnl).
        
        daily_pnl is maintained by record_trade_result() and reconciled
        from Trade Ledger on startup. We only track current_capital here.
        """
        self.current_capital = new_capital
    
    def _calc_unrealized_pnl(self, active_positions: List[Dict]) -> float:
        """Calculate total unrealized P&L from open positions"""
        unrealized = 0.0
        for pos in active_positions:
            if pos.get('status', 'OPEN') != 'OPEN':
                continue
            pnl = pos.get('pnl', 0)
            if pnl is not None and pnl != 0:
                unrealized += pnl
                continue
            # Fallback: compute from avg_price and ltp
            qty = pos.get('quantity', 0)
            entry = pos.get('avg_price', 0)
            ltp = pos.get('ltp', entry)
            side = pos.get('side', 'BUY')
            if side == 'BUY':
                unrealized += (ltp - entry) * qty
            else:
                unrealized += (entry - ltp) * qty
        return unrealized
    
    def can_trade_general(self, active_positions: List[Dict], setup_type: str = '') -> 'TradePermission':
        """
        Lightweight pre-scan check: can the system trade AT ALL?
        Checks system state, cooldown, daily loss, consecutive losses, trade count.
        Does NOT check sector/symbol exposure (that's done per-candidate).
        
        For GMM_SNIPER trades: skips total exposure check (they have separate ₹3L capital pool).
        """
        self._check_new_day()
        
        if self.state.system_state == SystemState.HALT_TRADING.value:
            return TradePermission(allowed=False, reason=f"Trading halted: {self.state.halt_reason}", warnings=[])
        
        if self.state.system_state == SystemState.CIRCUIT_BREAK.value:
            return TradePermission(allowed=False, reason=f"Circuit breaker active: {self.state.halt_reason}", warnings=[])
        
        if self.state.system_state == SystemState.COOLDOWN.value:
            if self.state.cooldown_until:
                cooldown_end = datetime.fromisoformat(self.state.cooldown_until)
                if datetime.now() < cooldown_end:
                    mins_left = (cooldown_end - datetime.now()).seconds // 60
                    return TradePermission(allowed=False, reason=f"Cooldown active: {mins_left} minutes remaining", warnings=[])
                else:
                    self.state.system_state = SystemState.ACTIVE.value
                    self._save_state()
        
        # Daily loss check using NET P&L (realized + unrealized)
        # Use cached unrealized from is_trading_allowed() (live ticker data),
        # falling back to _calc_unrealized_pnl (which may return 0 if positions lack LTP).
        unrealized_pnl = self._cached_unrealized_pnl if self._cached_unrealized_pnl != 0.0 else self._calc_unrealized_pnl(active_positions)
        net_pnl = self.state.daily_pnl + unrealized_pnl
        net_pnl_pct = (net_pnl / self.starting_capital) * 100
        
        if net_pnl_pct <= -self.limits.max_daily_loss_pct:
            return TradePermission(allowed=False, reason=f"Max daily loss exceeded: net {net_pnl_pct:.2f}% (realized {self.state.daily_pnl_pct:.2f}% + unrealized ₹{unrealized_pnl:+,.0f})", warnings=[])
        
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            return TradePermission(allowed=False, reason=f"Max consecutive losses: {self.state.consecutive_losses}", warnings=[])
        
        if self.state.trades_today >= self.limits.max_trades_per_day:
            return TradePermission(allowed=False, reason=f"Max trades per day reached: {self.state.trades_today}", warnings=[])
        
        # Total exposure check — use max_risk for IC positions (avg_price is just credit, understates risk)
        # GMM_SNIPER trades have separate capital — skip main exposure check for them
        if setup_type != 'GMM_SNIPER':
            total_exposure = sum(
                p.get('max_risk', p.get('avg_price', 0) * p.get('quantity', 0))
                if p.get('is_iron_condor', False)
                else p.get('avg_price', 0) * p.get('quantity', 0)
                for p in active_positions
                if not p.get('is_sniper', False)  # Exclude sniper positions from main exposure
            )
            total_exposure_pct = (total_exposure / self.current_capital) * 100
            if total_exposure_pct > self.limits.max_total_exposure_pct:
                return TradePermission(allowed=False, reason=f"Max total exposure: {total_exposure_pct:.1f}%", warnings=[])
        else:
            # Sniper-specific exposure check against separate capital
            from config import GMM_SNIPER
            sniper_capital = GMM_SNIPER.get('separate_capital', 300000)
            sniper_max_pct = GMM_SNIPER.get('max_exposure_pct', 90)
            sniper_exposure = sum(
                p.get('avg_price', 0) * p.get('quantity', 0)
                for p in active_positions
                if p.get('is_sniper', False)
            )
            sniper_exposure_pct = (sniper_exposure / sniper_capital) * 100
            if sniper_exposure_pct > sniper_max_pct:
                return TradePermission(allowed=False, reason=f"Sniper capital exhausted: {sniper_exposure_pct:.1f}% of ₹{sniper_capital/100000:.0f}L", warnings=[])
        
        warnings = []
        if self.state.trades_today == self.limits.max_trades_per_day - 1:
            warnings.append("Last trade allowed today")
        if self.state.daily_pnl_pct <= -(self.limits.max_daily_loss_pct * 0.5) and unrealized_pnl > 0:
            warnings.append(f"⚠️ Realized P&L stressed ({self.state.daily_pnl_pct:.1f}%), open winners keeping net OK ({net_pnl_pct:.1f}%)")
        return TradePermission(allowed=True, reason="", warnings=warnings, suggested_size_multiplier=1.0)
    
    def can_trade(
        self, 
        symbol: str, 
        position_value: float,
        active_positions: List[Dict]
    ) -> TradePermission:
        """
        Check if a new trade is allowed
        
        Args:
            symbol: Symbol to trade
            position_value: Value of proposed position
            active_positions: List of current open positions
        
        Returns:
            TradePermission with allowed/reason/warnings
        """
        self._check_new_day()
        warnings = []
        size_multiplier = 1.0
        
        # 1. Check system state
        if self.state.system_state == SystemState.HALT_TRADING.value:
            return TradePermission(
                allowed=False,
                reason=f"Trading halted: {self.state.halt_reason}",
                warnings=["System in HALT state"]
            )
        
        if self.state.system_state == SystemState.CIRCUIT_BREAK.value:
            return TradePermission(
                allowed=False,
                reason=f"Circuit breaker active: {self.state.halt_reason}",
                warnings=["Circuit breaker triggered"]
            )
        
        # 2. Check cooldown
        if self.state.system_state == SystemState.COOLDOWN.value:
            if self.state.cooldown_until:
                cooldown_end = datetime.fromisoformat(self.state.cooldown_until)
                if datetime.now() < cooldown_end:
                    mins_left = (cooldown_end - datetime.now()).seconds // 60
                    return TradePermission(
                        allowed=False,
                        reason=f"Cooldown active: {mins_left} minutes remaining",
                        warnings=[f"Lost last trade, cooling down"]
                    )
                else:
                    # Cooldown expired
                    self.state.system_state = SystemState.ACTIVE.value
                    self._save_state()
        
        # 3. Check max daily loss using NET P&L (realized + unrealized)
        # Profits run (target extension / trailing stops) while losses cut fast (SL),
        # so realized P&L skews negative. Using net P&L prevents premature halts
        # when open winners offset closed losers.
        # Use cached unrealized from is_trading_allowed() (live ticker data),
        # falling back to _calc_unrealized_pnl (which may return 0 if positions lack LTP).
        _unrealized = self._cached_unrealized_pnl if self._cached_unrealized_pnl != 0.0 else self._calc_unrealized_pnl(active_positions)
        _net_pnl = self.state.daily_pnl + _unrealized
        _net_pnl_pct = (_net_pnl / self.starting_capital) * 100 if self.starting_capital else 0.0
        
        if _net_pnl_pct <= -self.limits.max_daily_loss_pct:
            self._halt_trading(f"Max daily loss hit: net {_net_pnl_pct:.2f}% (realized {self.state.daily_pnl_pct:.2f}% + unrealized ₹{_unrealized:+,.0f})")
            return TradePermission(
                allowed=False,
                reason=f"Max daily loss exceeded: net {_net_pnl_pct:.2f}% (realized {self.state.daily_pnl_pct:.2f}%)",
                warnings=["Daily loss limit reached"]
            )
        
        # Warn if realized alone would trigger halt but unrealized saves it
        if self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct and _unrealized > 0:
            warnings.append(f"⚠️ Realized P&L {self.state.daily_pnl_pct:.1f}% past limit, open winners keeping net OK ({_net_pnl_pct:.1f}%)")
        
        # 4. Check consecutive losses
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt_trading(f"Max consecutive losses: {self.state.consecutive_losses}")
            return TradePermission(
                allowed=False,
                reason=f"Max consecutive losses: {self.state.consecutive_losses}",
                warnings=["Too many losses in a row"]
            )
        
        # 5. Check max trades per day
        if self.state.trades_today >= self.limits.max_trades_per_day:
            return TradePermission(
                allowed=False,
                reason=f"Max trades per day reached: {self.state.trades_today}",
                warnings=["Trade limit reached for today"]
            )
        
        # 6. Check symbol/sector exposure
        # For NFO symbols, map via underlying (e.g. NFO:SBIN26FEB1140CE → NSE:SBIN)
        def _get_sector(sym):
            if sym.startswith('NFO:'):
                # Extract underlying from NFO trading symbol
                ts = sym.replace('NFO:', '')
                for base in self.sector_map:
                    base_name = base.replace('NSE:', '')
                    if ts.startswith(base_name):
                        return self.sector_map[base]
            return self.sector_map.get(sym, 'OTHER')
        
        sector = _get_sector(symbol)
        sector_positions = sum(
            1 for p in active_positions 
            if _get_sector(p.get('symbol', '')) == sector
        )
        
        max_in_sector = self.sector_limits.get(sector, self.limits.max_symbol_exposure)
        if sector_positions >= max_in_sector:
            return TradePermission(
                allowed=False,
                reason=f"Max exposure in {sector} sector: {sector_positions}/{max_in_sector} positions",
                warnings=[f"Too many {sector} positions"]
            )
        
        # 7. Check position size limit
        position_pct = (position_value / self.current_capital) * 100
        if position_pct > self.limits.max_position_pct:
            # Reduce size instead of blocking
            size_multiplier = self.limits.max_position_pct / position_pct
            warnings.append(f"Position size reduced: {position_pct:.1f}% → {self.limits.max_position_pct}%")
        
        # 8. Check total exposure — use max_risk for IC positions
        total_exposure = sum(
            p.get('max_risk', p.get('avg_price', 0) * p.get('quantity', 0))
            if p.get('is_iron_condor', False)
            else p.get('avg_price', 0) * p.get('quantity', 0)
            for p in active_positions
        )
        total_exposure_pct = (total_exposure / self.current_capital) * 100
        
        if total_exposure_pct > self.limits.max_total_exposure_pct:
            return TradePermission(
                allowed=False,
                reason=f"Max total exposure reached: {total_exposure_pct:.1f}%",
                warnings=["Too much capital deployed"]
            )
        
        # 9. Check approaching limits (warnings)
        if self.state.trades_today == self.limits.max_trades_per_day - 1:
            warnings.append("Last trade allowed today")
        
        if self.state.consecutive_losses == self.limits.max_consecutive_losses - 1:
            warnings.append("One more loss will halt trading")
        
        if _net_pnl_pct < -(self.limits.max_daily_loss_pct * 0.7):
            warnings.append(f"Approaching daily loss limit: net {_net_pnl_pct:.1f}% (realized {self.state.daily_pnl_pct:.1f}%)")
        
        return TradePermission(
            allowed=True,
            reason="Trade allowed",
            warnings=warnings,
            suggested_size_multiplier=size_multiplier
        )
    
    def record_trade_result(
        self, 
        symbol: str, 
        pnl: float, 
        was_win: bool,
        unrealized_pnl: float = 0.0
    ):
        """
        Record result of a completed trade
        
        Args:
            symbol: Symbol traded
            pnl: Profit/loss amount (realized)
            was_win: True if profitable
            unrealized_pnl: Current unrealized P&L from open positions
        """
        self._check_new_day()
        
        # Cache unrealized if provided (from caller's live ticker data)
        if unrealized_pnl != 0.0:
            self._cached_unrealized_pnl = unrealized_pnl
        
        self.state.trades_today += 1
        self.state.daily_pnl += pnl
        self.state.daily_pnl_pct = (self.state.daily_pnl / self.starting_capital) * 100
        self.state.last_trade_time = datetime.now().isoformat()
        
        if was_win:
            self.state.wins_today += 1
            self.state.consecutive_losses = 0
            print(f"✅ Win recorded: ₹{pnl:+,.0f} | Day: {self.state.wins_today}W-{self.state.losses_today}L")
        else:
            self.state.losses_today += 1
            self.state.consecutive_losses += 1
            self.state.last_loss_time = datetime.now().isoformat()
            print(f"❌ Loss recorded: ₹{pnl:+,.0f} | Consecutive: {self.state.consecutive_losses}")
            
            # Enter cooldown
            if self.state.consecutive_losses < self.limits.max_consecutive_losses:
                self._enter_cooldown()
        
        self._save_state()
        
        # Check if we should halt using NET P&L (realized + unrealized)
        # This prevents halting when open winners offset a closed loser
        # Use cached unrealized if caller didn't provide it (live ticker data from is_trading_allowed)
        _effective_unrealized = unrealized_pnl if unrealized_pnl != 0.0 else self._cached_unrealized_pnl
        net_pnl = self.state.daily_pnl + _effective_unrealized
        net_pnl_pct = (net_pnl / self.starting_capital) * 100
        
        if net_pnl_pct <= -self.limits.max_daily_loss_pct:
            self._halt_trading(f"Max daily loss: {net_pnl_pct:.2f}% (realized: {self.state.daily_pnl_pct:.2f}% + unrealized: ₹{_effective_unrealized:+,.0f})")
        elif self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct and _effective_unrealized > 0:
            print(f"⚠️ Realized P&L {self.state.daily_pnl_pct:.2f}% exceeds limit, but open winners offset (net: {net_pnl_pct:.2f}%) — continuing")
        
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt_trading(f"Max consecutive losses: {self.state.consecutive_losses}")
    
    def _enter_cooldown(self):
        """Enter cooldown period after a loss"""
        cooldown_end = datetime.now() + timedelta(minutes=self.limits.cooldown_minutes)
        self.state.system_state = SystemState.COOLDOWN.value
        self.state.cooldown_until = cooldown_end.isoformat()
        print(f"⏸️ Cooldown: No trading until {cooldown_end.strftime('%H:%M')}")
        self._save_state()
    
    def _halt_trading(self, reason: str):
        """Halt trading for the day"""
        self.state.system_state = SystemState.HALT_TRADING.value
        self.state.halt_reason = reason
        print(f"\n🛑 TRADING HALTED: {reason}")
        print(f"   Day P&L: ₹{self.state.daily_pnl:+,.0f} ({self.state.daily_pnl_pct:+.2f}%)")
        print(f"   Trades: {self.state.trades_today} | W/L: {self.state.wins_today}/{self.state.losses_today}")
        self._save_state()
        
        # === DhanHQ Kill Switch — server-side backup halt ===
        self._activate_dhan_kill_switch(reason)
    
    def trigger_circuit_breaker(self, reason: str):
        """
        Trigger circuit breaker for emergency situations
        
        Call this when:
        - Broker rejects orders repeatedly
        - Data feed is stale
        - Position mismatch detected
        """
        self.state.system_state = SystemState.CIRCUIT_BREAK.value
        self.state.halt_reason = f"CIRCUIT BREAK: {reason}"
        print(f"\n🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        print(f"   Trading suspended until manual reset")
        self._save_state()
        
        # === DhanHQ Kill Switch — server-side EMERGENCY halt ===
        self._activate_dhan_kill_switch(f"CIRCUIT_BREAK: {reason}")
    
    def _activate_dhan_kill_switch(self, reason: str):
        """Activate DhanHQ server-side kill switch as backup safety net.
        
        This is a SUPPLEMENTARY halt on DhanHQ's side. Even if Titan crashes
        after this, DhanHQ will prevent any new orders.
        """
        try:
            from dhan_risk_tools import get_dhan_risk_tools
            drt = get_dhan_risk_tools()
            if drt.ready:
                result = drt.emergency_halt()
                ks = result.get('kill_switch', {})
                if ks.get('success'):
                    logger.info(f"🛑 DhanHQ Kill Switch activated (backup): {reason}")
                else:
                    logger.warning(f"⚠️ DhanHQ Kill Switch failed: {ks.get('message', 'unknown')}")
            else:
                logger.debug("DhanHQ not configured — kill switch skipped")
        except Exception as e:
            # Never let Dhan failure break the risk governor
            logger.warning(f"DhanHQ kill switch error (non-fatal): {e}")
    
    def record_order_rejection(self):
        """Record an order rejection (for circuit breaker)"""
        self.state.order_rejections += 1
        self._save_state()
        
        if self.state.order_rejections >= 3:
            self.trigger_circuit_breaker("Multiple order rejections")
    
    def record_stale_data(self):
        """Record stale data detection (for circuit breaker)"""
        self.state.data_stale_count += 1
        self._save_state()
        
        if self.state.data_stale_count >= 5:
            self.trigger_circuit_breaker("Data feed stale")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        if self.state.system_state == SystemState.CIRCUIT_BREAK.value:
            self.state.system_state = SystemState.ACTIVE.value
            self.state.halt_reason = ""
            self.state.order_rejections = 0
            self.state.data_stale_count = 0
            print("✅ Circuit breaker reset - trading resumed")
            self._save_state()
            
            # === Deactivate DhanHQ Kill Switch ===
            try:
                from dhan_risk_tools import get_dhan_risk_tools
                drt = get_dhan_risk_tools()
                if drt.ready:
                    ok, msg = drt.deactivate_kill_switch()
                    if ok:
                        logger.info("✅ DhanHQ Kill Switch deactivated")
                    else:
                        logger.warning(f"⚠️ DhanHQ Kill Switch deactivation failed: {msg}")
            except Exception as e:
                logger.warning(f"DhanHQ kill switch deactivation error: {e}")
    
    def force_halt(self, reason: str = "Manual halt"):
        """Manually halt trading"""
        self._halt_trading(reason)
    
    def get_status(self, active_positions: List[Dict] = None, unrealized_pnl: float = 0.0) -> str:
        """Get current risk governor status with live unrealized P&L
        
        Args:
            active_positions: Open positions (legacy fallback)
            unrealized_pnl: Pre-computed unrealized P&L from live ticker quotes (preferred)
        """
        self._check_new_day()
        
        state_emoji = {
            "ACTIVE": "🟢",
            "COOLDOWN": "🟡",
            "HALT_TRADING": "🔴",
            "CIRCUIT_BREAK": "🚨"
        }
        
        emoji = state_emoji.get(self.state.system_state, "❓")
        
        # Use pre-computed unrealized if provided, otherwise try from positions
        unrealized = unrealized_pnl if unrealized_pnl != 0.0 else (
            self._calc_unrealized_pnl(active_positions) if active_positions else 0.0
        )
        net_pnl = self.state.daily_pnl + unrealized
        net_pnl_pct = (net_pnl / self.starting_capital) * 100 if self.starting_capital else 0.0
        
        lines = [
            f"\n{emoji} RISK GOVERNOR STATUS:",
            f"   State: {self.state.system_state}",
            f"   Daily P&L: ₹{self.state.daily_pnl:+,.0f} ({self.state.daily_pnl_pct:+.2f}%)",
            f"   Trades: {self.state.trades_today}/{self.limits.max_trades_per_day}",
            f"   Win/Loss: {self.state.wins_today}/{self.state.losses_today}",
            f"   Consecutive Losses: {self.state.consecutive_losses}/{self.limits.max_consecutive_losses}",
        ]
        
        # Show live realized + unrealized breakdown
        if active_positions:
            lines.append(f"   Unrealized: ₹{unrealized:+,.0f} | Net P&L: ₹{net_pnl:+,.0f} ({net_pnl_pct:+.2f}%)")
        
        if self.state.halt_reason:
            lines.append(f"   Reason: {self.state.halt_reason}")
        
        if self.state.cooldown_until and self.state.system_state == SystemState.COOLDOWN.value:
            cooldown_end = datetime.fromisoformat(self.state.cooldown_until)
            if datetime.now() < cooldown_end:
                mins_left = (cooldown_end - datetime.now()).seconds // 60
                lines.append(f"   Cooldown: {mins_left} min remaining")
        
        return "\n".join(lines)
    
    def is_trading_allowed(self, active_positions: List[Dict] = None, unrealized_pnl: float = 0.0) -> bool:
        """Quick check if trading is allowed.
        
        If system is halted due to daily loss, re-evaluates using NET P&L
        (realized + unrealized). This prevents premature halts when open
        winners offset closed losers.
        
        Args:
            active_positions: Open positions (legacy fallback)
            unrealized_pnl: Pre-computed unrealized P&L from live ticker quotes
        """
        self._check_new_day()
        # Also clear expired cooldowns
        if self.state.system_state == SystemState.COOLDOWN.value:
            if self.state.cooldown_until:
                cooldown_end = datetime.fromisoformat(self.state.cooldown_until)
                if datetime.now() >= cooldown_end:
                    self.state.system_state = SystemState.ACTIVE.value
                    self.state.cooldown_until = ""
                    self._save_state()
                    print("✅ Cooldown expired - trading resumed")
        # Re-evaluate halt with live unrealized P&L (net P&L may clear it)
        if self.state.system_state == SystemState.HALT_TRADING.value:
            if unrealized_pnl != 0.0 or active_positions:
                self._reeval_halt(active_positions, unrealized_pnl=unrealized_pnl)
        # Cache the unrealized P&L so can_trade() uses live data instead of broken fallback
        if unrealized_pnl != 0.0:
            self._cached_unrealized_pnl = unrealized_pnl
        return self.state.system_state == SystemState.ACTIVE.value


# Singleton instance
_risk_governor: Optional[RiskGovernor] = None


def get_risk_governor(starting_capital: float = 100000) -> RiskGovernor:
    """Get singleton risk governor instance"""
    global _risk_governor
    if _risk_governor is None:
        _risk_governor = RiskGovernor(starting_capital)
    return _risk_governor


def reset_risk_governor():
    """Reset risk governor"""
    global _risk_governor
    _risk_governor = None


if __name__ == "__main__":
    # Test
    gov = RiskGovernor(100000)
    
    print(gov.get_status())
    
    # Test trade permission
    perm = gov.can_trade(
        symbol="NSE:INFY",
        position_value=10000,
        active_positions=[]
    )
    print(f"\nCan trade INFY: {perm}")
    
    # Simulate some losses
    gov.record_trade_result("NSE:INFY", -500, False)
    print(gov.get_status())
    
    time.sleep(1)  # Simulate cooldown check
    
    gov.record_trade_result("NSE:TCS", -300, False)
    print(gov.get_status())
