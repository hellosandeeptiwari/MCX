"""
TITAN — MCX Commodities Trading Module
══════════════════════════════════════════════════════════════
Standalone strategy engine for GOLD, SILVER, CRUDEOIL, NATURALGAS
on MCX via Kite Connect.

Strategies:
  1. OI-Breakout:   OI buildup + price breakout → directional futures entry
  2. Trend-Momentum: Multi-timeframe trend alignment (5m + 15m + 1hr)
  3. Mean-Reversion: Gold/Silver ratio extremes → pair trade
  4. Session-Break:  Asian→Europe / Europe→US session transitions

MCX Constraints:
  - NO IOC validity in algo segment (use DAY only)
  - Different trading hours per commodity
  - Futures-only (options on MCX have poor liq for most commodities)
  - High leverage → strict risk management
══════════════════════════════════════════════════════════════
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Trade ledger — shared event log for dashboard Trade History tab
try:
    from trade_ledger import TradeLedger
    _mcx_ledger = TradeLedger()
except Exception:
    _mcx_ledger = None

# ---------------------------------------------------------------------------
# MCX INSTRUMENT SPECS (lot sizes, tick sizes, margins, trading hours)
# ---------------------------------------------------------------------------
MCX_INSTRUMENTS = {
    "GOLD": {
        "lot_size": 100,           # 100 grams per lot
        "tick_size": 1.0,          # ₹1 per gram
        "margin_pct": 5.0,         # ~5% margin (NRML)
        "mis_multiplier": 2.0,     # MIS margin = NRML / 2
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "medium",
        "avg_daily_range_pct": 0.8,
        "sl_pct": 0.30,           # 0.30% SL — tighter (was 0.4)
        "target_pct": 0.60,       # 0.60% target → 2:1 RR
        "trailing_pct": 0.18,     # Trail at 0.18% from peak (was 0.25)
    },
    "GOLDM": {
        "lot_size": 10,            # 10 grams per lot (mini)
        "tick_size": 1.0,
        "margin_pct": 5.0,
        "mis_multiplier": 2.0,
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "medium",
        "avg_daily_range_pct": 0.8,
        "sl_pct": 0.30,
        "target_pct": 0.60,
        "trailing_pct": 0.18,
    },
    "SILVER": {
        "lot_size": 30,            # 30 kg per lot
        "tick_size": 1.0,          # ₹1 per kg
        "margin_pct": 6.0,
        "mis_multiplier": 2.0,
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "high",
        "avg_daily_range_pct": 1.2,
        "sl_pct": 0.35,            # Tighter SL (was 0.5)
        "target_pct": 0.70,        # 2:1 RR (was 0.8/0.5=1.6)
        "trailing_pct": 0.20,      # Tighter trail (was 0.3)
    },
    "SILVERM": {
        "lot_size": 5,             # 5 kg per lot (mini)
        "tick_size": 1.0,
        "margin_pct": 6.0,
        "mis_multiplier": 2.0,
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "high",
        "avg_daily_range_pct": 1.2,
        "sl_pct": 0.35,
        "target_pct": 0.70,
        "trailing_pct": 0.20,
    },
    "CRUDEOIL": {
        "lot_size": 100,           # 100 barrels per lot
        "tick_size": 1.0,          # ₹1 per barrel
        "margin_pct": 7.0,
        "mis_multiplier": 2.0,
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "very_high",
        "avg_daily_range_pct": 1.8,
        "sl_pct": 0.40,            # Tighter (was 0.6)
        "target_pct": 0.80,        # 2:1 RR (was 1.0/0.6=1.67)
        "trailing_pct": 0.22,      # Tighter trail (was 0.35)
    },
    "NATURALGAS": {
        "lot_size": 1250,          # 1250 mmBtu per lot
        "tick_size": 0.10,         # ₹0.10 per mmBtu
        "margin_pct": 10.0,
        "mis_multiplier": 2.0,
        "market_open": "09:00",
        "market_close": "23:30",
        "volatility_class": "extreme",
        "avg_daily_range_pct": 3.0,
        "sl_pct": 0.60,            # Tighter (was 1.0)
        "target_pct": 1.20,        # 2:1 RR (was 1.5/1.0=1.5)
        "trailing_pct": 0.30,      # Tighter trail (was 0.5)
    },
}

# Preferred trading instruments (use mini contracts for smaller capital)
MCX_PREFERRED = {
    "GOLD": "GOLD",           # Standard lot (switch to GOLDM for < ₹2L capital)
    "SILVER": "SILVERM",      # Mini lot (standard lot too large)
    "CRUDEOIL": "CRUDEOIL",
    "NATURALGAS": "NATURALGAS",
}

# ---------------------------------------------------------------------------
# MCX OPTION SPECS (premium-based risk management per commodity)
# ---------------------------------------------------------------------------
MCX_OPTION_SPECS = {
    "GOLD": {
        "otm_offset": 0,            # 0 = ATM, 1 = 1 strike OTM
        "premium_sl_pct": 35,       # Exit if premium drops 35%
        "premium_target_pct": 60,   # Take profit at 60% premium gain
        "premium_trail_pct": 25,    # Trail 25% from peak premium
        "trail_activation_pct": 40, # Start trailing after 40% gain
    },
    "GOLDM": {
        "otm_offset": 0,
        "premium_sl_pct": 35,
        "premium_target_pct": 60,
        "premium_trail_pct": 25,
        "trail_activation_pct": 40,
    },
    "SILVER": {
        "otm_offset": 0,
        "premium_sl_pct": 35,
        "premium_target_pct": 60,
        "premium_trail_pct": 25,
        "trail_activation_pct": 40,
    },
    "SILVERM": {
        "otm_offset": 0,
        "premium_sl_pct": 35,
        "premium_target_pct": 60,
        "premium_trail_pct": 25,
        "trail_activation_pct": 40,
    },
    "CRUDEOIL": {
        "otm_offset": 0,
        "premium_sl_pct": 35,
        "premium_target_pct": 60,
        "premium_trail_pct": 25,
        "trail_activation_pct": 40,
    },
    "NATURALGAS": {
        "otm_offset": 0,
        "premium_sl_pct": 40,       # Wider SL for extreme volatility
        "premium_target_pct": 80,   # Higher target for extreme vol
        "premium_trail_pct": 30,
        "trail_activation_pct": 50,
    },
}


# ---------------------------------------------------------------------------
# SIGNAL SCORING
# ---------------------------------------------------------------------------

class CommoditySignal:
    """Represents a scored trading signal for a commodity."""

    def __init__(self, commodity: str, direction: str, score: float,
                 strategy: str, meta: Dict[str, Any]):
        self.commodity = commodity
        self.direction = direction     # "BUY" or "SELL"
        self.score = score             # 0-100
        self.strategy = strategy       # e.g., "OI_BREAKOUT", "TREND_MOMENTUM"
        self.meta = meta               # Extra data (OI change, candle pattern, etc.)
        self.timestamp = datetime.now()

    def __repr__(self):
        return (f"CommoditySignal({self.commodity} {self.direction} "
                f"score={self.score:.1f} via {self.strategy})")


# ---------------------------------------------------------------------------
# COMMODITIES TRADER ENGINE
# ---------------------------------------------------------------------------

class CommoditiesTrader:
    """
    MCX Commodities Trading Engine — integrates with Titan's execution layer.
    
    Uses:
      - Kite Connect for order placement + historical data
      - DhanHQ for commodity OI data (richer than Kite for MCX)
      - Multi-strategy signal generation
      - Position tracking via shared state DB
    """

    def __init__(self, kite, config: Dict[str, Any], paper_mode: bool = True):
        self.kite = kite
        self.config = config
        self.paper_mode = paper_mode
        
        # Position tracking
        self._positions: List[Dict[str, Any]] = []
        self._positions_lock = threading.Lock()
        self._daily_pnl: float = 0.0
        self._trades_today: int = 0
        self._last_order_ts: float = 0.0   # For rate limiting
        
        # Candle cache: commodity -> deque of candles
        self._candle_cache: Dict[str, deque] = {}
        self._cache_lock = threading.Lock()
        
        # MCX instrument tokens (loaded on first scan)
        self._mcx_tokens: Dict[str, int] = {}       # symbol -> instrument_token
        self._mcx_instruments: List[Dict] = []       # Raw instrument list
        self._mcx_options: Dict[str, List[Dict]] = {}  # commodity -> option instruments
        self._instruments_loaded = False
        
        # DhanHQ OI fetcher
        self._dhan_fetcher = None
        try:
            from dhan_oi_fetcher import DhanOIFetcher
            self._dhan_fetcher = DhanOIFetcher()
        except Exception:
            pass
        
        # Signal history (for anti-repeat)
        self._signal_history: Dict[str, float] = {}  # "GOLD_BUY" -> last_signal_ts

        # State file for dashboard cross-process visibility
        self._state_file = os.path.join(os.path.dirname(__file__), 'mcx_state.json')
        
        print(f"📦 CommoditiesTrader initialized | mode={'PAPER' if paper_mode else 'LIVE'} "
              f"| commodities={list(config.get('commodities', []))}")

    # ===================================================================
    # INSTRUMENT LOADING
    # ===================================================================

    def _load_mcx_instruments(self):
        """Load MCX futures + options instruments from Kite."""
        if self._instruments_loaded or not self.kite:
            return
        try:
            mcx_all = self.kite.instruments("MCX")
            self._mcx_instruments = mcx_all
            
            today = date.today()
            near_month: Dict[str, Dict] = {}  # commodity -> nearest futures contract
            
            for inst in mcx_all:
                inst_type = inst.get("instrument_type", "")
                name = inst.get("name", "")
                expiry = inst.get("expiry")
                if not expiry or not name:
                    continue
                # expiry can be date or datetime
                exp_date = expiry if isinstance(expiry, date) else expiry.date()
                if exp_date < today:
                    continue  # Skip expired
                
                if inst_type == "FUT":
                    # Keep nearest unexpired contract per commodity
                    if name not in near_month or exp_date < near_month[name]["expiry_date"]:
                        near_month[name] = {
                            "token": inst["instrument_token"],
                            "tradingsymbol": inst["tradingsymbol"],
                            "expiry_date": exp_date,
                            "lot_size": inst.get("lot_size", 1),
                            "exchange": "MCX",
                            "name": name,
                        }
                elif inst_type in ("CE", "PE"):
                    # Collect option instruments for option chain
                    if name not in self._mcx_options:
                        self._mcx_options[name] = []
                    self._mcx_options[name].append({
                        "token": inst["instrument_token"],
                        "tradingsymbol": inst["tradingsymbol"],
                        "expiry": exp_date,
                        "strike": inst.get("strike", 0),
                        "option_type": inst_type,
                        "lot_size": inst.get("lot_size", 1),
                        "exchange": "MCX",
                        "name": name,
                    })
            
            for name, info in near_month.items():
                self._mcx_tokens[name] = info["token"]
            
            self._instruments_loaded = True
            loaded = [f"{k}={v['tradingsymbol']}" for k, v in near_month.items()
                      if k in self.config.get("commodities", [])]
            opt_count = sum(len(v) for k, v in self._mcx_options.items()
                           if k in [MCX_PREFERRED.get(c, c) for c in self.config.get("commodities", [])])
            print(f"📋 MCX instruments loaded: {len(near_month)} futures + {opt_count} options | "
                  f"Active: {', '.join(loaded)}")
        except Exception as e:
            print(f"⚠️ MCX instrument load failed: {e}")

    def _get_near_month(self, commodity: str) -> Optional[Dict]:
        """Get nearest futures contract info for a commodity."""
        self._load_mcx_instruments()
        # Check preferred mapping first
        pref = MCX_PREFERRED.get(commodity, commodity)
        for inst in self._mcx_instruments:
            if (inst.get("name") == pref and inst.get("instrument_type") == "FUT"
                    and inst.get("expiry")):
                exp = inst["expiry"]
                exp_date = exp if isinstance(exp, date) else exp.date()
                if exp_date >= date.today():
                    return {
                        "token": inst["instrument_token"],
                        "tradingsymbol": inst["tradingsymbol"],
                        "lot_size": inst.get("lot_size", 1),
                        "exchange": "MCX",
                        "name": pref,
                        "expiry": exp_date,
                    }
        # Fallback: use cached token
        token = self._mcx_tokens.get(pref)
        if token:
            return {"token": token, "tradingsymbol": pref, "lot_size": 1,
                    "exchange": "MCX", "name": pref}
        return None

    def _get_atm_option(self, commodity: str, direction: str,
                        underlying_price: float) -> Optional[Dict]:
        """Find ATM option for a commodity.
        
        direction: "BUY" → CE option, "SELL" → PE option.
        Picks nearest expiry with ≥ min_dte days, then closest strike to ATM.
        """
        self._load_mcx_instruments()
        pref = MCX_PREFERRED.get(commodity, commodity)
        options = self._mcx_options.get(pref, [])
        # Fallback: if preferred name has no options, try original name
        if not options and pref != commodity:
            options = self._mcx_options.get(commodity, [])

        if not options:
            return None

        opt_type = "CE" if direction == "BUY" else "PE"
        today = date.today()
        min_dte = self.config.get("mcx_options_min_dte", 3)
        min_expiry = today + timedelta(days=min_dte)

        # Filter by option type and minimum DTE
        filtered = [o for o in options
                    if o["option_type"] == opt_type
                    and o["expiry"] >= min_expiry
                    and o["strike"] > 0]
        if not filtered:
            return None

        # Get nearest expiry that satisfies min_dte
        nearest_expiry = min(o["expiry"] for o in filtered)
        expiry_options = [o for o in filtered if o["expiry"] == nearest_expiry]

        spec = MCX_OPTION_SPECS.get(pref, MCX_OPTION_SPECS.get(commodity, {}))
        otm_offset = spec.get("otm_offset", 0)

        # Sort by distance from underlying price to find ATM
        expiry_options.sort(key=lambda o: abs(o["strike"] - underlying_price))

        if not expiry_options:
            return None

        if otm_offset > 0:
            # For CE, OTM = higher strike; for PE, OTM = lower strike
            if opt_type == "CE":
                otm_candidates = sorted(
                    [o for o in expiry_options if o["strike"] >= underlying_price],
                    key=lambda o: o["strike"])
            else:
                otm_candidates = sorted(
                    [o for o in expiry_options if o["strike"] <= underlying_price],
                    key=lambda o: -o["strike"])
            idx = min(otm_offset, len(otm_candidates) - 1)
            selected = otm_candidates[idx] if otm_candidates else expiry_options[0]
        else:
            selected = expiry_options[0]  # ATM (closest to underlying)

        return selected

    # ===================================================================
    # DATA FETCHING
    # ===================================================================

    def _fetch_candles(self, commodity: str, interval: str = "5minute",
                       days: int = 5) -> List[Dict]:
        """Fetch historical candles from Kite for a commodity future."""
        contract = self._get_near_month(commodity)
        if not contract:
            return []
        try:
            from_dt = datetime.now() - timedelta(days=days)
            to_dt = datetime.now()
            candles = self.kite.historical_data(
                instrument_token=contract["token"],
                from_date=from_dt.strftime("%Y-%m-%d"),
                to_date=to_dt.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval,
            )
            return candles
        except Exception as e:
            print(f"⚠️ Candle fetch failed for {commodity} ({interval}): {e}")
            return []

    def _get_ltp(self, commodity: str) -> Optional[float]:
        """Get last traded price for a commodity future."""
        contract = self._get_near_month(commodity)
        if not contract:
            return None
        try:
            full_sym = f"MCX:{contract['tradingsymbol']}"
            quote = self.kite.ltp(full_sym)
            return quote.get(full_sym, {}).get("last_price")
        except Exception as e:
            print(f"⚠️ LTP fetch failed for {commodity}: {e}")
            return None

    def _get_commodity_oi(self, commodity: str) -> Optional[Dict]:
        """Get OI data from DhanHQ for commodity."""
        if not self._dhan_fetcher:
            return None
        try:
            from dhan_futures_oi import FuturesOIFetcher
            fut_fetcher = FuturesOIFetcher()
            if not fut_fetcher.ready:
                return None
            features = fut_fetcher.compute_daily_features(commodity)
            if features is None:
                return None
            if isinstance(features, dict):
                return features
            return dict(features.to_dict())  # type: ignore[union-attr]
        except Exception as e:
            print(f"⚠️ OI fetch failed for {commodity}: {e}")
            return None

    # ===================================================================
    # TECHNICAL INDICATORS
    # ===================================================================

    @staticmethod
    def _ema(values: List[float], period: int) -> List[float]:
        """Calculate EMA."""
        if not values or period <= 0:
            return []
        result = [values[0]]
        k = 2.0 / (period + 1)
        for v in values[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return result

    @staticmethod
    def _rsi(closes: List[float], period: int = 14) -> float:
        """Calculate RSI from close prices."""
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(candles: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return sum(trs[-period:]) / period

    @staticmethod
    def _vwap(candles: List[Dict]) -> float:
        """Calculate VWAP from candles."""
        total_vol = 0
        total_vp = 0
        for c in candles:
            vol = c.get("volume", 0)
            tp = (c["high"] + c["low"] + c["close"]) / 3
            total_vp += tp * vol
            total_vol += vol
        return total_vp / total_vol if total_vol > 0 else 0

    @staticmethod
    def _supertrend(candles: List[Dict], period: int = 10, multiplier: float = 3.0) -> str:
        """Calculate SuperTrend direction. Returns 'BUY' or 'SELL'."""
        if len(candles) < period + 1:
            return "NEUTRAL"
        atr_values = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            atr_values.append(tr)

        upper_band = []
        lower_band = []
        supertrend = []
        direction = 1  # 1=up, -1=down

        for i in range(len(candles)):
            if i < period:
                upper_band.append(0)
                lower_band.append(0)
                supertrend.append(candles[i]["close"])
                continue
            atr = sum(atr_values[max(0, i - period):i]) / period
            mid = (candles[i]["high"] + candles[i]["low"]) / 2
            basic_upper = mid + multiplier * atr
            basic_lower = mid - multiplier * atr

            if len(upper_band) > 0 and upper_band[-1] != 0:
                final_upper = min(basic_upper, upper_band[-1]) if candles[i - 1]["close"] <= upper_band[-1] else basic_upper
                final_lower = max(basic_lower, lower_band[-1]) if candles[i - 1]["close"] >= lower_band[-1] else basic_lower
            else:
                final_upper = basic_upper
                final_lower = basic_lower

            upper_band.append(final_upper)
            lower_band.append(final_lower)

            if direction == 1:
                if candles[i]["close"] < final_lower:
                    direction = -1
                    supertrend.append(final_upper)
                else:
                    supertrend.append(final_lower)
            else:
                if candles[i]["close"] > final_upper:
                    direction = 1
                    supertrend.append(final_lower)
                else:
                    supertrend.append(final_upper)

        return "BUY" if direction == 1 else "SELL"

    # ===================================================================
    # STRATEGY 1: OI-BREAKOUT
    # ===================================================================

    def _strategy_oi_breakout(self, commodity: str) -> Optional[CommoditySignal]:
        """
        OI Breakout: OI rises + price breaks out of range → strong directional move.
        
        Logic:
          - Fetch 5-min candles (2 days)
          - Compute today's range (high/low)
          - Check if price breaks above 2-day high or below 2-day low
          - Cross-check with OI buildup from DhanHQ
          - Score: breakout strength + OI confirmation + volume
        """
        candles_5m = self._fetch_candles(commodity, "5minute", days=3)
        if len(candles_5m) < 30:
            return None
        
        today_str = str(date.today())
        today_candles = [c for c in candles_5m
                         if str(c["date"])[:10] == today_str]
        if len(today_candles) < 6:
            return None  # Need at least 30 min of data
        
        prev_candles = [c for c in candles_5m
                        if str(c["date"])[:10] != today_str]
        if not prev_candles:
            return None
        
        # Previous day(s) range
        prev_high = max(c["high"] for c in prev_candles[-60:])  # Last ~5 hours
        prev_low = min(c["low"] for c in prev_candles[-60:])
        
        current_price = today_candles[-1]["close"]
        today_high = max(c["high"] for c in today_candles)
        today_low = min(c["low"] for c in today_candles)
        today_vol = sum(c.get("volume", 0) for c in today_candles)
        prev_vol = sum(c.get("volume", 0) for c in prev_candles[-len(today_candles):]) or 1
        vol_ratio = today_vol / prev_vol
        
        # Breakout detection
        breakout_up = current_price > prev_high and today_high > prev_high
        breakout_down = current_price < prev_low and today_low < prev_low
        
        if not breakout_up and not breakout_down:
            return None
        
        direction = "BUY" if breakout_up else "SELL"
        
        # Breakout strength
        if breakout_up:
            break_pct = (current_price - prev_high) / prev_high * 100
        else:
            break_pct = (prev_low - current_price) / prev_low * 100
        
        # OI confirmation
        oi_data = self._get_commodity_oi(commodity)
        oi_change_pct = 0
        oi_buildup = "NEUTRAL"
        if oi_data and isinstance(oi_data, dict):
            oi_change_pct = oi_data.get("oi_change_pct", 0)
            if oi_change_pct > 5 and breakout_up:
                oi_buildup = "LONG_BUILDUP"
            elif oi_change_pct > 5 and breakout_down:
                oi_buildup = "SHORT_BUILDUP"
            elif oi_change_pct < -5:
                oi_buildup = "UNWINDING"  # Counter-signal
        
        # Scoring
        score = 40  # Base for breakout
        score += min(break_pct * 15, 20)           # Breakout strength (max +20)
        score += min(vol_ratio * 5, 15)             # Volume confirmation (max +15)
        if oi_buildup in ("LONG_BUILDUP", "SHORT_BUILDUP"):
            score += 15                              # OI confirms direction
        elif oi_buildup == "UNWINDING":
            score -= 10                              # OI contradicts
        
        # RSI filter — avoid overbought/oversold entries
        closes = [c["close"] for c in candles_5m[-20:]]
        rsi = self._rsi(closes)
        if direction == "BUY" and rsi > 80:
            score -= 10
        elif direction == "SELL" and rsi < 20:
            score -= 10
        elif direction == "BUY" and 55 <= rsi <= 70:
            score += 5   # Healthy momentum
        elif direction == "SELL" and 30 <= rsi <= 45:
            score += 5
        
        score = max(0, min(100, score))
        
        if score < self.config.get("min_entry_score", 55):
            return None
        
        return CommoditySignal(
            commodity=commodity,
            direction=direction,
            score=score,
            strategy="OI_BREAKOUT",
            meta={
                "break_pct": round(break_pct, 3),
                "vol_ratio": round(vol_ratio, 2),
                "oi_change_pct": round(oi_change_pct, 2),
                "oi_buildup": oi_buildup,
                "rsi": round(rsi, 1),
                "prev_high": prev_high,
                "prev_low": prev_low,
                "current_price": current_price,
            },
        )

    # ===================================================================
    # STRATEGY 2: TREND-MOMENTUM (Multi-Timeframe)
    # ===================================================================

    def _strategy_trend_momentum(self, commodity: str) -> Optional[CommoditySignal]:
        """
        Multi-timeframe trend alignment: 5m + 15m + 1hr must agree.
        
        Uses EMA crossover + SuperTrend + RSI confirmation.
        High-conviction trend following — the bread & butter of commodities.
        """
        candles_5m = self._fetch_candles(commodity, "5minute", days=5)
        candles_15m = self._fetch_candles(commodity, "15minute", days=10)
        candles_1h = self._fetch_candles(commodity, "60minute", days=20)
        
        if len(candles_5m) < 50 or len(candles_15m) < 30 or len(candles_1h) < 20:
            return None
        
        # === 5-MIN SIGNALS ===
        closes_5m = [c["close"] for c in candles_5m]
        ema9_5m = self._ema(closes_5m, 9)
        ema21_5m = self._ema(closes_5m, 21)
        st_5m = self._supertrend(candles_5m, period=10, multiplier=2.0)
        rsi_5m = self._rsi(closes_5m)
        
        # 5m direction: EMA9 > EMA21 + SuperTrend BUY
        if ema9_5m and ema21_5m:
            ema_dir_5m = "BUY" if ema9_5m[-1] > ema21_5m[-1] else "SELL"
        else:
            return None
        
        # === 15-MIN SIGNALS ===
        closes_15m = [c["close"] for c in candles_15m]
        ema9_15m = self._ema(closes_15m, 9)
        ema21_15m = self._ema(closes_15m, 21)
        st_15m = self._supertrend(candles_15m, period=10, multiplier=3.0)
        rsi_15m = self._rsi(closes_15m)
        
        if ema9_15m and ema21_15m:
            ema_dir_15m = "BUY" if ema9_15m[-1] > ema21_15m[-1] else "SELL"
        else:
            return None
        
        # === 1-HOUR SIGNALS ===
        closes_1h = [c["close"] for c in candles_1h]
        ema9_1h = self._ema(closes_1h, 9)
        ema21_1h = self._ema(closes_1h, 21)
        st_1h = self._supertrend(candles_1h, period=10, multiplier=3.0)
        
        if ema9_1h and ema21_1h:
            ema_dir_1h = "BUY" if ema9_1h[-1] > ema21_1h[-1] else "SELL"
        else:
            return None
        
        # === ALIGNMENT CHECK ===
        directions = [ema_dir_5m, ema_dir_15m, ema_dir_1h, st_5m, st_15m, st_1h]
        buy_votes = sum(1 for d in directions if d == "BUY")
        sell_votes = sum(1 for d in directions if d == "SELL")
        
        if buy_votes >= 6:
            direction = "BUY"
            alignment = buy_votes
        elif sell_votes >= 6:
            direction = "SELL"
            alignment = sell_votes
        else:
            return None  # Need ALL 6 indicators aligned (was 5/6)
        
        # === SCORING ===
        score = 35  # Base
        score += alignment * 5                    # Alignment bonus (max 30)
        
        # RSI confirmation
        if direction == "BUY" and 50 < rsi_5m < 75 and 50 < rsi_15m < 75:
            score += 10  # Healthy bullish RSI
        elif direction == "SELL" and 25 < rsi_5m < 50 and 25 < rsi_15m < 50:
            score += 10
        
        # VWAP confirmation
        vwap_today = self._vwap([c for c in candles_5m
                                  if str(c["date"])[:10] == str(date.today())])
        current_price = closes_5m[-1]
        if vwap_today > 0:
            if direction == "BUY" and current_price > vwap_today:
                score += 5
            elif direction == "SELL" and current_price < vwap_today:
                score += 5
        
        # ATR-based volatility check
        atr = self._atr(candles_5m)
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0
        spec = MCX_INSTRUMENTS.get(MCX_PREFERRED.get(commodity, commodity), {})
        expected_range = spec.get("avg_daily_range_pct", 1.0)
        if atr_pct > expected_range * 0.3:
            score += 5  # Good intraday volatility
        
        # OI confirmation
        oi_data = self._get_commodity_oi(commodity)
        if oi_data and isinstance(oi_data, dict):
            oi_change = oi_data.get("oi_change_pct", 0)
            if oi_change > 3:
                score += 5   # OI building
        
        score = max(0, min(100, score))
        
        if score < self.config.get("min_entry_score", 55):
            return None
        
        return CommoditySignal(
            commodity=commodity,
            direction=direction,
            score=score,
            strategy="TREND_MOMENTUM",
            meta={
                "alignment": alignment,
                "ema_5m": ema_dir_5m, "ema_15m": ema_dir_15m, "ema_1h": ema_dir_1h,
                "st_5m": st_5m, "st_15m": st_15m, "st_1h": st_1h,
                "rsi_5m": round(rsi_5m, 1), "rsi_15m": round(rsi_15m, 1),
                "atr_pct": round(atr_pct, 3),
                "vwap": round(vwap_today, 2),
                "current_price": current_price,
            },
        )

    # ===================================================================
    # STRATEGY 3: MEAN-REVERSION (Gold/Silver Ratio)
    # ===================================================================

    def _strategy_gold_silver_ratio(self) -> Optional[CommoditySignal]:
        """
        Gold/Silver ratio mean-reversion.
        
        Historical ratio ~75-85. When stretched to extremes:
          - Ratio > 90: Silver undervalued → BUY SILVER (or SELL GOLD)
          - Ratio < 70: Silver overvalued → SELL SILVER (or BUY GOLD)
        
        Uses ratio of prices per gram for comparability.
        """
        if "GOLD" not in self.config.get("commodities", []):
            return None
        if "SILVER" not in self.config.get("commodities", []):
            return None
        
        gold_price = self._get_ltp("GOLD")    # ₹/gram
        silver_price = self._get_ltp("SILVER")  # ₹/kg → convert to ₹/gram
        
        if not gold_price or not silver_price or silver_price <= 0:
            return None
        
        # Silver is quoted per kg, gold per 10g (standard) or per gram
        # Ratio = gold_price_per_gram / silver_price_per_gram
        silver_per_gram = silver_price / 1000  # Convert ₹/kg to ₹/g
        ratio = gold_price / silver_per_gram if silver_per_gram > 0 else 0
        
        if ratio <= 0:
            return None
        
        # Historical mean ~80, std ~5
        mean_ratio = self.config.get("gs_ratio_mean", 80)
        std_ratio = self.config.get("gs_ratio_std", 5)
        
        z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        
        if abs(z_score) < 2.0:
            return None  # Need extreme deviation (was 1.5 — too loose)
        
        if z_score > 2.0:
            # Silver undervalued → BUY SILVER
            direction = "BUY"
            commodity = "SILVER"
            score = 55 + min(abs(z_score) * 8, 25)
        else:
            # Silver overvalued → SELL SILVER
            direction = "SELL"
            commodity = "SILVER"
            score = 55 + min(abs(z_score) * 8, 25)
        
        return CommoditySignal(
            commodity=commodity,
            direction=direction,
            score=score,
            strategy="GS_RATIO_REVERSION",
            meta={
                "gold_price": gold_price,
                "silver_price": silver_price,
                "ratio": round(ratio, 2),
                "z_score": round(z_score, 2),
                "mean": mean_ratio,
            },
        )

    # ===================================================================
    # STRATEGY 4: SESSION-BREAK (Asia→Europe, Europe→US transitions)
    # ===================================================================

    def _strategy_session_break(self, commodity: str) -> Optional[CommoditySignal]:
        """
        Session transition trading.
        
        Commodities are global — price reacts to session opens:
          - 14:00-14:30 IST: Europe open → GOLD/SILVER volatility spike
          - 19:00-19:30 IST: US open → CRUDE/NATGAS volatility spike
        
        Logic:
          - In 30-min window around session open, look for directional thrust
          - If first 15min candle of new session shows >0.3% move with volume,
            enter in that direction (institutional flow)
        """
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Define session windows per commodity
        windows = {
            "GOLD":       [(14, 0, 14, 30, "EUROPE"), (19, 0, 19, 30, "US")],
            "SILVER":     [(14, 0, 14, 30, "EUROPE"), (19, 0, 19, 30, "US")],
            "CRUDEOIL":   [(19, 0, 19, 30, "US")],
            "NATURALGAS": [(19, 0, 19, 30, "US")],
        }
        
        comm_windows = windows.get(commodity, [])
        active_session = None
        for h1, m1, h2, m2, session_name in comm_windows:
            if (h1, m1) <= (hour, minute) <= (h2, m2):
                active_session = session_name
                break
        
        if not active_session:
            return None
        
        # Get the last 6 × 5-min candles (30 min window)
        candles = self._fetch_candles(commodity, "5minute", days=1)
        if len(candles) < 6:
            return None
        
        recent = candles[-6:]  # Last 30 min
        first_close = recent[0]["close"]
        last_close = recent[-1]["close"]
        session_move_pct = (last_close - first_close) / first_close * 100
        session_volume = sum(c.get("volume", 0) for c in recent)
        
        # Need meaningful directional move (0.5% — was 0.3% which is noise)
        if abs(session_move_pct) < 0.5:
            return None
        
        direction = "BUY" if session_move_pct > 0 else "SELL"
        
        # Score based on move magnitude + volume
        score = 50
        score += min(abs(session_move_pct) * 12, 20)
        
        # Volume compared to earlier candles
        earlier = candles[-18:-6] if len(candles) >= 18 else candles[:-6]
        if earlier:
            earlier_vol = sum(c.get("volume", 0) for c in earlier) or 1
            vol_surge = session_volume / (earlier_vol / max(len(earlier) / 6, 1))
            if vol_surge > 1.5:
                score += 10
        
        score = max(0, min(100, score))
        
        if score < self.config.get("min_entry_score", 55):
            return None
        
        return CommoditySignal(
            commodity=commodity,
            direction=direction,
            score=score,
            strategy=f"SESSION_{active_session}",
            meta={
                "session": active_session,
                "move_pct": round(session_move_pct, 3),
                "volume": session_volume,
                "current_price": last_close,
            },
        )

    # ===================================================================
    # SIGNAL GENERATION (runs all strategies)
    # ===================================================================

    def generate_signals(self) -> List[CommoditySignal]:
        """Run all strategies across all configured commodities. Returns scored signals."""
        commodities = self.config.get("commodities", [])
        signals: List[CommoditySignal] = []
        
        for comm in commodities:
            if not self._is_market_open(comm):
                continue
            if not self._in_liquidity_window(comm):
                continue  # Outside high-liquidity session — skip
            
            # Anti-repeat cooldown
            cooldown = self.config.get("signal_cooldown_seconds", 300)
            
            # Strategy 1: OI Breakout
            try:
                sig = self._strategy_oi_breakout(comm)
                if sig and self._check_cooldown(sig, cooldown):
                    signals.append(sig)
            except Exception as e:
                print(f"⚠️ OI_BREAKOUT error for {comm}: {e}")
            
            # Strategy 2: Trend Momentum
            try:
                sig = self._strategy_trend_momentum(comm)
                if sig and self._check_cooldown(sig, cooldown):
                    signals.append(sig)
            except Exception as e:
                print(f"⚠️ TREND_MOMENTUM error for {comm}: {e}")
            
            # Strategy 4: Session Break
            try:
                sig = self._strategy_session_break(comm)
                if sig and self._check_cooldown(sig, cooldown):
                    signals.append(sig)
            except Exception as e:
                print(f"⚠️ SESSION_BREAK error for {comm}: {e}")
        
        # Strategy 3: Gold/Silver Ratio — now handled directly in scan_and_trade
        # with factor confirmation, so skipped here to avoid double-fire
        
        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    # ===================================================================
    # ORDER EXECUTION
    # ===================================================================

    def _place_commodity_order(self, signal: CommoditySignal) -> Optional[Dict]:
        """
        Place an MCX option order: buy CE for bullish, buy PE for bearish.
        
        Premium-based risk: max loss = premium paid.
        NO IOC validity (MCX algo restriction) — uses DAY.
        """
        commodity = signal.commodity

        # Get underlying futures price for strike selection & indicator context
        underlying_ltp = self._get_ltp(commodity)
        if not underlying_ltp or underlying_ltp <= 0:
            print(f"❌ Cannot get underlying LTP for {commodity}")
            return None

        # Find ATM option (CE for BUY signal, PE for SELL signal)
        option = self._get_atm_option(commodity, signal.direction, underlying_ltp)
        if not option:
            print(f"❌ No suitable option found for {commodity} {signal.direction}")
            return None

        opt_symbol = option["tradingsymbol"]
        opt_type = option["option_type"]
        strike = option["strike"]
        lot_size = option.get("lot_size", 1)

        # Get option premium
        try:
            full_sym = f"MCX:{opt_symbol}"
            quote = self.kite.ltp(full_sym)
            premium = quote.get(full_sym, {}).get("last_price")
        except Exception as e:
            print(f"❌ Cannot get premium for {opt_symbol}: {e}")
            return None

        if not premium or premium <= 0:
            print(f"❌ Invalid premium for {opt_symbol}: ₹{premium}")
            return None

        # === POSITION SIZING (risk = premium × lot_size × lots) ===
        pref = MCX_PREFERRED.get(commodity, commodity)
        opt_spec = MCX_OPTION_SPECS.get(pref, MCX_OPTION_SPECS.get(commodity, {}))
        max_risk_per_trade = self.config.get("max_risk_per_trade", 15000)
        premium_per_lot = premium * lot_size

        if premium_per_lot <= 0:
            return None

        num_lots = max(1, int(max_risk_per_trade / premium_per_lot))
        max_lots = self.config.get("max_lots_per_trade", 3)
        num_lots = min(num_lots, max_lots)
        quantity = num_lots * lot_size

        # === SL / TARGET (premium %) ===
        sl_pct = opt_spec.get("premium_sl_pct", 35) / 100
        target_pct = opt_spec.get("premium_target_pct", 60) / 100
        sl_premium = round(premium * (1 - sl_pct), 2)
        target_premium = round(premium * (1 + target_pct), 2)

        # === CHECK DAILY LIMITS ===
        max_trades = self.config.get("max_trades_per_day", 6)
        if self._trades_today >= max_trades:
            print(f"⚠️ Daily trade limit reached ({max_trades}) — skipping {commodity}")
            return None

        max_daily_loss = self.config.get("max_daily_loss", 30000)
        if self._daily_pnl <= -max_daily_loss:
            print(f"🚫 Daily loss limit hit (₹{self._daily_pnl:,.0f}) — no more trades")
            return None

        product = self.config.get("mcx_option_product", "NRML")

        # === EXECUTE ===
        if self.paper_mode:
            import random
            order_id = f"MCX_OPT_PAPER_{random.randint(100000, 999999)}"
            print(f"   📝 PAPER MCX OPT: BUY {num_lots}L × {lot_size} {opt_symbol} "
                  f"@ ₹{premium:.2f} (strike={strike} {opt_type}) "
                  f"| SL=₹{sl_premium:.2f} TGT=₹{target_premium:.2f} "
                  f"| underlying=₹{underlying_ltp:.2f} | {signal.strategy} "
                  f"score={signal.score:.0f}")
        else:
            try:
                import time as _time
                elapsed = _time.time() - self._last_order_ts
                if elapsed < 0.35:
                    _time.sleep(0.35 - elapsed)

                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange="MCX",
                    tradingsymbol=opt_symbol,
                    transaction_type="BUY",          # Always BUY (CE for bullish, PE for bearish)
                    quantity=quantity,
                    product=product,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_DAY,  # NO IOC on MCX algo!
                    market_protection=5,              # Kite April 2026 mandate
                    tag="TITAN_MCX_OPT",
                )
                self._last_order_ts = _time.time()

                print(f"   ✅ LIVE MCX OPT: BUY {num_lots}L × {lot_size} {opt_symbol} "
                      f"@ ₹{premium:.2f} (strike={strike} {opt_type}) "
                      f"| order={order_id} | {signal.strategy} score={signal.score:.0f}")

                # Place SL-M on premium
                try:
                    elapsed = _time.time() - self._last_order_ts
                    if elapsed < 0.35:
                        _time.sleep(0.35 - elapsed)

                    sl_order_id = self.kite.place_order(
                        variety=self.kite.VARIETY_REGULAR,
                        exchange="MCX",
                        tradingsymbol=opt_symbol,
                        transaction_type="SELL",
                        quantity=quantity,
                        product=product,
                        order_type=self.kite.ORDER_TYPE_SLM,
                        trigger_price=round(sl_premium, 2),
                        validity=self.kite.VALIDITY_DAY,
                        market_protection=5,
                        tag="TITAN_MCX_OPT_SL",
                    )
                    self._last_order_ts = _time.time()
                    print(f"   🛡️ OPT SL placed: trigger ₹{sl_premium:.2f} (order: {sl_order_id})")
                except Exception as sl_err:
                    print(f"   ⚠️ OPT SL order failed: {sl_err}")
                    sl_order_id = None

            except Exception as e:
                print(f"   ❌ MCX option order FAILED for {opt_symbol}: {e}")
                return None

        # === TRACK POSITION ===
        position = {
            "order_id": order_id,
            "commodity": commodity,
            "tradingsymbol": opt_symbol,
            "direction": signal.direction,          # BUY=bullish, SELL=bearish
            "option_type": opt_type,                # CE or PE
            "strike": strike,
            "trade_type": "option",
            "lots": num_lots,
            "lot_size": lot_size,
            "quantity": quantity,
            "entry_price": premium,                 # Option premium at entry
            "entry_underlying": underlying_ltp,
            "sl_price": sl_premium,
            "target_price": target_premium,
            "peak_premium": premium,                # For trailing
            "trailing_pct": opt_spec.get("premium_trail_pct", 25) / 100,
            "trail_activation_pct": opt_spec.get("trail_activation_pct", 40) / 100,
            "product": product,
            "strategy": signal.strategy,
            "score": signal.score,
            "meta": signal.meta,
            "entry_time": datetime.now().isoformat(),
            "status": "OPEN",
        }

        with self._positions_lock:
            self._positions.append(position)
            self._trades_today += 1
        self._save_state()

        # Log to trade ledger for Trade History tab
        if _mcx_ledger:
            try:
                _mcx_ledger.log_entry(
                    symbol=opt_symbol,
                    underlying=f"MCX:{commodity}",
                    direction=signal.direction,
                    source=signal.strategy or "MCX",
                    smart_score=signal.score,
                    final_score=signal.score,
                    option_symbol=opt_symbol,
                    strike=strike,
                    option_type=opt_type,
                    entry_price=premium,
                    quantity=quantity,
                    lots=num_lots,
                    stop_loss=sl_premium,
                    target=target_premium,
                    total_premium=round(premium * quantity, 2),
                    order_id=str(order_id) if order_id else "",
                    strategy_type="MCX_OPTION",
                    sector="MCX",
                    oi_signal=signal.meta.get("oi_signal", "") if signal.meta else "",
                    extra={"commodity": commodity, "trade_type": "option",
                           "exchange": "MCX"},
                )
            except Exception as e:
                print(f"   ⚠️ MCX ledger entry log: {e}")

        return position

    # ===================================================================
    # POSITION MONITORING & EXIT
    # ===================================================================

    def monitor_positions(self):
        """Check all open option positions for premium-based SL/target/trailing exits.
        
        Premium-Based Management:
          - SL: Exit if premium drops below sl_price (e.g. 35% loss)
          - Target: Exit if premium reaches target_price (e.g. 60% gain)
          - Trailing: Once premium rises 40%+ from entry, trail 25% from peak
          - Time exit: Close positions 10 min before market close
        """
        with self._positions_lock:
            open_positions = [p for p in self._positions if p["status"] == "OPEN"]

        for pos in open_positions:
            try:
                # Get current option premium
                ts = pos["tradingsymbol"]
                try:
                    full_sym = f"MCX:{ts}"
                    quote = self.kite.ltp(full_sym)
                    current_premium = quote.get(full_sym, {}).get("last_price")
                except Exception:
                    current_premium = None

                if not current_premium:
                    continue

                pos["current_price"] = current_premium
                entry = pos["entry_price"]
                sl = pos["sl_price"]
                target = pos["target_price"]

                # Update peak premium
                pos["peak_premium"] = max(pos.get("peak_premium", entry), current_premium)
                peak = pos["peak_premium"]

                # Premium gain %
                premium_gain_pct = (current_premium - entry) / entry if entry > 0 else 0

                # === TRAILING STOP (premium-based) ===
                trail_activation = pos.get("trail_activation_pct", 0.40)
                trail_pct = pos.get("trailing_pct", 0.25)

                if premium_gain_pct >= trail_activation and peak > entry:
                    trail_sl = round(peak * (1 - trail_pct), 2)
                    if trail_sl > sl:
                        pos["sl_price"] = trail_sl
                        sl = trail_sl

                # === CHECK SL / TARGET ===
                exit_reason = None
                if current_premium <= sl:
                    exit_reason = "PREMIUM_SL" if premium_gain_pct < 0 else "TRAILING_SL"
                elif current_premium >= target:
                    exit_reason = "PREMIUM_TARGET"

                # Time-based exit: close 10 min before market close
                commodity = pos.get("commodity", "")
                pref_key = MCX_PREFERRED.get(commodity, commodity) or commodity
                spec = MCX_INSTRUMENTS.get(pref_key,
                                           MCX_INSTRUMENTS.get(commodity, {}))
                close_time_str = spec.get("market_close", "23:30")
                ch, cm = map(int, close_time_str.split(":"))
                close_time = datetime.now().replace(hour=ch, minute=cm, second=0)
                if datetime.now() >= close_time - timedelta(minutes=10):
                    exit_reason = "MARKET_CLOSE"

                if exit_reason:
                    self._exit_position(pos, current_premium, exit_reason)

            except Exception as e:
                print(f"⚠️ Monitor error for {pos.get('commodity')}: {e}")
        self._save_state()

    def _exit_position(self, position: Dict, exit_price: float, reason: str):
        """Exit a commodity option position by selling the option."""
        ts = position["tradingsymbol"]
        quantity = position["quantity"]
        entry_premium = position["entry_price"]

        # PnL = (exit_premium - entry_premium) × quantity
        pnl = (exit_price - entry_premium) * quantity

        if self.paper_mode:
            gain_pct = (exit_price - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
            print(f"   📝 MCX OPT EXIT (PAPER): SELL {ts} × {quantity} "
                  f"@ ₹{exit_price:.2f} ({gain_pct:+.1f}%) "
                  f"| reason={reason} | P&L=₹{pnl:+,.0f}")
        else:
            try:
                import time as _time
                elapsed = _time.time() - self._last_order_ts
                if elapsed < 0.35:
                    _time.sleep(0.35 - elapsed)

                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange="MCX",
                    tradingsymbol=ts,
                    transaction_type="SELL",      # Close long option
                    quantity=quantity,
                    product=position.get("product", "NRML"),
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    validity=self.kite.VALIDITY_DAY,
                    market_protection=5,
                    tag="TITAN_MCX_OPT_EXIT",
                )
                self._last_order_ts = _time.time()
                print(f"   ✅ MCX OPT EXIT: SELL {ts} × {quantity} @ ₹{exit_price:.2f} "
                      f"| order={order_id} | reason={reason} | P&L=₹{pnl:+,.0f}")

                # Cancel pending SL order for this symbol
                try:
                    orders = self.kite.orders()
                    for o in orders:
                        if (o.get("tradingsymbol") == ts
                                and o.get("tag") == "TITAN_MCX_OPT_SL"
                                and o.get("status") in ("TRIGGER PENDING", "OPEN")):
                            self.kite.cancel_order(
                                variety=self.kite.VARIETY_REGULAR,
                                order_id=o["order_id"]
                            )
                except Exception:
                    pass

            except Exception as e:
                print(f"   🚨 MCX OPT EXIT FAILED: SELL {ts}: {e} — MANUAL ACTION NEEDED!")
                return

        # Update position status
        with self._positions_lock:
            position["status"] = "CLOSED"
            position["exit_price"] = exit_price
            position["exit_time"] = datetime.now().isoformat()
            position["exit_reason"] = reason
            position["realized_pnl"] = pnl
            self._daily_pnl += pnl

        gain_pct = (exit_price - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        pnl_emoji = "💰" if pnl > 0 else "💸"
        print(f"   {pnl_emoji} MCX {position.get('commodity','')} "
              f"{position.get('option_type','')}{position.get('strike','')} — "
              f"₹{entry_premium:.2f} → ₹{exit_price:.2f} ({gain_pct:+.1f}%) "
              f"| P&L=₹{pnl:+,.0f} | {reason}")

        # Log exit to trade ledger for Trade History tab
        if _mcx_ledger:
            try:
                entry_ts = position.get("entry_time", "")
                hold_mins = 0
                if entry_ts:
                    try:
                        entry_dt = datetime.fromisoformat(entry_ts)
                        hold_mins = int((datetime.now() - entry_dt).total_seconds() / 60)
                    except Exception:
                        pass
                _mcx_ledger.log_exit(
                    symbol=ts,
                    underlying=f"MCX:{position.get('commodity', '')}",
                    direction=position.get("direction", "BUY"),
                    source=position.get("strategy", "MCX"),
                    exit_type=reason,
                    entry_price=entry_premium,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=round(pnl, 2),
                    pnl_pct=round(gain_pct, 2),
                    smart_score=position.get("score", 0),
                    final_score=position.get("score", 0),
                    strategy_type="MCX_OPTION",
                    hold_minutes=hold_mins,
                    exit_reason=reason,
                    order_id=str(position.get("order_id", "")),
                    entry_time=entry_ts,
                    extra={"commodity": position.get("commodity", ""),
                           "option_type": position.get("option_type", ""),
                           "strike": position.get("strike", 0),
                           "exchange": "MCX", "trade_type": "option"},
                )
            except Exception as e:
                print(f"   ⚠️ MCX ledger exit log: {e}")

        self._save_state()

    # ===================================================================
    # HELPER METHODS
    # ===================================================================

    def _is_market_open(self, commodity: str) -> bool:
        """Check if MCX market is open for this commodity."""
        spec = MCX_INSTRUMENTS.get(MCX_PREFERRED.get(commodity, commodity),
                                    MCX_INSTRUMENTS.get(commodity, {}))
        if not spec:
            return False
        
        now = datetime.now()
        open_str = spec.get("market_open", "09:00")
        close_str = spec.get("market_close", "23:30")
        
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        
        market_open = now.replace(hour=oh, minute=om, second=0)
        market_close = now.replace(hour=ch, minute=cm, second=0)
        
        # Allow 5-min buffer after open (let prices settle)
        effective_open = market_open + timedelta(minutes=5)
        # Stop new entries 15 min before close
        effective_close = market_close - timedelta(minutes=15)
        
        return effective_open <= now <= effective_close

    def _in_liquidity_window(self, commodity: str) -> bool:
        """Check if we're in the trading window for this commodity.
        MCX has strong domestic liquidity from market open (9:00 IST)."""
        windows = self.config.get("liquidity_windows", {})
        window = windows.get(commodity)
        if not window:
            return True  # No window defined → allow
        now = datetime.now()
        sh, sm = map(int, window["start"].split(":"))
        eh, em = map(int, window["end"].split(":"))
        start = now.replace(hour=sh, minute=sm, second=0)
        end = now.replace(hour=eh, minute=em, second=0)
        return start <= now <= end

    def _get_1h_trend(self, commodity: str) -> Optional[str]:
        """Get 1-hour trend direction using EMA + SuperTrend.
        Returns 'BUY', 'SELL', or None if insufficient data."""
        candles_1h = self._fetch_candles(commodity, "60minute", days=15)
        if len(candles_1h) < 20:
            return None
        closes = [c["close"] for c in candles_1h]
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        st = self._supertrend(candles_1h, period=10, multiplier=3.0)
        if not ema9 or not ema21:
            return None
        ema_dir = "BUY" if ema9[-1] > ema21[-1] else "SELL"
        # Both EMA and SuperTrend must agree for strong signal
        if ema_dir == st:
            return ema_dir
        return None  # Conflicting → no clear trend

    # ===================================================================
    # MULTI-FACTOR SCORING ENGINE (OI_watcher style)
    # ===================================================================

    def _compute_factor_score(self, commodity: str) -> Dict[str, Any]:
        """
        Compute 12 independent market microstructure factors for a commodity.
        Returns {direction, factor_count, factors: {name: {vote, detail}}, strength}.

        Factors:
          1.  OI Buildup direction (Dhan: LONG_BUILDUP / SHORT_BUILDUP)
          2.  OI Buildup strength (>0.35 = counted)
          3.  5-min EMA crossover (EMA9 vs EMA21)
          4.  15-min EMA crossover
          5.  1-hour EMA crossover
          6.  5-min SuperTrend direction
          7.  15-min SuperTrend direction
          8.  1-hour SuperTrend direction
          9.  RSI zone (50-70 for BUY, 30-50 for SELL)
          10. VWAP position (price > VWAP = BUY bias)
          11. Volume surge (today vol > 1.3× yesterday)
          12. Previous day breakout (price outside prev range)
        """
        factors: Dict[str, Dict[str, Any]] = {}
        buy_votes = 0
        sell_votes = 0

        # --- Fetch data once ---
        candles_5m = self._fetch_candles(commodity, "5minute", days=3)
        candles_15m = self._fetch_candles(commodity, "15minute", days=5)
        candles_1h = self._fetch_candles(commodity, "60minute", days=15)

        if len(candles_5m) < 30:
            return {"direction": None, "factor_count": 0, "factors": {}, "strength": 0}

        closes_5m = [c["close"] for c in candles_5m]
        current_price = closes_5m[-1]

        # === FACTOR 1 & 2: OI BUILDUP ===
        oi_data = self._get_commodity_oi(commodity)
        oi_signal = "NEUTRAL"
        oi_strength = 0.0
        if oi_data and isinstance(oi_data, dict):
            oi_change_pct = oi_data.get("oi_change_pct", 0)
            # Classify OI buildup
            if oi_change_pct > 3:
                # OI increasing — need price direction to classify
                today_str = str(date.today())
                today_5m = [c for c in candles_5m if str(c["date"])[:10] == today_str]
                if today_5m and len(today_5m) >= 3:
                    price_chg = (today_5m[-1]["close"] - today_5m[0]["open"]) / today_5m[0]["open"] * 100
                    if price_chg > 0.05:
                        oi_signal = "LONG_BUILDUP"
                        oi_strength = min(abs(oi_change_pct) / 10, 1.0)
                    elif price_chg < -0.05:
                        oi_signal = "SHORT_BUILDUP"
                        oi_strength = min(abs(oi_change_pct) / 10, 1.0)
            elif oi_change_pct < -3:
                today_str = str(date.today())
                today_5m = [c for c in candles_5m if str(c["date"])[:10] == today_str]
                if today_5m and len(today_5m) >= 3:
                    price_chg = (today_5m[-1]["close"] - today_5m[0]["open"]) / today_5m[0]["open"] * 100
                    if price_chg > 0.05:
                        oi_signal = "SHORT_COVERING"
                    else:
                        oi_signal = "LONG_UNWINDING"

        min_oi_str = self.config.get("min_oi_strength", 0.35)
        if oi_signal == "LONG_BUILDUP":
            factors["oi_direction"] = {"vote": "BUY", "detail": f"{oi_signal} ({oi_strength:.2f})"}
            buy_votes += 1
            if oi_strength >= min_oi_str:
                factors["oi_strength"] = {"vote": "BUY", "detail": f"strength={oi_strength:.2f}≥{min_oi_str}"}
                buy_votes += 1
        elif oi_signal == "SHORT_BUILDUP":
            factors["oi_direction"] = {"vote": "SELL", "detail": f"{oi_signal} ({oi_strength:.2f})"}
            sell_votes += 1
            if oi_strength >= min_oi_str:
                factors["oi_strength"] = {"vote": "SELL", "detail": f"strength={oi_strength:.2f}≥{min_oi_str}"}
                sell_votes += 1

        # === FACTOR 3: 5-min EMA crossover ===
        ema9_5m = self._ema(closes_5m, 9)
        ema21_5m = self._ema(closes_5m, 21)
        if ema9_5m and ema21_5m:
            d = "BUY" if ema9_5m[-1] > ema21_5m[-1] else "SELL"
            factors["ema_5m"] = {"vote": d, "detail": f"EMA9={ema9_5m[-1]:.1f} vs EMA21={ema21_5m[-1]:.1f}"}
            if d == "BUY": buy_votes += 1
            else: sell_votes += 1

        # === FACTOR 4: 15-min EMA crossover ===
        if len(candles_15m) >= 25:
            closes_15m = [c["close"] for c in candles_15m]
            ema9_15m = self._ema(closes_15m, 9)
            ema21_15m = self._ema(closes_15m, 21)
            if ema9_15m and ema21_15m:
                d = "BUY" if ema9_15m[-1] > ema21_15m[-1] else "SELL"
                factors["ema_15m"] = {"vote": d, "detail": f"EMA9={ema9_15m[-1]:.1f} vs EMA21={ema21_15m[-1]:.1f}"}
                if d == "BUY": buy_votes += 1
                else: sell_votes += 1

        # === FACTOR 5: 1-hour EMA crossover ===
        if len(candles_1h) >= 22:
            closes_1h = [c["close"] for c in candles_1h]
            ema9_1h = self._ema(closes_1h, 9)
            ema21_1h = self._ema(closes_1h, 21)
            if ema9_1h and ema21_1h:
                d = "BUY" if ema9_1h[-1] > ema21_1h[-1] else "SELL"
                factors["ema_1h"] = {"vote": d, "detail": f"EMA9={ema9_1h[-1]:.1f} vs EMA21={ema21_1h[-1]:.1f}"}
                if d == "BUY": buy_votes += 1
                else: sell_votes += 1

        # === FACTOR 6: 5-min SuperTrend ===
        st_5m = self._supertrend(candles_5m, period=10, multiplier=2.0)
        if st_5m != "NEUTRAL":
            factors["st_5m"] = {"vote": st_5m, "detail": f"SuperTrend(10,2)={st_5m}"}
            if st_5m == "BUY": buy_votes += 1
            else: sell_votes += 1

        # === FACTOR 7: 15-min SuperTrend ===
        if len(candles_15m) >= 15:
            st_15m = self._supertrend(candles_15m, period=10, multiplier=3.0)
            if st_15m != "NEUTRAL":
                factors["st_15m"] = {"vote": st_15m, "detail": f"SuperTrend(10,3)={st_15m}"}
                if st_15m == "BUY": buy_votes += 1
                else: sell_votes += 1

        # === FACTOR 8: 1-hour SuperTrend ===
        if len(candles_1h) >= 15:
            st_1h = self._supertrend(candles_1h, period=10, multiplier=3.0)
            if st_1h != "NEUTRAL":
                factors["st_1h"] = {"vote": st_1h, "detail": f"SuperTrend(10,3)={st_1h}"}
                if st_1h == "BUY": buy_votes += 1
                else: sell_votes += 1

        # === FACTOR 9: RSI zone ===
        rsi = self._rsi(closes_5m)
        if 50 < rsi < 70:
            factors["rsi"] = {"vote": "BUY", "detail": f"RSI={rsi:.1f} (bullish zone)"}
            buy_votes += 1
        elif 30 < rsi < 50:
            factors["rsi"] = {"vote": "SELL", "detail": f"RSI={rsi:.1f} (bearish zone)"}
            sell_votes += 1
        # Otherwise RSI is neutral — no vote

        # === FACTOR 10: VWAP position ===
        today_str = str(date.today())
        today_candles = [c for c in candles_5m if str(c["date"])[:10] == today_str]
        vwap = self._vwap(today_candles) if today_candles else 0
        if vwap > 0:
            if current_price > vwap * 1.001:  # >0.1% above VWAP
                factors["vwap"] = {"vote": "BUY", "detail": f"price={current_price:.1f} > VWAP={vwap:.1f}"}
                buy_votes += 1
            elif current_price < vwap * 0.999:
                factors["vwap"] = {"vote": "SELL", "detail": f"price={current_price:.1f} < VWAP={vwap:.1f}"}
                sell_votes += 1

        # === FACTOR 11: Volume surge ===
        if today_candles:
            today_vol = sum(c.get("volume", 0) for c in today_candles)
            prev_candles = [c for c in candles_5m if str(c["date"])[:10] != today_str]
            prev_same_len = prev_candles[-len(today_candles):] if prev_candles else []
            prev_vol = sum(c.get("volume", 0) for c in prev_same_len) or 1
            vol_ratio = today_vol / prev_vol
            if vol_ratio > 1.3:
                # Volume surge — vote with price direction
                if today_candles[-1]["close"] > today_candles[0]["open"]:
                    factors["volume"] = {"vote": "BUY", "detail": f"vol_ratio={vol_ratio:.2f}× (surge + rising)"}
                    buy_votes += 1
                else:
                    factors["volume"] = {"vote": "SELL", "detail": f"vol_ratio={vol_ratio:.2f}× (surge + falling)"}
                    sell_votes += 1

        # === FACTOR 12: Previous day range breakout ===
        prev_candles = [c for c in candles_5m if str(c["date"])[:10] != today_str]
        if prev_candles:
            prev_high = max(c["high"] for c in prev_candles[-60:])
            prev_low = min(c["low"] for c in prev_candles[-60:])
            if current_price > prev_high:
                factors["prev_breakout"] = {"vote": "BUY", "detail": f"price {current_price:.1f} > prev_high {prev_high:.1f}"}
                buy_votes += 1
            elif current_price < prev_low:
                factors["prev_breakout"] = {"vote": "SELL", "detail": f"price {current_price:.1f} < prev_low {prev_low:.1f}"}
                sell_votes += 1

        # === DETERMINE DIRECTION ===
        if buy_votes > sell_votes:
            direction = "BUY"
            factor_count = buy_votes
        elif sell_votes > buy_votes:
            direction = "SELL"
            factor_count = sell_votes
        else:
            direction = None
            factor_count = 0

        strength = factor_count / 12.0

        return {
            "direction": direction,
            "factor_count": factor_count,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "factors": factors,
            "strength": round(strength, 3),
            "oi_signal": oi_signal,
            "oi_strength": oi_strength,
            "current_price": current_price,
        }

    def _check_cooldown(self, signal: CommoditySignal, cooldown_secs: int) -> bool:
        """Check if enough time has passed since last signal for same commodity+direction."""
        key = f"{signal.commodity}_{signal.direction}"
        last_ts = self._signal_history.get(key, 0)
        if time.time() - last_ts < cooldown_secs:
            return False
        self._signal_history[key] = time.time()
        return True

    def _save_state(self):
        """Persist positions & summary to JSON for dashboard cross-process access."""
        try:
            with self._positions_lock:
                positions_copy = list(self._positions)
            closed = [p for p in positions_copy if p.get('status') == 'CLOSED']
            open_pos = [p for p in positions_copy if p.get('status') == 'OPEN']
            wins = [p for p in closed if p.get('realized_pnl', 0) > 0]
            losses = [p for p in closed if p.get('realized_pnl', 0) < 0]
            total_pnl = sum(p.get('realized_pnl', 0) for p in closed)
            state = {
                'positions': positions_copy,
                'summary': {
                    'trades_today': self._trades_today,
                    'open_positions': len(open_pos),
                    'closed_positions': len(closed),
                    'wins': len(wins),
                    'losses': len(losses),
                    'win_rate': len(wins) / len(closed) * 100 if closed else 0,
                    'total_pnl': total_pnl,
                    'daily_pnl': self._daily_pnl,
                },
                'updated': datetime.now().isoformat(),
            }
            _tmp = self._state_file + '.tmp'
            with open(_tmp, 'w') as f:
                json.dump(state, f, default=str)
            os.replace(_tmp, self._state_file)
        except Exception as e:
            print(f"⚠️ MCX state save error: {e}")

    def get_open_positions(self) -> List[Dict]:
        """Return list of open commodity positions."""
        with self._positions_lock:
            return [p for p in self._positions if p["status"] == "OPEN"]

    def get_daily_summary(self) -> Dict:
        """Return daily trading summary."""
        with self._positions_lock:
            closed = [p for p in self._positions if p["status"] == "CLOSED"]
            open_pos = [p for p in self._positions if p["status"] == "OPEN"]
        
        wins = [p for p in closed if p.get("realized_pnl", 0) > 0]
        losses = [p for p in closed if p.get("realized_pnl", 0) < 0]
        total_pnl = sum(p.get("realized_pnl", 0) for p in closed)
        
        return {
            "trades_today": self._trades_today,
            "open_positions": len(open_pos),
            "closed_positions": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "total_pnl": total_pnl,
            "daily_pnl": self._daily_pnl,
        }

    # ===================================================================
    # MAIN SCAN LOOP (called from autonomous_trader)
    # ===================================================================

    def scan_and_trade(self) -> List[Dict]:
        """
        Main entry point — called by autonomous_trader on each cycle.
        
        OI_watcher-style multi-factor scoring:
          1. Monitor open positions (breakeven escalator)
          2. For each commodity in liquidity window:
             - Compute 12 independent factors (OI, EMA×3, SuperTrend×3, RSI, VWAP, volume, breakout)
             - Need 5+ factors aligned in same direction to fire
             - 7+ factors = strong signal, bypasses cooldown
          3. If factor score passes AND old strategies agree → highest conviction
        
        Returns list of new positions opened (if any).
        """
        new_positions = []
        
        try:
            # Monitor existing positions (includes breakeven escalator)
            self.monitor_positions()
            
            # Check limits
            max_positions = self.config.get("max_concurrent_positions", 3)
            open_count = len(self.get_open_positions())
            if open_count >= max_positions:
                return new_positions
            
            max_trades = self.config.get("max_trades_per_day", 6)
            if self._trades_today >= max_trades:
                return new_positions
            
            max_daily_loss = self.config.get("max_daily_loss", 30000)
            if self._daily_pnl <= -max_daily_loss:
                return new_positions
            
            min_factors = self.config.get("min_factor_confirmations", 5)
            strong_factors = self.config.get("strong_signal_factors", 7)
            cooldown = self.config.get("signal_cooldown_seconds", 300)
            
            commodities = self.config.get("commodities", [])
            candidates: List[Dict[str, Any]] = []
            
            for comm in commodities:
                if not self._is_market_open(comm):
                    continue
                if not self._in_liquidity_window(comm):
                    continue
                
                # Skip if already have open position in same commodity
                with self._positions_lock:
                    has_open = any(p["commodity"] == comm and p["status"] == "OPEN"
                                  for p in self._positions)
                if has_open:
                    continue
                
                # === MULTI-FACTOR SCORING ===
                score = self._compute_factor_score(comm)
                direction = score["direction"]
                factor_count = score["factor_count"]
                
                if not direction or factor_count < min_factors:
                    if factor_count >= 3:
                        print(f"   📊 MCX {comm}: {factor_count}/12 factors → {direction or 'NEUTRAL'} "
                              f"(need {min_factors}+, waiting...)")
                    continue
                
                # Cooldown check (strong signals bypass)
                key = f"{comm}_{direction}"
                last_ts = self._signal_history.get(key, 0)
                is_strong = factor_count >= strong_factors
                if not is_strong and (time.time() - last_ts < cooldown):
                    continue
                
                # === Cross-check with old strategies for score bonus ===
                strategy_score = 50 + factor_count * 4  # Base: 50 + 4 per factor
                strategy_name = "MULTI_FACTOR"
                strategy_hits = []
                
                # Run old strategies to see if they agree
                try:
                    sig_oi = self._strategy_oi_breakout(comm)
                    if sig_oi and sig_oi.direction == direction:
                        strategy_score += 8
                        strategy_hits.append("OI_BREAKOUT")
                except Exception:
                    pass
                
                try:
                    sig_trend = self._strategy_trend_momentum(comm)
                    if sig_trend and sig_trend.direction == direction:
                        strategy_score += 8
                        strategy_hits.append("TREND_MOMENTUM")
                except Exception:
                    pass
                
                try:
                    sig_session = self._strategy_session_break(comm)
                    if sig_session and sig_session.direction == direction:
                        strategy_score += 5
                        strategy_hits.append(sig_session.strategy)
                except Exception:
                    pass
                
                # OI buildup bonus
                if score["oi_signal"] in ("LONG_BUILDUP", "SHORT_BUILDUP"):
                    strategy_score += int(score["oi_strength"] * 10)
                
                strategy_score = min(100, strategy_score)
                
                if strategy_hits:
                    strategy_name = f"MULTI_FACTOR+{'+'.join(strategy_hits)}"
                
                factor_summary = ", ".join(f"{k}={v['vote']}" for k, v in score["factors"].items())
                
                candidates.append({
                    "commodity": comm,
                    "direction": direction,
                    "score": strategy_score,
                    "factor_count": factor_count,
                    "strategy_name": strategy_name,
                    "is_strong": is_strong,
                    "factor_summary": factor_summary,
                    "oi_signal": score["oi_signal"],
                    "oi_strength": score["oi_strength"],
                    "current_price": score["current_price"],
                })
            
            # Also check GS ratio (within market hours + liquidity)
            try:
                if (self._is_market_open("SILVER") and self._is_market_open("GOLD")
                        and self._in_liquidity_window("SILVER")):
                    sig = self._strategy_gold_silver_ratio()
                    if sig:
                        # GS ratio also needs factor confirmation on SILVER
                        silver_score = self._compute_factor_score("SILVER")
                        if silver_score["direction"] == sig.direction and silver_score["factor_count"] >= 3:
                            candidates.append({
                                "commodity": sig.commodity,
                                "direction": sig.direction,
                                "score": sig.score + silver_score["factor_count"] * 2,
                                "factor_count": silver_score["factor_count"],
                                "strategy_name": f"GS_RATIO+FACTORS({silver_score['factor_count']})",
                                "is_strong": silver_score["factor_count"] >= strong_factors,
                                "factor_summary": f"z={sig.meta.get('z_score', 0):.1f}",
                                "oi_signal": silver_score["oi_signal"],
                                "oi_strength": silver_score["oi_strength"],
                                "current_price": silver_score["current_price"],
                            })
            except Exception as e:
                print(f"⚠️ GS_RATIO error: {e}")
            
            # Sort by score descending, place trades
            candidates.sort(key=lambda c: c["score"], reverse=True)
            
            for cand in candidates:
                if open_count >= max_positions:
                    break
                
                comm = cand["commodity"]
                direction = cand["direction"]
                
                # Update cooldown
                self._signal_history[f"{comm}_{direction}"] = time.time()
                
                signal = CommoditySignal(
                    commodity=comm,
                    direction=direction,
                    score=cand["score"],
                    strategy=cand["strategy_name"],
                    meta={
                        "factor_count": cand["factor_count"],
                        "factor_summary": cand["factor_summary"],
                        "oi_signal": cand["oi_signal"],
                        "oi_strength": cand["oi_strength"],
                        "is_strong": cand["is_strong"],
                    },
                )
                
                strength_tag = "🔥 STRONG" if cand["is_strong"] else "📊"
                print(f"\n{strength_tag} MCX Signal: {comm} {direction} | "
                      f"{cand['factor_count']}/12 factors | score={cand['score']} | "
                      f"via {cand['strategy_name']}")
                print(f"   Factors: {cand['factor_summary']}")
                
                pos = self._place_commodity_order(signal)
                if pos:
                    new_positions.append(pos)
                    open_count += 1
            
        except Exception as e:
            print(f"⚠️ CommoditiesTrader scan error: {e}")
        
        return new_positions


# ===================================================================
# FACTORY FUNCTION
# ===================================================================

_instance = None

def get_commodities_trader(kite=None, config=None, paper_mode=True):
    """Get or create the singleton CommoditiesTrader instance."""
    global _instance
    if _instance is None and kite is not None:
        _instance = CommoditiesTrader(kite=kite, config=config or {}, paper_mode=paper_mode)
    return _instance
