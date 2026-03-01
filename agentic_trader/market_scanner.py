"""
MARKET SCANNER — Dynamic Discovery Layer
==========================================
Scans ALL F&O-eligible stocks on NSE using a single quote API call (~200 symbols)
to identify exceptional movers that may not be in the fixed 18-stock universe.

Categories scanned:
  • Top Gainers / Losers (by % change)
  • Volume Surges (volume vs 20-day avg proxy)
  • Breakout Candidates (price near day-high with volume)
  • Momentum Stocks (large absolute moves with sustained direction)

Surfaced stocks are injected as "wild-card" Tier-2 candidates into the scan cycle,
subject to the same Tier-2 gating (clear trend OR ORB breakout + volume).
"""

import os
import time
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from kiteconnect import KiteConnect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCANNER_CONFIG = {
    # How many wild-card stocks to surface per scan cycle
    "max_wildcards": 10,

    # Minimum % change (absolute) to consider a stock as a mover
    "min_change_pct": 1.5,

    # Minimum volume multiplier vs session average to flag volume surge
    # (we approximate with buy_qty+sell_qty / avg intraday volume proxy)
    "min_volume_ratio": 2.0,

    # Breakout: price within this % of day-high and change > 0
    "breakout_proximity_pct": 0.3,

    # Lot-size floor: skip illiquid contracts with very large lots (> ₹ premium)
    "max_lot_value_lakh": 15,  # Skip if lot_size * ltp > 15 lakh

    # Stocks to always EXCLUDE even if they show up as movers
    # (penny stocks, illiquid, circuit-prone)
    "exclude_symbols": [
        "IDEA", "SUZLON", "YESBANK", "PNB", "IRFC", "NHPC", "SAIL",
        "BANKBARODA", "UNIONBANK", "IDFCFIRSTB", "CANBK", "IOB",
        "INDIANB", "CENTRALBK", "UCOBANK", "BANKINDIA",
        "RECLTD", "PFC", "NBCC", "IRCTC", "ZOMATO", "PAYTM",
        "POLICYBZR", "DELHIVERY", "NYKAA",
    ],

    # Cache TTL for the F&O instrument list (seconds) — fetch once per day
    "instrument_cache_ttl": 86400,

    # Metal sector: reserve up to 2 wild-card slots on high-momentum metal days
    "metal_symbols": [
        "TATASTEEL", "JSWSTEEL", "JINDALSTEL", "HINDALCO",
        "HINDZINC", "VEDL", "NMDC", "NATIONALUM",
        "COALINDIA", "APLAPOLLO", "RATNAMANI", "WELCORP",
    ],
    "metal_max_reserve": 2,          # Max metal wild-card slots
    "metal_momentum_threshold": 1.5, # Avg abs % change across metals to trigger boost
    "metal_score_boost": 20,         # Extra score for metals on hot metal days
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScannerResult:
    """A single stock surfaced by the scanner."""
    symbol: str           # e.g. "TATAPOWER"
    nse_symbol: str       # e.g. "NSE:TATAPOWER"
    ltp: float
    change_pct: float
    volume: int
    day_high: float
    day_low: float
    prev_close: float
    oi: float             # open interest (0 for equity quotes)
    buy_qty: int
    sell_qty: int
    category: str         # GAINER | LOSER | VOLUME_SURGE | BREAKOUT
    score: float          # composite ranking score (higher = better)
    lot_size: int


@dataclass
class ScanSummary:
    """Summary of a scanner run."""
    timestamp: str
    total_fo_stocks: int
    scanned: int
    top_gainers: List[ScannerResult]
    top_losers: List[ScannerResult]
    volume_surges: List[ScannerResult]
    breakouts: List[ScannerResult]
    wildcards: List[ScannerResult]   # Final picks injected into the bot


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class MarketScanner:
    """
    Lightweight scanner that uses kite.quote() to scan all F&O stocks
    in 1 API call and surface exceptional movers.
    """

    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.config = SCANNER_CONFIG
        self._fo_stocks: List[Dict] = []       # [{name, lot_size, token}, ...]
        self._fo_cache_time: Optional[datetime] = None
        self._last_scan: Optional[ScanSummary] = None
        self._all_results: List[ScannerResult] = []  # ALL scanned stocks (for heat map)
        self._exclude = set(self.config["exclude_symbols"])

    # ------------------------------------------------------------------
    # F&O instrument discovery (cached per day)
    # ------------------------------------------------------------------

    def _refresh_fo_list(self):
        """Fetch NFO instruments and extract unique F&O stock names + lot sizes."""
        now = datetime.now()
        if (self._fo_cache_time and
                (now - self._fo_cache_time).total_seconds() < self.config["instrument_cache_ttl"]):
            return  # Cache still valid

        try:
            nfo = self.kite.instruments(exchange="NFO")
        except Exception as e:
            print(f"⚠️ Scanner: Failed to fetch NFO instruments: {e}")
            return

        # Extract unique stock names from FUT instruments (most reliable)
        stock_map = {}
        for inst in nfo:
            if inst.get("instrument_type") == "FUT" and inst.get("segment") == "NFO-FUT":
                name = inst["name"]
                if name not in stock_map:
                    stock_map[name] = {
                        "name": name,
                        "lot_size": inst.get("lot_size", 1),
                        "token": inst.get("instrument_token", 0),
                    }

        # Filter out excludes
        self._fo_stocks = [
            v for v in stock_map.values()
            if v["name"] not in self._exclude
        ]
        self._fo_cache_time = now
        # print(f"📡 Scanner: Loaded {len(self._fo_stocks)} F&O stocks from NFO instruments")

    def get_lot_map(self) -> Dict[str, int]:
        """Return {symbol: lot_size} for all F&O stocks discovered from Kite API."""
        self._refresh_fo_list()
        return {s["name"]: s["lot_size"] for s in self._fo_stocks}

    def get_all_fo_symbols(self) -> List[str]:
        """Return all F&O stock symbols as NSE:NAME for scan_universe expansion."""
        self._refresh_fo_list()
        return [f"NSE:{s['name']}" for s in self._fo_stocks]

    # ------------------------------------------------------------------
    # Core scan
    # ------------------------------------------------------------------

    def scan(self, existing_universe: List[str]) -> ScanSummary:
        """
        Run a full F&O market scan.

        Args:
            existing_universe: Current APPROVED_UNIVERSE (e.g. ["NSE:SBIN", ...])
                               Stocks already in universe are excluded from wild-cards
                               but still ranked (for logging).

        Returns:
            ScanSummary with categorised movers and final wild-card picks.
        """
        self._refresh_fo_list()

        if not self._fo_stocks:
            return ScanSummary(
                timestamp=str(datetime.now()),
                total_fo_stocks=0, scanned=0,
                top_gainers=[], top_losers=[],
                volume_surges=[], breakouts=[],
                wildcards=[],
            )

        # Build quote keys — all F&O stocks on NSE
        symbols = [f"NSE:{s['name']}" for s in self._fo_stocks]
        lot_map = {s["name"]: s["lot_size"] for s in self._fo_stocks}

        # Batch quote (max 500 per call — we have ~200, fits in 1 call)
        all_results: List[ScannerResult] = []
        try:
            quotes = self.kite.quote(symbols)
        except Exception as e:
            print(f"⚠️ Scanner: Quote API failed: {e}")
            return ScanSummary(
                timestamp=str(datetime.now()),
                total_fo_stocks=len(self._fo_stocks), scanned=0,
                top_gainers=[], top_losers=[],
                volume_surges=[], breakouts=[],
                wildcards=[],
            )

        existing_set = set(existing_universe)  # e.g. {"NSE:SBIN", ...}

        for sym_key, q in quotes.items():
            # sym_key = "NSE:RELIANCE"
            stock_name = sym_key.replace("NSE:", "")
            ohlc = q.get("ohlc", {})
            prev_close = ohlc.get("close", 0)
            ltp = q.get("last_price", 0)

            if prev_close <= 0 or ltp <= 0:
                continue

            change_pct = (ltp - prev_close) / prev_close * 100
            day_high = ohlc.get("high", ltp)
            day_low = ohlc.get("low", ltp)
            volume = q.get("volume", 0)
            oi = q.get("oi", 0)
            buy_qty = q.get("buy_quantity", 0)
            sell_qty = q.get("sell_quantity", 0)
            lot_size = lot_map.get(stock_name, 1)

            # Skip if lot value too large (illiquid/expensive)
            if lot_size * ltp > self.config["max_lot_value_lakh"] * 100000:
                continue

            # Determine category
            category = "NONE"
            score = 0.0

            abs_change = abs(change_pct)

            # --- GAINER ---
            if change_pct >= self.config["min_change_pct"]:
                category = "GAINER"
                score = change_pct * 10  # Higher % = higher score

            # --- LOSER ---
            elif change_pct <= -self.config["min_change_pct"]:
                category = "LOSER"
                score = abs_change * 10

            # --- BREAKOUT (near day-high, bullish) ---
            if day_high > 0 and ltp > prev_close:
                proximity = (day_high - ltp) / day_high * 100
                if proximity <= self.config["breakout_proximity_pct"] and change_pct > 0.5:
                    if category == "NONE":
                        category = "BREAKOUT"
                    elif category == "GAINER":
                        category = "GAINER+BREAKOUT"
                    score += 15  # Bonus for being at day-high

            # --- VOLUME SURGE ---
            # Approximate: if buy_qty + sell_qty (pending book depth) is huge
            # or raw volume is in top percentile, flag it.
            # We'll rank by volume later in relative terms.
            if volume > 0:
                # We'll compute relative volume after collecting all
                pass

            if category == "NONE" and abs_change < 0.5:
                continue  # Skip flat stocks entirely

            all_results.append(ScannerResult(
                symbol=stock_name,
                nse_symbol=sym_key,
                ltp=ltp,
                change_pct=change_pct,
                volume=volume,
                day_high=day_high,
                day_low=day_low,
                prev_close=prev_close,
                oi=oi,
                buy_qty=buy_qty,
                sell_qty=sell_qty,
                category=category,
                score=score,
                lot_size=lot_size,
            ))

        # --- Volume surge detection (relative) ---
        if all_results:
            volumes = [r.volume for r in all_results if r.volume > 0]
            if volumes:
                median_vol = sorted(volumes)[len(volumes) // 2]
                if median_vol > 0:
                    for r in all_results:
                        vol_ratio = r.volume / median_vol
                        if vol_ratio >= self.config["min_volume_ratio"]:
                            if "VOLUME" not in r.category:
                                if r.category == "NONE":
                                    r.category = "VOLUME_SURGE"
                                else:
                                    r.category += "+VOL"
                            r.score += vol_ratio * 5  # Bonus for high volume

        # --- Categorise ---
        gainers = sorted(
            [r for r in all_results if "GAINER" in r.category],
            key=lambda r: r.change_pct, reverse=True
        )[:10]

        losers = sorted(
            [r for r in all_results if "LOSER" in r.category],
            key=lambda r: r.change_pct  # Most negative first
        )[:10]

        vol_surges = sorted(
            [r for r in all_results if "VOL" in r.category],
            key=lambda r: r.volume, reverse=True
        )[:10]

        breakouts = sorted(
            [r for r in all_results if "BREAKOUT" in r.category],
            key=lambda r: r.score, reverse=True
        )[:10]

        # --- Metal momentum check: boost metals on hot metal days ---
        metal_names = set(self.config.get("metal_symbols", []))
        metal_results = [r for r in all_results if r.symbol in metal_names]
        avg_metal_move = 0.0
        if metal_results:
            avg_metal_move = sum(abs(r.change_pct) for r in metal_results) / len(metal_results)
        metal_is_hot = avg_metal_move >= self.config.get("metal_momentum_threshold", 1.5)

        if metal_is_hot:
            boost = self.config.get("metal_score_boost", 20)
            for r in all_results:
                if r.symbol in metal_names:
                    r.score += boost
                    if "METAL" not in r.category:
                        r.category += "+METAL🔥"

        # --- Select wild-cards ---
        # Exclude stocks already in the fixed universe
        # Prioritise by composite score
        candidates = sorted(
            [r for r in all_results if r.nse_symbol not in existing_set and r.score > 0],
            key=lambda r: r.score, reverse=True
        )

        # Reserve up to 2 slots for metals on hot days (if not already in top picks)
        max_wc = self.config["max_wildcards"]
        if metal_is_hot:
            metal_reserve = self.config.get("metal_max_reserve", 2)
            metal_candidates = [c for c in candidates if c.symbol in metal_names]
            non_metal_candidates = [c for c in candidates if c.symbol not in metal_names]
            # Take top metals (up to reserve) + fill rest with non-metals
            reserved_metals = metal_candidates[:metal_reserve]
            remaining_slots = max_wc - len(reserved_metals)
            # Merge: metals first, then top non-metals, avoid duplicates
            wildcards = reserved_metals + [c for c in candidates if c not in reserved_metals][:remaining_slots]
            wildcards = wildcards[:max_wc]
        else:
            wildcards = candidates[:max_wc]

        summary = ScanSummary(
            timestamp=str(datetime.now()),
            total_fo_stocks=len(self._fo_stocks),
            scanned=len(quotes),
            top_gainers=gainers,
            top_losers=losers,
            volume_surges=vol_surges,
            breakouts=breakouts,
            wildcards=wildcards,
        )

        self._last_scan = summary
        self._all_results = all_results  # Store ALL for heat map
        self._metal_is_hot = metal_is_hot
        self._avg_metal_move = avg_metal_move
        return summary

    # ------------------------------------------------------------------
    # Pretty-print for bot console
    # ------------------------------------------------------------------

    def format_scan_summary(self, summary: ScanSummary) -> str:
        """Format scan summary for console output."""
        lines = []
        lines.append(f"\n📡 MARKET SCANNER — {summary.scanned}/{summary.total_fo_stocks} F&O stocks scanned")
        
        if self._metal_is_hot:
            lines.append(f"  🔥 METALS HOT — avg move {self._avg_metal_move:.2f}% → reserving up to 2 metal wild-card slots")

        if summary.top_gainers:
            lines.append("  🟢 TOP GAINERS:")
            for g in summary.top_gainers[:5]:
                lines.append(f"     {g.symbol:15} +{g.change_pct:.2f}%  ₹{g.ltp:.2f}  Vol:{g.volume:,}")

        if summary.top_losers:
            lines.append("  🔴 TOP LOSERS:")
            for l in summary.top_losers[:5]:
                lines.append(f"     {l.symbol:15} {l.change_pct:.2f}%  ₹{l.ltp:.2f}  Vol:{l.volume:,}")

        if summary.volume_surges:
            lines.append("  📊 VOLUME SURGES:")
            for v in summary.volume_surges[:5]:
                lines.append(f"     {v.symbol:15} {v.change_pct:+.2f}%  ₹{v.ltp:.2f}  Vol:{v.volume:,}")

        if summary.breakouts:
            lines.append("  🚀 BREAKOUT CANDIDATES:")
            for b in summary.breakouts[:5]:
                lines.append(f"     {b.symbol:15} +{b.change_pct:.2f}%  ₹{b.ltp:.2f}  near day-high")

        if summary.wildcards:
            lines.append("  ⭐ WILD-CARD ADDITIONS:")
            for w in summary.wildcards:
                lines.append(f"     {w.symbol:15} {w.change_pct:+.2f}%  ₹{w.ltp:.2f}  [{w.category}]  Score:{w.score:.0f}")
        else:
            lines.append("  ⭐ No wild-cards — fixed universe covers the action today")

        return "\n".join(lines)

    def get_broad_market_heat(self, existing_universe: set = None) -> str:
        """Return a compact one-liner-per-stock heat map of ALL scanned F&O stocks.
        
        Sorted by absolute change_pct descending. Includes stocks NOT in the
        fixed universe so GPT can see opportunities outside the 31 curated.
        Shows top 40 movers (beyond that it's mostly flat/noise).
        """
        if not self._all_results:
            return "No scanner data available yet"
        
        existing_universe = existing_universe or set()
        
        # Sort by abs change — biggest movers first
        sorted_all = sorted(self._all_results, key=lambda r: abs(r.change_pct), reverse=True)
        
        lines = []
        for r in sorted_all[:40]:
            tag = "⭐NEW" if r.nse_symbol not in existing_universe else "CUR"
            arrow = "🟢" if r.change_pct > 0.5 else "🔴" if r.change_pct < -0.5 else "⚪"
            lines.append(f"  {arrow} {r.symbol:15} {r.change_pct:+.2f}% ₹{r.ltp:.0f} Vol:{r.volume:,} [{r.category}] {tag}")
        
        return "\n".join(lines) if lines else "No significant movers"

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    @property
    def last_scan(self) -> Optional[ScanSummary]:
        return self._last_scan


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_scanner_instance: Optional[MarketScanner] = None


def get_market_scanner(kite: KiteConnect = None) -> MarketScanner:
    """Get or create the market scanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        if kite is None:
            raise ValueError("MarketScanner requires a KiteConnect instance on first init")
        _scanner_instance = MarketScanner(kite)
    return _scanner_instance
