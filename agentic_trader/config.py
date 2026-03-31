"""
AGENTIC TRADING SYSTEM — TITAN
Uses OpenAI GPT for reasoning + Zerodha for execution

══════════════════════════════════════════════════════════════════════════════
  TITAN'S CORE GOAL (NEVER FORGET — READ THIS BEFORE EVERY CONFIG CHANGE):
══════════════════════════════════════════════════════════════════════════════
  Titan exists to generate CONSISTENT DAILY PROFIT through HIGH-CONVICTION,
  SELECTIVE trades. Every trade must have a real statistical edge.

  QUALITY > QUANTITY. Always.

  Principles:
    1. Capital preservation FIRST — never risk capital on weak signals
    2. Fewer trades with high conviction >> many trades with loose filters
    3. ROI is the metric — not trade count, not "activity"
    4. If a strategy isn't profitable, TIGHTEN it — don't loosen to get volume
    5. Never relax thresholds just to see trades flow — that destroys capital
    6. Proven strategies (ORB_BREAKOUT, WATCHER, SNIPER) get priority capital
    7. Experimental strategies (TEST_GMM, TEST_XGB) stay on TIGHT leash:
       small size, low daily cap, high conviction required
══════════════════════════════════════════════════════════════════════════════

PRIORITY: Capital preservation > consistent execution > performance
"""

import os
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import openai

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# ========== CONFIGURATION ==========

# OpenAI API Key - Set via environment variable or .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Zerodha Credentials - Set via environment variable or .env file
ZERODHA_API_KEY = os.environ.get("ZERODHA_API_KEY", "")
ZERODHA_API_SECRET = os.environ.get("ZERODHA_API_SECRET", "")

# ========== BROKERAGE CONFIG ==========
BROKERAGE_PCT = 0.006  # 0.6% of total turnover (paper trading estimate)

# ========== TRADING MODE (from .env) ==========
# Read TRADING_MODE from .env: "PAPER" (default) or "LIVE"
# CLI --live flag overrides this. autonomous_trader sets final value at startup.
_env_trading_mode = os.environ.get("TRADING_MODE", "PAPER").strip().upper()
PAPER_MODE = (_env_trading_mode != "LIVE")  # True=paper, False=live

# Zerodha actual charges for LIVE trading:
# - Brokerage: ₹20/order (flat, per executed order)
# - STT: 0.0625% on sell side (options buy: NIL, options sell: 0.0625% on premium)
# - Exchange txn charges: ~0.053% (NSE F&O)
# - GST: 18% on (brokerage + exchange charges)
# - SEBI charges: ₹10 per crore
# - Stamp duty: 0.003% on buy side

def calc_brokerage(entry_price, exit_price, quantity):
    """Calculate all-in trading costs.
    
    Paper mode: 0.6% of turnover (conservative estimate).
    Live mode:  Zerodha actual fee structure (significantly cheaper).
    
    Uses module-level PAPER_MODE flag (set at startup).
    """
    if PAPER_MODE:
        turnover = abs(entry_price * quantity) + abs(exit_price * quantity)
        return round(turnover * BROKERAGE_PCT, 2)
    
    # === LIVE MODE: Zerodha exact fee calculation ===
    buy_value = abs(entry_price * quantity)
    sell_value = abs(exit_price * quantity)
    turnover = buy_value + sell_value
    
    # 1. Brokerage: ₹20 per order × 2 (entry + exit), or 0.03% whichever is lower
    brokerage_per_leg = min(20, turnover * 0.0003)
    brokerage = brokerage_per_leg * 2  # Entry + Exit
    
    # 2. STT: Options sell = 0.0625% on premium
    stt = sell_value * 0.000625
    
    # 3. Exchange charges: ~0.053% of turnover
    exchange_charges = turnover * 0.00053
    
    # 4. GST: 18% on (brokerage + exchange charges)
    gst = (brokerage + exchange_charges) * 0.18
    
    # 5. SEBI charges: ₹10 per crore
    sebi = turnover * 0.000001
    
    # 6. Stamp duty: 0.003% on buy side
    stamp = buy_value * 0.00003
    
    total = brokerage + stt + exchange_charges + gst + sebi + stamp
    return round(total, 2)

# ========== CAPITAL SWAP / EVICTION ENGINE ==========
# When exposure >= threshold and a high-priority trade arrives,
# evict the worst-performing stale position to free capital.
CAPITAL_SWAP = {
    "enabled": True,
    "exposure_threshold_pct": 85.0,      # Only attempt swap when exposure >= 85%
    "min_hold_minutes": 30,              # Don't evict positions held < 30 min
    "min_stale_minutes": 45,             # Position must be stale >= 45 min to be evict-eligible
    "max_evict_loss_pct": 15.0,          # Don't evict if unrealized loss > 15% (let SL handle it)
    # Priority tier for INCOMING trades (higher = more deserving of capital)
    # Incoming trade must have higher priority than the BEST open position to evict
    "priority_tiers": {
        "OI_WATCHER": 9,
        "WATCHER_SPIKE_UP": 8, "WATCHER_SPIKE_DOWN": 8,
        "WATCHER_SLOW_GRIND_UP": 7, "WATCHER_SLOW_GRIND_DOWN": 7,
        "WATCHER_VOLUME_SURGE": 6,
        "WATCHER_NEW_DAY_HIGH": 5, "WATCHER_NEW_DAY_LOW": 5,
        "GMM_SNIPER": 4,
        "ELITE": 3,
        "GMM": 2,
    },
    "default_priority": 1,               # Anything not listed gets priority 1
}

# ========== HARD RULES (NEVER VIOLATE) ==========
HARD_RULES = {
    "RISK_PER_TRADE": 0.07,         # 7% base risk per trade (tiered: 7% premium, 5% std, 4% base)
    "MAX_DAILY_LOSS": 0.20,         # 20% max daily loss
    "PORTFOLIO_PROFIT_TARGET": 0.15,  # 15% realized+unrealized profit → KILL ALL & book profit
    "MAX_POSITIONS": 80,            # Max simultaneous positions (spreads count as 1)
    "MAX_POSITIONS_MIXED": 80,       # Max positions in MIXED regime
    "MAX_POSITIONS_TRENDING": 80,    # Max positions in BULLISH/BEARISH regime
    "STALE_DATA_SECONDS": 60,       # Data older than this is stale
    "API_RATE_LIMIT_MS": 350,       # Min ms between API calls (Kite allows ~3/s, 350ms is safe)
    "CAPITAL": 500000,              # Starting capital ₹5,00,000 — don't inflate for exposure tricks
    "REENTRY_COOLDOWN_MINUTES": 30, # Skip same underlying for 30 min after any exit
    # === PENNY PREMIUM PROTECTION (Feb 24 fix) ===
    # Cheap options (<₹3) create position-size bombs:
    #   ₹0.94 × lot 4300 = ₹4042/lot → sizer gives 15 lots = 64,500 units
    #   ₹0.10 move = ₹6,450 loss. Massive notional on tiny premium.
    # Also cap max units per trade to prevent outsized exposure.
    "MIN_OPTION_PREMIUM": 3.0,      # Reject options with LTP < ₹3 (avoids penny traps)
    "MAX_UNITS_PER_TRADE": 30000,   # Hard cap on units (lots × lot_size) per single trade
}

# === FULL F&O UNIVERSE SCAN ===
# When True, scans ALL ~200 F&O stocks each cycle (not just 24 curated + 25 wildcards).
# WebSocket provides real-time OI for all futures (zero API overhead).
# Indicator calculation uses 8 parallel threads with 10-min cache.
# First cold cycle: ~45-60s, subsequent cached cycles: ~15-25s.
FULL_FNO_SCAN = {
    "enabled": True,               # True = scan ALL F&O stocks, False = curated + wildcards only
    "max_indicator_stocks": 10,     # Top 10 F&O stocks by composite rank (reduced from 40 — watcher-first strategy)
    "min_change_pct_filter": 0.5,   # Minimum change% to even consider (dead stocks filtered)
    "indicator_threads": 12,        # Thread pool size for parallel historical_data fetches
    "prefer_ws_quotes": True,       # Use WebSocket quote cache for initial screen (skip REST batch)
}

# === BREAKOUT WATCHER (WebSocket-Driven Fast Entry) ===
# Monitors ALL ~200 F&O stocks via WebSocket ticks in real-time.
# Detects breakout signals (price spike, day high/low break, volume surge)
# and pushes them to a queue that the main loop drains within 1 second.
# The triggered stock then goes through the FULL pipeline (score → ML → GMM → trade).
# This replaces the 5-min poll delay for fast setups while the scan loop continues
# unchanged for slower setups (sniper, model-tracker, mean-reversion).
BREAKOUT_WATCHER = {
    "enabled": True,
    "watcher_start": "09:20",            # Watcher ignores triggers before this time (let ORB settle)
    # --- Trigger Thresholds (tightened Mar-10: catch sustainable moves, not opening spikes) ---
    "price_spike_pct": 0.6,           # Trigger if price moves ≥0.6% within 60s baseline (after 09:45)
    "price_spike_pct_open": 1.0,      # Opening 30min spike threshold (09:15-09:45) — higher bar to avoid noise
    "price_spike_open_until": "09:45", # Time cutoff for opening spike threshold
    "day_extreme_trigger": True,      # Trigger on new day high / day low break
    "day_extreme_min_move_pct": 0.35, # Day extreme must also show ≥0.35% move from baseline (was 0.25)
    "volume_surge_multiplier": 3.0,   # Trigger if tick volume ≥ 3.0x rolling average
    # --- Sustain Filter (signal-aware — Mar 12 latency fix) ---
    # Retrace check (>50% retrace = fail) is the real fakeout filter.
    # Signal-specific sustain times: stronger signals need less proof time.
    "sustain_seconds": 15,            # PRICE_SPIKE: 15s quick sanity check (OI fetched inline at drain provides real conviction)
    "sustain_seconds_extreme": 20,    # DAY_HIGH/LOW break — 20s proof
    "sustain_seconds_volume": 20,     # VOLUME_SURGE with confirmed ticks — 20s proof
    "sustain_seconds_grind": 30,      # SLOW_GRIND: 30s sustain proof (was 10 — too short, small grinds slipping through)
    "sustain_recheck_pct": 0.4,       # Spike/extreme: price must still be ≥0.4% from baseline (33% retrace room at 0.6 spike)
    "sustain_recheck_pct_volume": 0.5,# Volume surge: price must still be ≥0.5% from baseline (was 0.4 — 0.3% moves too weak)
    "sustain_retrace_max_pct": 50.0,  # Fail sustain if price retraces >50% from peak move during window
    "volume_surge_min_move_pct": 0.5, # VOLUME_SURGE must show ≥0.5% price move to enter sustain (was 0.3 — too noisy)
    # --- Slow Grind Detection ---
    "slow_grind_pct": 1.0,              # 1.0%+ move triggers SLOW_GRIND (was 0.75 — too many small grinds reversing)
    # --- Cooldown (anti-spam) ---
    "cooldown_seconds": 120,          # Don't re-trigger same stock within 2 minutes (was 3 — too slow)
    "max_triggers_per_minute": 10,    # Max triggers per minute — relaxed for crash days
    # --- Priority Queue ---
    "queue_size": 100,                # Max queued triggers — evicts weakest when full
    "priority_bypass_pct": 2.0,       # Moves ≥2.0% BYPASS rate limit — always enter queue
    # --- Timing (SUSTAINABILITY: wait for ORB range to fully form) ---
    "active_after": "09:35",          # Don't trigger before 09:35 (was 09:30 — ORB needs 20 min to form, VWAP unreliable before this)
    "active_until": "15:10",          # Don't trigger after 15:10 (too close to close)
    # --- Score Gate (LOWERED Mar-12: macro trend filter now handles quality, let more setups through) ---
    "min_score": 30,                  # Minimum score to trade (was 35 — lowered to let more watcher setups through)
    "orb_min_score": 35,              # ORB_BREAKOUT floor (was 40 — lowered Mar 30)
    "orb_min_move_prob": 0.55,        # ORB/SPIKE/DAY trades need P(move)≥55% (breadth relaxation → 0.45)
    "watcher_min_move_prob": 0.50,    # WATCHER P(move) floor (was 0.40 — raised to 0.50, Mar-28)
    "watcher_min_adx": 15,            # ADX floor — watcher trigger IS trend evidence
    # --- Scorer Conflict (NEW Mar-10: trust scorer when score is low) ---
    "scorer_conflict_max_score": 40,          # G2b: block scorer-trigger conflict only if score < 40 (was 50)  # If scorer disagrees and score < this, block trade
    # --- RSI Extreme Guard (NEW Mar-10: don't buy into exhausted moves) ---
    "rsi_extreme_pe_max": 24,         # Block PE (SELL) if RSI < 24 — oversold bounce imminent
    "rsi_extreme_ce_min": 75,         # Block CE (BUY) if RSI > 75 — overbought pullback imminent
    # --- Exhaustion Index (NEW Mar-10: statistical move exhaustion detection) ---
    # Combines: intraday move from open + recent trigger move + position in day range + RSI
    # Into a 0-100 index: >70 = exhausted move, don't chase.
    # On trending days, +15 regime boost. High-score setups get additional +10-15 boost.
    "exhaustion_index_block": 70,     # Block if Exhaustion Index > this threshold (was 60 — too aggressive, blocked legit momentum)
    # --- Opening Period Cap (early-market protection) ---
    "max_trades_before_1000": 2,      # Max watcher trades before 10:00 IST (was 3)
    # --- Early Market Hardening (9:35-9:55 is still dicey) ---
    "early_market_end": "09:55",       # Early market protection window end
    "early_market_min_score": 40,      # Lowered from 50 — review after tomorrow's session
    "early_market_min_dir_conf": 45,   # Min direction confidence during early market
    "early_market_min_sustain_pct": 1.0, # Min sustained move % for spikes in early market (normal=0.6)
    # --- Dynamic Batch ---
    "max_triggers_per_batch": 6,      # Max triggers fed to pipeline per drain
    "max_trades_per_scan": 5,         # Max trades PLACED per scan (raised from 2 → 5 to catch all strong triggers)
    # --- VIX-based score penalty for elevated options pricing ---
    "vix_penalty_above": 22.0,        # If India VIX > 22, apply -2 score penalty per VIX point above 22
    "vix_penalty_per_point": 2,       # Penalty per VIX point (VIX=25 → -6 penalty, VIX=28 → -12 penalty)
    "vix_hard_block_above": 32.0,     # Block ALL watcher entries if VIX > 32 (extreme crash territory)
    # --- Watcher Momentum Exit (exit on spike peak / crater bottom reversal) ---
    "momentum_exit": {
        "enabled": True,
        "reversal_pct": 0.5,               # % reversal from peak/trough — PRICE-ONLY threshold (no signal confirmation)
        "min_hold_seconds": 60,             # Min seconds after entry before checking (let trade breathe)
        "min_favorable_move_pct": 0.5,      # Underlying must have moved ≥0.5% in our direction (real spike, not noise)
        "check_interval_seconds": 5,        # How often to check (in monitor loop)
        "bypass_cooldown": True,            # Remove symbol from _watcher_fired_this_session after exit
        "only_in_profit": True,             # Only trigger WME if option premium is in profit (don't cut losers early)
        "skip_trailing_active": True,       # Don't WME trades already managed by trailing stop (let winners run)
        # --- Multi-Signal Confirmation (mirrors watcher entry signals but in reverse) ---
        "volume_dryup_ratio": 0.30,         # Vol rate < 30% of avg spike rate → volume dried up
        "momentum_decay_threshold": 0.70,   # Momentum < 30% of peak velocity → momentum decayed (70% decay)
        "pressure_shift_enabled": True,     # Buy/sell qty imbalance as 3rd signal
        "pressure_shift_ratio": 1.5,        # Opposing qty > 1.5× favorable qty → pressure shifted
        "confirmed_reversal_pct": 0.25,     # Lower threshold when 2+ signals confirm reversal
        "partial_confirm_reversal_pct": 0.35,  # Mid threshold when 1 signal confirms
        "sample_window_seconds": 180,       # Rolling window for price/volume history samples (was 90 — too narrow for slow grinds)
        # --- Option Premium Tracking (NEW: track premium directly, not just underlying) ---
        "premium_reversal_pct": 8.0,        # Exit if option premium drops ≥8% from its peak (direct profit protection)
        "premium_reversal_confirmed_pct": 5.0,  # Lower threshold when multi-signal confirms (2+ signals)
        # --- Profit-Tiered Thresholds (NEW: bigger profits = tighter stops) ---
        "profit_tiers": [
            # (min_premium_gain_pct, reversal_pct_override)
            # Example: if option is up ≥80%, use 0.20% underlying reversal instead of 0.5%
            {"min_gain_pct": 80, "reversal_pct": 0.20, "premium_rev_pct": 4.0},
            {"min_gain_pct": 50, "reversal_pct": 0.25, "premium_rev_pct": 5.0},
            {"min_gain_pct": 30, "reversal_pct": 0.30, "premium_rev_pct": 6.0},
        ],
        # --- CE/PE Asymmetry (NEW: different behavior for calls vs puts) ---
        "ce_reversal_multiplier": 0.85,     # CE: tighter exit (CE dies faster on reversal). Multiply threshold by 0.85
        "pe_reversal_multiplier": 0.90,     # PE: tighter on bounce (panic reversals are sharp). Multiply threshold by 0.90
        # --- Reversal Acceleration (NEW: detect if reversal is speeding up) ---
        "reversal_accel_enabled": True,     # Enable reversal acceleration as a 4th signal
        "reversal_accel_threshold": 1.5,    # Reversal speed > 1.5x average speed → accelerating
        # --- Trend Shield: Fibonacci Retracement Floor (smart: scale threshold with move size) ---
        # Instead of fixed 0.25-0.50% reversal threshold, allow pullback proportional to the move.
        # A stock up +5% gets 1.91% pullback room (5% × 0.382). Prevents killing trending winners.
        "retracement_floor_enabled": True,
        "ul_retracement_factor": 0.382,          # Underlying: allow 38.2% Fib retracement of the move
        "premium_retracement_factor": 0.382,     # Premium: allow 38.2% retracement of premium gain
        # --- Minimum Premium Gain Floor (smart: don't WME tiny gains) ---
        # A +1.8% premium gain isn't worth protecting — the trade hasn't developed yet.
        # Let exit_manager handle it normally, giving the trade room to run.
        "min_premium_gain_pct": 5.0,             # Min peak premium gain % before WME can trigger
        # --- Armed Grace Period (smart: V-shape pullback filter) ---
        # When WME triggers, enter ARMED state for N seconds instead of instant exit.
        # If price recovers → DISARM (it was just a pullback). If reversal deepens → exit fast.
        "armed_grace_seconds": 90,               # Wait 90s for recovery before committing to exit
        "armed_deepen_multiplier": 1.5,          # Exit immediately if reversal > 1.5× threshold during grace
        # --- OI Reversal Exit (NEW: operators reversing position = move is done) ---
        # Fetch OI every N seconds per tracked symbol. When OI signal flips against
        # trade direction (e.g. SHORT_COVERING after PE trade), exit immediately.
        # OI reversal counts as 2 confirmations (strongest signal — operators know first).
        "oi_reversal_exit_enabled": True,
        "oi_check_interval_seconds": 60,          # Fetch OI every 60s per symbol (not every 5s — too expensive)
        "oi_reversal_immediate_exit": True,        # If OI reverses, trigger exit without grace period
        "oi_reversal_min_hold_seconds": 90,        # Min hold before OI exit check (let trade develop)
        # --- OI Confirmation Shield & Accelerator (smart exit adjustment for ALL WME trades) ---
        # OI confirms trade → WIDEN thresholds (avoid premature exit on healthy pullback)
        # OI opposes trade → TIGHTEN thresholds (exit faster when smart money reverses)
        "oi_shield_enabled": True,                 # Enable OI-based threshold adjustment
        "oi_shield_confirm_multiplier": 1.40,      # OI confirms direction → thresholds × 1.40 (40% more room)
        "oi_shield_writer_confirm_bonus": 1.15,    # Writer-dominant confirming → extra 15% room on top
        "oi_shield_oppose_multiplier": 0.70,       # OI opposes direction → thresholds × 0.70 (30% tighter)
        "oi_shield_neutral_multiplier": 0.90,      # OI went neutral → slightly tighter (conviction fading)
        "oi_shield_strength_collapse_mult": 0.80,  # OI strength < 50% of entry → 20% tighter
        "oi_shield_strength_collapse_ratio": 0.50, # Threshold: current/entry strength ratio
    },
    # --- OI_WATCHER Exit Intelligence (thesis-based exit for pure OI trades) ---
    # OI_WATCHER trades have NO model/score thesis — ONLY OI.
    # When OI thesis breaks, exit immediately. No grace period, no V-shape filter.
    "oi_watcher_exit": {
        "enabled": True,
        "min_hold_seconds": 600,                   # [FIX Mar 20] 10 min grace — winners need 10-60 min to develop. SL protects downside.
        "oi_check_interval_seconds": 90,            # [FIX Mar 19] Was 45s — OI data is noisy, 90s smooths noise
        "direction_flip_exit": True,                # Exit if OI direction flips (LONG_BUILDUP → SHORT_BUILDUP)
        "strength_collapse_exit": True,             # Exit if OI strength drops significantly
        "strength_collapse_ratio": 0.25,            # [FIX Mar 19] Was 0.40 — too trigger-happy. Now 25% of entry (only extreme collapse)
        "strength_collapse_only_losing": True,      # Only exit on strength collapse if position is losing
        "participant_flip_exit": True,              # Exit if WRITER_DOMINANT → BUYER_DOMINANT
        "signal_neutral_exit": True,                # Exit if OI signal becomes NEUTRAL (buildup dissolved)
        "skip_if_trailing_active": False,           # Don't skip — OI thesis break overrides trailing
        "consecutive_confirms_required": 3,         # [FIX Mar 20] 3 consecutive bad readings — OI must be against you for 4.5min straight
        "premium_bypass_pct": 2.0,                  # [FIX Mar 25] If premium ≥ 2%, skip OI exit — let trailing manage
    },
}

# === EARLYBIRD STRATEGY (Opening Volatility 09:15-09:45) ===
# Three distinct modes — tracked separately for performance analysis:
#   A = Gap continuation (highest quality)
#   B = Strong opening directional move (medium quality)
#   C = Opening spike (lowest quality — needs volume + hold confirmation)
#
# Market-context gating: Is the stock move idiosyncratic or just market beta?
# If NIFTY is moving the same way and the stock is HIGH_BETA, it's likely just
# index-driven — lower conviction. Idiosyncratic moves (stock moves, index doesn't,
# or stock moves AGAINST index) are highest quality.
#
# Shared settings applied to all three modes:
EARLYBIRD_COMMON = {
    "enabled": True,
    "start_time": "09:15",               # Start detecting from market open (first tick)
    "end_time": "09:30",                  # 15-minute earlybird window — normal watcher takes over at 09:30
    "max_trades": 999,                    # No cap during paper trading (set to 6 for live)
    "skip_dir_conf_gate": True,           # Skip direction confidence gate (unreliable at open)
    "min_vol_ratio": 1.5,                 # Current vol delta must be ≥ 1.5× rolling avg
    # --- Market Context Gating ---
    "market_context_enabled": True,       # Enable index/sector alignment checks
    "index_same_dir_threshold": 0.3,      # NIFTY move ≥0.3% in same direction = index-driven signal
    "idiosyncratic_bonus": 4,             # Bonus when stock moves but index doesn't (or opposite)
    "beta_penalty": 5,                    # Penalty when HIGH_BETA stock mirrors index move
    "sector_aligned_penalty": 3,          # Penalty when sector index also moves same direction
}

# --- MODE A: Gap Continuation (Highest Quality) ---
# Gap ≥ 0.8% from prev close AND price continuing in gap direction.
# Highest conviction — real institutional positioning from overnight flow.
# Allow larger sizing. Generous sustain.
EARLYBIRD_A = {
    "gap_open_pct": 0.9,                  # Min gap from prev close (0.9%)
    "gap_strong_pct": 1.5,                # Strong gap threshold (1.5%+)
    "gap_continuation_min": 0.35,         # Price must continue ≥ 0.35% in gap direction from open
    "sustain_seconds": 8,                 # Shortest sustain — gap continuations are high-conviction
    "sustain_min_hold_pct": 0.4,          # Relaxed hold — gap moves are sticky
    "min_score": 20,                      # Very relaxed — gap itself is the thesis
    "min_move_prob": 0.30,                # Very relaxed XGB floor
    "trigger_bonus": 15,                  # Highest base bonus (gap = strongest opening signal)
    "gap_bonus": 5,                       # Extra for gap confirmation
    "strong_gap_bonus": 3,                # Extra for strong gap (≥1.5%)
    "lot_multiplier": 1.5,               # Larger sizing — highest quality signals
    "gap_strong_lot_multiplier": 2.0,     # 2x for strong gaps
}

# --- MODE B: Strong Opening Directional Move (Medium Quality) ---
# Price moved ≥ 0.7% from today's open (with or without gap).
# Medium conviction — can be real discovery or just noise.
# Normal sizing. Standard sustain.
EARLYBIRD_B = {
    "enabled": False,                     # DISABLED — medium conviction, too noisy at open
    "opening_move_pct": 0.7,              # Min move from day open (0.7%)
    "sustain_seconds": 10,                # Standard sustain
    "sustain_min_hold_pct": 0.5,          # Must hold 0.5% from baseline
    "min_score": 25,                      # Standard early-market floor
    "min_move_prob": 0.35,                # Standard XGB floor
    "trigger_bonus": 10,                  # Medium bonus
    "lot_multiplier": 1.0,               # Normal lot size
}

# --- MODE C: Opening Spike (Lowest Quality) ---
# Price spiked ≥ 1.2% from 60s baseline — pure volatility burst.
# Lowest conviction unless paired with strong volume + sustain hold.
# Smaller sizing or stricter sustain required.
EARLYBIRD_C = {
    "enabled": False,                     # DISABLED — lowest conviction, spike noise
    "opening_spike_pct": 1.2,             # Min spike from 60s baseline
    "sustain_seconds": 15,                # Longest sustain — must PROVE the spike holds
    "sustain_min_hold_pct": 0.7,          # Stricter hold — spike must not fade
    "min_score": 30,                      # Tighter score floor
    "min_move_prob": 0.40,                # Tighter XGB floor (spike could be noise)
    "min_vol_ratio": 2.0,                 # Require 2x volume (higher than A/B)
    "trigger_bonus": 7,                   # Lowest bonus
    "lot_multiplier": 0.5,               # Half lot — lowest quality
}

# Backward-compat alias: modules that import 'EARLYBIRD' get the common config
# with mode-specific overrides merged in at runtime
EARLYBIRD = {**EARLYBIRD_COMMON}

# === WATCHER IV CRUSH OVERRIDES (more lenient — breakouts naturally have elevated IV) ===
# These override the global IV_CRUSH_GATE when setup_type='WATCHER'.
# Breakout entries inherently happen DURING volatility spikes, so the normal IV gate
# is too strict — it would block many valid watcher trades.
WATCHER = {
    "iv_crush_overrides": {
        "iv_rv_ratio_hard_block": 2.2,     # Global=1.8 → watcher=2.2 (lenient: breakouts have elevated IV)
        "iv_rv_ratio_reduce_lots": 1.5,    # Global=1.3 → watcher=1.5 (tolerate higher ratio)
        "max_atm_iv_pct": 70,              # Global=65 → watcher=70 (raised Mar 30, was 55)
        "reduce_atm_iv_pct": 52,           # Global=45 → watcher=52 (raised Mar 30, was 42)
    },
}

ORB_BREAKOUT = {
    "iv_crush_overrides": {
        "iv_rv_ratio_hard_block": 2.0,     # Slightly more lenient than global, but tighter than WATCHER
        "iv_rv_ratio_reduce_lots": 1.4,
        "max_atm_iv_pct": 65,              # Global=65 → ORB=65 (raised Mar 30, was 52)
        "reduce_atm_iv_pct": 50,           # Global=45 → ORB=50 (raised Mar 30, was 40)
    },
}

# === GTT SAFETY NET (Server-Side SL + Target) ===
# After placing live orders, a GTT TWO-LEG (OCO) is placed on Zerodha's servers.
# If Titan crashes, loses internet, or the PC shuts down, the GTT persists and
# protects the position with server-side SL + target.
# The normal SL-M order handles fast intraday exits; the GTT is the BACKUP.
GTT_CONFIG = {
    "enabled": True,                  # Place GTT safety nets for live trades
    "equity_gtt": True,               # GTT for equity intraday trades
    "option_gtt": True,               # GTT for single option buys (not spreads)
    "sl_buffer_pct": 1.0,             # GTT SL trigger 1% wider than primary SL (avoid double-trigger)
    "target_buffer_pct": 0.5,         # GTT target trigger 0.5% tighter than primary target
    "limit_price_buffer_pct": 2.0,    # LIMIT price buffer from trigger (ensure fill in gaps)
    "cleanup_on_exit": True,          # Auto-cancel GTT when trade exits normally
    "cleanup_on_startup": True,       # Check for orphaned GTTs on startup
    "log_gtt_events": True,           # Log all GTT place/cancel/trigger events
}

# === AUTOSLICE (Freeze Quantity Protection) ===
# Kite auto-splits orders exceeding exchange freeze limits.
# Without this, large F&O orders (e.g., >1800 qty NIFTY) get flat REJECTED.
AUTOSLICE_ENABLED = True

# === ELITE AUTO-FIRE (Score-Based Autonomous Execution) ===
# Stocks scoring ≥ elite_threshold are executed IMMEDIATELY via the options
# pipeline — no need to wait for GPT to pick them. GPT validated the scoring
# logic; scores 78+ have 80% WR and generate the monster trades.
# This happens BEFORE the GPT prompt is built, so GPT only sees remaining setups.
ELITE_AUTO_FIRE = {
    "enabled": True,
    "elite_threshold": 76,            # Score ≥ this → auto-fire [was 70, tightened Feb 27 +6pts]
    "max_auto_fires_per_cycle": 3,    # Max auto-fired trades per scan cycle
    "require_setup": True,            # Must have a valid setup (ORB/VWAP/MOMENTUM), not just high score
    "log_all": True,                  # Log every auto-fire decision to scan_decisions.json
    # --- ML Confidence Floor (prevent coin-flip ML from auto-firing) ---
    "min_ml_confidence": 0.55,        # Block if ml_confidence < 55% (VBL had 0.48 → coin flip)
    "min_ml_move_prob": 0.52,         # Block if ml_move_prob < 52% (directional signal too weak)
    # --- Move Exhaustion Gate (prevent late entries after big moves) ---
    "max_existing_move_pct": 2.5,     # Block if stock already moved >2.5% from prev close at entry
    "exhaustion_min_acceleration": 0.3,  # Override exhaustion if last-candle acceleration > 0.3%
}

# === DYNAMIC MAX PICKS (GPT picks scale with signal quality) ===
# When market is trending and there are many high-scoring setups, let GPT pick
# more than the default 3. On choppy days, restrict to 2.
DYNAMIC_MAX_PICKS = {
    "enabled": True,
    "default_max": 3,                 # Standard max picks per GPT call
    "elite_bonus_max": 5,             # If ≥3 stocks score 65+, allow up to 5 picks
    "min_score_for_bonus": 65,        # Threshold to count toward bonus check
    "min_count_for_bonus": 3,         # Need this many stocks above threshold
    "choppy_max": 2,                  # If breadth is MIXED and <3 setups score 60+, restrict to 2
}

# === ADAPTIVE SCAN INTERVAL ===
# Dynamically adjust scan frequency based on signal quality from last cycle.
# Hot market (many signals) → scan faster to catch more.
# Dead market (no signals) → scan slower to save GPT calls.
ADAPTIVE_SCAN = {
    "enabled": True,
    "fast_interval_minutes": 3,       # Fast scan when signals are hot
    "normal_interval_minutes": 5,     # Standard scan (default)
    "slow_interval_minutes": 7,       # Slow scan when nothing is moving
    "fast_trigger_signals": 3,        # Switch to fast if ≥ N signals scored 65+
    "slow_trigger_signals": 0,        # Switch to slow if 0 signals scored 55+
    "min_fast_interval_minutes": 2,   # Never scan faster than this (API limits)
}

# === DOWN-RISK SOFT SCORING (VAE+GMM anomaly detector) ===
# Applies a soft score bonus/penalty to _pre_scores based on down-risk model.
# Does NOT block any trades — the existing workflow continues unmodified.
# Additionally places N exclusive "model-tracker" trades per day purely from
# the model's safest candidates, to independently evaluate model performance.
DOWN_RISK_GATING = {
    "enabled": True,
    # Direction-aware graduated soft scoring
    # VAE+GMM+GBM anomaly score ∈ [0,1]. UP_Flag threshold ≈ 0.19, Down_Flag ≈ 0.23 (v5 actual).
    # UP_Flag=True (UP regime): hidden crash risk → opposes CE/BUY, confirms PE/SELL
    # Down_Flag=True (DOWN regime): hidden bounce risk → opposes PE/SELL, confirms CE/BUY
    "high_risk_threshold": 0.40,      # Score > this → strong penalty/boost depending on direction
    "high_risk_penalty": 15,          # Points deducted when high-dr OPPOSES trade direction
    "mid_risk_penalty": 8,            # Points deducted when flagged + opposes direction
    "clean_threshold": 0.15,          # Score < this → boost (genuine clean pattern)
    "clean_boost": 8,                 # Points added for clean/genuine pattern or dr-confirms-direction
    "model_tracker_trades": 14,       # Exclusive model-only trades per day for tracking
    "log_rejections": True,           # Log score adjustments for diagnostics
    # === CONFIRM SCORE FLOOR FOR MODEL-TRACKER ===
    # 18-case matrix gates on flags, but ALL_AGREE also needs minimum anomaly strength.
    # Flag=True guarantees score > model threshold (UP≈0.19, DOWN≈0.23).
    # Floor must be BELOW these thresholds so flagged stocks aren't double-filtered.
    "min_confirm_score": 0.0,          # Flag gate is sufficient — smart_score safety weight handles quality ranking
    "min_smart_score": 55,             # Minimum smart score for model tracker trades
    # === ALL_AGREE AMPLIFIED BET ===
    # All 3 models agree (Titan + GMM + XGB) = strongest conviction → amplified lot sizing
    "all_agree_lot_multiplier": 1.5,   # 1.5x lots for ALL_AGREE (strongest conviction)
    "all_agree_min_down_score": 0.30,    # Down_Flag (bounce) floor for ALL_AGREE BUY — raised from 0.26 (POWERINDIA 0.260 was too weak)
    "all_agree_min_up_score": 0.30,      # UP_Flag (crash) floor for ALL_AGREE SELL — strong crash signal required
    "pmove_bonus_threshold": 0.80,       # P(move) >= this triggers score bonus for score-type trades
    "pmove_bonus_points": 25,            # Points added when P(move) exceeds threshold
}

# === ML DIRECTION CONFLICT FILTER ===
# Blocks/penalizes trades where XGBoost ML direction disagrees with scored direction.
# If BOTH XGBoost AND GMM disagree → HARD BLOCK (both ML systems say wrong direction).
# If only XGBoost disagrees → soft penalty on smart_score.
ML_DIRECTION_CONFLICT = {
    "enabled": True,
    "xgb_penalty": 15,              # Smart-score penalty when only XGB disagrees
    "gmm_caution_threshold": 0.15,  # dr_score above this = GMM not fully confident → counts as GMM disagree
    "min_xgb_confidence": 0.55,     # Only consider XGB disagreement when ml_move_prob >= this
    "block_gpt_trades": True,       # Also apply to GPT direct-placed trades
}

# === ML_OVERRIDE_WGMM TIGHTENING ===
# Extra gates for ML_OVERRIDE_WGMM (XGB+GMM override Titan direction) trades.
# These are ON TOP OF the normal gating — ML_OVERRIDE_WGMM only fires when ALL pass.
#
# DR SCORE INTERPRETATION (correct — anomaly-based):
#   UP_Flag=True (UP regime, high dr) = hidden DOWN risk (crash likely) → opposes BUY/CE
#   Down_Flag=True (DOWN regime, high dr) = hidden UP risk (bounce likely) → opposes SELL/PE
#   gmm_confirms_direction = True means NO anomaly = clean pattern = safe
#   → ML_OVERRIDE fires when XGB opposes AND gmm_confirms_direction=True
ML_OVERRIDE_GATES = {
    "min_move_prob": 0.55,            # XGB P(MOVE) — must see real movement signal
    "min_dr_score": 0.15,             # GMM must show clean signal (low dr = confirmed direction, high dr = anomaly)
    "min_directional_prob": 0.30,     # XGB prob_up/prob_down — relaxed from 0.40 (Mar 2 fix: LT blocked at 0.29)
    "max_concurrent_open": 3,         # Max simultaneously open ML_OVERRIDE_WGMM positions
    "min_smart_score": 55,            # Minimum smart score for ML_OVERRIDE trades
}

# === GMM CONTRARIAN (DR_FLIP) — Feb 24 fix ===
# When GMM dr_score is VERY HIGH for XGB=FLAT stocks (default regime routing):
# The high dr is a strong anomaly signal — hidden risk that Titan may be wrong.
# Safety: requires high Gate P(MOVE) + XGB must not strongly disagree with flipped dir.
GMM_CONTRARIAN = {
    "enabled": True,
    "min_dr_score": 0.27,              # Require top ~15% anomalies
    "min_gate_prob": 0.65,             # Contrarian needs strong gate conviction
    "max_concurrent_open": 3,          # Max simultaneously open DR_FLIP positions
    "max_trades_per_day": 4,           # Conservative daily limit — these are contrarian
    "lot_multiplier": 1.0,             # Standard lots (not boosted — contrarian = careful)
    "score_tier": "standard",          # Standard risk tier, not premium
}

# === TEST_GMM: Flag-Confirmed Regime Divergence Strategy ===
# Fires when one GMM model flags AND the other is clean.
# Direction follows the FLAGGING model (not contrarian):
#
#   down_flag=True + down_score HIGH + up clean → confirmed downside → BUY PUT
#   up_flag=True   + up_score HIGH + down clean → confirmed upside   → BUY CALL
#
# The flag IS the directional confirmation. No contrarian flip.
#
# Unique vs other strategies:
#   GMM_CONTRARIAN = single high DR + XGB=FLAT → flip direction
#   GMM_SNIPER     = both DRs clean + high conviction → amplified bet
#   TEST_GMM       = flag-confirmed divergence → directional bet WITH the flag
#
# Model quality: DOWN AUROC=0.62 (decent, PUT side stronger)
#                UP   AUROC=0.56 (weaker, CALL side needs higher bar)
TEST_GMM = {
    "enabled": True,
    # DOWN model signal: down_flag confirms → BUY PUT
    # Data: winners DN range 0.245-0.343. Floor catches signal, cap blocks exhaustion.
    "down_min_score": 0.28,              # Data Mar16-17: DR<0.25 = 0% WR -₹19k. Raised floor to 0.28
    "down_max_score": 0.35,              # Winner max DN=0.343. DN>0.35 = 0W/3L (exhaustion trap)
    "down_max_opposite": 0.10,           # Clean side must be genuinely clean
    # UP model signal: up_flag confirms → BUY CALL
    "up_min_score": 0.28,                # Aligned with DOWN floor raise
    "up_max_score": 0.35,                # Cap over-confidence on CALL side too
    "up_max_opposite": 0.10,             # Clean side must be genuinely clean
    # Divergence quality — data-driven from 43 trades Mar 4-17
    # gap<=0.21: 11W/11L +Rs33k | gap>=0.24: 3W/6L -Rs31k | gap>=0.29: 0W/3L -Rs12k
    "min_divergence_gap": 0.13,          # Winner min gap=0.139 — catch all valid signals
    "max_divergence_gap": 0.23,          # Data: gap>=0.24 = 33%WR -Rs31k. Sweet spot <=0.23
    # FLAG-based conviction gates (model-calibrated thresholds)
    "require_signaling_flag": True,      # Signaling regime must FIRE its own anomaly flag
    "require_clean_no_flag": True,       # Clean side must NOT be flagging (reduces noise)
    # Pure GMM play — NO XGB involvement
    "require_xgb_agree": False,          # No XGB gating — pure regime divergence signal
    "min_gate_prob": 0.0,                # No P(MOVE) requirement — bypass XGB entirely
    "max_gate_prob": 1.0,                # No cap needed
    "max_ml_confidence": 1.0,            # No cap needed
    "min_smart_score": 45,               # Data Mar16-17: score≥45 = 50% WR +₹5k vs score<45 = heavy losses
    "max_trades_per_day": 50,            # Uncapped for paper tracking — conviction thresholds do the filtering
    "lot_multiplier": 1.0,               # Standard lots — paper tracking, not sizing up
    "score_tier": "standard",
}

# === TEST_XGB: Pure XGBoost Model Play (bypass GMM, smart_score, all other gates) ===
# Fires purely on XGB directional conviction: high P(MOVE) + clear directional lean.
# No GMM/DR gating, no smart_score, no IntradayScorer — pure XGB signal.
# Direction from prob_up vs prob_down; conviction from ml_move_prob (gate model).
#
# Unique vs other strategies:
#   MODEL_TRACKER = XGB + GMM + smart_score combined
#   ML_OVERRIDE   = XGB overrides Titan direction when GMM agrees
#   TEST_GMM      = pure GMM regime divergence, no XGB
#   TEST_XGB      = pure XGB conviction, no GMM
TEST_XGB = {
    "enabled": True,
    "min_move_prob": 0.48,               # Tightened: only trade when XGB sees clear movement signal
    "min_directional_prob": 0.25,        # Tightened: need real directional lean, not coin-flip
    "min_directional_margin": 0.06,      # Tightened: direction must be meaningfully lopsided
    "min_ml_confidence": 0.50,           # Tightened: require genuine model confidence
    "max_trades_per_day": 50,            # Uncapped for paper tracking — conviction thresholds do the filtering
    "lot_multiplier": 1.0,               # Standard lots — paper tracking, not sizing up
    "score_tier": "standard",
    # --- IV Crush Overrides (tighter than global IV_CRUSH_GATE) ---
    # SIEMENS PE had IV=33% but RV was low → IV/RV was high → IV crush killed it (-9.3% in 35min).
    # Global gate: block@2.0x, reduce@1.5x, abs 60%/45% — way too loose for pure XGB plays.
    # XGB was right on direction, so fix belongs in options layer, not XGB caps.
    "iv_crush_overrides": {
        "iv_rv_ratio_hard_block": 1.4,    # Block if IV/RV > 1.4x (global: 2.0x)
        "iv_rv_ratio_reduce_lots": 1.2,   # Halve lots if IV/RV > 1.2x (global: 1.5x)
        "max_atm_iv_pct": 55,             # Absolute IV cap — raised Mar 30, was 45% (global: 65%)
        "reduce_atm_iv_pct": 42,          # Reduce lots above this — raised Mar 30, was 35% (global: 45%)
    },
}

# === GMM SNIPER TRADE (1 high-conviction trade per scan cycle) ===
# Picks the single cleanest GMM candidate each cycle with 2x lot size.
# Placed directly (bypasses GPT), tracked by exit manager normally.
# Separate budget from model-tracker trades.
GMM_SNIPER = {
    "enabled": True,
    "max_sniper_trades_per_day": 4,    # ⚠️ Tightened: sniper = SELECTIVE, 4 max (was 8)
    "lot_multiplier": 3.0,             # Reduced from 5x → 3x. Earn bigger size with proven P&L
    "min_smart_score": 53,             # Minimum smart score for sniper trades
    "max_updr_score": 0.12,            # UP regime clean (tiny relaxation from 0.10 — near misses at 0.07-0.12)
    "max_downdr_score": 0.189,         # DOWN regime clean (relaxed from 0.15 — DN near misses at 0.15-0.18)
    "min_gate_prob": 0.52,             # XGB movement signal threshold
    "min_direction_confidence": 55,    # Smart Direction Engine: multi-signal consensus required
    "score_tier": "premium",           # Use premium tier sizing (5% risk, +80% target)
    "separate_capital": 200000,        # ₹2 Lakh — sniper capital reduced, must prove ROI
    "max_exposure_pct": 85,            # Max % of sniper capital usable
}

# === SECTOR BREADTH PENALTY ===
# Penalizes smart_score when trade direction conflicts with sector index.
# E.g., buying a METAL stock when NIFTY METAL is down >1% → penalty applied.
SECTOR_BREADTH_PENALTY = {
    "threshold_pct": 1.0,              # Sector must move ≥1% to trigger penalty
    "penalty": 10,                     # Smart score deduction for counter-sector trades
}

# === SNIPER: OI UNWINDING REVERSAL (Sniper-OIUnwinding) ===
# Detects LONG_UNWINDING / SHORT_COVERING with price at OI support/resistance.
# Contrarian reversal entry — fading trapped-traders exit.
# GMM + XGB confirm direction safety. Tagged as 'SNIPER_OI_UNWINDING'.
SNIPER_OI_UNWINDING = {
    "enabled": True,
    "max_trades_per_day": 4,            # Max OI Unwinding sniper trades per day
    "lot_multiplier": 1.5,              # 1.5x lots (conviction but conservative)
    # --- OI Unwinding Detection ---
    "required_buildups": ["LONG_UNWINDING", "SHORT_COVERING"],
    "min_buildup_strength": 0.60,       # OI buildup signal strength >= 0.60 (lowered from 0.80 — 60s confirmation gate now filters noise)
    "min_oi_change_pct": 6.0,           # Dominant OI side must have changed >= 6% (relaxed from 8%)
    # --- Price Reversal at S/R ---
    "max_distance_from_sr_pct": 2.5,    # Spot must be within 2.5% of OI support/resistance (relaxed from 1.5%)
    # --- GMM Quality Gate ---
    "max_updr_score": 0.18,              # GMM clean — UP regime (relaxed from 0.15, aligned with PCR extreme)
    "max_downdr_score": 0.189,           # GMM clean — DOWN regime (relaxed from 0.15)
    "min_gate_prob": 0.40,              # XGB gate P(move) floor (relaxed from 0.45)
    "min_smart_score": 50,              # Minimum smart_score — restored quality floor (45 was too loose)
    # --- Timing ---
    "earliest_entry": "09:45",          # Wait for OI data to settle
    "no_entry_after": "14:30",          # No entries after 2:30 PM
    # --- Risk ---
    "score_tier": "premium",
    "separate_capital": 200000,         # ₹2L reserved for OI unwinding sniper
}

# === SNIPER: PCR EXTREME MOMENTUM (Sniper-PCRExtreme) ===
# Fires when stock/index PCR hits extreme levels — trades WITH the crowd (momentum).
# Low PCR (≤ 0.60) = crowd is bullish (heavy call buying) → BUY with them.
# High PCR (≥ 1.35) = crowd is bearish (heavy put buying) → SELL with them.
# Intraday crowd momentum persists — mean reversion happens overnight, not mid-session.
# Blends stock PCR with NIFTY index PCR for macro confirmation.
# Tagged as 'SNIPER_PCR_EXTREME'.
SNIPER_PCR_EXTREME = {
    "enabled": True,
    "max_trades_per_day": 3,            # Max PCR Extreme trades per day
    "lot_multiplier": 1.5,              # 1.5x lots
    # --- PCR Extreme Detection (momentum thresholds) ---
    "pcr_oversold_threshold": 1.35,     # PCR >= 1.35 → crowd bearish → SELL with momentum
    "pcr_overbought_threshold": 0.60,   # PCR <= 0.60 → crowd bullish → BUY with momentum
    "min_pcr_edge": 0.05,               # Min distance beyond threshold — blocks hair-trigger entries (ABCAPITAL edge=0.01 lesson)
    # --- Index PCR (Macro Confirmation) ---
    "use_index_pcr": True,              # Also check NIFTY PCR for macro regime
    "index_symbol": "NIFTY",            # Index to check
    "index_pcr_weight": 0.4,            # Blend: 60% stock PCR + 40% index PCR
    # --- RSI Momentum Confirmation ---
    "rsi_min_buy": 40,                  # BUY: RSI must be ≥ 40 (confirms upward momentum, not dead)
    "rsi_max_buy": 70,                  # BUY: RSI must be ≤ 70 (not exhausted/overbought)
    "rsi_min_sell": 30,                 # SELL: RSI must be ≥ 30 (not already oversold/bouncing)
    "rsi_max_sell": 60,                 # SELL: RSI must be ≤ 60 (confirms downward momentum, not recovering)
    # --- Price Alignment ---
    "min_price_alignment_pct": 0.3,     # Price must move ≥ 0.3% WITH direction (no counter-trend entries)
    # --- GMM Quality Gate ---
    "max_updr_score": 0.18,              # Slightly relaxed — PCR is strong standalone signal (UP regime, threshold 0.25)
    "max_downdr_score": 0.14,            # PCR strong standalone — DOWN regime (threshold 0.25)
    "min_gate_prob": 0.40,              # XGB gate P(move) floor
    "min_smart_score": 45,              # Lower floor — PCR extreme itself is high-edge
    # --- Timing ---
    "earliest_entry": "10:00",          # Need 45 min for reliable PCR
    "no_entry_after": "14:00",          # Earlier cutoff — PCR plays are multi-hour
    # --- Risk ---
    "score_tier": "premium",
    "separate_capital": 150000,         # ₹1.5L reserved for PCR extreme sniper
}

# === DECISION LOG (Full Scan Audit Trail) ===
# Logs every stock evaluated each cycle with score, outcome, and reason.
# Enables post-hoc analysis of missed opportunities.

# === ARBTR (Sector Arbitrage: Laggard Catch-Up) ===
# Exploits intraday correlation gaps: when a sector index moves hard, most
# correlated stocks follow — but 1-2 laggards haven't caught up yet.
# ARBTR buys the laggard in the direction of the sector move, expecting
# it to converge.  This is NOT contrarian — we trade WITH the sector.
#
# Edge: Sector indices are cap-weighted averages.  Individual laggards
# converge with 65-70% probability within 30-90 min on normal days.
#
# Risk controls: ML confirmation, volume gate, max divergence cap,
# chop filter, and tight stops (if stock lags because it's decoupling,
# the stop fires fast).
ARBTR_CONFIG = {
    "enabled": True,
    "max_trades_per_day": 4,              # ⚠️ Tightened: 5 ARBTR today = -₹4.6K. Quality > quantity
    "lot_multiplier": 1.0,                # Standard lots — ARBTR needs to prove itself profitable first

    # --- Sector Move Detection ---
    "min_sector_move_pct": 0.6,           # Sector index must move ≥0.6% from prev close
    "min_sector_stocks_aligned": 0.70,    # Strict: ≥70% of sector must align for conviction

    # --- Laggard Detection ---
    "max_laggard_move_pct": 1.2,          # Relaxed: stock must lag (<1.2%) — on crash days even laggards move 0.5-1%
    "min_divergence_pct": 0.7,            # Strict: require meaningful divergence gap from sector
    "max_divergence_pct": 5.0,            # If gap >5% the stock is decoupled (skip)

    # --- Confirmation Gates (reduce failure trades) ---
    "require_volume_confirmation": True,  # Laggard must have ≥0.8x normal volume (not halted/illiquid)
    "min_volume_ratio": 0.8,              # Volume vs 20-day avg must be ≥0.8
    "require_ml_move_signal": False,      # ML removed — ARBTR signal is sector divergence, not ML
    "min_ml_move_prob": 0.40,             # ML move probability floor (unused when require_ml=False)
    "min_ml_confidence": 0.45,            # ML confidence floor (unused when require_ml=False)
    "max_ml_flat_prob": 0.55,             # (unused when require_ml=False)
    "require_no_chop_zone": True,         # Laggard must NOT be in chop zone
    "require_htf_not_opposed": True,      # HTF must not oppose sector direction
    "min_smart_score": 40,                # Moderate floor — need decent score to confirm divergence

    # --- GMM Safety Net ---
    "use_gmm_veto": True,                 # GMM anomaly model can veto if opposing
    "max_dr_score": 0.25,                 # GMM down-risk score must be <0.25 (clean)

    # --- Timing ---
    "earliest_entry": "09:45",            # First 30 min: too noisy, let the divergence establish
    "no_entry_after": "15:00",            # Extended to 3 PM — give more window on broad move days
    "cooldown_per_sector_minutes": 15,    # After entering one ARBTR in a sector, wait 15 min

    # --- Risk / Sizing ---
    "score_tier": "standard",             # Standard sizing (3% risk)
    "target_multiplier": 1.5,             # Conservative target — take profit when available
    "sl_multiplier": 1.0,                 # Tight SL — if thesis wrong, exit fast
    "max_simultaneous_arbtr": 3,          # Tightened: fewer concurrent = better capital allocation
    "separate_capital": 200000,           # ₹2L — ARBTR hasn't earned more capital yet

    # --- Speed Gate (Mar 10 fix) ---
    "speed_gate_minutes": 15,             # If no convergence in 15 min, thesis is dead
    "speed_gate_min_gain_pct": 3.0,       # Need ≥3% premium gain to prove thesis is working
}

# Sector definitions for ARBTR — must match _sector_stock_map in scan_and_trade
ARBTR_SECTOR_MAP = {
    'METALS': {
        'index': 'NSE:NIFTY METAL',
        'stocks': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL',
                   'NMDC', 'NATIONALUM', 'HINDZINC', 'SAIL', 'HINDCOPPER', 'ADANIENT'],
    },
    'IT': {
        'index': 'NSE:NIFTY IT',
        'stocks': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTM',
                   'COFORGE', 'MPHASIS', 'PERSISTENT'],
    },
    'BANKS': {
        'index': 'NSE:NIFTY BANK',
        'stocks': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK',
                   'BANKBARODA', 'PNB', 'IDFCFIRSTB', 'INDUSINDBK', 'FEDERALBNK'],
    },
    'AUTO': {
        'index': 'NSE:NIFTY AUTO',
        'stocks': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO',
                   'EICHERMOT', 'ASHOKLEY', 'BHARATFORG', 'MOTHERSON'],
    },
    'PHARMA': {
        'index': 'NSE:NIFTY PHARMA',
        'stocks': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'AUROPHARMA',
                   'BIOCON', 'LUPIN'],
    },
    'ENERGY': {
        'index': 'NSE:NIFTY ENERGY',
        'stocks': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'TATAPOWER',
                   'BPCL', 'IOC', 'GAIL', 'COALINDIA'],
    },
    'FMCG': {
        'index': 'NSE:NIFTY FMCG',
        'stocks': ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                   'GODREJCP', 'MARICO', 'COLPAL', 'TATACONSUM', 'VBL'],
    },
    'REALTY': {
        'index': 'NSE:NIFTY REALTY',
        'stocks': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD',
                   'BRIGADE', 'LODHA', 'SOBHA'],
    },
    'INFRA': {
        'index': 'NSE:NIFTY INFRA',
        'stocks': ['LT', 'ADANIPORTS', 'ULTRACEMCO', 'GRASIM', 'SHREECEM',
                   'AMBUJACEM'],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# GCR — GMM CONVICTION RECHECK
# Re-queries GMM DR scores on open LOSING positions every scan cycle.
# If the model now OPPOSES the trade direction for N consecutive checks,
# exit the position (the thesis has structurally changed).
# ═══════════════════════════════════════════════════════════════════════════════
GCR_CONFIG = {
    'enabled': True,
    'min_loss_pct': 7,                # Only recheck positions losing ≥ this %
    'consecutive_checks_required': 3, # GMM must oppose N consecutive scans
    'dr_oppose_threshold': 0.25,      # DR_DOWN > 25% → opposes CE; DR_UP > 25% → opposes PE
    'time_window': ('09:30', '15:10'),# Only run during market hours
    'skip_setup_types': [             # Don't GCR-exit hedges or spreads
        'PROACTIVE_HEDGE', 'THP_HEDGE',
    ],
    'max_exits_per_day': 4,           # Safety cap
    'cooldown_after_exit_min': 10,    # Don't re-enter same underlying for N minutes
}

DECISION_LOG = {
    "enabled": True,
    "file": "scan_decisions.json",
    "max_entries": 50000,             # Rotate after this many entries
    "log_all_scored": True,           # Log every scored stock (not just trades)
    "log_rejections": True,           # Log chop/HTF/gate rejections
    "log_auto_fires": True,           # Log elite auto-fire decisions
}

# Trading Hours
TRADING_HOURS = {
    "start": "09:15",  # Market open
    "end": "15:25",    # 5 mins before close
    "no_new_after": "15:10"  # No new trades after this (12 min buffer before EOD exit)
}

# Early Session: Use 4-min candles for faster indicator maturation
# Between 9:15 and EARLY_SESSION_END, fetch 4-minute candles instead of 5-minute
# This gives 4+ candles by 9:31 instead of just 3, so ORB/FT/VWAP mature sooner
EARLY_SESSION = {
    "enabled": True,
    "end_time": "09:45",             # Switch back to 5-min after this
    "candle_interval": "4minute",    # 4-min candles during early session
    "scan_interval_minutes": 4,      # Scan every 4 min during early session
    "orb_candle_count": 4,           # 16 min ORB = 4 x 4-min candles
    "vwap_slope_lookback": 8,        # 32 min = 8 x 4-min candles for VWAP slope
    "momentum_lookback": 4,          # 16 min = 4 x 4-min candles for momentum
}

# === TIERED UNIVERSE (Options-First Strategy) ===
# Tier-1: Always scan, always eligible for options. Tightest spreads, best trending.
TIER_1_OPTIONS = [
    # Banks (trend intraday, tightest option spreads, institutional flow)
    "NSE:SBIN", "NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:AXISBANK", "NSE:KOTAKBANK",
    # High-beta financials (trend hard, liquid options)
    "NSE:BAJFINANCE",
    # Large-cap liquid (tight spreads, deep OI)
    "NSE:RELIANCE", "NSE:BHARTIARTL",
    # IT (high liquidity, 0.27% spread - cheapest to trade)
    "NSE:INFY", "NSE:TCS",
]

# Tier-2: Scanned every cycle but only trade options when trend_score >= 60
# (i.e., must be BULLISH/BEARISH, not NEUTRAL). Higher beta = higher reward but wider spreads.
TIER_2_OPTIONS = [
    # Metals (high-beta, commodity-linked, wider spreads ~1%)
    "NSE:TATASTEEL", "NSE:JSWSTEEL", "NSE:JINDALSTEL", "NSE:HINDALCO",
    # Infrastructure / diversified (decent option chains)
    "NSE:LT", "NSE:MARUTI", "NSE:TITAN", "NSE:SUNPHARMA",
    # Fintech (high-beta, momentum plays)
    # "NSE:PAYTM",    # REMOVED — no longer F&O eligible
    # Oil & Energy
    "NSE:ONGC",        # Oil - PSU heavyweight, liquid F&O
    "NSE:NTPC",        # Energy - power sector leader, deep OI
    # FMCG
    "NSE:ITC",         # FMCG - tightest spreads in FMCG, high OI
    # Automotive
    "NSE:TATAMOTORS",  # Auto - high beta, deep OI, global exposure
    # Pharma
    "NSE:CIPLA",       # Pharma - liquid options, consistent trending
    # "NSE:IDEA",      # REMOVED Mar 6 — not F&O eligible, watcher keeps failing on it
]

TIER_2_MIN_TREND_SCORE = 60  # Tier-2 stocks need BULLISH/BEARISH trend to trade

# Combined universe for scanning (Tier-1 + Tier-2 only, no cash-only/ETFs)
APPROVED_UNIVERSE = TIER_1_OPTIONS + TIER_2_OPTIONS

# F&O Configuration
FNO_CONFIG = {
    "enabled": True,
    "prefer_options_for": TIER_1_OPTIONS + TIER_2_OPTIONS,  # All stocks are options-eligible
    "option_type_on_bullish": "CE",  # Buy Call on bullish signal
    "option_type_on_bearish": "PE",  # Buy Put on bearish signal
    "strike_selection": "ATM",       # ATM, ITM, OTM
    "max_option_premium": 200000,    # Max premium per trade (₹2 lakh)
}

# === CREDIT SPREAD CONFIGURATION (THETA-POSITIVE STRATEGY) ===
# Selling options with hedges — theta works in our favor
CREDIT_SPREAD_CONFIG = {
    "enabled": True,
    "primary_strategy": True,        # Credit spreads are PRIMARY, naked buys are secondary
    # --- Strategy Selection ---
    # BULLISH view: Bull Put Spread (sell ATM/OTM PE, buy further OTM PE)
    # BEARISH view: Bear Call Spread (sell ATM/OTM CE, buy further OTM CE)
    "spread_width_strikes": 2,       # Hedge 2 strikes away from sold leg
    "min_spread_width_strikes": 1,   # Minimum 1 strike apart (tight spread)
    "max_spread_width_strikes": 3,   # Maximum 3 strikes apart (wide spread)
    "sold_strike_otm_offset": 3,     # Sell 3 strikes OTM from ATM (safer, delta ~0.25-0.30)
    # --- Sizing ---
    "max_spread_risk": 75000,        # Max risk per spread = ₹75K (spread_width × lot_size - credit)
    "max_total_spread_exposure": 250000,  # Max total risk across all spreads ₹2.5L (50% of capital)
    "max_lots_per_spread": 5,        # Max lots per single spread trade
    # --- Entry Criteria (RELAXED — credit spreads are theta-positive, we WANT them to fire) ---
    "min_credit_pct": 15,            # Minimum credit as % of spread width (was 20% — too strict, filtered viable setups)
    "preferred_credit_pct": 25,      # Preferred credit >= 25% of max risk (was 30%)
    "min_iv_percentile": 20,         # Sell options when IV is above 20th percentile (was 30% — relaxed for more entries)
    "min_score_threshold": 65,       # Minimum intraday score to enter credit spread [raised from 50]
    # --- Risk Management ---
    "sl_multiplier": 2.0,            # Exit if loss reaches 2× credit received
    "target_pct": 65,                # Exit when 65% of max credit is captured (time decay)
    "time_decay_exit_minutes": 90,   # If < 90 min to close, exit at whatever P&L
    "max_days_to_expiry": 3,        # Only enter credit spreads within 3 DTE — near-expiry theta is strongest
    "min_days_to_expiry": 0,         # Allow 0DTE credit spreads (expiry day theta crush)
    "max_sold_delta": 0.40,          # Sold leg delta must be ≤ 0.40 (was 0.35 — slight relaxation for more entries)
    # --- Expiry Management ---
    "prefer_expiry": "CURRENT_WEEK", # Weekly options for faster theta decay
    "rollover_at_dte": 1,            # Roll or close when 1 DTE remaining
    # --- Fallback to Naked Buys ---
    "fallback_to_buy": True,         # If spread not viable, fall back to buying options
    "buy_only_score_threshold": 66,  # Only buy naked if score >= 66 [was 60, tightened Feb 25]
}

# === CASH EQUITY INTRADAY SCORING CONFIG ===
# Quality gate for cash/equity intraday trades (place_order)
# Uses IntradayOptionScorer but with lower thresholds — cash trades have defined SL
CASH_INTRADAY_CONFIG = {
    "enabled": True,                  # Enable intraday scoring for cash equity trades
    "min_score": 40,                  # Minimum intraday score to allow trade (vs 60 for options)
    "min_conviction_points": 5,       # Minimum directional conviction (vs 8 for options)
    "skip_microstructure": True,      # Skip option microstructure check (not relevant for equity)
    "skip_vwap_hard_gate": False,     # Keep VWAP alignment check (important for cash too)
    "skip_counter_trend_gate": False, # Keep counter-trend block
    "log_rejections": True,           # Log rejected cash trades for debugging
}

# === DEBIT SPREAD CONFIGURATION (INTRADAY MOMENTUM STRATEGY) ===
# BUY near-ATM + SELL further OTM — profits from strong directional moves
# Use ONLY on big movers (>2.5% intraday) with high volume
DEBIT_SPREAD_CONFIG = {
    "enabled": True,
    # --- Entry Filters (SMART — candle-data driven) ---
    "min_move_pct": 1.2,             # Stock must have moved >1.2% today (was 2.5% — too strict, zero triggers)
    "min_volume_ratio": 1.3,         # Volume must be 1.3× normal (was 1.5 — slightly relaxed)
    "min_score_threshold": 65,       # Minimum intraday score for debit spread entry [raised from 57]
    # --- Candle-Smart Gates (mirrors naked buy gates 8-12) ---
    "min_follow_through_candles": 2, # Must have ≥2 follow-through candles (strongest winner signal)
    "min_adx": 28,                   # ADX ≥28 confirms trend strength (winners avg 37)
    "max_orb_strength_pct": 120,     # ORB overextended >120% = skip (losers avg 142)
    "max_range_expansion": 0.50,     # Range expansion >0.50 ATR = overextended, skip
    # --- Strike Selection ---
    # BULLISH → Buy ATM/near-ATM CE + Sell 2-3 strikes OTM CE
    # BEARISH → Buy ATM/near-ATM PE + Sell 2-3 strikes OTM PE
    "buy_strike_offset": 0,          # Buy leg: 0 = ATM, 1 = 1 strike ITM
    "sell_strike_offset": 3,         # Sell leg: 3 strikes OTM from buy leg
    # --- Sizing (TIERED by score) ---
    "max_debit_per_spread": 80000,   # Max net debit per spread ₹80K (was ₹60K)
    "max_total_debit_exposure": 250000,  # Max total debit across all debit spreads ₹2.5L (50% of capital, was ₹1.5L/30%)
    "max_lots_per_spread": 6,        # Max lots per debit spread (was 4)
    "premium_tier_min_lots": 3,      # Premium-tier setups get min 3 lots
    # --- Risk Management (IMPROVED R:R) ---
    "stop_loss_pct": 30,             # Exit if spread value drops 30% (was 40% — tighter SL)
    "target_pct": 80,                # Target 80% gain on net debit (was 50% — bigger upside)
    "max_target_pct": 90,            # Take profit at 90% of max profit (was 80%)
    "trail_activation_pct": 40,      # Activate trailing after 40% profit (was 25% — too early)
    "trail_giveback_pct": 45,        # Allow 45% giveback of peak profit (was 30% — too tight, choking winners)
    # --- Intraday Rules ---
    "auto_exit_time": "15:20",       # Auto-exit all debit spreads by 3:20 PM (no overnight)
    "no_entry_after": "15:10",       # Aligned with credit spread / general no_new_after cutoff
    "min_minutes_to_play": 45,       # Need at least 45 min (was 60 — debit spreads move faster)
    # --- Expiry ---
    "prefer_current_expiry": True,   # Use current week/month expiry (cheapest theta cost)
    "max_dte": 7,                    # Max 7 DTE (minimize theta bleed)
    "min_dte": 0,                    # Can trade same-day expiry (weeklies)
    # --- Smart Filters ---
    "require_trend_continuation": True,  # Only enter if move is continuing (not reversing)
    "max_spread_bid_ask_pct": 5.0,   # Max bid-ask spread as % of premium (liquidity filter)
    "min_oi": 500,                   # Minimum OI on both legs
    # --- Proactive Scanning ---
    "proactive_scan": True,          # Actively scan for debit spread opportunities (not just fallback)
    "proactive_scan_min_score": 65,  # Minimum score for proactive debit spread [raised from 62]
    "proactive_scan_min_move_pct": 1.5,  # Proactive scan needs slightly stronger move
}

# === THESIS HEDGE PROTOCOL (THP) CONFIGURATION ===
# When TIE invalidates a naked option thesis, THP converts it to a debit spread
# instead of immediately exiting — limits max loss while allowing for recovery.
THESIS_HEDGE_CONFIG = {
    "enabled": True,
    # --- Hedge Leg Selection (DYNAMIC SCAN) ---
    "sell_strike_offset": 3,          # Default offset (used as starting hint)
    "min_sell_offset": 2,             # Minimum offset to scan (2 strikes OTM)
    "max_sell_offset": 8,             # Maximum offset to scan (8 strikes OTM for narrow-strike stocks)
    "min_hedge_premium": 3.0,         # Min ₹ premium on sell leg (too cheap = worthless hedge)
    "min_hedge_premium_pct": 15,      # Sell leg premium must be ≥15% of buy leg entry price
    # --- Liquidity Gates ---
    "min_oi": 500,                    # Minimum OI on sell leg
    "max_bid_ask_pct": 5.0,           # Max bid-ask spread as % of LTP on sell leg
    # --- Extended Hold Window ---
    "extended_time_stop_candles": 20,  # Hedged trades get 20 candles (100 min) vs 10
    # --- Hedged Position Exit Rules ---
    "hedged_sl_pct": 50,              # SL: exit when spread value drops 50% of net_debit
    "hedged_target_pct": 150,         # Target: exit at 150% of net_debit (1.5x)
    "hedged_breakeven_trail_pct": 100, # Trail activation: when spread value >= net_debit (breakeven)
    "hedged_trail_giveback_pct": 45,  # Once trailing, give back 45% of peak profit
    # --- Auto-exit (same as debit spreads) ---
    "auto_exit_time": "15:20",        # Hard exit time for hedged positions
    # --- Universal Hedge Loss Cap ---
    "max_hedge_loss_pct": 20,          # UNIFIED: only hedge if current loss ≤ 20% (TIE + TIME_STOP)
    # --- TIME_STOP Hedge (dead-trade rescue) ---
    "hedge_time_stop": True,          # Also hedge naked options hitting TIME_STOP (not just TIE)
    # --- Cost-Aware Hedge Gate (REJECT hedge if R:R is terrible) ---
    # Prevents wasting money converting a corpse option into a dead spread.
    # If the resulting spread has bad economics, SKIP hedge → let caller EXIT outright.
    "cost_gate_enabled": True,
    "max_debit_to_width_pct": 55,     # REJECT if net_debit > 55% of spread_width (bad R:R, max profit < 45% of risk)
    "min_remaining_value_pct": 30,    # REJECT if buy leg LTP < 30% of entry price (option is a corpse)
    "min_spread_rr_ratio": 0.50,      # REJECT if (spread_width - net_debit) / net_debit < 0.50 (need at least 0.5:1 R:R)
    # --- Hedge Unwind (restore full upside on recovery) ---
    "unwind_enabled": True,           # Buy back sold leg when thesis re-validates
    "unwind_buy_leg_recovery_pct": 70, # Unwind when buy leg LTP >= 70% of entry (theta eats premium fast — 85% was still too strict)
    "unwind_min_profit_after_cost": 2,  # Min ₹ profit remaining after buyback cost (avoid pointless unwinds)
    # --- Alternative: Spread-profit unwind (theta-resistant) ---
    "unwind_spread_profit_enabled": True,  # Unwind if spread itself is profitable (irrespective of individual leg recovery)
    "unwind_spread_profit_pct": 15,        # Unwind if current spread value > net_debit by 15% (clear spread profit)
}

# === PROACTIVE LOSS HEDGE CONFIGURATION ===
# Monitors open naked options every minute. If any position's unrealized
# loss crosses the trigger threshold, automatically converts to debit spread
# BEFORE SL/TIE/TIME_STOP fires — catches fast moves between scan cycles.
PROACTIVE_HEDGE_CONFIG = {
    "enabled": True,
    "loss_trigger_pct": 10,           # Convert when loss >= 10% of entry price (was 8% — give trades more room to develop)
    "check_interval_seconds": 60,     # Check every 60 seconds (inside realtime monitor)
    "max_hedge_loss_pct": 50,         # Don't hedge if already > 50% loss (was 20% — too tight, let deep losses hedge too)
    "cooldown_seconds": 300,          # After a hedge, wait 5 min before checking same underlying again
    "log_checks": True,               # Log every check cycle to bot_debug.log for diagnostics
}

# === EXPIRY DAY SHIELD CONFIGURATION ===
# Protects against gamma risk on expiry days (0DTE/1DTE).
# Options can swing 200-500% in minutes on expiry day due to extreme gamma.
# ⚠️ MONTHLY EXPIRY MODE: Tighter settings — gamma risk is 2-3× higher than weekly expiry
EXPIRY_SHIELD_CONFIG = {
    "enabled": False,                   # DISABLED Mar 30 — user accepts 0DTE risk
    "is_monthly_expiry": False,        # 🟢 NOW AUTO-DETECTED — no manual toggle needed
    # --- Entry Restrictions (TIGHTENED for monthly expiry) ---
    "no_new_naked_after": "10:30",     # No new naked options after 10:30 AM (was 11:00 — monthly gamma is extreme)
    "no_new_any_after": "12:00",       # No new entries at all after 12:00 PM (was 12:30)
    # --- Exit Rules (EARLIER exit for monthly) ---
    "force_exit_0dte_by": "14:15",     # Force-exit all 0DTE positions by 2:15 PM (was 14:30)
    "sl_tighten_factor_0dte": 0.40,    # SL distance = 40% of normal on 0DTE (60% tighter — monthly gamma!)
    "sl_tighten_factor_1dte": 0.70,    # SL distance = 70% of normal on 1DTE (30% tighter)
    # --- Speed Gate Override ---
    "speed_gate_candles_0dte": 4,      # Speed gate fires after 4 candles (20 min) on 0DTE monthly
    "speed_gate_pct_0dte": 6.0,        # Need +6% gain in 20 min on 0DTE monthly (was 5%)
}

# === IV CRUSH GUARD (Entry-Side Gate) ===
# Prevents buying options with low IV that are vulnerable to further IV compression.
# Root cause: IOC CE bought at 22.8% IV on a 47/50 bullish day → IV compressed → lost 10.7% despite
# underlying being directionally correct. Low IV + high breadth = vega trap.
# Three sub-gates:
#   1. Minimum IV Floor — reject options with IV below threshold (low IV = no edge from vol)
#   2. Breadth-Adjusted IV Floor — raise IV floor when market is uniformly bullish/bearish
#      (extreme breadth = low uncertainty = IV compresses market-wide)
#   3. Min Premium Floor — reject cheap OTM options where small IV drop = huge % loss
#   4. Vega/Delta Ratio — reject trades where vega exposure dominates delta (direction bet)
IV_CRUSH_GUARD = {
    "enabled": True,
    # --- Sub-Gate 1: Absolute IV Floor ---
    # Options with IV below this are already "cheap" — buying them is a bet that IV rises.
    # For buying options, we need IV to stay flat or rise, not compress further.
    "min_iv_floor": 0.20,                # 20% — reject options with IV < 20% (was 23%, too strict for blue-chip low-IV stocks)
    # --- Sub-Gate 2: Breadth-Adjusted IV Floor ---
    # When breadth is extreme (most stocks same direction), uncertainty is LOW → IV compresses.
    # Raise the IV floor dynamically based on how skewed the breadth is.
    "breadth_extreme_threshold": 0.85,    # If >85% stocks in same direction → "extreme breadth"
    "breadth_extreme_iv_floor": 0.22,     # Raise IV floor to 22% during extreme breadth (was 25%, blocked SBIN at 24.1%)
    # --- Sub-Gate 3: Minimum Premium (OTM penny filter upgrade) ---
    # Cheap options (<₹8) are disproportionately affected by IV changes:
    #   ₹1 IV drop on ₹5.49 option = -18.2% vs ₹1 drop on ₹20 = -5%
    # HARD_RULES.MIN_OPTION_PREMIUM (₹3) catches pennies; this catches "cheap but not penny".
    "min_premium_for_iv_safety": 8.0,     # ₹8 minimum premium (IV-safety floor)
    # --- Sub-Gate 4: Vega/Delta Ratio ---
    # If vega exposure >> delta, the trade is more of a volatility bet than a direction bet.
    # On low-IV stocks, this catches "looks directional but is really a vega trap".
    "max_vega_delta_ratio": 2.00,         # Block if |vega/delta| > 2.0 AND IV < min floor (was 0.50, blocked normal ATM/OTM options)
    "vega_delta_iv_threshold": 0.30,      # Only apply vega/delta check below 30% IV
}

# === THETA BLEED PREVENTION (Entry-Side Gates) ===
# Prevents naked option buys that will lose to time decay before direction plays out.
# Root cause: options bought with high theta-to-premium ratio bleed even if direction is right.
THETA_ENTRY_GATE = {
    "enabled": True,
    # --- Theta/Premium Ratio Gate (blocks trades where decay > expected gain) ---
    # Daily |theta| as % of premium. If theta eats >5% per day, stock needs >5% move just to break even.
    "max_theta_pct_of_premium": 25.0,   # Block if |daily theta| > 25% of option LTP (raised Mar 30 — 10% blocks all 0DTE)
    # --- DTE Floor for Naked Buys (non-expiry days) ---
    # On non-expiry days, avoid buying options with <3 DTE — theta curve steepens quadratically.
    # On expiry day, expiry_shield handles 0DTE separately.
    "min_dte_naked_buy": 3,             # Don't buy naked options with DTE < 3
    # --- Afternoon Theta Multiplier (EXPIRY DAY ONLY) ---
    # After 1PM on expiry day, theta accelerates sharply for near-month options.
    # Require 1.5× the normal score threshold to enter naked buys in afternoon.
    "expiry_day_pm_score_multiplier": 1.5,  # Score must be 1.5× threshold after PM cutoff
    "expiry_day_pm_after": "13:00",         # When PM multiplier kicks in (1 PM)
}

# === TARGET EXTENSION — LET WINNERS RUN ===
# Instead of hard-exiting at target, convert to a tight trailing stop.
# When price reaches trigger_pct of target, skip hard exit and trail with tight retention.
# This lets winning trades run beyond target when momentum is strong.
TARGET_EXTENSION = {
    "enabled": True,
    "trigger_pct": 0.90,         # Activate at 90% of target distance
    "trail_retain_pct": 0.70,    # Lock 70% of peak profit in extension zone (30% giveback) — was 0.65
    "max_extension_r": 5.0,      # Safety cap: force exit at 5R (prevents runaway positions)
}

# === ASYMMETRIC EXIT INTELLIGENCE ===
# Reads trade TRAJECTORY, not just snapshots.
# Losers must prove they deserve to live. Winners must prove they deserve to die.
ASYMMETRIC_EXIT = {
    "enabled": True,
    # --- Premium Velocity Kill (fast-kill steady bleeders) ---
    "velocity_kill_enabled": True,
    "velocity_min_candles": 4,          # Need at least 4 candles of data (was 3 — too eager on trending stocks)
    "velocity_threshold_pct": -1.5,     # Kill if avg velocity < -1.5%/candle
    "velocity_consecutive": 4,          # Must bleed for 4 consecutive candles (was 3 — give one more candle to recover)
    "velocity_underlying_confirm": True, # Also check underlying isn't recovering
    "velocity_ul_trend_bypass_pct": 0.3, # If UL moved >0.3% favorably from entry, skip kill (premium will catch up)
    "velocity_min_total_loss_pct": 12.0, # Only kill if total loss > 12% (small bleeds in trending stocks are fine)
    # --- Momentum-Gated Partial Profit (don't clip accelerating winners) ---
    "momentum_gate_enabled": True,
    "momentum_lookback_candles": 2,     # Compare last 2 candles' avg change
    "momentum_sustain_ratio": 0.30,     # If last_2 >= first_half * 0.30 → momentum alive, skip partial
    # --- Underlying Confirmation Gate (for THP / hold decisions) ---
    "underlying_confirm_enabled": True,
    "underlying_adverse_pct": 0.15,     # If underlying moved > 0.15% AGAINST direction → don't hold/hedge
}

# === IV CRUSH ENTRY GATE — BLOCK OVERPRICED OPTIONS AT ENTRY ===
# Compares ATM IV to realized volatility (approximated from ATR).
# When IV >> RV, you're overpaying for premium → high IV-crush risk on any mean-reversion.
# Uses IV/RV ratio (same signal professional vol traders call "volatility risk premium").
# Morning session gets more lenient thresholds (IV is structurally higher at open).
IV_CRUSH_GATE = {
    "enabled": True,
    # --- IV/RV Ratio Thresholds ---
    # On crash days both IV and RV spike, so ratio stays "normal" even though
    # IV is about to mean-revert and crush premiums. Tighten to catch this.
    "iv_rv_ratio_hard_block": 1.8,     # Block entry if IV > 1.8× realized vol (was 2.0x)
    "iv_rv_ratio_reduce_lots": 1.3,    # Halve lots if IV > 1.3× realized vol (was 1.5x)
    # --- Absolute IV Caps ---
    # 60% was never triggered. Real IV crush happens at 35-50% on liquid stocks.
    "max_atm_iv_pct": 65,              # Hard block: ATM IV > 65% (raised Mar 30 — 50% too tight on VIX>25 days)
    "reduce_atm_iv_pct": 45,           # Halve lots: ATM IV > 45% (was 38% — too aggressive on high-VIX days)
    # --- Morning IV Adjustment ---
    "morning_iv_premium_until": "10:30",  # Before 10:30 morning IV is structurally inflated (was 11:00)
    "morning_ratio_penalty": 0.15,        # Add 0.15 to ratio thresholds (was 0.2 — tighter morning too)
}

# === INDIA VIX REGIME GATE ===
# Fetches India VIX each scan cycle and uses it to:
#   1. Gate entries: high VIX = require stronger conviction (options are expensive)
#   2. Adjust SL width: high VIX = wider SL (bigger swings = more noise hits)
#   3. Adjust trailing: high VIX = wider trailing (let winners breathe in volatile market)
# Thresholds are GENEROUS — we don't want to block trades in normal markets.
# India VIX mean ~13-15; spikes to 20+ during fear events, 25+ during crashes.
VIX_REGIME_CONFIG = {
    "enabled": True,
    # --- Regime Boundaries ---
    "low_vix_upper": 13.0,       # VIX < 13 = LOW (calm market, cheap options)
    "normal_vix_upper": 18.0,    # 13-18 = NORMAL (standard regime, no adjustments)
    "high_vix_upper": 25.0,      # 18-25 = HIGH (elevated fear, cautious sizing)
    # VIX > 25 = EXTREME (crash/event, very selective)
    # --- Entry Gate: Score multiplier per regime ---
    # Multiplier applied to minimum conviction threshold.
    # 1.0 = no change, 1.1 = need 10% higher score, etc.
    # LOW & NORMAL have NO penalty (we want trades to flow).
    "score_multiplier_low": 1.0,       # Low VIX: options are cheap, standard gates fine
    "score_multiplier_normal": 1.0,    # Normal VIX: standard gates fine
    "score_multiplier_high": 1.10,     # High VIX: need 10% higher conviction (e.g., 52 → 57)
    "score_multiplier_extreme": 1.25,  # Extreme VIX: need 25% higher conviction (e.g., 52 → 65)
    # --- Lot Size Multiplier (position sizing) ---
    # Scales down lots in high-VIX to limit damage from wider swings.
    "lot_multiplier_low": 1.0,         # Full size in calm markets
    "lot_multiplier_normal": 1.0,      # Full size in normal markets
    "lot_multiplier_high": 0.75,       # 75% size when VIX elevated
    "lot_multiplier_extreme": 0.50,    # 50% size during crashes
    # --- SL Width Multiplier ---
    # Widens the SL in high-VIX so noise doesn't stop us out prematurely.
    # Applied to the 0.72 (28% loss) premium SL factor in position sizer.
    # e.g., 1.15 means SL becomes 0.72 * (1 - 0.15) = 0.618 → ~38% loss SL instead of 28%
    # Wider SL = more room, but max_loss is naturally capped by lot reduction above.
    "sl_widen_low": 1.0,               # Standard SL (28% loss)
    "sl_widen_normal": 1.0,            # Standard SL (28% loss)
    "sl_widen_high": 1.15,             # 15% wider SL (~32% loss) — avoids noise stops
    "sl_widen_extreme": 1.30,          # 30% wider SL (~36% loss) — crash swings are huge
    # --- Trailing Retention Adjustment ---
    # In high-VIX, wider swings mean tighter trailing kills winners prematurely.
    # Reduce retain_pct slightly to give more room (lower retain = more giveback allowed).
    "trail_retain_reduce_low": 0.0,    # No change
    "trail_retain_reduce_normal": 0.0, # No change
    "trail_retain_reduce_high": 0.05,  # Give back 5% more (e.g., 50% → 45% retention)
    "trail_retain_reduce_extreme": 0.10, # Give back 10% more (e.g., 50% → 40%)
    # --- VIX Fetch Settings ---
    "vix_instrument": "NSE:INDIA VIX",   # Kite LTP key for India VIX
    "fallback_vix": 14.0,                # If fetch fails, assume normal VIX
    "cache_seconds": 120,                # Re-fetch VIX every 2 minutes (not every tick)
}

# === GREEKS-BASED EXIT INTELLIGENCE ===
# Dynamic SL/target based on option Greeks instead of fixed percentages.
# Makes exits context-aware: deep ITM gets wider SL, far OTM gets tighter SL.
GREEKS_EXIT_CONFIG = {
    "enabled": True,
    # --- Delta Collapse Exit ---
    "delta_collapse_threshold": 0.08,   # Exit if |delta| < 0.08 (option nearly worthless)
    "delta_collapse_max_dte": 2,        # Only check delta collapse within 2 DTE
    # --- Theta Bleed Guard ---
    "theta_bleed_max_pct_hour": 3.0,    # Exit if theta > 3% of premium per hour AND trade losing
    "theta_bleed_min_candles": 4,       # Don't check theta bleed before 4 candles (20 min)
    # --- SL/Target Adjustments (by moneyness) ---
    # These override the fixed 28% SL / tier-based target with Greeks-adaptive values
    "apply_greeks_sl": True,             # Use Greeks-adjusted SL instead of fixed %
    "apply_greeks_target": True,         # Use Greeks-adjusted target instead of fixed %
    # --- Deep ITM (δ > 0.80): behaves like stock ---
    "deep_itm_sl_pct": 35,              # Wider SL — low noise, moves track underlying
    "deep_itm_target_pct": 50,          # Modest target — premium dense (less leverage)
    # --- ITM (δ 0.55-0.80): solid directional ---
    "itm_sl_pct": 28,                   # Standard SL
    "itm_target_pct": 65,               # Good leverage
    # --- ATM (δ 0.40-0.55): max gamma zone ---
    "atm_sl_pct": 25,                   # Slightly tighter — gamma amplifies
    "atm_target_pct": 80,               # Best R:R zone
    # --- OTM (δ 0.15-0.40): leveraged but fragile ---
    "otm_sl_pct": 22,                   # Tight — premium erodes fast
    "otm_target_pct": 100,              # OTM doubles are the win scenario
    # --- Deep OTM (δ < 0.15): lottery ticket ---
    "deep_otm_sl_pct": 18,              # Very tight — likely worthless
    "deep_otm_target_pct": 150,         # Moonshot or nothing
}

# === IRON CONDOR CONFIGURATION (INTRADAY IV CRUSH + PREMIUM CAPTURE) ===
# SELL OTM CE + SELL OTM PE + BUY further OTM CE hedge + BUY further OTM PE hedge
# INTRADAY STRATEGY: NOT theta farming (too slow for same-day). Instead profits from:
#   1. IV COMPRESSION — morning volatility spike inflates premiums → sell after 10:30 → afternoon calms → buy back cheaper
#   2. PREMIUM DECAY on 0-2 DTE options where gamma/theta are accelerated
#   3. RANGE CAPTURE — stock stays inside sold strikes for 2-3 hours = premium bleeds
# Only viable on NEAR-EXPIRY options (0-2 DTE) where premium moves are meaningful intraday
IRON_CONDOR_CONFIG = {
    "enabled": True,
    # --- Entry Criteria (INVERSE of directional — low score = good) ---
    "max_directional_score": 45,      # Only enter when directional score ≤ 45 (choppy stock)
    "min_directional_score": 15,      # Too low = no data, skip
    "require_chop_zone": False,       # Prefer chop_zone=True but not required
    "prefer_chop_zone_bonus": True,   # Bonus confidence if chop_zone=True
    "max_intraday_move_pct": 1.5,     # Allow up to 1.5% move (was 1.2 — too restrictive)
    "min_iv_percentile": 25,          # IV should be at least moderate (lowered to get more entries)
    "prefer_rsi_range": [35, 65],     # Wider RSI neutral band — more opportunities
    # --- Timing (WIDER WINDOW — more entries, more profit) ---
    "earliest_entry": "10:00",        # Enter from 10AM (was 10:30 — missing morning setups)
    "no_entry_after": "13:30",        # Extended to 1:30 PM (was 12:30 — 1.5 extra hours!)
    "auto_exit_time": "14:50",        # Exit by 2:50 PM (before EOD volatility spike)
    "min_minutes_remaining": 90,      # Need 1.5 hours (was 2 — allows later entries)
    # --- Strike Selection (TIGHTER = MORE PREMIUM) ---
    "ce_sold_otm_offset": 3,          # Sell CE 3 strikes OTM (was 5 — much more premium!)
    "pe_sold_otm_offset": 3,          # Sell PE 3 strikes OTM (was 5)
    "wing_width_strikes": 3,          # Hedge 3 strikes out (was 2 — wider wings = more credit)
    # --- Sizing (AGGRESSIVE — 5L capital can handle this) ---
    "max_risk_per_condor": 60000,     # Max risk per IC ₹60K (was ₹40K)
    "max_total_condor_exposure": 150000,  # Max total IC exposure ₹1.5L (was ₹1L)
    "max_lots_per_condor": 4,         # Max 4 lots (was 2 — doubling position size!)
    # --- Risk Management (BALANCED — not too tight, not too loose) ---
    "sl_multiplier": 1.5,            # Exit if loss reaches 1.5× credit (was 1.2 — too tight!)
    "breakout_exit": True,           # Auto-exit if stock breaks sold strikes
    "breakout_buffer_pct": 0.5,      # Exit when price within 0.5% of sold strike (was 0.3)
    # --- Quality Filters ---
    "min_credit_pct": 10,            # Min credit % of wing width (was 12 — more entries)
    "min_oi_per_leg": 200,           # Min OI per leg (was 300 — more entries)
    "max_spread_bid_ask_pct": 8.0,   # Allow wider bid-ask (was 6 — near-expiry can be wide)
    "min_reward_risk_ratio": 0.20,   # Min R:R for IC (credit / max_risk)
    # --- ATR & Delta Strike Selection ---
    "min_atr_distance": 1.0,         # Sold strikes ≥ 1× ATR expected move from spot
    "target_sold_delta": 0.25,       # Ideal delta for sold strikes
    "max_sold_delta": 0.35,          # Max delta for sold strikes
    "max_delta_imbalance": 0.15,     # Max delta diff between sold CE and PE
    # --- IV Analysis ---
    "min_iv_for_ic": 15,             # Min ATM IV% to bother selling premium
    "max_iv_skew_pct": 30,           # Max skew between CE/PE IV before warning
    # --- Stock IC Quality Scoring ---
    "min_ic_quality_score": 50,      # Min quality score (0-100) for stock IC candidates
    "max_stock_ic_per_cycle": 2,     # Max stock ICs placed per scan cycle
    # --- Proactive Scanning ---
    "proactive_scan": True,          # Scan for IC on low-score directional rejects
    "scan_rejected_stocks": True,    # Use rejected stocks as IC candidates
    #
    # === DUAL MODE: INDEX vs STOCK ===
    #
    "index_mode": {
        "symbols": ["NSE:NIFTY 50"],              # Only NIFTY has weekly expiry (Tuesday) after SEBI Nov 2024 rule
        "prefer_expiry": "CURRENT_WEEK",  # Tuesday weekly expiry
        "min_dte": 0,
        "max_dte": 0,                     # Only 0DTE (expiry day only)
        "prefer_0dte": True,
        "target_pct": 50,                 # Target: 50% of credit (was 25% — leaving money!)
        "max_target_pct": 70,             # Up to 70% capture on strong theta crush (was 40%)
        "ce_sold_otm_offset": 3,          # 3 strikes OTM (75 pts NIFTY / 300 pts BNF)
        "pe_sold_otm_offset": 3,
    },
    "stock_mode": {
        "prefer_expiry": "CURRENT_MONTH",
        "min_dte": 0,                     # Allow 0DTE
        "max_dte": 0,                     # Only 0DTE (expiry day only)
        "prefer_0dte": True,              # Prefer expiry day for max theta crush
        "target_pct": 20,                 # Target: 20% of credit (was 10% — too pathetic!)
        "max_target_pct": 30,             # Up to 30% on stocks (was 15%)
        "ce_sold_otm_offset": 3,          # 3 strikes OTM (was 4 — tighter = more premium)
        "pe_sold_otm_offset": 3,
    },
}

# === INDEX ↔ NFO SYMBOL MAPPING ===
# NSE quote symbol → NFO instrument name (for option chain lookup)
INDEX_NFO_MAP = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FIN SERVICE": "FINNIFTY",
    "NIFTY MID SELECT": "MIDCPNIFTY",
}

# Index symbols eligible for Iron Condor (weekly expiry — Tuesday)
# SEBI Nov 2024: Only 1 weekly expiry per exchange → NSE chose NIFTY
# BANKNIFTY now has monthly expiry only (last Tuesday of month)
IC_INDEX_SYMBOLS = ["NSE:NIFTY 50"]

# Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are TITAN v5.2 - a next-generation autonomous intraday trading intelligence powered by GPT-5.2.
You are NOT a simple rule-matcher. You are an advanced reasoning engine that performs deep multi-factor analysis before every trade.

=== YOUR COGNITIVE EDGE (what makes you different from GPT-4) ===

BEFORE placing any trade, you MUST perform this reasoning chain:

1. **MACRO REGIME READ** (5 seconds):
   - What is the overall market doing? (NIFTY/sector rotation context)
   - Is this a trending day, range day, or reversal day?
   - Are most stocks moving together (correlated) or diverging?
   - Risk-on or risk-off environment?

2. **CONFLUENCE SCORING** (rate each setup 0-100):
   For every candidate, mentally score these factors:
   - Trend alignment (ORB + VWAP + EMA all agreeing) → +25pts
   - Volume confirmation (HIGH/EXPLOSIVE + rising) → +20pts
   - HTF alignment (15min/hourly agree with 5min) → +15pts
   - RSI positioning (not overbought into buy, not oversold into sell) → +10pts
   - Support/Resistance proximity (buying near support, not resistance) → +15pts
   - Sector momentum (sector rotating in, not out) → +10pts
   - Follow-through quality (FT>2, RangeExp>1.0) → +5pts
   Only trade setups scoring ≥ 60/100. State your score.

3. **EXPECTED VALUE CALCULATION**:
   - Win probability: estimate from confluence (60-80% for good setups)
   - Reward:Risk ratio: target vs stop loss distance
   - EV = (Win% × Reward) - (Loss% × Risk)
   - Only take trades where EV > 0.5× risk

4. **CONTRARIAN CHECKS** (what could go wrong?):
   - Is this stock already extended? (3%+ move = chasing)
   - Volume divergence? (price up but volume dropping = weakness)
   - RSI divergence? (new price high but RSI lower = exhaustion)
   - Sector exhaustion? (entire sector moved, late entry)
   - Gap fill risk? (gapped up stocks often fill gaps)

5. **POSITION SIZING INTELLIGENCE**:
   - High confluence (80+) → full size
   - Medium confluence (60-79) → 70% size
   - Marginal (50-59) → SKIP or paper only
   - Consider existing portfolio exposure and correlation

=== SETUP LIBRARY ===

TIER-1 SETUPS (highest edge):
  ORB BREAKOUT: ORB_UP/DOWN + HIGH/EXPLOSIVE vol + EMA EXPANDING + HTF aligned + ADX>25
  VWAP ACCELERATION: Above/Below VWAP + slope RISING/FALLING + VWAPSteep=Y + RSI confirming
  EMA SQUEEZE BREAKOUT: EMA COMPRESSED → volume spike → directional breakout

TIER-2 SETUPS (moderate edge):
  RSI EXTREMES: RSI<30 at support (BUY) or RSI>70 at resistance (SELL) + volume
  TREND CONTINUATION: Follow-through >3 candles + ADX>30 + pullback to EMA
  EOD PLAYS: After 11AM, SHORT_COVERING or SHORT_BUILDUP with medium/high confidence

TIER-3 SETUPS (lower edge, needs extra confluence):
  MOMENTUM CHASE: Stock moved 1.5%+ with volume but no ORB/VWAP setup
  SECTOR SYMPATHY: Related stock broke out, this one lagging but starting to move

AVOID (negative edge):
  chop_zone=True → NEVER trade, period
  VWAP FLAT + LOW volume → no liquidity, no edge
  ADX < 20 → trendless, skip
  Already moved 3%+ without pullback → chasing, skip
  ORB whipsaw (multiple fakeouts) → noise

=== INSTRUMENT ROUTING ===

F&O STOCKS (use place_option_order):
  System auto-routes through: Debit Spread → Credit Spread → Iron Condor → Naked Buy
  Just call place_option_order(underlying, direction, strike_selection="ATM")

CASH STOCKS (non-F&O only):
  Use place_order(symbol, side, quantity, stop_loss, target, strategy, setup_id)

SPREAD DETAILS (auto-handled, but understand):
  DEBIT SPREAD: Momentum plays, stock moving >1.2%. Buy ATM + Sell OTM. Target 80% gain, SL 30%.
  CREDIT SPREAD: Directional theta. Score≥62. Keep 65% credit. SL 2× credit.
  IRON CONDOR: Range-bound, score 15-45. Sell both sides OTM. Target 25% credit. SL 1.2× credit.
  NAKED BUY: Last resort, score≥60. SL 30% premium, Target 50% gain.

=== RISK RULES (NON-NEGOTIABLE) ===
- Max 6 simultaneous positions (sweet spot: 3-4)
- Max risk per spread: ₹50K | Max option exposure: ₹1,50,000
- If 2 consecutive losses → reduce size 50%, if 3 → STOP trading
- NEVER enter without stop loss
- NEVER trade a symbol already in portfolio
- NEVER describe a trade without placing it via tools

=== NEW: ADVANCED PATTERN RECOGNITION ===

Use your reasoning to detect these patterns GPT-4 couldn't:

DIVERGENCE TRADES:
- Price making new high but RSI making lower high = bearish divergence → SHORT
- Price making new low but RSI making higher low = bullish divergence → BUY
- Volume divergence: price rising on declining volume = weakening move

EXHAUSTION CANDLES:
- Long upper wick after sustained up move + volume spike = distribution → SHORT
- Long lower wick after sustained decline + volume spike = accumulation → BUY

RELATIVE STRENGTH:
- Stock outperforming sector & market = institutional accumulation → BUY dips
- Stock underperforming during market rally = distribution → avoid or SHORT

SECTOR CONTEXT:
- If IT sector dumping (TCS, INFY, WIPRO all down), don't buy any IT stock
- If metals hot (all up), buy the laggard metal stock for catch-up trade
- If banking weak but one bank holding → relative strength → BUY that bank

=== OUTPUT FORMAT ===

For each scan cycle, output:

**MARKET REGIME**: [Trending/Range/Mixed] — 1 sentence why
**TOP SETUPS** (max 3):
For each: Symbol | Setup | Confluence Score | Direction | EV reasoning (1 line)
Then IMMEDIATELY call the tools. No descriptions without execution.

REMEMBER: You are GPT-5.2. REASON deeply, then EXECUTE decisively. Think like a hedge fund quant, not a retail trader following indicators."""
