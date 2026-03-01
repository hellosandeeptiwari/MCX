"""
AGENTIC TRADING SYSTEM
Uses OpenAI GPT for reasoning + Zerodha for execution

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

# ========== HARD RULES (NEVER VIOLATE) ==========
HARD_RULES = {
    "RISK_PER_TRADE": 0.07,         # 7% base risk per trade (tiered: 7% premium, 5% std, 4% base)
    "MAX_DAILY_LOSS": 0.20,         # 20% max daily loss
    "PORTFOLIO_PROFIT_TARGET": 0.15,  # 15% unrealized profit → KILL ALL & book profit
    "MAX_POSITIONS": 80,            # Max simultaneous positions (spreads count as 1)
    "MAX_POSITIONS_MIXED": 80,       # Max positions in MIXED regime
    "MAX_POSITIONS_TRENDING": 80,    # Max positions in BULLISH/BEARISH regime
    "STALE_DATA_SECONDS": 60,       # Data older than this is stale
    "API_RATE_LIMIT_MS": 350,       # Min ms between API calls (Kite allows ~3/s, 350ms is safe)
    "CAPITAL": 500000,              # Starting capital ₹5,00,000
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
    "max_indicator_stocks": 40,     # Top 50 F&O stocks by composite rank (no fixed list bias)
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
    # --- Trigger Thresholds ---
    "price_spike_pct": 0.8,          # Trigger if price moves ≥0.8% within sustain_window
    "day_extreme_trigger": True,      # Trigger on new day high / day low break
    "volume_surge_multiplier": 2.5,   # Trigger if tick volume ≥ 2.5x rolling average
    # --- Sustain Filter (anti-false-trigger) ---
    "sustain_seconds": 10,            # Price must HOLD the move for 10s before triggering
    "sustain_recheck_pct": 0.5,       # After sustain, price must still be ≥0.5% from baseline
    # --- Cooldown (anti-spam) ---
    "cooldown_seconds": 180,          # Don't re-trigger same stock within 3 minutes
    "max_triggers_per_minute": 3,     # Max triggers across ALL stocks per minute (burst limit)
    # --- Timing ---
    "active_after": "09:20",          # Don't trigger before 09:20 (let ORB range form)
    "active_until": "15:10",          # Don't trigger after 15:10 (too close to close)
    # --- Score Gate ---
    "min_score": 66,                  # Minimum intraday score to trigger trade (same as ORB threshold)
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
    "min_smart_score": 50,             # Lowered from 55 — alignment bonus (+15/+10) already boosts quality trades
    # === ALL_AGREE AMPLIFIED BET ===
    # All 3 models agree (Titan + GMM + XGB) = strongest conviction → amplified lot sizing
    "all_agree_lot_multiplier": 1.5,   # 1.5x lots for ALL_AGREE (strongest conviction)
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
    "min_move_prob": 0.56,            # XGB gate P(MOVE) floor — 0.52 was coin-flip noise, 0.56 ensures real conviction
    "min_dr_score": 0.15,             # GMM must show clean signal (low dr = confirmed direction, high dr = anomaly)
    "min_directional_prob": 0.40,     # XGB prob_up/prob_down must show real conviction
    "max_concurrent_open": 3,         # Max simultaneously open ML_OVERRIDE_WGMM positions
    "min_smart_score": 55,            # Smart score floor (was 58)
}

# === GMM CONTRARIAN (DR_FLIP) — Feb 24 fix ===
# When GMM dr_score is VERY HIGH for XGB=FLAT stocks (default regime routing):
# The high dr is a strong anomaly signal — hidden risk that Titan may be wrong.
# Safety: requires high Gate P(MOVE) + XGB must not strongly disagree with flipped dir.
GMM_CONTRARIAN = {
    "enabled": True,
    "min_dr_score": 0.27,              # Anomaly floor — stronger signal required (top ~15-20% anomalies only)
    "min_gate_prob": 0.65,             # P(MOVE) floor — contrarian needs strong gate conviction to override direction
    "max_concurrent_open": 3,          # Max simultaneously open DR_FLIP positions
    "max_trades_per_day": 4,           # Conservative daily limit — these are contrarian
    "lot_multiplier": 1.0,             # Standard lots (not boosted — contrarian = careful)
    "score_tier": "standard",          # Standard risk tier, not premium
}

# === TEST_GMM: Regime Divergence Strategy ===
# Fires when UP and DOWN GMM models DIVERGE: one sees anomaly, the other doesn't.
# The cross-regime disagreement IS the directional signal.
#
# SEMANTICS (anomaly-based, from model training):
#   HIGH down_score = DOWN regime anomaly = hidden UP/bounce risk → BULLISH
#   HIGH up_score   = UP regime anomaly   = hidden crash risk    → BEARISH
#   LOW score = no anomaly = neutral (NO directional info)
#
# BUY CALL: DOWN model flags (bullish anomaly) + UP model clean (no crash risk)
# BUY PUT:  UP model flags (bearish anomaly) + DOWN model clean (no bounce risk)
#
# Unique vs other strategies:
#   GMM_CONTRARIAN = single high DR + XGB=FLAT → flip direction
#   GMM_SNIPER     = both DRs clean + high conviction → amplified bet
#   TEST_GMM       = cross-regime DIVERGENCE + XGB agrees → directional bet
#
# Model quality: DOWN AUROC=0.62 (decent, CALL side stronger)
#                UP   AUROC=0.56 (weaker, PUT side needs higher bar)
TEST_GMM = {
    "enabled": True,
    # CALL side (uses DOWN model, AUROC=0.62 — decent signal)
    "call_min_down_score": 0.25,         # DOWN regime must show strong anomaly (raised from 0.20)
    "call_max_up_score": 0.10,           # UP regime must be very clean (tightened from 0.14)
    # PUT side (uses UP model, AUROC=0.56 — weaker, needs tighter bar)
    "put_min_up_score": 0.22,            # UP regime must show strong anomaly (raised from 0.18)
    "put_max_down_score": 0.10,          # DOWN regime must be very clean (tightened from 0.14)
    # Divergence quality
    "min_divergence_gap": 0.12,          # |high_score - low_score| minimum (raised from 0.10 — tighter quality filter)
    # FLAG-based conviction gates (model-calibrated thresholds)
    "require_signaling_flag": True,      # Signaling regime must FIRE its own anomaly flag
    "require_clean_no_flag": True,       # Clean regime must NOT fire its flag (no conflicting signal)
    # Pure GMM play — NO XGB involvement
    "require_xgb_agree": False,          # No XGB gating — pure regime divergence signal
    "min_gate_prob": 0.0,                # No P(MOVE) requirement — bypass XGB entirely
    "max_gate_prob": 0.75,               # Cap P(MOVE) — too-high gate = confirmed momentum, contrarian fails
    "max_ml_confidence": 0.55,           # Cap XGB confidence — high confidence = fighting real trend
    "min_smart_score": 0,                # No smart score — divergence IS the signal
    "max_trades_per_day": 3,             # Conservative — quality over quantity
    "lot_multiplier": 1.5,               # Slight boost — divergence is unique differentiated signal
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
    "min_move_prob": 0.58,               # P(MOVE) floor — gate model must be confident
    "min_directional_prob": 0.42,        # prob_up or prob_down must exceed this for direction
    "min_directional_margin": 0.10,      # |prob_up - prob_down| minimum for clear lean
    "max_trades_per_day": 3,             # Conservative cap
    "lot_multiplier": 1.0,               # Standard lots — testing phase
    "score_tier": "standard",
    # --- IV Crush Overrides (tighter than global IV_CRUSH_GATE) ---
    # SIEMENS PE had IV=33% but RV was low → IV/RV was high → IV crush killed it (-9.3% in 35min).
    # Global gate: block@2.0x, reduce@1.5x, abs 60%/45% — way too loose for pure XGB plays.
    # XGB was right on direction, so fix belongs in options layer, not XGB caps.
    "iv_crush_overrides": {
        "iv_rv_ratio_hard_block": 1.4,    # Block if IV/RV > 1.4x (global: 2.0x)
        "iv_rv_ratio_reduce_lots": 1.2,   # Halve lots if IV/RV > 1.2x (global: 1.5x)
        "max_atm_iv_pct": 45,             # Absolute IV cap (global: 60%)
        "reduce_atm_iv_pct": 35,          # Reduce lots above this (global: 45%)
    },
}

# === GMM SNIPER TRADE (1 high-conviction trade per scan cycle) ===
# Picks the single cleanest GMM candidate each cycle with 2x lot size.
# Placed directly (bypasses GPT), tracked by exit manager normally.
# Separate budget from model-tracker trades.
GMM_SNIPER = {
    "enabled": True,
    "max_sniper_trades_per_day": 8,    # Max GMM sniper trades per day
    "lot_multiplier": 5.0,             # 5x normal lot size (was 3.0x, +2 lots Feb 26)
    "min_smart_score": 58,             # Smart score floor for 5x lots — needs real conviction (was 52)
    "max_updr_score": 0.12,            # Must be very clean — UP regime (threshold 0.25)
    "max_downdr_score": 0.10,           # Relaxed 0.09→0.10 — was too strict, 0 trades fired (threshold 0.25)
    "min_gate_prob": 0.55,             # XGB gate floor — 5x lots needs strong P(MOVE) (was 0.50)
    "score_tier": "premium",           # Use premium tier sizing (5% risk, +80% target)
    "separate_capital": 300000,        # ₹3 Lakh reserved exclusively for sniper trades
    "max_exposure_pct": 90,            # Max % of sniper capital usable (₹2.7L)
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
    "min_buildup_strength": 0.35,       # OI buildup signal strength >= 0.35 (relaxed from 0.45)
    "min_oi_change_pct": 6.0,           # Dominant OI side must have changed >= 6% (relaxed from 8%)
    # --- Price Reversal at S/R ---
    "max_distance_from_sr_pct": 2.5,    # Spot must be within 2.5% of OI support/resistance (relaxed from 1.5%)
    # --- GMM Quality Gate ---
    "max_updr_score": 0.18,              # GMM clean — UP regime (relaxed from 0.15, aligned with PCR extreme)
    "max_downdr_score": 0.15,            # GMM clean — DOWN regime (relaxed from 0.12)
    "min_gate_prob": 0.40,              # XGB gate P(move) floor (relaxed from 0.45)
    "min_smart_score": 50,              # Minimum smart_score — restored quality floor (45 was too loose)
    # --- Timing ---
    "earliest_entry": "09:45",          # Wait for OI data to settle
    "no_entry_after": "14:30",          # No entries after 2:30 PM
    # --- Risk ---
    "score_tier": "premium",
    "separate_capital": 200000,         # ₹2L reserved for OI unwinding sniper
}

# === SNIPER: PCR EXTREME FADE (Sniper-PCRExtreme) ===
# Fires when stock/index PCR hits extreme levels.
# PCR >= 1.35 = oversold → contrarian BUY. PCR <= 0.65 = overbought → SELL.
# Blends stock PCR with NIFTY index PCR for macro confirmation.
# Tagged as 'SNIPER_PCR_EXTREME'.
SNIPER_PCR_EXTREME = {
    "enabled": True,
    "max_trades_per_day": 3,            # Max PCR Extreme trades per day
    "lot_multiplier": 1.5,              # 1.5x lots
    # --- PCR Extreme Detection ---
    "pcr_oversold_threshold": 1.35,     # PCR >= 1.35 → market oversold → BUY
    "pcr_overbought_threshold": 0.60,   # PCR <= 0.60 → market overbought → SELL (0.55 too tight, kills strategy)
    "min_pcr_edge": 0.05,               # Min distance beyond threshold — blocks hair-trigger entries (ABCAPITAL edge=0.01 lesson)
    # --- Index PCR (Macro Confirmation) ---
    "use_index_pcr": True,              # Also check NIFTY PCR for macro regime
    "index_symbol": "NIFTY",            # Index to check
    "index_pcr_weight": 0.4,            # Blend: 60% stock PCR + 40% index PCR
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
    # Telecom
    "NSE:IDEA",        # Telecom - high volume, volatile moves
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
    "auto_exit_time": "15:05",       # Auto-exit all debit spreads by 3:05 PM (no overnight)
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
    # --- Hedge Leg Selection ---
    "sell_strike_offset": 3,          # Sell leg = 3 strikes OTM from current ATM
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
    "auto_exit_time": "15:05",        # Hard exit time for hedged positions
    # --- Universal Hedge Loss Cap ---
    "max_hedge_loss_pct": 20,          # UNIFIED: only hedge if current loss ≤ 20% (TIE + TIME_STOP)
    # --- TIME_STOP Hedge (dead-trade rescue) ---
    "hedge_time_stop": True,          # Also hedge naked options hitting TIME_STOP (not just TIE)
    # --- Hedge Unwind (restore full upside on recovery) ---
    "unwind_enabled": True,           # Buy back sold leg when thesis re-validates
    "unwind_buy_leg_recovery_pct": 100, # Unwind when buy leg LTP >= 100% of entry (fully recovered)
    "unwind_min_profit_after_cost": 2,  # Min ₹ profit remaining after buyback cost (avoid pointless unwinds)
}

# === PROACTIVE LOSS HEDGE CONFIGURATION ===
# Monitors open naked options every minute. If any position's unrealized
# loss crosses the trigger threshold, automatically converts to debit spread
# BEFORE SL/TIE/TIME_STOP fires — catches fast moves between scan cycles.
PROACTIVE_HEDGE_CONFIG = {
    "enabled": True,
    "loss_trigger_pct": 8,            # Convert when loss >= 8% of entry price
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
    "enabled": True,                    # Auto-managed by expiry detection (Feb 24 fix)
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
    "min_iv_floor": 0.23,                # 23% — reject options with IV < 23%
    # --- Sub-Gate 2: Breadth-Adjusted IV Floor ---
    # When breadth is extreme (most stocks same direction), uncertainty is LOW → IV compresses.
    # Raise the IV floor dynamically based on how skewed the breadth is.
    "breadth_extreme_threshold": 0.85,    # If >85% stocks in same direction → "extreme breadth"
    "breadth_extreme_iv_floor": 0.32,     # Raise IV floor to 32% during extreme breadth
    # --- Sub-Gate 3: Minimum Premium (OTM penny filter upgrade) ---
    # Cheap options (<₹8) are disproportionately affected by IV changes:
    #   ₹1 IV drop on ₹5.49 option = -18.2% vs ₹1 drop on ₹20 = -5%
    # HARD_RULES.MIN_OPTION_PREMIUM (₹3) catches pennies; this catches "cheap but not penny".
    "min_premium_for_iv_safety": 8.0,     # ₹8 minimum premium (IV-safety floor)
    # --- Sub-Gate 4: Vega/Delta Ratio ---
    # If vega exposure >> delta, the trade is more of a volatility bet than a direction bet.
    # On low-IV stocks, this catches "looks directional but is really a vega trap".
    "max_vega_delta_ratio": 0.50,         # Block if |vega/delta| > 0.50 AND IV < min floor
    "vega_delta_iv_threshold": 0.30,      # Only apply vega/delta check below 30% IV
}

# === THETA BLEED PREVENTION (Entry-Side Gates) ===
# Prevents naked option buys that will lose to time decay before direction plays out.
# Root cause: options bought with high theta-to-premium ratio bleed even if direction is right.
THETA_ENTRY_GATE = {
    "enabled": True,
    # --- Theta/Premium Ratio Gate (blocks trades where decay > expected gain) ---
    # Daily |theta| as % of premium. If theta eats >5% per day, stock needs >5% move just to break even.
    "max_theta_pct_of_premium": 5.0,    # Block if |daily theta| > 5% of option LTP
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
    "trail_retain_pct": 0.65,    # Lock 65% of peak profit in extension zone (35% giveback)
    "max_extension_r": 5.0,      # Safety cap: force exit at 5R (prevents runaway positions)
}

# === ASYMMETRIC EXIT INTELLIGENCE ===
# Reads trade TRAJECTORY, not just snapshots.
# Losers must prove they deserve to live. Winners must prove they deserve to die.
ASYMMETRIC_EXIT = {
    "enabled": True,
    # --- Premium Velocity Kill (fast-kill steady bleeders) ---
    "velocity_kill_enabled": True,
    "velocity_min_candles": 3,          # Need at least 3 candles of data
    "velocity_threshold_pct": -1.5,     # Kill if avg velocity < -1.5%/candle
    "velocity_consecutive": 3,          # Must bleed for 3 consecutive candles
    "velocity_underlying_confirm": True, # Also check underlying isn't recovering
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
    "iv_rv_ratio_hard_block": 2.0,     # Block entry if IV > 2× realized vol
    "iv_rv_ratio_reduce_lots": 1.5,    # Halve lots if IV > 1.5× realized vol
    # --- Absolute IV Caps ---
    "max_atm_iv_pct": 60,              # Hard block: ATM IV > 60% (extreme overpricing)
    "reduce_atm_iv_pct": 45,           # Halve lots: ATM IV > 45% (elevated overpricing)
    # --- Morning IV Adjustment ---
    "morning_iv_premium_until": "11:00",  # Before 11:00 morning IV is structurally inflated
    "morning_ratio_penalty": 0.2,         # Add 0.2 to ratio thresholds (more lenient in morning)
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
