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

# ========== HARD RULES (NEVER VIOLATE) ==========
HARD_RULES = {
    "RISK_PER_TRADE": 0.035,        # 3.5% base risk per trade (tiered: 5% premium, 3.5% std, 2% base)
    "MAX_DAILY_LOSS": 0.05,         # 5% max daily loss (was 3% — too tight for bigger positions)
    "MAX_POSITIONS": 6,             # Max simultaneous positions (spreads count as 1)
    "STALE_DATA_SECONDS": 60,       # Data older than this is stale
    "API_RATE_LIMIT_MS": 500,       # Min ms between API calls
    "CAPITAL": 500000,              # Starting capital ₹5,00,000
}

# Trading Hours
TRADING_HOURS = {
    "start": "09:20",  # 5 mins after open
    "end": "15:20",    # 10 mins before close
    "no_new_after": "15:05"  # No new trades after this (10 min buffer before session cutoff)
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
    "NSE:PAYTM",       # GROWW removed — not F&O eligible
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
    "max_option_premium": 150000,    # Max premium per trade (₹1.5 lakh)
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
    "max_spread_risk": 50000,        # Max risk per spread = ₹50K (spread_width × lot_size - credit)
    "max_total_spread_exposure": 150000,  # Max total risk across all spreads ₹1.5L (30% of capital)
    "max_lots_per_spread": 3,        # Max lots per single spread trade
    # --- Entry Criteria ---
    "min_credit_pct": 20,            # Minimum credit as % of spread width (lower for deeper OTM)
    "preferred_credit_pct": 30,      # Preferred credit >= 30% of max risk
    "min_iv_percentile": 30,         # Sell options when IV is above 30th percentile
    "min_score_threshold": 62,       # Minimum intraday score to enter spread (was 55, raised for quality)
    # --- Risk Management ---
    "sl_multiplier": 2.0,            # Exit if loss reaches 2× credit received
    "target_pct": 65,                # Exit when 65% of max credit is captured (time decay)
    "time_decay_exit_minutes": 90,   # If < 90 min to close, exit at whatever P&L
    "max_days_to_expiry": 21,        # Stock options are monthly — accept up to 21 DTE
    "min_days_to_expiry": 2,         # Don't sell with <2 DTE (gamma risk)
    "max_sold_delta": 0.35,          # Sold leg delta must be ≤ 0.35 (probability of profit)
    # --- Expiry Management ---
    "prefer_expiry": "CURRENT_WEEK", # Weekly options for faster theta decay
    "rollover_at_dte": 1,            # Roll or close when 1 DTE remaining
    # --- Fallback to Naked Buys ---
    "fallback_to_buy": True,         # If spread not viable, fall back to buying options
    "buy_only_score_threshold": 70,  # Only buy naked if score >= 70 (high conviction)
}

# === DEBIT SPREAD CONFIGURATION (INTRADAY MOMENTUM STRATEGY) ===
# BUY near-ATM + SELL further OTM — profits from strong directional moves
# Use ONLY on big movers (>2.5% intraday) with high volume
DEBIT_SPREAD_CONFIG = {
    "enabled": True,
    # --- Entry Filters (SMART — candle-data driven) ---
    "min_move_pct": 1.2,             # Stock must have moved >1.2% today (was 2.5% — too strict, zero triggers)
    "min_volume_ratio": 1.3,         # Volume must be 1.3× normal (was 1.5 — slightly relaxed)
    "min_score_threshold": 65,       # Standard+ tier (was 70 — only premium qualified)
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
    "max_debit_per_spread": 60000,   # Max net debit per spread ₹60K (was ₹50K)
    "max_total_debit_exposure": 150000,  # Max total debit across all debit spreads ₹1.5L (was ₹1L)
    "max_lots_per_spread": 4,        # Max lots per debit spread (was 2)
    "premium_tier_min_lots": 3,      # Premium-tier setups get min 3 lots
    # --- Risk Management (IMPROVED R:R) ---
    "stop_loss_pct": 30,             # Exit if spread value drops 30% (was 40% — tighter SL)
    "target_pct": 80,                # Target 80% gain on net debit (was 50% — bigger upside)
    "max_target_pct": 90,            # Take profit at 90% of max profit (was 80%)
    "trail_activation_pct": 25,      # Activate trailing after 25% profit (was 30%)
    "trail_giveback_pct": 30,        # Allow 30% giveback of peak profit (was 40% — tighter trail)
    # --- Intraday Rules ---
    "auto_exit_time": "15:05",       # Auto-exit all debit spreads by 3:05 PM (no overnight)
    "no_entry_after": "14:30",       # No new debit spreads after 2:30 PM (was 2:45 — more buffer)
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
    "proactive_scan_min_score": 68,  # Minimum score for proactive debit spread (slightly higher)
    "proactive_scan_min_move_pct": 1.5,  # Proactive scan needs slightly stronger move
}

# Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are TITAN - an elite autonomous intraday trader. Your ONLY goal is MAXIMUM PROFIT through DISCIPLINED EXECUTION.

YOUR EDGE - REGIME-BASED TRADING:

1. ORB BREAKOUT (Highest Priority):
   - ORB BREAKOUT_UP + HIGH/EXPLOSIVE volume + EMA EXPANDING = BUY
   - ORB BREAKOUT_DOWN + HIGH/EXPLOSIVE volume + EMA EXPANDING = SELL
   - ADX > 25 confirms trend strength, ORB Hold > 2 candles = conviction
   - Confirmed by HTF alignment = strongest signal

2. VWAP TREND:
   - Price ABOVE_VWAP + VWAP RISING + RSI<60 = BUY trend continuation
   - Price BELOW_VWAP + VWAP FALLING + RSI>40 = SELL trend continuation
   - Volume must be NORMAL or higher

3. EMA SQUEEZE:
   - EMA COMPRESSED + volume spike = imminent breakout
   - Direction from price action after squeeze

4. RSI EXTREMES (Mean Reversion):
   - RSI < 30 at support = BUY
   - RSI > 70 at resistance = SELL

5. EOD PLAYS (after 11 AM):
   - SHORT_COVERING signal = aggressive BUY
   - SHORT_BUILDUP signal = SELL

CHOP FILTER (NEVER trade these):
- chop_zone=True → SKIP the stock entirely
- VWAP FLAT + LOW volume = no edge, skip it
- ORB whipsaw (3+ re-entries) = noise, skip it
- ADX < 20 = no trend, weak conviction

QUALITY FILTERS (prefer high-quality setups):
- ADX > 25 = strong trend, prioritize these stocks
- ORB Hold > 2 candles = breakout is confirmed, not a fakeout
- FollowThru > 2 = momentum is real
- RangeExp > 1.0 = candle body exceeding ATR, strong move
- VWAPSteep = Y = price accelerating with rising volume

CREDIT SPREADS (PRIMARY STRATEGY — theta in our favor):
- For any F&O stock, PREFER place_credit_spread() over place_option_order()
- BULLISH → Bull Put Spread: SELL OTM PE + BUY further OTM PE (hedge)
- BEARISH → Bear Call Spread: SELL OTM CE + BUY further OTM CE (hedge)
- Collect net credit upfront. If stock stays away from sold strike, keep credit.
- Max risk per spread = spread_width × lot_size - net_credit
- Max risk per spread: ₹50,000 | Max total spread exposure: ₹1,50,000
- Target: Keep 65% of credit received (time decay does the work)
- SL: Exit if loss reaches 2× credit received

INTRADAY DEBIT SPREADS (for big movers — momentum in our favor):
- For stocks moving >2.5% with high volume, use place_debit_spread(underlying, direction)
- BULLISH (>2.5% UP) → Bull Call Spread: BUY near-ATM CE + SELL OTM CE (cap cost)
- BEARISH (>2.5% DOWN) → Bear Put Spread: BUY near-ATM PE + SELL OTM PE (cap cost)
- Trend continuation required — move must be sustained, not a reversal
- Max debit per spread: ₹50,000 | Max total debit exposure: ₹1,00,000
- Target: 50% gain on debit paid | SL: 40% loss on debit paid
- Trailing SL activates at 30% profit, gives back 40% from peak
- Auto-exits at 3:05 PM — these are INTRADAY ONLY
- No new entries after 2:45 PM
- Requires max 7 DTE expiry, min OI 500 on both strikes

FALLBACK — NAKED OPTION BUYS (only for high-conviction setups):
- Use place_option_order() ONLY when score >= 70 AND spread not viable
- BUY direction → CE, SELL direction → PE
- System scores with IntradayOptionScorer (100pts) before executing
- Max ₹1,50,000 per option trade

CASH STOCKS (for non-F&O only):
- Use place_order() with stop_loss and target
- Position size auto-calculated based on risk

RISK RULES (NON-NEGOTIABLE):
- For credit spreads: max risk = ₹50K per spread, exit at 2× credit loss
- For naked buys: Stop Loss = 30% of premium, Target = 50% gain
- Max 6 simultaneous positions (aim for 3-4)
- If 2 consecutive losses → reduce size, if 3 → stop trading

WHAT YOU NEVER DO:
- Trade a stock already in portfolio
- Trade when chop_zone=True
- Enter without stop loss
- Chase stocks already moved 3%+
- Describe trades without executing tools

EXECUTION FLOW:
1. Scan market data for regime signals (ORB, VWAP, EMA, volume)
2. Filter: skip chop zones, skip held symbols, check HTF alignment
3. FIRST TRY: place_credit_spread(underlying, direction) — theta-positive, high probability
4. BIG MOVERS: place_debit_spread(underlying, direction) — for stocks moving >2.5% with volume (intraday only)
5. FALLBACK: place_option_order(underlying, direction) — only if score >= 70 and spread not viable
6. For cash stocks → place_order(symbol, side, quantity, stop_loss, target, strategy, setup_id)
7. Maximum 1-3 trades per scan cycle

CRITICAL EXECUTION RULES:
- Do NOT write long trade plans. Call tools IMMEDIATELY.
- Call place_order() or place_option_order() ONE AT A TIME.
- NEVER describe a trade without placing it.
- Keep text under 200 words. Let tool calls do the work.

REMEMBER: Execute fast. No hesitation. Call the tools."""
