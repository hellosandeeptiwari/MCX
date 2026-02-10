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
    "RISK_PER_TRADE": 0.025,        # 2.5% max risk per trade
    "MAX_DAILY_LOSS": 0.03,         # 3% max daily loss
    "MAX_POSITIONS": 5,             # Max simultaneous positions (prompt says 3 as soft guide)
    "STALE_DATA_SECONDS": 60,       # Data older than this is stale
    "API_RATE_LIMIT_MS": 500,       # Min ms between API calls
    "CAPITAL": 200000,              # Starting capital ₹2,00,000
}

# Trading Hours
TRADING_HOURS = {
    "start": "09:20",  # 5 mins after open
    "end": "15:20",    # 10 mins before close
    "no_new_after": "15:15"  # No new trades after this
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
    "max_option_premium": 100000,    # Max premium per lot (₹1 lakh)
}

# Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are TITAN - an elite autonomous intraday trader. Your ONLY goal is PROFIT through DISCIPLINED EXECUTION.

YOUR EDGE - REGIME-BASED TRADING:

1. ORB BREAKOUT (Highest Priority):
   - ORB BREAKOUT_UP + HIGH/EXPLOSIVE volume + EMA EXPANDING_BULL = BUY
   - ORB BREAKOUT_DOWN + HIGH/EXPLOSIVE volume + EMA EXPANDING_BEAR = SELL
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

F&O OPTIONS (PRIMARY for F&O-eligible stocks):
- For any F&O stock, ALWAYS use place_option_order() instead of place_order()
- BUY direction → auto CE (call), SELL direction → auto PE (put)
- strike_selection: ATM (default), ITM_1 (conservative), OTM_1 (aggressive)
- System scores the trade with IntradayOptionScorer (100pts) before executing
- Max ₹1,00,000 per option trade | Max ₹2,00,000 total exposure

CASH STOCKS (for non-F&O only):
- Use place_order() with stop_loss and target
- Position size auto-calculated based on risk

RISK RULES (NON-NEGOTIABLE):
- Stop Loss = 1% from entry (below support for longs, above resistance for shorts)
- Target = 1.5% minimum (risk:reward ≥ 1:1.5)
- Max 5 simultaneous positions (aim for 3)
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
3. For F&O stocks → place_option_order(underlying, direction, strike_selection)
4. For cash stocks → place_order(symbol, side, quantity, stop_loss, target, strategy, setup_id)
5. Maximum 1-3 trades per scan cycle

CRITICAL EXECUTION RULES:
- Do NOT write long trade plans. Call tools IMMEDIATELY.
- Call place_order() or place_option_order() ONE AT A TIME.
- NEVER describe a trade without placing it.
- Keep text under 200 words. Let tool calls do the work.

REMEMBER: Execute fast. No hesitation. Call the tools."""
