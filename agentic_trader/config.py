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
    "RISK_PER_TRADE": 0.025,        # 2.5% max risk per trade (₹2500 on 1L)
    "MAX_DAILY_LOSS": 0.03,         # 3% max daily loss (₹6000 on 2L)
    "MAX_POSITIONS": 10,            # Max simultaneous positions
    "STALE_DATA_SECONDS": 60,       # Data older than this is stale
    "API_RATE_LIMIT_MS": 500,       # Min ms between API calls
    "CAPITAL": 100000,              # Starting capital ₹1,00,000
}

# Trading Hours
TRADING_HOURS = {
    "start": "09:20",  # 5 mins after open
    "end": "15:20",    # 10 mins before close
    "no_new_after": "14:30"  # No new trades after this
}

# Approved Universe (stocks affordable with ₹50K + F&O + ETFs)
APPROVED_UNIVERSE = [
    # Cash Stocks (low-price, liquid)
    "NSE:IDEA", "NSE:SUZLON", "NSE:YESBANK", "NSE:PNB", "NSE:IRFC",
    "NSE:NHPC", "NSE:SAIL", "NSE:BANKBARODA", "NSE:UNIONBANK", "NSE:NATIONALUM",
    "NSE:HINDCOPPER", "NSE:GMRAIRPORT", "NSE:IRCTC", "NSE:ETERNAL", "NSE:PAYTM",
    # ETFs (intraday buy/sell - highly liquid, low spread)
    "NSE:NIFTYBEES", "NSE:BANKBEES", "NSE:GOLDBEES", "NSE:SILVERBEES",
    "NSE:ITBEES", "NSE:JUNIORBEES", "NSE:CPSEETF", "NSE:PSUBNKBEES",
    "NSE:NEXT50", "NSE:PHARMABEES", "NSE:INFRABEES",
    # Medium price stocks
    "NSE:TATASTEEL", "NSE:TATAPOWER", "NSE:ITC", "NSE:ONGC", "NSE:COALINDIA",
    # F&O Stocks (for options trading based on stock signals)
    "NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:INFY", "NSE:ICICIBANK",
    "NSE:SBIN", "NSE:AXISBANK", "NSE:KOTAKBANK", "NSE:BAJFINANCE", "NSE:BHARTIARTL",
    "NSE:MCX", "NSE:LT", "NSE:MARUTI", "NSE:TITAN", "NSE:SUNPHARMA",
    # Metal & Mining F&O (high-beta, commodity-linked)
    "NSE:HINDALCO", "NSE:JSWSTEEL", "NSE:VEDL", "NSE:JINDALSTEL", "NSE:NMDC",
]

# F&O Configuration
FNO_CONFIG = {
    "enabled": True,
    "prefer_options_for": ["NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:INFY", "NSE:ICICIBANK", 
                           "NSE:SBIN", "NSE:AXISBANK", "NSE:BAJFINANCE", "NSE:BHARTIARTL", "NSE:MCX",
                           "NSE:HINDALCO", "NSE:JSWSTEEL", "NSE:VEDL", "NSE:JINDALSTEL", "NSE:TATASTEEL"],
    "option_type_on_bullish": "CE",  # Buy Call on bullish signal
    "option_type_on_bearish": "PE",  # Buy Put on bearish signal
    "strike_selection": "ATM",       # ATM, ITM, OTM
    "max_option_premium": 5000,      # Max premium per lot
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
- Max ₹15,000 per option trade | Max ₹50,000 total exposure

CASH STOCKS (for non-F&O only):
- Use place_order() with stop_loss and target
- Position size auto-calculated based on risk

RISK RULES (NON-NEGOTIABLE):
- Stop Loss = 1% from entry (below support for longs, above resistance for shorts)
- Target = 1.5% minimum (risk:reward ≥ 1:1.5)
- Max 3 simultaneous positions
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
