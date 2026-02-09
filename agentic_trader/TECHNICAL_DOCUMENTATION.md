# TITAN - Autonomous Options Trading System
## Technical Documentation

> **Goal:** ₹2L → ₹20L (10x returns) via fully autonomous intraday options trading on NSE F&O

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [Scoring Systems](#scoring-systems)
5. [Risk Management](#risk-management)
6. [Order Execution](#order-execution)
7. [Safety Gates](#safety-gates)
8. [Configuration](#configuration)
9. [Data Flow](#data-flow)

---

## System Overview

TITAN is an **AI-powered autonomous trading agent** that:
- Uses OpenAI GPT-4 for reasoning and strategy decisions
- Executes trades via Zerodha Kite Connect API
- Specializes in intraday options on NSE F&O stocks
- Implements multi-layer risk controls to prevent "risk of ruin"

### Core Philosophy
```
"Price escapes balance and does not come back."
- Don't predict, FOLLOW the trend
- Trend is your friend until it bends
- Add to winners, cut losers fast
- Let profits run with trailing stops
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMOUS TRADER                                 │
│                    (autonomous_trader.py)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  LLM Agent   │───▶│  Decision    │───▶│  Execution   │              │
│  │  (GPT-4)     │    │  Engines     │    │  Layer       │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │                   SAFETY LAYER                          │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │           │
│  │  │Risk     │ │Execution│ │Data     │ │Position │       │           │
│  │  │Governor │ │Guard    │ │Health   │ │Recon    │       │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │           │
│  └─────────────────────────────────────────────────────────┘           │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │           ZERODHA KITE CONNECT API                      │           │
│  │  (zerodha_tools.py - Orders, Positions, Quotes)         │           │
│  └─────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Trend Following Engine (`trend_following.py`)
**Purpose:** Score trend strength and determine entry/exit signals

#### Scoring Weights (100 points total)
| Factor | Points | Why It Matters |
|--------|--------|----------------|
| VWAP Slope | 25 | Institutional direction |
| EMA Expansion | 20 | Trend acceleration |
| Volume Regime | 20 | Participation validation |
| ORB Break Hold | 15 | Momentum confirmation |
| Pullback Quality | 10 | Continuation strength |
| ADX Strength | 10 | Trend conviction |

#### Trend States
```
STRONG_BULLISH  (80+)  - Aggressive entry
BULLISH         (60-79) - Standard entry
NEUTRAL         (40-59) - No trade
BEARISH         (20-39) - Short signals
STRONG_BEARISH  (0-19)  - Aggressive short
```

#### Enhanced Hysteresis System
Prevents whipsaw by making state transitions asymmetric:

| Transition | Threshold | Candles Required |
|------------|-----------|------------------|
| Upgrade to STRONG | ≥ 82 | 2 consecutive |
| Stay STRONG | ≥ 70 | - |
| Downgrade from STRONG | < 70 | Immediate |
| Upgrade to TREND | ≥ 60 | - |
| Downgrade to NEUTRAL | < 50 | Immediate |

#### Shock Override
Force immediate downgrade if:
- VWAP crossed against position + HIGH/EXPLOSIVE volume
- Price moves > 2x ATR against position

---

### 2. Intraday Option Scorer (`options_trader.py`)
**Purpose:** Score intraday signals for options trading decisions

#### Scoring Weights (~100 points total)
| Factor | Points | Description |
|--------|--------|-------------|
| Trend Following | 30 | From TrendFollowingEngine |
| Microstructure | 15 | Spread/depth/OI/fills |
| ORB Breakout | 20 | Morning breakout signals |
| Volume Regime | 15 | EXPLOSIVE/HIGH/NORMAL/LOW |
| VWAP Position | 10 | Price vs VWAP alignment |
| EMA Regime | 10 | Trend structure |
| Acceleration | 10 | Follow-through momentum |
| HTF Alignment | 5 | Higher timeframe |
| RSI Penalty | -3 | Overextension only |

#### Trade Tiers (3-Tier System)
| Score | Tier | Strike | Size |
|-------|------|--------|------|
| < 30 | BLOCK | - | No trade |
| 30-34 | Standard | ATM/ITM | 1.0x |
| 35-54 | Standard+ | ATM/ITM | 1.0x |
| ≥ 55 | Premium | ATM/ITM | Up to 1.2x |

#### Caller Direction Bonus
When GPT agent's direction aligns with intraday signals (VWAP, volume, EMA), up to +5 bonus points are added to the score.

#### OTM Strike Requirements (Special Case)
OTM strikes are **NOT default** - require ALL of:
- EXPLOSIVE volume regime
- Spread < 0.3%
- Acceleration score ≥ 8/10

#### Aggressive Sizing Requirements
1.2x size requires **ALL** of:
- Premium tier (≥ 55)
- Acceleration ≥ 8/10
- Microstructure ≥ 12/15

---

### 3. Options Microstructure (`options_trader.py`)
**Purpose:** Hard gate for options tradability - prevents losses from poor execution

#### Microstructure Data Structure
```python
@dataclass
class OptionMicrostructure:
    bid: float                    # Best bid price
    ask: float                    # Best ask price
    spread_pct: float             # Spread as % of mid
    bid_qty: int                  # Top-of-book depth
    ask_qty: int                  # Top-of-book depth
    open_interest: int            # OI
    option_volume: int            # Today's volume
    oi_volume_ratio: float        # OI / Volume
    partial_fill_rate: float      # 0-1
    cancel_rate: float            # 0-1
    ltp: float                    # Last traded price
```

#### Microstructure Thresholds
| Metric | Good | Acceptable | Bad | BLOCK |
|--------|------|------------|-----|-------|
| Spread % | < 0.3% | < 0.5% | < 1% | > 2% |
| Spread Absolute | - | - | - | > ₹3 (index) / ₹5 (stock) |
| Top Depth | ≥ 100 | ≥ 50 | ≥ 25 | < 25 |
| Open Interest | ≥ 2000 | ≥ 500 | ≥ 500 | < 500 |
| Option Volume | ≥ 100 | ≥ 100 | - | < 100 |
| Partial Fill Rate | < 0.3 | < 0.5 | ≥ 0.5 | - |

---

### 4. Exit Manager (`exit_manager.py`)
**Purpose:** First-class exit logic for consistent edge preservation

#### Exit Types
| Type | Trigger | Action |
|------|---------|--------|
| Hard Stop Loss | Structure-based (ORB, swing, VWAP) | Immediate exit |
| Time Stop | No +0.5R in 7 candles | Exit position |
| Break-even | +0.8R achieved | Move SL to entry |
| Session Cutoff | 15:20 IST | Exit all positions |
| Trailing Stop | After +1R, trail at 50% | Lock in profits |
| Target Hit | 2R or 3R target | Take profit |

#### R-Multiple Tracking
```
R = (Entry - Stop Loss)
+0.5R = Move halfway to break-even consideration
+0.8R = Move stop to break-even
+1.0R = Activate trailing stop
+2.0R = First profit target
+3.0R = Second profit target
```

---

### 5. Regime Scorer (`regime_score.py`)
**Purpose:** Confidence gate for trade quality

#### Thresholds by Trade Type
| Trade Type | Minimum Score |
|------------|---------------|
| ORB Breakout | 70 |
| VWAP Trend | 60 |
| EMA Squeeze | 65 |
| Mean Reversion | 65 |
| EOD Play | 55 |
| Momentum | 60 |

---

## Risk Management

### 1. Risk Governor (`risk_governor.py`)
**Purpose:** Account-level risk with kill-switch capability

#### System States
```
ACTIVE        - Normal trading
COOLDOWN      - 10 min pause after loss
HALT_TRADING  - Stopped for the day
CIRCUIT_BREAK - Emergency stop (data/order issues)
```

> **Critical Fix (v1.4):** `is_trading_allowed()` now clears expired cooldowns.
> Previously, cooldown expiry was only checked in `can_trade_general()` which was
> never reached because `is_trading_allowed()` returned False first, permanently
> halting the bot after any single loss.

#### Risk Limits
| Limit | Default Value |
|-------|---------------|
| Max Daily Loss | 2% of capital |
| Max Consecutive Losses | 3 |
| Max Trades Per Day | 8 |
| Max Symbol Exposure | 2 same sector |
| Cooldown Duration | 10 minutes |
| Max Position Size | 25% of capital |
| Max Total Exposure | 80% of capital |
| Max Premium Per Trade | ₹35,000 |
| Max Total Option Exposure | ₹80,000 |

### 2. Correlation Guard (`correlation_guard.py`)
**Purpose:** Prevent hidden overexposure from correlated positions

#### Beta Categories
- **HIGH_BETA (>1.2):** TATAMOTORS, BAJFINANCE, ADANIENT, Banks, Metals
- **MEDIUM_BETA (0.8-1.2):** RELIANCE, HDFCBANK, LT, TITAN
- **LOW_BETA (<0.8):** IT stocks (INFY, TCS), Pharma, FMCG
- **INDEX:** NIFTY, BANKNIFTY options/futures

#### Rules
- Max 2 HIGH_BETA positions at a time
- If holding index options, max 1 additional HIGH_BETA stock
- Block if real-time correlation exceeds threshold

---

## Order Execution

### 1. Execution Guard (`execution_guard.py`)
**Purpose:** Prevent edge leakage through poor fills

#### Spread Checking
| Spread | Action |
|--------|--------|
| < 0.3% | MARKET order OK |
| 0.3-0.5% | LIMIT with timeout |
| > 0.5% | BLOCK or LIMIT only |
| > ₹5 absolute | BLOCK trading |

#### Order Type by Volume Regime
| Volume Regime | Order Type | Max Slippage |
|---------------|------------|--------------|
| EXPLOSIVE | IOC | 0.5% |
| HIGH | MARKET | 0.3% |
| NORMAL | LIMIT | 0.15% |
| LOW | BLOCK | 0% (no trade) |

### 2. Idempotent Order Engine (`idempotent_order_engine.py`)
**Purpose:** Prevent duplicate orders on reconnect/retry

#### Client Order ID Generation
```
Format: SYMBOL_YYYYMMDD_STRATEGY_DIRECTION_HASH
Example: RELIANC_20260207_ORB_B_a1b2c3d4
```

#### Flow
1. Create intent with (symbol, direction, strategy, setup_id)
2. Generate deterministic client_order_id
3. Check broker for existing orders with same ID
4. If found: do nothing (idempotent)
5. If not found: place order and record

---

## Safety Gates

### 1. Data Health Gate (`data_health_gate.py`)
**Purpose:** Block trading on stale/dirty data

#### Health Checks
| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Last tick age | > 60 seconds | Block + increment stale counter |
| Candle gap | > 15 minutes | Block |
| Volume | = 0 | Block (feed glitch) |
| VWAP/EMA/ATR | NaN or 0 | Block (invalid indicators) |
| Stale counter | > 5 consecutive | Halt symbol for day |

### 2. Position Reconciliation (`position_reconciliation.py`)
**Purpose:** Periodic truth sync with broker

#### Reconciliation States
```
SYNCED     - Local matches broker
MISMATCH   - Discrepancy found
RECOVERY   - Fixing mismatches, no new trades
FROZEN     - Manual intervention required
INIT       - Startup sync in progress
```

#### Check Interval: 5-15 seconds
- Pull broker positions + open orders
- Compare to local state (active_trades.json)
- If mismatch: Cancel open orders, freeze entries, log for review

### 3. Hard Gates (Options Trading)
| Gate | Condition for BLOCK |
|------|---------------------|
| Score | < 30 (BLOCK threshold) |
| Direction | Cannot determine |
| Microstructure | Spread > 2%, OI < 500, depth < 25 |
| Duplicate | Already holding option on same underlying |
| Premium | Single trade > ₹35K or total exposure > ₹80K |

> **Note:** NEUTRAL trend no longer fully blocks — it applies a -5 point penalty
> instead of -10, allowing strong intraday signals to compensate.

---

## Configuration (`config.py`)

### Hard Rules
```python
HARD_RULES = {
    "RISK_PER_TRADE": 0.025,      # 2.5% max risk per trade
    "MAX_DAILY_LOSS": 0.05,       # 5% max daily loss
    "MAX_POSITIONS": 10,          # Max simultaneous positions
    "STALE_DATA_SECONDS": 60,     # Data freshness threshold
    "API_RATE_LIMIT_MS": 500,     # Min ms between API calls
    "CAPITAL": 100000,            # Starting capital ₹1L
}
```

### Trading Hours
```python
TRADING_HOURS = {
    "start": "09:20",      # 5 mins after market open
    "end": "15:20",        # 10 mins before close
    "no_new_after": "14:30"  # No new trades after 2:30 PM
}
```

### Approved Universe
- **Cash Stocks:** IDEA, SUZLON, YESBANK, PNB, IRFC, NHPC, SAIL, etc.
- **F&O Stocks:** RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, AXISBANK, etc.

---

## Data Flow

### Entry Signal Flow
```
Market Data (Zerodha)
        │
        ▼
┌──────────────────┐
│ Data Health Gate │ ──▶ BLOCK if stale/invalid
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ TrendFollowing   │ ──▶ TREND score + state
│ Engine           │     (with hysteresis)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ IntradayOption   │ ──▶ Trade decision
│ Scorer           │     + strike + size
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Regime Scorer    │ ──▶ Quality gate
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Risk Governor    │ ──▶ Account limits
│ Correlation Guard│     + correlation check
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Execution Guard  │ ──▶ Spread/slippage check
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Idempotent Engine│ ──▶ Duplicate prevention
└────────┬─────────┘
         │
         ▼
    PLACE ORDER (Zerodha)
```

### Exit Signal Flow
```
Position Monitoring
        │
        ▼
┌──────────────────┐
│ Exit Manager     │ ──▶ SL/Target/Trailing/Time
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Position Recon   │ ──▶ Verify broker state
└────────┬─────────┘
         │
         ▼
    EXIT ORDER (Zerodha)
```

---

## Anti-Double-Counting

### Problem
TrendFollowing and OptionScorer may score the same signal twice:
- VWAP Slope (TrendFollowing) + VWAP Position (OptionScorer)
- ORB Break (TrendFollowing) + ORB Breakout (OptionScorer)
- Volume (TrendFollowing) + Volume Regime (OptionScorer)

### Solution: Window Overlap Detection
```python
# If windows overlap > 50%, reduce score by half
if orb_window_minutes <= TREND_WINDOW_MINUTES:  # 15 min
    score_cap = WEIGHTS['orb_breakout'] / 2  # 10 instead of 20
```

### VWAP Position Fix
- Reduced weight: 15 → 10 points
- Penalizes misalignment: -8 points instead of +10
- Only rewards strong alignment: > 0.3% above/below VWAP

---

## Key Design Decisions

### 1. Balanced Risk & Opportunity
- Score ≥ 30 to trade (BLOCK), ≥ 35 Standard+, ≥ 55 Premium
- Max size 1.2x (Premium tier only)
- NEUTRAL trend applies -5 penalty (not full block)
- OTM strikes require special conditions
- Duplicate underlying check prevents double exposure

### 2. Institutional Signals First
- VWAP Slope (25 pts) > all other indicators
- Volume regime gates everything
- EMA expansion confirms trend acceleration

### 3. Quick Exits
- Time stop if no progress in 7 candles
- Session cutoff at 15:20 (EOD exit at 15:15)
- Break-even at +0.8R
- Trailing after +1R

### 4. No Predictions
- Follow price, don't anticipate
- Trend state with hysteresis (2 candles to upgrade)
- Shock override for violent reversals

---

## File Structure

```
agentic_trader/
├── autonomous_trader.py     # Main trading bot
├── config.py                # Configuration & hard rules
├── llm_agent.py             # GPT-4 agent wrapper
├── zerodha_tools.py         # Kite Connect API tools
├── trend_following.py       # Trend detection engine
├── options_trader.py        # Options scoring & decisions
├── regime_score.py          # Trade quality scoring
├── exit_manager.py          # Exit logic & trailing
├── execution_guard.py       # Order quality guard
├── risk_governor.py         # Account-level risk
├── correlation_guard.py     # Beta/correlation guard
├── position_reconciliation.py # Broker sync
├── data_health_gate.py      # Data quality gate
├── idempotent_order_engine.py # Duplicate prevention
├── dashboard.py             # Web dashboard
├── run.py                   # Entry point
└── templates/               # HTML templates
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | - | Initial TrendFollowing + Options |
| 1.1 | - | Added Microstructure (15 pts) |
| 1.2 | - | Replaced RSI with Acceleration |
| 1.3 | Feb 2026 | Risk of ruin fixes (7 changes) |
| 1.4 | 9 Feb 2026 | Scoring fix + bug fixes + ROI optimization |

### v1.3 Risk of Ruin Fixes
1. OTM + 1.5x → OTM special case + max 1.2x
2. Tightened spread gates (₹3 index, ₹5 stock)
3. Anti-double-counting by window overlap
4. TREND state gate (NEUTRAL blocks)
5. Aggressive requires Accel + Micro dominance
6. VWAP Position reduced + penalty for misalignment
7. Simplified to 2 thresholds (65 standard, 80 premium) → later revised to 30/35/55 in v1.4

### v1.4 Scoring Fix + Bug Fixes (9 Feb 2026)

**Problem:** Bot was not taking any trades. Investigation of the 19-gate pipeline
revealed the IntradayOptionScorer thresholds were mathematically unreachable when
trend=NEUTRAL (-10 penalty) + missing microstructure data. Best realistic score was ~31
but BLOCK_THRESHOLD was 45.

**Scoring Changes (options_trader.py):**
1. BLOCK_THRESHOLD: 45 → **30**
2. STANDARD_THRESHOLD: 45 → **35**
3. PREMIUM_THRESHOLD: 65 → **55**
4. NEUTRAL trend penalty: -10 → **-5**
5. No-trend-data penalty: -10 → **-5**
6. Added `caller_direction` parameter with +5 alignment bonus
7. Fallback direction from caller when signals ambiguous

**Critical Bug Fixes:**
8. **Cooldown expiry not clearing** (`risk_governor.py`): `is_trading_allowed()` checked
   `system_state == ACTIVE` but never cleared expired cooldowns. Only `can_trade_general()`
   did, which was unreachable. Bot permanently halted after any single loss.
9. **Option side='SELL' for PE buys** (`zerodha_tools.py`): Options stored with the market
   direction as `side` instead of always `'BUY'`. Caused wrong P&L formula (SHORT instead
   of LONG) and premature TARGET_HIT exits.
10. **Duplicate option positions** (`zerodha_tools.py`): No check for existing option on
    same underlying. GPT retried different strikes + retry mechanism created duplicates.
    Added underlying duplicate check.
11. **Default capital** (`autonomous_trader.py`): argparse default was ₹10,000 instead of
    ₹200,000.

**Configuration Changes:**
12. Cooldown: 15 → **10 minutes**
13. Max trades per day: 5 → **8**
14. Max consecutive losses: 2 → **3**
15. Per-trade premium cap: ₹25K → **₹35K**
16. Total option exposure: ₹75K → **₹80K**
17. Market close: 15:15 → **15:20** (EOD exit at 15:15)
18. F&O opportunity filter broadened (VWAP_TREND + MOMENTUM setups)

**Day 1 Results (Paper Trading):**
- Capital: ₹200,000
- Trades: 6 (5W / 1L)
- Day P&L: **+₹2,867 (+1.43%)**
- Winners: SBIN equity (+₹240), SBIN CE (+₹2,100), TATASTEEL PE (+₹550),
  INFY PE (+₹20), JINDALSTEL PE (+₹469)
- Loser: UNIONBANK equity (-₹512)

---

*TITAN Autonomous Trading System - Technical Documentation*
*Updated: 9 February 2026*
