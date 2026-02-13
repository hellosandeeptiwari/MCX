# TITAN - Autonomous Options Trading System
## Technical Documentation

> **Goal:** ₹2L → ₹20L (10x returns) via fully autonomous intraday options trading on NSE F&O
> **Capital:** ₹2,00,000 | **Mode:** Paper Trading | **Entry Point:** `python autonomous_trader.py`

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [Scoring Systems](#scoring-systems)
5. [Risk Management](#risk-management)
6. [Order Execution](#order-execution)
7. [Safety Gates](#safety-gates)
8. [LLM Agent](#llm-agent)
9. [Market Scanner](#market-scanner)
10. [Configuration](#configuration)
11. [Data Flow](#data-flow)
12. [Thread Safety](#thread-safety)
13. [File Structure](#file-structure)
14. [Known Issues & Fixes](#known-issues--fixes)
15. [Version History](#version-history)

---

## System Overview

TITAN is an **AI-powered autonomous trading agent** that:
- Uses **OpenAI GPT-4o** for reasoning and strategy decisions
- Executes trades via **Zerodha Kite Connect API**
- Specializes in **intraday options** on NSE F&O stocks (18-stock tiered universe + 7 wild-cards)
- Implements **19-gate safety pipeline** to prevent "risk of ruin"
- Runs two concurrent loops: **5-min scan cycle** (GPT agent) + **3-sec monitor daemon** (exits/SL/targets)

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
│                     AUTONOMOUS TRADER (Main Thread)                     │
│                      autonomous_trader.py                               │
│     schedule.every(5).minutes → scan_and_trade() → GPT-4o agent loop   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Market       │───▶│ LLM Agent    │───▶│ Zerodha      │              │
│  │ Scanner      │    │ (GPT-4o)     │    │ Tools        │              │
│  │ ~200 stocks  │    │ 10 tools     │    │ 2 order paths│              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                             │                   │                       │
│                             ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │              19-GATE SAFETY PIPELINE                    │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │           │
│  │  │LLM Arg  │ │Trading  │ │Risk     │ │Position │      │           │
│  │  │Validate │ │Hours    │ │Governor │ │Recon    │      │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │           │
│  │  │Data     │ │Idempotnt│ │Correlat │ │Regime   │      │           │
│  │  │Health   │ │Engine   │ │Guard    │ │Score    │      │           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                  │           │
│  │  │Execution│ │Intraday │ │Micro-   │                  │           │
│  │  │Guard    │ │Scorer   │ │structure│                  │           │
│  │  └─────────┘ └─────────┘ └─────────┘                  │           │
│  └─────────────────────────────────────────────────────────┘           │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │           ZERODHA KITE CONNECT API                      │           │
│  │  (Orders, Positions, Quotes, Option Chains)             │           │
│  └─────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     MONITOR DAEMON (Background Thread)                  │
│              threading.Thread(daemon=True) — every 3 seconds            │
├─────────────────────────────────────────────────────────────────────────┤
│  check_and_update_trades() → SL/Target hits                            │
│  exit_manager.check_exit() → Time stops, trailing, session cutoff      │
│  Candle counter increment every ~300s                                   │
│  Thread-safe via threading.RLock() on paper_positions                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Two Order Paths

| Path | Function | Used For | Safety Gates |
|------|----------|----------|--------------|
| **Equity** | `place_order()` | Non-F&O stocks | 9 gates (reconciliation → data health → idempotent → duplicate → correlation → regime score → validate_trade → execution guard → adaptive sizing) |
| **Options** | `place_option_order()` | F&O stocks | 5 shared gates (trading hours → risk governor → reconciliation → data health → correlation) + option-specific (duplicate underlying → intraday scorer → microstructure → premium caps) |

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
STRONG_BULLISH  (80+)  → Aggressive entry
BULLISH         (60-79) → Standard entry
NEUTRAL         (40-59) → No trade (penalty applied)
BEARISH         (20-39) → Short signals
STRONG_BEARISH  (0-19)  → Aggressive short
```

#### Hysteresis System (Prevents Whipsaw)
| Transition | Threshold | Candles Required |
|------------|-----------|------------------|
| Upgrade to STRONG | ≥ 82 | 2 consecutive |
| Stay STRONG | ≥ 70 | — |
| Downgrade from STRONG | < 70 | Immediate |
| Upgrade to TREND | ≥ 60 | — |
| Downgrade to NEUTRAL | < 50 | Immediate |

#### Shock Override
Force immediate downgrade if:
- VWAP crossed against position + HIGH/EXPLOSIVE volume
- Price moves > 2x ATR against position

---

### 2. Intraday Option Scorer (`options_trader.py :: IntradayOptionScorer`)
**Purpose:** Score intraday signals for options trading decisions (100-point scale)

#### Scoring Weights
| Factor | Points | Description |
|--------|--------|-------------|
| Trend Following | 30 | From TrendFollowingEngine |
| ORB Breakout | 20 | Morning breakout signals (capped to 10 if window overlap) |
| Microstructure | 15 | Spread/depth/OI/fills |
| Volume Regime | 15 | EXPLOSIVE/HIGH/NORMAL/LOW (capped to 8 if window overlap) |
| VWAP Position | 10 | Price vs VWAP alignment (penalty -8 for misalignment) |
| EMA Regime | 10 | Trend structure |
| Acceleration | 10 | Follow-through + range expansion |
| HTF Alignment | 5 | Higher timeframe confirmation |
| RSI Penalty | -3 | Overextension penalty only |
| Caller Direction Bonus | +5 | When GPT direction aligns with signals |
| NEUTRAL Trend Penalty | -5 | Applied when trend is NEUTRAL |
| Chop Zone Penalty | -12 | Applied when in CHOP zone |

#### Trade Tiers (3-Tier System)
| Score | Tier | Strike | Size | Requirements |
|-------|------|--------|------|--------------|
| < 30 | **BLOCK** | — | No trade | — |
| 30–34 | Standard | ATM/ITM | 1.0x | — |
| 35–54 | Standard+ | ATM/ITM | 1.0x | — |
| ≥ 55 | Premium | ATM/ITM | Up to 1.2x | Acceleration ≥ 8 AND Microstructure ≥ 12 |

#### OTM Strike Requirements (Special Case)
OTM strikes are **NOT default** — require ALL of:
- EXPLOSIVE volume regime
- Spread < 0.3%
- Acceleration score ≥ 8/10

---

### 3. Options Microstructure (`options_trader.py :: IntradayOptionScorer`)
**Purpose:** Hard gate for options tradability — prevents losses from poor execution

#### Microstructure Data Structure
```python
@dataclass
class OptionMicrostructure:
    bid: float                    # Best bid price
    ask: float                    # Best ask price
    spread_pct: float             # Spread as % of mid (default 100.0 if unknown)
    bid_qty: int                  # Top-of-book depth (shares)
    ask_qty: int                  # Top-of-book depth (shares)
    open_interest: int            # OI
    option_volume: int            # Today's volume
    oi_volume_ratio: float        # OI / Volume
    partial_fill_rate: float      # 0-1
    cancel_rate: float            # 0-1
    ltp: float                    # Last traded price
```

#### Microstructure Scoring (15 points max)
| Metric | Excellent | Acceptable | Warning | HARD BLOCK |
|--------|-----------|------------|---------|------------|
| Spread % | < 0.3% (+5) | < 0.5% (+3) | 0.5–0.8% (0) | > 0.8% |
| Spread Absolute | — | — | — | > ₹3 (index) / > ₹5 (stock) |
| Top Depth | ≥ 4 lots (+3) | ≥ 2 lots (+1) | — | < 2 lots = BLOCK |
| Open Interest | ≥ 2000 (+3) | ≥ 500 (+1) | — | < 500 = BLOCK |
| Option Volume | ≥ 100 (+2) | — | — | < 100 = penalty |
| Partial Fill Rate | < 0.3 (0) | 0.3–0.5 (-1) | ≥ 0.5 (-2) | — |
| Cancel Rate | < 0.4 (0) | — | ≥ 0.4 (-1) | — |

> **Note:** When `option_data is None` (chain fetch failure), microstructure returns `blocked=True` — does NOT silently pass.

#### Depth Normalization
Kite API returns depth in **shares**, but TITAN normalizes to **lots** using `FNO_LOT_SIZES`:
```python
bid_lots = bid_qty / lot_size  # e.g., 5500 shares / 5500 lot = 1 lot
ask_lots = ask_qty / lot_size
# MIN_DEPTH_LOTS = 2 → need ≥ 2 lots at best bid/ask
```

---

### 4. Exit Manager (`exit_manager.py`)
**Purpose:** First-class exit logic for consistent edge preservation

#### Exit Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `session_cutoff` | 15:15 IST | Exit all positions (aligned with `TRADING_HOURS['no_new_after']`) |
| `time_stop_candles` | 7 | Exit if no progress in 7 candles (~35 min) |
| `time_stop_min_r` | 0.5R | Must achieve at least +0.5R to stay alive |
| `breakeven_trigger_r` | 0.8R | Move SL to entry price |
| `trailing_start_r` | 1.0R | Activate trailing stop |
| `trailing_pct` | 50% | Trail at 50% of max profit from peak |

#### Option Speed Gate (Early Exit)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `option_speed_gate_candles` | 4 | Check after 4 candles (20 min) |
| `option_speed_gate_pct` | 12.0% | Need +12% premium gain |
| `option_speed_gate_max_r` | 0.3R | Only exit if R-multiple < 0.3 too |

**Logic:** After 4 candles, if premium hasn't gained +12% AND R < 0.3, the trade is killed as a slow mover. Both conditions must be true (dual check prevents exiting profitable but slow trades).

**Logging:** Speed gate decisions logged to `speed_gate_log.jsonl` with structured JSON.

#### R-Multiple Tracking
```
R = abs(Entry - Stop Loss)
+0.5R = Time stop threshold
+0.8R = Move stop to break-even
+1.0R = Activate trailing stop
+2.0R = First profit target
+3.0R = Second profit target
```

#### Exit Types
| Type | Trigger | Action |
|------|---------|--------|
| Hard Stop Loss | LTP ≤ SL (BUY) or LTP ≥ SL (SELL) | Immediate exit |
| Time Stop | No +0.5R in 7 candles | Exit position |
| Break-even | +0.8R achieved | Move SL to entry |
| Session Cutoff | 15:15 IST | Exit all positions |
| Trailing Stop | After +1R, trail at 50% of max profit | Lock in profits |
| Target Hit | 2R or 3R target | Take profit |
| Speed Gate | After 4 candles: premium < +12% AND R < 0.3 | Kill slow option |

---

### 5. Regime Scorer (`regime_score.py`)
**Purpose:** Confidence gate for trade quality

#### Minimum Score by Strategy
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
ACTIVE        → Normal trading
COOLDOWN      → 10 min pause after loss (was 15 min, reduced in v1.4)
HALT_TRADING  → Stopped for the day (daily loss limit hit)
CIRCUIT_BREAK → Emergency stop (data/order issues)
```

> **Critical Fix (v1.4):** `is_trading_allowed()` now clears expired cooldowns inline.
> Previously, cooldown expiry was only checked in `can_trade_general()` which was
> never reached because `is_trading_allowed()` returned False first, permanently
> halting the bot after any single loss.

#### Risk Limits (`RiskLimits` dataclass)
| Limit | Value | Description |
|-------|-------|-------------|
| `max_daily_loss_pct` | 3.0% | Max daily loss (₹6,000 on ₹2L) |
| `max_consecutive_losses` | 3 | Halt after 3 consecutive losses |
| `max_trades_per_day` | 8 | Max trades per day |
| `max_symbol_exposure` | 2 | Max positions in same sector |
| `cooldown_minutes` | 10 | Pause after each loss |
| `max_position_pct` | 25.0% | Max 25% of capital in one position |
| `max_total_exposure_pct` | 80.0% | Max 80% of capital deployed |

#### Risk State (`RiskState` dataclass)
| Field | Type | Description |
|-------|------|-------------|
| `system_state` | str | ACTIVE / COOLDOWN / HALT_TRADING / CIRCUIT_BREAK |
| `daily_pnl` | float | Running day P&L in ₹ |
| `daily_pnl_pct` | float | Daily P&L as % of starting capital |
| `trades_today` | int | Number of trades placed today |
| `wins_today` / `losses_today` | int | Win/Loss counters |
| `consecutive_losses` | int | Streak counter (resets on win) |
| `cooldown_until` | str | ISO timestamp when cooldown expires |
| `halt_reason` | str | Why trading was halted |
| `positions_by_sector` | dict | Sector exposure tracking |
| `order_rejections` | int | Broker rejection counter |
| `data_stale_count` | int | Stale data event counter |

**Persistence:** State saved to `risk_state.json` (uses `os.path.dirname(__file__)` for absolute path).

### 2. Correlation Guard (`correlation_guard.py`)
**Purpose:** Prevent hidden overexposure from correlated positions

#### Beta Categories
| Category | Beta Range | Stocks |
|----------|-----------|--------|
| **HIGH_BETA** | > 1.2 | BHARTIARTL, BAJFINANCE, ADANIENT, AXISBANK, ICICIBANK, SBIN, TATASTEEL, HINDALCO, JSWSTEEL, JINDALSTEL, COALINDIA, VEDL, NMDC, ONGC, BPCL, MARUTI, M&M, EICHERMOT |
| **MEDIUM_BETA** | 0.8–1.2 | RELIANCE, HDFCBANK, KOTAKBANK, LT, ULTRACEMCO, GRASIM, TITAN, ASIANPAINT, NTPC, POWERGRID |
| **LOW_BETA** | < 0.8 | INFY, TCS, WIPRO, HCLTECH, TECHM, SUNPHARMA, DRREDDY, CIPLA, HINDUNILVR, ITC, NESTLEIND |
| **INDEX** | — | NIFTY, BANKNIFTY options/futures |

#### Correlation Rules
- Max **2 HIGH_BETA** positions simultaneously
- If holding index options → max **1 additional HIGH_BETA** stock
- Block if real-time correlation exceeds threshold (declared but not yet implemented)

---

## Order Execution

### Equity Order Gate Sequence (`place_order()` — 9 gates)
```
1. Position Reconciliation   → Is local state synced with broker?
2. Data Health Gate          → Is market data fresh and valid?
3. Idempotent Order Engine   → Has this exact order been placed already?
4. Duplicate Symbol Check    → Already holding this symbol?
5. Correlation Guard         → Too many correlated positions?
6. Regime Scorer             → Is trade quality above threshold?
7. Validate Trade            → Do SL/target/risk meet hard rules?
8. Execution Guard           → Is spread acceptable? What order type?
9. Adaptive Position Sizing  → Adjust quantity for slippage/volume
```

### Option Order Gate Sequence (`place_option_order()` — 10 gates)
```
1. F&O Eligibility           → Is underlying in FNO_LOT_SIZES?
2. Trading Hours Check       → Between 09:20 and 15:15?
3. Risk Governor             → Daily loss, consecutive losses, cooldown OK?
4. Position Reconciliation   → Local state synced?
5. Data Health Gate          → Market data fresh?
6. Correlation Guard         → Not too many correlated positions?
7. Duplicate Underlying      → No existing option on same underlying?
8. Intraday Option Scorer    → Score ≥ 30 (BLOCK_THRESHOLD)?
9. Premium Cap Check         → Single trade ≤ ₹1,00,000? Total exposure ≤ ₹2,00,000?
10. Microstructure Gate      → Spread, depth, OI all acceptable?
```

### Execution Guard (`execution_guard.py`)
**Purpose:** Prevent edge leakage through poor fills

#### Spread Checking
| Spread | Action |
|--------|--------|
| < 0.3% | MARKET order OK |
| 0.3–0.5% | LIMIT with 30s timeout |
| > 0.5% (50 bps) | **BLOCK** trading |
| > ₹5 absolute | **BLOCK** trading |

> **Critical Fix (v1.5):** Execution guard failure now **blocks the order** instead of
> silently proceeding with default values. Previously `except Exception: proceed with defaults`
> allowed unvalidated orders through.

#### Order Type by Volume Regime
| Volume Regime | Order Type | Max Slippage |
|---------------|------------|--------------|
| EXPLOSIVE | IOC (immediate-or-cancel) | 0.5% |
| HIGH | MARKET | 0.3% |
| NORMAL | LIMIT (30s timeout) | 0.15% |
| LOW | **BLOCK** (no trade) | 0% |

#### Slippage Tracking
- Records every fill: expected vs actual price, regime, order type
- Maintains rolling log of last 1000 fills in `slippage_log.json`
- Used for statistical analysis of execution quality

### Idempotent Order Engine (`idempotent_order_engine.py`)
**Purpose:** Prevent duplicate orders on reconnect/retry

#### Client Order ID Format
```
Format: SYMBOL_YYYYMMDD_STRATEGY_DIRECTION_HASH
Example: RELIANC_20260207_ORB_B_a1b2c3d4
```

#### Flow
1. Create intent with (symbol, direction, strategy, setup_id)
2. Generate deterministic `client_order_id`
3. Check broker for existing orders with same ID
4. If found: do nothing (idempotent)
5. If not found: place order and record
6. Persisted to `order_idempotency.json`

### Paper Trading Mode
When `paper_mode=True`:
- Orders simulated with LTP as fill price
- **Hard-coded SL/target** for equity: 1% SL, 1.5% target (ignores GPT's values — known P2 issue)
- Options use GPT-specified SL/target from IntradayOptionScorer
- Positions stored in `active_trades.json` (date-partitioned, fresh each day)
- Trade history appended to `trade_history.json`

---

## Safety Gates

### Hard Gates Summary
| Gate | Condition for BLOCK | Module |
|------|---------------------|--------|
| Trading Hours | Before 09:20 or after 15:15 | config.py |
| Risk Governor | Daily loss > 3%, consecutive losses > 3, cooldown active | risk_governor.py |
| Position Recon | Local/broker state mismatch | position_reconciliation.py |
| Data Health | Stale data > 60s, zero volume, NaN indicators | data_health_gate.py |
| Correlation | > 2 HIGH_BETA positions | correlation_guard.py |
| Execution Guard | Spread > 0.5%, LOW volume | execution_guard.py |
| Intraday Scorer | Score < 30 | options_trader.py |
| Microstructure | Spread > 0.8%, depth < 2 lots, OI < 500 | options_trader.py |
| Duplicate | Already holding option on same underlying | zerodha_tools.py |
| Premium Cap | Single trade > ₹1,00,000 or total exposure > ₹2,00,000 | zerodha_tools.py |
| Capital Guard | Single lot premium > 50% of capital | options_trader.py |
| Idempotent | Same order ID already placed | idempotent_order_engine.py |

### Data Health Gate (`data_health_gate.py`)
| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Last tick age | > 60 seconds | Block + increment stale counter |
| Candle gap | > 15 minutes | Block |
| Volume | = 0 | Block (feed glitch) |
| VWAP/EMA/ATR | NaN or 0 | Block (invalid indicators) |
| Stale counter | > 5 consecutive | Halt symbol for day |

### Position Reconciliation (`position_reconciliation.py`)
#### States
```
SYNCED     → Local matches broker
MISMATCH   → Discrepancy found
RECOVERY   → Fixing mismatches, no new trades
FROZEN     → Manual intervention required
INIT       → Startup sync in progress
```
- Check interval: 5–15 seconds
- Pulls broker positions + open orders
- Compares to local state (`active_trades.json`)
- If mismatch: cancel open orders, freeze entries, log for review

---

## LLM Agent (`llm_agent.py`)

### Agent Configuration
| Parameter | Value |
|-----------|-------|
| Model | `gpt-4o` |
| Max tokens per response | 4096 |
| Max iterations per scan | 10 |
| Auto-execute | `True` (in autonomous mode) |
| Truncation recovery | Yes (asks GPT to continue) |

### Available Tools (10 total)
| Tool | Purpose |
|------|---------|
| `get_account_state` | Account margins, positions, P&L, trading permission |
| `get_market_data` | OHLCV + indicators for specified symbols |
| `calculate_position_size` | Safe sizing based on entry/SL/capital |
| `validate_trade` | Check trade against all hard rules |
| `get_portfolio_risk` | Current portfolio risk exposure |
| `place_order` | Equity order placement (9-gate pipeline) |
| `get_options_chain` | Option chain with ATM strikes + recommendation |
| `place_option_order` | F&O order placement (10-gate pipeline) |
| `get_volume_analysis` | Futures OI analysis for EOD momentum plays |
| `get_oi_analysis` | Detailed OI breakdown for single stock |

### Argument Validation (Added v1.5)
Before executing `place_order` or `place_option_order`, the agent validates:

**`place_order` validation:**
- `symbol`: Required, must be string, auto-prepends `NSE:` if missing
- `quantity`: Must be integer 1–50,000
- `side`: Must be `"BUY"` or `"SELL"`

**`place_option_order` validation:**
- `underlying`: Required, auto-prepends `NSE:` if missing
- `direction`: Must be `"BUY"` or `"SELL"`
- `strike_selection`: Must be one of `{ATM, OTM1, OTM2, ITM1, ITM2}` (defaults to ATM)

---

## Market Scanner (`market_scanner.py`)

### Purpose
Scans the entire NSE F&O universe (~200 stocks) every cycle and surfaces up to **7 wild-card** opportunities beyond the fixed 18-stock universe.

### Configuration (`SCANNER_CONFIG`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_wildcards` | 7 | Max wild-card stocks per scan |
| `min_change_pct` | 1.5% | Minimum absolute % change to flag as mover |
| `min_volume_ratio` | 2.0x | Min volume vs session average for surge |
| `breakout_proximity_pct` | 0.3% | Price within 0.3% of day-high = breakout |
| `max_lot_value_lakh` | 15 | Skip if lot × LTP > ₹15 lakh |
| `instrument_cache_ttl` | 86400s | Refresh F&O instrument list once daily |

### Metal Momentum Boost
| Parameter | Value |
|-----------|-------|
| `metal_max_reserve` | 2 slots reserved for metals |
| `metal_momentum_threshold` | 1.5% avg change across metal sector |
| `metal_score_boost` | +20 points for metals on hot metal days |

**Metal Symbols:** TATASTEEL, JSWSTEEL, JINDALSTEL, HINDALCO, HINDZINC, VEDL, NMDC, NATIONALUM, COALINDIA, APLAPOLLO, RATNAMANI, WELCORP

### Exclusion List
Penny stocks, illiquid, circuit-prone: IDEA, SUZLON, YESBANK, PNB, IRFC, NHPC, SAIL, BANKBARODA, UNIONBANK, IDFCFIRSTB, CANBK, IOB, INDIANB, CENTRALBK, UCOBANK, BANKINDIA, RECLTD, PFC, NBCC, IRCTC, ZOMATO, PAYTM, POLICYBZR, DELHIVERY, NYKAA

---

## Configuration (`config.py`)

### Hard Rules
```python
HARD_RULES = {
    "RISK_PER_TRADE": 0.025,      # 2.5% max risk per trade (₹5,000 on ₹2L)
    "MAX_DAILY_LOSS": 0.03,       # 3% max daily loss (₹6,000 on ₹2L)
    "MAX_POSITIONS": 5,           # Max 5 simultaneous positions (prompt guides to 3)
    "STALE_DATA_SECONDS": 60,     # Data freshness threshold
    "API_RATE_LIMIT_MS": 500,     # Min ms between API calls
    "CAPITAL": 200000,            # Starting capital ₹2,00,000
}
```

### Trading Hours
```python
TRADING_HOURS = {
    "start": "09:20",       # 5 mins after market open
    "end": "15:20",         # 10 mins before close
    "no_new_after": "15:15" # No new trades after 3:15 PM (aligned everywhere)
}
```

### Tiered Universe (18 fixed stocks)

**Tier 1 — Always Eligible (10 stocks)**
Tightest spreads, best trending, always scan + always trade options:
```
NSE:SBIN, NSE:HDFCBANK, NSE:ICICIBANK, NSE:AXISBANK, NSE:KOTAKBANK,
NSE:BAJFINANCE, NSE:RELIANCE, NSE:BHARTIARTL, NSE:INFY, NSE:TCS
```

**Tier 2 — Conditional (8 stocks)**
Scanned every cycle, only trade options when `trend_score ≥ 60` (BULLISH/BEARISH required):
```
NSE:TATASTEEL, NSE:JSWSTEEL, NSE:JINDALSTEL, NSE:HINDALCO,
NSE:LT, NSE:MARUTI, NSE:TITAN, NSE:SUNPHARMA
```

### F&O Configuration
```python
FNO_CONFIG = {
    "enabled": True,
    "option_type_on_bullish": "CE",    # Buy Call on bullish
    "option_type_on_bearish": "PE",    # Buy Put on bearish
    "strike_selection": "ATM",         # Default: At-The-Money
    "max_option_premium": 100000,      # ₹1,00,000 max per lot
}
```

### F&O Lot Sizes (Complete)
```python
FNO_LOT_SIZES = {
    # Index
    "NIFTY": 65,  "BANKNIFTY": 30,  "FINNIFTY": 60,  "MIDCPNIFTY": 120,
    # Banks
    "SBIN": 750,  "HDFCBANK": 550,  "ICICIBANK": 700,  "AXISBANK": 625,
    "KOTAKBANK": 2000,  "INDUSINDBK": 450,
    # Financials
    "BAJFINANCE": 750,  "SBILIFE": 750,
    # IT
    "INFY": 400,  "TCS": 175,  "WIPRO": 3000,  "HCLTECH": 350,  "TECHM": 600,
    # Large-cap
    "RELIANCE": 500,  "BHARTIARTL": 475,  "LT": 175,  "MARUTI": 50,  "TITAN": 175,
    # Metals
    "TATASTEEL": 5500,  "JSWSTEEL": 675,  "JINDALSTEL": 625,
    "HINDALCO": 700,  "VEDL": 1150,  "NATIONALUM": 3750,  "NMDC": 6750,
    "SAIL": 4700,  "COALINDIA": 1350,
    # Pharma
    "SUNPHARMA": 350,  "DRREDDY": 125,  "CIPLA": 650,  "DIVISLAB": 100,
    "APOLLOHOSP": 125,
    # FMCG / Consumer
    "HINDUNILVR": 300,  "ITC": 1600,  "NESTLEIND": 25,  "TATACONSUM": 550,
    "ASIANPAINT": 300,
    # Infrastructure / Energy
    "NTPC": 1500,  "POWERGRID": 2700,  "ONGC": 2250,  "BPCL": 1800,
    "TATAPOWER": 1450,
    # Others
    "ADANIENT": 250,  "ADANIPORTS": 1250,  "ETERNAL": 2425,  "MCX": 625,
    "M&M": 350,  "EICHERMOT": 175,  "HEROMOTOCO": 150,  "GRASIM": 475,
    "SHREECEM": 25,  "ULTRACEMCO": 100,  "UPL": 1300,
}
```

### Position Sizing (Options — `OptionPositionSizer`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_premium_per_trade` | 50% of capital (₹1,00,000) | Max premium for single trade |
| `max_premium_per_day` | 100% of capital (₹2,00,000) | Max total premium per day |
| `max_lots_per_trade` | 10 | Maximum lots per single trade |
| `risk_per_trade` | 2.5% of capital (₹5,000) | From `HARD_RULES["RISK_PER_TRADE"]` |

#### Capital Guard (Added v1.5)
If `premium_per_lot > max_premium` (i.e., a single lot costs more than 50% of capital), the trade is **blocked** with `blocked=True`. Previously, `max(1, ...)` would force at least 1 lot through regardless of cost.

#### Three Sizing Methods (minimum wins)
1. **Premium-based:** `lots = max_premium / premium_per_lot`
2. **Risk-based:** `lots = max_loss / (premium_per_lot × 0.5)` (assumes 50% max drawdown)
3. **Delta-adjusted:** Scale down for deep ITM (0.7x) or deep OTM (0.8x)

Final quantity = `min(lots_by_premium, lots_by_risk, lots_by_delta, max_lots_per_trade)`

---

## Data Flow

### Entry Signal Flow
```
Market Data (Zerodha Kite)
        │
        ▼
┌──────────────────┐
│ Market Scanner   │ ──▶ ~200 F&O stocks scanned
│ (every cycle)    │     7 wild-cards surfaced
└────────┬─────────┘
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
│ GPT-4o Agent     │ ──▶ Picks 1-3 best trades
│ (10 tools)       │     per cycle
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ IntradayOption   │ ──▶ 100-pt scoring
│ Scorer           │     + strike + sizing
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Safety Pipeline  │ ──▶ 10-gate validation
│ (all gates)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Execution Guard  │ ──▶ Spread/slippage check
└────────┬─────────┘
         │
         ▼
    PLACE ORDER (Zerodha) or PAPER FILL
```

### Exit Signal Flow
```
Position Monitoring (every 3 seconds)
        │
        ▼
┌──────────────────┐
│ check_and_update │ ──▶ SL/Target hit (from LTP)
│ _trades()        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Exit Manager     │ ──▶ Time stop / Trailing / Speed gate
│ check_exit()     │     / Session cutoff / Break-even
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ update_trade_    │ ──▶ Record result, update risk state
│ status()         │     Save to trade_history.json
└────────┬─────────┘
         │
         ▼
    EXIT ORDER or PAPER CLOSE
```

### Anti-Double-Counting

**Problem:** TrendFollowing and OptionScorer may score the same signal twice:
- VWAP Slope (TrendFollowing) + VWAP Position (OptionScorer)
- ORB Break (TrendFollowing) + ORB Breakout (OptionScorer)
- Volume (TrendFollowing) + Volume Regime (OptionScorer)

**Solution — Window Overlap Detection:**
```python
# If windows overlap > 50%, reduce score by half
if orb_window_minutes <= TREND_WINDOW_MINUTES:  # 15 min
    score_cap = WEIGHTS['orb_breakout'] / 2  # 10 instead of 20
```

**VWAP Position Fix:**
- Reduced weight: 15 → 10 points
- Penalizes misalignment: -8 points
- Only rewards strong alignment: > 0.3% above/below VWAP

---

## Thread Safety

### Problem
Two concurrent threads access `paper_positions`:
- **Main thread:** GPT agent → `place_order()` / `place_option_order()` → `append()`
- **Monitor daemon:** `check_and_update_trades()` → `pop()` on SL/target hits

### Solution (Added v1.5)
```python
self._positions_lock = threading.RLock()  # Reentrant lock
```

**RLock used (not Lock)** because `check_and_update_trades()` calls `update_trade_status()` which also acquires the lock — a regular Lock would deadlock.

#### Protected Operations
| Operation | Method | Lock Scope |
|-----------|--------|------------|
| Add position | `place_order()` | `with self._positions_lock: append + save` |
| Add option position | `place_option_order()` | `with self._positions_lock: append + save` |
| Update/close trade | `update_trade_status()` | `with self._positions_lock: modify + pop + save` |
| Check SL/targets | `check_and_update_trades()` | `with self._positions_lock: iterate + call update_trade_status` |
| Read positions | `is_symbol_in_active_trades()` | Reads only (eventual consistency OK) |

### Persistence Files
| File | Purpose | Thread-Safe |
|------|---------|-------------|
| `active_trades.json` | Current open positions (date-partitioned) | Yes (via RLock) |
| `trade_history.json` | All closed trades (append-only) | Via update_trade_status lock |
| `risk_state.json` | Risk governor state | Single-writer (risk_governor) |
| `exit_manager_state.json` | TradeState persistence | Single-writer (exit_manager) |
| `order_idempotency.json` | Dedup engine state | Single-writer |
| `slippage_log.json` | Execution quality records | Single-writer |
| `speed_gate_log.jsonl` | Speed gate decisions | Append-only logging |
| `data_health_state.json` | Data health state | Single-writer |
| `reconciliation_state.json` | Recon state | Single-writer |

---

## File Structure

```
agentic_trader/
├── autonomous_trader.py       # Main bot: scan loop + monitor daemon
├── config.py                  # Hard rules, universe, trading hours, GPT prompt
├── llm_agent.py               # GPT-4o agent wrapper + 10 tool definitions
├── zerodha_tools.py           # Kite Connect API tools + 2 order paths + paper mode
├── trend_following.py         # Trend detection engine (100-pt scoring)
├── options_trader.py          # Options scoring, Greeks, microstructure, lot sizes
├── market_scanner.py          # F&O universe scanner (~200 stocks → 7 wild-cards)
├── regime_score.py            # Trade quality scoring by strategy type
├── exit_manager.py            # Exit logic: SL/trailing/time stop/speed gate
├── execution_guard.py         # Order quality: spread/slippage/order type
├── risk_governor.py           # Account-level risk: daily loss/cooldown/halt
├── correlation_guard.py       # Beta/sector correlation guard
├── position_reconciliation.py # Broker state sync
├── data_health_gate.py        # Data quality gate
├── idempotent_order_engine.py # Duplicate order prevention
├── dashboard.py               # Web dashboard
├── run.py                     # Entry point
├── .env                       # API keys (not committed)
├── active_trades.json         # Current positions (auto-generated)
├── trade_history.json         # All closed trades (auto-generated)
├── risk_state.json            # Risk governor state (auto-generated)
├── exit_manager_state.json    # Exit manager state (auto-generated)
├── slippage_log.json          # Execution quality log (auto-generated)
├── speed_gate_log.jsonl       # Speed gate decisions (auto-generated)
├── order_idempotency.json     # Dedup engine state (auto-generated)
├── TECHNICAL_DOCUMENTATION.md # This file
└── templates/                 # HTML templates for dashboard
```

---

## Known Issues & Fixes

### Open Issues (P2/P3)
| # | Issue | Severity | Module |
|---|-------|----------|--------|
| P2-1 | Paper mode hard-codes 1% SL / 1.5% target for equity, ignoring GPT values | P2 | zerodha_tools.py |
| P2-2 | `daily_pnl` in autonomous_trader never resets for multi-day runs | P2 | autonomous_trader.py |
| P2-3 | Auto-retry regex can match symbols in GPT's natural language | P2 | autonomous_trader.py |
| P2-4 | Volume ratio biased LOW in early morning (small denominator) | P2 | zerodha_tools.py |
| P2-5 | 3x redundant `get_market_data()` calls per order (~4.5s latency) | P2 | zerodha_tools.py |
| P2-6 | `candles_since_entry` requires external increment — can drift | P2 | exit_manager.py |
| P2-7 | NFO symbols bypass beta mapping in correlation guard | P2 | correlation_guard.py |
| P2-8 | Conversation grows unbounded (no trimming) | P2 | llm_agent.py |
| P3-1 | Dead code: `monitor_positions()`, `_exit_position()`, `self.positions` | P3 | autonomous_trader.py |
| P3-2 | `starting_capital` never resets daily in risk governor | P3 | risk_governor.py |
| P3-3 | Sector names differ: `BANKING` vs `BANK` between modules | P3 | correlation_guard.py / risk_governor.py |
| P3-4 | Division by zero possible if capital=0 in risk governor | P3 | risk_governor.py |
| P3-5 | Paper option exits (speed gate, time stop) never trigger through monitor | P3 | zerodha_tools.py |
| P3-6 | Stale Greeks used for exit decisions (not refreshed after entry) | P3 | options_trader.py |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | — | Initial TrendFollowing + Options |
| 1.1 | — | Added Microstructure (15 pts) |
| 1.2 | — | Replaced RSI with Acceleration |
| 1.3 | Feb 2026 | Risk of ruin fixes (7 changes) |
| 1.4 | 9 Feb 2026 | Scoring fix + critical bug fixes + Day 1 paper trading |
| 1.5 | 9 Feb 2026 | Full code audit (38 issues) + P0/P1 fixes |

### v1.3 Risk of Ruin Fixes
1. OTM + 1.5x → OTM special case + max 1.2x
2. Tightened spread gates (₹3 index, ₹5 stock)
3. Anti-double-counting by window overlap
4. TREND state gate (NEUTRAL blocks)
5. Aggressive requires Accel + Micro dominance
6. VWAP Position reduced + penalty for misalignment
7. Simplified to 2 thresholds (65 standard, 80 premium) → later revised to 30/35/55 in v1.4

### v1.4 Scoring Fix + Bug Fixes (9 Feb 2026)

**Problem:** Bot was not taking any trades. IntradayOptionScorer thresholds were mathematically unreachable when trend=NEUTRAL (-10 penalty) + missing microstructure data. Best realistic score was ~31 but BLOCK_THRESHOLD was 45.

**Scoring Changes (options_trader.py):**
1. BLOCK_THRESHOLD: 45 → **30**
2. STANDARD_THRESHOLD: 45 → **35**
3. PREMIUM_THRESHOLD: 65 → **55**
4. NEUTRAL trend penalty: -10 → **-5**
5. No-trend-data penalty: -10 → **-5**
6. Added `caller_direction` parameter with +5 alignment bonus
7. Fallback direction from caller when signals ambiguous

**Critical Bug Fixes:**
8. **Cooldown expiry not clearing** (`risk_governor.py`): `is_trading_allowed()` checked `system_state == ACTIVE` but never cleared expired cooldowns. Only `can_trade_general()` did, which was unreachable. Bot permanently halted after any single loss.
9. **Option side='SELL' for PE buys** (`zerodha_tools.py`): Options stored with market direction as `side` instead of always `'BUY'`. Caused wrong P&L formula and premature TARGET_HIT exits.
10. **Duplicate option positions** (`zerodha_tools.py`): No check for existing option on same underlying. GPT retried different strikes + retry mechanism created duplicates.
11. **Default capital** (`autonomous_trader.py`): argparse default was ₹10,000 instead of ₹200,000.

**Configuration Changes:**
12. Cooldown: 15 → **10 minutes**
13. Max trades per day: 5 → **8**
14. Max consecutive losses: 2 → **3**
15. Per-trade premium cap: ₹25K → **₹35K**
16. Total option exposure: ₹75K → **₹80K**
17. Market close: 15:15 → **15:20** (EOD exit at 15:15)
18. F&O opportunity filter broadened (VWAP_TREND + MOMENTUM setups)

### v1.5 Full Code Audit + P0/P1 Fixes (9 Feb 2026)

**Code Audit:** 38 issues found across 10 modules, prioritized P0–P3.

**P0 Fixes (Critical — could cause real money loss):**
1. **Option order safety gates** (`zerodha_tools.py`): `place_option_order()` now runs 5 shared safety gates (trading hours, risk governor, reconciliation, data health, correlation guard) — previously had **zero gates**, bypassing ALL safety checks that `place_order()` had.
2. **Lot sizing capital guard** (`options_trader.py`): If single lot premium > 50% of capital, trade is blocked. Previously `max(1, ...)` forced at least 1 lot through regardless of cost.
3. **NIFTY lot size** (`options_trader.py`): Corrected to **65** (Jan–Apr 2026 NSE value). BANKNIFTY: 30, FINNIFTY: 60, MIDCPNIFTY: 120.
4. **Execution guard failure blocks** (`zerodha_tools.py`): Exception in execution guard now returns error and blocks order. Previously `except Exception: proceed with defaults` let unvalidated orders through.
5. **Thread lock on paper_positions** (`zerodha_tools.py`): Added `threading.RLock()` wrapping all mutations (append, pop, save). Prevents race condition between main thread (GPT agent adding positions) and monitor daemon (checking SL/targets every 3s).
6. **LLM tool argument validation** (`llm_agent.py`): Validates symbol format, quantity bounds (1–50,000), side, direction before any order tool executes. Prevents GPT hallucinated values from reaching the order pipeline.

**P1 Fixes (Important — affects trade quality):**
7. **Spread grace zone** (`options_trader.py`): Added `SPREAD_WARN_PCT = 0.8`. Spreads 0.5–0.8% get 0 points (warning zone); > 0.8% hard blocked in scorer. Execution guard still blocks at 0.5% as backstop. Previously `SPREAD_OK_PCT == SPREAD_BLOCK_PCT` (both 0.5%) meant no grace zone.
8. **Config conflicts unified** (`config.py`, `options_trader.py`):
   - `CAPITAL`: 100000 → **200000** (actual paper capital)
   - `MAX_POSITIONS`: 10 → **5** (prompt guides to 3)
   - `max_option_premium`: 5000 → **100000** (₹1 lakh)
   - `risk_per_trade` in options_trader now reads from `HARD_RULES["RISK_PER_TRADE"]` (2.5%) instead of hard-coded 2%
9. **Session cutoff unified** (`exit_manager.py`): Changed 15:20 → **15:15** to match `TRADING_HOURS['no_new_after']` and autonomous_trader.

**Premium Cap Changes:**
- Per-trade premium: ₹35K → **₹1,00,000**
- Total option exposure: ₹80K → **₹2,00,000**
- Position sizer max premium: 5% → **50%** of capital

### Trading Results (Paper — 9 Feb 2026)
- Capital: ₹200,000
- Trades: 9 total (7 options + 2 equity)
- Day P&L: **+₹3,929 (+1.96%)**
- Record: 8W / 0L / 1 SL = 89% win rate
- Best trade: SBIN 1140CE +₹2,100 (+14% premium, held 2h37m)
- Only loss: UNIONBANK equity -₹512 (SL_HIT)

---

*TITAN Autonomous Trading System — Technical Documentation*
*Updated: 9 February 2026 — v1.5*
