# TITAN API AUDIT REPORT — Unused Kite & DhanHQ Capabilities
**Generated: February 15, 2026**

---

## TASK 1: Kite API Methods Currently Used

| # | Method | File(s) | Lines |
|---|--------|---------|-------|
| 1 | `kite.instruments(exchange)` | zerodha_tools.py, options_trader.py, kite_ticker.py, diagnose_symbols.py, candle_replay.py, market_scanner.py, backtest_trend_following.py, ml_models/data_fetcher.py, diagnose_data_feeds.py, test_data_limits.py | 206, 1206, 1300, 1588, 4096, 4217; 2212; 98, 105, 502; 31, 71, 170; 30; 138; 735; 110; 167, 200; 139 |
| 2 | `kite.place_order(**kwargs)` | zerodha_tools.py, options_trader.py | 279, 285; 2738, 2744 |
| 3 | `kite.ltp(symbols)` | zerodha_tools.py, autonomous_trader.py, options_trader.py, kite_ticker.py, diagnose_data_feeds.py | 698, 2866, 4092; 752, 754, 921, 924, 931, 937, 976, 1280; 2397, 5173; 248, 281; 142, 526 |
| 4 | `kite.quote(symbols)` | zerodha_tools.py, options_trader.py, kite_ticker.py, diagnose_symbols.py, market_scanner.py, diagnose_data_feeds.py | 1016, 1224, 1233, 1287, 1323, 1403, 2760, 4126; 2421, 2941; 314, 339; 31; 207; 152 |
| 5 | `kite.historical_data(...)` | zerodha_tools.py, diagnose_symbols.py, candle_replay.py, backtest_trend_following.py, ml_models/data_fetcher.py, diagnose_data_feeds.py, test_data_limits.py | 1608, 1654; 119, 152; 56, 61; 752; 157, 303; 177, 182; 149 |
| 6 | `kite.margins()` | zerodha_tools.py | 906 |
| 7 | `kite.positions()` | zerodha_tools.py, position_reconciliation.py, diagnose_data_feeds.py | 912; 234; 210 |
| 8 | `kite.orders()` | zerodha_tools.py, position_reconciliation.py, diagnose_data_feeds.py | 920, 2566; 245, 419; 211 |
| 9 | `kite.place_gtt(...)` | zerodha_tools.py | 395 |
| 10 | `kite.get_gtts()` | zerodha_tools.py | 459 |
| 11 | `kite.delete_gtt(id)` | zerodha_tools.py | 439, 478 |
| 12 | `kite.modify_order(...)` | autonomous_trader.py | 1221 |
| 13 | `kite.cancel_order(...)` | position_reconciliation.py | 423 |
| 14 | `kite.basket_margins(...)` | options_trader.py | 2838 |
| 15 | `kite.profile()` | zerodha_tools.py, quick_auth.py, _check_kite.py, _auth_kite.py, refresh_token.py, diagnose_data_feeds.py | 795, 843; 50; 37, 54; 38; 66; 132 |
| 16 | `kite.login_url()` | zerodha_tools.py, quick_auth.py, _check_kite.py | 816; 11; 65 |
| 17 | `kite.generate_session(...)` | zerodha_tools.py, quick_auth.py, _auth_kite.py, refresh_token.py | 827; 29; 12; 31 |
| 18 | `kite.set_access_token(...)` | zerodha_tools.py, diagnose_symbols.py, quick_auth.py, candle_replay.py, _check_kite.py, _auth_kite.py, refresh_token.py, diagnose_data_feeds.py, ml_models/data_fetcher.py | Multiple |

**Constants used:** `VARIETY_REGULAR`, `TRANSACTION_TYPE_BUY/SELL`, `PRODUCT_MIS`, `ORDER_TYPE_MARKET/LIMIT/SLM`, `VALIDITY_DAY/IOC`, `GTT_TYPE_OCO`

---

## TASK 2: DhanHQ Endpoints Currently Used

| # | Endpoint | File(s) | Purpose |
|---|----------|---------|---------|
| 1 | `POST /v2/optionchain` | dhan_oi_fetcher.py | Full option chain with OI, Greeks, IV, bid/ask |
| 2 | `POST /v2/optionchain/expirylist` | dhan_oi_fetcher.py | Active expiry dates |
| 3 | `POST /v2/killswitch?killSwitchStatus=ACTIVATE` | dhan_risk_tools.py:146 | Emergency trading halt |
| 4 | `POST /v2/killswitch?killSwitchStatus=DEACTIVATE` | dhan_risk_tools.py:155 | Re-enable trading |
| 5 | `GET /v2/killswitch` | dhan_risk_tools.py:164 | Check kill switch status |
| 6 | `GET /v2/fundlimit` | dhan_risk_tools.py:400, diagnose_data_feeds.py:330 | Available balance check |
| 7 | `POST /v2/charts/historical` | dhan_risk_tools.py:435, dhan_futures_oi.py:261,266,372, test_data_limits.py, test_dhan_oi_hist.py, diagnose_data_feeds.py:371 | Historical OHLCV+OI candles |
| 8 | `POST /v2/charts/intraday` | dhan_risk_tools.py:483, dhan_futures_oi.py:325,330, test_dhan_oi_hist.py:14 | Intraday candles with OI |
| 9 | `POST /v2/marketfeed/ltp` | diagnose_data_feeds.py:538 | LTP comparison only (diagnostic) |

**DhanHQ methods used via `DhanRiskTools`:** `activate_kill_switch`, `deactivate_kill_switch`, `get_kill_switch_status`, `set_pnl_exit`, `stop_pnl_exit`, `get_pnl_exit_status`, `check_margin`, `check_multi_margin`, `check_ic_margin`, `check_spread_margin`, `get_fund_limit`, `fetch_daily_candles`, `fetch_intraday_candles`, `setup_daily_safety`, `emergency_halt`

---

## TASK 3: All Available KiteConnect Methods (Library)

```
basket_order_margins    cancel_mf_order        cancel_mf_sip          cancel_order
convert_position        delete_gtt             exit_order             generate_session
get_auction_instruments get_gtt                get_gtts               get_virtual_contract_note
historical_data         holdings               instruments            invalidate_access_token
invalidate_refresh_token login_url             ltp                    margins
mf_holdings             mf_instruments         mf_orders              mf_sips
modify_gtt              modify_mf_sip          modify_order           ohlc
order_history           order_margins          order_trades           orders
place_gtt               place_mf_order         place_mf_sip           place_order
positions               profile                quote                  renew_access_token
set_access_token        set_session_expiry_hook trades                 trigger_range
```

**Constants:** `EXCHANGE_*`, `GTT_*`, `MARGIN_*`, `ORDER_TYPE_*`, `POSITION_TYPE_*`, `PRODUCT_*`, `STATUS_*`, `TRANSACTION_TYPE_*`, `VALIDITY_*`, `VARIETY_*` (including `VARIETY_AMO`, `VARIETY_AUCTION`, `VARIETY_CO`, `VARIETY_ICEBERG`)

---

## TASK 4: DhanHQ API Documentation References in Code

Found in:
- `dhan_oi_fetcher.py` lines 13-16: Documents `/v2/optionchain`, `/v2/optionchain/expirylist`, `/v2/marketfeed/quote`
- `dhan_risk_tools.py` lines 1-12: Documents Kill Switch, P&L Auto-Exit, Margin Calculator, Fund Limit, Historical OI
- `dhan_futures_oi.py` lines 2-13: Documents futures OI backfill from DhanHQ

---

## TASK 5: High-Value Feature Audit — 14 Items

### 1. `kite.margins()` — Real-time margin utilization
- **Status: YES — USED**
- **Where:** [zerodha_tools.py](agentic_trader/zerodha_tools.py#L906) — called in `get_account_status()` to report equity/commodity margins
- **Gap:** Only called in account status. NOT used for pre-trade margin validation. Kite margin data is NOT checked before placing orders. DhanHQ's fund_limit is used instead.

### 2. `kite.holdings()` — Portfolio holdings
- **Status: NO — NOT USED**
- **Where:** Never called anywhere in the codebase
- **Value it could add:**
  - **Cover order hedging:** Check existing equity holdings that could serve as margin cover for options selling
  - **Collateral awareness:** Holdings pledged as margin affect total available capital
  - **Portfolio delta management:** Know total portfolio exposure including overnight positions
  - **Prevent duplicate entries:** Avoid buying into existing long-term positions

### 3. `kite.trades()` — Executed trades list
- **Status: NO — NOT USED**
- **Where:** Never called anywhere
- **Value it could add:**
  - **Fill price verification:** Compare actual fill price vs expected to detect slippage in real-time
  - **Partial fill detection:** Know instantly when orders are partially filled
  - **Execution quality analytics:** Track spread between trigger price and fill price across strategies
  - **Audit trail for P&L:** More accurate than relying on order status alone — trades show exact fill timestamps

### 4. `kite.order_history(order_id)` — Order audit trail
- **Status: NO — NOT USED**
- **Where:** Never called anywhere
- **Value it could add:**
  - **Debug rejected orders:** See exact rejection reason (margin, quantity, price band)
  - **Track order lifecycle:** OPEN → TRIGGER_PENDING → COMPLETE shows SL trigger behavior
  - **Latency measurement:** Compare `order_timestamp` vs `exchange_timestamp` for execution speed
  - **Modify audit:** See history of price modifications on trailing stops

### 5. `kite.order_margins(params)` — Pre-order margin check
- **Status: NO — NOT USED**  
- **Where:** Never called. DhanHQ's margin calculator is used instead.
- **Value it could add:**
  - **Kite-native margin check:** More accurate than DhanHQ for Kite-executed orders since margin rules differ by broker
  - **Single-leg pre-validation:** Before placing individual orders, check if margin is sufficient
  - **Avoid rejected orders:** Currently orders can fail due to insufficient margin — this would prevent that
  - **Real-time margin impact:** Unlike DhanHQ estimate, this uses Kite's own SPAN engine

### 6. `kite.basket_order_margins(orders)` — Basket margin with hedging benefit
- **Status: YES — USED**
- **Where:** [options_trader.py](agentic_trader/options_trader.py#L2838) — used for spread/IC margin calculation with `consider_positions=True`
- **Gap:** Only used in `options_trader.py`. Not used in `zerodha_tools.py` where naked spreads and iron condors are also placed.

### 7. `kite.trigger_range(params)` — SL trigger price range
- **Status: NO — NOT USED**
- **Where:** Never called anywhere
- **Value it could add:**
  - **Valid SL placement:** Know the allowed trigger price range before setting stop-loss — avoids rejections due to out-of-range triggers
  - **Dynamic SL optimization:** Set SL at the optimal allowed boundary rather than arbitrary percentages
  - **Options SL accuracy:** Options have wide SL ranges — this tells you exact valid range per instrument

### 8. GTT Orders (place_gtt, get_gtts, delete_gtt)
- **Status: YES — USED**
- **Where:** [zerodha_tools.py](agentic_trader/zerodha_tools.py#L395) — `place_gtt()` with OCO type, [L459](agentic_trader/zerodha_tools.py#L459) — `get_gtts()`, [L439](agentic_trader/zerodha_tools.py#L439) / [L478](agentic_trader/zerodha_tools.py#L478) — `delete_gtt()`
- **Gap:** `modify_gtt()` and `get_gtt(single_id)` are available but NOT used. Cannot update GTT targets/SLs without deleting and re-creating. `GTT_TYPE_SINGLE` is not used — only `OCO`.

### 9. Market Depth (5-level bid/ask from KiteTicker FULL mode)
- **Status: PARTIAL — Available but NOT used to its potential**
- **Where:** 
  - [kite_ticker.py](agentic_trader/kite_ticker.py#L429) — WebSocket reconnect defaults to `'quote'` mode (no depth)
  - [kite_ticker.py](agentic_trader/kite_ticker.py#L203) — `subscribe_symbols()` supports `'full'` mode parameter
  - [options_trader.py](agentic_trader/options_trader.py#L183) — `MIN_DEPTH_LOTS` constant exists, `bid_qty`/`ask_qty` referenced at L1413
- **Gap:** 
  - **WebSocket defaults to `'quote'` mode** on connect/reconnect (L429), missing all 5-level depth data
  - No code ever calls `subscribe_symbols(symbols, mode='full')`
  - Market depth data (5 levels × bid/ask × price + qty + orders) is never parsed from ticks
  - The `options_trader.py` depth check uses `kite.quote()` REST data (only top-of-book), NOT WebSocket full depth
- **Value it could add:**
  - **Liquidity scoring:** See if there's real depth behind the best bid/ask before entering illiquid options
  - **Smart order routing:** Place limit orders inside the spread based on visible depth
  - **Impact cost estimation:** Calculate what price you'll actually get for 5-10 lots
  - **Spoofing detection:** Sudden depth disappearance signals fake orders

### 10. DhanHQ `/v2/marketfeed/quote` — Individual instrument depth
- **Status: NO — NOT USED**
- **Where:** Documented in [dhan_oi_fetcher.py](agentic_trader/dhan_oi_fetcher.py#L16) as available but never called
- **Value it could add:**
  - **Cross-broker depth validation:** Compare DhanHQ depth vs Kite depth for same instrument
  - **OI + depth in one call:** Single endpoint gives OI, depth, Greeks for any instrument
  - **Pre-trade depth check for options:** Before entering, verify the specific strike has adequate market depth on DhanHQ's feed

### 11. DhanHQ `/v2/orders` — Mirror order placement
- **Status: NO — NOT USED**
- **Where:** No `/v2/orders` endpoint is called for order placement — DhanHQ is data-only
- **Value it could add:**
  - **Redundant execution:** If Kite API is down, fall back to DhanHQ for order placement
  - **Multi-broker execution:** Split large orders across brokers for better fills
  - **Arbitrage detection:** Compare Kite vs DhanHQ fill prices for same instrument
  - **Disaster recovery:** If Kite session expires mid-trade, DhanHQ can close positions

### 12. DhanHQ `/v2/positions` — Position tracking
- **Status: NO — NOT USED**
- **Where:** Not called anywhere
- **Value it could add:**
  - **Cross-broker reconciliation:** Validate Kite positions agree with DhanHQ records
  - **Position monitoring backup:** If Kite positions API fails, DhanHQ provides backup
  - **Greek aggregation:** DhanHQ positions may include Greeks for options positions

### 13. DhanHQ Forever Orders (GTC/GTD)
- **Status: NO — NOT USED**
- **Where:** Not referenced anywhere
- **Value it could add:**
  - **Multi-day stop-losses:** GTT orders on Kite are good-till-triggered but only for 1 year. DhanHQ GTC/GTD adds another layer
  - **Overnight protection:** Set GTC orders on DhanHQ as backup SL if Titan isn't running overnight
  - **Swing trade support:** If Titan ever moves to positional trades, GTC orders would hold SLs across days

### 14. DhanHQ `/v2/charts/intraday` — Intraday candles with OI
- **Status: YES — USED**
- **Where:** [dhan_risk_tools.py](agentic_trader/dhan_risk_tools.py#L483), [dhan_futures_oi.py](agentic_trader/dhan_futures_oi.py#L325), [test_dhan_oi_hist.py](agentic_trader/test_dhan_oi_hist.py#L14)
- **Gap:** Used for historical backfill only, NOT for real-time intraday OI tracking during live trading. Could be polled every 5 minutes to get live futures OI changes.

---

## SUMMARY: Unused KiteConnect Methods

| Method | Priority | Potential Impact |
|--------|----------|-----------------|
| `kite.holdings()` | HIGH | Collateral & portfolio delta awareness |
| `kite.trades()` | HIGH | Fill verification, slippage detection, partial fills |
| `kite.order_history()` | HIGH | Rejection debugging, latency measurement |
| `kite.order_margins()` | HIGH | Pre-trade margin validation (prevents rejections) |
| `kite.trigger_range()` | MEDIUM | Valid SL range to avoid trigger rejections |
| `kite.modify_gtt()` | MEDIUM | Update GTT without delete+recreate cycle |
| `kite.get_gtt(id)` | LOW | Inspect single GTT details |
| `kite.ohlc()` | LOW | Lightweight OHLC without full quote overhead |
| `kite.convert_position()` | LOW | Convert MIS↔NRML/CNC intraday |
| `kite.exit_order()` | LOW | Simplified exit for bracket/cover orders |
| `kite.order_trades()` | MEDIUM | Trades for a specific order (partial fill audit) |
| `kite.get_virtual_contract_note()` | LOW | Brokerage/tax estimate for trade |
| `kite.get_auction_instruments()` | LOW | Auction session instruments |
| `kite.renew_access_token()` | MEDIUM | Token renewal without re-login |
| `kite.invalidate_access_token()` | LOW | Security — logout old sessions |
| `kite.set_session_expiry_hook()` | MEDIUM | Auto-handler for token expiry |
| `kite.basket_order_margins()` | LOW | Available but `basket_margins()` already used |
| **Varieties:** `VARIETY_AMO` | MEDIUM | After-market orders for next-day execution |
| **Varieties:** `VARIETY_ICEBERG` | HIGH | Large orders split automatically by exchange |
| **Varieties:** `VARIETY_CO` | LOW | Cover orders (deprecated for new use) |
| **Products:** `PRODUCT_NRML` | MEDIUM | Positional trades (currently MIS-only) |
| **Products:** `PRODUCT_CNC` | LOW | Cash-and-carry for equity delivery |

## SUMMARY: Unused DhanHQ Capabilities

| Endpoint/Feature | Priority | Potential Impact |
|-------------------|----------|-----------------|
| `/v2/marketfeed/quote` | HIGH | Real-time depth + OI per instrument |
| `/v2/orders` (placement) | MEDIUM | Backup execution engine / multi-broker |
| `/v2/positions` | MEDIUM | Cross-broker position reconciliation |
| Forever Orders (GTC/GTD) | LOW | Multi-day stop-losses as backup |
| Live intraday OI polling | HIGH | Real-time futures OI trend during trading |

---

## TOP 5 RECOMMENDATIONS (Highest ROI)

1. **Add `kite.trades()` + `kite.order_history()`** — Detect slippage & partial fills in real-time. Currently Titan has no post-execution verification.

2. **Add `kite.order_margins()` pre-check** — Before every order, validate margin is sufficient. Prevents rejected orders that currently cause silent failures.

3. **Switch KiteTicker to FULL mode for active options** — Get 5-level market depth for liquidity scoring. Currently using `'quote'` mode which has no depth data.

4. **Add `kite.holdings()` check at startup** — Know collateral value, pledged securities, and existing portfolio exposure before sizing trades.

5. **Use `VARIETY_ICEBERG`** — For large orders (>freeze qty), instead of manual autoslice hack (`_routes["orders.place"]`), use the native Kite iceberg variety which splits orders at exchange level.
