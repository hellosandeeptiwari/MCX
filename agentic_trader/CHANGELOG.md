# TITAN CHANGELOG

All notable changes to the Titan trading bot are documented here.  
Format: `[YYYY-MM-DD] Category: Description`

---

## 2026-03-09

### Architecture
- **Added SQLite `trades` table** to `state_db.py` — proper queryable trade ledger with indexed columns (date, symbol, source, underlying, status, order_id)
- **Hooked `trade_ledger.py` → SQLite** — every ENTRY/EXIT/CONVERSION event now writes to both JSONL and SQLite `trades` table
- **Created `trade_query.py`** — CLI tool for instant trade analysis (`--week`, `--month`, `--source`, `--underlying`, `--backfill`, `--exit-types`)
- **Created `TITAN_INSTRUCTIONS.md`** — comprehensive architecture doc (strategies, thresholds, exit types, ML models, deployment)
- **Created `CHANGELOG.md`** — this file

### XGB Direction Removal (Major)
- **Removed XGB direction (UP/DOWN) from ALL decision-making:**
  - `_ml_override_allowed`: Gate 4 (direction alignment) removed, max DR score used instead
  - Herd filter: disabled entirely
  - MODEL_TRACKER: ALL_AGREE concept removed — no score bonus, gate floor, or lot multiplier for direction agreement
  - SNIPER: XGB direction alignment + flip + ML_DIRECTION_CONFLICT filter removed
  - TEST_GMM: `require_xgb` direction gate removed
  - ARBTR: XGB direction opposition gate removed
  - TEST_XGB: strategy disabled (returns empty list)
  - Debit spreads: ML UP/DOWN boost replaced with P(move)-based quality
  - Iron Condors: ML FLAT/UP/DOWN quality replaced with P(move)-based quality
  - GPT direct-place: ML direction conflict filter removed entirely
- **Rationale:** Direction predictions were unreliable and caused false rejections. P(move) is more reliable for gating.

### Fixes
- Direction is still predicted and logged (for analysis) but never used to gate, boost, or reject trades

---

## 2026-03-08

### THP Improvements
- THP now skips interception when trailing SL is active (trade already in profit, let trailing SL manage it)
- THP skips when breakeven is already applied

---

## 2026-03-06

### State Persistence
- `state_db.py` created — SQLite WAL-mode database replacing all JSON state files
- Tables: active_trades, daily_state, exit_states, risk_state, order_idempotency, data_health, slippage_log, scan_decisions, live_pnl
- Migration from JSON → SQLite via `db.migrate_from_json()`

---

## 2026-03-02

### Trade Ledger
- `trade_ledger.py` created — append-only JSONL trade event logger
- Events: ENTRY, EXIT, SCAN, CONVERSION
- Daily files: `trade_ledger/trade_ledger_YYYY-MM-DD.jsonl`
- CLI viewer: `python -m trade_ledger [date]`

---

## Pre-March 2026

### Core System
- Autonomous scan loop (5-min cycles + real-time watcher)
- IntradaySignaler + IntradayScorer (0-100 scoring)
- ML pipeline: XGBoost move predictor, direction model, GMM/VAE DR anomaly
- Exit manager with 10+ exit types
- THP (Thesis Hedge Protocol) — SL/time stop → debit spread conversion
- VIX regime scaling (LOW/NORMAL/HIGH/EXTREME/PANIC)
- Correlation guard (sector concentration limits)
- Expiry shield (0DTE/1DTE gamma protection)
- Position sizing tiers (BASE/STANDARD/PREMIUM/ELITE)
- Dashboard (Flask SSE real-time monitoring)
- Zerodha Kite API integration (paper + live mode ready)
- GTT safety nets (server-side SL + target)
- Idempotent order engine

---

*Update this file EVERY TIME you make changes. Include the date, category, and a clear description.*
