"""
TITAN STATE DATABASE — Atomic SQLite Persistence Layer
========================================================
Replaces all JSON state files with a single crash-safe SQLite database
using WAL journal mode. Provides:

  - Atomic writes (no half-written state on crash/power failure)
  - Thread-safe access (SQLite WAL + Python threading.Lock)
  - Queryable history (old data stays, new day = new rows)
  - Drop-in replacement for every _save_state / _load_state pair

Usage:
    from state_db import get_state_db
    db = get_state_db()          # Singleton, one connection per process
    db.save_risk_state({...})    # Replaces atomic_json_save('risk_state.json', ...)
    data = db.load_risk_state()  # Replaces json.load(open('risk_state.json'))

Migration:
    db.migrate_from_json()       # One-time import of all .json files → SQLite

Author: Titan v5.2
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, date
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ─── Database path ────────────────────────────────────────────────────
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, 'titan_state.db')

# ─── Schema version (bump when adding/altering tables) ───────────────
SCHEMA_VERSION = 1


# ======================================================================
#  SCHEMA
# ======================================================================
_SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- ── active_trades (replaces active_trades.json → active_trades list) ──
CREATE TABLE IF NOT EXISTS active_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    trade_json      TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_at_date ON active_trades(date);

-- ── daily_state (replaces active_trades.json → top-level fields) ─────
CREATE TABLE IF NOT EXISTS daily_state (
    date            TEXT    PRIMARY KEY,
    realized_pnl    REAL    DEFAULT 0.0,
    paper_capital   REAL    DEFAULT 0.0,
    last_updated    TEXT
);

-- ── exit_states (replaces exit_manager_state.json) ───────────────────
CREATE TABLE IF NOT EXISTS exit_states (
    symbol          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    state_json      TEXT    NOT NULL,
    last_updated    TEXT,
    PRIMARY KEY (symbol, date)
);

-- ── risk_state (replaces risk_state.json) ────────────────────────────
CREATE TABLE IF NOT EXISTS risk_state (
    date            TEXT    PRIMARY KEY,
    state_json      TEXT    NOT NULL,
    last_updated    TEXT
);

-- ── order_idempotency (replaces order_idempotency.json) ──────────────
CREATE TABLE IF NOT EXISTS order_idempotency (
    order_id        TEXT    PRIMARY KEY,
    date            TEXT    NOT NULL,
    record_json     TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_oi_date ON order_idempotency(date);

-- ── data_health (replaces data_health_state.json) ────────────────────
CREATE TABLE IF NOT EXISTS data_health (
    date            TEXT    PRIMARY KEY,
    stale_counters  TEXT    NOT NULL DEFAULT '{}',
    halted_symbols  TEXT    NOT NULL DEFAULT '[]',
    health_history  TEXT    NOT NULL DEFAULT '[]',
    last_updated    TEXT
);

-- ── slippage_log (replaces slippage_log.json) ────────────────────────
CREATE TABLE IF NOT EXISTS slippage_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record_json     TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);

-- ── reconciliation_state (replaces reconciliation_state.json) ────────
CREATE TABLE IF NOT EXISTS reconciliation_state (
    date            TEXT    PRIMARY KEY,
    state_json      TEXT    NOT NULL,
    last_updated    TEXT
);

-- ── orb_trades (replaces orb_trades_state.json) ──────────────────────
CREATE TABLE IF NOT EXISTS orb_trades (
    date            TEXT    PRIMARY KEY,
    trades_json     TEXT    NOT NULL DEFAULT '{}',
    last_updated    TEXT
);

-- ── scan_decisions (replaces scan_decisions JSONL) ───────────────────
CREATE TABLE IF NOT EXISTS scan_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    decision_json   TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_sd_date   ON scan_decisions(date);
CREATE INDEX IF NOT EXISTS idx_sd_sym    ON scan_decisions(date, symbol);
"""


# ======================================================================
#  TitanStateDB — Singleton
# ======================================================================

class TitanStateDB:
    """Thread-safe SQLite state database with WAL journal mode."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._ensure_schema()

    # ──────────────────────────────────────────────────────────────────
    #  CONNECTION
    # ──────────────────────────────────────────────────────────────────
    def _connect(self):
        """Open SQLite connection with WAL mode and safe pragmas."""
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=10,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")   # Safe with WAL
        self._conn.execute("PRAGMA busy_timeout=5000")     # 5s retry on lock
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            # Set schema version
            cur = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cur.fetchone()
            if row is None:
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ──────────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────────
    def _today(self) -> str:
        return str(date.today())

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    # ==================================================================
    #  1. ACTIVE TRADES  (replaces active_trades.json)
    # ==================================================================

    def save_active_trades(self, positions: list, realized_pnl: float, paper_capital: float):
        """Save the full active trades list + P&L atomically.

        Replaces ALL rows for today — mirrors the old full-file-overwrite pattern.
        """
        today = self._today()
        now = self._now_iso()
        with self._lock:
            c = self._conn
            c.execute("BEGIN IMMEDIATE")
            try:
                # Clear today's rows and rewrite (mirrors JSON overwrite)
                c.execute("DELETE FROM active_trades WHERE date = ?", (today,))
                for pos in positions:
                    c.execute(
                        "INSERT INTO active_trades (date, trade_json) VALUES (?, ?)",
                        (today, json.dumps(pos)),
                    )
                # Upsert daily state
                c.execute(
                    """INSERT INTO daily_state (date, realized_pnl, paper_capital, last_updated)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(date) DO UPDATE SET
                           realized_pnl = excluded.realized_pnl,
                           paper_capital = excluded.paper_capital,
                           last_updated  = excluded.last_updated""",
                    (today, realized_pnl, paper_capital, now),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def load_active_trades(self, date_str: str = None) -> Tuple[list, float, float]:
        """Load active trades for today (or given date).

        Returns: (positions_list, realized_pnl, paper_capital)
        """
        d = date_str or self._today()
        with self._lock:
            c = self._conn
            rows = c.execute(
                "SELECT trade_json FROM active_trades WHERE date = ?", (d,)
            ).fetchall()
            positions = [json.loads(r['trade_json']) for r in rows]

            ds = c.execute(
                "SELECT realized_pnl, paper_capital FROM daily_state WHERE date = ?", (d,)
            ).fetchone()
            pnl = ds['realized_pnl'] if ds else 0.0
            cap = ds['paper_capital'] if ds else 0.0
        return positions, pnl, cap

    # ==================================================================
    #  2. EXIT STATES  (replaces exit_manager_state.json)
    # ==================================================================

    def save_exit_states(self, states_dict: Dict[str, dict]):
        """Save all exit manager trade states atomically.

        Args:
            states_dict: {symbol: state_dict} as produced by TradeState.to_dict()
        """
        today = self._today()
        now = self._now_iso()
        with self._lock:
            c = self._conn
            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute("DELETE FROM exit_states WHERE date = ?", (today,))
                for sym, sd in states_dict.items():
                    c.execute(
                        """INSERT INTO exit_states (symbol, date, state_json, last_updated)
                           VALUES (?, ?, ?, ?)""",
                        (sym, today, json.dumps(sd), now),
                    )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def save_exit_state_single(self, symbol: str, state_dict: dict):
        """Upsert a single exit state (avoids full rewrite on every tick)."""
        today = self._today()
        now = self._now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO exit_states (symbol, date, state_json, last_updated)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(symbol, date) DO UPDATE SET
                       state_json   = excluded.state_json,
                       last_updated = excluded.last_updated""",
                (symbol, today, json.dumps(state_dict), now),
            )
            self._conn.commit()

    def delete_exit_state(self, symbol: str):
        """Remove a single exit state (trade closed)."""
        today = self._today()
        with self._lock:
            self._conn.execute(
                "DELETE FROM exit_states WHERE symbol = ? AND date = ?",
                (symbol, today),
            )
            self._conn.commit()

    def load_exit_states(self, date_str: str = None) -> Dict[str, dict]:
        """Load exit states for today. Returns {symbol: state_dict}."""
        d = date_str or self._today()
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, state_json FROM exit_states WHERE date = ?", (d,)
            ).fetchall()
        return {r['symbol']: json.loads(r['state_json']) for r in rows}

    # ==================================================================
    #  3. RISK STATE  (replaces risk_state.json)
    # ==================================================================

    def save_risk_state(self, state_dict: dict):
        """Save risk governor state (upsert today's row)."""
        today = self._today()
        now = self._now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO risk_state (date, state_json, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       state_json   = excluded.state_json,
                       last_updated = excluded.last_updated""",
                (today, json.dumps(state_dict), now),
            )
            self._conn.commit()

    def load_risk_state(self, date_str: str = None) -> Optional[dict]:
        """Load risk state for today. Returns dict or None."""
        d = date_str or self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM risk_state WHERE date = ?", (d,)
            ).fetchone()
        return json.loads(row['state_json']) if row else None

    # ==================================================================
    #  4. ORDER IDEMPOTENCY  (replaces order_idempotency.json)
    # ==================================================================

    def save_order_record(self, order_id: str, record_dict: dict):
        """Record or update a placed order (upsert by PK)."""
        today = self._today()
        with self._lock:
            self._conn.execute(
                """INSERT INTO order_idempotency
                   (order_id, date, record_json)
                   VALUES (?, ?, ?)
                   ON CONFLICT(order_id) DO UPDATE SET
                       record_json = excluded.record_json""",
                (order_id, today, json.dumps(record_dict)),
            )
            self._conn.commit()

    def is_order_placed(self, order_id: str) -> bool:
        """Check if an order ID was already placed today."""
        today = self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM order_idempotency WHERE order_id = ? AND date = ?",
                (order_id, today),
            ).fetchone()
        return row is not None

    def load_order_records(self, date_str: str = None) -> Tuple[Set[str], Dict[str, dict]]:
        """Load all order records for today.

        Returns: (placed_order_ids_set, {order_id: record_dict})
        """
        d = date_str or self._today()
        with self._lock:
            rows = self._conn.execute(
                "SELECT order_id, record_json FROM order_idempotency WHERE date = ?", (d,)
            ).fetchall()
        ids = set()
        records = {}
        for r in rows:
            oid = r['order_id']
            ids.add(oid)
            records[oid] = json.loads(r['record_json'])
        return ids, records

    def clear_orders_for_day(self, date_str: str = None):
        """Clear all order records for a date (new-day reset)."""
        d = date_str or self._today()
        with self._lock:
            self._conn.execute(
                "DELETE FROM order_idempotency WHERE date = ?", (d,)
            )
            self._conn.commit()

    # ==================================================================
    #  5. DATA HEALTH  (replaces data_health_state.json)
    # ==================================================================

    def save_data_health(self, stale_counters: dict, halted_symbols: list,
                         health_history: list):
        """Save data health state (upsert today's row)."""
        today = self._today()
        now = self._now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO data_health (date, stale_counters, halted_symbols,
                       health_history, last_updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       stale_counters = excluded.stale_counters,
                       halted_symbols = excluded.halted_symbols,
                       health_history = excluded.health_history,
                       last_updated   = excluded.last_updated""",
                (today, json.dumps(stale_counters), json.dumps(halted_symbols),
                 json.dumps(health_history[-100:]), now),
            )
            self._conn.commit()

    def load_data_health(self, date_str: str = None) -> Optional[dict]:
        """Load data health for today. Returns dict with keys or None."""
        d = date_str or self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT stale_counters, halted_symbols, health_history FROM data_health WHERE date = ?",
                (d,),
            ).fetchone()
        if row is None:
            return None
        return {
            'stale_counters': json.loads(row['stale_counters']),
            'halted_symbols': json.loads(row['halted_symbols']),
            'health_history': json.loads(row['health_history']),
        }

    # ==================================================================
    #  6. SLIPPAGE LOG  (replaces slippage_log.json)
    # ==================================================================

    def log_slippage(self, record_dict: dict):
        """Append a slippage record."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO slippage_log (record_json) VALUES (?)",
                (json.dumps(record_dict),),
            )
            self._conn.commit()

    def load_slippage_log(self, limit: int = 1000) -> list:
        """Load recent slippage records (most recent first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_json FROM slippage_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        # Return in chronological order
        return [json.loads(r['record_json']) for r in reversed(rows)]

    # ==================================================================
    #  7. RECONCILIATION STATE  (replaces reconciliation_state.json)
    # ==================================================================

    def save_reconciliation_state(self, state_dict: dict):
        """Save reconciliation state (upsert today)."""
        today = self._today()
        now = self._now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO reconciliation_state (date, state_json, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       state_json   = excluded.state_json,
                       last_updated = excluded.last_updated""",
                (today, json.dumps(state_dict), now),
            )
            self._conn.commit()

    def load_reconciliation_state(self, date_str: str = None) -> Optional[dict]:
        """Load reconciliation state for today."""
        d = date_str or self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM reconciliation_state WHERE date = ?", (d,)
            ).fetchone()
        return json.loads(row['state_json']) if row else None

    # ==================================================================
    #  8. ORB TRADES  (replaces orb_trades_state.json)
    # ==================================================================

    def save_orb_trades(self, trades_dict: dict):
        """Save ORB trade tracking (upsert today)."""
        today = self._today()
        now = self._now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO orb_trades (date, trades_json, last_updated)
                   VALUES (?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       trades_json  = excluded.trades_json,
                       last_updated = excluded.last_updated""",
                (today, json.dumps(trades_dict), now),
            )
            self._conn.commit()

    def load_orb_trades(self, date_str: str = None) -> Optional[dict]:
        """Load ORB trade state for today. Returns trades dict or None."""
        d = date_str or self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT trades_json FROM orb_trades WHERE date = ?", (d,)
            ).fetchone()
        return json.loads(row['trades_json']) if row else None

    # ==================================================================
    #  9. SCAN DECISIONS  (replaces scan_decisions JSONL)
    # ==================================================================

    def log_scan_decision(self, symbol: str, decision_dict: dict):
        """Log a single scan decision."""
        today = self._today()
        with self._lock:
            self._conn.execute(
                "INSERT INTO scan_decisions (date, symbol, decision_json) VALUES (?, ?, ?)",
                (today, symbol, json.dumps(decision_dict)),
            )
            self._conn.commit()

    def get_scan_decisions(self, date_str: str = None, symbol: str = None,
                           limit: int = 5000) -> list:
        """Query scan decisions for today (optionally filtered by symbol)."""
        d = date_str or self._today()
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT decision_json FROM scan_decisions WHERE date = ? AND symbol = ? ORDER BY id LIMIT ?",
                    (d, symbol, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT decision_json FROM scan_decisions WHERE date = ? ORDER BY id LIMIT ?",
                    (d, limit),
                ).fetchall()
        return [json.loads(r['decision_json']) for r in rows]

    # ==================================================================
    #  MIGRATION — JSON → SQLite (one-time import)
    # ==================================================================

    def migrate_from_json(self, base_dir: str = None):
        """
        Import all existing JSON state files into SQLite.

        - Reads each JSON file
        - Inserts into the corresponding table
        - Renames the JSON file to .json.migrated (backup)
        - Idempotent: skips files that don't exist or are already migrated
        """
        base = base_dir or DB_DIR
        migrated = []
        errors = []

        # ── 1. active_trades.json ──
        f = os.path.join(base, 'active_trades.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                d = data.get('date', self._today())
                positions = data.get('active_trades', [])
                pnl = data.get('realized_pnl', 0.0)
                cap = data.get('paper_capital', 0.0)
                self.save_active_trades(positions, pnl, cap)
                os.rename(f, f + '.migrated')
                migrated.append('active_trades.json')
            except Exception as e:
                errors.append(f'active_trades.json: {e}')

        # ── 2. exit_manager_state.json ──
        f = os.path.join(base, 'exit_manager_state.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and 'trade_states' in data:
                    data = data['trade_states']
                if isinstance(data, dict):
                    self.save_exit_states(data)
                os.rename(f, f + '.migrated')
                migrated.append('exit_manager_state.json')
            except Exception as e:
                errors.append(f'exit_manager_state.json: {e}')

        # ── 3. risk_state.json ──
        f = os.path.join(base, 'risk_state.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                self.save_risk_state(data)
                os.rename(f, f + '.migrated')
                migrated.append('risk_state.json')
            except Exception as e:
                errors.append(f'risk_state.json: {e}')

        # ── 4. order_idempotency.json ──
        f = os.path.join(base, 'order_idempotency.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for oid, rec in data.get('order_records', {}).items():
                    self.save_order_record(oid, rec)
                os.rename(f, f + '.migrated')
                migrated.append('order_idempotency.json')
            except Exception as e:
                errors.append(f'order_idempotency.json: {e}')

        # ── 5. data_health_state.json ──
        f = os.path.join(base, 'data_health_state.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                self.save_data_health(
                    data.get('stale_counters', {}),
                    data.get('halted_symbols', []),
                    data.get('health_history', []),
                )
                os.rename(f, f + '.migrated')
                migrated.append('data_health_state.json')
            except Exception as e:
                errors.append(f'data_health_state.json: {e}')

        # ── 6. slippage_log.json ──
        f = os.path.join(base, 'slippage_log.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for rec in data:
                        self.log_slippage(rec)
                os.rename(f, f + '.migrated')
                migrated.append('slippage_log.json')
            except Exception as e:
                errors.append(f'slippage_log.json: {e}')

        # ── 7. reconciliation_state.json ──
        f = os.path.join(base, 'reconciliation_state.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                self.save_reconciliation_state(data)
                os.rename(f, f + '.migrated')
                migrated.append('reconciliation_state.json')
            except Exception as e:
                errors.append(f'reconciliation_state.json: {e}')

        # ── 8. orb_trades_state.json ──
        f = os.path.join(base, 'orb_trades_state.json')
        if os.path.exists(f):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                trades = data.get('trades', {})
                self.save_orb_trades(trades)
                os.rename(f, f + '.migrated')
                migrated.append('orb_trades_state.json')
            except Exception as e:
                errors.append(f'orb_trades_state.json: {e}')

        # ── Summary ──
        print(f"SQLite migration: {len(migrated)} files imported, {len(errors)} errors")
        for m in migrated:
            print(f"  OK  {m}")
        for e in errors:
            print(f"  ERR {e}")
        return migrated, errors

    # ==================================================================
    #  UTILITIES
    # ==================================================================

    def get_daily_realized_pnl(self, date_str: str = None) -> float:
        """Quick lookup of today's realized P&L."""
        d = date_str or self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT realized_pnl FROM daily_state WHERE date = ?", (d,)
            ).fetchone()
        return row['realized_pnl'] if row else 0.0

    def vacuum(self):
        """Reclaim disk space (run occasionally, not during trading hours)."""
        with self._lock:
            self._conn.execute("VACUUM")

    def checkpoint(self):
        """Force WAL checkpoint (flush WAL to main DB)."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# ======================================================================
#  SINGLETON ACCESS
# ======================================================================

_instance: Optional[TitanStateDB] = None
_instance_lock = threading.Lock()


def get_state_db(db_path: str = DB_PATH) -> TitanStateDB:
    """Get the singleton TitanStateDB instance (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TitanStateDB(db_path)
    return _instance
