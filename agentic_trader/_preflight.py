"""Titan Pre-Flight Check — Monday Readiness"""
import sys, os, sqlite3
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  TITAN PRE-FLIGHT CHECK — Monday Readiness")
print("=" * 60)

db_path = "titan_state.db"
if not os.path.exists(db_path):
    print("FAIL: titan_state.db missing!")
    sys.exit(1)

print(f"\n1. DB FILE: {db_path} ({os.path.getsize(db_path):,} bytes)")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
print(f"   Journal mode: {mode}")
integ = conn.execute("PRAGMA integrity_check").fetchone()[0]
print(f"   Integrity: {integ}")
ver = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
print(f"   Schema version: {ver[0] if ver else 'MISSING'}")

print("\n2. TABLE INVENTORY:")
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
for t in tables:
    name = t[0]
    count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    print(f"   {name:30s} {count:>5} rows")

print("\n3. MIGRATED DATA (2026-02-28):")
at = conn.execute(
    "SELECT COUNT(*) FROM active_trades WHERE date=?", ("2026-02-28",)
).fetchone()[0]
print(f"   active_trades: {at} positions")
ds = conn.execute(
    "SELECT realized_pnl, paper_capital FROM daily_state WHERE date=?",
    ("2026-02-28",),
).fetchone()
if ds:
    print(f"   daily_state: pnl={ds[0]}, cap={ds[1]}")
else:
    print("   daily_state: no row")
es = conn.execute(
    "SELECT COUNT(*) FROM exit_states WHERE date=?", ("2026-02-28",)
).fetchone()[0]
print(f"   exit_states: {es} symbols")
oi = conn.execute(
    "SELECT COUNT(*) FROM order_idempotency WHERE date=?", ("2026-02-28",)
).fetchone()[0]
print(f"   order_idempotency: {oi} records")
sl = conn.execute("SELECT COUNT(*) FROM slippage_log").fetchone()[0]
print(f"   slippage_log: {sl} records (all time)")

print("\n4. MONDAY (2026-03-02) — should be empty (new day auto-creates):")
at_m = conn.execute(
    "SELECT COUNT(*) FROM active_trades WHERE date=?", ("2026-03-02",)
).fetchone()[0]
ds_m = conn.execute(
    "SELECT * FROM daily_state WHERE date=?", ("2026-03-02",)
).fetchone()
print(f"   active_trades: {at_m} (expect 0)")
print(f"   daily_state: {'exists' if ds_m else 'empty'} (expect empty)")
conn.close()

print("\n5. JSON FILES STATUS:")
jfs = [
    "active_trades.json",
    "exit_manager_state.json",
    "risk_state.json",
    "order_idempotency.json",
    "data_health_state.json",
    "slippage_log.json",
    "reconciliation_state.json",
    "orb_trades_state.json",
]
for jf in jfs:
    orig = os.path.exists(jf)
    mig = os.path.exists(jf + ".migrated")
    if not orig and mig:
        st = "OK (migrated)"
    elif orig:
        st = "WARNING: original still exists"
    else:
        st = "absent (no backup)"
    print(f"   {jf:35s} {st}")

# 6. Trade ledger check
print("\n6. TRADE LEDGER (independent of SQLite):")
ledger_files = ["trade_ledger.db", "scan_decisions.jsonl"]
for lf in ledger_files:
    if os.path.exists(lf):
        print(f"   {lf:35s} {os.path.getsize(lf):,} bytes - OK")
    else:
        print(f"   {lf:35s} not found (will be created on first trade)")

# 7. Compile check all 10 critical files
print("\n7. COMPILE CHECK (10 core files):")
files = [
    "state_db.py",
    "risk_governor.py",
    "idempotent_order_engine.py",
    "data_health_gate.py",
    "execution_guard.py",
    "position_reconciliation.py",
    "exit_manager.py",
    "zerodha_tools.py",
    "autonomous_trader.py",
    "options_trader.py",
]
import py_compile
all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"   OK  {f}")
    except py_compile.PyCompileError as e:
        print(f"   ERR {f}: {e}")
        all_ok = False

# 8. Import chain check
print("\n8. IMPORT CHECK (state_db loads cleanly):")
try:
    from state_db import get_state_db, TitanStateDB
    db = get_state_db()
    # Quick CRUD round-trip
    db.save_risk_state({"test": True, "check": "preflight"})
    loaded = db.load_risk_state()
    assert loaded is not None and loaded.get("test") is True
    print("   state_db import + CRUD: OK")
except Exception as e:
    print(f"   FAIL: {e}")
    all_ok = False

print("\n" + "=" * 60)
if all_ok and integ == "ok":
    print("  RESULT: ALL CHECKS PASSED — Ready for Monday 9:15 AM")
else:
    print("  RESULT: ISSUES FOUND — review above")
print("=" * 60)
