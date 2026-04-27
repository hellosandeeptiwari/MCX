#!/bin/bash
# deploy_check.sh — Pre-deploy regression guard
# Run BEFORE scp'ing files to EC2. Catches silent overwrites.
set -e

# Auto-cd to script directory (works whether run from cron, SSH, or interactively)
cd "$(dirname "$0")"

FILE="agentic_trader/autonomous_trader.py"

echo "=== DEPLOY PRE-FLIGHT CHECK ==="

# 1. OI_WATCHER engine must be imported (not inlined)
if ! grep -q "from oi_watcher_engine import OIWatcherEngine" "$FILE"; then
    echo "❌ FATAL: OIWatcherEngine import MISSING — OI code regression detected!"
    echo "   autonomous_trader.py must import from oi_watcher_engine.py"
    exit 1
fi

# 2. Engine must be initialized
if ! grep -q "self._oi_engine = OIWatcherEngine" "$FILE"; then
    echo "❌ FATAL: self._oi_engine init MISSING — OI engine not wired!"
    exit 1
fi

# 3. OI_WATCHER must delegate, not inline
if ! grep -q "_oi_engine.run_watcher_scan" "$FILE"; then
    echo "❌ FATAL: run_watcher_scan delegation MISSING — inline OI code regression!"
    exit 1
fi

# 4. AGGR must delegate, not inline
if ! grep -q "_oi_engine.aggressive_buildup_scan" "$FILE"; then
    echo "❌ FATAL: aggressive_buildup_scan delegation MISSING — inline AGGR regression!"
    exit 1
fi

# 5. Should NOT have inline OI scoring (the old pattern)
if grep -q "# (A) Participant quality: GRANULAR writer/buyer ratio" "$FILE"; then
    echo "❌ FATAL: Inline OI factor scoring detected in autonomous_trader.py!"
    echo "   This code belongs in oi_watcher_engine.py, not here."
    exit 1
fi

# 6. Line count sanity — if file is >13000 lines, OI code probably got re-inlined
LINE_COUNT=$(wc -l < "$FILE")
if [ "$LINE_COUNT" -gt 13000 ]; then
    echo "⚠️  WARNING: autonomous_trader.py is $LINE_COUNT lines (expected <13000)"
    echo "   Check if OI_WATCHER code was accidentally re-inlined."
fi

echo "✅ All pre-flight checks passed ($LINE_COUNT lines)"
echo "   OIWatcherEngine: imported + initialized + delegated"
echo "   No inline OI scoring detected"

# ============================================================
# SECTION 2: ENGINE MODULE EXISTENCE CHECK
# ============================================================
echo ""
echo "=== ENGINE MODULE EXISTENCE CHECK ==="
FAIL=0

REQUIRED_ENGINES="oi_watcher_engine.py commodities_trader.py sniper_strategies.py risk_governor.py market_scanner.py trade_ledger.py state_db.py data_health_gate.py exit_manager.py execution_guard.py settings_manager.py thesis_validator.py correlation_guard.py llm_agent.py gmm_data_collector.py dhan_oi_fetcher.py dhan_risk_tools.py greeks_engine.py watcher_pipeline.py watcher_exit_engine.py capital_swap_engine.py config.py"

for engine in $REQUIRED_ENGINES; do
    if [ ! -f "agentic_trader/$engine" ]; then
        echo "❌ MISSING: agentic_trader/$engine"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 1 ]; then
    echo "❌ FATAL: Missing engine modules!"
    exit 1
fi
echo "✅ All 22 engine modules present"

# ============================================================
# SECTION 3: CRITICAL IMPORT VERIFICATION
# ============================================================
echo ""
echo "=== CRITICAL IMPORT CHECKS ==="

# commodities_trader must be imported
if ! grep -q "from commodities_trader import" "$FILE"; then
    echo "❌ FATAL: commodities_trader not imported!"
    exit 1
fi

# sniper_strategies must be imported
if ! grep -q "from sniper_strategies import\|import sniper_strategies" "$FILE"; then
    echo "❌ FATAL: sniper_strategies not imported!"
    exit 1
fi

# risk_governor must be imported
if ! grep -q "from risk_governor import" "$FILE"; then
    echo "❌ FATAL: risk_governor not imported!"
    exit 1
fi

# market_scanner must be imported
if ! grep -q "from market_scanner import" "$FILE"; then
    echo "❌ FATAL: market_scanner not imported!"
    exit 1
fi

# settings_manager must be imported
if ! grep -q "from settings_manager import" "$FILE"; then
    echo "❌ FATAL: settings_manager not imported!"
    exit 1
fi

echo "✅ All critical imports verified"

# ============================================================
# SECTION 4: FINGERPRINT CHECK (detect drift)
# ============================================================
echo ""
echo "=== FILE FINGERPRINTS ==="
for engine in $REQUIRED_ENGINES; do
    LINES=$(wc -l < "agentic_trader/$engine")
    MD5=$(md5sum "agentic_trader/$engine" | cut -c1-12)
    echo "  $engine: ${LINES}L  md5=$MD5"
done
TRADER_LINES=$(wc -l < "$FILE")
TRADER_MD5=$(md5sum "$FILE" | cut -c1-12)
echo "  autonomous_trader.py: ${TRADER_LINES}L  md5=$TRADER_MD5"

echo ""
echo "=== DEPLOY CHECK COMPLETE ==="
echo "   No inline OI scoring detected"
