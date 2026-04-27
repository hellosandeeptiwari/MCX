"""
FULL INTEGRITY AUDIT: Check every engine module is actually USED (not re-inlined)
Generates a comprehensive report.
"""
import os, re

BASE = '/home/ubuntu/titan/agentic_trader'
TRADER = os.path.join(BASE, 'autonomous_trader.py')

with open(TRADER, encoding='utf-8') as f:
    trader_code = f.read()
trader_lines = trader_code.splitlines()
total_lines = len(trader_lines)

# ===== 1. Engine modules that MUST be imported and delegated =====
CRITICAL_ENGINES = {
    'oi_watcher_engine': {
        'class': 'OIWatcherEngine',
        'methods': ['run_watcher_scan', 'aggressive_buildup_scan'],
        'anti_patterns': ['# (A) Participant quality: GRANULAR', 'oi_conviction_score +='],
        'desc': 'OI 13-factor conviction engine'
    },

    'sniper_strategies': {
        'class': 'SniperStrategies',
        'methods': ['evaluate_orb', 'evaluate_vwap_rejection', 'evaluate_gap_fill'],
        'anti_patterns': [],
        'desc': 'Sniper entry strategies'
    },
    'risk_governor': {
        'class': None,
        'factory': 'get_risk_governor',
        'methods': ['check_system_state', 'daily_pnl', 'can_trade'],
        'anti_patterns': [],
        'desc': 'Risk management governor'
    },
    'market_scanner': {
        'class': None,
        'factory': 'get_market_scanner',
        'methods': ['scan', 'get_market_context'],
        'anti_patterns': [],
        'desc': 'Market scanning engine'
    },
    'trade_ledger': {
        'class': None,
        'factory': 'get_trade_ledger',
        'methods': ['log_trade', 'get_today_trades'],
        'anti_patterns': [],
        'desc': 'Trade logging ledger'
    },
    'state_db': {
        'class': None,
        'factory': 'get_state_db',
        'methods': ['get', 'set', 'save'],
        'anti_patterns': [],
        'desc': 'Persistent state database'
    },
    'data_health_gate': {
        'class': None,
        'factory': 'get_data_health_gate',
        'methods': ['check', 'is_healthy'],
        'anti_patterns': [],
        'desc': 'Data quality gatekeeper'
    },
    'exit_manager': {
        'class': None,
        'methods': ['check_exits', 'manage_trailing_stops'],
        'anti_patterns': [],
        'desc': 'Position exit management'
    },
    'execution_guard': {
        'class': None,
        'methods': ['validate_order', 'check_duplicate'],
        'anti_patterns': [],
        'desc': 'Order execution safety'
    },
    'settings_manager': {
        'class': None,
        'methods': ['sync_defaults'],
        'anti_patterns': [],
        'desc': 'Dynamic settings management'
    },
    'thesis_validator': {
        'class': None,
        'methods': ['should_hedge_instead_of_exit'],
        'anti_patterns': [],
        'desc': 'Trade thesis validation'
    },
    'watcher_pipeline': {
        'class': None,
        'methods': [],
        'anti_patterns': [],
        'desc': 'Watcher trade pipeline'
    },
    'watcher_exit_engine': {
        'class': None,
        'methods': [],
        'anti_patterns': [],
        'desc': 'Watcher exit engine'
    },
    'capital_swap_engine': {
        'class': None,
        'methods': [],
        'anti_patterns': [],
        'desc': 'Capital swap logic'
    },
    'greeks_engine': {
        'class': None,
        'methods': [],
        'anti_patterns': [],
        'desc': 'Options greeks calculator'
    },
    'correlation_guard': {
        'class': None,
        'methods': [],
        'anti_patterns': [],
        'desc': 'Correlation-based position guard'
    },
    'llm_agent': {
        'class': 'TradingAgent',
        'methods': [],
        'anti_patterns': [],
        'desc': 'LLM-based trading agent'
    },
    'gmm_data_collector': {
        'class': 'GMMDataCollector',
        'methods': [],
        'anti_patterns': [],
        'desc': 'GMM data collection'
    },
    'dhan_oi_fetcher': {
        'class': 'DhanOIFetcher',
        'methods': [],
        'anti_patterns': [],
        'desc': 'Dhan OI data fetcher'
    },
    'dhan_risk_tools': {
        'class': None,
        'factory': 'get_dhan_risk_tools',
        'methods': [],
        'anti_patterns': [],
        'desc': 'Dhan risk management tools'
    },
}

print("=" * 70)
print(f"TITAN v5 FULL INTEGRITY AUDIT")
print(f"autonomous_trader.py: {total_lines} lines")
print("=" * 70)

issues = []
warnings = []
ok_count = 0

for module, info in CRITICAL_ENGINES.items():
    fname = f"{module}.py"
    fpath = os.path.join(BASE, fname)
    
    # Check file exists
    if not os.path.exists(fpath):
        issues.append(f"MISSING FILE: {fname} — {info['desc']}")
        continue
    
    with open(fpath, encoding='utf-8') as f:
        mod_lines = len(f.readlines())
    
    # Check imported in autonomous_trader.py
    import_patterns = [
        f'from {module} import',
        f'import {module}'
    ]
    imported = any(p in trader_code for p in import_patterns)
    
    if not imported:
        issues.append(f"NOT IMPORTED: {module} ({mod_lines}L) — {info['desc']} exists but is NEVER imported!")
        continue
    
    # Check anti-patterns (code that should NOT be in autonomous_trader.py)
    for ap in info.get('anti_patterns', []):
        if ap in trader_code:
            issues.append(f"INLINED CODE: {module} — found anti-pattern '{ap[:50]}' in autonomous_trader.py. This code should only be in {fname}!")
    
    # Check class/factory usage
    cls = info.get('class')
    factory = info.get('factory')
    if cls:
        if cls not in trader_code:
            warnings.append(f"CLASS NOT USED: {module}.{cls} imported but never instantiated")
    if factory:
        if factory not in trader_code:
            warnings.append(f"FACTORY NOT CALLED: {module}.{factory}() imported but never called")
    
    ok_count += 1

# ===== 2. Check for orphan modules (files with code that nothing imports) =====
print(f"\n{'='*70}")
print("SECTION 1: ENGINE IMPORT AUDIT")
print(f"{'='*70}")

all_py = [f for f in os.listdir(BASE) if f.endswith('.py') and not f.startswith('_')]
# Which modules are never imported by autonomous_trader.py?
orphans = []
for f in sorted(all_py):
    mod = f[:-3]
    if mod == 'autonomous_trader':
        continue
    if mod == 'run':
        continue
    if mod.startswith(('analyze', 'backtest', 'check_', 'count_', 'day', 'debug_',
                       'diagnose_', 'download_', 'fetch_', 'fix_', 'impact_',
                       'loss_', 'predict_', 'quick_', 'refresh_', 'roi_',
                       'show_', 'simulate_', 'test_', 'trade_analysis', 'trade_query',
                       'watcher_diag', 'candle_')):
        continue  # utility/analysis scripts, not engines
    # Check if imported
    pats = [f'from {mod} import', f'import {mod}']
    if not any(p in trader_code for p in pats):
        fpath = os.path.join(BASE, f)
        with open(fpath, encoding='utf-8') as fh:
            lines = len(fh.readlines())
        orphans.append((mod, lines))

if issues:
    print(f"\n🚨 CRITICAL ISSUES ({len(issues)}):")
    for i in issues:
        print(f"  ❌ {i}")
else:
    print(f"\n✅ No critical issues found")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠️  {w}")

print(f"\n✅ {ok_count}/{len(CRITICAL_ENGINES)} critical engines verified: imported + file exists")

if orphans:
    print(f"\n📦 ORPHAN MODULES (exist but NOT imported by autonomous_trader.py):")
    for mod, lines in orphans:
        print(f"  📦 {mod}.py ({lines}L) — may be imported by other modules or unused")

# ===== 3. Check for code duplication (engine code copy-pasted into trader) =====
print(f"\n{'='*70}")
print("SECTION 2: CODE DUPLICATION CHECK")
print(f"{'='*70}")

# For each engine, check if its key functions also exist inline in trader
duplication_suspects = []
for module, info in CRITICAL_ENGINES.items():
    fpath = os.path.join(BASE, f"{module}.py")
    if not os.path.exists(fpath):
        continue
    with open(fpath, encoding='utf-8') as f:
        mod_code = f.read()
    
    # Find all function definitions in the module
    mod_funcs = re.findall(r'def (\w+)\(', mod_code)
    # Check if any of these show up as inline defs in trader (excluding class methods of the engine itself)
    for func in mod_funcs:
        if func.startswith('_') and len(func) < 4:
            continue
        # Check if this function is defined AGAIN in autonomous_trader.py
        pattern = f'def {func}('
        count_in_trader = trader_code.count(pattern)
        if count_in_trader > 0:
            # Find line numbers
            for i, line in enumerate(trader_lines):
                if pattern in line:
                    duplication_suspects.append((module, func, i+1))

if duplication_suspects:
    print(f"\n⚠️  POSSIBLE DUPLICATED FUNCTIONS ({len(duplication_suspects)}):")
    for mod, func, line in duplication_suspects:
        print(f"  ⚠️  {func}() defined in both {mod}.py AND autonomous_trader.py L{line}")
else:
    print(f"\n✅ No duplicated function definitions found")

# ===== 4. Line count analysis =====
print(f"\n{'='*70}")
print("SECTION 3: SIZE SANITY CHECK")
print(f"{'='*70}")
print(f"\nautonomous_trader.py: {total_lines} lines")
if total_lines > 13000:
    print(f"  🚨 BLOATED! Expected <13000. Possible code re-inlining.")
elif total_lines > 12800:
    print(f"  ⚠️  Getting big. Watch for growth.")
else:
    print(f"  ✅ Size is healthy")

# Module sizes
print(f"\nEngine module sizes:")
total_engine_lines = 0
for module in sorted(CRITICAL_ENGINES.keys()):
    fpath = os.path.join(BASE, f"{module}.py")
    if os.path.exists(fpath):
        with open(fpath, encoding='utf-8') as f:
            lines = len(f.readlines())
        total_engine_lines += lines
        marker = "📌" if lines > 500 else "  "
        print(f"  {marker} {module}.py: {lines}L")

print(f"\n  Total engine code: {total_engine_lines}L")
print(f"  Combined (trader + engines): {total_lines + total_engine_lines}L")

# ===== 5. MD5 fingerprints for drift detection =====
print(f"\n{'='*70}")
print("SECTION 4: FILE FINGERPRINTS (for future drift detection)")
print(f"{'='*70}")
import hashlib
print()
for module in sorted(CRITICAL_ENGINES.keys()):
    fpath = os.path.join(BASE, f"{module}.py")
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()[:12]
        with open(fpath, encoding='utf-8') as f:
            lines = len(f.readlines())
        print(f"  {module}.py  md5={h}  lines={lines}")

# autonomous_trader.py
with open(TRADER, 'rb') as f:
    h = hashlib.md5(f.read()).hexdigest()[:12]
print(f"  autonomous_trader.py  md5={h}  lines={total_lines}")

print(f"\n{'='*70}")
print(f"AUDIT COMPLETE")
if issues:
    print(f"🚨 {len(issues)} CRITICAL ISSUES — FIX IMMEDIATELY")
elif warnings:
    print(f"⚠️  {len(warnings)} warnings — review recommended")
else:
    print(f"✅ ALL CLEAR — system integrity verified")
print(f"{'='*70}")
