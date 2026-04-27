#!/usr/bin/env python3
"""
EARLYBIRD REMOVAL PATCH — Surgically removes all earlybird trade types
=====================================================================
Removes earlybird code from: config.py, kite_ticker.py, autonomous_trader.py,
zerodha_tools.py, options_trader.py, dashboard.py

Strategy: 
  1. Config: Replace EARLYBIRD block with disabled stubs (keeps imports safe)
  2. kite_ticker: Remove detection block, state init, sustain handling, priority entries
  3. autonomous_trader: Remove state vars, earlybird branches, clean interleaved conditions
  4. zerodha_tools: Remove threshold override
  5. options_trader: Remove bypass logic
"""
import sys
import shutil

BASE = '/home/ubuntu/titan/agentic_trader'
FILES = {
    'config': f'{BASE}/config.py',
    'kite': f'{BASE}/kite_ticker.py',
    'auto': f'{BASE}/autonomous_trader.py',
    'zerodha': f'{BASE}/zerodha_tools.py',
    'options': f'{BASE}/options_trader.py',
    'dashboard': f'{BASE}/dashboard.py',
}

results = {}

def patch_file(name, patches):
    """Apply patches to a file. Each patch is (description, old, new)."""
    path = FILES[name]
    with open(path, 'r') as f:
        code = f.read()
    shutil.copy2(path, path + '.bak_eb_removal')
    
    applied = 0
    for desc, old, new in patches:
        if old in code:
            code = code.replace(old, new, 1)
            applied += 1
            print(f"  ✅ {name}: {desc}")
        else:
            print(f"  ❌ {name}: {desc} — ANCHOR NOT FOUND")
            # Show first 80 chars of anchor for debugging
            print(f"     Looking for: {old[:80]}...")
    
    with open(path, 'w') as f:
        f.write(code)
    results[name] = applied
    return applied


# ═══════════════════════════════════════════════════════════
# CONFIG.PY — Replace earlybird block with disabled stubs
# ═══════════════════════════════════════════════════════════
print("\n=== CONFIG.PY ===")

config_old = """# === EARLYBIRD STRATEGY (Opening Volatility 09:15-09:45) ===
# Three distinct modes — tracked separately for performance analysis:
#   A = Gap continuation (highest quality)
#   B = Strong opening directional move (medium quality)
#   C = Opening spike (lowest quality — needs volume + hold confirmation)       
#
# Market-context gating: Is the stock move idiosyncratic or just market beta?   
# If NIFTY is moving the same way and the stock is HIGH_BETA, it's likely just  
# index-driven — lower conviction. Idiosyncratic moves (stock moves, index doesn't,
# or stock moves AGAINST index) are highest quality.
#
# Shared settings applied to all three modes:
EARLYBIRD_COMMON = {"""

config_new = """# === EARLYBIRD STRATEGY — REMOVED Apr 20, 2026 ===
# All earlybird trade types removed. Stubs kept for import compatibility.
EARLYBIRD_COMMON = {"enabled": False}
EARLYBIRD_A = {"enabled": False}
EARLYBIRD_B = {"enabled": False}
EARLYBIRD_C = {"enabled": False}
EARLYBIRD_D = {"enabled": False}
EARLYBIRD = {"enabled": False}

# REMOVED_EARLYBIRD_STUB = {"""

# Read current config to find end of earlybird block
with open(FILES['config'], 'r') as f:
    config_code = f.read()

# Find the earlybird block boundaries
eb_start = config_code.find('# === EARLYBIRD STRATEGY (Opening Volatility')
if eb_start == -1:
    print("  ❌ config: EARLYBIRD block start not found")
    sys.exit(1)

# Find end: "EARLYBIRD = {**EARLYBIRD_COMMON}" followed by next section
eb_end_marker = 'EARLYBIRD = {**EARLYBIRD_COMMON}'
eb_end = config_code.find(eb_end_marker, eb_start)
if eb_end == -1:
    print("  ❌ config: EARLYBIRD block end marker not found")
    sys.exit(1)
# Move to end of that line
eb_end = config_code.find('\n', eb_end) + 1

# Extract the block to remove
eb_block = config_code[eb_start:eb_end]

# Replace with stubs
config_replacement = """# === EARLYBIRD STRATEGY — REMOVED Apr 20, 2026 ===
# All earlybird trade types removed. Stubs kept for import compatibility.
EARLYBIRD_COMMON = {"enabled": False}
EARLYBIRD_A = {"enabled": False}
EARLYBIRD_B = {"enabled": False}
EARLYBIRD_C = {"enabled": False}
EARLYBIRD_D = {"enabled": False}
EARLYBIRD = {"enabled": False}
"""

shutil.copy2(FILES['config'], FILES['config'] + '.bak_eb_removal')
config_code = config_code[:eb_start] + config_replacement + config_code[eb_end:]
with open(FILES['config'], 'w') as f:
    f.write(config_code)
print("  ✅ config: Replaced EARLYBIRD block with disabled stubs")
results['config'] = 1


# ═══════════════════════════════════════════════════════════
# KITE_TICKER.PY — Remove earlybird detection, state, sustain
# ═══════════════════════════════════════════════════════════
print("\n=== KITE_TICKER.PY ===")

kite_patches = []

# 1. State init: replace earlybird state block with minimal stub
kite_patches.append((
    "Remove earlybird state init",
    """        # === EARLYBIRD STATE (Opening Volatility 09:15-09:45) ===
        from config import EARLYBIRD_COMMON, EARLYBIRD_A, EARLYBIRD_B, EARLYBIRD_C, EARLYBIRD_D
        self._eb_common = EARLYBIRD_COMMON
        self._eb_a = EARLYBIRD_A
        self._eb_b = EARLYBIRD_B
        self._eb_c = EARLYBIRD_C
        self._eb_d = EARLYBIRD_D
        self._earlybird_enabled = EARLYBIRD_COMMON.get('enabled', False)        
        self._earlybird_trades_fired = 0  # Count of earlybird triggers queued today
        self._prev_close: Dict[str, float] = {}  # sym → previous day close price (from OHLC)
        self._day_open: Dict[str, float] = {}     # sym → today's open price (from OHLC)
        self._earlybird_fired: Dict[str, set] = {'A': set(), 'B': set(), 'C': set(), 'D': set()}  # Per-mode fired symbols
        self._news_targets: Dict[str, dict] = {}  # symbol → news target dict (set by autonomous_trader)""",
    """        # === EARLYBIRD — REMOVED Apr 20 ===
        self._earlybird_enabled = False
        self._earlybird_trades_fired = 0
        self._prev_close: Dict[str, float] = {}
        self._day_open: Dict[str, float] = {}
        self._earlybird_fired: Dict[str, set] = {}
        self._news_targets: Dict[str, dict] = {}"""
))

# 2. Sustain earlybird handling — remove earlybird branches only
kite_patches.append((
    "Remove earlybird sustain time branch",
    """            if 'EARLYBIRD' in _ttype_s:
                _effective_sustain = pending.get('_earlybird_sustain', 10)      
            elif 'GRIND' in _ttype_s:""",
    """            if 'GRIND' in _ttype_s:"""
))

kite_patches.append((
    "Remove earlybird sustain hold threshold",
    """                # Earlybird uses per-mode sustain hold threshold
                if 'EARLYBIRD' in _ttype:
                    _eb_mode = pending.get('earlybird_mode', 'B')
                    if _eb_mode == 'A':
                        _recheck = self._eb_a.get('sustain_min_hold_pct', 0.4)  
                    elif _eb_mode == 'C':
                        _recheck = self._eb_c.get('sustain_min_hold_pct', 0.7)  
                    else:
                        _recheck = self._eb_b.get('sustain_min_hold_pct', 0.5)  
                elif 'SPIKE' in _ttype:""",
    """                if 'SPIKE' in _ttype:"""
))

# 3. Remove earlybird skip of early market hardening
kite_patches.append((
    "Remove earlybird early-market hardening skip",
    """                # NOTE: Earlybird triggers SKIP early market hardening (they ARE early market)
                if 'EARLYBIRD' not in _ttype:
                    _em_end = self._config.get('early_market_end', '09:55')""",
    """                if True:  # Early market hardening
                    _em_end = self._config.get('early_market_end', '09:55')"""
))

# 4. Remove earlybird priority entries from _TRIGGER_PRIORITY
kite_patches.append((
    "Remove earlybird priority map entries",
    """            'PRICE_SPIKE_UP': 4, 'PRICE_SPIKE_DOWN': 4,
            'EARLYBIRD_UP': 5, 'EARLYBIRD_DOWN': 5,
            'EARLYBIRD_C_UP': 5, 'EARLYBIRD_C_DOWN': 5,
            'EARLYBIRD_B_UP': 6, 'EARLYBIRD_B_DOWN': 6,
            'EARLYBIRD_A_UP': 7, 'EARLYBIRD_A_DOWN': 7,
            'EARLYBIRD_D_UP': 8, 'EARLYBIRD_D_DOWN': 8,  # News-based — highest priority
        }""",
    """            'PRICE_SPIKE_UP': 4, 'PRICE_SPIKE_DOWN': 4,
        }"""
))

# 5. Remove earlybird metadata from trigger data field list
kite_patches.append((
    "Remove earlybird fields from trigger metadata",
    """                          'gap_pct', 'open_move_pct', 'has_gap', 'strong_gap',  
                          'earlybird_mode', 'earlybird_reason',
                          'news_confidence', 'news_sentiment', 'news_reason', 'news_headline'):""",
    """                          'gap_pct', 'open_move_pct', 'has_gap', 'strong_gap',  
                          'news_confidence', 'news_sentiment', 'news_reason', 'news_headline'):"""
))

# 6. Remove earlybird daily reset
kite_patches.append((
    "Remove earlybird daily reset",
    """        # Reset Earlybird state
        self._earlybird_trades_fired = 0
        self._prev_close.clear()
        self._day_open.clear()
        self._earlybird_fired = {'A': set(), 'B': set(), 'C': set(), 'D': set()}
        self._news_targets = {}  # Clear news targets for new day""",
    """        # Earlybird removed — minimal reset
        self._earlybird_trades_fired = 0
        self._prev_close.clear()
        self._day_open.clear()
        self._earlybird_fired = {}
        self._news_targets = {}"""
))

# 7. Remove prev_close/day_open tracking comment (keep the tracking — other code may use it)
kite_patches.append((
    "Clean prev_close tracking comment",
    """        # === TRACK PREV CLOSE & DAY OPEN (for Earlybird gap detection) ===""",
    """        # === TRACK PREV CLOSE & DAY OPEN ==="""
))

# 8. Remove NIFTY earlybird comment
kite_patches.append((
    "Clean NIFTY tracking comment",
    """                            # Feed NIFTY 50 data for Earlybird market-context gating""",
    """                            # Feed NIFTY 50 data for market-context gating"""
))

# 9. BIGGEST: Remove the entire earlybird detection block (lines 1592-1789)
# This is the SOURCE of all earlybird triggers
kite_patches.append((
    "Remove earlybird detection block (source of triggers)",
    """        # === EARLYBIRD DETECTION (09:16-09:45 opening volatility) ===
        # Three modes tracked separately:
        #   A = Gap continuation (highest quality, larger sizing)
        #   B = Strong opening directional move (medium quality)
        #   C = Opening spike (lowest quality, stricter sustain)
        # Market-context: compare stock move vs NIFTY 50 to detect beta vs idiosyncratic.
        _earlybird_triggered = False
        if self._earlybird_enabled:""",
    """        # === EARLYBIRD DETECTION — REMOVED Apr 20, 2026 ===
        _earlybird_triggered = False
        if False:  # EARLYBIRD REMOVED"""
))

patch_file('kite', kite_patches)


# ═══════════════════════════════════════════════════════════
# AUTONOMOUS_TRADER.PY — Remove earlybird state, branches, gates
# ═══════════════════════════════════════════════════════════
print("\n=== AUTONOMOUS_TRADER.PY ===")

auto_patches = []

# 1. Remove state vars
auto_patches.append((
    "Remove earlybird state vars",
    """        self._earlybird_total_placed = 0             # Total EARLYBIRD trades placed today""",
    """        self._earlybird_total_placed = 0             # EARLYBIRD REMOVED — stub for compat"""
))

auto_patches.append((
    "Remove earlybird mode counter",
    """        self._earlybird_mode_placed = {'A': 0, 'B': 0, 'C': 0, 'D': 0}  # Per-mode counter""",
    """        self._earlybird_mode_placed = {}  # EARLYBIRD REMOVED"""
))

# 2. Remove earlybird sector penalty comment
auto_patches.append((
    "Clean sector map comment",
    """        # ARBTR: DISABLED Apr 8 — keep sector map for earlybird sector penalty""",
    """        # ARBTR: DISABLED Apr 8"""
))

# 3. Clean the watcher drain earlybird bypass
auto_patches.append((
    "Remove earlybird drain bypass",
    """        # EARLYBIRD triggers bypass this gate — they fire from 09:16       
        from config import EARLYBIRD_COMMON as _EB_COM, EARLYBIRD_A as _EB_A, EARLYBIRD_B as _EB_B, EARLYBIRD_C as _EB_C, EARLYBIRD_D as _EB_D
        _earlybird_active = _EB_COM.get('enabled', False)""",
    """        _earlybird_active = False  # EARLYBIRD REMOVED"""
))

# 4. Remove from config imports in scoring section
auto_patches.append((
    "Remove earlybird config imports in scoring",
    """            EARLYBIRD_COMMON as _EB_COM,
            EARLYBIRD_A as _EB_A,
            EARLYBIRD_B as _EB_B,
            EARLYBIRD_C as _EB_C,
            EARLYBIRD_D as _EB_D,""",
    ""
))

# 5. Remove earlybird trigger direction mapping entries
auto_patches.append((
    "Remove EARLYBIRD_UP from direction mapping",
    """                if _trigger_type in ('PRICE_SPIKE_UP', 'NEW_DAY_HIGH', 'SLOW_GRIND_UP', 'EARLYBIRD_UP',
                                     'EARLYBIRD_A_UP', 'EARLYBIRD_B_UP', 'EARLYBIRD_C_UP', 'EARLYBIRD_D_UP'):""",
    """                if _trigger_type in ('PRICE_SPIKE_UP', 'NEW_DAY_HIGH', 'SLOW_GRIND_UP'):"""
))

auto_patches.append((
    "Remove EARLYBIRD_DOWN from direction mapping",
    """                elif _trigger_type in ('PRICE_SPIKE_DOWN', 'NEW_DAY_LOW', 'SLOW_GRIND_DOWN', 'EARLYBIRD_DOWN',
                                       'EARLYBIRD_A_DOWN', 'EARLYBIRD_B_DOWN', 'EARLYBIRD_C_DOWN', 'EARLYBIRD_D_DOWN'):""",
    """                elif _trigger_type in ('PRICE_SPIKE_DOWN', 'NEW_DAY_LOW', 'SLOW_GRIND_DOWN'):"""
))

# 6. Remove earlybird score gate override — the _is_earlybird block
auto_patches.append((
    "Remove earlybird score gate",
    """                # EARLYBIRD uses per-mode min_score (A most relaxed, C tightest)
                _is_earlybird = 'EARLYBIRD' in _trigger_type""",
    """                _is_earlybird = False  # EARLYBIRD REMOVED"""
))

# 7. Simplify B2 threshold
auto_patches.append((
    "Simplify B2 threshold (remove earlybird relaxation)",
    """                # Apr 2: EARLYBIRD uses relaxed threshold (1/4) — technicals unreliable at open
                _b2_min_needed = 1 if _is_earlybird else 2""",
    """                _b2_min_needed = 2"""
))

# 8. Remove earlybird range budget skip
auto_patches.append((
    "Remove earlybird range budget skip",
    """                _is_earlybird = 'EARLYBIRD' in _trigger_type
                if not _is_earlybird and _c2_budget_used >= _c2_thresh and _c2_range > 0.5:""",
    """                if _c2_budget_used >= _c2_thresh and _c2_range > 0.5:"""
))

# But we need to check what's on the line with the range budget comment. Let me handle both patterns.

# 9. Remove earlybird ADX skip
auto_patches.append((
    "Remove earlybird ADX gate skip",
    """                # [Apr-2] Skip for EARLYBIRD — ADX unreliable at market open
                if adx_val < _adx_min and _trigger_type not in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN') and not _is_earlybird:""",
    """                if adx_val < _adx_min and _trigger_type not in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):"""
))

# 10. Remove earlybird metadata storage
auto_patches.append((
    "Remove earlybird metadata storage",
    """                # EARLYBIRD metadata for position tracking (per-mode)      
                if 'EARLYBIRD' in _trigger_type:
                    _ml_data['earlybird'] = True
                    _ml_data['earlybird_mode'] = _trigger.get('earlybird_mode', 'B')
                    _ml_data['earlybird_gap_pct'] = _trigger.get('gap_pct', 0)
                    _ml_data['earlybird_reason'] = _trigger.get('earlybird_reason', '')
                    _ml_data['earlybird_has_gap'] = _trigger.get('has_gap', False)
                    _ml_data['earlybird_strong_gap'] = _trigger.get('strong_gap', False)
                    _ml_data['earlybird_nifty_change'] = _trigger.get('nifty_change_pct', 0)
                    _ml_data['earlybird_idiosyncratic'] = _trigger.get('is_idiosyncratic', False)
                    _ml_data['earlybird_beta_driven'] = _trigger.get('is_beta_driven', False)""",
    ""
))

# 11. Remove earlybird scoring gate skip
auto_patches.append((
    "Remove earlybird scoring gate skip",
    """                # EARLYBIRD and INTRADAY_NEWS skip this gate (have their own scoring)
                if not _is_earlybird and not _is_news_trigger and ('DAY' in _trigger_type or 'SPIKE' in _trigger_type):""",
    """                if not _is_news_trigger and ('DAY' in _trigger_type or 'SPIKE' in _trigger_type):"""
))

# 12. Remove earlybird OI bypass
auto_patches.append((
    "Remove earlybird OI bypass",
    """                        or 'EARLYBIRD' in _bt_ttype  # Earlybird always bypasses OI (too early for reliable OI)""",
    ""
))

# 13. Remove earlybird max trades cap
auto_patches.append((
    "Remove earlybird max trades cap",
    """                    # [Apr 2] EARLYBIRD max_trades cap — limit earlybird trades per morning
                    if 'EARLYBIRD' in _trigger_type:
                        _eb_max_trades = _EB_COM.get('max_trades', 3)      
                        if self._earlybird_total_placed >= _eb_max_trades: 
                            self._wlog(f"  ⛔ EARLYBIRD CAP: {_sym.replace('NSE:', '')} — already placed {self._earlybird_total_placed}/{_eb_max_trades} earlybird trades today")""",
    ""
))

# Alternate max trades cap format (whitespace may differ)
auto_patches.append((
    "Remove earlybird max trades cap (alt format)",
    """                    # [Apr 2] EARLYBIRD max_trades cap — limit earlybird trades per morning
                    if 'EARLYBIRD' in _trigger_type:
                        _eb_max_trades = _EB_COM.get('max_trades', 3)
                        if self._earlybird_total_placed >= _eb_max_trades:
                            self._wlog(f"  ⛔ EARLYBIRD CAP: {_sym.replace('NSE:', '')} — already placed {self._earlybird_total_placed}/{_eb_max_trades} earlybird trades today")""",
    ""
))

# 14. Remove earlybird placement counter/logging
auto_patches.append((
    "Remove earlybird placement logging",
    """                        if 'EARLYBIRD' in _trigger_type:
                            self._earlybird_total_placed += 1""",
    """                        if False:  # EARLYBIRD REMOVED
                            self._earlybird_total_placed += 1"""
))

# 15. Remove earlybird startup init
auto_patches.append((
    "Remove earlybird startup init",
    """            from config import EARLYBIRD_D as _EB_D_INIT
            if _EB_D_INIT.get('enabled', False):""",
    """            if False:  # EARLYBIRD REMOVED"""
))

patch_file('auto', auto_patches)


# ═══════════════════════════════════════════════════════════
# ZERODHA_TOOLS.PY — Remove earlybird threshold override
# ═══════════════════════════════════════════════════════════
print("\n=== ZERODHA_TOOLS.PY ===")

zerodha_patches = []

zerodha_patches.append((
    "Remove earlybird threshold override",
    """        # === EARLYBIRD THRESHOLD OVERRIDE: Watcher gates already validated the trade ===""",
    """        # === EARLYBIRD THRESHOLD OVERRIDE — REMOVED Apr 20 ==="""
))

zerodha_patches.append((
    "Disable earlybird threshold logic",
    """        _eb_threshold_override = None
        if 'EARLYBIRD' in (setup_type or ''):""",
    """        _eb_threshold_override = None
        if False:  # EARLYBIRD REMOVED"""
))

zerodha_patches.append((
    "Remove earlybird theta relaxation",
    """                # EARLYBIRD: relax to 12% — theta is always high at 9:15, gap edge compensates""",
    """                # EARLYBIRD theta relaxation — REMOVED Apr 20"""
))

zerodha_patches.append((
    "Disable earlybird theta check",
    """                    if 'EARLYBIRD' in (setup_type or ''):""",
    """                    if False:  # EARLYBIRD REMOVED"""
))

patch_file('zerodha', zerodha_patches)


# ═══════════════════════════════════════════════════════════
# OPTIONS_TRADER.PY — Remove earlybird bypass logic
# ═══════════════════════════════════════════════════════════
print("\n=== OPTIONS_TRADER.PY ===")

options_patches = []

options_patches.append((
    "Remove earlybird conviction override",
    """        # EARLYBIRD override: at 9:15-9:25, follow-through/ORB/volume barely exist.""",
    """        # EARLYBIRD conviction override — REMOVED Apr 20"""
))

options_patches.append((
    "Disable conviction bypass check",
    """        _eb_conv_bypass = getattr(self, '_eb_conviction_override', False)
        if _eb_conv_bypass:""",
    """        _eb_conv_bypass = False  # EARLYBIRD REMOVED
        if False:"""
))

options_patches.append((
    "Disable micro bypass",
    """        # EARLYBIRD: downgrade hard-block to warning (9:15 spreads are always wide,""",
    """        # EARLYBIRD micro bypass — REMOVED Apr 20"""
))

options_patches.append((
    "Disable micro bypass check",
    """        _eb_micro_bypass = getattr(self, '_eb_micro_override', False)
        if microstructure_block and not _eb_micro_bypass:""",
    """        _eb_micro_bypass = False  # EARLYBIRD REMOVED
        if microstructure_block:"""
))

options_patches.append((
    "Remove earlybird micro bypass branch",
    """        elif microstructure_block and _eb_micro_bypass:
            warnings.append(f"⚠️ EARLYBIRD micro-bypass: {microstructure_block_reason} (downgraded to wwarning)")""",
    """        # EARLYBIRD micro-bypass removed"""
))

options_patches.append((
    "Remove earlybird scoring override",
    """        # EARLYBIRD trades go through intraday scoring (with relaxed threshold set upstream)
        _eb_use_scoring = 'EARLYBIRD' in (setup_type or '')
        if market_data and (not setup_type or _eb_use_scoring):""",
    """        if market_data and not setup_type:"""
))

patch_file('options', options_patches)


# ═══════════════════════════════════════════════════════════
# DASHBOARD.PY — Remove earlybird news trigger
# ═══════════════════════════════════════════════════════════
print("\n=== DASHBOARD.PY ===")

dash_patches = []

dash_patches.append((
    "Remove earlybird news trigger setup_type",
    """            'setup_type': 'WATCHER_EARLYBIRD_D_UP' if sentiment == 'BULLISH' else 'WATCHER_EARLYBIRD_D_DOWN',""",
    """            'setup_type': 'WATCHER_NEWS_UP' if sentiment == 'BULLISH' else 'WATCHER_NEWS_DOWN',"""
))

patch_file('dashboard', dash_patches)


# ═══════════════════════════════════════════════════════════
# COMPILE ALL
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMPILATION CHECK:")
import py_compile
all_ok = True
for name, path in FILES.items():
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ {name}: compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ {name}: COMPILE ERROR: {e}")
        all_ok = False

print(f"\n{'='*60}")
print("RESULTS:")
total = sum(results.values())
for name, count in results.items():
    print(f"  {name}: {count} patches applied")
print(f"  TOTAL: {total} patches")
if all_ok:
    print(f"\n✅ ALL FILES COMPILE OK — safe to restart")
else:
    print(f"\n❌ COMPILE ERRORS — DO NOT RESTART, fix first")
    print(f"   Backups: {BASE}/*.bak_eb_removal")
