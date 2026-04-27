#!/usr/bin/env python3
"""Final earlybird cleanup — remove remaining references that failed or were missed."""

AUTO = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'

with open(AUTO, 'r') as f:
    code = f.read()

fixes = 0

# ═══ 1: Remove EARLYBIRD trigger bonus elif block ═══
old = """                elif 'EARLYBIRD' in _trigger_type:
                    # Per-mode Earlybird bonus: D (news) > A > B > C
                    _eb_mode = _trigger.get('earlybird_mode', 'B')
                    if _eb_mode == 'D':
                        # NEWS-BASED Mode D — highest base bonus
                        _trigger_bonus = _EB_D.get('trigger_bonus', 18)
                        _news_conf = _trigger.get('news_confidence', 0)
                        if _news_conf >= 65:
                            _trigger_bonus += _EB_D.get('news_confidence_bonus', 5)
                elif _trigger_type == 'INTRADAY_NEWS':"""
new = """                elif _trigger_type == 'INTRADAY_NEWS':"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 1: Removed EARLYBIRD trigger bonus elif block")
else:
    print("❌ 1: EARLYBIRD trigger bonus block not found")

# ═══ 2: Clean INTRADAY_NEWS earlybird _eb_mode branches ═══
# Remove from "elif _eb_mode == 'A':" through sector alignment break
# Keep INTRADAY_NEWS as just: bonus=12, +3 if conf>=65
old = """                    if _news_conf >= 65:
                        _trigger_bonus += 3
                    elif _eb_mode == 'A':
                        _trigger_bonus = _EB_A.get('trigger_bonus', 15)
                        if _trigger.get('has_gap', False):
                            _trigger_bonus += _EB_A.get('gap_bonus', 5)
                        if _trigger.get('strong_gap', False):
                            _trigger_bonus += _EB_A.get('strong_gap_bonus', 3)  
                    elif _eb_mode == 'C':
                        _trigger_bonus = _EB_C.get('trigger_bonus', 7)
                    else:
                        _trigger_bonus = _EB_B.get('trigger_bonus', 10)
                    # Large opening move bonus (all modes)
                    _eb_open_move = abs(_trigger.get('open_move_pct', 0))       
                    if _eb_open_move >= 1.5:
                        _trigger_bonus += 3
                    elif _eb_open_move >= 1.0:
                        _trigger_bonus += 1
                    # Market-context adjustments
                    if _EB_COM.get('market_context_enabled', True):
                        if _trigger.get('is_idiosyncratic', False):
                            _trigger_bonus += _EB_COM.get('idiosyncratic_bonus', 4)
                        if _trigger.get('is_beta_driven', False):
                            _trigger_bonus -= _EB_COM.get('beta_penalty', 5)    
                        # Sector alignment check: if sector index moves same direction, penalize
                        _eb_sec_chgs = getattr(self, '_sector_index_changes_cache', {})
                        if _eb_sec_chgs:
                            _eb_stock_clean = _sym.replace('NSE:', '')
                            _eb_sec_penalty_applied = False
                            for _eb_sec_name, _eb_sec_info in self._arbtr_sector_map.items():
                                if _eb_stock_clean in _eb_sec_info.get('stocks', []):
                                    _eb_sec_idx = _eb_sec_info.get('index', '') 
                                    _eb_sec_chg = _eb_sec_chgs.get(_eb_sec_idx, 0)
                                    _eb_stock_dir = _trigger.get('open_move_pct', 0)
                                    if abs(_eb_sec_chg) >= 0.3 and (
                                        (_eb_sec_chg > 0 and _eb_stock_dir > 0) or
                                        (_eb_sec_chg < 0 and _eb_stock_dir < 0)):
                                        _trigger_bonus -= _EB_COM.get('sector_aligned_penalty', 3)
                                        _eb_sec_penalty_applied = True
                                    break"""
new = """                    if _news_conf >= 65:
                        _trigger_bonus += 3"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 2: Cleaned INTRADAY_NEWS earlybird branches")
else:
    print("❌ 2: INTRADAY_NEWS earlybird branches not found")

# ═══ 3: Remove earlybird setup_type branch ═══
old = """                if 'EARLYBIRD' in _trigger_type:
                    _setup_type = f'WATCHER_{_trigger_type}'
                elif 'DAY' in _trigger_type or 'SPIKE' in _trigger_type:"""
new = """                if 'DAY' in _trigger_type or 'SPIKE' in _trigger_type:"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 3: Removed earlybird setup_type branch")
else:
    print("❌ 3: Setup type branch not found")

# ═══ 4: Remove earlybird P(move) floor ═══
old = """                    # EARLYBIRD uses per-mode P(move) floor
                    if _is_earlybird:
                        _eb_mode_cfg = {'A': _EB_A, 'B': _EB_B, 'C': _EB_C, 'D': _EB_D}.get(_eb_mode, _EB_B)
                        _w_min_move = _eb_mode_cfg.get('min_move_prob', 0.35)   
                    elif _is_news_trigger:
                        _w_min_move = _EB_D.get('min_move_prob', 0.25)  # News uses Mode D P(move) floor
                    else:"""
new = """                    if _is_news_trigger:
                        _w_min_move = 0.25  # News P(move) floor
                    else:"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 4: Removed earlybird P(move) floor")
else:
    print("❌ 4: P(move) floor not found")

# ═══ 5: Remove earlybird lot sizing block ═══
old = """                    elif 'EARLYBIRD' in _trigger_type:
                        _eb_trigger = _trigger_map.get(_sym, {})
                        _eb_m = _eb_trigger.get('earlybird_mode', 'B')
                        _eb_m_cfg = {'A': _EB_A, 'B': _EB_B, 'C': _EB_C, 'D': _EB_D}.get(_eb_m, _EB_B)
                        if _eb_m == 'D':
                            # News-based Mode D — size by news confidence       
                            _news_conf = _eb_trigger.get('news_confidence', 0)  
                            if _news_conf >= 70:
                                _lot_mult = _EB_D.get('high_conf_lot_multiplier', 2.0)
                                _surge_tag = f' [EB-D NEWS conf={_news_conf}→{_lot_mult}x LOT]'
                            else:
                                _lot_mult = _EB_D.get('lot_multiplier', 1.5)    
                                _surge_tag = f' [EB-D NEWS→{_lot_mult}x LOT]'   
                        elif _eb_m == 'A' and _eb_trigger.get('strong_gap', False):
                            _lot_mult = _EB_A.get('gap_strong_lot_multiplier', 2.0)
                            _surge_tag = f' [EB-A STRONG GAP→{_lot_mult}x LOT]' 
                        else:
                            _lot_mult = _eb_m_cfg.get('lot_multiplier', 1.0)    
                            if _lot_mult != 1.0:
                                _surge_tag = f' [EB-{_eb_m}→{_lot_mult}x LOT]'"""
new = ""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 5: Removed earlybird lot sizing block")
else:
    print("❌ 5: Lot sizing block not found")

# ═══ 6: Fix second _is_earlybird at line ~2981 ═══
old = """                _is_earlybird = 'EARLYBIRD' in _trigger_type
                adx_val = _data.get('adx', 20)"""
new = """                adx_val = _data.get('adx', 20)"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 6: Removed second _is_earlybird assignment")
else:
    print("❌ 6: Second _is_earlybird not found")

# ═══ 7: Fix before-watcher-start filter ═══
old = """            # Before watcher_start, only EARLYBIRD triggers are allowed
            if _before_watcher_start and 'EARLYBIRD' not in _ttype_filter:"""
new = """            # Before watcher_start, skip all triggers (earlybird removed)
            if _before_watcher_start:"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 7: Simplified before-watcher-start filter")
else:
    print("❌ 7: Before-watcher filter not found")

# ═══ 8: Clean earlybird comment in before-watcher block ═══
old = """        # If before watcher_start, only allow EARLYBIRD triggers through"""
new = """        # If before watcher_start, skip all triggers"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 8: Cleaned before-watcher comment")
else:
    print("❌ 8: Before-watcher comment not found")

# ═══ 9: Clean earlybird lot comment ═══
old = """                    # SPIKE + SURGE co-fire → double the lot (2x sizing)        
                    # EARLYBIRD: use configurable lot multiplier (strong gap = 1.5x)"""
new = """                    # SPIKE + SURGE co-fire → double the lot (2x sizing)"""
if old in code:
    code = code.replace(old, new, 1)
    fixes += 1
    print("✅ 9: Cleaned earlybird lot comment")
else:
    print("❌ 9: Earlybird lot comment not found")

# ═══ 10: Clean earlybird range budget comment ═══
old = """                #   - EARLYBIRD: skip (range hasn't formed yet)"""
if old in code:
    code = code.replace(old, '', 1)
    fixes += 1
    print("✅ 10: Removed earlybird range budget comment")
else:
    print("❌ 10: Range budget comment not found")

# ═══ 11: Clean earlybird ADX comment ═══
old = """                # [Apr-2] Skip for EARLYBIRD — ADX unreliable at market open"""
if old in code:
    code = code.replace(old, '', 1)
    fixes += 1
    print("✅ 11: Removed earlybird ADX comment")
else:
    print("❌ 11: ADX comment not found")

# ═══ 12: Clean earlybird placement logging block ═══
# The earlier patch changed this to "if False: # EARLYBIRD REMOVED"
# Let's find and remove the entire if False block
old = """                        if False:  # EARLYBIRD REMOVED
                            self._earlybird_total_placed += 1"""
if old in code:
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'if False:  # EARLYBIRD REMOVED' in line:
            # Find end of this if block (next line at same or lower indent)
            j = i + 1
            base_indent = len(line) - len(line.lstrip())
            while j < len(lines) and (lines[j].strip() == '' or 
                  (len(lines[j]) - len(lines[j].lstrip())) > base_indent):
                j += 1
            del lines[i:j]
            code = '\n'.join(lines)
            fixes += 1
            print("✅ 12: Removed earlybird placement logging if-False block")
            break
else:
    print("❌ 12: Placement logging not found")

with open(AUTO, 'w') as f:
    f.write(code)

# ═══ COMPILE CHECK ═══
print(f"\n{'='*60}")
print("COMPILATION CHECK:")
import py_compile
all_ok = True
for path in [AUTO, 
             '/home/ubuntu/titan/agentic_trader/config.py',
             '/home/ubuntu/titan/agentic_trader/kite_ticker.py',
             '/home/ubuntu/titan/agentic_trader/zerodha_tools.py',
             '/home/ubuntu/titan/agentic_trader/options_trader.py',
             '/home/ubuntu/titan/agentic_trader/dashboard.py']:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ {path.split('/')[-1]}: compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ {path.split('/')[-1]}: COMPILE ERROR: {e}")
        all_ok = False

if all_ok:
    print(f"\n✅ {fixes} fixes applied, ALL files compile OK — safe to restart")
else:
    print(f"\n❌ COMPILE ERRORS remain — check above")
