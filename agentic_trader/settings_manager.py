"""
settings_manager.py — Single source of truth for ALL Titan runtime settings.

Architecture:
  ┌──────────────┐        ┌───────────────────┐
  │  config.py   │──────▶ │  titan_settings    │ ◀──── Dashboard UI writes
  │  (defaults)  │ sync   │  .json (persisted) │       Bot hot-reload reads
  └──────────────┘        └───────────────────┘
                                │
                                ▼  apply()
                          config.py module
                          (in-memory, ALL code reads from here)

Rules:
  1. titan_settings.json is the PERSISTENT source of truth.
  2. config.py provides FACTORY DEFAULTS for keys missing from json.
  3. On startup, missing keys are auto-populated from config.py.
  4. set() writes to json AND updates config module in-memory.
  5. apply() pushes json → config module (hot-reload, every 30s).
  6. Existing code reads config module directly (zero refactor needed).
"""

import json
import os
import threading
from pathlib import Path
from datetime import datetime

SETTINGS_FILE = Path(__file__).parent / 'titan_settings.json'

# ── Key Mapping ──────────────────────────────────────────────────────
# Maps titan_settings.json flat keys to (config_attr, config_key) or
# (config_attr, config_key, sub_key) for nested dicts.
#
# Convention: watcher_{key} → BREAKOUT_WATCHER[key]
#             strategy_{name} → {NAME}.enabled
#             lots_{name} → {NAME}.lot_multiplier (or special key)
#             CAPITAL etc. → HARD_RULES[key]

# All BREAKOUT_WATCHER keys that should be synced.
# Format: json_key → config dict key
_WATCHER_KEYS = [
    'min_score', 'orb_min_score', 'orb_min_move_prob',
    'watcher_min_move_prob', 'watcher_min_adx',
    'max_trades_per_scan', 'max_triggers_per_batch',
    'sustain_seconds', 'sustain_seconds_extreme',
    'sustain_seconds_volume', 'sustain_seconds_grind',
    'sustain_recheck_pct', 'sustain_recheck_pct_volume',
    'sustain_retrace_max_pct', 'volume_surge_min_move_pct',
    'price_spike_pct', 'day_extreme_trigger', 'day_extreme_min_move_pct',
    'volume_surge_multiplier', 'slow_grind_pct',
    'cooldown_seconds', 'max_triggers_per_minute',
    'queue_size', 'priority_bypass_pct',
    'active_after', 'active_until', 'watcher_start',
    'scorer_conflict_max_score',
    'rsi_extreme_pe_max', 'rsi_extreme_ce_min',
    'exhaustion_index_block',
    'max_trades_before_1000',
    'early_market_end', 'early_market_min_score',
    'early_market_min_dir_conf', 'early_market_min_sustain_pct',
    'vix_penalty_above', 'vix_penalty_per_point', 'vix_hard_block_above',
]

# HARD_RULES keys
_HARD_RULES_KEYS = [
    'CAPITAL', 'RISK_PER_TRADE', 'MAX_DAILY_LOSS', 'MAX_POSITIONS',
    'REENTRY_COOLDOWN_MINUTES', 'MIN_OPTION_PREMIUM', 'PORTFOLIO_PROFIT_TARGET',
]

# Strategy names (enabled toggle)
_STRATEGY_NAMES = [
    'BREAKOUT_WATCHER', 'ELITE_AUTO_FIRE', 'DOWN_RISK_GATING',
    'GMM_SNIPER', 'GMM_CONTRARIAN', 'TEST_GMM', 'TEST_XGB',
    'SNIPER_OI_UNWINDING', 'SNIPER_PCR_EXTREME', 'ARBTR_CONFIG',
    'IRON_CONDOR_CONFIG', 'CREDIT_SPREAD_CONFIG', 'DEBIT_SPREAD_CONFIG',
    'ML_DIRECTION_CONFLICT', 'GCR_CONFIG',
]

# Lot multiplier strategies
_LOT_STRATEGIES = {
    'GMM_SNIPER': 'lot_multiplier',
    'ARBTR_CONFIG': 'lot_multiplier',
    'SNIPER_OI_UNWINDING': 'lot_multiplier',
    'SNIPER_PCR_EXTREME': 'lot_multiplier',
    'DOWN_RISK_GATING': 'all_agree_lot_multiplier',
    'GMM_CONTRARIAN': 'lot_multiplier',
}


class SettingsManager:
    """Centralized settings manager. Thread-safe."""

    def __init__(self):
        self._data = {}
        self._mtime = 0
        self._lock = threading.Lock()
        self._load()

    # ── Core I/O ─────────────────────────────────────────────────────

    def _load(self):
        """Load titan_settings.json from disk."""
        try:
            if SETTINGS_FILE.exists():
                mtime = os.path.getmtime(SETTINGS_FILE)
                if mtime != self._mtime:
                    with open(SETTINGS_FILE, 'r') as f:
                        self._data = json.load(f)
                    self._mtime = mtime
                    return True
        except Exception as e:
            print(f"⚠️ settings_manager load error: {e}")
        return False

    def _save(self):
        """Atomically write titan_settings.json."""
        tmp = SETTINGS_FILE.with_suffix('.tmp')
        try:
            self._data['_last_updated'] = datetime.now().isoformat()
            with open(tmp, 'w') as f:
                json.dump(self._data, f, indent=4, default=str)
            if os.name == 'nt' and SETTINGS_FILE.exists():
                SETTINGS_FILE.unlink()
            tmp.rename(SETTINGS_FILE)
            self._mtime = os.path.getmtime(SETTINGS_FILE)
        except Exception as e:
            print(f"⚠️ settings_manager save error: {e}")
            if tmp.exists():
                tmp.unlink()

    # ── Public API ───────────────────────────────────────────────────

    def get(self, json_key, default=None):
        """Read a value from titan_settings.json."""
        with self._lock:
            return self._data.get(json_key, default)

    def set(self, json_key, value):
        """Write a value to titan_settings.json AND update config in-memory."""
        with self._lock:
            self._data[json_key] = value
            self._save()
        # Push to config module in-memory
        self._apply_single(json_key, value)

    def set_many(self, updates: dict):
        """Write multiple values atomically."""
        with self._lock:
            self._data.update(updates)
            self._save()
        for k, v in updates.items():
            self._apply_single(k, v)

    def get_all(self) -> dict:
        """Return a copy of all settings."""
        with self._lock:
            return dict(self._data)

    def reload(self):
        """Re-read from disk if changed, then apply ALL to config module.
        Call this periodically from the bot main loop."""
        with self._lock:
            changed = self._load()
        if changed:
            self.apply_all()
            return True
        return False

    # ── Watcher helpers (used by dashboard) ──────────────────────────

    def get_watcher(self, key, default=None):
        """Get a BREAKOUT_WATCHER setting. Checks json first, then config.py."""
        json_key = f'watcher_{key}'
        with self._lock:
            if json_key in self._data:
                return self._data[json_key]
        # Fallback to config.py
        import config as _cfg
        bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
        return bw.get(key, default)

    def get_watcher_dict(self) -> dict:
        """Get all watcher tunables as {key: value} for dashboard display."""
        import config as _cfg
        bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
        result = {}
        for key in _WATCHER_KEYS:
            json_key = f'watcher_{key}'
            with self._lock:
                if json_key in self._data:
                    result[key] = self._data[json_key]
                else:
                    result[key] = bw.get(key)
        # momentum_exit.enabled
        with self._lock:
            if 'watcher_momentum_exit' in self._data:
                result['momentum_exit_enabled'] = self._data['watcher_momentum_exit']
            else:
                result['momentum_exit_enabled'] = bw.get('momentum_exit', {}).get('enabled', True)
        return result

    # ── Sync & Apply ─────────────────────────────────────────────────

    def sync_defaults(self):
        """Populate titan_settings.json with config.py defaults for any MISSING keys.
        Called once at startup. Never overwrites existing values."""
        import config as _cfg
        bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
        hr = getattr(_cfg, 'HARD_RULES', {})
        added = []

        with self._lock:
            # Watcher keys
            for key in _WATCHER_KEYS:
                json_key = f'watcher_{key}'
                if json_key not in self._data and key in bw:
                    self._data[json_key] = bw[key]
                    added.append(json_key)

            # momentum_exit.enabled
            if 'watcher_momentum_exit' not in self._data:
                self._data['watcher_momentum_exit'] = bw.get('momentum_exit', {}).get('enabled', True)
                added.append('watcher_momentum_exit')

            # Hard rules
            for key in _HARD_RULES_KEYS:
                if key not in self._data and key in hr:
                    self._data[key] = hr[key]
                    added.append(key)

            # Strategy toggles
            for name in _STRATEGY_NAMES:
                skey = f'strategy_{name}'
                if skey not in self._data:
                    cfg = getattr(_cfg, name, {})
                    if isinstance(cfg, dict):
                        self._data[skey] = cfg.get('enabled', True)
                        added.append(skey)

            # Lot multipliers
            for name, cfg_key in _LOT_STRATEGIES.items():
                lkey = f'lots_{name}'
                if lkey not in self._data:
                    cfg = getattr(_cfg, name, {})
                    if isinstance(cfg, dict) and cfg_key in cfg:
                        self._data[lkey] = cfg[cfg_key]
                        added.append(lkey)

            if added:
                self._save()

        if added:
            print(f"📋 settings_manager: synced {len(added)} defaults → titan_settings.json")

        return added

    def apply_all(self):
        """Push ALL titan_settings.json values → config module in-memory.
        Ensures config.py always reflects the persisted truth."""
        import config as _cfg
        bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
        hr = getattr(_cfg, 'HARD_RULES', {})
        applied = []

        with self._lock:
            data = dict(self._data)

        # Watcher keys
        for key in _WATCHER_KEYS:
            json_key = f'watcher_{key}'
            if json_key in data:
                old = bw.get(key)
                new = data[json_key]
                if old != new:
                    bw[key] = new
                    applied.append(f"BREAKOUT_WATCHER.{key}={new}")

        # momentum_exit.enabled
        if 'watcher_momentum_exit' in data:
            me = bw.get('momentum_exit', {})
            new_me = bool(data['watcher_momentum_exit'])
            if me.get('enabled') != new_me:
                me['enabled'] = new_me
                applied.append(f"momentum_exit.enabled={new_me}")

        # Hard rules
        for key in _HARD_RULES_KEYS:
            if key in data and hr.get(key) != data[key]:
                hr[key] = data[key]
                applied.append(f"HARD_RULES.{key}={data[key]}")

        # Strategy toggles
        for name in _STRATEGY_NAMES:
            skey = f'strategy_{name}'
            if skey in data:
                cfg = getattr(_cfg, name, None)
                if isinstance(cfg, dict) and 'enabled' in cfg:
                    new_val = bool(data[skey])
                    if cfg['enabled'] != new_val:
                        cfg['enabled'] = new_val
                        applied.append(f"{name}.enabled={new_val}")

        # Lot multipliers
        for name, cfg_key in _LOT_STRATEGIES.items():
            lkey = f'lots_{name}'
            if lkey in data:
                cfg = getattr(_cfg, name, None)
                if isinstance(cfg, dict):
                    new_v = float(data[lkey])
                    if cfg.get(cfg_key) != new_v:
                        cfg[cfg_key] = new_v
                        applied.append(f"{name}.{cfg_key}={new_v}")

        # Global lot multiplier (stored in HARD_RULES at runtime)
        if 'global_lot_multiplier' in data:
            new_glm = float(data['global_lot_multiplier'])
            if hr.get('GLOBAL_LOT_MULTIPLIER') != new_glm:
                hr['GLOBAL_LOT_MULTIPLIER'] = new_glm
                applied.append(f"GLOBAL_LOT_MULTIPLIER={new_glm}")

        # Kill switch
        if data.get('kill_switch', False):
            hr['MAX_POSITIONS'] = 0
            applied.append("KILL_SWITCH=ON → MAX_POSITIONS=0")

        # Trading hours
        th = getattr(_cfg, 'TRADING_HOURS', {})
        for skey, cfgkey in [('hours_start', 'start'), ('hours_end', 'end'),
                              ('hours_no_new_after', 'no_new_after')]:
            if skey in data and th.get(cfgkey) != data[skey]:
                th[cfgkey] = data[skey]
                applied.append(f"TRADING_HOURS.{cfgkey}={data[skey]}")

        if applied:
            print(f"🔄 settings_manager: applied {len(applied)} overrides → {', '.join(applied[:5])}{'...' if len(applied)>5 else ''}")

        return applied

    # ── Internal ─────────────────────────────────────────────────────

    def _apply_single(self, json_key, value):
        """Push a single setting to config module in-memory."""
        import config as _cfg

        # Watcher key?
        if json_key.startswith('watcher_') and json_key != 'watcher_momentum_exit':
            cfg_key = json_key[len('watcher_'):]
            bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
            if cfg_key in bw or cfg_key in _WATCHER_KEYS:
                bw[cfg_key] = value

        elif json_key == 'watcher_momentum_exit':
            bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
            me = bw.get('momentum_exit', {})
            me['enabled'] = bool(value)

        elif json_key.startswith('strategy_'):
            name = json_key[len('strategy_'):]
            cfg = getattr(_cfg, name, None)
            if isinstance(cfg, dict) and 'enabled' in cfg:
                cfg['enabled'] = bool(value)

        elif json_key.startswith('lots_'):
            name = json_key[len('lots_'):]
            cfg_key = _LOT_STRATEGIES.get(name, 'lot_multiplier')
            cfg = getattr(_cfg, name, None)
            if isinstance(cfg, dict):
                cfg[cfg_key] = float(value)

        elif json_key in _HARD_RULES_KEYS:
            hr = getattr(_cfg, 'HARD_RULES', {})
            hr[json_key] = value

        elif json_key == 'global_lot_multiplier':
            hr = getattr(_cfg, 'HARD_RULES', {})
            hr['GLOBAL_LOT_MULTIPLIER'] = float(value)

        elif json_key == 'kill_switch' and value:
            hr = getattr(_cfg, 'HARD_RULES', {})
            hr['MAX_POSITIONS'] = 0

        elif json_key.startswith('hours_'):
            suffix = json_key[len('hours_'):]
            th = getattr(_cfg, 'TRADING_HOURS', {})
            th[suffix] = value


# ── Module-level singleton ───────────────────────────────────────────
settings = SettingsManager()
