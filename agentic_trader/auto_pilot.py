"""
Auto-Pilot Agent — real-time reverse/+1 adjuster for naked-option positions.

Runs as a daemon thread inside the titan-bot process. Every 1 s it:
  1. Pulls live quote data from the shared TitanTicker cache for every open
     NAKED_OPTION position.
  2. Extracts short-horizon features (velocity, depth imbalance, volume surge,
     tape aggression).
  3. If aggressive rule gates fire → asks GPT-5.2 to confirm/veto the action.
  4. On GPT confirm → POSTs to the dashboard's own endpoints:
        /api/reverse_trade  — flip PE↔CE at same strike
        /api/add_lot        — strengthen with +1 lot at LTP
     Both endpoints already enforce per-symbol cooldowns + in-flight locks,
     so the agent can't run away.

Scope:
    - Only NAKED_OPTION positions (skip credit/debit spreads, iron condors)
    - Trading window: 09:20 – 15:10 IST
    - Min hold before reverse: 60 s (avoids tick-flicker flips)
    - Max reverses per symbol per day: 7
    - Max +1 adds per symbol per day: 3
    - No daily loss kill-switch (user's choice)

Persistence:
    auto_pilot_state.json
        {
          "enabled": bool,
          "date": "YYYY-MM-DD",
          "counters": {symbol: {"reverses": int, "adds": int}},
          "last_action_ts": {symbol: epoch_seconds},
          "last_decision": [{...recent decisions for UI...}]
        }

Controlled via:
    GET  /api/auto_pilot            — returns state
    POST /api/auto_pilot {enabled}  — toggles on/off
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error

try:
    from config import OPENAI_API_KEY
except Exception:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

try:
    import openai  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# exit_manager singleton — used to check if a position has already armed its
# trailing SL. Once trail is active, let the exit manager run the exit; AP
# should not second-guess it with a REVERSE / ADD_LOT.
try:
    from exit_manager import get_exit_manager  # type: ignore
except Exception:
    get_exit_manager = None  # type: ignore


STATE_PATH = os.path.join(os.path.dirname(__file__), "auto_pilot_state.json")
DISABLE_FLAG_PATH = os.path.join(os.path.dirname(__file__), "auto_pilot_disabled.flag")
DASHBOARD_URL = os.environ.get("TITAN_DASHBOARD_URL", "http://127.0.0.1:5000")


# ── Slope-driven thresholds ─────────────────────────────────────────
# Philosophy: read the UNDERLYING's tick-by-tick price (option premium is a
# lagging, noisy echo). Fit a short OLS slope on the last few seconds, require
# the slope to SUSTAIN (same-sign seconds) and reject CHOPPY windows. Act on
# the underlying's trend, not on option-premium P&L wiggles.
class Thresholds:
    TICK_INTERVAL_SEC = 1.0               # feature-extraction cadence
    WINDOW_SEC = 320                      # rolling underlying-tick window (5-min macro + 20s buffer)
    SLOPE_WINDOW_SEC = 5                  # primary slope regression window
    CONFIRM_WINDOW_SEC = 3                # short confirmation window (must agree in sign)
    MACRO_WINDOW_SEC = 60                 # higher-timeframe trend filter
    MACRO5M_WINDOW_SEC = 300              # 5-minute timeframe — catches chart-visible trend
                                          # reversals that short-term filters (5s/3s/60s) miss
                                          # because their thresholds are designed for tick noise,
                                          # not 5-min drifts of 0.3-0.6 bps/s.
    # SUSTAIN_SEC tuned 3 → 0 after tick-level calibration: |slope_5s_bps| is
    # an OLS regression over 5 seconds — it already encodes sustained direction.
    # Layering a 'last 1-sec delta must match' on top is ~50/50 noise from bid/ask
    # jitter and was zeroing out the gate. Set to 0 → sustain gate is effectively
    # subsumed by the slope regression.
    SUSTAIN_SEC = 0                       # 0 disables sustain check (slope5s is the sustain)

    # Slope magnitudes in bps/sec on the UNDERLYING
    #   1 bps/sec ≈ 0.6 %/min — i.e. a normal intraday drift
    #   2 bps/sec ≈ 1.2 %/min — a decisive move
    STEEP_SLOPE_BPS_S = 2.0               # required to ADD_LOT in direction
    MILD_SLOPE_BPS_S = 0.5                # below this the move is "flat" (noise)
    REVERSE_MIN_BPS_S = 0.6               # minimum counter-slope magnitude to REVERSE (relaxed 0.8→0.6)
    # REVERSE also requires the 5-sec counter-slope to be stronger than the macro
    # trend (break-the-macro check) by this multiplier — stops retracement flips.
    REVERSE_VS_MACRO_RATIO = 1.5

    # Adaptive thresholds: scale STEEP/REVERSE thresholds by the instrument's
    # recent realised vol of 1-sec log-returns (annualised into bps/sec terms).
    # Effective threshold = max(MIN, base * vol_ratio).
    VOL_ADAPTIVE = True
    VOL_LOOKBACK_SEC = 60                 # realised-vol window (we only have WINDOW_SEC=75 buffer)
    VOL_BASELINE_BPS_S = 1.0              # baseline stdev of 1-sec returns in bps — "normal" tape
    STEEP_FLOOR_BPS_S = 0.4               # never let adaptive STEEP fall below this
    REVERSE_FLOOR_BPS_S = 0.2             # never let adaptive REVERSE fall below this

    # Flash-spike filter: reject REVERSE if 5-sec slope direction OPPOSES the
    # cumulative 15-sec return (i.e. the "dip" is a reversion toward the 15s
    # mean, not a trend break). Catches dip-and-snap-back whipsaws.
    FLASH_LOOKBACK_SEC = 15

    # Wick-rejection filter: reject REVERSE if the 30-sec net move is small
    # compared to the 30-sec high-low RANGE (V-shape wick-and-bounce). A real
    # trend break travels net ≈ range; a wick travels far and comes back.
    WICK_LOOKBACK_SEC = 30
    WICK_NET_OVER_RANGE_MIN = 0.4         # |ret_30s| / range_30s must be ≥ this

    # Exhaustion-spike guard: reject REVERSE when the 15-sec cumulative move
    # is TOO LARGE in the same direction as the counter-slope. A ≥10 bps
    # (0.10%) move over 15 seconds is capitulation / climactic spike — the
    # next few minutes typically mean-revert. Flipping INTO the spike = buying
    # the extreme. Real trend-breaks build over minutes, not seconds.
    # Seen live on PERSISTENT 2026-04-24 12:02: AP reversed on ret_15s=-12.4bps
    # (a capitulation down-spike) right at the 5-min bottom; PE then bled -10%
    # as underlying mean-reverted up.
    EXHAUSTION_RET15_BPS = 10.0           # |ret_15s| ≥ this + same sign as slope5 → reject REVERSE

    # Macro-magnitude floor: the 60-sec macro must have REAL direction, not
    # noise near zero. Without this, a 15-sec bounce can nudge slope_60s to
    # +0.05 bps/s and pass the sign-agreement check even when the day-level
    # trend is clearly against the counter-slope.
    # Seen live on INFY 2026-04-24 12:16: AP reversed PE→CE on macro=+0.06bps/s
    # (basically 0) while INFY was in a 3-hour 1220→1180 downtrend; flipped CE
    # immediately bled as underlying resumed down.
    MACRO_MIN_ABS_BPS_S = 0.08            # |slope_60s| must be ≥ this for counter-slope REVERSE (relaxed 0.15→0.08)

    # Multi-timeframe macro confirmation: if slope_300s (5-min) is computable
    # (non-zero = window warmed) and has the SAME sign as trade_dir, the
    # position is on the right side of the multi-minute trend. The current
    # bleed is theta/IV noise, NOT a trend flip. Reject counter-slope REVERSE.
    # Also protects against the "post-restart blind spot" (first ~2 min after
    # titan-bot restart, slope_300s is still warming at 0).
    REQUIRE_MACRO5M_AGAINST_FOR_REV = True

    # Option-liquidity / spread gate: block AP actions when the option's own
    # bid-ask spread is too wide — flipping through a fat spread pays most of
    # the edge straight to the market-maker. Computed from L1 depth as
    # (ask - bid) / mid * 100.
    # Typical values on NFO weeklies: 0.3-1.5% on ATM, 2-5% on OTM, 5-15% on deep OTM.
    MAX_OPTION_SPREAD_PCT = 3.0           # above this: skip AP action
    MAX_OPTION_SPREAD_PCT_ADD = 2.0       # tighter for ADDs (no urgency; wait for liquidity)

    # CHOP_FLIP_LIMIT retuned 3 → 12 after tick-level calibration:
    # on live NFO underlying tape, 1-sec delta sign flips occur 10-25 times /
    # 75-sec window even during a clean 20% drawdown. CHOP<3 was passing only
    # ~2% of ticks → gate effectively permanently closed. Real chop looks
    # like 25+ flips/window; normal trending tape sits in the 8-15 range.
    CHOP_FLIP_LIMIT = 12                  # ≥ this many slope-sign flips in WINDOW_SEC → chop, skip

    # Deep-bleed override: when the option's pnl_pct is this red AND the
    # underlying slope is against us (any magnitude above BLEED_MIN_SLOPE),
    # fire a REVERSE even if sustain/wick/flash filters would normally block.
    # Rationale: those filters exist to avoid reversing into wicks / spikes.
    # A 15%+ drawdown is by definition NOT a wick — it's a sustained move.
    BLEED_PNL_PCT = -8.0                  # pnl% at which bleed-override arms (relaxed -12→-8)
    BLEED_MIN_SLOPE_BPS_S = 0.3           # min |slope5s| against us to fire override

    # Momentum-ADD override (symmetric to BLEED): when position is already
    # meaningfully in profit AND underlying slope still agrees, fire ADD_LOT
    # even if |slope5s| hasn't reached the normal STEEP threshold (2.0 bps/s).
    # Rationale: a winning position with aligned drift is the ideal pyramid
    # candidate — don't wait for a fresh slope spike that may never come.
    # Note: hands-off guard on trailing-active symbols still applies upstream,
    # so this only fires on winners that haven't armed their trailing stop.
    MOMENTUM_PNL_PCT = 8.0                # pnl% at which momentum-ADD arms
    MOMENTUM_MIN_SLOPE_BPS_S = 1.0        # min |slope5s| aligned with us to fire

    MIN_HOLD_AFTER_ENTRY = 15             # don't touch a position in its first 15s
    COOLDOWN_BETWEEN_DECISIONS = 15       # min seconds between any two decisions on same symbol
    # Post-REVERSE per-symbol lockout. Once we flip a symbol, freeze AP on it
    # for this many seconds regardless of signal. Kills the whipsaw loop we
    # saw on JIOFIN / INFY where a reverse at -12% immediately got another
    # -12% reverse on the new leg, compounding the loss.
    POST_REVERSE_LOCKOUT_SEC = 120
    # Per-symbol cap. Raised 10 → 50 per user directive 2026-04-24:
    # "I don't care how many reversals, cut my loss wherever it occurs".
    # 120s lockout + chop/exhaustion gates still bound ping-pong in practice.
    MAX_REVERSES_PER_DAY = 50
    MAX_ADDS_PER_DAY = 5

    # Emergency re-reverse escape: allow ONE REVERSE to bypass the post-reverse
    # lockout when the previous flip is clearly wrong. Conditions (all required):
    #   - proposed action is REVERSE
    #   - pnl_pct on the newly-flipped leg ≤ EMERGENCY_REREV_PNL_PCT (deep red)
    #   - |slope_5s_bps| ≥ EMERGENCY_REREV_SLOPE_BPS_S (strong counter signal)
    #   - sign(slope_60s_bps) == sign(slope_5s_bps) (macro agrees, not a wick)
    #   - chop_flips < AUTO_VETO_CHOP_FLIPS
    #   - escape hasn't been used for this lockout period yet
    # Rationale: intraday we don't carry bad flips — if the market tells us
    # within seconds that the reverse was wrong, flip back once. Can't ping-pong
    # because each reverse gets exactly one escape.
    EMERGENCY_REREV_PNL_PCT = -6.0
    EMERGENCY_REREV_SLOPE_BPS_S = 1.2

    # Pre-LLM auto-veto: when chop_flips is this high, even bleed-override
    # and momentum-override skip firing. The normal REV path already rejects
    # chop >= CHOP_FLIP_LIMIT (12), but the two overrides currently bypass
    # that filter. In brutal chop (>=30 flips/60s), any reverse is just
    # another whipsaw entry — saves LLM tokens and prevents chop losses.
    AUTO_VETO_CHOP_FLIPS = 40             # relaxed 30→40 (bleed-override chop cap)

    # Macro-5min REVERSE path. Fires when the 5-minute underlying slope has
    # turned strongly against the position, EVEN IF short-term filters
    # (sustain/3s-agree/macro60s-agree/wick/flash) would block. Designed to
    # catch the chart-visible scenario where user says "the slope reversed,
    # why didn't AP flip?" — a sustained multi-minute move that the tick-noise
    # oriented short-term gates never see.
    #
    # Examples calibrated from real tape:
    #   0.4 bps/s over 5min = ~120 bps = 1.2% underlying move against us
    #   0.6 bps/s over 5min = ~180 bps = 1.8% underlying move against us
    # For a typical ATM option with |delta| ≈ 0.5, 1.5% underlying = 0.75% premium
    # swing — enough to justify flipping.
    # Loosened 2026-04-24 per user directive ("cut my loss wherever it occurs"):
    # 0.4 → 0.10 bps/s (30 bps over 5min = ~0.3% underlying) so slow grinds
    # like WAAREEENER/UNOMINDA trigger before they round-trip.
    # -5 → -3 so we don't wait for the premium quote to catch up before flipping.
    MACRO5M_REVERSE_BPS_S = 0.06          # min |slope_300s| against us to trigger (relaxed 0.10→0.06)
    MACRO5M_PNL_PCT = -1.5                # require pnl% ≤ -1.5% (relaxed -3→-1.5)
    MACRO5M_MAX_CHOP_FLIPS = 25           # skip if tape is choppier than this (whipsaw risk)

    # Clean-tape drift REVERSE. Catches the "premium-stale" scenario: low-liquidity
    # stock where option quote is frozen while the underlying quietly drifts against
    # us. Tick slopes never spike (below normal REV 0.8 bar), pnl stays shallow
    # because the option hasn't repriced yet — but by the time quote catches up,
    # the move has already round-tripped.
    # Seen 2026-04-24 on WAAREEENER 3300PE (underlying +36 bps over 25min, opt_ltp
    # frozen 60s at a time, pnl stuck at -3.4% while AP slopes showed +0.75 peak)
    # and UNOMINDA 1100PE (s5 never exceeded 0.62 on a clear up-move).
    # Uses u_drift_against_bps (underlying move from entry, signed against trade_dir)
    # as the primary trigger — it's the ground truth when tick slopes / premium lag.
    # All four conditions required; conjunction makes false positives rare.
    CLEAN_DRIFT_MIN_HOLD = 180            # 3 min minimum (relaxed 300→180)
    CLEAN_DRIFT_MAX_CHOP = 25             # clean tape (relaxed 15→25)
    CLEAN_DRIFT_MIN_U_DRIFT_BPS = 5.0     # underlying moved ≥5 bps (relaxed 10→5)
    CLEAN_DRIFT_PNL_PCT = -1.0            # any shallow red (relaxed -2→-1)

    # Macro-5min ADD_LOT path (symmetric to macro5m REVERSE). Fires when the
    # 5-minute underlying slope is with the position and the trade is running
    # profitably, even if 5s/3s filters are too small to trip STEEP_SLOPE_BPS_S.
    # Designed to catch the scenario the user flagged on ADANIENT: a slow
    # 0.2%/min drift over 10+ minutes that the short-term ADD gate never sees.
    MACRO5M_ADD_BPS_S = 0.35              # min |slope_300s| WITH us to pyramid
    MACRO5M_ADD_PNL_PCT = 4.0             # require pnl% ≥ +4% (already winning)

    # ── Macro-trend CHOP-BYPASS (Option A) ──
    # When evidence is overwhelming (severe loss/profit + large entry-to-now
    # underlying drift), tape chop should NOT block action. This catches the
    # stairstep-grind scenario (e.g. RELIANCE 2026-04-27: 1325→1345 over 3h,
    # chop=89 blocking macro5m + clean-drift even though the chart-visible
    # move was unambiguous). Higher floors than CLEAN_DRIFT to compensate
    # for the dropped chop guard.
    BYPASS_REV_MIN_HOLD = 300             # 5 min minimum (filters entry noise)
    BYPASS_REV_PNL_PCT = -6.0             # severe red (≥4× CLEAN_DRIFT floor)
    BYPASS_REV_U_DRIFT_BPS = 25.0         # 25 bps = 0.25% underlying move (5× CLEAN_DRIFT)
    BYPASS_ADD_MIN_HOLD = 300
    BYPASS_ADD_PNL_PCT = 4.0              # already running
    BYPASS_ADD_U_DRIFT_WITH_BPS = 25.0    # underlying drifted WITH us by ≥25 bps

    # Stale-loss REVERSE path. Catches the SBILIFE-style scenario where AP
    # entered, underlying drifted a little against us, then parked there in
    # chop for 10+ minutes. Every slope gate is flat (no *current* move) yet
    # we are quietly bleeding. Uses cumulative drift-from-entry on the
    # underlying as the signal instead of a slope. Fires when ALL hold:
    #   - pnl_pct ≤ STALE_LOSS_PNL_PCT     (position is meaningfully red)
    #   - hold_sec ≥ STALE_LOSS_MIN_HOLD   (been stuck long enough)
    #   - u_drift_against_bps ≥ STALE_LOSS_DRIFT_BPS  (underlying moved against us)
    #   - chop_flips < STALE_LOSS_MAX_CHOP (high tolerance — feature is chop-insensitive)
    # Bypasses the normal chop/sustain/slope filters like bleed-override does.
    STALE_LOSS_PNL_PCT = -4.0             # relaxed -6→-4
    STALE_LOSS_MIN_HOLD = 300             # 5 minutes (relaxed 600→300)
    STALE_LOSS_DRIFT_BPS = 8.0            # 0.08% adverse on underlying (relaxed 15→8)
    STALE_LOSS_MAX_CHOP = 110             # only skip when tape is a pure washing machine

    # Medium-bleed REVERSE path. Fills the gap between the normal REV gate
    # (which is blocked by even mild chop) and bleed-override (which requires
    # -12% pnl). Catches the LUPIN-style scenario: position is moderately red,
    # a strong counter-spike is in progress with 3s+60s confirmation, but NO
    # single dimension is extreme enough to trip an existing gate. Fires only
    # when FOUR medium conditions co-occur — the multi-axis evidence makes up
    # for not having one extreme axis.
    MEDIUM_BLEED_PNL_PCT = -5.0           # meaningful red (relaxed -7→-5)
    MEDIUM_BLEED_MIN_SLOPE_BPS_S = 0.7    # spike magnitude (relaxed 1.0→0.7)
    MEDIUM_BLEED_MAX_CHOP = 45            # medium tape (relaxed 35→45)
    MEDIUM_BLEED_MIN_MACRO_BPS_S = 0.05   # macro60s must at least mildly agree

    # Profit-peak REVERSE path. Flips a WINNING position at the top/bottom of
    # a swing when a counter-slope is forming. Existing REVERSE gates all
    # require pnl% ≤ negative — so AP never flips green positions. Manual
    # trader (2026-04-24 VEDL) flipped 5× between CE/PE same day, banking
    # profit each swing; every flip was while in profit (peak catch), not
    # after bleeding. This gate replicates that behavior. Symmetric (fires
    # equally on long-CE peaks and long-PE peaks) and can re-fire up to
    # MAX_REVERSES_PER_DAY (50), enabling back-and-forth peak harvesting
    # on oscillating names.
    PROFIT_PEAK_PNL_PCT = 3.0             # need ≥+3% open profit
    PROFIT_PEAK_MIN_HOLD = 60             # let the winner develop first
    PROFIT_PEAK_MAX_CHOP = 35             # skip pure whipsaw tape
    PROFIT_PEAK_MIN_S5_BPS_S = 0.8        # counter-slope magnitude
    PROFIT_PEAK_MIN_S3_BPS_S = 1.2        # 3s acceleration confirms turn

    # Per-symbol realized-loss cap. After a single underlying has accumulated
    # this much realized loss in today's ledger (from ANY source — AP reverses,
    # SL hits, time stops), AutoPilot freezes on that symbol for the rest of
    # the session. Base trader SL/target and exit_manager still run normally;
    # only AP reverse/add_lot decisions are blocked. Prevents a single ticker
    # from becoming a money pit (the INFY-style rot scenario).
    PER_SYMBOL_REALIZED_LOSS_CAP = -3000.0
    PER_SYMBOL_CAP_REFRESH_SEC = 30  # ledger-scan cache interval

    # Kept for LLM context only (NOT gating)
    VOL_SURGE_MULTIPLIER = 1.3
    DEPTH_IMBALANCE_RATIO = 1.25

    TRADING_WINDOW_START = (9, 20)        # 09:20 IST
    TRADING_WINDOW_END = (15, 10)         # 15:10 IST


class AutoPilot:
    """Background decision loop. Created by autonomous_trader."""

    def __init__(self, tools_ref):
        # tools_ref is the ZerodhaTools singleton (so we can read paper_positions + ticker)
        self.tools = tools_ref
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Per-symbol rolling tick windows: {symbol: deque[(ts, ltp, buy_qty, sell_qty, volume)]}
        # Also stores float timestamps under sentinel keys (e.g. "__diag_{sym}__") for log throttling.
        self._windows: Dict[str, Any] = {}

        # Per-underlying realized-loss cache + refresh timestamp.
        # Populated from trade_ledger_{today}.jsonl every PER_SYMBOL_CAP_REFRESH_SEC.
        # Used to gate AP actions on symbols that have already bled out.
        self._realized_by_underlying: Dict[str, float] = {}
        self._realized_cache_ts: float = 0.0

        # Persistent state
        self._state = self._load_state()

        # LLM client
        self._llm: Any = None
        if _HAS_OPENAI and OPENAI_API_KEY:
            try:
                self._llm = openai.OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                print(f"⚠️ AutoPilot: OpenAI init failed: {e}")
        else:
            print("⚠️ AutoPilot: openai not available — will run in rule-only mode")

        # === Concurrency hardening (Apr 27) ============================
        # Per (symbol, action) LLM verdict cache so repeating gates don't re-pay
        # LLM cost. TTL kept short so signals stay fresh during real trends.
        # Halves typical-tick LLM cost when 5+ symbols share a regime.
        self._llm_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
        self._llm_cache_ttl: float = 10.0  # seconds
        # Bounded thread pool for parallel LLM verdicts within a single tick.
        # max_workers=4 — enough to absorb 8–10 simultaneous candidates without
        # overwhelming OpenAI rate limits. State mutations (counters, exec)
        # remain SEQUENTIAL after futures resolve, so no race conditions.
        self._llm_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ap_llm")

    # ── Public control ──────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="AutoPilot", daemon=True)
        self._thread.start()
        enabled = self.is_enabled()
        print(f"🤖 AutoPilot thread started (enabled={enabled})")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self._llm_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def is_enabled(self) -> bool:
        if os.path.exists(DISABLE_FLAG_PATH):
            return False
        # Re-read the state file every time so dashboard toggles (written by
        # the dashboard process) are picked up without restarting the bot.
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    disk = json.load(f) or {}
                if isinstance(disk, dict) and "enabled" in disk:
                    # sync into in-memory state so counters / recent_decisions
                    # read paths stay coherent
                    with self._lock:
                        self._state["enabled"] = bool(disk.get("enabled"))
                    return bool(disk.get("enabled"))
        except Exception:
            pass
        return bool(self._state.get("enabled", False))

    def set_enabled(self, on: bool) -> Dict[str, Any]:
        with self._lock:
            self._state["enabled"] = bool(on)
            self._save_state()
        if on:
            try:
                if os.path.exists(DISABLE_FLAG_PATH):
                    os.remove(DISABLE_FLAG_PATH)
            except Exception:
                pass
        print(f"🤖 AutoPilot toggled → {'ON' if on else 'OFF'}")
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            if self._state.get("date") != today:
                self._state["date"] = today
                self._state["counters"] = {}
                self._state["last_action_ts"] = {}
                self._save_state()
            return {
                "enabled": self.is_enabled(),
                "date": self._state.get("date"),
                "counters": dict(self._state.get("counters", {})),
                "limits": {
                    "max_reverses_per_symbol": Thresholds.MAX_REVERSES_PER_DAY,
                    "max_adds_per_symbol": Thresholds.MAX_ADDS_PER_DAY,
                },
                "recent_decisions": list(self._state.get("last_decision", []))[-10:],
                "scope": "NAKED_OPTION only",
                "aggression": "Aggressive",
                "llm": "gpt-5.2" if self._llm else "rules-only",
            }

    # ── Core loop ───────────────────────────────────────────────────
    def _loop(self):
        self._last_heartbeat = 0.0
        self._skip_counters = {
            "no_positions": 0, "no_quote": 0, "no_ltp": 0,
            "warmup": 0, "no_gate": 0, "cooldown": 0, "day_cap": 0,
            "wide_spread": 0, "trailing_active": 0,
            "vetoed": 0, "executed": 0,
        }
        while not self._stop_event.is_set():
            try:
                if self.is_enabled() and self._in_trading_window():
                    self._tick()
                    # Heartbeat every 15s so user can see the agent is alive
                    now = time.time()
                    if now - self._last_heartbeat >= 15:
                        self._last_heartbeat = now
                        sc = self._skip_counters
                        print(
                            f"💓 AutoPilot heartbeat | last-15s: "
                            f"no_pos={sc['no_positions']} no_quote={sc['no_quote']} "
                            f"no_ltp={sc['no_ltp']} warmup={sc['warmup']} "
                            f"no_gate={sc['no_gate']} cooldown={sc['cooldown']} "
                            f"capped={sc['day_cap']} wide_spread={sc['wide_spread']} "
                            f"trailing={sc['trailing_active']} "
                            f"vetoed={sc['vetoed']} executed={sc['executed']}",
                            flush=True,
                        )
                        for k in sc:
                            sc[k] = 0
            except Exception as e:
                print(f"⚠️ AutoPilot tick error: {e}", flush=True)
                traceback.print_exc()
            self._stop_event.wait(Thresholds.TICK_INTERVAL_SEC)

    def _in_trading_window(self) -> bool:
        now = datetime.now().time()
        sh, sm = Thresholds.TRADING_WINDOW_START
        eh, em = Thresholds.TRADING_WINDOW_END
        from datetime import time as _t
        return _t(sh, sm) <= now <= _t(eh, em)

    def _tick(self):
        positions = self._get_naked_option_positions()
        if not positions:
            self._skip_counters["no_positions"] += 1
            return

        now_wall = time.time()
        symbols_seen = set()

        # Pending LLM verdicts collected during the per-symbol loop. Drained
        # at the end of the tick so multiple symbols' LLM round-trips run
        # concurrently in the bounded thread pool while gate-eval stays
        # strictly sequential. Tuple format:
        # (pos, features, decision, sym, counters, gate_ts, future)
        _pending_llm: List[Tuple[dict, dict, dict, str, dict, float, Any]] = []

        # Refresh per-underlying realized-loss cache once per tick (cached
        # internally for PER_SYMBOL_CAP_REFRESH_SEC so this is cheap).
        self._refresh_realized_by_underlying(now_wall)

        for pos in positions:
            sym = pos.get("symbol") or pos.get("option_symbol") or ""
            if not sym:
                continue
            symbols_seen.add(sym)

            # Hands-off trailing-active state is computed here but enforced
            # AFTER the rule gate (so ADD_LOT can still pyramid a winner that's
            # already in harvest mode; only REVERSE is blocked because the
            # trail is already handling the exit).
            trailing_is_active = False
            trailing_maxR = 0.0
            if get_exit_manager is not None:
                try:
                    em = get_exit_manager()
                    em_state = em.trade_states.get(sym) if em else None
                    if em_state is not None and getattr(em_state, "trailing_active", False):
                        trailing_is_active = True
                        trailing_maxR = float(getattr(em_state, "max_favorable_move", 0) or 0)
                except Exception:
                    pass

            quote = self._get_quote(sym)
            if not quote:
                self._skip_counters["no_quote"] += 1
                continue

            ltp = float(quote.get("ltp") or 0)
            if ltp <= 0:
                self._skip_counters["no_ltp"] += 1
                continue
            buy_qty = int(quote.get("buy_qty") or 0)
            sell_qty = int(quote.get("sell_qty") or 0)
            volume = int(quote.get("volume") or 0)

            # Underlying LTP — the real signal. Skip tick if unavailable
            # (we MUST have an underlying series to compute slope).
            u_ltp = self._get_underlying_ltp(pos)
            if not u_ltp:
                self._skip_counters["no_quote"] += 1
                continue

            win = self._windows.setdefault(sym, deque(maxlen=int(Thresholds.WINDOW_SEC / Thresholds.TICK_INTERVAL_SEC)))
            win.append((now_wall, ltp, buy_qty, sell_qty, volume, u_ltp))

            features = self._extract_features(win, pos)
            if not features:
                self._skip_counters["warmup"] += 1
                continue

            # Per-symbol diagnostic log every 20s so the user can see the
            # *live* slope picture for every open position.
            last_diag = self._windows.get(f"__diag_{sym}__", 0)
            if now_wall - last_diag >= 20:
                self._windows[f"__diag_{sym}__"] = now_wall
                print(
                    f"🔎 AutoPilot eval {sym} "
                    f"u_ltp={features['u_ltp']} opt_ltp={features['ltp']} "
                    f"trade_dir={'+' if features['trade_dir']>0 else '-'} "
                    f"slope5s={features['slope_5s_bps']:+.2f}bps/s "
                    f"slope3s={features['slope_3s_bps']:+.2f}bps/s "
                    f"slope60s={features['slope_60s_bps']:+.2f}bps/s "
                    f"slope300s={features.get('slope_300s_bps', 0.0):+.2f}bps/s "
                    f"rv={features['rv_bps']:.2f}bps ret15={features['ret_15s_bps']:+.1f}bps "
                    f"sustain={features['sustain_sec']}s "
                    f"chop={features['chop_flips']} "
                    f"aligned={features['aligned']} "
                    f"pnl%={features['pnl_pct']:+.2f} hold={features['hold_sec']}s "
                    f"| gates: ADD(|s5|≥{Thresholds.STEEP_SLOPE_BPS_S} aligned sustain≥{Thresholds.SUSTAIN_SEC} no-chop) "
                    f"REV(|s5|≥{Thresholds.REVERSE_MIN_BPS_S} against sustain≥{Thresholds.SUSTAIN_SEC} no-chop)",
                    flush=True,
                )

            decision = self._rule_gate(pos, features)
            if not decision:
                self._skip_counters["no_gate"] += 1
                continue

            # Trailing-active selective guard: REVERSE is blocked (exit_manager
            # trail is already handling the downside); ADD_LOT is allowed so
            # we can still pyramid a winner that's in harvest mode.
            if trailing_is_active and decision["action"] == "REVERSE":
                self._skip_counters["trailing_active"] += 1
                last_ta = self._windows.get(f"__ta_log_{sym}__", 0)
                if now_wall - last_ta >= 30:
                    self._windows[f"__ta_log_{sym}__"] = now_wall
                    print(
                        f"🛡️ AutoPilot HANDS_OFF {sym} REVERSE — trailing SL active, "
                        f"deferring to exit_manager (maxR={trailing_maxR:.2f})",
                        flush=True,
                    )
                continue

            # Cooldown guard to prevent decision flood on same symbol
            last_ts = self._state.get("last_action_ts", {}).get(sym, 0.0)
            if now_wall - last_ts < Thresholds.COOLDOWN_BETWEEN_DECISIONS:
                self._skip_counters["cooldown"] += 1
                continue

            # Per-underlying realized-loss cap. If this ticker has already bled
            # past the cap today (from any source — AP reverses, base trader
            # SL, time stops), freeze AP on it. Base trader SL/target keep
            # running; only AP decisions (add_lot / reverse) are gated.
            if self._is_symbol_loss_capped(pos):
                self._skip_counters["cooldown"] += 1
                under_k = pos.get('underlying') or ''
                log_key = f"__loss_cap_log_{under_k}__"
                last_lk = self._windows.get(log_key, 0)
                if now_wall - last_lk >= 60:
                    self._windows[log_key] = now_wall
                    realized_v = self._realized_by_underlying.get(under_k, 0.0)
                    print(
                        f"🚫 AutoPilot LOSS_CAP {under_k} — realized ₹{realized_v:,.0f} "
                        f"≤ cap ₹{Thresholds.PER_SYMBOL_REALIZED_LOSS_CAP:,.0f} (frozen for today)",
                        flush=True,
                    )
                continue

            # Post-REVERSE per-symbol lockout — after flipping this symbol,
            # freeze AP on it for POST_REVERSE_LOCKOUT_SEC regardless of signal.
            # Kills the -12% → flip → -12% → flip whipsaw loop.
            # EXCEPT: emergency re-reverse escape (one per lockout) when the
            # flip was clearly wrong — deep red on new leg + strong macro-
            # agreed counter-slope. Intraday: don't sit duck on a bad flip.
            last_rev_ts = self._state.get("last_reverse_ts", {}).get(sym, 0.0)
            if now_wall - last_rev_ts < Thresholds.POST_REVERSE_LOCKOUT_SEC:
                # Evaluate escape predicate
                escape_used_map = self._state.setdefault("rerev_escape_used_ts", {})
                already_used_for = escape_used_map.get(sym, 0.0)
                escape_available = (already_used_for != last_rev_ts)

                s5 = float(features.get("slope_5s_bps") or 0.0)
                s60 = float(features.get("slope_60s_bps") or 0.0)
                pnl_pct = float(features.get("pnl_pct") or 0.0)
                chop = int(features.get("chop_flips") or 0)
                trade_dir = int(features.get("trade_dir") or 0)

                # For REVERSE on a long-option leg, slope_5s must push AGAINST
                # trade_dir (so sign(s5) * trade_dir < 0) with strong magnitude,
                # and slope_60s must confirm (same sign as s5).
                sign5 = 1 if s5 > 0 else (-1 if s5 < 0 else 0)
                sign60 = 1 if s60 > 0 else (-1 if s60 < 0 else 0)
                is_emergency = (
                    decision["action"] == "REVERSE"
                    and escape_available
                    and pnl_pct <= Thresholds.EMERGENCY_REREV_PNL_PCT
                    and abs(s5) >= Thresholds.EMERGENCY_REREV_SLOPE_BPS_S
                    and sign5 != 0
                    and sign5 == sign60
                    and trade_dir != 0
                    and sign5 * trade_dir < 0  # move is against us
                    and chop < Thresholds.AUTO_VETO_CHOP_FLIPS
                )

                if is_emergency:
                    # Consume escape for this lockout period
                    escape_used_map[sym] = last_rev_ts
                    decision.setdefault("boosters", []).append("emergency_rerev_escape")
                    remaining = Thresholds.POST_REVERSE_LOCKOUT_SEC - (now_wall - last_rev_ts)
                    print(
                        f"🔓 AutoPilot RE_REVERSE_ESCAPE {sym} — bypassing {remaining:.0f}s "
                        f"lockout | pnl%={pnl_pct:+.2f} s5={s5:+.2f} s60={s60:+.2f} "
                        f"chop={chop} (one-shot, intraday rescue)",
                        flush=True,
                    )
                    # fall through — do NOT continue
                else:
                    self._skip_counters["cooldown"] += 1
                    # Throttled log so we see it's working
                    log_key = f"__rev_lock_log_{sym}__"
                    last_lk = self._windows.get(log_key, 0)
                    if now_wall - last_lk >= 30:
                        self._windows[log_key] = now_wall
                        remaining = Thresholds.POST_REVERSE_LOCKOUT_SEC - (now_wall - last_rev_ts)
                        print(
                            f"🔒 AutoPilot POST_REVERSE_LOCKOUT {sym} — {remaining:.0f}s left",
                            flush=True,
                        )
                    continue

            # Day-cap guard
            counters = self._state.setdefault("counters", {}).setdefault(sym, {"reverses": 0, "adds": 0})
            if decision["action"] == "REVERSE" and counters["reverses"] >= Thresholds.MAX_REVERSES_PER_DAY:
                self._skip_counters["day_cap"] += 1
                continue
            if decision["action"] == "ADD_LOT" and counters["adds"] >= Thresholds.MAX_ADDS_PER_DAY:
                self._skip_counters["day_cap"] += 1
                continue

            # Option-spread gate: block when the option's own bid-ask is too
            # wide to execute without giving away most of the edge to the MM.
            # REVERSE fires two orders (close + open), so spread cost is paid
            # twice — but it's still usually better than holding a losing leg.
            # ADD fires one order, no urgency, tighter gate.
            bid = float(quote.get("bid") or 0)
            ask = float(quote.get("ask") or 0)
            if bid > 0 and ask > 0 and ask > bid:
                mid = (bid + ask) / 2.0
                spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0.0
                limit = (Thresholds.MAX_OPTION_SPREAD_PCT
                         if decision["action"] == "REVERSE"
                         else Thresholds.MAX_OPTION_SPREAD_PCT_ADD)
                if spread_pct > limit:
                    self._skip_counters["wide_spread"] += 1
                    print(
                        f"🚫 AutoPilot SKIP_WIDE_SPREAD {decision['action']} {sym} | "
                        f"bid={bid:.2f} ask={ask:.2f} spread={spread_pct:.2f}% > {limit:.1f}%",
                        flush=True,
                    )
                    continue

            # Rule fired → log BEFORE calling LLM (so we see the trigger even if LLM fails)
            print(
                f"🎯 AutoPilot RULE_FIRED {decision['action']} {sym} | "
                f"rule={decision.get('rule')} "
                f"u_slope5s={features['slope_5s_bps']:+.2f}bps/s "
                f"u_slope3s={features['slope_3s_bps']:+.2f}bps/s "
                f"u_slope60s={features['slope_60s_bps']:+.2f}bps/s "
                f"u_slope300s={features.get('slope_300s_bps', 0.0):+.2f}bps/s "
                f"sustain={features['sustain_sec']}s chop={features['chop_flips']} "
                f"trade_dir={'+' if features['trade_dir']>0 else '-'} "
                f"pnl%={features['pnl_pct']:+.2f} — asking LLM…",
                flush=True,
            )

            # LLM confirmation (always on per user's choice)
            # ── Concurrency hardening (Apr 27) ─────────────────────
            # 1) Cache: if a verdict for this (symbol, action) was issued
            #    in the last 10s, reuse it. Halves LLM cost when a regime
            #    holds for several ticks.
            # 2) Parallel: cache misses are submitted to a thread pool so
            #    multiple symbols can ask the LLM concurrently within the
            #    same tick. State mutations stay sequential after the
            #    futures resolve, preventing any race condition.
            _llm_key = (sym, decision["action"])
            _cached = self._llm_cache.get(_llm_key)
            if _cached and (now_wall - _cached[0]) < self._llm_cache_ttl:
                # Cache hit — handle inline, no LLM round-trip.
                _v = dict(_cached[1])
                _v["reason"] = "[cached] " + str(_v.get("reason", ""))[:180]
                self._handle_verdict(pos, features, decision, sym,
                                     counters, now_wall, _v)
                continue
            # Cache miss → submit to thread pool, defer handling
            try:
                _fut = self._llm_executor.submit(
                    self._ask_llm, pos, features, decision
                )
                _pending_llm.append(
                    (pos, features, decision, sym, counters, now_wall, _fut)
                )
            except Exception as _e:
                # Pool exhausted / shut down → fall back to sequential call
                _v = self._ask_llm(pos, features, decision)
                self._llm_cache[_llm_key] = (now_wall, _v)
                self._handle_verdict(pos, features, decision, sym,
                                     counters, now_wall, _v)

        # ── Drain pending LLM futures (parallel verdicts, sequential exec) ──
        # All gate evaluation is done. Now resolve the cache-miss verdicts in
        # the order they were submitted and run their actions sequentially so
        # state mutations (counters / last_action_ts / save_state) cannot race.
        # max_workers=4 in the pool keeps OpenAI fan-out polite.
        if _pending_llm:
            for (pos, features, decision, sym, counters, gate_ts, fut) in _pending_llm:
                try:
                    llm_verdict = fut.result(timeout=15)
                except Exception as _e:
                    llm_verdict = {"go": False, "reason": f"llm_timeout: {str(_e)[:80]}"}
                # Cache the resolved verdict (success OR error) so we don't
                # hammer LLM with retries during the same regime burst.
                self._llm_cache[(sym, decision["action"])] = (gate_ts, llm_verdict)
                self._handle_verdict(pos, features, decision, sym,
                                     counters, gate_ts, llm_verdict)
            # One save at end of tick covers all pending mutations.
            self._save_state()

        # Prune windows for symbols that are no longer open
        for stale in list(self._windows.keys()):
            if stale not in symbols_seen:
                self._windows.pop(stale, None)
        # Prune entry-u_ltp snapshots for closed symbols so re-entries start fresh.
        snap_map = self._state.get("u_entry_ltp", {})
        for stale in list(snap_map.keys()):
            if stale not in symbols_seen:
                snap_map.pop(stale, None)

    # ── Feature extraction ──────────────────────────────────────────
    def _extract_features(self, win: deque, pos: dict) -> Optional[Dict[str, Any]]:
        """Build slope features on the UNDERLYING tick stream.

        Window tuple: (ts, opt_ltp, buy_qty, sell_qty, volume, u_ltp)
        Key outputs (all named on the underlying):
          - slope_5s_bps  : OLS slope of u_ltp over last SLOPE_WINDOW_SEC, in bps/sec
          - slope_3s_bps  : same over last CONFIRM_WINDOW_SEC (must agree in sign)
          - sustain_sec   : consecutive same-sign 1-sec deltas at the tail
          - chop_flips    : sign-flips of 1-sec deltas across the WINDOW_SEC window
          - trade_dir     : +1 long CE / short PE (wants UP), -1 long PE / short CE (wants DOWN)
          - aligned       : bool, sign(slope_5s) == trade_dir
        """
        if len(win) < Thresholds.SLOPE_WINDOW_SEC + 1:
            return None
        pts = list(win)
        now_ts = pts[-1][0]

        # --- Underlying slope via simple OLS on (t, u_ltp) ---
        def _slope_bps_per_sec(series):
            """OLS slope normalised to bps/sec of the reference price."""
            if len(series) < 2:
                return 0.0
            n = len(series)
            ts = [p[0] for p in series]
            ys = [p[1] for p in series]  # underlying ltp
            t_mean = sum(ts) / n
            y_mean = sum(ys) / n
            num = sum((ts[i] - t_mean) * (ys[i] - y_mean) for i in range(n))
            den = sum((ts[i] - t_mean) ** 2 for i in range(n)) or 1e-9
            slope_rs_per_sec = num / den
            ref = ys[-1] or 1e-9
            return (slope_rs_per_sec / ref) * 1e4  # bps / sec

        # Underlying series restricted to the last SLOPE_WINDOW_SEC seconds
        u_series_full = [(p[0], p[5]) for p in pts if p[5]]
        if len(u_series_full) < Thresholds.SLOPE_WINDOW_SEC + 1:
            return None
        u5 = [p for p in u_series_full if now_ts - p[0] <= Thresholds.SLOPE_WINDOW_SEC + 0.5]
        u3 = [p for p in u_series_full if now_ts - p[0] <= Thresholds.CONFIRM_WINDOW_SEC + 0.5]
        u60 = [p for p in u_series_full if now_ts - p[0] <= Thresholds.MACRO_WINDOW_SEC + 0.5]
        u300 = [p for p in u_series_full if now_ts - p[0] <= Thresholds.MACRO5M_WINDOW_SEC + 0.5]
        slope_5s_bps = _slope_bps_per_sec(u5)
        slope_3s_bps = _slope_bps_per_sec(u3)
        slope_60s_bps = _slope_bps_per_sec(u60) if len(u60) >= 10 else 0.0
        # slope_300s requires ≥ 120 samples (2 min) to be meaningful;
        # until the window warms up, leave as 0 so the macro5m path can't fire.
        slope_300s_bps = _slope_bps_per_sec(u300) if len(u300) >= 120 else 0.0

        # Realised-vol of 1-sec log-returns over VOL_LOOKBACK_SEC, in bps.
        u_vol = [p for p in u_series_full if now_ts - p[0] <= Thresholds.VOL_LOOKBACK_SEC + 0.5]
        rv_bps = 0.0
        if len(u_vol) >= 10:
            import math
            rets = []
            for i in range(1, len(u_vol)):
                y0, y1 = u_vol[i - 1][1], u_vol[i][1]
                if y0 > 0 and y1 > 0:
                    rets.append(math.log(y1 / y0) * 1e4)   # bps
            if len(rets) >= 5:
                m = sum(rets) / len(rets)
                var = sum((r - m) ** 2 for r in rets) / len(rets)
                rv_bps = var ** 0.5  # stdev in bps per 1-sec step

        # Cumulative 15-sec underlying return in bps (for flash-spike filter).
        u_flash = [p for p in u_series_full if now_ts - p[0] <= Thresholds.FLASH_LOOKBACK_SEC + 0.5]
        ret_15s_bps = 0.0
        if len(u_flash) >= 2 and u_flash[0][1] > 0:
            ret_15s_bps = ((u_flash[-1][1] - u_flash[0][1]) / u_flash[0][1]) * 1e4

        # 30-sec net-over-range ratio (for wick-rejection filter). A real trend
        # break has |net| ≈ range; a wick-and-bounce has |net| << range.
        u_wick = [p for p in u_series_full if now_ts - p[0] <= Thresholds.WICK_LOOKBACK_SEC + 0.5]
        ret_30s_bps = 0.0
        range_30s_bps = 0.0
        net_over_range = 1.0  # default pass-through if insufficient data
        if len(u_wick) >= 5 and u_wick[0][1] > 0:
            ref = u_wick[0][1]
            ret_30s_bps = ((u_wick[-1][1] - ref) / ref) * 1e4
            hi = max(p[1] for p in u_wick)
            lo = min(p[1] for p in u_wick)
            range_30s_bps = ((hi - lo) / ref) * 1e4
            if range_30s_bps > 1e-6:
                net_over_range = abs(ret_30s_bps) / range_30s_bps

        # --- Sustain: consecutive same-sign 1-sec deltas from the tail ---
        u_ltps = [p[1] for p in u_series_full]
        u_deltas = [u_ltps[i] - u_ltps[i - 1] for i in range(1, len(u_ltps))]
        sustain = 0
        if u_deltas:
            ref_sign = 1 if u_deltas[-1] > 0 else (-1 if u_deltas[-1] < 0 else 0)
            if ref_sign != 0:
                for d in reversed(u_deltas):
                    if (d > 0 and ref_sign == 1) or (d < 0 and ref_sign == -1):
                        sustain += 1
                    else:
                        break

        # --- Chop flips across whole window ---
        flips = 0
        prev_s = 0
        for d in u_deltas:
            s = 1 if d > 0 else (-1 if d < 0 else 0)
            if s == 0:
                continue
            if prev_s != 0 and s != prev_s:
                flips += 1
            prev_s = s

        # --- Trade direction (what underlying move we want) ---
        opt_type = (pos.get("option_type") or "").upper()
        side = (pos.get("side") or "BUY").upper()
        long_like = +1 if side == "BUY" else -1
        wants_up = +1 if opt_type == "CE" else -1
        trade_dir = long_like * wants_up  # +1 → want u_ltp UP, -1 → want u_ltp DOWN
        slope_sign = 1 if slope_5s_bps > 0 else (-1 if slope_5s_bps < 0 else 0)
        aligned = (slope_sign != 0) and (slope_sign == trade_dir)

        # --- Context (for logging / LLM only, NOT gating) ---
        cur = pts[-1]
        buys = cur[2] or 1
        sells = cur[3] or 1
        depth_ratio = buys / sells
        vols = [p[4] for p in pts]
        vol_deltas = [vols[i] - vols[i - 1] for i in range(1, len(vols))]
        baseline = (sum(vol_deltas) / len(vol_deltas)) if vol_deltas else 0
        last_vol_delta = vol_deltas[-1] if vol_deltas else 0
        vol_surge = (last_vol_delta / baseline) if baseline > 0 else 1.0

        entry = float(pos.get("avg_price") or pos.get("entry_price") or 0)
        qty = int(pos.get("quantity") or 0)
        opt_ltp_now = cur[1]
        pnl = (opt_ltp_now - entry) * qty if side == "BUY" else (entry - opt_ltp_now) * qty
        premium = max(1.0, entry * qty)
        pnl_pct = (pnl / premium) * 100

        # --- Underlying drift from AP's first-seen u_ltp for this symbol ---
        # We don't have entry_underlying on positions, so snapshot on first
        # tick AP sees for this symbol. Key is the position symbol so a fresh
        # entry (post-exit same name) starts a new snapshot if AP was stopped
        # and restarted; otherwise the same snapshot is reused for the life
        # of this AP process which is what we want for intraday.
        u_ltp_now = float(cur[5] or 0)
        sym_k = pos.get("symbol") or ""
        snap_map = self._state.setdefault("u_entry_ltp", {})
        u_entry = float(snap_map.get(sym_k, 0.0) or 0.0)
        if u_entry <= 0 and u_ltp_now > 0:
            snap_map[sym_k] = u_ltp_now
            u_entry = u_ltp_now
        if u_entry > 0 and u_ltp_now > 0 and trade_dir != 0:
            # Positive means underlying moved AGAINST our direction.
            u_drift_against_bps = ((u_ltp_now - u_entry) / u_entry) * 1e4 * (-trade_dir)
        else:
            u_drift_against_bps = 0.0

        hold_sec = 0
        try:
            ts_str = pos.get("timestamp") or pos.get("entry_time") or ""
            if ts_str:
                from dateutil.parser import parse as _dp
                hold_sec = (datetime.now() - _dp(ts_str)).total_seconds()
        except Exception:
            pass

        return {
            "symbol": pos.get("symbol"),
            "option_type": opt_type,
            "side": side,
            "ltp": round(opt_ltp_now, 2),
            "u_ltp": round(cur[5] or 0, 2),
            "entry": round(entry, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_inr": round(pnl, 0),
            "hold_sec": int(hold_sec),
            "trade_dir": trade_dir,
            "u_drift_against_bps": round(u_drift_against_bps, 2),
            "slope_5s_bps": round(slope_5s_bps, 3),
            "slope_3s_bps": round(slope_3s_bps, 3),
            "slope_60s_bps": round(slope_60s_bps, 3),
            "slope_300s_bps": round(slope_300s_bps, 3),
            "rv_bps": round(rv_bps, 3),
            "ret_15s_bps": round(ret_15s_bps, 2),
            "ret_30s_bps": round(ret_30s_bps, 2),
            "range_30s_bps": round(range_30s_bps, 2),
            "net_over_range": round(net_over_range, 3),
            "sustain_sec": int(sustain),
            "chop_flips": int(flips),
            "aligned": bool(aligned),
            "depth_ratio": round(depth_ratio, 2),
            "vol_surge": round(vol_surge, 2),
        }

    # ── Rule gate — SLOPE of UNDERLYING, not premium noise ──────────
    def _rule_gate(self, pos: dict, f: dict) -> Optional[Dict[str, Any]]:
        """
        Decision tree mirrors the user's manual process:
          1. Already sat out entry-noise window (MIN_HOLD_AFTER_ENTRY)
          2. Not in chop (flips < CHOP_FLIP_LIMIT)
          3. Require the 3-sec confirm slope to agree in sign with the 5-sec slope
             (i.e. trend hasn't started turning at the tail)
          4. Require SUSTAIN_SEC of consecutive same-sign 1-sec deltas
          5. If aligned + steep → ADD_LOT
             If NOT aligned + min-strength counter-slope → REVERSE
             Else → hold
        """
        if f["hold_sec"] < Thresholds.MIN_HOLD_AFTER_ENTRY:
            return None

        slope5 = f["slope_5s_bps"]
        slope3 = f["slope_3s_bps"]
        sign5 = 1 if slope5 > 0 else (-1 if slope5 < 0 else 0)
        sign3 = 1 if slope3 > 0 else (-1 if slope3 < 0 else 0)

        # ── Deep-bleed REVERSE override ──
        # Fires BEFORE normal chop/sustain/wick filters because a sustained
        # deep drawdown is by definition a real move, not a wick. Requires:
        #   - pnl_pct <= BLEED_PNL_PCT (position deep red)
        #   - slope5 against our position (aligned=False) with any magnitude
        #     ≥ BLEED_MIN_SLOPE_BPS_S (so we don't flip on flat tape)
        #   - slope5 and slope3 both non-zero & same sign (tail not turning)
        #   - macro60s sign AGREES with slope5 sign — i.e. the higher-timeframe
        #     trend is also against us, this is a regime not a pullback.
        pnl_pct_val = f.get("pnl_pct", 0.0) or 0.0
        macro_val = f.get("slope_60s_bps", 0.0) or 0.0
        macro_sign_val = 1 if macro_val > 0 else (-1 if macro_val < 0 else 0)
        ret15_val = f.get("ret_15s_bps", 0.0) or 0.0
        ret15_sign_val = 1 if ret15_val > 0 else (-1 if ret15_val < 0 else 0)
        exhausted = (
            abs(ret15_val) >= Thresholds.EXHAUSTION_RET15_BPS
            and ret15_sign_val == sign5
        )
        if (
            pnl_pct_val <= Thresholds.BLEED_PNL_PCT
            and not f.get("aligned", True)
            and sign5 != 0
            and sign3 == sign5
            and macro_sign_val == sign5
            and abs(slope5) >= Thresholds.BLEED_MIN_SLOPE_BPS_S
            and f.get("chop_flips", 0) < Thresholds.AUTO_VETO_CHOP_FLIPS
            and not exhausted
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"bleed-override(pnl%={pnl_pct_val:.1f}≤{Thresholds.BLEED_PNL_PCT}"
                    f",|s5|={abs(slope5):.2f}bps/s,macro{macro_val:+.2f}"
                    f",chop={f.get('chop_flips')})"
                ),
                "boosters": ["bleed_override"],
            }

        # ── Stale-loss REVERSE ──
        # Drift-based rescue for positions that have been quietly bleeding in
        # chop for a long time with no active slope to trigger any other path.
        # All four conditions independent — safe against false positives.
        u_drift_ag = f.get("u_drift_against_bps", 0.0) or 0.0
        if (
            pnl_pct_val <= Thresholds.STALE_LOSS_PNL_PCT
            and f.get("hold_sec", 0) >= Thresholds.STALE_LOSS_MIN_HOLD
            and u_drift_ag >= Thresholds.STALE_LOSS_DRIFT_BPS
            and f.get("chop_flips", 0) < Thresholds.STALE_LOSS_MAX_CHOP
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"stale-loss(pnl%={pnl_pct_val:.1f},hold={f.get('hold_sec')}s"
                    f",u_drift_against={u_drift_ag:+.1f}bps"
                    f",chop={f.get('chop_flips')})"
                ),
                "boosters": ["stale_loss"],
            }

        # ── Medium-bleed REVERSE ──
        # Multi-axis counter-spike rescue. Requires SIX independent conditions
        # (pnl red, against us, strong s5, 3s confirms, macro confirms w/
        # magnitude, not exhausted, medium chop). Each condition on its own
        # is mild; the conjunction is rare and only matches a genuine reversal.
        if (
            pnl_pct_val <= Thresholds.MEDIUM_BLEED_PNL_PCT
            and not f.get("aligned", True)
            and sign5 != 0
            and sign3 == sign5
            and macro_sign_val == sign5
            and abs(slope5) >= Thresholds.MEDIUM_BLEED_MIN_SLOPE_BPS_S
            and abs(macro_val) >= Thresholds.MEDIUM_BLEED_MIN_MACRO_BPS_S
            and f.get("chop_flips", 0) < Thresholds.MEDIUM_BLEED_MAX_CHOP
            and not exhausted
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"medium-bleed(pnl%={pnl_pct_val:.1f},|s5|={abs(slope5):.2f}bps/s"
                    f",s3{slope3:+.2f},macro{macro_val:+.2f}"
                    f",chop={f.get('chop_flips')})"
                ),
                "boosters": ["medium_bleed"],
            }

        # ── Momentum-ADD override ──
        # Symmetric to bleed-override but for pyramiding winners. Hands-off guard
        # on trailing-active symbols runs upstream in _tick(), so this can only
        # fire on winners that haven't armed their trailing stop yet.
        if (
            pnl_pct_val >= Thresholds.MOMENTUM_PNL_PCT
            and f.get("aligned", False)
            and sign5 != 0
            and sign3 == sign5
            and abs(slope5) >= Thresholds.MOMENTUM_MIN_SLOPE_BPS_S
            and f.get("chop_flips", 0) < Thresholds.AUTO_VETO_CHOP_FLIPS
        ):
            return {
                "action": "ADD_LOT",
                "rule": (
                    f"momentum-override(pnl%={pnl_pct_val:.1f}≥{Thresholds.MOMENTUM_PNL_PCT}"
                    f",|s5|={abs(slope5):.2f}bps/s,chop={f.get('chop_flips')})"
                ),
                "boosters": ["momentum_override"],
            }

        # ── Profit-peak REVERSE ──
        # Flips a green position at a swing peak when a counter-slope is forming.
        # Fills the gap where all other REVERSE gates require pnl% ≤ negative.
        # Enables back-and-forth peak harvesting on oscillating names (e.g.
        # VEDL 2026-04-24 manual 5× flips, each banking profit). Symmetric:
        # fires on both long-CE peaks and long-PE peaks. Re-arms each cycle
        # via MAX_REVERSES_PER_DAY.
        if (
            pnl_pct_val >= Thresholds.PROFIT_PEAK_PNL_PCT
            and not f.get("aligned", False)
            and sign5 != 0
            and sign3 == sign5
            and abs(slope5) >= Thresholds.PROFIT_PEAK_MIN_S5_BPS_S
            and abs(slope3) >= Thresholds.PROFIT_PEAK_MIN_S3_BPS_S
            and f.get("hold_sec", 0) >= Thresholds.PROFIT_PEAK_MIN_HOLD
            and f.get("chop_flips", 0) < Thresholds.PROFIT_PEAK_MAX_CHOP
            and not exhausted
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"profit-peak(pnl%={pnl_pct_val:+.1f}≥{Thresholds.PROFIT_PEAK_PNL_PCT}"
                    f",|s5|={abs(slope5):.2f}bps/s,|s3|={abs(slope3):.2f}bps/s"
                    f",chop={f.get('chop_flips')},hold={f.get('hold_sec')}s)"
                ),
                "boosters": ["profit_peak"],
            }

        # ── Clean-tape drift REVERSE ──
        # Last-resort catch for slow grinds on illiquid/clean tape where tick
        # slopes never trip and premium is stale. Primary trigger is the
        # underlying's move vs our entry (u_drift_against_bps) — ground truth
        # when slopes and premium lag. Requires the tape be clean (low chop)
        # so we don't flip into whipsaw.
        trade_dir_for_drift = f.get("trade_dir", 0)
        if (
            trade_dir_for_drift != 0
            and f.get("hold_sec", 0) >= Thresholds.CLEAN_DRIFT_MIN_HOLD
            and f.get("chop_flips", 0) < Thresholds.CLEAN_DRIFT_MAX_CHOP
            and u_drift_ag >= Thresholds.CLEAN_DRIFT_MIN_U_DRIFT_BPS
            and pnl_pct_val <= Thresholds.CLEAN_DRIFT_PNL_PCT
            and not exhausted
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"clean-drift(u_drift_against={u_drift_ag:+.1f}bps"
                    f",pnl%={pnl_pct_val:.1f},hold={f.get('hold_sec')}s"
                    f",chop={f.get('chop_flips')})"
                ),
                "boosters": ["clean_drift"],
            }

        # ── Macro-5min counter-trend REVERSE ──
        # Catches chart-visible multi-minute trend reversals that the 5s/3s/60s
        # stack misses because its thresholds are tuned for tick noise, not for
        # slow 0.3-0.6 bps/s drifts over 5 minutes. Fires when:
        #   - slope_300s_bps is non-zero (window warmed up, ≥120s of history)
        #   - sign(slope_300s_bps) is AGAINST trade_dir (5-min trend against us)
        #   - |slope_300s_bps| ≥ MACRO5M_REVERSE_BPS_S (real move, not drift)
        #   - pnl_pct ≤ MACRO5M_PNL_PCT (already bleeding; don't flip winners)
        #   - chop_flips < MACRO5M_MAX_CHOP_FLIPS (tape not in pure whipsaw)
        # Intentionally DOES NOT require sustain/3s-agree/macro60s-agree/wick —
        # those are tick-scale filters that are noise at the 5-min timescale.
        # LLM gate downstream still vetoes nonsense.
        slope300 = f.get("slope_300s_bps", 0.0) or 0.0
        sign300 = 1 if slope300 > 0 else (-1 if slope300 < 0 else 0)
        trade_dir_val = f.get("trade_dir", 0)
        if (
            sign300 != 0
            and trade_dir_val != 0
            and sign300 != trade_dir_val
            and abs(slope300) >= Thresholds.MACRO5M_REVERSE_BPS_S
            and pnl_pct_val <= Thresholds.MACRO5M_PNL_PCT
            and f.get("chop_flips", 0) < Thresholds.MACRO5M_MAX_CHOP_FLIPS
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"macro5min(s300={slope300:+.2f}bps/s vs dir={trade_dir_val}"
                    f",pnl%={pnl_pct_val:.1f},chop={f.get('chop_flips')})"
                ),
                "boosters": ["macro5min_counter"],
            }

        # ── Macro-5min WITH-trend ADD_LOT ──
        # Symmetric to macro5min REVERSE. Pyramids winners when the 5-minute
        # underlying slope is running our way at ≥0.35 bps/s AND pnl% ≥ +4%
        # AND tape isn't pure chop. Catches slow sustained drifts (e.g. a
        # 0.2%/min grind) where 5s slope never clears the 2.0 bps/s STEEP gate.
        if (
            sign300 != 0
            and trade_dir_val != 0
            and sign300 == trade_dir_val
            and abs(slope300) >= Thresholds.MACRO5M_ADD_BPS_S
            and pnl_pct_val >= Thresholds.MACRO5M_ADD_PNL_PCT
            and f.get("chop_flips", 0) < Thresholds.MACRO5M_MAX_CHOP_FLIPS
        ):
            return {
                "action": "ADD_LOT",
                "rule": (
                    f"macro5min-add(s300={slope300:+.2f}bps/s with dir={trade_dir_val}"
                    f",pnl%={pnl_pct_val:.1f},chop={f.get('chop_flips')})"
                ),
                "boosters": ["macro5min_with"],
            }

        # ── Macro-trend CHOP-BYPASS REVERSE (Option A) ──
        # Last-resort override when chop is too high for every other path but
        # the macro evidence is overwhelming. Requires SEVERE pnl loss AND a
        # large entry-to-now underlying drift in our face — both must agree
        # so we never flip on tape noise alone. No chop guard: stairstep
        # grinds (RELIANCE 1325→1345 / 3h with chop=89) need this escape.
        u_drift_against_now = f.get("u_drift_against_bps", 0.0) or 0.0
        if (
            trade_dir_for_drift != 0
            and f.get("hold_sec", 0) >= Thresholds.BYPASS_REV_MIN_HOLD
            and pnl_pct_val <= Thresholds.BYPASS_REV_PNL_PCT
            and u_drift_against_now >= Thresholds.BYPASS_REV_U_DRIFT_BPS
            and not exhausted
        ):
            return {
                "action": "REVERSE",
                "rule": (
                    f"chop-bypass-rev(pnl%={pnl_pct_val:.1f}≤{Thresholds.BYPASS_REV_PNL_PCT}"
                    f",u_drift_against={u_drift_against_now:+.1f}bps≥{Thresholds.BYPASS_REV_U_DRIFT_BPS}"
                    f",hold={f.get('hold_sec')}s,chop={f.get('chop_flips')})"
                ),
                "boosters": ["chop_bypass_rev"],
            }

        # ── Macro-trend CHOP-BYPASS ADD_LOT (Option A, symmetric) ──
        # Pyramid winners on chart-visible runs even when 5s tape is choppy.
        # Mirrors REVERSE: needs decent pnl + large WITH-direction underlying drift.
        u_drift_with_now = -u_drift_against_now  # WITH-direction drift in bps
        if (
            trade_dir_for_drift != 0
            and f.get("hold_sec", 0) >= Thresholds.BYPASS_ADD_MIN_HOLD
            and pnl_pct_val >= Thresholds.BYPASS_ADD_PNL_PCT
            and u_drift_with_now >= Thresholds.BYPASS_ADD_U_DRIFT_WITH_BPS
        ):
            return {
                "action": "ADD_LOT",
                "rule": (
                    f"chop-bypass-add(pnl%={pnl_pct_val:+.1f}≥{Thresholds.BYPASS_ADD_PNL_PCT}"
                    f",u_drift_with={u_drift_with_now:+.1f}bps≥{Thresholds.BYPASS_ADD_U_DRIFT_WITH_BPS}"
                    f",hold={f.get('hold_sec')}s,chop={f.get('chop_flips')})"
                ),
                "boosters": ["chop_bypass_add"],
            }

        if f["chop_flips"] >= Thresholds.CHOP_FLIP_LIMIT:
            return None

        # Tail of the move must agree in sign with the 5s trend —
        # otherwise the slope is already turning and we should wait.
        if sign5 == 0 or sign3 == 0 or sign5 != sign3:
            return None

        if f["sustain_sec"] < Thresholds.SUSTAIN_SEC:
            return None

        trade_dir = f["trade_dir"]
        abs5 = abs(slope5)

        # --- Adaptive thresholds scaled by realised vol ----------------
        # vol_ratio = rv_bps / VOL_BASELINE_BPS_S (clamped to [0.25, 4.0])
        if Thresholds.VOL_ADAPTIVE and f.get("rv_bps", 0) > 0:
            vol_ratio = f["rv_bps"] / max(1e-6, Thresholds.VOL_BASELINE_BPS_S)
            vol_ratio = max(0.25, min(4.0, vol_ratio))
        else:
            vol_ratio = 1.0
        steep_thr = max(Thresholds.STEEP_FLOOR_BPS_S,
                        Thresholds.STEEP_SLOPE_BPS_S * vol_ratio)
        reverse_thr = max(Thresholds.REVERSE_FLOOR_BPS_S,
                          Thresholds.REVERSE_MIN_BPS_S * vol_ratio)

        # ADD_LOT: slope agrees with our direction AND is steep (vol-adjusted)
        if f["aligned"] and abs5 >= steep_thr:
            return {
                "action": "ADD_LOT",
                "rule": f"aligned+steep({abs5:.2f}≥{steep_thr:.2f}bps/s,rv={f.get('rv_bps')}bps)+sustain{f['sustain_sec']}s",
                "boosters": [],
            }

        # REVERSE: slope AGAINST our direction with enough magnitude (vol-adjusted)
        # AND the counter-move must be BREAKING the 60-sec macro trend
        # AND must NOT be a flash-spike reversion (5-sec slope sign must agree
        #     with the cumulative 15-sec return sign — otherwise the dip is
        #     pulling price back toward the 15s mean, i.e. mean-reversion, not a turn).
        if (not f["aligned"]) and abs5 >= reverse_thr:
            # Flash-spike rejection: if 15-sec cumulative return opposes the
            # 5-sec slope, the move is a reversion, not a trend break.
            ret15 = f.get("ret_15s_bps", 0.0) or 0.0
            ret15_sign = 1 if ret15 > 0 else (-1 if ret15 < 0 else 0)
            if ret15_sign != 0 and ret15_sign != sign5:
                return None

            # Exhaustion-spike rejection: when |ret_15s| is TOO LARGE in the
            # SAME direction as slope_5s, we're about to flip INTO a capitulation
            # spike. These mean-revert more often than they continue.
            if abs(ret15) >= Thresholds.EXHAUSTION_RET15_BPS and ret15_sign == sign5:
                return None

            # Wick rejection: |net_30s| must be at least 40% of range_30s.
            # A V-shape wick travels far but ends near where it started
            # (net/range ≈ 0.1-0.3); a real trend break has net/range ≈ 0.7-1.0.
            nor = f.get("net_over_range", 1.0)
            if nor < Thresholds.WICK_NET_OVER_RANGE_MIN:
                return None

            macro = f.get("slope_60s_bps", 0.0) or 0.0
            macro_sign = 1 if macro > 0 else (-1 if macro < 0 else 0)
            # Strict macro-agreement: the 60-sec trend must point in the
            # same direction as the 5-sec counter-slope. A flat or opposing
            # macro means the counter-move is a pullback, not a regime flip.
            if macro_sign != sign5:
                return None
            # Macro-magnitude floor: reject when macro is near zero (noise),
            # even if the sign happens to agree — a marginal +0.06 bps/s
            # doesn't count as "macro confirms the flip".
            if abs(macro) < Thresholds.MACRO_MIN_ABS_BPS_S:
                return None
            # Multi-timeframe confirmation: 5-min macro must NOT be with
            # the position. If slope_300s agrees with trade_dir, the bleed
            # is theta/noise, not a real regime flip — don't reverse.
            # When slope_300s == 0 (warming up after restart) we skip this
            # check so AP isn't totally frozen for 2 minutes on boot.
            slope300_val = f.get("slope_300s_bps", 0.0) or 0.0
            if (
                Thresholds.REQUIRE_MACRO5M_AGAINST_FOR_REV
                and slope300_val != 0.0
                and trade_dir != 0
                and (1 if slope300_val > 0 else -1) == trade_dir
            ):
                return None
            return {
                "action": "REVERSE",
                "rule": (f"counter-slope({abs5:.2f}≥{reverse_thr:.2f}bps/s)"
                         f"+sustain{f['sustain_sec']}s+macro({macro:+.2f}bps/s agree)"
                         f"+flash_ok({ret15:+.1f}bps)+nor({nor:.2f})"),
                "boosters": [],
            }

        return None

    # ── Verdict handler (extracted Apr 27 for parallel-LLM refactor) ─
    def _handle_verdict(self, pos: dict, features: dict, decision: dict,
                        sym: str, counters: dict, now_wall: float,
                        llm_verdict: Dict[str, Any]) -> None:
        """Apply an LLM verdict: veto-skip or execute + state mutate + log.

        Called both for cache hits (inline within per-symbol loop) and for
        resolved futures during the post-loop drain. ALL state mutations live
        here so they remain strictly sequential — no two threads ever touch
        counters / last_action_ts concurrently.
        """
        if not llm_verdict.get("go"):
            self._skip_counters["vetoed"] += 1
            self._log_decision(sym, decision["action"], "VETOED_BY_LLM", features,
                               llm_verdict.get("reason", ""), None)
            return

        # Execute via dashboard HTTP endpoints
        ok, msg = self._execute(pos, decision["action"])
        # Compute flipped symbol for REVERSE so the UI badge follows the
        # new (post-flip) position symbol as well as the original. Without
        # this the badge disappears from the new row because the position
        # row is keyed on the flipped option_type (PE↔CE).
        _flipped_sym = None
        if decision["action"] == "REVERSE":
            try:
                import re as _re_ap
                _opt = (pos.get("option_type") or "").upper()
                _new_opt = "CE" if _opt == "PE" else "PE" if _opt == "CE" else None
                if _new_opt and _opt:
                    _flipped_sym = _re_ap.sub(rf'{_opt}$', _new_opt, sym)
            except Exception:
                _flipped_sym = None
        if ok:
            self._skip_counters["executed"] += 1
            counters[decision["action"] == "REVERSE" and "reverses" or "adds"] += 1
            self._state.setdefault("last_action_ts", {})[sym] = now_wall
            if decision["action"] == "REVERSE":
                self._state.setdefault("last_reverse_ts", {})[sym] = now_wall
            self._log_decision(sym, decision["action"], "EXECUTED", features,
                               llm_verdict.get("reason", ""), msg,
                               flipped_symbol=_flipped_sym)
        elif msg == "http_409_conflict_inflight":
            # Prior action still being consumed by bot pipeline — suppress
            # retries via cooldown, don't log as failure. The position
            # state will settle on next tick and a fresh decision (if
            # still valid) can fire after cooldown.
            self._skip_counters["cooldown"] += 1
            self._state.setdefault("last_action_ts", {})[sym] = now_wall
            # For REVERSE, also arm the 120s lockout since the flip IS
            # happening — just not visible to us yet.
            if decision["action"] == "REVERSE":
                self._state.setdefault("last_reverse_ts", {})[sym] = now_wall
            self._log_decision(sym, decision["action"], "DEFERRED_INFLIGHT",
                               features, llm_verdict.get("reason", ""), msg,
                               flipped_symbol=_flipped_sym)
        else:
            self._log_decision(sym, decision["action"], "EXECUTION_FAILED", features,
                               llm_verdict.get("reason", ""), msg,
                               flipped_symbol=_flipped_sym)

    # ── LLM sanity check ────────────────────────────────────────────
    def _ask_llm(self, pos: dict, features: dict, decision: dict) -> Dict[str, Any]:
        if not self._llm:
            # Rules-only fallback — approve
            return {"go": True, "reason": "rules-only-mode"}

        system = (
            "You are an options trading co-pilot. A rules engine has ALREADY fired on "
            "a sustained SLOPE of the UNDERLYING stock (bps/second), after filtering out "
            "chop and tail-reversal. Your only job is to catch OBVIOUS traps — default "
            "is GO.\n\n"
            "Features you receive describe the UNDERLYING's move:\n"
            "  slope_5s_bps : OLS slope over last 5s, bps/sec (short-term)\n"
            "  slope_3s_bps : confirmation slope over last 3s (same sign required)\n"
            "  slope_60s_bps: macro trend over last 60s — REVERSE only fires when slope_5s "
            "breaks the macro (flips sign OR dominates by 1.5x)\n"
            "  slope_300s_bps: 5-minute trend — a 'macro5min' REVERSE may fire when this "
            "is against the position by ≥0.4 bps/s AND pnl_pct ≤ -5% even if short-term "
            "slopes are noisy. This catches chart-visible multi-minute trend reversals.\n"
            "  sustain_sec  : consecutive same-sign 1s deltas at the tail\n"
            "  chop_flips   : slope sign flips in last 30s (already <3 or rule wouldn't fire)\n"
            "  trade_dir    : +1 if position wants UP, -1 if wants DOWN\n"
            "  aligned      : True → ADD_LOT, False → REVERSE\n\n"
            "Veto (go=false) ONLY for concrete red flags such as:\n"
            "  • ADD_LOT when slope_5s magnitude is > 8 bps/sec AND pnl_pct > 80% → "
            "likely an exhaustion spike; adding near the top is bad.\n"
            "  • ADD_LOT on a deep-OTM lottery (pnl_pct extremely negative and entry price tiny).\n"
            "  • REVERSE when trade is actually recovering strongly (pnl_pct already > +3 "
            "and slope is only marginally against).\n\n"
            "Noise like depth_ratio near 1.0 or vol_surge < 1.3 are NOT red flags — NFO "
            "depth is thin by nature. Slope magnitude + sustain is what matters.\n\n"
            'Respond with ONLY a compact JSON: {"go": true|false, "reason": "<=20 words"}.'
        )
        user = json.dumps({
            "proposed_action": decision["action"],
            "rule_triggered": decision.get("rule"),
            "position": {
                "symbol": pos.get("symbol"),
                "underlying": pos.get("underlying"),
                "option_type": pos.get("option_type"),
                "side": pos.get("side"),
                "qty": pos.get("quantity"),
                "lots": pos.get("lots"),
                "entry": pos.get("avg_price"),
            },
            "features": features,
            "thresholds": {
                "steep_slope_bps_s": Thresholds.STEEP_SLOPE_BPS_S,
                "reverse_min_bps_s": Thresholds.REVERSE_MIN_BPS_S,
                "sustain_sec": Thresholds.SUSTAIN_SEC,
                "chop_flip_limit": Thresholds.CHOP_FLIP_LIMIT,
            },
        })

        try:
            resp = self._llm.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=8,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Strip markdown fences if any
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:].strip()
            verdict = json.loads(text)
            return {
                "go": bool(verdict.get("go")),
                "reason": str(verdict.get("reason", ""))[:200],
            }
        except Exception as e:
            # On LLM failure, be conservative — VETO (don't execute on broken link)
            return {"go": False, "reason": f"llm_error: {str(e)[:80]}"}

    # ── Action execution via dashboard HTTP ─────────────────────────
    def _execute(self, pos: dict, action: str) -> tuple:
        sym = pos.get("symbol") or pos.get("option_symbol") or ""
        if action == "REVERSE":
            # Build payload matching /api/reverse_trade
            opt_type = (pos.get("option_type") or "").upper()
            rev_opt = "CE" if opt_type == "PE" else "PE" if opt_type == "CE" else opt_type
            payload = {
                "symbol": sym,
                "underlying": pos.get("underlying", ""),
                "direction": "BUY",   # paper engine always BUYs options
                "quantity": int(pos.get("quantity") or 0),
                "ltp": float(pos.get("avg_price") or 0),
                "option_type": rev_opt,
                "strike": pos.get("strike", 0),
                "expiry": pos.get("expiry", ""),
                "lots": int(pos.get("lots") or 1),
                "is_option": True,
            }
            return self._post("/api/reverse_trade", payload)

        if action == "ADD_LOT":
            return self._post("/api/add_lot", {"symbol": sym})

        return False, f"unknown action {action}"

    def _post(self, path: str, payload: dict) -> tuple:
        url = f"{DASHBOARD_URL}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read().decode("utf-8")
                resp = json.loads(body) if body else {}
            return bool(resp.get("ok")), resp.get("msg", "")
        except urllib.error.HTTPError as e:
            # 409 CONFLICT = a prior manual-exit signal is still being consumed
            # by the bot's position pipeline. Surface a distinct marker so the
            # caller suppresses retries (via cooldown) instead of treating it
            # as either success or a real failure.
            if e.code == 409:
                return False, "http_409_conflict_inflight"
            return False, f"http_error: HTTP Error {e.code}: {e.reason}"
        except Exception as e:
            return False, f"http_error: {str(e)[:120]}"

    # ── Helpers ─────────────────────────────────────────────────────
    def _get_naked_option_positions(self) -> List[dict]:
        try:
            with self.tools._positions_lock:  # type: ignore[attr-defined]
                snap = list(self.tools.paper_positions or [])  # type: ignore[attr-defined]
        except Exception:
            snap = list(getattr(self.tools, "paper_positions", []) or [])

        out = []
        excluded = {"closed": 0, "spread": 0, "not_option": 0, "wrong_strategy": 0}
        for t in snap:
            if (t.get("status") or "OPEN") != "OPEN":
                excluded["closed"] += 1
                continue
            if t.get("is_credit_spread") or t.get("is_debit_spread") or t.get("is_iron_condor"):
                excluded["spread"] += 1
                continue
            if not t.get("is_option"):
                excluded["not_option"] += 1
                continue
            stype = (t.get("strategy_type") or "").upper()
            if stype and stype != "NAKED_OPTION":
                excluded["wrong_strategy"] += 1
                continue
            out.append(t)

        # Log once per 20s so user sees the universe the agent is scanning
        now = time.time()
        last = getattr(self, "_last_universe_log", 0)
        if now - last >= 20:
            self._last_universe_log = now
            print(
                f"📋 AutoPilot universe: eligible={len(out)} "
                f"(total paper_positions={len(snap)}; excluded: "
                f"closed={excluded['closed']} spreads={excluded['spread']} "
                f"equity={excluded['not_option']} other_strategy={excluded['wrong_strategy']})",
                flush=True,
            )
            if not out and snap:
                # Show one sample so we understand why nothing matched
                sample = snap[0]
                print(
                    f"   sample: sym={sample.get('symbol')} "
                    f"status={sample.get('status')} "
                    f"is_option={sample.get('is_option')} "
                    f"strategy_type={sample.get('strategy_type')} "
                    f"is_credit={sample.get('is_credit_spread')} "
                    f"is_debit={sample.get('is_debit_spread')} "
                    f"is_ic={sample.get('is_iron_condor')}",
                    flush=True,
                )
        return out

    def _refresh_realized_by_underlying(self, now_wall: float) -> None:
        """Rebuild self._realized_by_underlying from today's trade ledger.
        Cached for PER_SYMBOL_CAP_REFRESH_SEC to avoid hammering disk. Silent on
        any error \u2014 loss-cap gate is best-effort, not critical path.
        """
        if now_wall - self._realized_cache_ts < Thresholds.PER_SYMBOL_CAP_REFRESH_SEC:
            return
        self._realized_cache_ts = now_wall
        import re as _re
        try:
            from datetime import datetime as _dt
            today = _dt.now().strftime('%Y-%m-%d')
            ledger_file = os.path.join(
                os.path.dirname(__file__), 'trade_ledger', f'trade_ledger_{today}.jsonl'
            )
            if not os.path.exists(ledger_file):
                return
            totals: Dict[str, float] = {}
            with open(ledger_file, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get('event') != 'EXIT':
                        continue
                    under = ev.get('underlying') or ''
                    if not under:
                        sym = ev.get('symbol', '') or ''
                        m = _re.match(r'(?:NFO:)?([A-Z&]+)\d', sym.replace('NFO:', ''))
                        under = f"NSE:{m.group(1)}" if m else sym
                    pnl = float(ev.get('pnl', 0) or 0)
                    totals[under] = totals.get(under, 0.0) + pnl
            self._realized_by_underlying = totals
        except Exception:
            # Best-effort; keep previous cache.
            pass

    def _is_symbol_loss_capped(self, pos: dict) -> bool:
        """True when the position's underlying has already breached the
        per-symbol realized-loss cap today. AP gates on this to stop throwing
        good money after bad on a single ticker."""
        under = pos.get('underlying') or ''
        if not under:
            return False
        realized = self._realized_by_underlying.get(under, 0.0)
        return realized <= Thresholds.PER_SYMBOL_REALIZED_LOSS_CAP

    def _get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return {ltp, buy_qty, sell_qty, volume, bid, ask} from ticker cache."""
        ticker = getattr(self.tools, "ticker", None)
        if not ticker:
            return None
        key = symbol if ":" in symbol else f"NFO:{symbol}"
        try:
            q = ticker.get_quote(key)
        except Exception:
            return None
        if not q:
            return None
        # Best bid / ask from L1 depth (KiteTicker FULL-mode payload).
        bid = 0.0
        ask = 0.0
        try:
            depth = q.get("depth") or {}
            buys = depth.get("buy") or []
            sells = depth.get("sell") or []
            if buys:
                bid = float(buys[0].get("price") or 0)
            if sells:
                ask = float(sells[0].get("price") or 0)
        except Exception:
            pass
        return {
            "ltp": q.get("last_price") or 0,
            "buy_qty": q.get("buy_quantity") or q.get("total_buy_quantity") or 0,
            "sell_qty": q.get("sell_quantity") or q.get("total_sell_quantity") or 0,
            "volume": q.get("volume") or 0,
            "bid": bid,
            "ask": ask,
        }

    def _get_underlying_ltp(self, pos: dict) -> Optional[float]:
        """Read underlying stock LTP from the shared ticker cache.

        position.underlying is stored as e.g. "NSE:MANKIND". The KiteTicker
        already subscribes to all universe stocks at startup, so this is a
        zero-API-call lookup.
        """
        key = pos.get("underlying") or ""
        if not key:
            return None
        if ":" not in key:
            key = f"NSE:{key}"
        ticker = getattr(self.tools, "ticker", None)
        if not ticker:
            return None
        try:
            ltp = ticker.get_ltp(key)
            return float(ltp) if ltp else None
        except Exception:
            return None

    def _log_decision(self, symbol: str, action: str, outcome: str,
                      features: dict, reason: str, exec_msg: Optional[str],
                      flipped_symbol: Optional[str] = None):
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            # For REVERSE EXECUTED, the *new* (post-flip) symbol carries the
            # actual open position. UI uses this so the 🤖 badge follows the
            # flipped row instead of disappearing on every reverse.
            "flipped_symbol": flipped_symbol,
            "action": action,
            "outcome": outcome,
            "pnl_pct": features.get("pnl_pct"),
            "slope_5s_bps": features.get("slope_5s_bps"),
            "slope_3s_bps": features.get("slope_3s_bps"),
            "slope_60s_bps": features.get("slope_60s_bps"),
            "slope_300s_bps": features.get("slope_300s_bps"),
            "sustain_sec": features.get("sustain_sec"),
            "chop_flips": features.get("chop_flips"),
            "trade_dir": features.get("trade_dir"),
            "aligned": features.get("aligned"),
            # keep legacy keys so existing dashboard code doesn't blow up
            "velocity": features.get("slope_5s_bps"),
            "depth_ratio": features.get("depth_ratio"),
            "vol_surge": features.get("vol_surge"),
            "reason": reason,
            "exec_msg": exec_msg,
        }
        with self._lock:
            lst = self._state.setdefault("last_decision", [])
            lst.append(entry)
            # keep last 50
            if len(lst) > 50:
                del lst[: len(lst) - 50]
        msg_suffix = f" | exec_msg={exec_msg}" if outcome == "EXECUTION_FAILED" and exec_msg else ""
        print(f"🤖 AutoPilot {outcome} {action} {symbol} | "
              f"u_slope5s={features.get('slope_5s_bps')}bps/s "
              f"sustain={features.get('sustain_sec')}s "
              f"chop={features.get('chop_flips')} "
              f"pnl%={features.get('pnl_pct')} | {reason}{msg_suffix}")

    # ── State persistence ───────────────────────────────────────────
    def _load_state(self) -> Dict[str, Any]:
        default = {
            "enabled": False,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "counters": {},
            "last_action_ts": {},
            "last_reverse_ts": {},
            "last_decision": [],
        }
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    data = json.load(f)
                # Day rollover
                today = datetime.now().strftime("%Y-%m-%d")
                if data.get("date") != today:
                    data["date"] = today
                    data["counters"] = {}
                    data["last_action_ts"] = {}
                    data["last_reverse_ts"] = {}
                for k, v in default.items():
                    data.setdefault(k, v)
                return data
        except Exception as e:
            print(f"⚠️ AutoPilot state load failed: {e}")
        return default

    def _save_state(self):
        try:
            # Merge: read current disk (may contain a fresh "enabled" flip from
            # the dashboard) and combine with our in-memory counters/decisions.
            disk = {}
            try:
                if os.path.exists(STATE_PATH):
                    with open(STATE_PATH, "r") as f:
                        disk = json.load(f) or {}
            except Exception:
                disk = {}
            merged = dict(self._state)
            # dashboard only mutates "enabled" — trust it if present
            if isinstance(disk, dict) and "enabled" in disk:
                merged["enabled"] = bool(disk.get("enabled"))
            tmp = STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2, default=str)
            os.replace(tmp, STATE_PATH)
            self._state = merged
        except Exception as e:
            print(f"⚠️ AutoPilot state save failed: {e}")


# ── Singleton accessor ──────────────────────────────────────────────
_SINGLETON: Optional[AutoPilot] = None


def get_auto_pilot(tools_ref=None) -> Optional[AutoPilot]:
    global _SINGLETON
    if _SINGLETON is None and tools_ref is not None:
        _SINGLETON = AutoPilot(tools_ref)
    return _SINGLETON
