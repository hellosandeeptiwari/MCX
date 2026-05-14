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

# ── Feature snapshot writer (May 2026) ──────────────────────────────
# Per-tick dump of (ts, sym, features, decision) to a daily jsonl. Forms
# the replay corpus for the offline MC backtest harness — without this,
# we cannot tune thresholds against historical reality. Disabled by
# setting AP_FEATURE_SNAPSHOT=0 in env. ~50-150 MB/day uncompressed
# (depends on open-position count). Compress old days as needed.
FEATURE_SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "bandit_corpus")
FEATURE_SNAPSHOT_ENABLED = os.environ.get("AP_FEATURE_SNAPSHOT", "1") not in ("0", "false", "False", "no")
# Policy version — bump whenever AP gating logic changes so off-policy
# evaluation can segment the corpus by policy era. Format: ap_v_YYYY_MM_DD.
POLICY_VERSION = os.environ.get("AP_POLICY_VERSION", "ap_v_2026_05_04")


# ── Slope-driven thresholds ─────────────────────────────────────────
# Philosophy: read the UNDERLYING's tick-by-tick price (option premium is a
# lagging, noisy echo). Fit a short OLS slope on the last few seconds, require
# the slope to SUSTAIN (same-sign seconds) and reject CHOPPY windows. Act on
# the underlying's trend, not on option-premium P&L wiggles.
class Thresholds:
    TICK_INTERVAL_SEC = 1.0               # feature-extraction cadence
    WINDOW_SEC = 420                      # rolling underlying-tick window (5-min macro + 2-min headroom for tick latency / GC pauses)
    SLOPE_WINDOW_SEC = 5                  # primary slope regression window
    CONFIRM_WINDOW_SEC = 3                # short confirmation window (must agree in sign)
    MACRO_WINDOW_SEC = 45                 # higher-timeframe trend filter (2026-05-04: 60→45s for faster macro confirmation; feature key 'slope_60s_bps' retained for back-compat)
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
    REVERSE_MIN_BPS_S = 0.5               # minimum counter-slope magnitude to REVERSE (2026-05-04: 0.6→0.5 to fire earlier; new quality gate filters noise)
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

    # Slope-magnitude adaptive threshold (May 1). Complements VOL_ADAPTIVE
    # (which scales by tick-level realised vol) by also scaling by the
    # underlying's *directional* slope magnitude over a 5-min rolling window.
    # Rationale: a name with consistently strong slopes (BANKNIFTY-class) should
    # require a proportionally stronger spike to count as a regime flip; a slow
    # grinder (NATIONALUM-class) keeps the static floor. Effective REVERSE
    # threshold = max(static_floor, vol_adaptive, slope_adaptive).
    # Never lowers the bar — only raises it when warranted.
    SLOPE_ADAPTIVE = True
    SLOPE_ADAPTIVE_LOOKBACK_SEC = 300     # 5-min rolling history of |slope_5s|
    SLOPE_ADAPTIVE_MULT = 1.15            # threshold = 1.15 × recent avg |slope| (2026-05-04: 1.3→1.15 to fire earlier in trending regimes)
    SLOPE_ADAPTIVE_MIN_SAMPLES = 60       # require ≥1 min warm-up before activating

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
    MACRO_MIN_ABS_BPS_S = 0.08            # |slope_45s| must be ≥ this for counter-slope REVERSE (DISABLED via MACRO45S_GATE_ENABLED=False on 2026-05-05)
    MACRO45S_GATE_ENABLED = False         # 2026-05-05: removed 45s macro-confirm gate entirely (user request) — reverses fire on 5s+3s + reverse-quality-gate only

    # Multi-timeframe macro confirmation: if slope_300s (5-min) is computable
    # (non-zero = window warmed) and has the SAME sign as trade_dir, the
    # position is on the right side of the multi-minute trend. The current
    # bleed is theta/IV noise, NOT a trend flip. Reject counter-slope REVERSE.
    # Also protects against the "post-restart blind spot" (first ~2 min after
    # titan-bot restart, slope_300s is still warming at 0).
    REQUIRE_MACRO5M_AGAINST_FOR_REV = False  # 2026-05-05: disabled (over-restrictive on flat-trend tape)

    # PRESTIGE 2026-04-29 RCA: when slope_300s rounds to 0.0 (slow grinder
    # under bps resolution), the warm-up exception used to bypass macro
    # protection — flipping winning positions on 60-sec wicks. After 120s
    # of hold, slope_300s==0.0 is treated as "trend not broken" and we
    # require entry-to-now adverse underlying drift ≥ this floor before
    # allowing a counter-slope REVERSE. This is the macro-authority check
    # when the slope feature is rounding-flat.
    MACRO5M_FLAT_REV_MIN_DRIFT_BPS = 8.0   # 0.08% adverse u-drift required

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
    BLEED_PNL_PCT = -4.0                  # pnl% at which bleed-override arms (2026-05-05: -8.0→-4.0 scaled with MEDIUM_BLEED -5→-2.5)
    BLEED_MIN_SLOPE_BPS_S = 0.3           # min |slope5s| against us to fire override
    BLEED_MAX_CHOP = 80                   # 2026-05-06: cap chop — was bypassing chop ceiling, net −₹1.26L
    BLEED_MIN_MACRO_BPS_S = 0.10          # 2026-05-06: require slope_60s adverse ≥ 0.10 bps/s

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
    MIN_HOLD_FOR_REVERSE = 60             # Apr 30: REVERSE specifically needs 60s of tape (was 15s; FEDERALBNK 280PE flipped at 23s for ₹1500)
    COOLDOWN_BETWEEN_DECISIONS = 8        # Apr 29: 15→8 (entry chop already filtered by MIN_HOLD; 15s was too laggy on real moves)
    # Post-REVERSE per-symbol lockout. Once we flip a symbol, freeze AP on it
    # for this many seconds regardless of signal. Kills the whipsaw loop we
    # saw on JIOFIN / INFY where a reverse at -12% immediately got another
    # -12% reverse on the new leg, compounding the loss.
    # Apr 29: tightened 120 → 75 because slope_300s no longer needs ~120s to
    # warm after a flip (underlying-keyed window now survives flips), so the
    # macro signal is available earlier and EMERGENCY_REREV_ESCAPE remains the
    # one-shot safety valve for genuinely wrong flips.
    POST_REVERSE_LOCKOUT_SEC = 45
    # Per-symbol cap. Raised 10 → 50 per user directive 2026-04-24:
    # "I don't care how many reversals, cut my loss wherever it occurs".
    # 120s lockout + chop/exhaustion gates still bound ping-pong in practice.
    MAX_REVERSES_PER_DAY = 50
    MAX_ADDS_PER_DAY = 2

    # ── Reverse Quality Gate (2026-05-04 RCA, hard-block) ──────────
    # Counterfactual analysis on 2026-05-04's 42 reverse trades showed
    # that a 2-feature gate would have changed day P&L from −₹59,728 to
    # +₹52,345 by blocking 24 of 25 losers while keeping 12 of 17 winners
    # (kept-pnl +₹134,504 vs keep-all +₹21,998 → +₹112k delta on the day).
    # Gate (must satisfy BOTH; applied to every REVERSE before LLM/exec):
    #   1. slope_5s in NEW reverse direction ≥ +0.23 bps/s
    #      (= -slope_5s_bps * trade_dir; new_dir = -trade_dir)
    #      Rationale: rejects stale-trigger reverses where the 5s slope
    #      has already mean-reverted by the time we'd fire.
    #   2. chop_flips ≤ 98
    #      Rationale: rejects pure-noise tape (≥99 direction changes /
    #      5min = no real direction to reverse INTO).
    # Source: _ap_reverse_quality_classifier.py 2026-05-04 output.
    # Single-day backtest — expect 30-60% shrinkage out-of-sample;
    # realistic expected lift ~₹40-70k/day.
    REVERSE_GATE_SLOPE5S_WITH_NEW_MIN_BPS_S = 0.05
    REVERSE_GATE_CHOP_FLIPS_MAX = 80  # 2026-05-06: data-driven (3-day per-tick oracle) — chop≤80 was net +₹1.03cr, no cap was net negative
    # Bleed-bypass kept for safety symmetry (no longer needed since chop cap is off)
    REVERSE_GATE_BLEED_BYPASS_PNL_PCT = -5.0
    REVERSE_GATE_BLEED_BYPASS_MIN_HOLD = 600

    # ── Whipsaw circuit-breaker (PRESTIGE 2026-05 RCA) ─────────────
    # POST_REVERSE_LOCKOUT_SEC blocks immediate re-REVERSE but does NOT
    # block ADD_LOT firing on a brief 5s blip favouring the doomed
    # direction between reverses. Result: REVERSE→ADD→ADD→ADD→REVERSE
    # ping-pong while underlying grinds against the original position.
    # When 4+ REVERSEs hit in a 15-min window, the macro thesis is
    # demonstrably broken — halt all AP action on this symbol and let
    # the exit_manager / SL handle exit. Don't keep paying spread.
    WHIPSAW_FLIP_WINDOW_SEC = 900    # rolling 15-min window
    WHIPSAW_MAX_FLIPS = 4            # ≥ this many REVERSEs in window → freeze
    # Same RCA: ADD_LOT must not reinforce a position whose UNDERLYING
    # has already drifted meaningfully against entry. PRESTIGE PE got
    # ADDed on 5s downward blips while underlying made new day-highs.
    # u_drift_against_bps measures cumulative entry-to-now adverse move
    # in the underlying — if ≥ this, the macro thesis is wrong; refuse ADD.
    ADD_LOT_VETO_DRIFT_AGAINST_BPS = 8.0   # symmetric with MACRO_DRIFT_U_BPS (was 12.0; if drift is enough to REVERSE, never ADD)
    ADD_LOT_VETO_OPT_STALE_SEC = 20.0      # if option quote frozen ≥20s, refuse ADD — can't pyramid into a frozen book

    # Per-underlying flip cooldown (Apr 29 follow-up). The existing
    # POST_REVERSE_LOCKOUT_SEC is keyed off the OPTION symbol — so a
    # CE→PE flip on the same underlying resets it, allowing instant
    # PE→CE re-flip. PRESTIGE 11:05:00→11:05:32 (32-second whipsaw)
    # is the canonical bug. Track the last flip ts on the UNDERLYING
    # so any CE/PE flip on that name respects the gap.
    MIN_U_FLIP_GAP_SEC = 45                # any reverse on same underlying must wait ≥45s

    # ── Late-session BLEED override (Apr 29) ──
    # Root cause from FEDERALBNK 290CE 15:09→15:22 EOD nuke (−₹16k, −25%):
    # entry was a panic REVERSE @ 15:09; 75s lockout + AP gate cooldown +
    # exit_manager not having installed an SL on a manually-flipped leg yet
    # meant the position rode straight into the 15:22 EOD auto-close at
    # the worst possible price. After ~15:00 IST, theta + close-of-day
    # liquidity-thinning makes any deeply-red position a one-way bleed —
    # there is no time left for it to recover. Override:
    #   - if HH:MM ≥ LATE_SESSION_HHMM AND pnl_pct ≤ LATE_SESSION_BLEED_PCT
    #     and proposed action is REVERSE, BYPASS post-reverse lockout AND
    #     per-underlying flip cooldown so we can flip / cut without delay.
    LATE_SESSION_HH = 15
    LATE_SESSION_MM = 0                    # active from 15:00 IST onwards
    LATE_SESSION_BLEED_PNL_PCT = -4.0      # any leg ≤ −4% in last 15min → cut/flip immediately (2026-05-05: -8.0→-4.0 scaled)

    # ── Order-book / spread veto (Apr 29) ──
    # On pinned strikes (e.g. FEDERALBNK 287.5 today) the bid-ask spread
    # widens to 1.5–2% of premium during chop. Every REVERSE pays this
    # spread on both legs — 11 flips at 2% = 22% premium burn before any
    # directional move. Refuse to flip when spread is too wide; tape is
    # telling us the strike is illiquid right now.
    SPREAD_VETO_PCT = 1.5                  # max bid-ask spread as %% of mid
    SPREAD_RELAX_LATE_PCT = 2.5            # late-session: allow wider (cut > spread)

    # ── PARK action (Apr 29) ──
    # Selective participation: when a symbol is in confirmed chop AND we
    # have no edge, exit to cash and stop trading it. Re-entry is handled
    # by the watcher pipeline naturally when a real signal returns.
    # Triggers (any one is enough):
    #   1. WHIPSAW_FREEZE just fired (≥4 reverses in 15min)
    #   2. chop_flips ≥ PARK_CHOP_FLIPS AND |slope_300s| ≤ PARK_FLAT_SLOPE_BPS_S
    #      (high-frequency tick reversal with no macro drift = pure noise)
    # Guards:
    #   - PARK only when pnl_pct ≤ PARK_MAX_PNL_PCT (do NOT park a winner)
    #   - per-symbol PARK cooldown so we don't re-park the watcher's
    #     legitimate fresh entries
    PARK_CHOP_FLIPS = 25
    PARK_FLAT_SLOPE_BPS_S = 0.05           # macro is essentially flat
    PARK_MIN_HOLD_SEC = 240                # don't park before 4min (Apr 30: was 90 — too aggressive on early winners)
    PARK_MAX_PNL_PCT = 1.0                 # only park flat / red positions
    PARK_COOLDOWN_SEC = 600                # 10min before re-evaluating same symbol
    # Early-winner protection (Apr 30): never PARK a young, green position.
    # Must satisfy BOTH guards — a 5-min-old +0.4% trade is still nascent.
    PARK_EARLY_WINNER_PNL_PCT = 0.3        # if pnl > 0.3%
    PARK_EARLY_WINNER_HOLD_SEC = 480       #   AND hold < 8min, do NOT park.

    # ── Decoupled-strike detection (Apr 29 — RBLBANK 340CE RCA) ──
    # When the underlying moves favorably (or at least neutrally) but the
    # option premium is decaying, the strike is wrong — too far OTM, or
    # ATM with theta dominating delta on a stalled underlying. Two paths:
    #   1. ADD_LOT veto: refuse to double-down into a decoupled strike.
    #      A losing option with NO underlying reason to lose = pure theta;
    #      adding more multiplies the theta exposure.
    #   2. PARK: after a long hold of confirmed decoupling, flatten so the
    #      watcher can re-pick a properly-tuned strike (or stand aside).
    # u_drift_against_bps is signed: + means underlying went against us,
    # − means underlying went WITH us. ≤ 0 = neutral-or-favorable.
    DECOUPLE_ADD_HOLD_SEC = 300            # 5min — long enough to be stable
    DECOUPLE_ADD_PNL_PCT = -0.5            # option clearly red
    DECOUPLE_ADD_U_DRIFT_AGAINST_BPS_MAX = 0  # underlying not against us

    PARK_DECOUPLE_HOLD_SEC = 600           # 10min — confident decoupling call
    PARK_DECOUPLE_PNL_PCT = -1.0           # ≥1% bleed
    PARK_DECOUPLE_U_DRIFT_AGAINST_BPS_MAX = 50  # underlying near-neutral
    PARK_DECOUPLE_OPT_SLOPE_MAX_BPS_S = 0.5     # option tape calm (no recovery)

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
    MACRO5M_PNL_PCT = -1.5                # 2026-05-06: 0.0→-1.5 (per adverse analysis: zero-pnl flips were net negative)
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
    CLEAN_DRIFT_MIN_HOLD = 90             # 1.5 min minimum (relaxed 180→90; underlying window survives flips)
    CLEAN_DRIFT_MAX_CHOP = 25             # clean tape (relaxed 15→25)
    CLEAN_DRIFT_MIN_U_DRIFT_BPS = 5.0     # underlying moved ≥5 bps (relaxed 10→5)
    CLEAN_DRIFT_PNL_PCT = -1.0            # any shallow red (relaxed -2→-1)

    # ── Macro-drift REVERSE (Apr 29) ──
    # Drift-primary trigger that does NOT depend on premium pnl. Catches the
    # "underlying moved 30+ bps but option premium hasn't caught up" scenario
    # on low-delta / illiquid options. The user's specific complaint: AP is
    # slow because every existing REVERSE gate keys off pnl_pct, and on a low-
    # delta strike the underlying can drift 0.3-0.5% before premium drops the
    # required -1.5%. This gate fires on underlying drift + macro slope alone.
    # Conditions (ALL required):
    #   - hold_sec >= MACRO_DRIFT_MIN_HOLD     (warm-up filter, not entry chop)
    #   - u_drift_against_bps >= MACRO_DRIFT_U_BPS  (underlying moved meaningfully against us)
    #   - sign(slope_300s) opposes trade_dir AND |slope_300s| >= MACRO_DRIFT_S300_BPS_S
    #     (macro confirms the move is still going, not retrace)
    #   - chop_flips < MACRO_DRIFT_MAX_CHOP    (skip pure whipsaw)
    # Intentionally has NO premium-pnl floor — that's the whole point.
    MACRO_DRIFT_MIN_HOLD = 60
    MACRO_DRIFT_U_BPS = 8.0                # 0.08% adverse underlying drift (relaxed 12→8 — same floor as STALE_LOSS_DRIFT_BPS)
    MACRO_DRIFT_S300_BPS_S = 0.10         # macro 5-min still moving against us
    MACRO_DRIFT_MAX_CHOP = 30
    MACRO_DRIFT_PNL_PCT = -1.0            # 2026-05-06: don't fire on flat positions; require some bleed

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
    BYPASS_REV_PNL_PCT = -10.0            # 2026-05-06: -6→-10 (per adverse-rule analysis: 3094 fires net −₹4.18cr)
    BYPASS_REV_U_DRIFT_BPS = 40.0         # 2026-05-06: 25→40 (require very large underlying drift)
    BYPASS_ADD_MIN_HOLD = 300
    BYPASS_ADD_PNL_PCT = 4.0              # already running
    BYPASS_ADD_U_DRIFT_WITH_BPS = 25.0    # underlying drifted WITH us by ≥25 bps

    # Macro-trend ADD_LOT (Apr 29). 56% of AP evals show slope_5s≈0 because
    # the 5s window's bps resolution is coarser than typical tick spacing on
    # liquid stocks (e.g. PRESTIGE @ ₹1432 unchanged for 5s = literally 0).
    # The strict |s5|≥2.0 ADD gate is unreachable for slow-grinder winners
    # (ABCAPITAL hit +36% with s5 only 0.86). Macro-trend ADD lets us pyramid
    # winners on the 5-MINUTE trend even when the tick tape is flat:
    #   - aligned (sign5 == trade_dir) and sign3 agrees (no tail reversal)
    #   - slope_300s aligned with trade_dir AND |slope_300s| ≥ 0.5 bps/s
    #   - pnl% ≥ 4 (already running)
    #   - chop < 30 (clean tape — stricter than chop_bypass_add)
    #   - |slope_5s| ≥ 0.5 (not pure noise; relaxed from 2.0)
    MACRO_TREND_ADD_MIN_S5_BPS_S = 0.5
    MACRO_TREND_ADD_MIN_S300_BPS_S = 0.5
    MACRO_TREND_ADD_MIN_PNL_PCT = 4.0
    MACRO_TREND_ADD_MAX_CHOP = 30

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
    STALE_LOSS_PNL_PCT = -3.0             # relaxed -4→-3 (cut sooner, don't wait for deeper bleed)
    STALE_LOSS_MIN_HOLD = 120             # 2 minutes (relaxed 300→120 — was wasting time waiting)
    STALE_LOSS_DRIFT_BPS = 6.0            # 0.06% adverse on underlying (relaxed 8→6)
    STALE_LOSS_MAX_CHOP = 110             # only skip when tape is a pure washing machine

    # Medium-bleed REVERSE path. Fills the gap between the normal REV gate
    # (which is blocked by even mild chop) and bleed-override (which requires
    # -12% pnl). Catches the LUPIN-style scenario: position is moderately red,
    # a strong counter-spike is in progress with 3s+60s confirmation, but NO
    # single dimension is extreme enough to trip an existing gate. Fires only
    # when FOUR medium conditions co-occur — the multi-axis evidence makes up
    # for not having one extreme axis.
    MEDIUM_BLEED_PNL_PCT = -2.5           # meaningful red (2026-05-05: -5.0→-3.0→-2.5 — fire reverse on shallower bleed; quality gate filters noise)
    MEDIUM_BLEED_MIN_SLOPE_BPS_S = 0.7    # 2026-05-06: 0.4→0.7 (per adverse analysis: 59 fires net −₹3.32L)
    MEDIUM_BLEED_MAX_CHOP = 50            # 2026-05-06: 45→50 — actually keep the cap but tighten slope, chop wasn't the issue
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
    # 2026-05-06: relaxed per 3-day per-tick oracle (610 legs / 330k ticks).
    # Best-net green rule was pnl≥1.5, |s5|≥0.3, |s3|≥0.3, chop≤80, hold≥60.
    # Old thresholds (0.8/1.2/35) missed CHOLAFIN/HAL/BHARATFORG gave-back legs.
    PROFIT_PEAK_PNL_PCT = 2.0             # ≥+2% open profit (was 3.0)
    PROFIT_PEAK_MIN_HOLD = 60             # let the winner develop first
    PROFIT_PEAK_MAX_CHOP = 80             # 2026-05-06: 35→80 per oracle
    PROFIT_PEAK_MIN_S5_BPS_S = 0.3        # 2026-05-06: 0.8→0.3 per oracle
    PROFIT_PEAK_MIN_S3_BPS_S = 0.3        # 2026-05-06: 1.2→0.3 per oracle

    # ── Stale-Winner BOOK_PROFIT (2026-05-05) ──
    # Lock in modest profits when a position has gone flat for a long time.
    # Triggers when:
    #   - STALE_WINNER_PNL_PCT_MIN ≤ pnl_pct ≤ STALE_WINNER_PNL_PCT_MAX (5–11%)
    #   - hold_sec  ≥ STALE_WINNER_MIN_HOLD
    #   - |slope_300s_bps| < STALE_WINNER_MAX_SLOPE_BPS_S (5-min trend flat)
    #   - chop_flips     < STALE_WINNER_MAX_CHOP (not pure whipsaw)
    # Above 11% the existing partial_profit / trailing-SL rules take over.
    STALE_WINNER_PNL_PCT = 5.0
    STALE_WINNER_PNL_PCT_MAX = 11.0
    STALE_WINNER_MIN_HOLD = 400
    STALE_WINNER_MAX_SLOPE_BPS_S = 0.05
    STALE_WINNER_MAX_CHOP = 60

    # Per-symbol realized-loss cap. After a single underlying has accumulated
    # this much realized loss in today's ledger (from ANY source — AP reverses,
    # SL hits, time stops), AP applies action-aware gating on that symbol:
    #   • ADD_LOT → BLOCKED (no doubling down on losers).
    #   • REVERSE → ALLOWED up to MAX_POST_CAP_REVERSES, since REVERSE is the
    #     *escape* path — freezing it locks the position into further bleed.
    #     Existing POST_REVERSE_LOCKOUT_SEC + macro-agreement gates filter
    #     weak flips. After MAX_POST_CAP_REVERSES, truly frozen (money sink).
    # Apr 29 redesign — the prior "hard freeze" blocked PERSISTENT from being
    # rescued via flip when chart clearly turned against it.
    PER_SYMBOL_REALIZED_LOSS_CAP = -10000.0  # 2026-05-05: loosened from -3000 (BHEL RCA — cap was triggering rescue-mode prematurely)
    PER_SYMBOL_CAP_REFRESH_SEC = 30  # ledger-scan cache interval
    MAX_POST_CAP_REVERSES = 3        # max REVERSE flips per cap-breached symbol

    # ── Regime detector (Apr 27): oscillator vs breakout ──
    # Built from the 5-min underlying range + position-within-range:
    #   range_300s_bps = (hi-lo)/start * 1e4 over last 300s
    #   pos_in_range_300s = (u_now - lo) / (hi - lo)   in [0, 1]
    #   ret_over_range_300s = |net_300s| / range_300s  in [0, 1+]
    # An OSCILLATOR is: small/medium range, low net-over-range (ping-pong),
    # i.e. the market is swinging within a band, not breaking out.
    # A BREAKOUT is: meaningful range AND price pinned at one extreme AND
    # macro 60s slope confirming the breakout direction.
    OSC_RANGE_MIN_BPS = 20.0              # need at least 0.2% range to bother
    OSC_RANGE_MAX_BPS = 80.0              # above 0.8% it's trending, not oscillating
    OSC_NET_OVER_RANGE_MAX = 0.40         # |net|/range < 0.4 = ping-pong (vs trending which is ~0.7-1.0)
    OSC_MIN_SAMPLES = 120                 # need ~2min of tape before declaring regime

    # Oscillator-friendly PROFIT_PEAK — fires earlier on smaller swings BUT
    # only when the tape is genuinely an oscillator (above flags true).
    # Catches the VEDL-style 4-5 swings/day each ~0.4-0.6% on the underlying
    # that wouldn't trip the standard PROFIT_PEAK +3% / 0.8 bps/s thresholds.
    # 2026-05-06: oscillator slightly tighter than standard profit-peak
    OSC_PROFIT_PEAK_PNL_PCT = 1.5         # smaller pnl threshold
    OSC_PROFIT_PEAK_MIN_S5_BPS_S = 0.3    # 2026-05-06: 0.6→0.3 per oracle
    OSC_PROFIT_PEAK_MIN_S3_BPS_S = 0.3    # 2026-05-06: 0.9→0.3 per oracle
    OSC_PROFIT_PEAK_MIN_HOLD = 60         # same as standard
    OSC_PROFIT_PEAK_MAX_CHOP = 60         # 2026-05-06: 30→60 per oracle

    # Breakout-friendly ADD_LOT — fires when price is at one extreme of the
    # 5-min range AND macro 60s slope is breaking in our favor. Lower
    # threshold than macro5m_add because the range break itself is the signal.
    BREAKOUT_RANGE_MIN_BPS = 30.0         # need real range to call it a breakout
    BREAKOUT_POS_HIGH = 0.85              # CE breakout: price in top 15% of range
    BREAKOUT_POS_LOW = 0.15               # PE breakout: price in bottom 15% of range
    BREAKOUT_MACRO_BPS_S = 0.15           # 60s macro must agree with breakout direction
    BREAKOUT_ADD_PNL_PCT = 4.0            # 2026-05-06: 2.0→4.0 (per adverse-rule analysis: 22 fires net −₹78k)
    BREAKOUT_ADD_MAX_CHOP = 30            # tape must be cleanish at the breakout edge

    # Kept for LLM context only (NOT gating)
    VOL_SURGE_MULTIPLIER = 1.3
    DEPTH_IMBALANCE_RATIO = 1.25

    # Book-imbalance confirmation on REVERSE (Apr 30):
    # book_imb = buy_qty / (buy_qty + sell_qty) on the HELD option.
    # Reversing CE→PE (or PE→CE) implies the held leg should be unloading
    # (sellers stacking the book) — book_imb ≤ confirm threshold confirms.
    # If buyers are stacking (book_imb ≥ veto threshold), the book disagrees
    # with reversing and we skip. Late-bleed / emergency-rerev escape bypass.
    BOOK_IMB_REVERSE_VETO = 0.55         # ≥ this → veto REVERSE (book buying held leg)
    BOOK_IMB_REVERSE_NEUTRAL = 0.45      # 0.45–0.55 = neutral, allowed through
    BOOK_IMB_MIN_TOTAL_QTY = 50          # below this depth, skip the gate (noise)

    # Strike migration (Apr 30, log-only Phase 1):
    # When held option is deep OTM after a favorable underlying move, the
    # leg has lost gamma and theta is now eating premium faster than delta
    # is feeding it. Migration = exit current strike + re-enter ATM same
    # type (CE→CE / PE→PE), recentering on max gamma. Only fires when:
    #   • trade is ALIGNED (slope still in our direction)
    #   • GREEN ≥ STRIKE_MIGRATE_MIN_PNL_PCT (lock realized gain on exit)
    #   • underlying drifted FAVORABLY ≥ STRIKE_MIGRATE_MIN_DRIFT_BPS
    #   • current strike is now ≥ N strikes OTM from new ATM
    #   • hold ≥ STRIKE_MIGRATE_MIN_HOLD_SEC (no entry-tape noise)
    # Phase 1 = log-only (DRY_RUN). Phase 2 = wire execute path once user
    # reviews picks and confirms threshold tuning.
    STRIKE_MIGRATE_ENABLED = True
    STRIKE_MIGRATE_DRY_RUN = True              # log plans, don't execute
    STRIKE_MIGRATE_MIN_PNL_PCT = 3.0           # need real gain to lock
    STRIKE_MIGRATE_MIN_HOLD_SEC = 120          # avoid entry-tape noise
    STRIKE_MIGRATE_MIN_DRIFT_BPS = 100.0       # ≥1% favorable underlying drift
    STRIKE_MIGRATE_MIN_SLOPE_BPS_S = 0.20      # momentum still with us
    STRIKE_MIGRATE_MIN_STRIKES_OTM = 1         # current strike ≥ 1 OTM from new ATM
    STRIKE_MIGRATE_COOLDOWN_SEC = 600          # one migration per name / 10min
    STRIKE_MIGRATE_MAX_PER_DAY = 4             # day-cap per symbol

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

        # Per-symbol rolling tick windows: {symbol: deque[(ts, ltp, buy_qty, sell_qty, volume, u_ltp)]}
        # Also stores float timestamps under sentinel keys (e.g. "__diag_{sym}__") for log throttling.
        self._windows: Dict[str, Any] = {}

        # Per-UNDERLYING tick window — survives option-symbol flips. Without this,
        # a REVERSE wipes the slope/300s/range/regime history (option-keyed deque
        # is dropped) and AP sits in warmup blind for ~2 min on the new leg, which
        # is exactly when the macro signal is most needed. Keyed e.g. "NSE:RELIANCE".
        # Tuple: (ts, u_ltp).
        self._u_windows: Dict[str, deque] = {}

        # Per-underlying realized-loss cache + refresh timestamp.
        # Populated from trade_ledger_{today}.jsonl every PER_SYMBOL_CAP_REFRESH_SEC.
        # Used to gate AP actions on symbols that have already bled out.
        self._realized_by_underlying: Dict[str, float] = {}
        self._realized_cache_ts: float = 0.0

        # Feature snapshot writer state (per-tick replay corpus for MC
        # backtest harness). File handle is opened lazily on first call
        # and rotated on date change.
        self._fs_lock = threading.Lock()
        self._fs_handle: Optional[Any] = None
        self._fs_date: Optional[str] = None
        # Regime tag cache (refreshed every 60s in _get_regime_tags). Lets
        # off-policy slicing pivot on VIX bucket / Nifty trend / day-of-week.
        self._regime_cache: Dict[str, Any] = {}
        self._regime_cache_ts: float = 0.0
        self._nifty_open_by_date: Dict[str, float] = {}

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
        # === LLM outage fallback (Apr 27) ============================
        # If the LLM fails N times in a row, switch to RULES_ONLY mode for
        # high-confidence rules (bleed_override, macro5min_counter,
        # stale_loss, chop_bypass_rev, emergency_rerev_escape) so AP isn't
        # totally frozen during an OpenAI outage. Marginal rules still need
        # LLM. Recovers on first successful call.
        self._llm_consecutive_failures: int = 0
        self._llm_outage_threshold: int = 3
        self._llm_high_conf_boosters = {
            "bleed_override", "macro5min_counter", "macro_drift_rev", "stale_loss",
            "chop_bypass_rev", "chop_bypass_add", "emergency_rerev_escape",
            "momentum_override", "breakout_add", "osc_profit_peak",
            "macro_trend_add", "late_session_bleed", "park_chop",
            "park_decoupled",
        }

        # ── Failed-winner protection (2026-05-10) ─────────────────────
        # Per-symbol peak P&L ratchet, used to refuse REVERSE on positions
        # that previously showed real profit (≥+3%). Counterfactual on May
        # 5-8 OI_WATCHER trades showed 5/15 "failed winner" exits came
        # via AP REVERSE — trades that peaked +5% to +14% then reversed
        # at -2.3% into the give-back, compounding losses. Hard SL /
        # trailing SL handle these exits more cleanly than a flip.
        # Format: {sym: (last_update_ts, peak_pnl_pct)}.
        # TTL via timestamp — entries auto-stale after 5min inactivity
        # (= position closed, no longer being tick-fed).
        self._peak_pnl_by_sym: Dict[str, Tuple[float, float]] = {}
        # 2026-05-14: per-symbol loss-sustain counter for the -2.3% reverse rule.
        # Single bad ticks (bid-side prints, wide-spread option noise) used to
        # trigger false reverses (ADANIENT 14:50 incident: one tick at ~₹90
        # showed pnl%=-4.4% on a position truly at -1%). Now require 3
        # consecutive ticks below threshold before firing.
        self._loss_sustain_count: Dict[str, int] = {}

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
                # Repeated-error alarm (Apr 30): a swallowed tick error that
                # keeps recurring means AP is silently dead. Track frequency
                # and emit a LOUD alarm + full stack at escalating thresholds
                # so a regression like 'reverses' KeyError can't silently
                # disable AP for an hour again.
                err_key = str(e)[:120]
                err_state = self._state.setdefault("_tick_err_alarm", {})
                rec = err_state.setdefault(err_key, {"count": 0, "first_ts": time.time(), "last_alarm": 0.0})
                rec["count"] += 1
                rec["last_ts"] = time.time()
                cnt = rec["count"]
                # Escalation thresholds: 5, 25, 100, then every 200
                fire_alarm = (
                    cnt in (5, 25, 100)
                    or (cnt > 100 and cnt % 200 == 0)
                )
                if cnt == 1 or fire_alarm:
                    print(f"⚠️ AutoPilot tick error: {e}", flush=True)
                    traceback.print_exc()
                if fire_alarm:
                    age = time.time() - rec["first_ts"]
                    print(
                        f"🚨🚨🚨 AutoPilot ALARM: same tick error repeated "
                        f"{cnt}x in {age:.0f}s — AP IS LIKELY DEAD on this code path. "
                        f"Error: {err_key}",
                        flush=True,
                    )
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

        # Dashboard pause flag — block all reverses while paused.
        # Trailing SL / target / EOD logic in exit_manager are unaffected.
        try:
            _pause_flag = os.path.join(os.path.dirname(__file__), 'trading_paused.flag')
            if os.path.exists(_pause_flag):
                self._skip_counters["dashboard_paused"] = self._skip_counters.get("dashboard_paused", 0) + 1
                return
        except Exception:
            pass

        now_wall = time.time()
        symbols_seen = set()
        # Per-tick underlying LTP cache: positions sharing an underlying
        # (e.g. multi-leg PRESTIGE) hit ticker.get_ltp() multiple times per
        # tick otherwise. Trivial savings but cleaner.
        u_ltp_cache: Dict[str, float] = {}
        # Track which underlyings we already appended this tick — multiple
        # open positions can share the same underlying; we only want one
        # u_ltp sample per underlying per tick (otherwise slope regression
        # would over-weight the latest sample and corrupt timestamps).
        underlyings_written: set = set()
        # Track whether ANY state mutation happened this tick. Cache-hit-only
        # ticks (likely during sustained regimes) need _save_state too — without
        # this, counters/last_action_ts mutations are only persisted on ticks
        # that drained at least one future, risking state loss on crash.
        self._tick_mutated: bool = False

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

            # Order-book microstructure (Apr 29):
            #   spread_pct  — bid-ask width as %% of mid (proxy for liquidity)
            #   book_imb    — total_buy_qty / (total_buy_qty + total_sell_qty)
            #                 0.0 = all sell pressure, 1.0 = all buy pressure
            _bid = float(quote.get("bid") or 0)
            _ask = float(quote.get("ask") or 0)
            spread_pct = 0.0
            if _bid > 0 and _ask > 0 and _ask >= _bid:
                _mid = (_bid + _ask) * 0.5
                if _mid > 0:
                    spread_pct = ((_ask - _bid) / _mid) * 100.0
            _tot = buy_qty + sell_qty
            book_imb = (buy_qty / _tot) if _tot > 0 else 0.5

            # Underlying LTP — the real signal. Skip tick if unavailable
            # (we MUST have an underlying series to compute slope).
            u_key_lookup = pos.get("underlying") or ""
            if u_key_lookup in u_ltp_cache:
                u_ltp = u_ltp_cache[u_key_lookup]
            else:
                u_ltp = self._get_underlying_ltp(pos)
                if u_ltp and u_key_lookup:
                    u_ltp_cache[u_key_lookup] = u_ltp
            if not u_ltp:
                self._skip_counters["no_quote"] += 1
                continue

            win = self._windows.setdefault(sym, deque(maxlen=int(Thresholds.WINDOW_SEC / Thresholds.TICK_INTERVAL_SEC)))
            win.append((now_wall, ltp, buy_qty, sell_qty, volume, u_ltp))

            # Append to the underlying-keyed window once per (underlying, tick).
            # This window survives REVERSE flips, so slope_300s / range_300s /
            # regime classifier stay warm and AP is NOT blind for the first 2 min
            # after a flip.
            u_key = pos.get("underlying") or ""
            if u_key and u_key not in underlyings_written:
                u_win = self._u_windows.setdefault(
                    u_key,
                    deque(maxlen=int(Thresholds.WINDOW_SEC / Thresholds.TICK_INTERVAL_SEC)),
                )
                u_win.append((now_wall, u_ltp))
                underlyings_written.add(u_key)

            u_win_for_feat = self._u_windows.get(u_key) if u_key else None
            features = self._extract_features(win, pos, u_win=u_win_for_feat)
            if not features:
                self._skip_counters["warmup"] += 1
                continue
            # Inject microstructure features (post-extraction so we don't
            # touch the slope-pipeline contract).
            features["spread_pct"] = round(spread_pct, 2)
            features["book_imb"] = round(book_imb, 3)

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

            # Per-tick replay corpus for offline MC backtest harness.
            # Captures EVERY tick (whether or not a rule fires) so we can
            # re-simulate threshold sweeps against historical reality.
            self._log_feature_snapshot(sym, pos, features, decision)

            # ── Strike migration diagnostic (Apr 30, Phase 1 log-only) ──
            # Independent of REVERSE/ADD gates — fires when held leg has gone
            # too far OTM after a favorable move. Recentering at fresh ATM
            # restores gamma and locks realized gain on the exit leg.
            if Thresholds.STRIKE_MIGRATE_ENABLED:
                try:
                    self._check_strike_migrate(pos, features, now_wall)
                except Exception as _mig_e:
                    # Best-effort — never break the main loop
                    log_key = f"__migrate_err_{sym}__"
                    last_lk = self._windows.get(log_key, 0)
                    if now_wall - last_lk >= 300:
                        self._windows[log_key] = now_wall
                        print(f"⚠️ AutoPilot strike_migrate error {sym}: {str(_mig_e)[:120]}", flush=True)

            if not decision:
                self._skip_counters["no_gate"] += 1
                continue

            # ── HARD-RULE: simple-loss-rev (pnl<=-2.5% → REVERSE) ──
            # The ONLY rule. No gates, no LLM, no cooldowns, no lockouts.
            # User directive 2026-05-07: brutal hard stop, fires on same tick.
            counters = self._state.setdefault("counters", {}).setdefault(
                sym, {"reverses": 0, "adds": 0}
            )
            counters.setdefault("reverses", 0)
            counters.setdefault("adds", 0)

            # ── DAILY REVERSE CAP (per underlying symbol) ──
            # User-requested 2026-05-08: hard cap of 6 reverses/day per symbol
            # to prevent whipsaw cascades. After 6 flips, the position rides
            # without further reversal until next day. Keeps the rule alive
            # for legitimate moves but kills the noise tail.
            MAX_REVERSES_PER_DAY = 6
            if (decision.get("action") == "REVERSE"
                    and counters["reverses"] >= MAX_REVERSES_PER_DAY):
                # Throttle the log to once per minute per symbol.
                _cap_key = f"__capwarn_{sym}"
                _last = self._windows.get(_cap_key, 0)
                if now_wall - _last >= 60:
                    print(
                        f"🛑 AutoPilot CAP_HIT {sym} reverses={counters['reverses']}/"
                        f"{MAX_REVERSES_PER_DAY} — blocking further reverses today",
                        flush=True,
                    )
                    self._windows[_cap_key] = now_wall
                self._skip_counters["reverse_cap"] = (
                    self._skip_counters.get("reverse_cap", 0) + 1
                )
                continue

            print(
                f"🎯 AutoPilot RULE_FIRED {decision['action']} {sym} | "
                f"rule={decision.get('rule')} pnl%={features['pnl_pct']:+.2f} "
                f"— HARD-RULE (no gates, no LLM)",
                flush=True,
            )
            _v = {
                "action": decision["action"],
                "go": True,
                "confidence": 1.0,
                "reason": f"[hard-rule] {decision.get('rule', '')}",
                "_hard_rule": True,
            }
            self._handle_verdict(pos, features, decision, sym,
                                 counters, now_wall, _v)
            continue

        # Persist if anything mutated this tick (handles cache-hit-only ticks too).
        if self._tick_mutated:
            self._save_state()

        # LLM cache sweep: drop entries past TTL to prevent unbounded growth
        # over a 6-hour session. Cheap O(N) once per tick.
        if len(self._llm_cache) > 200:
            _cutoff = now_wall - self._llm_cache_ttl
            self._llm_cache = {
                k: v for k, v in self._llm_cache.items() if v[0] >= _cutoff
            }

        # Prune windows for symbols that are no longer open. Skip sentinel
        # throttling keys (start with '__') — those are NOT position symbols
        # and pruning them every tick was nuking all the rate-limited logs
        # (diag/loss_cap/rev_lock/trailing-active throttles).
        for stale in list(self._windows.keys()):
            if stale.startswith("__"):
                continue
            if stale not in symbols_seen:
                self._windows.pop(stale, None)
        # Also prune sentinel keys whose owning symbol is gone (e.g.
        # __diag_NFOXYZ__ when NFOXYZ closed). Keeps memory bounded without
        # destroying live throttles.
        _live_suffixes = symbols_seen
        for stale in list(self._windows.keys()):
            if not stale.startswith("__"):
                continue
            if not any(s in stale for s in _live_suffixes):
                self._windows.pop(stale, None)
        # Prune entry-u_ltp snapshots for closed symbols so re-entries start fresh.
        snap_map = self._state.get("u_entry_ltp", {})
        for stale in list(snap_map.keys()):
            if stale not in symbols_seen:
                snap_map.pop(stale, None)

        # Prune underlying-keyed windows for underlyings no longer represented
        # by any open position. Survives flips within the same underlying
        # (e.g. PE→CE on RELIANCE both share "NSE:RELIANCE") but releases
        # memory once the trade thesis is fully closed.
        underlyings_alive = {p.get("underlying") for p in positions if p.get("underlying")}
        for stale in list(self._u_windows.keys()):
            if stale not in underlyings_alive:
                self._u_windows.pop(stale, None)

    # ── Feature extraction ──────────────────────────────────────────
    def _extract_features(self, win: deque, pos: dict, u_win: Optional[deque] = None) -> Optional[Dict[str, Any]]:
        """Build slope features on the UNDERLYING tick stream.

        Window tuple: (ts, opt_ltp, buy_qty, sell_qty, volume, u_ltp)
        u_win (optional) is the long-lived underlying-keyed window
        (ts, u_ltp). When supplied, ALL underlying-derived features
        (slopes, sustain, chop, ret_15s/30s, range_300s, regime, vol)
        are computed from it so they survive REVERSE flips. The option-keyed
        `win` is only used for opt_ltp / depth / volume / current snapshot.
        Falls back to in-window u_ltp (pts[i][5]) when u_win is None
        (e.g. legacy callers / first tick before u_win is populated).

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

        # Underlying series — prefer the long-lived underlying-keyed window
        # (survives REVERSE flips). Fall back to the option-keyed in-window
        # u_ltp slot if u_win is unavailable / shorter (very early in life).
        if u_win is not None and len(u_win) >= len(pts):
            u_series_full = [(p[0], p[1]) for p in u_win if p[1]]
        else:
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

        # --- Slope-adaptive baseline (May 1) ---
        # Per-underlying rolling history of |slope_5s_bps| over the last
        # SLOPE_ADAPTIVE_LOOKBACK_SEC. Used downstream in _rule_gate to scale
        # the REVERSE threshold so a slow-grinder name keeps a low floor while
        # a fast-mover requires a proportionally bigger spike to flip. Stored
        # at most once per second per underlying to avoid duplicate inserts
        # when both CE and PE legs of the same underlying evaluate this tick.
        slope_avg_recent_bps = 0.0
        if Thresholds.SLOPE_ADAPTIVE:
            u_key = pos.get("underlying") or pos.get("symbol") or ""
            hist_map = self._state.setdefault("_u_slope_hist", {})
            hist = hist_map.setdefault(u_key, [])
            if not hist or (now_ts - hist[-1][0]) >= 0.9:
                hist.append((now_ts, abs(slope_5s_bps)))
            cutoff = now_ts - Thresholds.SLOPE_ADAPTIVE_LOOKBACK_SEC
            while hist and hist[0][0] < cutoff:
                hist.pop(0)
            if len(hist) >= Thresholds.SLOPE_ADAPTIVE_MIN_SAMPLES:
                slope_avg_recent_bps = sum(v for _, v in hist) / len(hist)

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

        # ── Fast-window features (May 4 corpus enrichment) ──────────────────
        # Today's flip-quant analysis showed the corpus has many lagging
        # signals (slopes 60s/300s) and few decision-time ones. Add 3 fast
        # microstructure prints so next week's training has more 5-15s class
        # signal to learn from. None of these gate live behaviour today —
        # they only ride along in the per-tick feature snapshot.
        # 1) u_ret_5s_bps — strict speed-up of ret_15s (3× faster window).
        u_5s = [p for p in u_series_full if now_ts - p[0] <= 5.5]
        u_ret_5s_bps = 0.0
        if len(u_5s) >= 2 and u_5s[0][1] > 0:
            u_ret_5s_bps = ((u_5s[-1][1] - u_5s[0][1]) / u_5s[0][1]) * 1e4
        # 2) u_ret_15s_z — z-score of current ret_15s vs rolling 60s history
        # of past ret_15s prints for THIS underlying. Strips out the
        # per-symbol vol regime so a +20bps tick on a sleepy name reads
        # bigger than a +20bps tick on a volatile one.
        u_ret_15s_z = 0.0
        try:
            u_key_z = pos.get("underlying") or pos.get("symbol") or ""
            r15_map = self._state.setdefault("_u_ret15_hist", {})
            r15h = r15_map.setdefault(u_key_z, [])
            if not r15h or (now_ts - r15h[-1][0]) >= 0.9:
                r15h.append((now_ts, ret_15s_bps))
            cutoff_z = now_ts - 60.0
            while r15h and r15h[0][0] < cutoff_z:
                r15h.pop(0)
            if len(r15h) >= 10:
                vals = [v for _, v in r15h]
                m_v = sum(vals) / len(vals)
                var_v = sum((v - m_v) ** 2 for v in vals) / len(vals)
                sd_v = var_v ** 0.5
                if sd_v > 1e-6:
                    u_ret_15s_z = (ret_15s_bps - m_v) / sd_v
        except Exception:
            pass
        # 3) depth_ratio_5s_avg — mean of (buys/sells) over last 5s of ticks
        # for this option leg. Today's corpus showed depth_ratio (single
        # tick) carries the highest IV among microstructure features (0.18);
        # a smoothed version should de-noise it without losing signal.
        depth_ratio_5s_avg = 0.0
        try:
            recent_pts = [p for p in pts if now_ts - p[0] <= 5.5]
            if recent_pts:
                ratios = []
                for _, _, b, s, *_ in recent_pts:
                    bb = float(b or 1)
                    ss = float(s or 1)
                    ratios.append(bb / ss)
                if ratios:
                    depth_ratio_5s_avg = sum(ratios) / len(ratios)
        except Exception:
            pass

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

        # ── 5-min range / regime detector (Apr 27) ──
        # Computes range, position-in-range, and ret-over-range over the
        # last 5 minutes of underlying tape. Drives oscillator vs breakout
        # regime classification used by relaxed PROFIT_PEAK / ADD_LOT paths.
        range_300s_bps = 0.0
        pos_in_range_300s = 0.5            # 0 = at low, 1 = at high
        ret_over_range_300s = 0.0
        regime_warm = len(u300) >= Thresholds.OSC_MIN_SAMPLES
        if regime_warm and u300[0][1] > 0:
            hi300 = max(p[1] for p in u300)
            lo300 = min(p[1] for p in u300)
            ref0 = u300[0][1]
            u_now = u300[-1][1]
            range_300s_bps = ((hi300 - lo300) / ref0) * 1e4
            if hi300 > lo300:
                pos_in_range_300s = (u_now - lo300) / (hi300 - lo300)
            net_300s = u_now - ref0
            if range_300s_bps > 1e-6:
                ret_over_range_300s = abs((net_300s / ref0) * 1e4) / range_300s_bps
        # Oscillator: range bounded, ping-ponging within it (low net-over-range)
        is_oscillator = bool(
            regime_warm
            and Thresholds.OSC_RANGE_MIN_BPS <= range_300s_bps <= Thresholds.OSC_RANGE_MAX_BPS
            and ret_over_range_300s < Thresholds.OSC_NET_OVER_RANGE_MAX
        )
        # Breakout up: at top of range + macro climbing
        is_breakout_up = bool(
            regime_warm
            and range_300s_bps >= Thresholds.BREAKOUT_RANGE_MIN_BPS
            and pos_in_range_300s >= Thresholds.BREAKOUT_POS_HIGH
            and slope_60s_bps >= Thresholds.BREAKOUT_MACRO_BPS_S
        )
        # Breakout down: at bottom of range + macro falling
        is_breakout_dn = bool(
            regime_warm
            and range_300s_bps >= Thresholds.BREAKOUT_RANGE_MIN_BPS
            and pos_in_range_300s <= Thresholds.BREAKOUT_POS_LOW
            and slope_60s_bps <= -Thresholds.BREAKOUT_MACRO_BPS_S
        )

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

        # --- Option premium staleness ---
        # How many seconds since the option LTP last *changed*. On illiquid
        # strikes the broker tape can hold the same premium for 30-60s while
        # the underlying drifts steadily. Slope/pnl gates miss this because
        # they all read the (stale) option premium. We use this as a tie-
        # breaker in the macro-drift path so a frozen quote can't keep the
        # rule gate inert when underlying evidence is otherwise sufficient.
        opt_ltp_stale_sec = 0.0
        try:
            for prev_ts, prev_ltp, *_ in reversed(pts[:-1]):
                if abs((prev_ltp or 0) - opt_ltp_now) > 1e-9:
                    opt_ltp_stale_sec = now_ts - prev_ts
                    break
            else:
                # All in-window points equal opt_ltp_now → window-length stale
                if pts:
                    opt_ltp_stale_sec = now_ts - pts[0][0]
        except Exception:
            pass

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
            "slope_avg_recent_bps": round(slope_avg_recent_bps, 3),
            "ret_15s_bps": round(ret_15s_bps, 2),
            "ret_30s_bps": round(ret_30s_bps, 2),
            # Fast-window enrichment (May 4) — passive, training-only
            "u_ret_5s_bps": round(u_ret_5s_bps, 2),
            "u_ret_15s_z": round(u_ret_15s_z, 3),
            "depth_ratio_5s_avg": round(depth_ratio_5s_avg, 3),
            "range_30s_bps": round(range_30s_bps, 2),
            "net_over_range": round(net_over_range, 3),
            "sustain_sec": int(sustain),
            "chop_flips": int(flips),
            "aligned": bool(aligned),
            "depth_ratio": round(depth_ratio, 2),
            "vol_surge": round(vol_surge, 2),
            "opt_ltp_stale_sec": round(opt_ltp_stale_sec, 1),
            # Regime detector outputs
            "range_300s_bps": round(range_300s_bps, 2),
            "pos_in_range_300s": round(pos_in_range_300s, 3),
            "ret_over_range_300s": round(ret_over_range_300s, 3),
            "is_oscillator": is_oscillator,
            "is_breakout_up": is_breakout_up,
            "is_breakout_dn": is_breakout_dn,
        }

    # ── Rule gate — SLOPE of UNDERLYING, not premium noise ──────────
    def _rule_gate(self, pos: dict, f: dict) -> Optional[Dict[str, Any]]:
        """SIMPLIFIED 2026-05-07 (user directive): one rule only.
        REVERSE if pnl_pct <= -2.3%. Never reverse a positive trade.
        All other rules removed (profit-peak, stale-loss, bleed-override,
        chop-bypass, macro-drift, counter-slope, breakout-add, momentum-add, etc.).

        2026-05-10 update: also refuse REVERSE if the position previously
        peaked ≥+3% (failed-winner protection). Counterfactual on
        May 5-8 OI_WATCHER trades: 5/15 of trades that peaked ≥+3%
        were exited by AP REVERSE at -2.3%, compounding the give-back
        instead of letting trailing SL / hard SL manage. Hard SL takes
        these gracefully; AP REVERSE flips into the wrong direction.
        """
        sym = pos.get("symbol", "") or ""
        pnl_pct_val = f.get("pnl_pct", 0.0) or 0.0

        # Per-symbol peak P&L ratchet, with 5-min staleness TTL.
        # On position close the symbol stops being tick-fed → entry goes
        # stale → next-time peak resets to current pnl (correct for re-entry).
        peak = pnl_pct_val
        if sym:
            now_ts = time.time()
            prev = self._peak_pnl_by_sym.get(sym)
            if prev is not None and now_ts - prev[0] <= 300:
                peak = max(prev[1], pnl_pct_val)
            self._peak_pnl_by_sym[sym] = (now_ts, peak)

        # 2026-05-14: sustain check — reset counter on any tick above threshold.
        # Must run BEFORE the rule body so transient bounces clear the count.
        if pnl_pct_val > -2.3:
            if sym in self._loss_sustain_count:
                self._loss_sustain_count.pop(sym, None)

        if pnl_pct_val <= -2.3:
            # Increment sustain counter — track CONSECUTIVE ticks below
            # threshold. Single-tick spikes (e.g. ADANIENT 2026-05-14:
            # one bid-side print at ₹90 on a position at ₹93) used to
            # fire the rule. Now needs 3 ticks (≈3 sec) of sustained
            # loss before reversing.
            _sustain_cnt = self._loss_sustain_count.get(sym, 0) + 1
            self._loss_sustain_count[sym] = _sustain_cnt
            if _sustain_cnt < 3:
                # Not yet sustained — log once per (sym, count) to avoid spam
                _sk = f"__sustain_log__{sym}_{_sustain_cnt}"
                if not self._state.get(_sk):
                    self._state[_sk] = True
                    try:
                        print(f"⏳ AP loss-sustain: {sym} pnl%={pnl_pct_val:+.2f} "
                              f"hit threshold ({_sustain_cnt}/3 consecutive) — waiting")
                    except Exception:
                        pass
                return None

            # MIN-HOLD ENFORCEMENT (2026-05-13 root-cause fix)
            # Thresholds.MIN_HOLD_FOR_REVERSE was a dead constant — defined but
            # never read after the May 7 simplification stripped the gates.
            # Restoring the documented behavior: REVERSE needs ≥60s of tape
            # so we don't flip on tick noise immediately after a fresh entry
            # (e.g. NAM-INDIA 8s flip on 2026-05-13 was the canonical bug).
            _hold_sec = int(f.get("hold_sec") or 0)
            if _hold_sec < Thresholds.MIN_HOLD_FOR_REVERSE:
                _log_key = f"__min_hold_skip__{sym}_{_hold_sec}"
                if not self._state.get(_log_key):
                    self._state[_log_key] = True
                    try:
                        print(f"⏸ AP min-hold skip: {sym} hold={_hold_sec}s "
                              f"< {Thresholds.MIN_HOLD_FOR_REVERSE}s "
                              f"(pnl={pnl_pct_val:+.2f}%) — too fresh to reverse")
                    except Exception:
                        pass
                return None

            # FAILED-WINNER PROTECTION v2 (2026-05-12 — dynamic threshold)
            # Only protect if peak crossed the trailing engagement point.
            # That's the "did this become a real winner" question. Brief
            # +3% pokes that didn't arm trailing get the -2.3% reverse as
            # the user designed.
            #
            # threshold = 0.18 * R_pct  (must match exit_manager.trailing_start_r)
            # If exit_manager.trailing_start_r changes, update this too.
            _sl_val = float(pos.get("stop_loss") or 0)
            _entry_val = float(pos.get("avg_price") or pos.get("entry_price") or 0)
            _fw_threshold = 999.0  # if SL/entry unreadable, protection effectively off
            if _sl_val > 0 and _entry_val > 0:
                _R_pct = abs(_entry_val - _sl_val) / _entry_val * 100.0
                _fw_threshold = 0.18 * _R_pct  # trailing engage point in %
            if peak >= _fw_threshold and _fw_threshold < 999.0:
                _log_key = f"__fw_skip_logged__{sym}_{int(peak)}"
                if not self._state.get(_log_key):
                    self._state[_log_key] = True
                    try:
                        print(f"🛡️ AP failed-winner skip: {sym} "
                              f"peak={peak:+.2f}% pnl={pnl_pct_val:+.2f}% "
                              f"(trailing engage={_fw_threshold:.2f}%) "
                              f"— letting trailing SL / hard SL handle exit")
                    except Exception:
                        pass
                return None
            return {
                "action": "REVERSE",
                "rule": f"simple-loss-rev(pnl%={pnl_pct_val:.2f}≤-2.3)",
                "boosters": ["simple_loss_rev"],
            }

        # ── R-based ADD_LOT (2026-05-11, user directive) ──
        # First +1 lot at 0.35R in profit. Second +1 lot at 0.70R.
        # Max 2 adds per position (already enforced by MAX_ADDS_PER_DAY=2 in
        # execute path). R = SL distance from entry, as %. Only fires when
        # in profit; never adds to a losing position. Existing AP safety
        # (drift-against veto, spread veto, LLM confirm) still applies.
        if pnl_pct_val > 0 and sym:
            sl_val = float(pos.get("stop_loss") or 0)
            entry_val = float(pos.get("avg_price") or pos.get("entry_price") or 0)
            if sl_val > 0 and entry_val > 0:
                R_pct = abs(entry_val - sl_val) / entry_val * 100.0
                if R_pct > 0:
                    adds_done = (self._state.get("counters", {})
                                 .get(sym, {}).get("adds", 0))
                    if adds_done == 0 and pnl_pct_val >= 0.30 * R_pct:
                        return {
                            "action": "ADD_LOT",
                            "rule": f"r-add-1st(0.30R={0.30*R_pct:.2f}%, "
                                    f"pnl={pnl_pct_val:.2f}%)",
                            "boosters": ["r_add_first"],
                        }
                    if adds_done == 1 and pnl_pct_val >= 0.70 * R_pct:
                        return {
                            "action": "ADD_LOT",
                            "rule": f"r-add-2nd(0.70R={0.70*R_pct:.2f}%, "
                                    f"pnl={pnl_pct_val:.2f}%)",
                            "boosters": ["r_add_second"],
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
            # De-dup cached-verdict spam: AP runs at 1Hz but the LLM verdict
            # is cached ~5s, so the same "[cached]" veto would otherwise log
            # 4-5 identical rows per cycle. Only emit a decision row when:
            #   - the verdict is FRESH (not [cached] prefix), OR
            #   - the reason changed since the last logged veto for this sym+action
            _reason = llm_verdict.get("reason", "") or ""
            _is_cached = _reason.startswith("[cached]")
            _vk = f"{sym}|{decision['action']}"
            _last = self._state.setdefault("_last_veto_log", {}).get(_vk)
            _should_log = (not _is_cached) or (_last != _reason)
            if _should_log:
                self._log_decision(sym, decision["action"], "VETOED_BY_LLM", features,
                                   _reason, None)
                self._state["_last_veto_log"][_vk] = _reason
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
            # Reset veto-dedup cache on successful execution so any future
            # veto cluster for this symbol logs cleanly.
            try:
                self._state.get("_last_veto_log", {}).pop(f"{sym}|{decision['action']}", None)
            except Exception:
                pass
            counters[{"REVERSE": "reverses", "ADD_LOT": "adds"}.get(decision["action"], "adds")] += 1
            self._state.setdefault("last_action_ts", {})[sym] = now_wall
            if decision["action"] == "REVERSE":
                self._state.setdefault("last_reverse_ts", {})[sym] = now_wall
                _u_for_ts = pos.get("underlying") or ""
                if _u_for_ts:
                    self._state.setdefault("last_u_flip_ts", {})[_u_for_ts] = now_wall
                # Whipsaw circuit-breaker: record flip for rolling-window count.
                self._state.setdefault("flip_history", {}).setdefault(sym, []).append(now_wall)
                # Track post-cap rescue-reverse usage per underlying so
                # MAX_POST_CAP_REVERSES can hard-stop runaway flipping on a
                # symbol that has already breached the realized-loss cap.
                _under_rev = pos.get("underlying") or ""
                if _under_rev and self._is_symbol_loss_capped(pos):
                    _bucket = self._state.setdefault("post_cap_reverses", {})
                    _bucket[_under_rev] = int(_bucket.get(_under_rev, 0)) + 1
            self._tick_mutated = True
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
                _u_for_ts = pos.get("underlying") or ""
                if _u_for_ts:
                    self._state.setdefault("last_u_flip_ts", {})[_u_for_ts] = now_wall
                self._state.setdefault("flip_history", {}).setdefault(sym, []).append(now_wall)
                _under_rev = pos.get("underlying") or ""
                if _under_rev and self._is_symbol_loss_capped(pos):
                    _bucket = self._state.setdefault("post_cap_reverses", {})
                    _bucket[_under_rev] = int(_bucket.get(_under_rev, 0)) + 1
            self._tick_mutated = True
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

        # LLM-outage fallback: if the LLM has been failing for the last N
        # consecutive calls, auto-approve HIGH-CONFIDENCE rules (bleed_override,
        # macro5min_counter, stale_loss, chop_bypass_rev, emergency_rerev_escape,
        # momentum_override) so AP isn't totally frozen during an OpenAI outage.
        # Marginal rules still go through LLM (and likely fail-veto). Recovers
        # automatically on first successful call below.
        if self._llm_consecutive_failures >= self._llm_outage_threshold:
            _boosters = set(decision.get("boosters") or [])
            if _boosters & self._llm_high_conf_boosters:
                return {
                    "go": True,
                    "reason": f"llm_outage_fallback({self._llm_consecutive_failures}f) — rules-only on high-conf rule",
                }

        system = (
            "You are an options trading co-pilot. A rules engine has ALREADY fired on "
            "a sustained SLOPE of the UNDERLYING stock (bps/second), after filtering "
            "chop and tail-reversal. Your only job is to catch OBVIOUS traps — default "
            "is GO. Most rules pass; veto only on concrete red flags listed below.\n\n"
            "SIGN CONVENTION (CRITICAL): The payload's `clarity` block exposes "
            "position-relative metrics. `*_with_position_*` fields are POSITIVE when "
            "the move HELPS the position and NEGATIVE when adverse — use these for "
            "sign reasoning. Do NOT infer sign from `u_drift_against_bps` (legacy "
            "name; positive there means against, easy to misread).\n\n"
            "UNDERLYING SLOPE FEATURES:\n"
            "  slope_5s_bps  : OLS slope over last 5s, bps/sec (short-term)\n"
            "  slope_3s_bps  : confirmation slope over last 3s (same sign required)\n"
            "  slope_60s_bps : 60s macro — REVERSE counter-trend rules require this to "
            "agree with the new direction (flip-sign or 1.5x dominance)\n"
            "  slope_300s_bps: 5-minute trend — 'macro5min' REVERSE on bleeders.\n"
            "  sustain_sec   : consecutive same-sign 1s deltas at the tail\n"
            "  chop_flips    : slope sign flips in last 30s (rule already filtered <30/<35)\n"
            "  trade_dir     : +1 if position wants UP, -1 if wants DOWN\n"
            "  aligned       : True ⇒ ADD_LOT family, False ⇒ REVERSE family\n\n"
            "REGIME FEATURES (5-minute window on underlying):\n"
            "  range_300s_bps     : (high − low) / start × 1e4 over 5 min\n"
            "  pos_in_range_300s  : 0 = at low, 1 = at high\n"
            "  ret_over_range_300s: |net move| / range; ~0 ping-pongs, ~1 trends\n"
            "  is_oscillator      : range bounded (20-80 bps) + ping-ponging — bank peaks earlier\n"
            "  is_breakout_up/dn  : at top/bottom 15% of range + macro confirming — pyramid\n\n"
            "RULE FAMILIES (action ↔ rule_triggered ↔ intent):\n"
            "  REVERSE — loss-cutting on a real macro flip:\n"
            "    bleed_override / macro5min_counter / stale_loss / chop_bypass_rev / "
            "emergency_rerev_escape — pnl is negative, macro turned against, exit before "
            "more damage. APPROVE unless trade is actively recovering.\n"
            "  REVERSE — profit-banking on a winner (DESIGN INTENT, not a trap):\n"
            "    profit_peak / osc_profit_peak — pnl is POSITIVE (≥+2-3%), counter-slope is "
            "sustained, macro 60s confirms the turn. We INTENTIONALLY exit a winner "
            "when momentum exhausts. ALMOST ALWAYS APPROVE — do NOT veto these for "
            "'recovering' just because pnl is green; that's exactly when we bank.\n"
            "  ADD_LOT — trend-continuation:\n"
            "    macro5min_add / momentum_override / macro_trend_add — established move, pyramid in. "
            "`macro_trend_add` fires when the 5-minute trend is with us and tape is clean (chop<30) "
            "even if 5s slope is small — APPROVE for slow-grinder winners; do not require |slope_5s|≥2.\n"
            "  ADD_LOT — breakout pyramiding:\n"
            "    breakout_add / chop_bypass_add — price at range extreme + macro confirms; "
            "early entry on regime change. APPROVE unless exhaustion-spike criteria below.\n\n"
            "VETO (go=false) ONLY FOR:\n"
            "  • ADD_LOT when |slope_5s| > 8 bps/sec AND pnl_pct > 80% — exhaustion spike, "
            "adding near the top is bad. (Does NOT apply to breakout_add at modest pnl.)\n"
            "  • ADD_LOT on a deep-OTM lottery (pnl_pct very negative + entry price tiny).\n"
            "  • REVERSE-loss-cut when pnl_pct > +3% and slope is only marginally against. "
            "(Does NOT apply to profit_peak / osc_profit_peak — those are MEANT to fire on "
            "winners.)\n"
            "  • Any direction wholly inconsistent with sign of slope_5s and slope_3s.\n\n"
            "NOT red flags: depth_ratio near 1.0, vol_surge < 1.3 — NFO depth is thin by "
            "nature. Slope magnitude + sustain + macro agreement is what matters.\n\n"
            'Respond with ONLY compact JSON: {"go": true|false, "reason": "<=20 words"}.'
        )
        user = json.dumps({
            "proposed_action": decision["action"],
            "rule_triggered": decision.get("rule"),
            "boosters": decision.get("boosters") or [],
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
            # === LLM-clarity injections (Apr 29) =====================
            # Add unambiguous, position-relative metric names so the LLM
            # cannot confuse "u_drift_against_bps" sign convention. The
            # `_with_position_bps` field is positive when the underlying
            # has moved IN OUR FAVOUR since entry, negative when adverse.
            "clarity": {
                "u_drift_with_position_bps": -float(features.get("u_drift_against_bps", 0.0) or 0.0),
                "slope_5s_with_position_bps_s": float(features.get("slope_5s_bps", 0.0) or 0.0) * float(features.get("trade_dir", 0) or 0),
                "slope_60s_with_position_bps_s": float(features.get("slope_60s_bps", 0.0) or 0.0) * float(features.get("trade_dir", 0) or 0),
                "slope_300s_with_position_bps_s": float(features.get("slope_300s_bps", 0.0) or 0.0) * float(features.get("trade_dir", 0) or 0),
                "note": "all *_with_position_* fields: POSITIVE = favourable to position, NEGATIVE = adverse. Ignore u_drift_against_bps for sign reasoning.",
            },
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
            # Reset outage counter on first successful call.
            self._llm_consecutive_failures = 0
            return {
                "go": bool(verdict.get("go")),
                "reason": str(verdict.get("reason", ""))[:200],
            }
        except Exception as e:
            # Track consecutive failures for outage-fallback path.
            self._llm_consecutive_failures += 1
            # On LLM failure, be conservative — VETO (don't execute on broken link)
            # Mark with _llm_error so the caller skips caching this verdict.
            return {"go": False, "reason": f"llm_error: {str(e)[:80]}", "_llm_error": True}

    # ── Strike-spacing heuristic (mirrors dashboard.py news endpoint) ──
    @staticmethod
    def _strike_gap_for(underlying_name: str, u_price: float) -> float:
        nm = (underlying_name or "").upper().replace("NSE:", "")
        if nm in ("NIFTY", "NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            return 50.0
        if u_price > 5000:
            return 100.0
        if u_price > 2500:
            return 50.0
        if u_price > 1000:
            return 20.0
        if u_price > 500:
            return 10.0
        return 5.0

    # ── Strike-migration check (Apr 30, log-only Phase 1) ────────────
    def _check_strike_migrate(self, pos: dict, f: dict, now_wall: float) -> None:
        """Diagnostic: emit a migration plan when held strike has drifted
        too far OTM and trade is aligned + green. Pure logging — no execute."""
        sym = pos.get("symbol") or ""
        if not sym:
            return

        # Per-name cooldown + day-cap
        last_mig_ts = self._state.setdefault("last_migrate_ts", {}).get(sym, 0.0)
        if now_wall - last_mig_ts < Thresholds.STRIKE_MIGRATE_COOLDOWN_SEC:
            return
        mig_counters = self._state.setdefault("counters", {}).setdefault(
            sym, {"reverses": 0, "adds": 0, "migrations": 0}
        )
        if mig_counters.get("migrations", 0) >= Thresholds.STRIKE_MIGRATE_MAX_PER_DAY:
            return

        # Hard preconditions
        aligned = bool(f.get("aligned"))
        if not aligned:
            return
        pnl_pct = float(f.get("pnl_pct") or 0.0)
        if pnl_pct < Thresholds.STRIKE_MIGRATE_MIN_PNL_PCT:
            return
        hold_sec = int(f.get("hold_sec") or 0)
        if hold_sec < Thresholds.STRIKE_MIGRATE_MIN_HOLD_SEC:
            return
        # Underlying must have drifted FAVORABLY (negative against = with us)
        u_drift_against = float(f.get("u_drift_against_bps") or 0.0)
        u_drift_with = -u_drift_against
        if u_drift_with < Thresholds.STRIKE_MIGRATE_MIN_DRIFT_BPS:
            return
        # Slope still pushing
        s5 = float(f.get("slope_5s_bps") or 0.0)
        if abs(s5) < Thresholds.STRIKE_MIGRATE_MIN_SLOPE_BPS_S:
            return

        # Current strike + new ATM
        cur_strike = float(pos.get("strike") or 0)
        u_ltp = float(f.get("u_ltp") or 0)
        if cur_strike <= 0 or u_ltp <= 0:
            return
        opt_type = (pos.get("option_type") or "").upper()
        if opt_type not in ("CE", "PE"):
            return
        underlying = pos.get("underlying") or ""
        gap = self._strike_gap_for(underlying, u_ltp)
        new_atm = round(u_ltp / gap) * gap
        # Distance in strike-units that current strike is OTM relative to new ATM
        if opt_type == "CE":
            otm_units = (cur_strike - new_atm) / gap   # positive = OTM
        else:  # PE
            otm_units = (new_atm - cur_strike) / gap

        if otm_units < Thresholds.STRIKE_MIGRATE_MIN_STRIKES_OTM:
            return  # not far enough OTM to bother re-centering

        # Spread veto piggyback
        spread_pct = float(f.get("spread_pct") or 0.0)
        if spread_pct > Thresholds.SPREAD_VETO_PCT:
            return  # illiquid current strike, skip migration too

        # Throttle: log at most once per cooldown window
        log_key = f"__migrate_plan_{sym}__"
        last_lk = self._windows.get(log_key, 0)
        if now_wall - last_lk < 60:
            return
        self._windows[log_key] = now_wall

        print(
            f"🎯 AutoPilot STRIKE_MIGRATE_PLAN [DRY-RUN] {sym} — "
            f"current_strike={cur_strike:.0f} {opt_type}, new_ATM={new_atm:.0f} "
            f"({otm_units:.0f} strikes OTM, gap={gap:.0f}) | "
            f"pnl%={pnl_pct:+.2f} hold={hold_sec}s u_drift_with={u_drift_with:+.1f}bps "
            f"s5={s5:+.2f} aligned=True spread={spread_pct:.2f}% | "
            f"plan: EXIT {cur_strike:.0f}{opt_type} → BUY {new_atm:.0f}{opt_type} "
            f"(re-center on max-gamma; lock +{pnl_pct:.1f}% realized)",
            flush=True,
        )

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
                "source": "AUTOPILOT",  # 2026-05-13: distinguish AP from user clicks
            }
            return self._post("/api/reverse_trade", payload)

        if action == "ADD_LOT":
            return self._post("/api/add_lot", {"symbol": sym})

        if action == "BOOK_PROFIT":
            # Stale-winner profit lock — exit at market via dashboard.
            return self._post("/api/exit_position", {"symbol": sym})

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
        per-symbol realized-loss cap today. Used by action-aware gating:
        ADD_LOT is always blocked at cap, REVERSE is allowed up to
        MAX_POST_CAP_REVERSES (the escape path is preserved)."""
        under = pos.get('underlying') or ''
        if not under:
            return False
        realized = self._realized_by_underlying.get(under, 0.0)
        return realized <= Thresholds.PER_SYMBOL_REALIZED_LOSS_CAP

    def _post_cap_reverses_used(self, under: str) -> int:
        """Count of REVERSEs AP has executed on this underlying since the
        cap was breached today. Used to enforce MAX_POST_CAP_REVERSES."""
        return int(self._state.get("post_cap_reverses", {}).get(under, 0))

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

    def _get_regime_tags(self) -> Dict[str, Any]:
        """Return regime tags (VIX bucket, Nifty trend bucket, day-of-week,
        policy version) for the current moment. Cached for 60s — these
        change slowly and the bandit logger fires at 1Hz per position.

        Tags use ZERO API calls: VIX & Nifty come from the shared ticker
        cache that's already subscribed. If unavailable, returns 'unknown'
        for that field rather than failing.
        """
        now = time.time()
        if self._regime_cache and (now - self._regime_cache_ts) < 60.0:
            return self._regime_cache
        ticker = getattr(self.tools, "ticker", None)
        # India VIX
        vix_val = 0.0
        vix_bucket = "unknown"
        if ticker is not None:
            try:
                _v = ticker.get_ltp("NSE:INDIA VIX")
                if _v:
                    vix_val = float(_v)
            except Exception:
                pass
        if vix_val > 0:
            if vix_val < 14:
                vix_bucket = "low"
            elif vix_val < 18:
                vix_bucket = "mid"
            elif vix_val < 22:
                vix_bucket = "elevated"
            else:
                vix_bucket = "high"
        # Nifty trend (pct change from day open)
        nifty_ltp = 0.0
        nifty_pct = 0.0
        nifty_trend = "unknown"
        today = datetime.now().strftime("%Y-%m-%d")
        if ticker is not None:
            try:
                _n = ticker.get_ltp("NSE:NIFTY 50")
                if _n:
                    nifty_ltp = float(_n)
            except Exception:
                pass
        if nifty_ltp > 0:
            if today not in self._nifty_open_by_date:
                self._nifty_open_by_date[today] = nifty_ltp
                # prune old dates (keep last 5)
                if len(self._nifty_open_by_date) > 5:
                    for _k in sorted(self._nifty_open_by_date.keys())[:-5]:
                        self._nifty_open_by_date.pop(_k, None)
            n_open = self._nifty_open_by_date.get(today, 0.0)
            if n_open > 0:
                nifty_pct = (nifty_ltp - n_open) / n_open * 100.0
                if nifty_pct < -0.3:
                    nifty_trend = "down"
                elif nifty_pct > 0.3:
                    nifty_trend = "up"
                else:
                    nifty_trend = "flat"
        # Day of week (Mon..Sun)
        dow = datetime.now().strftime("%a")
        self._regime_cache = {
            "vix": round(vix_val, 2) if vix_val else None,
            "vix_b": vix_bucket,
            "nifty_pct": round(nifty_pct, 3) if nifty_ltp else None,
            "nifty_t": nifty_trend,
            "dow": dow,
            "pv": POLICY_VERSION,
        }
        self._regime_cache_ts = now
        return self._regime_cache

    def _log_feature_snapshot(self, symbol: str, pos: dict,
                              features: dict,
                              decision: Optional[Dict[str, Any]]) -> None:
        """Append per-tick (ts, sym, features, decision) to a daily jsonl
        for the offline MC backtest harness. Best-effort; never raises.

        Disabled by env AP_FEATURE_SNAPSHOT=0. ~50-150 MB/day depending
        on open-position count. File rotates on date change.
        """
        if not FEATURE_SNAPSHOT_ENABLED:
            return
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with self._fs_lock:
                if self._fs_handle is None or self._fs_date != today:
                    if self._fs_handle is not None:
                        try:
                            self._fs_handle.close()
                        except Exception:
                            pass
                    os.makedirs(FEATURE_SNAPSHOT_DIR, exist_ok=True)
                    path = os.path.join(
                        FEATURE_SNAPSHOT_DIR,
                        f"bandit_corpus_{today}.jsonl",
                    )
                    self._fs_handle = open(path, "a", buffering=1)
                    self._fs_date = today

                def _safe(v):
                    if v is None or isinstance(v, (int, float, str, bool)):
                        return v
                    try:
                        return float(v)
                    except Exception:
                        return str(v)[:80]

                entry = {
                    "ts": time.time(),
                    "sym": symbol,
                    "u": pos.get("underlying"),
                    "side": pos.get("transaction_type") or pos.get("side"),
                    "qty": pos.get("quantity"),
                    "entry_px": pos.get("entry_price") or pos.get("avg_price"),
                    "f": {k: _safe(v) for k, v in features.items()},
                    "r": self._get_regime_tags(),
                }
                if decision:
                    entry["d"] = {
                        "a": decision.get("action"),
                        "r": (decision.get("rule") or "")[:80],
                    }
                self._fs_handle.write(
                    json.dumps(entry, separators=(",", ":")) + "\n"
                )
        except Exception:
            # Never break the main loop on snapshot-writer failure.
            pass

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
            "last_u_flip_ts": {},
            "last_park_ts": {},
            "post_cap_reverses": {},
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
                    data["last_u_flip_ts"] = {}
                    data["last_park_ts"] = {}
                    data["post_cap_reverses"] = {}
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
