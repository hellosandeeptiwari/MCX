#!/usr/bin/env python3
"""
Narrow & combine the two best GMM filters:
  Filter A: Score 52-70 + UPDR 0.10-0.16 + DOWNDR >= 0.19
  Filter B: Score 52-70 + DR_spread >= 0.10

Drill into finer sub-ranges, add GMM direction/alignment, max_favorable
distribution, and per-symbol breakdown to find the tightest profitable zone.
"""
import json
from collections import Counter

DATA_FILE = "/home/ubuntu/titan/logs/gmm_calibration_data.jsonl"

data = []
with open(DATA_FILE) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

completed = [x for x in data if x.get("completed")]


def stats(bucket):
    if not bucket:
        return None
    w = sum(1 for x in bucket if "PROFIT" in x.get("outcome", ""))
    l = sum(1 for x in bucket if "LOSS" in x.get("outcome", ""))
    f = sum(1 for x in bucket if x.get("outcome") == "FLAT")
    mf = sum(x.get("max_favorable", 0) for x in bucket) / len(bucket)
    ma = sum(x.get("max_adverse", 0) for x in bucket) / len(bucket)
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    return {"w": w, "l": l, "f": f, "n": len(bucket), "wr": wr,
            "avg_fav": mf, "avg_adv": ma, "edge": mf + ma}


def pr(label, s, baseline_wr=None):
    if not s:
        print(f"  {label:<50} -- no data --")
        return
    uplift = f"{s['wr']/baseline_wr:.1f}x" if baseline_wr and baseline_wr > 0 else ""
    print(f"  {label:<50} {s['w']:>3}W {s['l']:>3}L {s['f']:>2}F  n={s['n']:<4} "
          f"WR={s['wr']:>5.1f}%  Fav={s['avg_fav']:>+6.2f}%  Adv={s['avg_adv']:>+7.2f}%  "
          f"Edge={s['edge']:>+6.2f}%  {uplift}")


# ============================================================
# BASELINES
# ============================================================
print("=" * 110)
print("BASELINES")
print("=" * 110)
all_s = stats(completed)
base52 = stats([x for x in completed if 52 <= x.get("smart_score", 0) < 70])
pr("All completed tracks", all_s)
pr("Score 52-70 (no DR filter)", base52)
base_wr = base52["wr"] if base52 else 0

# ============================================================
# FILTER A vs FILTER B vs INTERSECTION
# ============================================================
print()
print("=" * 110)
print("FILTER A vs B vs A∩B")
print("=" * 110)

filter_a = [x for x in completed
            if 52 <= x.get("smart_score", 0) < 70
            and 0.10 <= x.get("updr_score", 0) < 0.16
            and x.get("downdr_score", 0) >= 0.19]

filter_b = [x for x in completed
            if 52 <= x.get("smart_score", 0) < 70
            and (x.get("downdr_score", 0) - x.get("updr_score", 0)) >= 0.10]

# A∩B: both conditions simultaneously
filter_ab = [x for x in completed
             if 52 <= x.get("smart_score", 0) < 70
             and 0.10 <= x.get("updr_score", 0) < 0.16
             and x.get("downdr_score", 0) >= 0.19
             and (x.get("downdr_score", 0) - x.get("updr_score", 0)) >= 0.10]

# B minus A (in B but not A -- what does B add beyond A?)
filter_b_only = [x for x in filter_b if x not in filter_a]

pr("Filter A: Score 52-70 + UPDR 0.10-0.16 + DOWNDR>=0.19", stats(filter_a), base_wr)
pr("Filter B: Score 52-70 + DR_spread>=0.10", stats(filter_b), base_wr)
pr("A ∩ B (both conditions)", stats(filter_ab), base_wr)
pr("B \\ A (in B, not A)", stats(filter_b_only), base_wr)

# ============================================================
# NARROW SCORE: 52-56, 56-60, 60-65, 65-70 within Filter A
# ============================================================
print()
print("=" * 110)
print("NARROW SCORE BINS within Filter A (UPDR 0.10-0.16 + DOWNDR>=0.19)")
print("=" * 110)
for lo, hi, lbl in [(52, 56, "52-56"), (56, 60, "56-60"), (60, 65, "60-65"), (65, 70, "65-70")]:
    b = [x for x in filter_a if lo <= x.get("smart_score", 0) < hi]
    pr(f"Score {lbl}", stats(b), base_wr)

# ============================================================
# NARROW UPDR within Filter A zone
# ============================================================
print()
print("=" * 110)
print("NARROW UPDR BINS (Score 52-70, DOWNDR>=0.19)")
print("=" * 110)
for lo, hi, lbl in [(0.08, 0.10, "0.08-0.10"), (0.10, 0.12, "0.10-0.12"),
                     (0.12, 0.14, "0.12-0.14"), (0.14, 0.16, "0.14-0.16"),
                     (0.16, 0.18, "0.16-0.18"), (0.18, 0.22, "0.18-0.22")]:
    b = [x for x in completed
         if 52 <= x.get("smart_score", 0) < 70
         and lo <= x.get("updr_score", 0) < hi
         and x.get("downdr_score", 0) >= 0.19]
    pr(f"UPDR {lbl}", stats(b), base_wr)

# ============================================================
# NARROW DOWNDR within Filter A zone
# ============================================================
print()
print("=" * 110)
print("NARROW DOWNDR BINS (Score 52-70, UPDR 0.10-0.16)")
print("=" * 110)
for lo, hi, lbl in [(0.15, 0.19, "0.15-0.19"), (0.19, 0.22, "0.19-0.22"),
                     (0.22, 0.25, "0.22-0.25"), (0.25, 0.30, "0.25-0.30"),
                     (0.30, 0.40, "0.30-0.40"), (0.40, 1.0, "0.40+")]:
    b = [x for x in completed
         if 52 <= x.get("smart_score", 0) < 70
         and 0.10 <= x.get("updr_score", 0) < 0.16
         and lo <= x.get("downdr_score", 0) < hi]
    pr(f"DOWNDR {lbl}", stats(b), base_wr)

# ============================================================
# NARROW DR_SPREAD within Score 52-70
# ============================================================
print()
print("=" * 110)
print("NARROW DR_SPREAD BINS (Score 52-70)")
print("=" * 110)
for lo, hi, lbl in [(0.05, 0.08, "0.05-0.08"), (0.08, 0.10, "0.08-0.10"),
                     (0.10, 0.12, "0.10-0.12"), (0.12, 0.15, "0.12-0.15"),
                     (0.15, 0.20, "0.15-0.20"), (0.20, 0.30, "0.20-0.30"),
                     (0.30, 1.0, "0.30+")]:
    b = [x for x in completed
         if 52 <= x.get("smart_score", 0) < 70
         and lo <= (x.get("downdr_score", 0) - x.get("updr_score", 0)) < hi]
    pr(f"DR_spread {lbl}", stats(b), base_wr)

# ============================================================
# GMM DIRECTION OVERLAY on Filter A
# ============================================================
print()
print("=" * 110)
print("GMM DIRECTION OVERLAY on Filter A")
print("=" * 110)
gmm_yes = [x for x in filter_a if x.get("gmm_confirms_dir") is True]
gmm_no = [x for x in filter_a if x.get("gmm_confirms_dir") is False]
pr("Filter A + GMM confirms=True", stats(gmm_yes), base_wr)
pr("Filter A + GMM confirms=False", stats(gmm_no), base_wr)

# Scorer-GMM alignment overlay
aligned = [x for x in filter_a if x.get("scorer_direction") and x.get("direction_hint") and
           ((x["scorer_direction"] == "BUY" and "BULL" in x["direction_hint"]) or
            (x["scorer_direction"] == "SELL" and "BEAR" in x["direction_hint"]))]
misaligned = [x for x in filter_a if x.get("scorer_direction") and x.get("direction_hint") and x not in aligned]
pr("Filter A + Scorer-GMM aligned", stats(aligned), base_wr)
pr("Filter A + Scorer-GMM misaligned", stats(misaligned), base_wr)

# ============================================================
# GMM DIRECTION OVERLAY on Filter B
# ============================================================
print()
print("=" * 110)
print("GMM DIRECTION OVERLAY on Filter B (DR_spread>=0.10)")
print("=" * 110)
gmm_yes_b = [x for x in filter_b if x.get("gmm_confirms_dir") is True]
gmm_no_b = [x for x in filter_b if x.get("gmm_confirms_dir") is False]
pr("Filter B + GMM confirms=True", stats(gmm_yes_b), base_wr)
pr("Filter B + GMM confirms=False", stats(gmm_no_b), base_wr)

aligned_b = [x for x in filter_b if x.get("scorer_direction") and x.get("direction_hint") and
             ((x["scorer_direction"] == "BUY" and "BULL" in x["direction_hint"]) or
              (x["scorer_direction"] == "SELL" and "BEAR" in x["direction_hint"]))]
misaligned_b = [x for x in filter_b if x.get("scorer_direction") and x.get("direction_hint") and x not in aligned_b]
pr("Filter B + Scorer-GMM aligned", stats(aligned_b), base_wr)
pr("Filter B + Scorer-GMM misaligned", stats(misaligned_b), base_wr)

# ============================================================
# DIRECTION (CE vs PE) within Filter A
# ============================================================
print()
print("=" * 110)
print("DIRECTION (CE vs PE) within Filter A")
print("=" * 110)
ce_a = [x for x in filter_a if x.get("scorer_direction") == "BUY"]
pe_a = [x for x in filter_a if x.get("scorer_direction") == "SELL"]
pr("Filter A + CE (BUY)", stats(ce_a), base_wr)
pr("Filter A + PE (SELL)", stats(pe_a), base_wr)

# ============================================================
# MAX FAVORABLE DISTRIBUTION within Filter A (Winners vs Losers)
# ============================================================
print()
print("=" * 110)
print("MAX FAVORABLE DISTRIBUTION -- Filter A Winners vs Losers")
print("=" * 110)
a_wins = [x for x in filter_a if "PROFIT" in x.get("outcome", "")]
a_losses = [x for x in filter_a if "LOSS" in x.get("outcome", "")]
if a_wins:
    favs = sorted([x.get("max_favorable", 0) for x in a_wins])
    print(f"  Winners (n={len(a_wins)}): min={favs[0]:.2f}%  med={favs[len(favs)//2]:.2f}%  max={favs[-1]:.2f}%  avg={sum(favs)/len(favs):.2f}%")
if a_losses:
    favs = sorted([x.get("max_favorable", 0) for x in a_losses])
    advs = sorted([x.get("max_adverse", 0) for x in a_losses])
    print(f"  Losers  (n={len(a_losses)}): max_fav min={favs[0]:.2f}% med={favs[len(favs)//2]:.2f}% max={favs[-1]:.2f}%")
    print(f"                         max_adv min={advs[0]:.2f}% med={advs[len(advs)//2]:.2f}% max={advs[-1]:.2f}%")

# ============================================================
# PER-SYMBOL breakdown within Filter A
# ============================================================
print()
print("=" * 110)
print("PER-SYMBOL BREAKDOWN -- Filter A")
print("=" * 110)
syms = {}
for x in filter_a:
    sym = x.get("symbol", "?")
    if sym not in syms:
        syms[sym] = []
    syms[sym].append(x)

print(f"  {'Symbol':<20} {'W':>3} {'L':>3} {'F':>2} {'N':>4} {'WR':>6} {'Edge':>7}")
print(f"  {'-'*52}")
for sym in sorted(syms, key=lambda s: -len(syms[s])):
    s = stats(syms[sym])
    if s:
        print(f"  {sym:<20} {s['w']:>3} {s['l']:>3} {s['f']:>2} {s['n']:>4} {s['wr']:>5.1f}% {s['edge']:>+6.2f}%")

# ============================================================
# TIME-OF-DAY within Filter A
# ============================================================
print()
print("=" * 110)
print("TIME-OF-DAY BREAKDOWN -- Filter A")
print("=" * 110)
for hr_lo, hr_hi, lbl in [(9, 10, "09:00-10:00"), (10, 11, "10:00-11:00"),
                            (11, 12, "11:00-12:00"), (12, 14, "12:00-14:00"),
                            (14, 16, "14:00-15:30")]:
    b = [x for x in filter_a
         if x.get("entry_time") and hr_lo <= int(x["entry_time"].split("T")[-1].split(":")[0]) < hr_hi]
    pr(f"{lbl}", stats(b), base_wr)

# ============================================================
# COMBINED TIGHTEST FILTER SCAN (Score x UPDR x DOWNDR x DR_spread)
# ============================================================
print()
print("=" * 110)
print("EXHAUSTIVE NARROW SCAN -- Score x UPDR x DOWNDR x DR_spread (min 5 samples)")
print("=" * 110)
print(f"  {'Score':<8} {'UPDR':<12} {'DOWNDR':<12} {'DRspread':<12} "
      f"{'W':>3} {'L':>3} {'F':>2} {'N':>4} {'WR':>6} {'Edge':>7}")
print(f"  {'-'*75}")

combos = []
score_ranges = [(52, 58, "52-58"), (58, 64, "58-64"), (64, 70, "64-70")]
updr_ranges = [(0.08, 0.12, "0.08-0.12"), (0.12, 0.16, "0.12-0.16"), (0.10, 0.16, "0.10-0.16")]
downdr_ranges = [(0.19, 0.25, "0.19-0.25"), (0.25, 0.35, "0.25-0.35"), (0.19, 1.0, "0.19+")]
spread_ranges = [(0.05, 0.10, "sp05-10"), (0.10, 0.15, "sp10-15"), (0.15, 1.0, "sp15+"), (0.10, 1.0, "sp10+")]

for slo, shi, slbl in score_ranges:
    for ulo, uhi, ulbl in updr_ranges:
        for dlo, dhi, dlbl in downdr_ranges:
            for splo, sphi, splbl in spread_ranges:
                b = [x for x in completed
                     if slo <= x.get("smart_score", 0) < shi
                     and ulo <= x.get("updr_score", 0) < uhi
                     and dlo <= x.get("downdr_score", 0) < dhi
                     and splo <= (x.get("downdr_score", 0) - x.get("updr_score", 0)) < sphi]
                s = stats(b)
                if s and s["n"] >= 5:
                    combos.append((s["wr"], s["edge"], slbl, ulbl, dlbl, splbl, s))

combos.sort(key=lambda c: (-c[0], -c[1]))
for i, (_, _, slbl, ulbl, dlbl, splbl, s) in enumerate(combos[:25]):
    marker = " <<<" if i < 3 else ""
    print(f"  {slbl:<8} {ulbl:<12} {dlbl:<12} {splbl:<12} "
          f"{s['w']:>3} {s['l']:>3} {s['f']:>2} {s['n']:>4} {s['wr']:>5.1f}% {s['edge']:>+6.2f}%{marker}")

# ============================================================
# FINAL RECOMMENDATION
# ============================================================
print()
print("=" * 110)
print("RECOMMENDATION")
print("=" * 110)
if combos:
    best = combos[0]
    print(f"  Tightest profitable zone: Score {best[2]} + UPDR {best[3]} + DOWNDR {best[4]} + {best[5]}")
    print(f"    {best[6]['w']}W / {best[6]['l']}L / {best[6]['f']}F  n={best[6]['n']}  WR={best[6]['wr']:.1f}%  Edge={best[6]['edge']:+.2f}%")
    print(f"    vs baseline WR={base_wr:.1f}% => {best[6]['wr']/base_wr:.1f}x uplift" if base_wr > 0 else "")
