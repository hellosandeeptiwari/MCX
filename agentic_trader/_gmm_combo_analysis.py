#!/usr/bin/env python3
"""Analyze GMM calibration: UPDR x DOWNDR x smart_score cross-tabulation."""
import json

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
    return w, l, f, len(bucket), wr, mf, ma, mf + ma

# ========== SCORE x UPDR ==========
print("=" * 75)
print("SMART_SCORE x UPDR -- Win Rate Heatmap")
print("=" * 75)
score_bins = [(0, 40, "<40"), (40, 52, "40-52"), (52, 60, "52-60"), (60, 70, "60-70"), (70, 200, "70+")]
updr_bins = [(0, 0.10, "<0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.20, "0.15-0.20"), (0.20, 1.0, "0.20+")]

print(f"{'':>12}", end="")
for _, _, ulbl in updr_bins:
    print(f" {ulbl:>12}", end="")
print()
print("-" * 65)

for slo, shi, slbl in score_bins:
    print(f"{slbl:>12}", end="")
    for ulo, uhi, ulbl in updr_bins:
        b = [x for x in completed if slo <= x.get("smart_score", 0) < shi and ulo <= x.get("updr_score", 0) < uhi]
        s = stats(b)
        if s and s[3] >= 3:
            w, l, f, n, wr, mf, ma, edge = s
            marker = "*" if wr >= 25 else " "
            print(f" {wr:>4.0f}%/{n:<3}{marker}  ", end="")
        else:
            print(f" {'---':>12}", end="")
    print()

# ========== SCORE x DOWNDR ==========
print()
print("=" * 75)
print("SMART_SCORE x DOWNDR -- Win Rate Heatmap")
print("=" * 75)
downdr_bins = [(0, 0.12, "<0.12"), (0.12, 0.17, "0.12-0.17"), (0.17, 0.22, "0.17-0.22"), (0.22, 0.30, "0.22-0.30"), (0.30, 1.0, "0.30+")]

print(f"{'':>12}", end="")
for _, _, dlbl in downdr_bins:
    print(f" {dlbl:>12}", end="")
print()
print("-" * 78)

for slo, shi, slbl in score_bins:
    print(f"{slbl:>12}", end="")
    for dlo, dhi, dlbl in downdr_bins:
        b = [x for x in completed if slo <= x.get("smart_score", 0) < shi and dlo <= x.get("downdr_score", 0) < dhi]
        s = stats(b)
        if s and s[3] >= 3:
            w, l, f, n, wr, mf, ma, edge = s
            marker = "*" if wr >= 25 else " "
            print(f" {wr:>4.0f}%/{n:<3}{marker}  ", end="")
        else:
            print(f" {'---':>12}", end="")
    print()

# ========== SCORE x DR SPREAD ==========
print()
print("=" * 75)
print("SMART_SCORE x DR_SPREAD (downdr - updr) -- Win Rate Heatmap")
print("=" * 75)
spread_bins = [(-1.0, 0.0, "UR>DR"), (0.0, 0.05, "DR+0-5"), (0.05, 0.10, "DR+5-10"), (0.10, 0.20, "DR+10-20"), (0.20, 1.0, "DR+20+")]

print(f"{'':>12}", end="")
for _, _, splbl in spread_bins:
    print(f" {splbl:>12}", end="")
print()
print("-" * 78)

for slo, shi, slbl in score_bins:
    print(f"{slbl:>12}", end="")
    for splo, sphi, splbl in spread_bins:
        b = [x for x in completed 
             if slo <= x.get("smart_score", 0) < shi
             and splo <= (x.get("downdr_score", 0) - x.get("updr_score", 0)) < sphi]
        s = stats(b)
        if s and s[3] >= 3:
            w, l, f, n, wr, mf, ma, edge = s
            marker = "*" if wr >= 25 else " "
            print(f" {wr:>4.0f}%/{n:<3}{marker}  ", end="")
        else:
            print(f" {'---':>12}", end="")
    print()

# ========== DETAILED TOP COMBOS ==========
print()
print("=" * 80)
print("TOP 20 COMBOS: SCORE x UPDR x DOWNDR -- sorted by WR (min 8 samples)")
print("=" * 80)
print(f"{'Score':<8} {'UPDR':<11} {'DOWNDR':<11} {'W':>3} {'L':>3} {'F':>3} {'Tot':>4} {'WR':>6} {'AvgFav':>8} {'AvgAdv':>8} {'Edge':>7}")
print("-" * 80)

combos = []
score_ranges = [(0, 40, "<40"), (40, 52, "40-52"), (52, 60, "52-60"), (60, 70, "60-70"), (70, 200, "70+")]
updr_ranges = [(0, 0.10, "<0.10"), (0.10, 0.16, "0.10-0.16"), (0.16, 1.0, "0.16+")]
downdr_ranges = [(0, 0.15, "<0.15"), (0.15, 0.22, "0.15-0.22"), (0.22, 1.0, "0.22+")]

for slo, shi, slbl in score_ranges:
    for ulo, uhi, ulbl in updr_ranges:
        for dlo, dhi, dlbl in downdr_ranges:
            b = [x for x in completed
                 if slo <= x.get("smart_score", 0) < shi
                 and ulo <= x.get("updr_score", 0) < uhi
                 and dlo <= x.get("downdr_score", 0) < dhi]
            s = stats(b)
            if s and s[3] >= 8:
                combos.append((s[4], slbl, ulbl, dlbl, *s))

combos.sort(reverse=True)
for i, (_, slbl, ulbl, dlbl, w, l, f, n, wr, mf, ma, edge) in enumerate(combos[:20]):
    marker = " <== BEST" if i < 3 else ""
    print(f"{slbl:<8} {ulbl:<11} {dlbl:<11} {w:>3} {l:>3} {f:>3} {n:>4} {wr:>5.1f}% {mf:>7.2f}% {ma:>7.2f}% {edge:>6.2f}%{marker}")

# ========== ACTIONABLE SUMMARY ==========
print()
print("=" * 80)
print("ACTIONABLE SUMMARY -- Best entry filters")
print("=" * 80)

# Best overall
best_all = [x for x in completed
            if 52 <= x.get("smart_score", 0) < 70
            and 0.10 <= x.get("updr_score", 0) < 0.16
            and x.get("downdr_score", 0) >= 0.19]
s = stats(best_all)
if s:
    print(f"\n  BEST COMBO: Score 52-70 + UPDR 0.10-0.16 + DOWNDR >= 0.19")
    print(f"    W={s[0]} L={s[1]} F={s[2]} N={s[3]}  WR={s[4]:.1f}%  Fav={s[5]:.2f}%  Adv={s[6]:.2f}%  Edge={s[7]:.2f}%")

# DR spread filter
best_spread = [x for x in completed
               if 52 <= x.get("smart_score", 0) < 70
               and (x.get("downdr_score", 0) - x.get("updr_score", 0)) >= 0.10]
s2 = stats(best_spread)
if s2:
    print(f"\n  SPREAD FILTER: Score 52-70 + DR_spread >= 0.10")
    print(f"    W={s2[0]} L={s2[1]} F={s2[2]} N={s2[3]}  WR={s2[4]:.1f}%  Fav={s2[5]:.2f}%  Adv={s2[6]:.2f}%  Edge={s2[7]:.2f}%")

# Baseline comparison
baseline = [x for x in completed if x.get("smart_score", 0) >= 52]
sb = stats(baseline)
if sb:
    print(f"\n  BASELINE: Score >= 52 (no DR filter)")
    print(f"    W={sb[0]} L={sb[1]} F={sb[2]} N={sb[3]}  WR={sb[4]:.1f}%  Fav={sb[5]:.2f}%  Adv={sb[6]:.2f}%  Edge={sb[7]:.2f}%")

# All trades baseline
sb_all = stats(completed)
if sb_all:
    print(f"\n  ALL TRACKS BASELINE:")
    print(f"    W={sb_all[0]} L={sb_all[1]} F={sb_all[2]} N={sb_all[3]}  WR={sb_all[4]:.1f}%  Fav={sb_all[5]:.2f}%  Adv={sb_all[6]:.2f}%  Edge={sb_all[7]:.2f}%")
