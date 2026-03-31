#!/usr/bin/env python3
"""Analyze GMM calibration data by UPDR/DOWNDR score ranges."""
import json
from collections import defaultdict

DATA_FILE = "/home/ubuntu/titan/logs/gmm_calibration_data.jsonl"

data = []
with open(DATA_FILE) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

completed = [x for x in data if x.get("completed")]
total = len(completed)

# ========== UPDR RANGE ==========
print("=" * 65)
print(f"UPDR SCORE RANGE ANALYSIS -- {total} completed tracks, 2 days")
print("=" * 65)
header = f"{'Range':<14} {'W':>4} {'L':>4} {'F':>4} {'Tot':>5} {'WR':>6} {'AvgFav':>8} {'AvgAdv':>8} {'Edge':>7}"
print(header)
print("-" * 65)
for lo, hi, lbl in [(0, 0.08, "<0.08"), (0.08, 0.12, "0.08-0.12"), (0.12, 0.16, "0.12-0.16"),
                     (0.16, 0.20, "0.16-0.20"), (0.20, 0.25, "0.20-0.25"), (0.25, 1.0, "0.25+")]:
    b = [x for x in completed if lo <= x.get("updr_score", 0) < hi]
    if b:
        w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
        l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
        f = sum(1 for x in b if x.get("outcome") == "FLAT")
        mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
        ma = sum(x.get("max_adverse", 0) for x in b) / len(b)
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        print(f"{lbl:<14} {w:>4} {l:>4} {f:>4} {len(b):>5} {wr:>5.1f}% {mf:>7.2f}% {ma:>7.2f}% {mf+ma:>6.2f}%")

# ========== DOWNDR RANGE ==========
print()
print("=" * 65)
print("DOWNDR SCORE RANGE ANALYSIS")
print("=" * 65)
print(header)
print("-" * 65)
for lo, hi, lbl in [(0, 0.10, "<0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.19, "0.15-0.19"),
                     (0.19, 0.25, "0.19-0.25"), (0.25, 0.35, "0.25-0.35"), (0.35, 1.0, "0.35+")]:
    b = [x for x in completed if lo <= x.get("downdr_score", 0) < hi]
    if b:
        w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
        l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
        f = sum(1 for x in b if x.get("outcome") == "FLAT")
        mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
        ma = sum(x.get("max_adverse", 0) for x in b) / len(b)
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        print(f"{lbl:<14} {w:>4} {l:>4} {f:>4} {len(b):>5} {wr:>5.1f}% {mf:>7.2f}% {ma:>7.2f}% {mf+ma:>6.2f}%")

# ========== UPDR x DOWNDR COMBINED ==========
print()
print("=" * 65)
print("UPDR x DOWNDR COMBINED -- best combos")
print("=" * 65)
print(f"{'UPDR':<10} {'DOWNDR':<10} {'W':>3} {'L':>3} {'F':>3} {'Tot':>4} {'WR':>6} {'AvgFav':>8} {'Edge':>7}")
print("-" * 60)
combos = []
for ulo, uhi, ulbl in [(0, 0.10, "<0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.20, "0.15-0.20"), (0.20, 1.0, "0.20+")]:
    for dlo, dhi, dlbl in [(0, 0.12, "<0.12"), (0.12, 0.19, "0.12-0.19"), (0.19, 0.25, "0.19-0.25"), (0.25, 1.0, "0.25+")]:
        b = [x for x in completed if ulo <= x.get("updr_score", 0) < uhi and dlo <= x.get("downdr_score", 0) < dhi]
        if len(b) >= 5:
            w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
            l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
            f = sum(1 for x in b if x.get("outcome") == "FLAT")
            mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
            ma = sum(x.get("max_adverse", 0) for x in b) / len(b)
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            combos.append((wr, ulbl, dlbl, w, l, f, len(b), mf, mf + ma))

combos.sort(reverse=True)
for wr, ul, dl, w, l, f, t, mf, edge in combos:
    marker = " ***" if wr > 20 else ""
    print(f"{ul:<10} {dl:<10} {w:>3} {l:>3} {f:>3} {t:>4} {wr:>5.1f}% {mf:>7.2f}% {edge:>6.2f}%{marker}")

# ========== DIRECTION FLAGS ==========
print()
print("=" * 65)
print("DIRECTION FLAG ANALYSIS -- up_flag / down_flag")
print("=" * 65)
for flag_name in ["up_flag", "down_flag"]:
    yes = [x for x in completed if x.get(flag_name)]
    no = [x for x in completed if not x.get(flag_name)]
    yw = sum(1 for x in yes if "PROFIT" in x.get("outcome", ""))
    yl = sum(1 for x in yes if "LOSS" in x.get("outcome", ""))
    nw = sum(1 for x in no if "PROFIT" in x.get("outcome", ""))
    nl = sum(1 for x in no if "LOSS" in x.get("outcome", ""))
    ywr = yw / (yw + yl) * 100 if (yw + yl) > 0 else 0
    nwr = nw / (nw + nl) * 100 if (nw + nl) > 0 else 0
    print(f"  {flag_name}=True:  W={yw} L={yl} -- {len(yes)} total, WR={ywr:.1f}%")
    print(f"  {flag_name}=False: W={nw} L={nl} -- {len(no)} total, WR={nwr:.1f}%")
    print()

# ========== UPDR/DOWNDR SPREAD ==========
print("=" * 65)
print("UPDR - DOWNDR SPREAD -- does imbalance predict direction?")
print("=" * 65)
print(f"{'Spread':<16} {'W':>4} {'L':>4} {'F':>4} {'Tot':>5} {'WR':>6} {'AvgFav':>8} {'Edge':>7}")
print("-" * 60)
for lo, hi, lbl in [(-1.0, -0.10, "DR>>UR heavy"), (-0.10, -0.03, "DR>UR mod"),
                     (-0.03, 0.03, "Balanced"), (0.03, 0.10, "UR>DR mod"), (0.10, 1.0, "UR>>DR heavy")]:
    b = [x for x in completed
         if lo <= (x.get("updr_score", 0) - x.get("downdr_score", 0)) < hi]
    if b:
        w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
        l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
        f = sum(1 for x in b if x.get("outcome") == "FLAT")
        mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
        ma = sum(x.get("max_adverse", 0) for x in b) / len(b)
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        print(f"{lbl:<16} {w:>4} {l:>4} {f:>4} {len(b):>5} {wr:>5.1f}% {mf:>7.2f}% {mf+ma:>6.2f}%")

# ========== PER-DAY UPDR BREAKDOWN ==========
print()
for dt in sorted(set(x.get("date") for x in completed)):
    day = [x for x in completed if x.get("date") == dt]
    print(f"\n{'='*65}")
    print(f"DATE: {dt} -- UPDR breakdown")
    print(f"{'Range':<14} {'W':>4} {'L':>4} {'F':>4} {'Tot':>5} {'WR':>6} {'AvgFav':>8}")
    print("-" * 50)
    for lo, hi, lbl in [(0, 0.08, "<0.08"), (0.08, 0.12, "0.08-0.12"), (0.12, 0.16, "0.12-0.16"),
                         (0.16, 0.20, "0.16-0.20"), (0.20, 1.0, "0.20+")]:
        b = [x for x in day if lo <= x.get("updr_score", 0) < hi]
        if b:
            w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
            l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
            f = sum(1 for x in b if x.get("outcome") == "FLAT")
            mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"{lbl:<14} {w:>4} {l:>4} {f:>4} {len(b):>5} {wr:>5.1f}% {mf:>7.2f}%")
    
    print(f"\nDATE: {dt} -- DOWNDR breakdown")
    print(f"{'Range':<14} {'W':>4} {'L':>4} {'F':>4} {'Tot':>5} {'WR':>6} {'AvgFav':>8}")
    print("-" * 50)
    for lo, hi, lbl in [(0, 0.10, "<0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.19, "0.15-0.19"),
                         (0.19, 0.25, "0.19-0.25"), (0.25, 1.0, "0.25+")]:
        b = [x for x in day if lo <= x.get("downdr_score", 0) < hi]
        if b:
            w = sum(1 for x in b if "PROFIT" in x.get("outcome", ""))
            l = sum(1 for x in b if "LOSS" in x.get("outcome", ""))
            f = sum(1 for x in b if x.get("outcome") == "FLAT")
            mf = sum(x.get("max_favorable", 0) for x in b) / len(b)
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"{lbl:<14} {w:>4} {l:>4} {f:>4} {len(b):>5} {wr:>5.1f}% {mf:>7.2f}%")
