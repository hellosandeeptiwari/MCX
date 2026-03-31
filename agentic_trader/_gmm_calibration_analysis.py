#!/usr/bin/env python3
"""Analyze GMM calibration data for range/score insights."""
import json
from collections import Counter
from typing import Any, Dict, List

DATA_FILE = "/home/ubuntu/titan/logs/gmm_calibration_data.jsonl"

data: List[Dict[str, Any]] = []
with open(DATA_FILE) as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"Total entries: {len(data)}\n")

# Per date stats
dates: Dict[str, Dict[str, Any]] = {}
for d in data:
    dt: str = d.get("date", "?")
    if dt not in dates:
        dates[dt] = {"total": 0, "completed": 0, "outcomes": Counter(),
                     "max_fav": [], "smart_scores": [], "entries": []}
    dates[dt]["total"] += 1
    if d.get("completed"):
        dates[dt]["completed"] += 1
    dates[dt]["outcomes"][d.get("outcome", "?")] += 1
    if d.get("max_favorable") is not None:
        dates[dt]["max_fav"].append(d["max_favorable"])
    if d.get("smart_score") is not None:
        dates[dt]["smart_scores"].append(d["smart_score"])
    dates[dt]["entries"].append(d)

for dt in sorted(dates.keys()):
    s = dates[dt]
    max_fav_list: List[float] = s["max_fav"]
    scores_list: List[float] = s["smart_scores"]
    entries_list: List[Dict[str, Any]] = s["entries"]
    avg_fav = sum(max_fav_list) / len(max_fav_list) if max_fav_list else 0
    avg_sc = sum(scores_list) / len(scores_list) if scores_list else 0
    print(f"{'='*60}")
    print(f"DATE: {dt}  |  {s['total']} tracks ({s['completed']} completed)")
    print(f"  Outcomes: {dict(s['outcomes'])}")
    print(f"  Avg max_favorable: {avg_fav:.3f}%  |  Avg smart_score: {avg_sc:.1f}")

    # Score distribution
    scores = scores_list
    if scores:
        bins = {"<40": 0, "40-52": 0, "52-60": 0, "60-70": 0, "70+": 0}
        for sc in scores:
            if sc < 40: bins["<40"] += 1
            elif sc < 52: bins["40-52"] += 1
            elif sc < 60: bins["52-60"] += 1
            elif sc < 70: bins["60-70"] += 1
            else: bins["70+"] += 1
        print(f"  Score dist: {bins}")

    # Win rate by score range
    print(f"\n  {'Range':<10} {'W':>4} {'L':>4} {'F':>4} {'Total':>6} {'WinR':>6} {'AvgFav':>8} {'AvgAdv':>8}")
    print(f"  {'-'*56}")
    for lo, hi, lbl in [(0, 40, "<40"), (40, 52, "40-52"), (52, 60, "52-60"),
                         (60, 70, "60-70"), (70, 200, "70+")]:
        bucket = [x for x in entries_list
                  if x.get("smart_score", 0) >= lo and x.get("smart_score", 0) < hi
                  and x.get("completed")]
        if bucket:
            wins = sum(1 for x in bucket if "PROFIT" in x.get("outcome", ""))
            losses = sum(1 for x in bucket if "LOSS" in x.get("outcome", ""))
            flats = sum(1 for x in bucket if x.get("outcome") == "FLAT")
            avg_mf = sum(x.get("max_favorable", 0) for x in bucket) / len(bucket)
            avg_ma = sum(x.get("max_adverse", 0) for x in bucket) / len(bucket)
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            print(f"  {lbl:<10} {wins:>4} {losses:>4} {flats:>4} {len(bucket):>6} {wr:>5.1f}% {avg_mf:>7.2f}% {avg_ma:>7.2f}%")

    # GMM direction accuracy
    confirms = [x for x in entries_list if x.get("completed") and x.get("gmm_confirms_dir") is not None]
    if confirms:
        gmm_yes = [x for x in confirms if x.get("gmm_confirms_dir")]
        gmm_no = [x for x in confirms if not x.get("gmm_confirms_dir")]
        
        gmm_yes_wins = sum(1 for x in gmm_yes if "PROFIT" in x.get("outcome", ""))
        gmm_yes_total = len(gmm_yes)
        gmm_no_wins = sum(1 for x in gmm_no if "PROFIT" in x.get("outcome", ""))
        gmm_no_total = len(gmm_no)
        
        print(f"\n  GMM confirms=True:  {gmm_yes_wins}W / {gmm_yes_total} ({gmm_yes_wins/gmm_yes_total*100:.1f}% WR)" if gmm_yes_total else "")
        print(f"  GMM confirms=False: {gmm_no_wins}W / {gmm_no_total} ({gmm_no_wins/gmm_no_total*100:.1f}% WR)" if gmm_no_total else "")
    
    # Direction conflict analysis (scorer vs GMM)
    conflicts = [x for x in entries_list if x.get("completed") 
                 and x.get("scorer_direction") and x.get("direction_hint")]
    if conflicts:
        aligned = [x for x in conflicts if 
                   (x["scorer_direction"] == "BUY" and "BULL" in x["direction_hint"]) or
                   (x["scorer_direction"] == "SELL" and "BEAR" in x["direction_hint"])]
        misaligned = [x for x in conflicts if x not in aligned]
        al_wins = sum(1 for x in aligned if "PROFIT" in x.get("outcome", ""))
        mis_wins = sum(1 for x in misaligned if "PROFIT" in x.get("outcome", ""))
        print(f"\n  Scorer-GMM aligned:    {al_wins}W / {len(aligned)} ({al_wins/len(aligned)*100:.1f}% WR)" if aligned else "")
        print(f"  Scorer-GMM misaligned: {mis_wins}W / {len(misaligned)} ({mis_wins/len(misaligned)*100:.1f}% WR)" if misaligned else "")
    
    print()

# === OVERALL BEST RANGE ===
print("=" * 60)
print("OVERALL: WHICH SCORE RANGE IS PROFITABLE?")
print("=" * 60)
all_completed = [x for x in data if x.get("completed")]
for lo, hi, lbl in [(0, 40, "<40"), (40, 52, "40-52"), (52, 60, "52-60"),
                     (60, 70, "60-70"), (70, 200, "70+")]:
    bucket = [x for x in all_completed if x.get("smart_score", 0) >= lo and x.get("smart_score", 0) < hi]
    if bucket:
        wins = sum(1 for x in bucket if "PROFIT" in x.get("outcome", ""))
        losses = sum(1 for x in bucket if "LOSS" in x.get("outcome", ""))
        flats = sum(1 for x in bucket if x.get("outcome") == "FLAT")
        avg_mf = sum(x.get("max_favorable", 0) for x in bucket) / len(bucket)
        avg_ma = sum(x.get("max_adverse", 0) for x in bucket) / len(bucket)
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        edge = avg_mf + avg_ma  # avg_ma is negative
        print(f"  {lbl:<10} W={wins:>3} L={losses:>3} F={flats:>3}  WR={wr:>5.1f}%  AvgFav={avg_mf:>6.2f}%  AvgAdv={avg_ma:>7.2f}%  Edge={edge:>6.2f}%")
