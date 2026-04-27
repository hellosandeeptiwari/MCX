#!/usr/bin/env python3
"""Quick summary of GMM calibration data collected today."""
import json

lines = open("/home/ubuntu/titan/logs/gmm_calibration_data.jsonl").readlines()
records = [json.loads(l) for l in lines]
completed = [r for r in records if r.get("completed")]
pending = [r for r in records if not r.get("completed")]

outcomes = {}
for r in completed:
    o = r.get("outcome", "UNKNOWN")
    outcomes[o] = outcomes.get(o, 0) + 1

unique_syms = len(set(r["symbol"] for r in records))
unique_cycles = len(set(r["cycle_time"] for r in records))

print(f"=== GMM CALIBRATION DATA — {records[0]['date'] if records else 'N/A'} ===")
print(f"Total records: {len(records)}")
print(f"Completed: {len(completed)}")
print(f"Still pending: {len(pending)}")
print(f"Unique symbols: {unique_syms}")
print(f"Unique scan cycles: {unique_cycles}")
print(f"\nOutcomes: {outcomes}")

if completed:
    ups = [r["updr_score"] for r in completed]
    dns = [r["downdr_score"] for r in completed]
    smarts = [r["smart_score"] for r in completed]
    avg_u = sum(ups) / len(ups)
    avg_d = sum(dns) / len(dns)
    avg_s = sum(smarts) / len(smarts)
    print(f"\n--- ALL COMPLETED ({len(completed)}) ---")
    print(f"  UPDR range:  {min(ups):.4f} - {max(ups):.4f} (avg {avg_u:.4f})")
    print(f"  DownDR range: {min(dns):.4f} - {max(dns):.4f} (avg {avg_d:.4f})")
    print(f"  Smart range:  {min(smarts):.1f} - {max(smarts):.1f} (avg {avg_s:.1f})")

    # Profitable
    profit_up = [r for r in completed if r["outcome"] == "PROFIT_UP"]
    profit_dn = [r for r in completed if r["outcome"] == "PROFIT_DOWN"]
    profit = profit_up + profit_dn
    if profit:
        p_ups = [r["updr_score"] for r in profit]
        p_dns = [r["downdr_score"] for r in profit]
        p_smarts = [r["smart_score"] for r in profit]
        p_favs = [r["max_favorable"] for r in profit]
        print(f"\n--- PROFITABLE ({len(profit)}: {len(profit_up)} UP + {len(profit_dn)} DOWN) ---")
        print(f"  UPDR:   {min(p_ups):.4f} - {max(p_ups):.4f} (avg {sum(p_ups)/len(p_ups):.4f})")
        print(f"  DownDR: {min(p_dns):.4f} - {max(p_dns):.4f} (avg {sum(p_dns)/len(p_dns):.4f})")
        print(f"  Smart:  {min(p_smarts):.1f} - {max(p_smarts):.1f} (avg {sum(p_smarts)/len(p_smarts):.1f})")
        print(f"  Max favorable move: {min(p_favs):.2f}% - {max(p_favs):.2f}% (avg {sum(p_favs)/len(p_favs):.2f}%)")

    # Losers
    loss = [r for r in completed if "LOSS" in r.get("outcome", "")]
    if loss:
        l_ups = [r["updr_score"] for r in loss]
        l_dns = [r["downdr_score"] for r in loss]
        print(f"\n--- LOSERS ({len(loss)}) ---")
        print(f"  UPDR:   {min(l_ups):.4f} - {max(l_ups):.4f} (avg {sum(l_ups)/len(l_ups):.4f})")
        print(f"  DownDR: {min(l_dns):.4f} - {max(l_dns):.4f} (avg {sum(l_dns)/len(l_dns):.4f})")

    # Flat
    flat = [r for r in completed if r["outcome"] == "FLAT"]
    if flat:
        print(f"\n--- FLAT ({len(flat)}) ---")

    # Direction accuracy
    dir_correct = 0
    dir_total = 0
    for r in completed:
        sd = r.get("scorer_direction", "")
        outcome = r.get("outcome", "")
        if sd in ("BUY", "SELL") and outcome not in ("FLAT",):
            dir_total += 1
            if (sd == "BUY" and "UP" in outcome) or (sd == "SELL" and "DOWN" in outcome):
                dir_correct += 1
    if dir_total > 0:
        print(f"\n--- DIRECTION ACCURACY ---")
        print(f"  Correct: {dir_correct}/{dir_total} ({dir_correct/dir_total*100:.1f}%)")

# Pending details
if pending:
    pend_candles = [r.get("candles_remaining", 8) for r in pending]
    print(f"\n--- PENDING ({len(pending)}) ---")
    print(f"  Avg candles remaining: {sum(pend_candles)/len(pend_candles):.1f}")

# Top profitable symbols
if completed:
    sym_profits = {}
    for r in completed:
        s = r["symbol"].replace("NSE:", "")
        fav = r.get("max_favorable", 0)
        if s not in sym_profits or fav > sym_profits[s]:
            sym_profits[s] = fav
    top = sorted(sym_profits.items(), key=lambda x: -x[1])[:10]
    print(f"\n--- TOP 10 FAVORABLE MOVES ---")
    for s, f in top:
        print(f"  {s}: +{f:.2f}%")
