#!/usr/bin/env python3
"""Analyze strength scores: winners vs losers."""
import json, re
from collections import defaultdict

LEDGER = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-23.jsonl"

entries = {}
trades = []

with open(LEDGER) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ev = rec.get("event", "")
        sym = rec.get("symbol", "")

        if ev == "ENTRY":
            # Extract strength from rationale
            rat = rec.get("rationale", "")
            m = re.search(r'strength=([\d.]+)', rat)
            strength = float(m.group(1)) if m else None
            rec["_strength"] = strength
            entries[sym] = rec
        elif ev == "EXIT":
            entry_rec = entries.get(sym, {})
            if not entry_rec:
                continue
            pnl = rec.get("pnl", 0) or 0
            trade = {
                "symbol": sym,
                "strength": entry_rec.get("_strength"),
                "smart_score": entry_rec.get("smart_score", 0),
                "oi_signal": entry_rec.get("oi_signal", ""),
                "entry_price": entry_rec.get("entry_price", 0),
                "exit_price": rec.get("exit_price", 0),
                "pnl": pnl,
                "pnl_pct": rec.get("pnl_pct", 0) or 0,
                "exit_reason": rec.get("exit_reason", ""),
                "entry_time": entry_rec.get("ts", ""),
                "lots": entry_rec.get("lots", 1),
                "lot_multiplier": entry_rec.get("lot_multiplier", 0),
                "delta": entry_rec.get("delta", 0),
                "iv": entry_rec.get("iv", 0),
                "rationale": entry_rec.get("rationale", ""),
            }
            trades.append(trade)

# Split winners and losers
winners = [t for t in trades if (t["pnl"] or 0) > 0]
losers = [t for t in trades if (t["pnl"] or 0) < 0]

print(f"=== STRENGTH ANALYSIS: WINNERS vs LOSERS ===\n")
print(f"Total trades: {len(trades)}  |  Winners: {len(winners)}  |  Losers: {len(losers)}\n")

# Strength distribution
print(f"{'':30} {'WINNERS':>10} {'LOSERS':>10}")
print("=" * 55)

w_str = [t["strength"] for t in winners if t["strength"] is not None]
l_str = [t["strength"] for t in losers if t["strength"] is not None]
if w_str:
    print(f"{'Avg Strength':<30} {sum(w_str)/len(w_str):>10.3f} {sum(l_str)/len(l_str) if l_str else 0:>10.3f}")
    print(f"{'Min Strength':<30} {min(w_str):>10.3f} {min(l_str) if l_str else 0:>10.3f}")
    print(f"{'Max Strength':<30} {max(w_str):>10.3f} {max(l_str) if l_str else 0:>10.3f}")

w_iv = [t["iv"] for t in winners if t["iv"]]
l_iv = [t["iv"] for t in losers if t["iv"]]
if w_iv:
    print(f"{'Avg IV at Entry':<30} {sum(w_iv)/len(w_iv):>10.3f} {sum(l_iv)/len(l_iv) if l_iv else 0:>10.3f}")

w_delta = [abs(t["delta"]) for t in winners if t["delta"]]
l_delta = [abs(t["delta"]) for t in losers if t["delta"]]
if w_delta:
    print(f"{'Avg |Delta| at Entry':<30} {sum(w_delta)/len(w_delta):>10.3f} {sum(l_delta)/len(l_delta) if l_delta else 0:>10.3f}")

w_lots = [t["lot_multiplier"] for t in winners if t["lot_multiplier"]]
l_lots = [t["lot_multiplier"] for t in losers if t["lot_multiplier"]]
if w_lots:
    print(f"{'Avg Lot Multiplier':<30} {sum(w_lots)/len(w_lots):>10.3f} {sum(l_lots)/len(l_lots) if l_lots else 0:>10.3f}")

w_pnl = [t["pnl"] for t in winners]
l_pnl = [t["pnl"] for t in losers]
print(f"{'Avg PnL':<30} {sum(w_pnl)/len(w_pnl):>+10,.0f} {sum(l_pnl)/len(l_pnl) if l_pnl else 0:>+10,.0f}")
print(f"{'Total PnL':<30} {sum(w_pnl):>+10,.0f} {sum(l_pnl):>+10,.0f}")

# Per-trade detail with strength
print(f"\n\n=== PER-TRADE STRENGTH DETAIL (sorted by PnL) ===\n")
trades.sort(key=lambda x: x["pnl"] or 0, reverse=True)
print(f"{'SYMBOL':<32} {'STR':>5} {'IV':>6} {'|D|':>5} {'LOTS':>4} {'PNL':>10} {'PCT':>7} {'OI_SIGNAL':<18} {'EXIT_REASON':<35}")
print("=" * 145)
for t in trades:
    pnl = t["pnl"] or 0
    s = t["strength"]
    s_str = f"{s:.2f}" if s is not None else "N/A"
    iv = t["iv"] or 0
    d = abs(t["delta"] or 0)
    er = t["exit_reason"]
    if len(er) > 34:
        er = er[:34]
    print(f"{t['symbol']:<32} {s_str:>5} {iv:>5.1f}% {d:>5.2f} {t['lots']:>4} {pnl:>+10,.0f} {t['pnl_pct']:>+6.1f}% {t['oi_signal']:<18} {er:<35}")

# Strength buckets
print(f"\n\n=== STRENGTH BUCKETS ===\n")
buckets = {"0.00-0.30": [], "0.31-0.50": [], "0.51-0.70": [], "0.71-0.90": [], "0.91-1.00": []}
for t in trades:
    s = t["strength"]
    if s is None:
        continue
    if s <= 0.30:
        buckets["0.00-0.30"].append(t)
    elif s <= 0.50:
        buckets["0.31-0.50"].append(t)
    elif s <= 0.70:
        buckets["0.51-0.70"].append(t)
    elif s <= 0.90:
        buckets["0.71-0.90"].append(t)
    else:
        buckets["0.91-1.00"].append(t)

print(f"{'BUCKET':<15} {'TRADES':>6} {'WINNERS':>8} {'WIN%':>6} {'TOTAL PNL':>12} {'AVG PNL':>10}")
print("=" * 65)
for bucket, bt in buckets.items():
    if not bt:
        print(f"{bucket:<15} {0:>6} {0:>8} {'N/A':>6} {0:>+12,.0f} {0:>+10,.0f}")
        continue
    pnls = [t["pnl"] or 0 for t in bt]
    w = sum(1 for p in pnls if p > 0)
    print(f"{bucket:<15} {len(bt):>6} {w:>8} {w/len(bt)*100:>5.0f}% {sum(pnls):>+12,.0f} {sum(pnls)/len(pnls):>+10,.0f}")

# OI signal breakdown
print(f"\n\n=== OI SIGNAL BREAKDOWN ===\n")
by_sig = defaultdict(list)
for t in trades:
    by_sig[t["oi_signal"]].append(t)

print(f"{'OI_SIGNAL':<25} {'TRADES':>6} {'WINNERS':>8} {'WIN%':>6} {'TOTAL PNL':>12} {'AVG PNL':>10}")
print("=" * 75)
for sig in sorted(by_sig.keys()):
    st = by_sig[sig]
    pnls = [t["pnl"] or 0 for t in st]
    w = sum(1 for p in pnls if p > 0)
    print(f"{sig:<25} {len(st):>6} {w:>8} {w/len(st)*100:>5.0f}% {sum(pnls):>+12,.0f} {sum(pnls)/len(pnls):>+10,.0f}")

# Time-of-entry analysis
print(f"\n\n=== ENTRY TIME ANALYSIS ===\n")
by_hour = defaultdict(list)
for t in trades:
    et = t.get("entry_time", "")
    if et and "T" in et:
        h = et.split("T")[1][:2]
        by_hour[h].append(t)

print(f"{'HOUR':<8} {'TRADES':>6} {'WINNERS':>8} {'WIN%':>6} {'TOTAL PNL':>12} {'AVG PNL':>10}")
print("=" * 55)
for h in sorted(by_hour.keys()):
    ht = by_hour[h]
    pnls = [t["pnl"] or 0 for t in ht]
    w = sum(1 for p in pnls if p > 0)
    print(f"{h}:00     {len(ht):>6} {w:>8} {w/len(ht)*100:>5.0f}% {sum(pnls):>+12,.0f} {sum(pnls)/len(pnls):>+10,.0f}")
