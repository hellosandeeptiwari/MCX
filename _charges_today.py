"""Compute today's Zerodha brokerage + charges for all closed option trades.

Per-leg Zerodha charges for NFO options (2025-26 rates):
- Brokerage:        min(Rs 20, 0.03% * turnover)   per order
- STT:              0.1% of SELL-side premium      (sell leg only)
- Transaction:      0.03503% of premium turnover   (NSE)
- SEBI:             Rs 10 per Crore (0.0001%)      of turnover
- Stamp duty:       0.003% on BUY-side turnover    (buy leg only)
- GST:              18% on (brokerage + transaction + SEBI)

One closed option position = 2 orders (1 BUY + 1 SELL on the option contract).
Turnover per leg = price * qty.
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-24.jsonl")


def charges_for_leg(price, qty, side):
    turnover = price * qty
    brokerage = min(20.0, 0.0003 * turnover)
    stt = 0.0015 * turnover if side == "SELL" else 0.0  # Budget 2026: 0.15% from 1-Apr-2026
    txn = 0.00003503 * turnover
    sebi = 0.0000010 * turnover
    stamp = 0.00003 * turnover if side == "BUY" else 0.0
    gst = 0.18 * (brokerage + txn + sebi)
    total = brokerage + stt + txn + sebi + stamp + gst
    return {"turnover": turnover, "brokerage": brokerage, "stt": stt,
            "txn": txn, "sebi": sebi, "stamp": stamp, "gst": gst, "total": total}


def charges_for_roundtrip(entry, exitp, qty):
    buy = charges_for_leg(entry, qty, "BUY")
    sell = charges_for_leg(exitp, qty, "SELL")
    return {k: buy[k] + sell[k] for k in buy}


agg = {"turnover": 0.0, "brokerage": 0.0, "stt": 0.0, "txn": 0.0,
       "sebi": 0.0, "stamp": 0.0, "gst": 0.0, "total": 0.0}
legs = 0
realized = 0.0
exits = 0
with open(path) as fh:
    for line in fh:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "EXIT":
            continue
        entry = float(d.get("entry_price") or 0)
        exitp = float(d.get("exit_price") or 0)
        qty = int(d.get("quantity") or 0)
        pnl = float(d.get("pnl") or 0)
        if qty == 0 or entry == 0 or exitp == 0:
            continue
        nlegs = len(d.get("symbol", "").split("|"))
        for _ in range(nlegs):
            rt = charges_for_roundtrip(abs(entry), abs(exitp), qty)
            for k in agg:
                agg[k] += rt[k]
        legs += nlegs
        exits += 1
        realized += pnl

print("Closed exits:", exits)
print("Option legs:", legs, "| orders:", legs * 2)
print()
print(f"Turnover:           Rs {agg['turnover']:>14,.2f}")
print(f"Brokerage:          Rs {agg['brokerage']:>14,.2f}")
print(f"STT (sell only):    Rs {agg['stt']:>14,.2f}")
print(f"Exchange txn:       Rs {agg['txn']:>14,.2f}")
print(f"SEBI:               Rs {agg['sebi']:>14,.2f}")
print(f"Stamp duty:         Rs {agg['stamp']:>14,.2f}")
print(f"GST @ 18%:          Rs {agg['gst']:>14,.2f}")
print("-" * 38)
print(f"TOTAL CHARGES:      Rs {agg['total']:>14,.2f}")
print()
print(f"Gross realized P&L: Rs {realized:>14,.2f}")
print(f"Net after charges:  Rs {realized - agg['total']:>14,.2f}")
