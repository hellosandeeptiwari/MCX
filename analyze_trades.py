#!/usr/bin/env python3
"""Analyze today's trade ledger."""
import json
from collections import defaultdict

LEDGER = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-23.jsonl"

entries = {}  # symbol -> entry record
exits = {}   # symbol -> exit record
trades = []  # matched trades

with open(LEDGER) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ev = rec.get("event", "")
        sym = rec.get("symbol", "")
        
        if ev == "ENTRY":
            entries[sym] = rec
        elif ev == "EXIT":
            exits[sym] = rec
            # Match with entry
            entry_rec = entries.get(sym, {})
            # Strategy from "source" field in ENTRY, fallback to exit_reason prefix
            strat = entry_rec.get("source", "")
            if not strat:
                er = rec.get("exit_reason", rec.get("reason", ""))
                if "OI_WATCHER" in er:
                    strat = "OI_WATCHER"
                elif "SNIPER" in er:
                    strat = "GMM_SNIPER"
                elif "ORB" in er:
                    strat = "ORB_BREAKOUT"
                else:
                    strat = "UNKNOWN"
            trade = {
                "symbol": sym,
                "strategy": strat,
                "direction": entry_rec.get("direction", rec.get("direction", "?")),
                "entry_price": entry_rec.get("entry_price", entry_rec.get("premium", 0)),
                "exit_price": rec.get("exit_price", rec.get("premium", 0)),
                "pnl": rec.get("pnl", rec.get("realized_pnl", 0)),
                "pnl_pct": rec.get("pnl_pct", 0),
                "exit_reason": rec.get("exit_reason", rec.get("reason", "?")),
                "lots": entry_rec.get("lots", rec.get("lots", 1)),
                "entry_time": entry_rec.get("ts", "?"),
                "exit_time": rec.get("ts", "?"),
                "oi_signal": entry_rec.get("oi_signal", ""),
                "is_sniper": entry_rec.get("is_sniper", False),
            }
            trades.append(trade)

# Sort by PnL
trades.sort(key=lambda x: x.get("pnl", 0) or 0, reverse=True)

print(f"=== TOTAL TRADES: {len(trades)} ===\n")

# Per-trade detail
print(f"{'SYMBOL':<30} {'STRATEGY':<20} {'DIR':<5} {'ENTRY':>8} {'EXIT':>8} {'PNL':>10} {'PCT':>7} {'EXIT_REASON':<35}")
print("=" * 150)
for t in trades:
    pnl = t.get("pnl", 0) or 0
    pct = t.get("pnl_pct", 0) or 0
    print(f"{t['symbol']:<30} {t['strategy']:<20} {t['direction']:<5} {t.get('entry_price',0):>8.2f} {t.get('exit_price',0):>8.2f} {pnl:>+10,.0f} {pct:>+6.1f}% {t['exit_reason']:<35}")

# Strategy summary
print(f"\n\n=== STRATEGY SUMMARY ===\n")
by_strat = defaultdict(list)
for t in trades:
    by_strat[t["strategy"]].append(t)

print(f"{'STRATEGY':<25} {'TRADES':>6} {'WINNERS':>8} {'LOSERS':>7} {'WIN%':>6} {'TOTAL PNL':>12} {'AVG PNL':>10} {'BEST':>10} {'WORST':>10}")
print("=" * 110)

total_pnl = 0
total_trades = 0
total_winners = 0

for strat in sorted(by_strat.keys()):
    strades = by_strat[strat]
    pnls = [t.get("pnl", 0) or 0 for t in strades]
    winners = sum(1 for p in pnls if p > 0)
    losers = sum(1 for p in pnls if p < 0)
    tot = sum(pnls)
    avg = tot / len(pnls) if pnls else 0
    best = max(pnls)
    worst = min(pnls)
    win_pct = (winners / len(pnls) * 100) if pnls else 0
    
    print(f"{strat:<25} {len(strades):>6} {winners:>8} {losers:>7} {win_pct:>5.0f}% {tot:>+12,.0f} {avg:>+10,.0f} {best:>+10,.0f} {worst:>+10,.0f}")
    total_pnl += tot
    total_trades += len(strades)
    total_winners += winners

print("=" * 110)
print(f"{'TOTAL':<25} {total_trades:>6} {total_winners:>8} {total_trades-total_winners:>7} {(total_winners/total_trades*100 if total_trades else 0):>5.0f}% {total_pnl:>+12,.0f}")

# Exit reason breakdown
print(f"\n\n=== EXIT REASON BREAKDOWN ===\n")
by_reason = defaultdict(list)
for t in trades:
    by_reason[t["exit_reason"]].append(t)

print(f"{'EXIT_REASON':<40} {'COUNT':>6} {'TOTAL PNL':>12} {'AVG PNL':>10}")
print("=" * 75)
for reason in sorted(by_reason.keys(), key=lambda r: sum(t.get("pnl",0) or 0 for t in by_reason[r])):
    rtrades = by_reason[reason]
    pnls = [t.get("pnl", 0) or 0 for t in rtrades]
    print(f"{reason:<40} {len(rtrades):>6} {sum(pnls):>+12,.0f} {sum(pnls)/len(pnls):>+10,.0f}")

# Top winners and losers
print(f"\n\n=== TOP 10 WINNERS ===")
for t in trades[:10]:
    pnl = t.get("pnl", 0) or 0
    if pnl <= 0:
        break
    print(f"  {t['symbol']:<30} {t['strategy']:<18} PnL: {pnl:>+10,.0f}  ({t['exit_reason']})")

print(f"\n=== TOP 10 LOSERS ===")
for t in trades[-10:]:
    pnl = t.get("pnl", 0) or 0
    if pnl >= 0:
        continue
    print(f"  {t['symbol']:<30} {t['strategy']:<18} PnL: {pnl:>+10,.0f}  ({t['exit_reason']})")
