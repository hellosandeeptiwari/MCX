"""Audit AutoPilot's P&L impact for today."""
import json
from collections import defaultdict
from datetime import datetime

with open("auto_pilot_state.json") as f:
    ap = json.load(f)
decs = ap.get("last_decision", [])
print(f"Total AutoPilot decisions stored (last 50): {len(decs)}")
by = defaultdict(int)
for d in decs:
    by[d["outcome"]] += 1
for k, v in by.items():
    print(f"  {k}: {v}")

# Day counters (authoritative total for the day)
counters = ap.get("counters", {})
tot_adds = sum(c.get("adds", 0) for c in counters.values())
tot_rev = sum(c.get("reverses", 0) for c in counters.values())
print(f"\nDay totals from counters: {tot_adds} adds, {tot_rev} reverses across {len(counters)} symbols")

rows = []
with open("trade_ledger/trade_ledger_2026-04-23.jsonl") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

entries = [r for r in rows if r.get("event") == "ENTRY"]
exits = [r for r in rows if r.get("event") == "EXIT"]
print(f"\nLedger today: {len(entries)} ENTRYs, {len(exits)} EXITs")

executed = [d for d in decs if d["outcome"] == "EXECUTED"]
print(f"\nExecuted AutoPilot actions (in last-50 window): {len(executed)}")

# Build (ts, symbol, action) tuples
ap_actions = []
for d in executed:
    try:
        ap_actions.append((datetime.fromisoformat(d["ts"]), d["symbol"], d["action"]))
    except Exception:
        pass

# REVERSE = exit old leg + open opposite leg. The opposite leg entry happens at same ts.
# When we find an EXIT whose entry_time ≈ an AP REVERSE timestamp, it means AP opened that
# leg and it has since closed — pnl attributable to AP.
ap_exits = []
for ex in exits:
    sym = ex.get("symbol", "")
    et = ex.get("entry_time", "")
    if not et:
        continue
    try:
        et_dt = datetime.fromisoformat(et.replace("Z", ""))
    except Exception:
        continue
    for (ap_dt, asym, act) in ap_actions:
        if asym != sym:
            continue
        if abs((et_dt - ap_dt).total_seconds()) < 180:
            ap_exits.append((ap_dt, sym, act, ex.get("pnl", 0), ex.get("exit_reason", ""), ex.get("pnl_pct", 0)))
            break

print(f"\n--- AutoPilot-sourced trades that ALREADY EXITED ({len(ap_exits)}) ---")
tot = 0.0
wins = 0
losses = 0
for e in ap_exits:
    tot += e[3]
    if e[3] > 0:
        wins += 1
    elif e[3] < 0:
        losses += 1
    print(f"  {e[0].strftime('%H:%M:%S')} {e[2]:<8} {e[1]:<42} pnl=Rs{e[3]:+10.0f} ({e[5]:+5.1f}%)  {e[4][:45]}")
print(f"\nREALISED P&L from AutoPilot closed legs: Rs{tot:+,.0f} | W={wins} L={losses}")

# For still-open positions that AP touched, compute unrealised
with open("active_trades.json") as f:
    act = json.load(f)

ap_open = []
for p in act:
    if p.get("status", "OPEN") != "OPEN":
        continue
    sym = p.get("symbol", "")
    for (_, asym, _) in ap_actions:
        if asym == sym:
            ltp = p.get("ltp", 0) or 0
            qty = p.get("quantity", 0)
            avg = p.get("avg_price", 0)
            side = p.get("side", "BUY")
            if ltp and qty and avg:
                upnl = (ltp - avg) * qty if side == "BUY" else (avg - ltp) * qty
            else:
                upnl = p.get("unrealized_pnl", 0)
            ap_open.append((sym, qty, avg, ltp, upnl, counters.get(sym, {})))
            break

print(f"\n--- Still-OPEN positions AutoPilot has touched ({len(ap_open)}) ---")
total_upnl = 0.0
for sym, qty, avg, ltp, upnl, c in ap_open:
    total_upnl += upnl
    adds = c.get("adds", 0)
    revs = c.get("reverses", 0)
    print(f"  {sym:<42} qty={qty:>6} avg={avg:>7.2f} ltp={ltp:>7.2f} uPnL=Rs{upnl:>+10.0f}  AP:[+{adds}add rev{revs}]")
print(f"\nUNREALISED P&L on AP-touched open positions: Rs{total_upnl:+,.0f}")
print(f"\nTOTAL AP impact (realised + unrealised): Rs{tot + total_upnl:+,.0f}")
