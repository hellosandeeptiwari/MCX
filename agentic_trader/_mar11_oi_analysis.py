"""Correlate March 11 trades with OI scoring from AUDIT strings and scorer data."""
import json

# The March 11 ledger doesn't have oi_signal (added Mar 12).
# But the scorer writes OI scoring into the AUDIT trail.
# Let's check what data IS available in the ENTRY records.

ledger_file = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-11.jsonl"

entries = []
exits = {}

for line in open(ledger_file):
    if not line.strip():
        continue
    t = json.loads(line.strip())
    if t.get("event") == "ENTRY":
        entries.append(t)
    elif t.get("event") == "EXIT":
        # Key by order_id to match with entries
        oid = t.get("order_id", "")
        if oid not in exits:
            exits[oid] = t
        else:
            # Multiple exits (partial), accumulate PnL
            exits[oid]["pnl"] = exits[oid].get("pnl", 0) + t.get("pnl", 0)

# Now check SCAN entries for OI-related AUDIT strings
scans = {}
for line in open(ledger_file):
    if not line.strip():
        continue
    t = json.loads(line.strip())
    if t.get("event") == "SCAN":
        sym = t.get("symbol", "")
        reason = t.get("reason", "")
        outcome = t.get("outcome", "")
        # Check for AUDIT string with OI info
        if "OI" in reason.upper() or "AUDIT" in reason.upper():
            if sym not in scans:
                scans[sym] = []
            scans[sym].append({"reason": reason, "outcome": outcome})

# Print all entries with matched exit PnL
print("=" * 100)
print("MARCH 11, 2026 — ALL TRADES WITH OI CORRELATION")
print("=" * 100)
print(f"{'#':>2} {'Stock':<14} {'Dir':>4} {'Source':<18} {'Score':>5} {'PnL':>10} {'Status':<12} {'Rationale'}")
print("-" * 100)

total_pnl = 0
wins = 0
losses = 0
for i, entry in enumerate(entries, 1):
    sym = entry.get("underlying", "").replace("NSE:", "")
    d = entry.get("direction", "")
    src = entry.get("source", "")
    score = entry.get("smart_score", 0)
    oid = entry.get("order_id", "")
    rat = entry.get("rationale", "")
    
    exit_data = exits.get(oid, {})
    pnl = exit_data.get("pnl", 0)
    exit_type = exit_data.get("exit_type", "OPEN?")
    total_pnl += pnl
    if pnl > 0:
        wins += 1
    elif pnl < 0:
        losses += 1
    
    # Truncate rationale for display
    rat_short = rat[:60] if len(rat) > 60 else rat
    pnl_str = f"Rs{pnl:>+,.0f}" if pnl != 0 else "OPEN"
    
    print(f"{i:>2} {sym:<14} {d:>4} {src:<18} {score:>5.0f} {pnl_str:>10} {exit_type:<12} {rat_short}")

print("-" * 100)
print(f"   TOTAL: {len(entries)} trades, {wins}W/{losses}L, PnL = Rs{total_pnl:+,.0f}")
print()

# Now check: do any SCAN entries mention OI?
if scans:
    print("SCAN entries mentioning OI:")
    for sym, scan_list in scans.items():
        for s in scan_list:
            print(f"  {sym}: {s['outcome']} — {s['reason'][:100]}")
else:
    print("NOTE: No SCAN entries found with 'OI' in reason field for March 11.")

# Check ALL unique fields in ENTRY records
print("\nAll fields in March 11 ENTRY records:")
if entries:
    fields = sorted(entries[0].keys())
    print(f"  {', '.join(fields)}")

# Check if there's any OI data stored anywhere in entries
print("\nSearching for OI-related data in any ENTRY field...")
for entry in entries:
    for k, v in entry.items():
        if isinstance(v, str) and "OI" in v.upper() and k != "order_id" and k != "option_symbol":
            sym = entry.get("underlying", "").replace("NSE:", "")
            print(f"  {sym}: field '{k}' = {v[:80]}")
        elif isinstance(v, str) and ("BUILDUP" in v.upper() or "UNWINDING" in v.upper() or "COVERING" in v.upper()):
            sym = entry.get("underlying", "").replace("NSE:", "")
            print(f"  {sym}: field '{k}' = {v[:80]}")

# Also search EXIT records for OI references
print("\nSearching EXIT records for OI references...")
for oid, ex in exits.items():
    for k, v in ex.items():
        if isinstance(v, str) and k not in ("order_id", "option_symbol", "symbol"):
            if "OI" in v.upper() or "BUILDUP" in v.upper() or "UNWINDING" in v.upper():
                sym = ex.get("underlying", "").replace("NSE:", "")
                print(f"  {sym}: field '{k}' = {v[:80]}")

# Now let's check the SCAN entries for WATCHER_OI_CONFLICT or ELITE_OI_CONFLICT outcomes
print("\nSearching for OI CONFLICT decisions in SCAN events...")
for line in open(ledger_file):
    if not line.strip():
        continue
    t = json.loads(line.strip())
    if t.get("event") == "SCAN":
        outcome = t.get("outcome", "")
        if "OI" in outcome.upper():
            sym = t.get("symbol", "").replace("NSE:", "")
            reason = t.get("reason", "")
            direction = t.get("direction", "")
            score = t.get("score", 0)
            print(f"  {sym}: outcome={outcome} dir={direction} score={score} reason={reason[:80]}")
