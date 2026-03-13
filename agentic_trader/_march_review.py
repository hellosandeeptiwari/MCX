"""March 2026 performance review — run on EC2."""
import json, glob, os

files = sorted(glob.glob("/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-*.jsonl"))
all_closed = []
print("=" * 60)
print("TITAN v5 — MARCH 2026 PERFORMANCE REVIEW")
print("=" * 60)

for f in files:
    trades = [json.loads(l) for l in open(f) if l.strip()]
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    if not closed:
        continue
    wins = len([t for t in closed if t.get("pnl", 0) > 0])
    losses = len(closed) - wins
    pnl = sum(t.get("pnl", 0) for t in closed)
    day = os.path.basename(f).replace("trade_ledger_", "").replace(".jsonl", "")
    print(f"  {day}: {len(closed):2d}T  {wins}W/{losses}L  PnL=Rs{pnl:>+9,.0f}")
    all_closed.extend(closed)

print("-" * 60)
if all_closed:
    wins = len([t for t in all_closed if t.get("pnl", 0) > 0])
    losses = len(all_closed) - wins
    pnl = sum(t.get("pnl", 0) for t in all_closed)
    avg_win = sum(t.get("pnl", 0) for t in all_closed if t.get("pnl", 0) > 0) / max(wins, 1)
    avg_loss = sum(t.get("pnl", 0) for t in all_closed if t.get("pnl", 0) <= 0) / max(losses, 1)
    print(f"  TOTAL:  {len(all_closed):2d}T  {wins}W/{losses}L  WR={wins/len(all_closed)*100:.0f}%  PnL=Rs{pnl:>+9,.0f}")
    print(f"  Avg Win=Rs{avg_win:,.0f}  Avg Loss=Rs{avg_loss:,.0f}  RR={abs(avg_win/avg_loss) if avg_loss else 0:.2f}")

    # OI analysis
    oi_conflict = []
    oi_aligned = []
    oi_na = []
    for t in all_closed:
        oi = t.get("oi_signal", "")
        d = t.get("direction", "")
        if not oi or oi in ("NEUTRAL", "NO_FUTURES", "ERROR", "N/A", ""):
            oi_na.append(t)
            continue
        conflict = (d == "BUY" and oi in ("SHORT_BUILDUP", "LONG_UNWINDING")) or \
                   (d == "SELL" and oi in ("LONG_BUILDUP", "SHORT_COVERING"))
        if conflict:
            oi_conflict.append(t)
        else:
            oi_aligned.append(t)

    print(f"\n  OI BREAKDOWN:")
    if oi_conflict:
        cw = len([t for t in oi_conflict if t.get("pnl", 0) > 0])
        cp = sum(t.get("pnl", 0) for t in oi_conflict)
        print(f"    CONFLICT: {len(oi_conflict)}T {cw}W/{len(oi_conflict)-cw}L WR={cw/len(oi_conflict)*100:.0f}% PnL=Rs{cp:>+9,.0f}")
        for t in oi_conflict:
            sym = t.get("symbol", "?").replace("NSE:", "")
            print(f"      {sym}: {t.get('direction')} vs OI={t.get('oi_signal')} -> Rs{t.get('pnl', 0):>+,.0f}")
    if oi_aligned:
        aw = len([t for t in oi_aligned if t.get("pnl", 0) > 0])
        ap = sum(t.get("pnl", 0) for t in oi_aligned)
        print(f"    ALIGNED:  {len(oi_aligned)}T {aw}W/{len(oi_aligned)-aw}L WR={aw/len(oi_aligned)*100:.0f}% PnL=Rs{ap:>+9,.0f}")
    if oi_na:
        nw = len([t for t in oi_na if t.get("pnl", 0) > 0])
        np_ = sum(t.get("pnl", 0) for t in oi_na)
        print(f"    NO OI:    {len(oi_na)}T {nw}W/{len(oi_na)-nw}L PnL=Rs{np_:>+9,.0f}")

    # Pipeline breakdown
    print(f"\n  PIPELINE BREAKDOWN:")
    elite = [t for t in all_closed if t.get("entry_type", "") in ("ELITE_AUTO", "ELITE")]
    watcher = [t for t in all_closed if "WATCHER" in t.get("entry_type", "").upper() or "TICKER" in t.get("entry_type", "").upper()]
    other = [t for t in all_closed if t not in elite and t not in watcher]
    for label, group in [("ELITE", elite), ("WATCHER", watcher), ("OTHER", other)]:
        if group:
            gw = len([t for t in group if t.get("pnl", 0) > 0])
            gp = sum(t.get("pnl", 0) for t in group)
            print(f"    {label:8s}: {len(group)}T {gw}W/{len(group)-gw}L PnL=Rs{gp:>+9,.0f}")
print("=" * 60)
