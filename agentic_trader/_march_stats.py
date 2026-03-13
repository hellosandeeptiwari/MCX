import json, os
from collections import defaultdict

pp = json.load(open("/home/ubuntu/titan/agentic_trader/paper_positions.json"))
closed = [p for p in pp if p.get("status") == "CLOSED" and "2026-03" in p.get("entry_time", "")]
openp = [p for p in pp if p.get("status") == "OPEN"]

print(f"Total positions: {len(pp)}, Open: {len(openp)}, March closed: {len(closed)}")

if closed:
    wins = [p for p in closed if p.get("pnl", 0) > 0]
    losses = [p for p in closed if p.get("pnl", 0) <= 0]
    pnl = sum(p.get("pnl", 0) for p in closed)
    avg_w = sum(p.get("pnl", 0) for p in wins) / max(len(wins), 1)
    avg_l = sum(p.get("pnl", 0) for p in losses) / max(len(losses), 1)
    wr = len(wins) / len(closed) * 100
    rr = abs(avg_w / avg_l) if avg_l else 0
    print(f"\nMARCH: {len(closed)}T {len(wins)}W/{len(losses)}L WR={wr:.0f}% PnL=Rs{pnl:+,.0f}")
    print(f"Avg Win=Rs{avg_w:+,.0f}  Avg Loss=Rs{avg_l:+,.0f}  RR={rr:.2f}")

    # By day
    by_day = defaultdict(list)
    for p in closed:
        day = p.get("entry_time", "")[:10]
        by_day[day].append(p)
    print("\nDAILY:")
    cum_pnl = 0
    for day in sorted(by_day):
        trades = by_day[day]
        w = len([t for t in trades if t.get("pnl", 0) > 0])
        l = len(trades) - w
        dp = sum(t.get("pnl", 0) for t in trades)
        cum_pnl += dp
        print(f"  {day}: {len(trades):2d}T {w}W/{l}L PnL=Rs{dp:>+9,.0f}  cum=Rs{cum_pnl:>+9,.0f}")

    # OI analysis
    oi_conflict = []
    oi_aligned = []
    oi_none = []
    for p in closed:
        oi = p.get("oi_signal", "")
        d = p.get("direction", "")
        if not oi or oi in ("NEUTRAL", "NO_FUTURES", "ERROR", "N/A", ""):
            oi_none.append(p)
            continue
        conflict = (d == "BUY" and oi in ("SHORT_BUILDUP", "LONG_UNWINDING")) or \
                   (d == "SELL" and oi in ("LONG_BUILDUP", "SHORT_COVERING"))
        if conflict:
            oi_conflict.append(p)
        else:
            oi_aligned.append(p)

    print(f"\nOI ANALYSIS (trades with OI data: {len(oi_conflict)+len(oi_aligned)}):")
    if oi_conflict:
        cw = len([t for t in oi_conflict if t.get("pnl", 0) > 0])
        cl = len(oi_conflict) - cw
        cp = sum(t.get("pnl", 0) for t in oi_conflict)
        print(f"  CONFLICT: {len(oi_conflict)}T {cw}W/{cl}L WR={cw/len(oi_conflict)*100:.0f}% PnL=Rs{cp:>+9,.0f}")
        for t in oi_conflict:
            s = t.get("symbol", "").replace("NFO:", "")
            print(f"    {s}: {t.get('direction')} vs OI={t.get('oi_signal')} PnL=Rs{t.get('pnl',0):>+,.0f}")
    if oi_aligned:
        aw = len([t for t in oi_aligned if t.get("pnl", 0) > 0])
        al = len(oi_aligned) - aw
        ap = sum(t.get("pnl", 0) for t in oi_aligned)
        print(f"  ALIGNED:  {len(oi_aligned)}T {aw}W/{al}L WR={aw/len(oi_aligned)*100:.0f}% PnL=Rs{ap:>+9,.0f}")
    if oi_none:
        nw = len([t for t in oi_none if t.get("pnl", 0) > 0])
        nl = len(oi_none) - nw
        np_ = sum(t.get("pnl", 0) for t in oi_none)
        print(f"  NO OI:    {len(oi_none)}T {nw}W/{nl}L PnL=Rs{np_:>+9,.0f}")

    # What-if: if we had flipped all OI conflicts
    if oi_conflict:
        hypothetical_saved = sum(-t.get("pnl", 0) for t in oi_conflict if t.get("pnl", 0) < 0)
        hypothetical_lost = sum(t.get("pnl", 0) for t in oi_conflict if t.get("pnl", 0) > 0)
        print(f"\n  WHAT-IF OI FLIP WAS ACTIVE EARLIER:")
        print(f"    Would have avoided Rs{hypothetical_saved:,.0f} in losses from conflict trades")
        print(f"    Would have missed Rs{hypothetical_lost:,.0f} in wins from conflict trades (flipped other way)")

# Today open
today_open = [p for p in openp if "2026-03-12" in p.get("entry_time", "")]
if today_open:
    print(f"\nTODAY OPEN ({len(today_open)}):")
    for p in today_open:
        s = p.get("symbol", "").replace("NFO:", "")
        pnl = p.get("pnl", p.get("unrealized_pnl", 0))
        oi = p.get("oi_signal", "N/A")
        d = p.get("direction", "?")
        entry = p.get("entry_price", 0)
        ltp = p.get("ltp", p.get("current_price", 0))
        print(f"  {s}: {d} OI={oi} entry={entry} ltp={ltp} PnL=Rs{pnl:>+,.0f}")
