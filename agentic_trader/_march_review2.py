import json, glob, os

# Check event types across all March
files = sorted(glob.glob("/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-*.jsonl"))
events = {}
entries = []
exits = []
for f in files:
    for l in open(f):
        if not l.strip(): continue
        t = json.loads(l.strip())
        ev = t.get("event", "?")
        events[ev] = events.get(ev, 0) + 1
        if ev == "ENTRY":
            entries.append(t)
        elif ev == "EXIT":
            exits.append(t)

print("Event types:")
for k, v in sorted(events.items()):
    print(f"  {k}: {v}")

print(f"\nEntries: {len(entries)}, Exits: {len(exits)}")

# Check paper positions file for closed trades
pp_file = "/home/ubuntu/titan/agentic_trader/paper_positions.json"
if os.path.exists(pp_file):
    pp = json.load(open(pp_file))
    closed = [p for p in pp if p.get("status") == "CLOSED"]
    open_p = [p for p in pp if p.get("status") == "OPEN"]
    print(f"\npaper_positions.json: {len(open_p)} OPEN, {len(closed)} CLOSED")
    
    # March closed trades
    mar_closed = [p for p in closed if "2026-03" in p.get("entry_time", "")]
    if mar_closed:
        wins = [p for p in mar_closed if p.get("pnl", 0) > 0]
        losses = [p for p in mar_closed if p.get("pnl", 0) <= 0]
        pnl = sum(p.get("pnl", 0) for p in mar_closed)
        print(f"\nMARCH CLOSED: {len(mar_closed)}T {len(wins)}W/{len(losses)}L PnL=Rs{pnl:,.0f}")
        print(f"Avg Win=Rs{sum(p.get('pnl',0) for p in wins)/max(len(wins),1):,.0f}")
        print(f"Avg Loss=Rs{sum(p.get('pnl',0) for p in losses)/max(len(losses),1):,.0f}")
        
        # Day breakdown
        from collections import defaultdict
        by_day = defaultdict(list)
        for p in mar_closed:
            day = p.get("entry_time", "")[:10]
            by_day[day].append(p)
        print("\nBy Day:")
        for day in sorted(by_day):
            trades = by_day[day]
            w = len([t for t in trades if t.get("pnl",0) > 0])
            l = len(trades) - w
            dp = sum(t.get("pnl",0) for t in trades)
            print(f"  {day}: {len(trades)}T {w}W/{l}L PnL=Rs{dp:>+9,.0f}")
        
        # OI analysis
        oi_conflict = []
        oi_aligned = []
        oi_none = []
        for p in mar_closed:
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
        
        print(f"\nOI ANALYSIS:")
        if oi_conflict:
            cw = len([t for t in oi_conflict if t.get("pnl",0) > 0])
            cp = sum(t.get("pnl",0) for t in oi_conflict)
            print(f"  CONFLICT: {len(oi_conflict)}T {cw}W/{len(oi_conflict)-cw}L PnL=Rs{cp:>+9,.0f}")
            for t in oi_conflict:
                s = t.get("symbol","").replace("NSE:","").replace("NFO:","")
                print(f"    {s}: {t.get('direction')} vs OI={t.get('oi_signal')} PnL=Rs{t.get('pnl',0):>+,.0f}")
        if oi_aligned:
            aw = len([t for t in oi_aligned if t.get("pnl",0) > 0])
            ap = sum(t.get("pnl",0) for t in oi_aligned)
            print(f"  ALIGNED:  {len(oi_aligned)}T {aw}W/{len(oi_aligned)-aw}L PnL=Rs{ap:>+9,.0f}")
        if oi_none:
            nw = len([t for t in oi_none if t.get("pnl",0) > 0])
            np_ = sum(t.get("pnl",0) for t in oi_none)
            print(f"  NO OI:    {len(oi_none)}T {nw}W/{len(oi_none)-nw}L PnL=Rs{np_:>+9,.0f}")
    
    # Today's open trades
    today_open = [p for p in open_p if "2026-03-12" in p.get("entry_time", "")]
    if today_open:
        print(f"\nTODAY OPEN: {len(today_open)} positions")
        for p in today_open:
            s = p.get("symbol","").replace("NFO:","")
            pnl = p.get("pnl", p.get("unrealized_pnl", 0))
            oi = p.get("oi_signal", "N/A")
            d = p.get("direction", "?")
            print(f"  {s}: {d} OI={oi} PnL=Rs{pnl:>+,.0f}")
