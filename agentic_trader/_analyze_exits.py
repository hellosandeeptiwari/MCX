"""Quick analysis of TIME_STOP and OPTION_SPEED_GATE exits"""
import json, statistics

trades = json.load(open('trade_history.json'))

# =====================================================
# 1. TIME_STOP ANALYSIS
# =====================================================
print("=" * 75)
print("  TIME_STOP TRADES — Deep Dive")
print("=" * 75)

ts_trades = []
for t in trades:
    ed = t.get('exit_detail', {}) or {}
    result = t.get('result', '')
    if ed.get('exit_type') == 'TIME_STOP' or result == 'TIME_STOP':
        ts_trades.append(t)

total_pnl = sum(t.get('pnl', 0) or 0 for t in ts_trades)
print(f"Total: {len(ts_trades)} trades, PnL: {total_pnl:+,.0f}")

winners = [t for t in ts_trades if (t.get('pnl', 0) or 0) > 0]
losers = [t for t in ts_trades if (t.get('pnl', 0) or 0) < 0]
flat = [t for t in ts_trades if (t.get('pnl', 0) or 0) == 0]
print(f"Winners: {len(winners)}, Losers: {len(losers)}, Flat: {len(flat)}")
if winners:
    print(f"  Winner PnL: {sum(t.get('pnl', 0) for t in winners):+,.0f}")
if losers:
    print(f"  Loser PnL:  {sum(t.get('pnl', 0) for t in losers):+,.0f}")
print()

header = f"{'Symbol':<25s} {'PnL':>10s} {'R':>6s} {'MaxR':>6s} {'Candles':>8s} {'Type':>12s} {'Score':>6s}"
print(header)
print("-" * 75)
for t in sorted(ts_trades, key=lambda x: x.get('pnl', 0)):
    ed = t.get('exit_detail', {}) or {}
    sym = (t.get('symbol', '') or t.get('underlying', ''))[:24]
    pnl = t.get('pnl', 0) or 0
    r = ed.get('r_multiple_achieved', 0) or 0
    maxr = ed.get('max_favorable_excursion', 0) or 0
    candles = ed.get('candles_held', 0) or 0
    stype = (t.get('strategy_type', 'EQ') or 'EQ')[:12]
    score = t.get('entry_score', 0) or 0
    print(f"{sym:<25s} {pnl:>+10,.0f} {r:>+6.2f} {maxr:>6.2f} {candles:>8d} {stype:>12s} {score:>6.0f}")

print()
r_at_stop = [(t.get('exit_detail', {}) or {}).get('r_multiple_achieved', 0) or 0 for t in ts_trades]
if r_at_stop:
    print(f"R at time-stop — Mean: {statistics.mean(r_at_stop):+.3f}, Median: {statistics.median(r_at_stop):+.3f}")
    neg_r = [r for r in r_at_stop if r < 0]
    pos_r = [r for r in r_at_stop if r > 0]
    print(f"  Trades at negative R: {len(neg_r)}/{len(r_at_stop)} → time-stop SAVED these from full SL")
    if neg_r:
        print(f"  Their avg R: {statistics.mean(neg_r):+.3f}")
    if pos_r:
        print(f"  Trades at positive R: {len(pos_r)}/{len(r_at_stop)} → time-stop KILLED these despite being profitable")
        print(f"  Their avg R: {statistics.mean(pos_r):+.3f}")

# Estimate savings: if time-stopped trades had gone to full SL (-1R), how much worse?
# Each trade's risk = avg_price * quantity * SL%
saved_estimate = 0
for t in ts_trades:
    ed = t.get('exit_detail', {}) or {}
    r = ed.get('r_multiple_achieved', 0) or 0
    pnl = t.get('pnl', 0) or 0
    if r > -1.0 and pnl < 0:
        # This trade exited at partial loss instead of full SL
        # Rough estimate: at full SL, loss would be pnl * (1/abs(r)) if r<0, or the initial risk
        risk = t.get('risk_amount', 0) or t.get('position_risk', 0) or 0
        if risk > 0:
            full_sl_loss = -risk
            saved = full_sl_loss - pnl
            saved_estimate += saved

print(f"\n  Estimated additional loss if these went to full SL: ~{saved_estimate:+,.0f}")

# =====================================================
# 2. OPTION_SPEED_GATE ANALYSIS
# =====================================================
print("\n" + "=" * 75)
print("  OPTION_SPEED_GATE TRADES — Deep Dive")
print("=" * 75)

sg_trades = []
for t in trades:
    ed = t.get('exit_detail', {}) or {}
    result = t.get('result', '')
    if ed.get('exit_type') == 'OPTION_SPEED_GATE' or result == 'OPTION_SPEED_GATE':
        sg_trades.append(t)

total_pnl_sg = sum(t.get('pnl', 0) or 0 for t in sg_trades)
print(f"Total: {len(sg_trades)} trades, PnL: {total_pnl_sg:+,.0f}")

winners_sg = [t for t in sg_trades if (t.get('pnl', 0) or 0) > 0]
losers_sg = [t for t in sg_trades if (t.get('pnl', 0) or 0) < 0]
print(f"Winners: {len(winners_sg)}, Losers: {len(losers_sg)}")
if winners_sg:
    print(f"  Winner PnL: {sum(t.get('pnl', 0) for t in winners_sg):+,.0f}")
if losers_sg:
    print(f"  Loser PnL:  {sum(t.get('pnl', 0) for t in losers_sg):+,.0f}")
print()

print(header)
print("-" * 75)
for t in sorted(sg_trades, key=lambda x: x.get('pnl', 0)):
    ed = t.get('exit_detail', {}) or {}
    sym = (t.get('symbol', '') or t.get('underlying', ''))[:24]
    pnl = t.get('pnl', 0) or 0
    r = ed.get('r_multiple_achieved', 0) or 0
    maxr = ed.get('max_favorable_excursion', 0) or 0
    candles = ed.get('candles_held', 0) or 0
    stype = (t.get('strategy_type', 'EQ') or 'EQ')[:12]
    score = t.get('entry_score', 0) or 0
    print(f"{sym:<25s} {pnl:>+10,.0f} {r:>+6.2f} {maxr:>6.2f} {candles:>8d} {stype:>12s} {score:>6.0f}")

print()

# Speed gate log analysis — how many passed vs failed?
import os
sg_log = os.path.join(os.path.dirname(__file__), 'speed_gate_log.jsonl')
if os.path.exists(sg_log):
    passes = 0
    exits = 0
    with open(sg_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if 'PASS' in entry.get('reason', ''):
                    passes += 1
                elif 'EXIT' in entry.get('reason', ''):
                    exits += 1
            except:
                pass
    
    total = passes + exits
    if total > 0:
        exit_rate = exits / total * 100
        print(f"Speed Gate Log: {total} evaluations")
        print(f"  PASS: {passes} ({passes/total*100:.1f}%)")
        print(f"  EXIT: {exits} ({exit_rate:.1f}%)")
        print(f"  Current kill rate: {exit_rate:.1f}%")
        
        if exit_rate < 5:
            print(f"  → Gate is LENIENT (kills <5% of options)")
        elif exit_rate < 15:
            print(f"  → Gate is MODERATE")
        else:
            print(f"  → Gate is AGGRESSIVE (kills >{exit_rate:.0f}% of options)")

# Compare: what happened to options that PASSED the speed gate?
print()
print("--- OPTIONS THAT PASSED SPEED GATE ---")
option_trades = [t for t in trades if t.get('strategy_type') == 'NAKED_OPTION']
non_sg_options = [t for t in option_trades if
    (t.get('exit_detail', {}) or {}).get('exit_type') != 'OPTION_SPEED_GATE' and 
    t.get('result') != 'OPTION_SPEED_GATE']
print(f"Options that survived gate: {len(non_sg_options)}")
if non_sg_options:
    survived_pnl = sum(t.get('pnl', 0) or 0 for t in non_sg_options)
    survived_wins = sum(1 for t in non_sg_options if (t.get('pnl', 0) or 0) > 0)
    survived_wr = survived_wins / len(non_sg_options) * 100
    print(f"  PnL: {survived_pnl:+,.0f}, Win Rate: {survived_wr:.1f}%")
    print(f"  Avg PnL/trade: {survived_pnl/len(non_sg_options):+,.0f}")

sg_avg_pnl = total_pnl_sg / len(sg_trades) if sg_trades else 0
surv_avg_pnl = survived_pnl / len(non_sg_options) if non_sg_options else 0
print(f"\n  Speed-gated avg PnL/trade: {sg_avg_pnl:+,.0f}")
print(f"  Survived avg PnL/trade:   {surv_avg_pnl:+,.0f}")

if sg_avg_pnl < surv_avg_pnl:
    print(f"  → Gate IS helping: gated trades averaged {sg_avg_pnl:+,.0f} vs survivors {surv_avg_pnl:+,.0f}")
else:
    print(f"  → Gate is HURTING: gated trades would have done better than survivors!")

# Current config
print("\n" + "=" * 75)
print("  CURRENT SETTINGS")
print("=" * 75)
print(f"  TIME_STOP:          10 candles (50 min), min R: 0.3R")
print(f"  OPTION_SPEED_GATE:  12 candles (60 min), need +3% premium OR +0.10R")
