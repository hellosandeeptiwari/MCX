#!/usr/bin/env python3
"""Calculate brokerage impact on Friday OI_WATCHER trades: Paper vs Live (Zerodha)."""
import json, re

ledger_file = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-20.jsonl'
entries = {}
exits = {}

with open(ledger_file) as f:
    for line in f:
        rec = json.loads(line.strip())
        oid = rec.get('order_id', '')
        src = rec.get('source', '')
        if src != 'OI_WATCHER':
            continue
        if rec.get('event') == 'ENTRY':
            entries[oid] = rec
        elif rec.get('event') == 'EXIT':
            exits[oid] = rec

print(f'OI_WATCHER trades: {len(exits)}')
print()

total_paper_brok = 0
total_live_brok = 0
total_pnl_raw = 0
trade_count = 0
turnovers = []

total_br = 0; total_stt = 0; total_ec = 0; total_gst = 0; total_sebi = 0; total_stamp = 0

for oid, ex in exits.items():
    en = entries.get(oid)
    if not en:
        continue
    entry_price = en.get('entry_price', 0)
    exit_price = ex.get('exit_price', 0)
    qty = en.get('quantity', 0)
    pnl = ex.get('pnl', 0)

    if entry_price <= 0 or qty <= 0:
        continue

    buy_value = abs(entry_price * qty)
    sell_value = abs(exit_price * qty)
    turnover = buy_value + sell_value

    # Paper mode (current: 0.6% of turnover)
    paper_brok = round(turnover * 0.006, 2)

    # Live mode (Zerodha actual)
    brokerage_per_leg = min(20, turnover * 0.0003)
    brokerage = brokerage_per_leg * 2
    stt = sell_value * 0.000625
    exchange_charges = turnover * 0.00053
    gst = (brokerage + exchange_charges) * 0.18
    sebi = turnover * 0.000001
    stamp = buy_value * 0.00003
    live_brok = round(brokerage + stt + exchange_charges + gst + sebi + stamp, 2)

    total_paper_brok += paper_brok
    total_live_brok += live_brok
    total_pnl_raw += pnl
    trade_count += 1

    total_br += brokerage
    total_stt += stt
    total_ec += exchange_charges
    total_gst += gst
    total_sebi += sebi
    total_stamp += stamp

    sym = en.get('underlying', '').replace('NSE:', '')
    turnovers.append((sym, qty, entry_price, exit_price, pnl, paper_brok, live_brok, turnover))

print(f'Trades with data: {trade_count}')
print(f'Total raw P&L (before brokerage): Rs {total_pnl_raw:,.0f}')
print()
print(f'Paper brokerage (0.6% turnover): Rs {total_paper_brok:,.0f}')
print(f'Live brokerage (Zerodha actual):  Rs {total_live_brok:,.0f}')
print(f'Saving going live: Rs {total_paper_brok - total_live_brok:,.0f}')
print()
print(f'P&L after paper brokerage: Rs {total_pnl_raw - total_paper_brok:,.0f}')
print(f'P&L after live brokerage:  Rs {total_pnl_raw - total_live_brok:,.0f}')
print()

if trade_count > 0:
    avg_paper = total_paper_brok / trade_count
    avg_live = total_live_brok / trade_count
    print(f'Avg paper brokerage/trade: Rs {avg_paper:.1f}')
    print(f'Avg live brokerage/trade:  Rs {avg_live:.1f}')
    print(f'Avg turnover/trade: Rs {sum(t[7] for t in turnovers)/trade_count:,.0f}')
    print()

turnovers.sort(key=lambda x: x[6], reverse=True)
print('Top-15 trades by LIVE brokerage:')
print(f'{"Symbol":<15} {"Qty":>5} {"Entry":>8} {"Exit":>8} {"P&L":>8} {"Paper":>7} {"Live":>7} {"Turnover":>10}')
for s, q, ep, xp, p, pb, lb, tv in turnovers[:15]:
    print(f'{s:<15} {q:>5} {ep:>8.1f} {xp:>8.1f} {p:>8.0f} {pb:>7.0f} {lb:>7.1f} {tv:>10.0f}')

print()
print(f'Min turnover: Rs {min(t[7] for t in turnovers):,.0f}')
print(f'Max turnover: Rs {max(t[7] for t in turnovers):,.0f}')
tvs = sorted(t[7] for t in turnovers)
print(f'Median turnover: Rs {tvs[len(tvs)//2]:,.0f}')

print()
print('Live brokerage component breakdown:')
print(f'  Brokerage (Rs20/leg):  Rs {total_br:>8.0f}  ({total_br/total_live_brok*100:.1f}%)')
print(f'  STT (0.0625% sell):    Rs {total_stt:>8.0f}  ({total_stt/total_live_brok*100:.1f}%)')
print(f'  Exchange charges:      Rs {total_ec:>8.0f}  ({total_ec/total_live_brok*100:.1f}%)')
print(f'  GST (18%):             Rs {total_gst:>8.0f}  ({total_gst/total_live_brok*100:.1f}%)')
print(f'  SEBI charges:          Rs {total_sebi:>8.1f}  ({total_sebi/total_live_brok*100:.1f}%)')
print(f'  Stamp duty:            Rs {total_stamp:>8.0f}  ({total_stamp/total_live_brok*100:.1f}%)')
print(f'  TOTAL:                 Rs {total_live_brok:>8.0f}')

# Impact: How many winning trades become losers after live brokerage?
print()
winners_before = 0
winners_after = 0
flipped = []
for s, q, ep, xp, p, pb, lb, tv in turnovers:
    if p > 0:
        winners_before += 1
        if p - lb > 0:
            winners_after += 1
        else:
            flipped.append((s, p, lb, p - lb))

print(f'Winners before brokerage: {winners_before}')
print(f'Winners after live brokerage: {winners_after}')
print(f'Trades flipped to losers by brokerage: {len(flipped)}')
if flipped:
    print()
    print('Flipped trades (were winners, now losers):')
    for s, p, lb, net in flipped:
        print(f'  {s}: P&L={p:.0f} - brokerage={lb:.1f} = {net:.1f}')

# What-if: With 3 fixes (fewer trades, faster exits)
print()
print('='*60)
print('IMPACT ANALYSIS: Brokerage cost per trade at different premiums')
print('='*60)
# Typical lot sizes and premiums
scenarios = [
    ('Low premium (Rs 50, qty=900)', 50, 45, 900),
    ('Mid premium (Rs 150, qty=400)', 150, 140, 400),
    ('High premium (Rs 300, qty=200)', 300, 280, 200),
    ('Quick scalp win (Rs 100, qty=500)', 100, 108, 500),
    ('Quick scalp loss (Rs 100, qty=500)', 100, 92, 500),
]
print(f'{"Scenario":<45} {"P&L":>7} {"Brok":>6} {"Net":>7} {"Brok%":>6}')
for label, ep, xp, qty in scenarios:
    raw_pnl = (xp - ep) * qty
    bv = abs(ep * qty)
    sv = abs(xp * qty)
    tv = bv + sv
    bpl = min(20, tv * 0.0003)
    br = bpl * 2
    stt_v = sv * 0.000625
    ec_v = tv * 0.00053
    gst_v = (br + ec_v) * 0.18
    sebi_v = tv * 0.000001
    stamp_v = bv * 0.00003
    live_b = br + stt_v + ec_v + gst_v + sebi_v + stamp_v
    net = raw_pnl - live_b
    brok_pct = (live_b / bv) * 100 if bv > 0 else 0
    print(f'{label:<45} {raw_pnl:>7.0f} {live_b:>6.1f} {net:>7.0f} {brok_pct:>5.2f}%')
