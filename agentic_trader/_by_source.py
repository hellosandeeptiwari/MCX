#!/usr/bin/env python3
"""Break down all March 5 trades by source AND exit type."""
import json
from collections import defaultdict

f = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-05.jsonl'

exits = []
for line in open(f):
    try:
        t = json.loads(line.strip())
        if t.get('event') == 'EXIT':
            exits.append(t)
    except:
        pass

# Categorize exit type
def exit_cat(t):
    et = str(t.get('exit_type', '') or t.get('exit_reason', ''))
    if 'MANUAL' in et.upper(): return 'MANUAL'
    if 'IV_CRUSH' in et.upper() or 'IV CRUSH' in et.upper(): return 'IV_CRUSH'
    if 'NEVER_SHOW' in et.upper() or 'NEVER SHOW' in et.upper(): return 'NEVER_SHOWED'
    if 'SL' in et.upper() or 'STOP' in et.upper(): return 'STOP_LOSS'
    if 'TARGET' in et.upper(): return 'TARGET'
    if 'TRAIL' in et.upper(): return 'TRAILING'
    if 'DEBIT_SPREAD' in et.upper() or 'TIME_EXIT' in et.upper(): return 'SPREAD_TIME_EXIT'
    if 'SESSION_CUTOFF' in et.upper() or 'CUTOFF' in et.upper(): return 'SESSION_CUTOFF'
    if 'GREEKS' in et.upper(): return 'GREEKS_EXIT'
    if 'EOD' in et.upper() or 'ORPHAN' in et.upper(): return 'EOD_ORPHAN'
    return et[:30]

sources = ['TEST_XGB', 'WATCHER', 'GMM_SNIPER', 'VWAP_TREND', 'SNIPER_PCR_EXTREME', 'GPT', 'MANUAL', 'TEST_GMM', 'ELITE']

for src in sources:
    trades = [t for t in exits if t.get('source') == src]
    if not trades:
        continue
    
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    wins = len([t for t in trades if t.get('pnl', 0) >= 0])
    
    print(f"{'='*80}")
    print(f"SOURCE: {src} | {len(trades)} trades | {wins}W/{len(trades)-wins}L | Rs {total_pnl:+,.0f}")
    print(f"{'='*80}")
    
    # Group by exit type
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[exit_cat(t)].append(t)
    
    for ecat in sorted(by_exit.keys()):
        etrades = by_exit[ecat]
        epnl = sum(t.get('pnl', 0) for t in etrades)
        print(f"\n  [{ecat}] ({len(etrades)} trades, Rs {epnl:+,.0f}):")
        for t in etrades:
            sym = t.get('underlying', '?').replace('NSE:', '')
            pnl = t.get('pnl', 0)
            m = '+' if pnl >= 0 else '-'
            opt = t.get('symbol', '').split('|')[0]
            # option type
            if 'CE' in opt and 'PE' not in opt:
                ot = 'CE'
            elif 'PE' in opt:
                ot = 'PE'
            else:
                ot = '??'
            
            entry_px = t.get('entry_price', 0)
            exit_px = t.get('exit_price', 0)
            pnl_pct = t.get('pnl_pct', 0)
            hold = t.get('hold_minutes', 0)
            qty = t.get('quantity', 0)
            
            # Extract IV info from entry
            iv = t.get('iv', 0)
            delta = t.get('delta', 0)
            
            # Exit reason detail
            reason = str(t.get('exit_reason', t.get('exit_type', '')))[:60]
            
            entry_t = t.get('entry_time', '')
            if entry_t: entry_t = entry_t[11:16]
            exit_t = t.get('ts', '')[11:16]
            
            print(f"    {m} {sym:15s} {ot} {entry_t}->{exit_t} ({hold:>3}m) | "
                  f"qty={qty:>5} | {entry_px:.2f}->{exit_px:.2f} | "
                  f"Rs {pnl:>+8,.0f} ({pnl_pct:>+5.1f}%) | {reason}")
    
    print()
