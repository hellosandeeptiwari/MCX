"""Show active trades with full entry basis (reads from SQLite)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from state_db import get_state_db
from datetime import date

db = get_state_db()
today = str(date.today())
trades, pnl, cap = db.load_active_trades(today)
print(f"Date: {today} | Capital: Rs{cap:,.0f} | Realized PnL: Rs{pnl:,.0f}")
print(f"Active trades: {len(trades)}")

for t in trades:
    if t.get('status','') != 'OPEN':
        continue
    sym = t.get('symbol','?')
    underlying = t.get('underlying','?')
    direction = t.get('direction','?')
    opt_type = t.get('option_type','?')
    strike = t.get('strike',0)
    entry = t.get('avg_price',0)
    sl = t.get('stop_loss',0)
    tgt = t.get('target',0)
    score = t.get('entry_score',0)
    tier = t.get('score_tier','?')
    setup = t.get('setup_type','?')
    strategy = t.get('strategy_type','?')
    smart = t.get('smart_score','N/A')
    p_sc = t.get('p_score','N/A')
    dr = t.get('dr_score','N/A')
    ml_prob = t.get('ml_move_prob','N/A')
    ml_conf = t.get('ml_confidence','N/A')
    ml_dir = t.get('ml_scored_direction','N/A')
    lots = t.get('lots',0)
    premium = t.get('total_premium',0)
    rationale = (t.get('rationale','') or '')[:300]
    ts = t.get('timestamp','?')
    sector = t.get('sector','?')
    delta = t.get('delta','?')
    theta = t.get('theta','?')
    iv = t.get('iv','?')
    expiry = t.get('expiry','?')
    meta = t.get('entry_metadata',{}) or {}
    
    print(f"\n{'='*65}")
    print(f"  SYMBOL: {sym}")
    print(f"  Underlying: {underlying} | Sector: {sector}")
    print(f"  Direction: {direction} | {opt_type} {strike} | Expiry: {expiry}")
    print(f"  Entry: Rs{entry} | SL: Rs{sl:.2f} | Target: Rs{tgt:.2f}")
    print(f"  Lots: {lots} | Premium: Rs{premium:.0f}")
    print(f"  Greeks: delta={delta}, theta={theta}, IV={iv}")
    print(f"  Setup: {setup} | Strategy: {strategy}")
    print(f"  ENTRY SCORE: {score} | Tier: {tier}")
    print(f"  ML Scores: smart={smart}, p_score={p_sc}, dr={dr}")
    print(f"  ML: move_prob={ml_prob}, conf={ml_conf}, direction={ml_dir}")
    print(f"  Entry Time: {ts}")
    print(f"  Rationale: {rationale}")
    if meta:
        print(f"  --- Entry Metadata ---")
        for k in ['candle_pattern','volume_signal','sector_score','regime','entry_score',
                   'score_tier','gate_results','trade_id','strategy_type','sizing_rationale']:
            v = meta.get(k)
            if v:
                print(f"    {k}: {str(v)[:200]}")

# Also show today's ledger entries
import glob, os
today = '2026-02-25'
ledger_files = glob.glob(f'trade_ledger_{today}*.json') + glob.glob(f'logs/trade_ledger_{today}*.json')
for lf in ledger_files:
    print(f"\n{'='*65}")
    print(f"LEDGER: {lf}")
    ld = json.load(open(lf))
    entries = [e for e in ld if e.get('event') == 'ENTRY']
    exits = [e for e in ld if e.get('event') == 'EXIT']
    print(f"  Entries: {len(entries)} | Exits: {len(exits)}")
    for e in entries:
        print(f"  ENTRY: {e.get('symbol')} | score={e.get('entry_score')} | setup={e.get('setup_type')} | {e.get('timestamp','')[:19]}")
    for e in exits:
        pnl = e.get('pnl', 0)
        print(f"  EXIT: {e.get('symbol')} | PnL=Rs{pnl:,.0f} | reason={e.get('exit_reason','')} | {e.get('timestamp','')[:19]}")
