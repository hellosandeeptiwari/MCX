"""Diagnose OI strength from DhanHQ fetcher directly"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhan_oi_fetcher import get_dhan_oi_fetcher

dhan = get_dhan_oi_fetcher()
print(f"Dhan ready: {dhan.ready}")

test_syms = [
    'NSE:HINDUNILVR', 'NSE:BSE', 'NSE:SIEMENS', 'NSE:MARUTI',
    'NSE:PERSISTENT', 'NSE:RVNL', 'NSE:COLPAL',
]

print(f"\n{'Symbol':<20} {'Signal':<20} {'Strength':>10} {'PartID':<18} {'CallOIChg':>12} {'PutOIChg':>12}")
print("-" * 95)
for sym in test_syms:
    try:
        res = dhan.fetch(sym)
        if not res:
            print(f"{sym:<20} {'NO DATA / None':<20}")
            continue
        sig = res.get('oi_buildup_signal', 'NEUTRAL')
        strength = res.get('oi_buildup_strength', 0.0)
        part = res.get('oi_participant_id', 'UNKNOWN')
        ce_chg = res.get('total_call_oi_change', 0)
        pe_chg = res.get('total_put_oi_change', 0)
        print(f"{sym:<20} {sig:<20} {strength:>10.4f} {part:<18} {ce_chg:>12,} {pe_chg:>12,}")
    except Exception as e:
        print(f"{sym:<20} ERROR: {e}")

# Also check nse_oi_fetcher
print("\n\n=== NSE OI FETCHER ===")
try:
    from nse_oi_fetcher import get_nse_oi_fetcher
    nse = get_nse_oi_fetcher()
    print(f"\n{'Symbol':<20} {'Signal':<20} {'Strength':>10}")
    print("-" * 55)
    for sym in test_syms[:3]:
        try:
            res = nse.fetch(sym)
            if not res:
                print(f"{sym:<20} {'NO DATA / None':<20}")
                continue
            sig = res.get('oi_buildup_signal', 'NEUTRAL')
            strength = res.get('oi_buildup_strength', 0.0)
            print(f"{sym:<20} {sig:<20} {strength:>10.4f}")
        except Exception as e:
            print(f"{sym:<20} ERROR: {e}")
except Exception as e:
    print(f"NSE fetcher error: {e}")
