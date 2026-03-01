"""Inspect Dhan instrument master CSV."""
import csv
from collections import Counter

exch = Counter()
inst = Counter()
sbin_rows = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        exch[row.get('SEM_EXM_EXCH_ID', '')] += 1
        inst[row.get('SEM_INSTRUMENT_NAME', '')] += 1
        sym = row.get('SM_SYMBOL_NAME', '').strip()
        instr = row.get('SEM_INSTRUMENT_NAME', '')
        if sym == 'SBIN' and 'FUT' in instr:
            sbin_rows.append(row)

print('Exchanges:', dict(exch))
print('Top instruments:', dict(inst.most_common(15)))
print(f'\nSBIN FUT rows: {len(sbin_rows)}')
for r in sbin_rows[:5]:
    e = r.get('SEM_EXM_EXCH_ID', '')
    i = r.get('SEM_INSTRUMENT_NAME', '')
    s = r.get('SEM_TRADING_SYMBOL', '')
    sid = r.get('SEM_SMST_SECURITY_ID', '')
    exp = r.get('SEM_EXPIRY_DATE', '')
    print(f'  exch={e} inst={i} sym={s} ID={sid} exp={exp}')

# Also check for any NSE FNO contracts
print('\nNSE FNO contracts sample:')
nse_fno = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        seg = row.get('SEM_SEGMENT', '')
        exch_id = row.get('SEM_EXM_EXCH_ID', '')
        instr = row.get('SEM_INSTRUMENT_NAME', '')
        if exch_id == 'NSE' and 'FUT' in instr and 'STK' in instr:
            nse_fno.append(row)
            if len(nse_fno) >= 5:
                break

for r in nse_fno:
    print(f"  {r.get('SEM_TRADING_SYMBOL', '')} seg={r.get('SEM_SEGMENT', '')} inst={r.get('SEM_INSTRUMENT_NAME', '')} ID={r.get('SEM_SMST_SECURITY_ID', '')}")

# Check segment values
seg_counts = Counter()
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        seg_counts[row.get('SEM_SEGMENT', '')] += 1
print(f'\nSegments: {dict(seg_counts)}')
