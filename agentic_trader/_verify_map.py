#!/usr/bin/env python3
from dhan_oi_fetcher import DHAN_SCRIP_MAP, _SYMBOL_ALIASES
fno = [k for k,v in DHAN_SCRIP_MAP.items() if v['segment']=='NSE_FNO']
idx = [k for k,v in DHAN_SCRIP_MAP.items() if v['segment']=='IDX_I']
print(f'F&O stocks: {len(fno)}')
print(f'Index entries: {len(idx)}')
print(f'Aliases: {_SYMBOL_ALIASES}')
for sym in ['KEI','CGPOWER','BSE','MCX','TATAMOTORS','HAL','BAJAJ-AUTO','NAM-INDIA','INFY','SBIN']:
    r = DHAN_SCRIP_MAP.get(sym) or DHAN_SCRIP_MAP.get(_SYMBOL_ALIASES.get(sym,''))
    status = 'OK' if r else 'MISS'
    print(f'  {sym:15s} → {r}  [{status}]')
