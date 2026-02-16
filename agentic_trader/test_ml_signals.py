"""Quick ML signal diagnostic."""
import warnings; warnings.filterwarnings('ignore')
import sys, time
sys.path.insert(0, '.')
from ml_models.predictor import MovePredictor
from zerodha_tools import ZerodhaTools
from dotenv import load_dotenv
load_dotenv()

pred = MovePredictor()
tools = ZerodhaTools()
time.sleep(3)

test_stocks = [
    'NSE:SBIN','NSE:RELIANCE','NSE:HDFCBANK','NSE:IREDA','NSE:POWERGRID',
    'NSE:GMRAIRPORT','NSE:BSE','NSE:INOXWIND','NSE:AXISBANK','NSE:TATAMOTORS',
    'NSE:ICICIBANK','NSE:ANGELONE','NSE:MCX','NSE:WIPRO'
]
for sym in test_stocks:
    tools._calculate_indicators(sym)

cc = getattr(tools, '_candle_cache', {})
dc = getattr(tools, '_daily_cache', {})

up = 0; down = 0; flat = 0
for sym in test_stocks:
    candles = cc.get(sym)
    daily = dc.get(sym)
    if candles is not None and len(candles) >= 50:
        ml_c = candles; s = 'int'
    elif daily is not None and len(daily) >= 50:
        ml_c = daily; s = 'dly'
    else:
        print(f'{sym}: NO DATA')
        continue
    r = pred.get_titan_signals(ml_c)
    sig = r.get('ml_signal', '?')
    if sig == 'UP': up += 1
    elif sig == 'DOWN': down += 1
    else: flat += 1
    pm = r.get('ml_p_move')
    pu = r.get('ml_p_up_given_move')
    pd = r.get('ml_p_down_given_move')
    boost = r.get('ml_score_boost')
    hint = r.get('ml_direction_hint')
    print(f'{sym}[{s}]: p_move={pm} p_up={pu} p_down={pd} sig={sig} boost={boost} hint={hint}')

print(f'\nSummary: {up} UP | {down} DOWN | {flat} FLAT')
