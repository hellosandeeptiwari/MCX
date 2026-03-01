"""Check actual GMM dual-regime scores from decision log and recent predictions."""
import json, os, sys

# 1. Check scan_decisions.json
f = 'scan_decisions.json'
if os.path.exists(f):
    data = json.load(open(f))
    print(f"=== scan_decisions.json: {len(data)} entries ===\n")
    
    agree = [e for e in data if e.get('outcome') in ('ALL_AGREE', 'GMM_CONTRARIAN')]
    aa = [e for e in agree if e.get('outcome') == 'ALL_AGREE']
    gc = [e for e in agree if e.get('outcome') == 'GMM_CONTRARIAN']
    print(f"ALL_AGREE: {len(aa)}, GMM_CONTRARIAN: {len(gc)}")
    
    for e in agree[-10:]:
        print(f"  {e.get('time','?')} | {e.get('symbol','?'):15s} | {e.get('outcome')} | dir={e.get('direction')} | score={e.get('score')} | {e.get('reason','')}")
    
    # Show placed trades
    placed = [e for e in data if '_PLACED' in str(e.get('outcome', ''))]
    print(f"\nPlaced trades: {len(placed)}")
    for e in placed[-10:]:
        print(f"  {e.get('time','?')} | {e.get('symbol','?'):15s} | {e.get('outcome')} | dir={e.get('direction')} | {e.get('reason','')}")
    
    # Extract confirm scores from reason strings
    import re
    confirm_scores = []
    for e in data:
        reason = e.get('reason', '')
        m = re.search(r'score=(\d+\.\d+)', reason)
        if m:
            confirm_scores.append(float(m.group(1)))
    
    if confirm_scores:
        print(f"\nConfirm scores from reasons: {len(confirm_scores)} entries")
        print(f"  Min: {min(confirm_scores):.4f}")
        print(f"  Max: {max(confirm_scores):.4f}")
        print(f"  Mean: {sum(confirm_scores)/len(confirm_scores):.4f}")
        print(f"  >0.25: {sum(1 for s in confirm_scores if s >= 0.25)}")
        print(f"  >0.30: {sum(1 for s in confirm_scores if s >= 0.30)}")
        print(f"  >0.40: {sum(1 for s in confirm_scores if s >= 0.40)}")
else:
    print(f"{f} not found")

# 2. Run predictor on recent data to get live dual-regime scores
print("\n\n=== LIVE DUAL-REGIME SCORES (current models) ===\n")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ml_models.down_risk_detector import DownRiskDetector
    from ml_models.feature_engineering import build_features_for_symbols
    import pandas as pd
    
    det = DownRiskDetector()
    if not det.load():
        print("Detector not loaded")
        sys.exit(0)
    
    print(f"UP threshold: {det.thresholds.get('UP', 'N/A')}")
    print(f"DOWN threshold: {det.thresholds.get('DOWN', 'N/A')}")
    
    # Build features for a sample of stocks
    from config import UNIVERSE
    symbols = UNIVERSE.get('stocks', [])[:30]  # first 30
    
    features_df = build_features_for_symbols(symbols, lookback_days=5)
    if features_df is not None and len(features_df) > 0:
        # Get latest row per symbol
        latest = features_df.groupby(level=0).tail(1) if isinstance(features_df.index, pd.MultiIndex) else features_df.tail(30)
        
        print(f"\nFeatures shape: {features_df.shape}, Latest: {latest.shape}")
        
        results = []
        for idx, row in latest.iterrows():
            sym = idx[0] if isinstance(idx, tuple) else idx
            feats = row.values.reshape(1, -1)
            
            up_res = det.predict_single(feats, 'UP')
            down_res = det.predict_single(feats, 'DOWN')
            
            up_score = float(up_res['anomaly_score'][0])
            down_score = float(down_res['anomaly_score'][0])
            up_flag = bool(up_res['down_risk_flag'][0])
            down_flag = bool(down_res['down_risk_flag'][0])
            
            results.append({
                'sym': str(sym),
                'up_score': up_score,
                'down_score': down_score,
                'up_flag': up_flag,
                'down_flag': down_flag,
            })
        
        # Sort by down_score desc (bounce signals for BUY/CE)
        results.sort(key=lambda x: -x['down_score'])
        
        print(f"\n{'Symbol':15s} {'UP_Score':>10s} {'DN_Score':>10s} {'UP_Flag':>8s} {'DN_Flag':>8s}  Action for BUY  Action for SELL")
        print("-" * 95)
        for r in results:
            buy_action = "CONFIRM" if r['down_flag'] and not r['up_flag'] else "OPPOSE" if r['up_flag'] and not r['down_flag'] else "CONFLICT" if r['up_flag'] and r['down_flag'] else "CLEAN(block)"
            sell_action = "CONFIRM" if r['up_flag'] and not r['down_flag'] else "OPPOSE" if r['down_flag'] and not r['up_flag'] else "CONFLICT" if r['up_flag'] and r['down_flag'] else "CLEAN(block)"
            print(f"{r['sym']:15s} {r['up_score']:10.4f} {r['down_score']:10.4f} {str(r['up_flag']):>8s} {str(r['down_flag']):>8s}  {buy_action:15s} {sell_action}")
        
        # Summary
        n = len(results)
        up_flags = sum(1 for r in results if r['up_flag'])
        dn_flags = sum(1 for r in results if r['down_flag'])
        both = sum(1 for r in results if r['up_flag'] and r['down_flag'])
        clean = sum(1 for r in results if not r['up_flag'] and not r['down_flag'])
        buy_confirm = sum(1 for r in results if r['down_flag'] and not r['up_flag'])
        sell_confirm = sum(1 for r in results if r['up_flag'] and not r['down_flag'])
        
        print(f"\n--- SUMMARY ({n} stocks) ---")
        print(f"UP_Flag (crash risk):   {up_flags}/{n} ({100*up_flags/n:.0f}%)")
        print(f"Down_Flag (bounce risk): {dn_flags}/{n} ({100*dn_flags/n:.0f}%)")
        print(f"Both flags (conflict):  {both}/{n} ({100*both/n:.0f}%)")
        print(f"Clean (no edge):        {clean}/{n} ({100*clean/n:.0f}%)")
        print(f"BUY confirmable:        {buy_confirm}/{n} ({100*buy_confirm/n:.0f}%)")
        print(f"SELL confirmable:       {sell_confirm}/{n} ({100*sell_confirm/n:.0f}%)")
        
        # Score distributions for confirmed stocks
        buy_scores = [r['down_score'] for r in results if r['down_flag'] and not r['up_flag']]
        sell_scores = [r['up_score'] for r in results if r['up_flag'] and not r['down_flag']]
        
        if buy_scores:
            print(f"\nBUY confirm down_scores: min={min(buy_scores):.4f} max={max(buy_scores):.4f} mean={sum(buy_scores)/len(buy_scores):.4f}")
            above_25 = sum(1 for s in buy_scores if s >= 0.25)
            print(f"  >=0.25 (min_confirm): {above_25}/{len(buy_scores)}")
        
        if sell_scores:
            print(f"SELL confirm up_scores: min={min(sell_scores):.4f} max={max(sell_scores):.4f} mean={sum(sell_scores)/len(sell_scores):.4f}")
            above_25 = sum(1 for s in sell_scores if s >= 0.25)
            print(f"  >=0.25 (min_confirm): {above_25}/{len(sell_scores)}")
    else:
        print("No features built")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
