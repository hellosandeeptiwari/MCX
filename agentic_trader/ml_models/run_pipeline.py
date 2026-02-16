"""
PIPELINE ORCHESTRATOR — One-command fetch → engineer → train → evaluate

CLI commands:
  # Full pipeline: fetch data + train model
  python -m ml_models.run_pipeline --full
  
  # Fetch data only (needs Kite auth)
  python -m ml_models.run_pipeline --fetch --symbols SBIN HDFCBANK RELIANCE
  
  # Train only (uses cached parquet files)
  python -m ml_models.run_pipeline --train
  
  # Train with custom params
  python -m ml_models.run_pipeline --train --lookahead 6 --threshold 0.5 --test-days 20
  
  # Show model info
  python -m ml_models.run_pipeline --info
  
  # Quick predict on a symbol (test)
  python -m ml_models.run_pipeline --predict SBIN
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_fetch(args):
    """Fetch historical candles from Kite API."""
    from ml_models.data_fetcher import fetch_and_save_all, DEFAULT_SYMBOLS
    
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    days = args.days if hasattr(args, 'days') and args.days else 90
    
    print(f"\n{'='*60}")
    print(f"  DATA FETCH")
    print(f"  Symbols: {len(symbols)}  |  Days: {days}")
    print(f"{'='*60}\n")
    
    summary = fetch_and_save_all(symbols=symbols, days=days)
    
    print(f"\n✅ Fetch complete: {summary['success']}/{summary['total']} symbols")
    if summary.get('failed'):
        print(f"⚠ Failed: {', '.join(summary['failed'])}")


def cmd_train(args):
    """Train XGBoost model on cached data."""
    from ml_models.trainer import quick_train
    
    print(f"\n{'='*60}")
    print(f"  MODEL TRAINING")
    print(f"  Lookahead: {args.lookahead} candles ({args.lookahead * 5} min)")
    print(f"  Threshold: {args.threshold}%")
    print(f"  Test days: {args.test_days}")
    print(f"{'='*60}\n")
    
    result = quick_train(
        symbols=args.symbols,
        lookahead=args.lookahead,
        threshold=args.threshold,
        test_days=args.test_days,
    )
    
    print(f"\n🎯 Training complete!")
    print(f"   Accuracy: {result['accuracy']:.1%}")
    print(f"   Model saved to: {result['model_path']}")
    
    return result


def cmd_info(args):
    """Show info about the latest trained model."""
    models_dir = Path(__file__).parent / "saved_models"
    meta_path = models_dir / "move_predictor_latest_meta.json"
    
    if not meta_path.exists():
        print("❌ No trained model found. Run: python -m ml_models.run_pipeline --train")
        return
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"  MOVE PREDICTOR MODEL INFO")
    print(f"{'='*60}")
    print(f"  Trained at:     {meta.get('timestamp', '?')}")
    print(f"  Train samples:  {meta.get('train_samples', '?'):,}")
    print(f"  Test samples:   {meta.get('test_samples', '?'):,}")
    print(f"  Accuracy:       {meta.get('accuracy', 0):.1%}")
    print(f"  Best iteration: {meta.get('best_iteration', '?')}")
    print(f"  Train time:     {meta.get('train_time_seconds', '?')}s")
    
    print(f"\n  Per-class metrics:")
    for cls in ['BIG_DOWN', 'NO_MOVE', 'BIG_UP']:
        p = meta.get('precision_per_class', {}).get(cls, 0)
        r = meta.get('recall_per_class', {}).get(cls, 0)
        f1 = meta.get('f1_per_class', {}).get(cls, 0)
        print(f"    {cls:>10s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    
    print(f"\n  Top 10 features:")
    feat_imp = meta.get('feature_importance', {})
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, imp in sorted_feats:
        print(f"    {fname:<25s}  {imp:.4f}")
    
    print(f"{'='*60}\n")


def cmd_predict(args):
    """Quick prediction test on a symbol."""
    from ml_models.predictor import MovePredictor
    from ml_models.data_fetcher import load_candles
    
    symbol = args.predict_symbol
    predictor = MovePredictor()
    
    if not predictor.ready:
        print("❌ No trained model. Run: python -m ml_models.run_pipeline --train")
        return
    
    candles = load_candles(symbol)
    if candles is None or candles.empty:
        print(f"❌ No cached candles for {symbol}. Run --fetch first.")
        return
    
    # Use last 100 candles for prediction
    recent = candles.tail(100).copy()
    result = predictor.predict(recent)
    
    print(f"\n{'='*60}")
    print(f"  PREDICTION: {symbol}")
    print(f"  (using last 100 candles ending {recent['date'].iloc[-1]})")
    print(f"{'='*60}")
    
    if result:
        print(f"  Signal:      {result['ml_signal']}")
        print(f"  Confidence:  {result['ml_confidence']:.1%}")
        print(f"  P(UP):       {result['ml_move_prob_up']:.1%}")
        print(f"  P(DOWN):     {result['ml_move_prob_down']:.1%}")
        print(f"  P(NONE):     {result['ml_move_prob_none']:.1%}")
        print(f"  Score boost: {result['ml_score_boost']:+d}")
    else:
        print("  ❌ Prediction failed (insufficient data?)")
    
    print(f"{'='*60}\n")


def cmd_full(args):
    """Full pipeline: fetch → train."""
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE")
    print(f"  Step 1: Fetch data from Kite")
    print(f"  Step 2: Train model")
    print(f"{'='*60}\n")
    
    cmd_fetch(args)
    cmd_train(args)


def cmd_validate(args):
    """Run full validation suite on trained model data."""
    from ml_models.model_validator import run_full_validation
    
    use_synthetic = args.synthetic if hasattr(args, 'synthetic') else False
    folds = args.folds if hasattr(args, 'folds') else 5
    
    run_full_validation(use_synthetic=use_synthetic, n_cv_folds=folds)


def main():
    parser = argparse.ArgumentParser(
        description='Move Predictor ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml_models.run_pipeline --full
  python -m ml_models.run_pipeline --fetch --symbols SBIN HDFCBANK
  python -m ml_models.run_pipeline --train --threshold 0.5
  python -m ml_models.run_pipeline --validate            # Full quant validation
  python -m ml_models.run_pipeline --validate --synthetic # Test on synthetic data
  python -m ml_models.run_pipeline --info
  python -m ml_models.run_pipeline --predict SBIN
        """
    )
    
    # Action flags
    parser.add_argument('--full', action='store_true', help='Run full pipeline (fetch + train)')
    parser.add_argument('--fetch', action='store_true', help='Fetch historical data from Kite')
    parser.add_argument('--train', action='store_true', help='Train model on cached data')
    parser.add_argument('--validate', action='store_true', help='Run full validation suite')
    parser.add_argument('--info', action='store_true', help='Show model info')
    parser.add_argument('--predict', dest='predict_symbol', metavar='SYMBOL', help='Test prediction on symbol')
    
    # Parameters
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--days', type=int, default=90, help='Days of history to fetch (default: 90)')
    parser.add_argument('--lookahead', type=int, default=6, help='Lookahead candles (default: 6)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Move threshold %% (default: 0.5)')
    parser.add_argument('--test-days', type=int, default=20, help='Test period days (default: 20)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (for --validate)')
    parser.add_argument('--folds', type=int, default=5, help='CV folds (for --validate, default: 5)')
    
    args = parser.parse_args()
    
    # Default to --info if no action specified
    if not any([args.full, args.fetch, args.train, args.validate, args.info, args.predict_symbol]):
        parser.print_help()
        return
    
    if args.full:
        cmd_full(args)
    elif args.fetch:
        cmd_fetch(args)
    elif args.train:
        cmd_train(args)
    elif args.validate:
        cmd_validate(args)
    elif args.info:
        cmd_info(args)
    elif args.predict_symbol:
        cmd_predict(args)


if __name__ == '__main__':
    main()
