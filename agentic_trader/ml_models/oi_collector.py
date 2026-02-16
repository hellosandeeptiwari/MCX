"""
OI SNAPSHOT COLLECTOR — Collect option-chain OI data for ML training

Saves periodic OI snapshots per symbol to parquet files.
These snapshots are used by the trainer to add OI context features
to each 5-min candle during model training.

Data stored per snapshot:
    timestamp, symbol, pcr_oi, pcr_oi_change, buildup_strength,
    spot_vs_max_pain, iv_skew, atm_iv, atm_delta, call_resistance_dist

Storage: ml_models/data/oi_snapshots/{SYMBOL}.parquet

Usage:
    # During Titan's scan cycle (autonomous_trader.py):
    from ml_models.oi_collector import OICollector
    collector = OICollector()
    collector.collect(symbol, oi_data_dict)   # oi_data from DhanOIFetcher
    
    # During training:
    from ml_models.oi_collector import load_oi_snapshots
    oi_df = load_oi_snapshots("SBIN")  # Returns DataFrame for merge_asof
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger('oi_collector')

OI_DATA_DIR = Path(__file__).parent / "data" / "oi_snapshots"


class OICollector:
    """Collect and store OI snapshots during market hours."""

    def __init__(self):
        OI_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._buffer = {}  # symbol -> list of snapshot dicts

    def collect(self, symbol: str, oi_data: dict) -> bool:
        """Extract ML-relevant OI features from a DhanHQ/NSE snapshot and buffer.
        
        Args:
            symbol: Stock symbol (e.g., "SBIN")
            oi_data: Raw output from DhanOIFetcher.fetch() or to_flow_analyzer_format()
                Expected keys: pcr_oi, pcr_oi_change, oi_buildup_strength,
                    spot_price, max_pain, iv_skew, atm_greeks, call_resistance, strikes
        
        Returns:
            True if snapshot was stored successfully
        """
        try:
            if not oi_data or not oi_data.get('spot_price'):
                return False

            spot = oi_data.get('spot_price', 0)
            max_pain = oi_data.get('max_pain', 0)
            call_resistance = oi_data.get('call_resistance', 0)
            atm_greeks = oi_data.get('atm_greeks', {})

            # Compute OI features for ML
            # 1. PCR (put-call ratio by OI)
            pcr_oi = oi_data.get('pcr_oi', 1.0)

            # 2. PCR change from previous close
            pcr_oi_change = oi_data.get('pcr_oi_change', 0.0)

            # 3. Buildup strength: -1 (strong short/bearish) to +1 (strong long/bullish)
            buildup_signal = oi_data.get('oi_buildup_signal', 'NEUTRAL')
            raw_strength = oi_data.get('oi_buildup_strength', 0.0)
            if buildup_signal in ('SHORT_BUILDUP', 'LONG_UNWINDING'):
                buildup_strength = -abs(raw_strength)
            elif buildup_signal in ('LONG_BUILDUP', 'SHORT_COVERING'):
                buildup_strength = abs(raw_strength)
            else:
                buildup_strength = 0.0

            # 4. Spot vs max pain (%)
            spot_vs_max_pain = 0.0
            if spot > 0 and max_pain > 0:
                spot_vs_max_pain = (spot - max_pain) / spot * 100

            # 5. IV skew (ATM put IV - call IV, positive = bearish fear)
            iv_skew = oi_data.get('iv_skew', 0.0)

            # 6. ATM average IV
            ce_iv = atm_greeks.get('ce_iv', 0)
            pe_iv = atm_greeks.get('pe_iv', 0)
            atm_iv = (ce_iv + pe_iv) / 2.0 if (ce_iv > 0 and pe_iv > 0) else max(ce_iv, pe_iv)

            # 7. ATM call delta (~ 0.5 at ATM, market direction expectation)
            atm_delta = abs(atm_greeks.get('ce_delta', 0.5))

            # 8. Call resistance distance (% from spot to highest-OI call strike)
            call_resistance_dist = 0.0
            if spot > 0 and call_resistance > 0:
                call_resistance_dist = (call_resistance - spot) / spot * 100

            # Build timestamp
            ts_str = oi_data.get('timestamp')
            if ts_str:
                try:
                    ts = pd.to_datetime(ts_str)
                except Exception:
                    ts = datetime.now()
            else:
                ts = datetime.now()

            snapshot = {
                'timestamp': ts,
                'symbol': symbol.upper(),
                'pcr_oi': round(float(pcr_oi), 4),
                'pcr_oi_change': round(float(pcr_oi_change), 4),
                'buildup_strength': round(float(buildup_strength), 4),
                'spot_vs_max_pain': round(float(spot_vs_max_pain), 4),
                'iv_skew': round(float(iv_skew), 4),
                'atm_iv': round(float(atm_iv), 4),
                'atm_delta': round(float(atm_delta), 4),
                'call_resistance_dist': round(float(call_resistance_dist), 4),
            }

            # Buffer in memory
            sym = symbol.upper()
            if sym not in self._buffer:
                self._buffer[sym] = []
            self._buffer[sym].append(snapshot)

            # Flush every 10 snapshots per symbol to reduce disk I/O
            if len(self._buffer[sym]) >= 10:
                self._flush(sym)

            return True

        except Exception as e:
            logger.warning(f"OICollector: Failed to collect for {symbol}: {e}")
            return False

    def _flush(self, symbol: str):
        """Flush buffered snapshots to parquet."""
        try:
            sym = symbol.upper()
            if sym not in self._buffer or not self._buffer[sym]:
                return

            new_df = pd.DataFrame(self._buffer[sym])
            fpath = OI_DATA_DIR / f"{sym}.parquet"

            if fpath.exists():
                existing = pd.read_parquet(fpath)
                combined = pd.concat([existing, new_df], ignore_index=True)
                # Deduplicate by timestamp (within 30-second window)
                combined = combined.sort_values('timestamp')
                combined['_ts_round'] = combined['timestamp'].dt.round('30s')
                combined = combined.drop_duplicates(subset=['_ts_round'], keep='last')
                combined = combined.drop(columns=['_ts_round'])
            else:
                combined = new_df

            combined.to_parquet(fpath, index=False)
            self._buffer[sym] = []
            logger.debug(f"OICollector: Flushed {len(new_df)} snapshots for {sym} ({len(combined)} total)")

        except Exception as e:
            logger.warning(f"OICollector: Flush failed for {symbol}: {e}")

    def flush_all(self):
        """Flush all buffered snapshots (call at EOD or shutdown)."""
        for sym in list(self._buffer.keys()):
            self._flush(sym)

    def get_snapshot_count(self, symbol: str) -> int:
        """Get total number of stored snapshots for a symbol."""
        fpath = OI_DATA_DIR / f"{symbol.upper()}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            return len(df)
        return 0


def load_oi_snapshots(symbol: str) -> Optional[pd.DataFrame]:
    """Load OI snapshots for a symbol (used by trainer/predictor).
    
    Returns:
        DataFrame with columns: timestamp, pcr_oi, pcr_oi_change,
            buildup_strength, spot_vs_max_pain, iv_skew, atm_iv,
            atm_delta, call_resistance_dist
        Or None if no data exists.
    """
    fpath = OI_DATA_DIR / f"{symbol.upper()}.parquet"
    if not fpath.exists():
        return None

    try:
        df = pd.read_parquet(fpath)
        if len(df) == 0:
            return None
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception:
        return None


def load_all_oi_snapshots(symbols: list) -> dict:
    """Load OI snapshots for multiple symbols.
    
    Returns:
        {symbol: oi_df, ...} — only symbols with data are included
    """
    result = {}
    for sym in symbols:
        df = load_oi_snapshots(sym)
        if df is not None:
            result[sym] = df
    return result


if __name__ == '__main__':
    # Quick test: check available OI data
    if OI_DATA_DIR.exists():
        files = list(OI_DATA_DIR.glob("*.parquet"))
        if files:
            print(f"OI snapshots stored for {len(files)} symbols:")
            for f in sorted(files):
                df = pd.read_parquet(f)
                days = df['timestamp'].dt.date.nunique() if len(df) > 0 else 0
                print(f"  {f.stem}: {len(df)} snapshots, {days} days")
        else:
            print("No OI snapshots collected yet.")
            print(f"Directory: {OI_DATA_DIR}")
            print("Run Titan with DhanHQ OI enabled to start collecting.")
    else:
        print(f"OI data directory not yet created: {OI_DATA_DIR}")
        print("It will be created when Titan starts collecting OI data.")
