"""
Centralized Trade Logger for Titan
===================================
Single source of truth for all trade events: entries, exits, partial exits.
Replaces the need to cross-reference scan_decisions.json, trade_history.json,
active_trades.json, and trade_decisions.log for post-trade analysis.

File: trade_ledger_YYYY-MM-DD.jsonl  (one JSON object per line, append-only)
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, Optional


LEDGER_DIR = os.path.join(os.path.dirname(__file__), 'trade_ledger')


class TradeLedger:
    """Append-only trade event log — one file per day, JSONL format."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._write_lock = threading.Lock()
        os.makedirs(LEDGER_DIR, exist_ok=True)
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_entry(self, *,
                  symbol: str,
                  underlying: str,
                  direction: str,
                  source: str,            # ALL_AGREE, SNIPER, MODEL_TRACKER, ML_OVERRIDE_WGMM, SCORE, ORB
                  smart_score: float = 0,
                  pre_score: float = 0,   # score before DR adjustments
                  final_score: float = 0, # score after DR adjustments  
                  dr_score: float = 0,
                  dr_flag: bool = False,
                  gate_prob: float = 0,
                  gmm_action: str = '',   # BOOST / BLOCK / ALLOW
                  ml_direction: str = '',
                  ml_move_prob: float = 0,
                  ml_confidence: str = '',
                  xgb_disagrees: bool = False,
                  sector: str = '',
                  strategy_type: str = '',  # NAKED_OPTION, CREDIT_SPREAD, etc.
                  score_tier: str = '',     # PREMIUM, STANDARD, BLOCK
                  option_symbol: str = '',
                  strike: float = 0,
                  option_type: str = '',    # CE / PE
                  expiry: str = '',
                  entry_price: float = 0,
                  quantity: int = 0,
                  lots: int = 0,
                  lot_multiplier: float = 1.0,
                  stop_loss: float = 0,
                  target: float = 0,
                  total_premium: float = 0,
                  delta: float = 0,
                  iv: float = 0,
                  rationale: str = '',
                  order_id: str = '',
                  trade_id: str = '',
                  is_sniper: bool = False,
                  extra: Optional[Dict] = None):
        """Log a new trade entry."""
        record = {
            'event': 'ENTRY',
            'ts': datetime.now().isoformat(),
            'symbol': symbol,
            'underlying': underlying,
            'direction': direction,
            'source': source,
            # Scores
            'smart_score': round(smart_score, 1),
            'pre_score': round(pre_score, 1),
            'final_score': round(final_score, 1),
            'score_tier': score_tier,
            # DR / GMM
            'dr_score': round(dr_score, 4),
            'dr_flag': dr_flag,
            'gate_prob': round(gate_prob, 3),
            'gmm_action': gmm_action,
            # ML
            'ml_direction': ml_direction,
            'ml_move_prob': round(ml_move_prob, 3),
            'ml_confidence': ml_confidence,
            'xgb_disagrees': xgb_disagrees,
            # Context
            'sector': sector,
            'strategy_type': strategy_type,
            'is_sniper': is_sniper,
            'lot_multiplier': lot_multiplier,
            # Option details
            'option_symbol': option_symbol,
            'strike': strike,
            'option_type': option_type,
            'expiry': expiry,
            # Sizing
            'entry_price': round(entry_price, 2),
            'quantity': quantity,
            'lots': lots,
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'total_premium': round(total_premium, 2),
            'delta': round(delta, 4),
            'iv': round(iv, 4),
            # IDs
            'rationale': rationale,
            'order_id': order_id,
            'trade_id': trade_id,
        }
        if extra:
            record['extra'] = extra
        self._append(record)

    def log_exit(self, *,
                 symbol: str,
                 underlying: str,
                 direction: str,
                 source: str = '',
                 sector: str = '',
                 exit_type: str,          # QUICK_PROFIT, PARTIAL_PROFIT, TARGET_HIT, STOPLOSS_HIT, TIME_STOP, SESSION_CUTOFF
                 entry_price: float = 0,
                 exit_price: float = 0,
                 quantity: int = 0,
                 pnl: float = 0,
                 pnl_pct: float = 0,
                 # Entry context (carried from entry)
                 smart_score: float = 0,
                 final_score: float = 0,
                 dr_score: float = 0,
                 score_tier: str = '',
                 strategy_type: str = '',
                 is_sniper: bool = False,
                 # Exit quality
                 candles_held: int = 0,
                 r_multiple: float = 0,
                 max_favorable: float = 0,
                 exit_reason: str = '',
                 breakeven_applied: bool = False,
                 trailing_active: bool = False,
                 partial_booked: bool = False,
                 current_sl: float = 0,
                 hold_minutes: int = 0,
                 order_id: str = '',
                 trade_id: str = '',
                 entry_time: str = '',
                 extra: Optional[Dict] = None):
        """Log a trade exit (full or partial)."""
        record = {
            'event': 'EXIT',
            'ts': datetime.now().isoformat(),
            'symbol': symbol,
            'underlying': underlying,
            'direction': direction,
            'source': source,
            'setup': source,  # alias for easier analysis
            'sector': sector,
            'exit_type': exit_type,
            # Prices
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'quantity': quantity,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            # Entry scores (carried)
            'smart_score': round(smart_score, 1),
            'final_score': round(final_score, 1),
            'dr_score': round(dr_score, 4),
            'score_tier': score_tier,
            'strategy_type': strategy_type,
            'is_sniper': is_sniper,
            # Exit quality
            'candles_held': candles_held,
            'r_multiple': round(r_multiple, 3),
            'max_favorable': round(max_favorable, 2),
            'exit_reason': exit_reason,
            'breakeven_applied': breakeven_applied,
            'trailing_active': trailing_active,
            'partial_booked': partial_booked,
            'current_sl': round(current_sl, 2),
            'hold_minutes': hold_minutes,
            # IDs
            'order_id': order_id,
            'trade_id': trade_id,
            'entry_time': entry_time,
        }
        if extra:
            record['extra'] = extra
        self._append(record)

    def log_scan_decision(self, *,
                          symbol: str,
                          score: float,
                          outcome: str,
                          reason: str = '',
                          direction: str = '',
                          cycle: str = '',
                          extra: Optional[Dict] = None):
        """Log a scan/scoring decision (PLACED, SCORED_LOW, CHOP_FILTERED, etc.)."""
        record = {
            'event': 'SCAN',
            'ts': datetime.now().isoformat(),
            'cycle': cycle,
            'symbol': symbol,
            'score': round(score, 1),
            'outcome': outcome,
            'reason': reason,
            'direction': direction,
        }
        if extra:
            record['extra'] = extra
        self._append(record)

    # ------------------------------------------------------------------
    # Read API — for analysis scripts
    # ------------------------------------------------------------------

    def read_day(self, date_str: str = None) -> list:
        """Read all events for a given date (default today). Returns list of dicts."""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        filepath = os.path.join(LEDGER_DIR, f'trade_ledger_{date_str}.jsonl')
        if not os.path.exists(filepath):
            return []
        events = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def get_entries(self, date_str: str = None) -> list:
        """Get all ENTRY events for a day."""
        return [e for e in self.read_day(date_str) if e.get('event') == 'ENTRY']

    def get_exits(self, date_str: str = None) -> list:
        """Get all EXIT events for a day."""
        return [e for e in self.read_day(date_str) if e.get('event') == 'EXIT']

    def get_trades_with_pnl(self, date_str: str = None) -> list:
        """
        Join entries with their exits (by order_id, or underlying+direction+time proximity).
        Returns list of dicts with entry info + aggregated exit PnL.
        Handles partial exits by summing all exit legs for the same entry.
        """
        entries = self.get_entries(date_str)
        exits = self.get_exits(date_str)

        # Group exits by order_id first, then by (underlying, direction)
        from collections import defaultdict
        exit_by_order = defaultdict(list)
        exit_by_sym = defaultdict(list)
        for ex in exits:
            oid = ex.get('order_id', '')
            if oid:
                exit_by_order[oid].append(ex)
            else:
                key = (ex.get('underlying', ''), ex.get('direction', ''))
                exit_by_sym[key].append(ex)

        used_exits = set()
        results = []
        for entry in entries:
            oid = entry.get('order_id', '')
            matched_exits = []

            # Primary match: by order_id
            if oid and oid in exit_by_order:
                matched_exits = exit_by_order[oid]
            else:
                # Fallback: match by underlying + direction + closest time
                key = (entry.get('underlying', ''), entry.get('direction', ''))
                candidates = exit_by_sym.get(key, []) + exit_by_order.get('', [])
                entry_ts = entry.get('ts', '')
                for ex in candidates:
                    ex_id = id(ex)
                    if ex_id in used_exits:
                        continue
                    # Check entry_time proximity (within 5 minutes)
                    try:
                        from datetime import datetime as dt
                        et = dt.fromisoformat(entry_ts)
                        ext = dt.fromisoformat(ex.get('entry_time', '') or ex.get('ts', ''))
                        if abs((et - ext).total_seconds()) < 300:
                            matched_exits.append(ex)
                            used_exits.add(ex_id)
                    except Exception:
                        pass

            total_pnl = sum(ex.get('pnl', 0) for ex in matched_exits)
            exit_types = [ex.get('exit_type', '') for ex in matched_exits]
            final_result = exit_types[-1] if exit_types else 'OPEN'
            total_candles = max((ex.get('candles_held', 0) for ex in matched_exits), default=0)
            best_r = max((ex.get('r_multiple', 0) for ex in matched_exits), default=0)
            hold_mins = max((ex.get('hold_minutes', 0) for ex in matched_exits), default=0)

            results.append({
                **entry,
                'total_pnl': round(total_pnl, 2),
                'exit_count': len(matched_exits),
                'exit_types': exit_types,
                'final_result': final_result,
                'max_candles_held': total_candles,
                'best_r_multiple': round(best_r, 3),
                'hold_minutes': hold_mins,
                'is_winner': total_pnl > 0,
            })
        return results

    def daily_summary(self, date_str: str = None) -> dict:
        """Generate a summary dict for the day."""
        trades = self.get_trades_with_pnl(date_str)
        if not trades:
            return {'date': date_str or datetime.now().strftime('%Y-%m-%d'), 'total_trades': 0}

        total_pnl = sum(t['total_pnl'] for t in trades)
        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]

        # Source breakdown
        from collections import defaultdict
        by_source = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
        for t in trades:
            src = t.get('source', 'UNKNOWN')
            by_source[src]['count'] += 1
            by_source[src]['pnl'] += t['total_pnl']
            if t['is_winner']:
                by_source[src]['wins'] += 1

        return {
            'date': date_str or datetime.now().strftime('%Y-%m-%d'),
            'total_trades': len(trades),
            'wins': len(winners),
            'losses': len(losers),
            'win_rate': round(len(winners) / len(trades) * 100, 1) if trades else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_winner': round(sum(t['total_pnl'] for t in winners) / len(winners), 2) if winners else 0,
            'avg_loser': round(sum(t['total_pnl'] for t in losers) / len(losers), 2) if losers else 0,
            'by_source': dict(by_source),
            'trades': trades,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _today_file(self) -> str:
        return os.path.join(LEDGER_DIR, f'trade_ledger_{datetime.now().strftime("%Y-%m-%d")}.jsonl')

    def _append(self, record: dict):
        """Thread-safe append a single JSON line."""
        try:
            with self._write_lock:
                with open(self._today_file(), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            print(f"⚠️ TradeLedger write error: {e}")


# Module-level singleton accessor
def get_trade_ledger() -> TradeLedger:
    return TradeLedger()


# ======================================================================
# CLI: python -m trade_ledger [date]
# ======================================================================
if __name__ == '__main__':
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else None
    ledger = get_trade_ledger()
    summary = ledger.daily_summary(date)

    if summary.get('total_trades', 0) == 0:
        print(f"No trades found for {summary['date']}")
        sys.exit(0)

    print(f"\n{'='*80}")
    print(f"  TITAN TRADE LEDGER — {summary['date']}")
    print(f"{'='*80}")
    print(f"  Total Trades: {summary['total_trades']} | W:{summary['wins']} L:{summary['losses']} | WR: {summary['win_rate']}%")
    print(f"  Total PnL: Rs {summary['total_pnl']:+,.0f}")
    print(f"  Avg Winner: Rs {summary['avg_winner']:+,.0f} | Avg Loser: Rs {summary['avg_loser']:+,.0f}")
    print()

    # Source breakdown
    print(f"  {'Source':>14} | {'Count':>5} | {'Wins':>4} | {'WR%':>5} | {'PnL':>12}")
    print(f"  {'-'*14}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*12}")
    for src, data in sorted(summary['by_source'].items()):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {src:>14} | {data['count']:>5} | {data['wins']:>4} | {wr:>5.1f} | Rs {data['pnl']:>+10,.0f}")
    print()

    # Individual trades
    print(f"  {'#':>2} | {'Time':>8} | {'Symbol':>14} | {'Dir':>4} | {'Score':>5} | {'Smart':>5} | {'DR':>5} | {'Source':>14} | {'PnL':>10} | {'Result':>15}")
    print(f"  {'-'*2}-+-{'-'*8}-+-{'-'*14}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*14}-+-{'-'*10}-+-{'-'*15}")
    for i, t in enumerate(summary['trades'], 1):
        ts = t.get('ts', '')
        if 'T' in ts:
            ts = ts.split('T')[1][:8]
        sym = t.get('underlying', t.get('symbol', '')).replace('NSE:', '')
        d = t.get('direction', '?')
        sc = t.get('final_score', 0)
        smart = t.get('smart_score', 0)
        dr = t.get('dr_score', 0)
        src = t.get('source', '?')
        pnl = t.get('total_pnl', 0)
        result = t.get('final_result', '?')
        print(f"  {i:>2} | {ts:>8} | {sym:>14} | {d:>4} | {sc:>5.1f} | {smart:>5.1f} | {dr:>5.3f} | {src:>14} | {pnl:>+10,.0f} | {result:>15}")

    print(f"\n{'='*80}")
