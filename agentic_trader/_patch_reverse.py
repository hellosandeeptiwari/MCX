"""Patch reverse_trade endpoint to exit current + open opposite."""
import sys

filepath = r'c:\Users\SandeepTiwari\MCX\agentic_trader\dashboard.py'
lines = open(filepath, 'r', encoding='utf-8', errors='surrogateescape').readlines()

# Find start line (the comment line before @app.route)
start_line = None
for i, line in enumerate(lines):
    if 'Reverse paper trade' in line or ('Reverse trade' in line and 'shadow' in line):
        start_line = i
        break
    if 'reverse_trade' in line and 'route' in line and start_line is None:
        # Check if previous line is a comment
        if i > 0 and lines[i-1].strip().startswith('#'):
            start_line = i - 1
        else:
            start_line = i
        break

if start_line is None:
    # Fallback: find @app.route('/api/reverse_trade'
    for i, line in enumerate(lines):
        if "'/api/reverse_trade'" in line:
            start_line = i - 2  # include comment lines above
            break

# Find end: the except + return + blank lines before MANUAL_EXIT_FILE
end_line = None
for i in range(start_line + 10, len(lines)):
    if 'Manual position exit' in lines[i] or 'MANUAL_EXIT_FILE' in lines[i]:
        end_line = i
        break

if start_line is None or end_line is None:
    print(f"ERROR: Could not find boundaries. start={start_line}, end={end_line}")
    sys.exit(1)

print(f"Replacing lines {start_line}-{end_line-1} (old function)")
print(f"First old line: {lines[start_line][:60]!r}")
print(f"Last old line:  {lines[end_line-1][:60]!r}")

NEW_FUNCTION = r'''# ── Reverse trade: EXIT current position + OPEN opposite ───────

@app.route('/api/reverse_trade', methods=['POST'])
def reverse_trade():
    """Reverse a position: exit current, then open opposite (PE↔CE, same strike/lots).
    Fetches live market LTP via Kite API, calculates SL/target like the bot does."""
    try:
        import re, random
        data = request.get_json(force=True, silent=True) or {}
        symbol = data.get('symbol', '').strip()
        underlying = data.get('underlying', '').strip()
        direction = data.get('direction', '').strip()  # reversed direction
        quantity = int(data.get('quantity', 0))
        is_option = data.get('is_option', False)
        option_type = data.get('option_type', '')
        strike = data.get('strike', 0)
        expiry = data.get('expiry', '')
        lots = int(data.get('lots', 1))

        if not symbol or not direction or quantity <= 0:
            return jsonify({'ok': False, 'msg': 'Missing required fields'}), 400

        db = get_state_db()
        today = _today()

        # ── STEP 1: EXIT the current position ──
        positions, realized_pnl, paper_capital = db.load_active_trades(today)
        target_pos = None
        remaining = []
        for pos in positions:
            pos_sym = pos.get('symbol') or pos.get('option_symbol') or ''
            if pos_sym == symbol and pos.get('status', 'OPEN') == 'OPEN' and target_pos is None:
                target_pos = pos
            else:
                remaining.append(pos)

        exit_pnl = 0
        exit_price = 0
        exit_msg = ''
        if target_pos:
            # Get current LTP for P&L calc
            live = db.load_live_pnl() or {}
            lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
            ltp_exit = 0
            if isinstance(lp, dict):
                ltp_exit = lp.get('ltp', 0)
            elif isinstance(lp, (int, float)):
                ltp_exit = float(lp)

            entry_price = target_pos.get('avg_price') or target_pos.get('entry_price') or 0
            exit_qty = abs(target_pos.get('quantity', 0))
            side = target_pos.get('side') or target_pos.get('direction', 'BUY')

            if ltp_exit > 0 and entry_price > 0:
                if side in ('BUY', 'LONG'):
                    exit_pnl = (ltp_exit - entry_price) * exit_qty
                else:
                    exit_pnl = (entry_price - ltp_exit) * exit_qty
            else:
                exit_pnl = target_pos.get('unrealized_pnl', 0)

            exit_price = ltp_exit if ltp_exit > 0 else entry_price

            # Write exit signal for bot (in-memory cleanup)
            exit_signal = {
                'symbol': symbol,
                'exit_price': round(exit_price, 2),
                'pnl': round(exit_pnl, 2),
                'exit_time': datetime.now().isoformat(),
                'exit_type': 'AUTOPILOT_REVERSE_EXIT',
                'direction': side,
                'quantity': exit_qty,
                'entry_price': round(entry_price, 2),
                'trade': target_pos,
                'live_exit_placed': False,
            }
            exit_pending = []
            if MANUAL_EXIT_FILE.exists():
                try:
                    exit_pending = json.loads(MANUAL_EXIT_FILE.read_text())
                except Exception:
                    exit_pending = []
            exit_pending.append(exit_signal)
            MANUAL_EXIT_FILE.write_text(json.dumps(exit_pending, indent=2, default=str))

            # Log EXIT in trade_ledger
            try:
                ledger = get_trade_ledger()
                _underlying = target_pos.get('underlying', '')
                if not _underlying:
                    m = re.match(r'(?:NFO:)?([A-Z]+)\d', symbol.replace('NFO:', ''))
                    _underlying = f"NSE:{m.group(1)}" if m else symbol
                _hold_mins = 0
                try:
                    from dateutil.parser import parse as _dp
                    _hold_mins = int((datetime.now() - _dp(target_pos.get('timestamp', ''))).total_seconds() / 60)
                except Exception:
                    pass
                ledger.log_exit(
                    symbol=symbol,
                    underlying=_underlying,
                    exit_price=round(exit_price, 2),
                    exit_time=datetime.now().isoformat(),
                    exit_type='AUTOPILOT_REVERSE_EXIT',
                    pnl=round(exit_pnl, 2),
                    hold_time_minutes=_hold_mins,
                    entry_data=target_pos,
                )
            except Exception:
                pass

            realized_pnl += exit_pnl
            exit_msg = f'Exited {symbol} P&L={exit_pnl:+,.0f} | '

        # ── STEP 2: BUILD reverse position (PE↔CE) ──
        rev_symbol = symbol
        if is_option and option_type in ('CE', 'PE'):
            orig_type = 'PE' if option_type == 'CE' else 'CE'
            rev_symbol = re.sub(rf'{orig_type}$', option_type, symbol)

        # Fetch REAL market LTP via Kite API
        market_ltp = 0
        kite = _get_dashboard_kite()
        if not kite:
            # Still save the exit
            if target_pos:
                db.save_active_trades(remaining, realized_pnl, paper_capital)
            return jsonify({'ok': False, 'msg': f'{exit_msg}Kite API not available for reverse'}), 503

        nfo_symbol = f'NFO:{rev_symbol}' if not rev_symbol.startswith('NFO:') else rev_symbol
        try:
            ltp_data = kite.ltp([nfo_symbol])
            if nfo_symbol in ltp_data:
                market_ltp = ltp_data[nfo_symbol]['last_price']
        except Exception as e:
            print(f"\u26a0\ufe0f Reverse trade LTP fetch failed for {nfo_symbol}: {e}")

        if market_ltp <= 0:
            if target_pos:
                db.save_active_trades(remaining, realized_pnl, paper_capital)
            return jsonify({'ok': False, 'msg': f'{exit_msg}Could not fetch price for {rev_symbol}'}), 400

        stoploss_premium = round(market_ltp * 0.72, 2)
        target_premium = round(market_ltp * 1.60, 2)
        total_premium = round(market_ltp * quantity, 2)
        max_loss = round((market_ltp - stoploss_premium) * quantity, 2)

        paper_id = f'PAPER_REV_{random.randint(100000, 999999)}'
        trade_id = f'REV_{datetime.now().strftime("%H%M%S")}_{random.randint(1000,9999)}'

        new_pos = {
            'symbol': rev_symbol,
            'underlying': underlying or symbol,
            'quantity': quantity,
            'lots': lots,
            'avg_price': market_ltp,
            'side': 'BUY',
            'direction': direction,
            'option_type': option_type if is_option else '',
            'strike': strike if is_option else 0,
            'expiry': expiry if is_option else '',
            'stop_loss': stoploss_premium,
            'target': target_premium,
            'order_id': paper_id,
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'OPEN',
            'is_option': is_option,
            'total_premium': total_premium,
            'max_loss': max_loss,
            'setup_type': 'REVERSE_SHADOW',
            'strategy_type': 'NAKED_OPTION',
            'rationale': f'Reverse of {symbol} \u2192 {rev_symbol} @ \u20b9{market_ltp:.2f} (same strike/lots)',
            'entry_score': 0,
            'score_tier': 'manual',
            'smart_score': 0,
            'lot_multiplier': 1.0,
            'sector': '',
            'trigger_type': 'AUTOPILOT_REVERSE',
            'is_sniper': False,
            'delta': 0, 'theta': 0, 'iv': 0,
        }

        # Write entry signal for bot (in-memory injection)
        signal_file = os.path.join(os.path.dirname(__file__), 'manual_entry_requests.json')
        pending = []
        if os.path.exists(signal_file):
            try:
                with open(signal_file, 'r') as sf:
                    pending = json.loads(sf.read().strip() or '[]')
            except Exception:
                pending = []
        pending.append(new_pos)
        with open(signal_file, 'w') as sf:
            json.dump(pending, sf)

        # Save to state_db: remaining (without exited) + new reverse position
        remaining.append(new_pos)
        db.save_active_trades(remaining, realized_pnl, paper_capital)

        return jsonify({
            'ok': True,
            'msg': f'{exit_msg}Opened {rev_symbol} @ \u20b9{market_ltp:.2f} | SL \u20b9{stoploss_premium:.2f} | TGT \u20b9{target_premium:.2f}',
            'order_id': paper_id,
            'ltp': market_ltp,
            'stop_loss': stoploss_premium,
            'target': target_premium,
        })

    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Reverse trade failed: {str(e)}'}), 500


'''

new_lines = lines[:start_line] + [NEW_FUNCTION] + lines[end_line:]
open(filepath, 'w', encoding='utf-8', errors='surrogateescape').writelines(new_lines)
print(f"SUCCESS: Replaced {end_line - start_line} lines with new function")
