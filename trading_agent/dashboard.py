"""
TRADING AGENT DASHBOARD
Web interface for monitoring and controlling the trading agent

Features:
- Real-time status display
- Kill switch button
- Position management
- P&L tracking
- Signal viewer
- Manual controls
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import json
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_manager import get_data_manager
from signal_engine import get_signal_engine
from risk_manager import get_risk_manager
from execution_engine import get_execution_engine
from config import UNIVERSE, CAPITAL

app = Flask(__name__)
CORS(app)

# Global agent reference
agent = None
agent_thread = None


def get_modules():
    """Get all module instances"""
    return {
        'dm': get_data_manager(),
        'se': get_signal_engine(),
        'rm': get_risk_manager(),
        'ee': get_execution_engine()
    }


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('agent_dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current agent status"""
    modules = get_modules()
    rm = modules['rm']
    dm = modules['dm']
    
    risk_summary = rm.get_risk_summary()
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'agent_running': agent is not None and agent.running if agent else False,
        'websocket_connected': dm.connected,
        'authenticated': dm.is_authenticated(),
        'kill_switch': rm.kill_switch,
        'market_hours': rm.is_trading_hours(),
        'can_trade': rm.can_take_new_trade() and not rm.kill_switch,
        'risk': risk_summary
    })


@app.route('/api/positions')
def get_positions():
    """Get open positions"""
    rm = get_modules()['rm']
    positions = rm.get_open_positions()
    
    return jsonify({
        'positions': [p.to_dict() for p in positions],
        'count': len(positions)
    })


@app.route('/api/quotes')
def get_quotes():
    """Get live quotes for universe"""
    dm = get_modules()['dm']
    
    try:
        quotes = dm.get_ltp(UNIVERSE[:20])  # Limit to 20 for speed
        return jsonify({'quotes': quotes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan', methods=['POST'])
def run_scan():
    """Manually trigger a signal scan"""
    modules = get_modules()
    dm = modules['dm']
    se = modules['se']
    
    try:
        # Get data and scan
        ohlc_data = {}
        for symbol in UNIVERSE[:10]:
            df = dm.get_historical_data(symbol, "5minute", days=3)
            if not df.empty:
                ohlc_data[symbol] = df
        
        signals = se.scan_universe(ohlc_data)
        
        return jsonify({
            'signals': [s.to_dict() for s in signals],
            'count': len(signals),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kill-switch', methods=['POST'])
def toggle_kill_switch():
    """Toggle kill switch"""
    rm = get_modules()['rm']
    action = request.json.get('action', 'toggle')
    reason = request.json.get('reason', 'Dashboard manual toggle')
    
    if action == 'activate' or (action == 'toggle' and not rm.kill_switch):
        rm.activate_kill_switch(reason)
        return jsonify({'kill_switch': True, 'message': 'Kill switch ACTIVATED'})
    else:
        rm.deactivate_kill_switch()
        return jsonify({'kill_switch': False, 'message': 'Kill switch deactivated'})


@app.route('/api/exit-all', methods=['POST'])
def exit_all():
    """Exit all positions"""
    ee = get_modules()['ee']
    
    try:
        closed = ee.exit_all_positions("DASHBOARD_EXIT")
        return jsonify({'closed': closed, 'message': f'Exited {closed} positions'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/exit-position', methods=['POST'])
def exit_position():
    """Exit a specific position"""
    ee = get_modules()['ee']
    symbol = request.json.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        success = ee.exit_position(symbol, "DASHBOARD_EXIT")
        return jsonify({'success': success, 'symbol': symbol})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/orders')
def get_orders():
    """Get today's orders"""
    ee = get_modules()['ee']
    
    try:
        orders = ee.get_order_book()
        return jsonify({'orders': orders, 'count': len(orders)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades-log')
def get_trades_log():
    """Get trades log"""
    try:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'trading_agent', 'trades_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
            return jsonify({'logs': logs[-50:]})  # Last 50 entries
        return jsonify({'logs': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """Get detailed analysis for a symbol"""
    modules = get_modules()
    dm = modules['dm']
    se = modules['se']
    
    # Format symbol
    if ':' not in symbol:
        symbol = f"NSE:{symbol}"
    
    try:
        df = dm.get_historical_data(symbol, "5minute", days=5)
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        summary = se.get_strategy_summary(symbol, df)
        signal = se.analyze(symbol, df)
        
        return jsonify({
            'symbol': symbol,
            'summary': summary,
            'signal': signal.to_dict() if signal else None,
            'candles': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_dashboard(host='0.0.0.0', port=5001, debug=False):
    """Run the dashboard server"""
    print(f"\n🌐 Trading Agent Dashboard")
    print(f"   URL: http://localhost:{port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_dashboard(debug=True)
