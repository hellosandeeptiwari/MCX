"""
AGENTIC TRADING DASHBOARD
Web interface for the LLM-powered trading agent
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import OPENAI_API_KEY, HARD_RULES, APPROVED_UNIVERSE
from zerodha_tools import get_tools

app = Flask(__name__)
CORS(app)

# Global agent instance
agent = None


def get_or_create_agent():
    """Get or create the agent instance"""
    global agent
    if agent is None:
        if not OPENAI_API_KEY:
            return None
        from llm_agent import TradingAgent
        agent = TradingAgent(auto_execute=False)
    return agent


@app.route('/')
def dashboard():
    return render_template('agentic_dashboard.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    tools = get_tools()
    
    has_api_key = bool(OPENAI_API_KEY)
    account = tools.get_account_state()
    risk = tools.get_portfolio_risk()
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'openai_configured': has_api_key,
        'zerodha_connected': 'error' not in account,
        'account': account,
        'risk': risk,
        'hard_rules': HARD_RULES,
        'universe_count': len(APPROVED_UNIVERSE)
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the agent"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({
            'error': 'OpenAI API key not configured',
            'response': 'Please add your OpenAI API key to agentic_trader/config.py'
        }), 400
    
    message = request.json.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = agent.run(message)
        return jsonify({
            'response': response,
            'pending_trades': agent.get_pending_approvals()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_market():
    """Ask agent to analyze market"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({'error': 'OpenAI API key not configured'}), 400
    
    symbols = request.json.get('symbols', APPROVED_UNIVERSE[:5])
    
    try:
        response = agent.analyze_market(symbols)
        return jsonify({
            'response': response,
            'pending_trades': agent.get_pending_approvals()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plan', methods=['POST'])
def create_plan():
    """Create trade plan for a symbol"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({'error': 'OpenAI API key not configured'}), 400
    
    symbol = request.json.get('symbol', '')
    bias = request.json.get('bias', 'neutral')
    
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    
    try:
        response = agent.create_trade_plan(symbol, bias)
        return jsonify({
            'response': response,
            'pending_trades': agent.get_pending_approvals()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pending')
def get_pending():
    """Get pending trade approvals"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({'pending_trades': []})
    
    return jsonify({
        'pending_trades': agent.get_pending_approvals()
    })


@app.route('/api/approve', methods=['POST'])
def approve_trade():
    """Approve a pending trade"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 400
    
    index = request.json.get('index', 0)
    
    try:
        result = agent.approve_trade(index)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reject', methods=['POST'])
def reject_trade():
    """Reject a pending trade"""
    agent = get_or_create_agent()
    
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 400
    
    index = request.json.get('index', 0)
    
    try:
        result = agent.reject_trade(index)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/universe')
def get_universe():
    """Get approved trading universe"""
    return jsonify({
        'universe': APPROVED_UNIVERSE
    })


def run_dashboard(host='0.0.0.0', port=5002, debug=False):
    """Run the dashboard"""
    print(f"\n🤖 Agentic Trading Dashboard")
    print(f"   URL: http://localhost:{port}")
    print(f"   OpenAI Key: {'✅ Configured' if OPENAI_API_KEY else '❌ Not Set'}")
    print(f"\n   Add your API key to: agentic_trader/config.py")
    print(f"   Press Ctrl+C to stop\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_dashboard(debug=True)
