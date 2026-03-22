"""
AGENTIC TRADING SYSTEM - QUICK START

Complete autonomous trading agent with ChatGPT API integration.

FILES:
------
agentic_trader/
  ├── config.py          - API keys and hard rules
  ├── zerodha_tools.py   - Zerodha API integration
  ├── llm_agent.py       - OpenAI GPT agent with function calling
  ├── dashboard.py       - Web dashboard
  └── templates/
      └── agentic_dashboard.html

SETUP:
------
1. Add your OpenAI API key to config.py:
   OPENAI_API_KEY = "sk-your-key-here"

2. Get Zerodha access token (run once daily):
   python -c "from zerodha_tools import ZerodhaTools; t = ZerodhaTools(); t.authenticate()"

3. Run the dashboard:
   python dashboard.py

4. Open in browser:
   http://localhost:5002

USAGE:
------
CLI Mode:
  python llm_agent.py
  
Commands:
  analyze           - Analyze market for opportunities
  plan RELIANCE     - Create trade plan for symbol
  status            - Check account status
  pending           - Show pending approvals
  approve 0         - Approve trade at index 0

Dashboard Mode:
  python dashboard.py
  - Chat with the agent
  - Click quick actions
  - Approve/reject trades

HARD RULES:
-----------
1. Risk per trade <= 0.5%
2. Max daily loss <= 1.5%
3. Max positions = 3
4. Stop-loss required
5. Only approved symbols
6. Trading hours only
"""

import os
import sys

# Check if running from correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def check_setup():
    """Check if everything is configured"""
    print("\n🔍 Checking Agentic Trading System Setup...")
    print("=" * 50)
    
    issues = []
    
    # Check OpenAI key
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY:
            print("✅ OpenAI API Key: Configured")
        else:
            print("❌ OpenAI API Key: NOT SET")
            issues.append("Add OPENAI_API_KEY to config.py")
    except Exception as e:
        print(f"❌ config.py: Error - {e}")
        issues.append("Fix config.py")
    
    # Check Zerodha
    try:
        from zerodha_tools import ZerodhaTools
        tools = ZerodhaTools()
        account = tools.get_account_state()
        if "error" not in account:
            print(f"✅ Zerodha: Connected (₹{account['available_margin']:,.0f} margin)")
        else:
            print(f"⚠️ Zerodha: {account.get('error', 'Not authenticated')}")
            issues.append("Run: ZerodhaTools().authenticate()")
    except Exception as e:
        print(f"⚠️ Zerodha: {e}")
        issues.append("Install kiteconnect: pip install kiteconnect")
    
    # Check dependencies
    deps = ["openai", "flask", "flask_cors", "kiteconnect"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}: Installed")
        except Exception as e:
            print(f"❌ {dep}: NOT INSTALLED ({e})")
            issues.append(f"pip install {dep}")
    
    print("\n" + "=" * 50)
    
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"   → {issue}")
    else:
        print("✅ All systems ready!")
        print("\nStart options:")
        print("  CLI:       python llm_agent.py")
        print("  Dashboard: python dashboard.py")
    
    return len(issues) == 0


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_setup()
        
        elif command == "cli":
            from llm_agent import run_interactive_agent
            run_interactive_agent()
        
        elif command == "dashboard":
            from dashboard import run_dashboard
            run_dashboard(debug=True)
        
        elif command == "auth":
            from zerodha_tools import ZerodhaTools
            tools = ZerodhaTools()
            tools.authenticate()
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run.py [check|cli|dashboard|auth]")
    else:
        print("🤖 Agentic Trading System")
        print()
        print("Usage: python run.py <command>")
        print()
        print("Commands:")
        print("  check      Check system setup")
        print("  cli        Run interactive CLI")
        print("  dashboard  Run web dashboard")
        print("  auth       Authenticate with Zerodha")


if __name__ == "__main__":
    main()
