"""
LLM AGENT - The Brain of the Trading System
Uses OpenAI GPT to reason about trades and make decisions
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import openai

from config import OPENAI_API_KEY, AGENT_SYSTEM_PROMPT, HARD_RULES, APPROVED_UNIVERSE
from zerodha_tools import get_tools, ZerodhaTools


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_state",
            "description": "Get current account state including margins, positions, open orders, P&L, and whether trading is allowed. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Get market data (OHLCV + indicators) for specified symbols. Only symbols in approved universe will return data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symbols like ['NSE:RELIANCE', 'NSE:TCS']"
                    }
                },
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_position_size",
            "description": "Calculate safe position size based on entry, stop loss, and capital. Ensures risk <= 0.5% per trade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_price": {"type": "number", "description": "Planned entry price"},
                    "stop_loss": {"type": "number", "description": "Stop loss price"},
                    "capital": {"type": "number", "description": "Available capital"},
                    "lot_size": {"type": "integer", "description": "Lot size (1 for stocks)", "default": 1}
                },
                "required": ["entry_price", "stop_loss", "capital"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_trade",
            "description": "Validate a trade plan against all hard rules. Returns pass/fail for each rule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "side": {"type": "string", "enum": ["BUY", "SELL"], "description": "Trade direction"},
                    "entry_price": {"type": "number", "description": "Entry price"},
                    "stop_loss": {"type": "number", "description": "Stop loss price"},
                    "target": {"type": "number", "description": "Target price"},
                    "quantity": {"type": "integer", "description": "Number of shares"},
                    "risk_pct": {"type": "number", "description": "Risk as percentage of capital"}
                },
                "required": ["symbol", "side", "entry_price", "stop_loss", "target", "quantity", "risk_pct"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_risk",
            "description": "Get current portfolio risk exposure including total exposure, estimated risk, and remaining daily loss limit.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place an order with Zerodha. Only use after validate_trade passes. Places both entry and stop loss orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["BUY", "SELL"]},
                    "quantity": {"type": "integer"},
                    "entry_price": {"type": "number"},
                    "stop_loss": {"type": "number"},
                    "target": {"type": "number"},
                    "risk_pct": {"type": "number"}
                },
                "required": ["symbol", "side", "quantity", "stop_loss"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_options_chain",
            "description": "Get options chain for a stock with ATM strikes and recommendation. Use this for F&O trading on high-priced stocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol like 'NSE:RELIANCE'"},
                    "bias": {"type": "string", "enum": ["bullish", "bearish", "neutral"], "description": "Market view on the stock"}
                },
                "required": ["symbol", "bias"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_option_order",
            "description": "FALLBACK: Place a naked option BUY order. Only use this when place_credit_spread() is not viable or returns fallback. The system auto-selects CE for BUY direction and PE for SELL direction. Requires score >= 70 for entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "description": "Underlying stock symbol like 'NSE:RELIANCE'"},
                    "direction": {"type": "string", "enum": ["BUY", "SELL"], "description": "BUY = bullish (auto CE), SELL = bearish (auto PE)"},
                    "strike_selection": {"type": "string", "enum": ["ATM", "ITM_1", "ITM_2", "OTM_1", "OTM_2"], "description": "Strike selection. ATM recommended for most trades."},
                    "rationale": {"type": "string", "description": "Trade rationale (e.g., 'ORB breakout + high volume')"}
                },
                "required": ["underlying", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_credit_spread",
            "description": "PRIMARY STRATEGY: Place a credit spread (SELL option + BUY hedge). Theta-positive — time decay works in our favor. ALWAYS try this FIRST before place_option_order(). BULLISH view = Bull Put Spread (sell PE + buy further OTM PE). BEARISH view = Bear Call Spread (sell CE + buy further OTM CE). Max risk is defined. If this returns a fallback suggestion, then try place_option_order().",
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "description": "Underlying stock symbol like 'NSE:RELIANCE'"},
                    "direction": {"type": "string", "enum": ["BUY", "SELL"], "description": "BUY = bullish (bull put spread), SELL = bearish (bear call spread)"},
                    "spread_width": {"type": "integer", "description": "Number of strikes between sold and hedge leg (default: 2)", "default": 2},
                    "rationale": {"type": "string", "description": "Trade rationale"}
                },
                "required": ["underlying", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_debit_spread",
            "description": "INTRADAY MOMENTUM: Place a debit spread on a big mover (>2.5% move + high volume). BUY near-ATM + SELL further OTM in direction of move. Cheaper than naked buy, defined risk = net debit. BULLISH → Bull Call Spread (BUY CE + SELL OTM CE). BEARISH → Bear Put Spread (BUY PE + SELL OTM PE). Auto-exits by 3:05 PM. Only use on stocks with strong intraday momentum. Score must be >= 70.",
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "description": "Underlying stock symbol like 'NSE:EICHERMOT' — must be moving >2.5% intraday"},
                    "direction": {"type": "string", "enum": ["BUY", "SELL"], "description": "BUY = bullish (bull call spread), SELL = bearish (bear put spread)"},
                    "spread_width": {"type": "integer", "description": "Number of strikes between buy and sell leg (default: 3)", "default": 3},
                    "rationale": {"type": "string", "description": "Trade rationale — must mention the big move and momentum"}
                },
                "required": ["underlying", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_iron_condor",
            "description": "Place an Iron Condor on a range-bound/choppy stock or index. Sells OTM CE + PE (collect premium) and buys further OTM CE + PE (hedge). Best for low-directional stocks with score 15-45. Works on NIFTY/BANKNIFTY (weekly expiry, 0-2 DTE) or F&O stocks (monthly expiry, 3-15 DTE). Auto-exits at 2:50 PM. Max profit = total credit, max loss = spread width - credit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string", "description": "Underlying symbol like 'NSE:NIFTY 50', 'NSE:NIFTY BANK', or 'NSE:TATASTEEL'"},
                    "rationale": {"type": "string", "description": "Why this stock is range-bound/choppy — mention low score, flat VWAP, compressed regime, etc."}
                },
                "required": ["underlying", "rationale"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_volume_analysis",
            "description": """Analyze intraday momentum using Futures OI (not delivery volume) to predict EOD movement.
            
            Returns for each symbol:
            - oi_signal: LONG_BUILDUP (bullish), SHORT_BUILDUP (bearish), SHORT_COVERING (rally), LONG_UNWINDING (weak)
            - eod_prediction: UP, DOWN, or NEUTRAL
            - eod_confidence: HIGH, MEDIUM, or LOW
            - trade_signal: BUY_FOR_EOD or SHORT_FOR_EOD
            - buy_pressure_pct / sell_pressure_pct: Order book depth imbalance
            
            Use this to find EOD momentum plays - especially after 11 AM.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symbols like ['NSE:RELIANCE', 'NSE:SBIN']"
                    }
                },
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_oi_analysis",
            "description": """Get detailed Open Interest analysis for a single F&O stock.
            
            OI Interpretation:
            - LONG_BUILDUP: Price UP + OI UP = Strong Bullish, fresh longs entering
            - SHORT_BUILDUP: Price DOWN + OI UP = Strong Bearish, fresh shorts entering
            - SHORT_COVERING: Price UP + OI DOWN = Rally from shorts buying back
            - LONG_UNWINDING: Price DOWN + OI DOWN = Longs exiting, weakness
            
            Use this for deeper analysis on specific stocks before trading.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol like 'NSE:RELIANCE'"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]


@dataclass
class AgentMessage:
    """A message in the agent conversation"""
    role: str  # system, user, assistant, tool
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


class TradingAgent:
    """
    LLM-powered trading agent that reasons about trades
    """
    
    def __init__(self, auto_execute: bool = False, paper_mode: bool = True, paper_capital: float = None):
        """
        Initialize the trading agent
        
        Args:
            auto_execute: If True, agent can place orders automatically.
                         If False, it only produces trade plans for approval.
            paper_mode: If True, use simulated capital instead of real margins.
            paper_capital: Capital amount for paper trading.
        """
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not set. Add it to config.py")
        
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.tools = get_tools(paper_mode=paper_mode, paper_capital=paper_capital)
        self.auto_execute = auto_execute
        self.paper_mode = paper_mode
        self.paper_capital = paper_capital
        self.conversation: List[Dict] = []
        self.trade_plans: List[Dict] = []
        
        # Initialize with system prompt
        self.conversation.append({
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT + f"""

CURRENT CONFIGURATION:
- Auto-execution mode: {auto_execute}
- Approved universe: {len(APPROVED_UNIVERSE)} symbols
- Risk per trade: {HARD_RULES['RISK_PER_TRADE']*100}%
- Max daily loss: {HARD_RULES['MAX_DAILY_LOSS']*100}%
- Max positions: {HARD_RULES['MAX_POSITIONS']}

If auto_execute is False, you must output trade plans for user approval instead of placing orders."""
        })
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return the result as a string"""
        try:
            if tool_name == "get_account_state":
                result = self.tools.get_account_state()
            elif tool_name == "get_market_data":
                result = self.tools.get_market_data(arguments.get("symbols", []))
            elif tool_name == "calculate_position_size":
                result = self.tools.calculate_position_size(
                    entry_price=arguments["entry_price"],
                    stop_loss=arguments["stop_loss"],
                    capital=arguments["capital"],
                    lot_size=arguments.get("lot_size", 1)
                )
            elif tool_name == "validate_trade":
                result = self.tools.validate_trade(arguments)
            elif tool_name == "get_portfolio_risk":
                result = self.tools.get_portfolio_risk()
            elif tool_name == "get_options_chain":
                result = self.tools.get_options_chain(
                    symbol=arguments["symbol"],
                    bias=arguments.get("bias", "neutral")
                )
            elif tool_name == "place_option_order":
                # === ARGUMENT VALIDATION ===
                underlying = arguments.get("underlying", "")
                direction = arguments.get("direction", "")
                
                # Validate underlying format
                if not underlying or not isinstance(underlying, str):
                    return json.dumps({"success": False, "error": "VALIDATION: 'underlying' is required and must be a string"})
                if not underlying.startswith("NSE:"):
                    underlying = f"NSE:{underlying}"
                    arguments["underlying"] = underlying
                
                # Validate direction
                if direction not in ("BUY", "SELL"):
                    return json.dumps({"success": False, "error": f"VALIDATION: 'direction' must be 'BUY' or 'SELL', got '{direction}'"})
                
                # Validate strike_selection
                valid_strikes = {"ATM", "OTM1", "OTM2", "ITM1", "ITM2"}
                strike_sel = arguments.get("strike_selection", "ATM")
                if strike_sel not in valid_strikes:
                    arguments["strike_selection"] = "ATM"
                
                if not self.auto_execute:
                    result = {
                        "success": False,
                        "message": "Auto-execution disabled. Option trade plan saved for user approval.",
                        "trade_plan": arguments
                    }
                    self.trade_plans.append(arguments)
                else:
                    result = self.tools.place_option_order(
                        underlying=arguments["underlying"],
                        direction=arguments["direction"],
                        strike_selection=arguments.get("strike_selection", "ATM"),
                        rationale=arguments.get("rationale", ""),
                        use_intraday_scoring=True
                    )
            elif tool_name == "place_credit_spread":
                # === ARGUMENT VALIDATION ===
                underlying = arguments.get("underlying", "")
                direction = arguments.get("direction", "")
                
                if not underlying or not isinstance(underlying, str):
                    return json.dumps({"success": False, "error": "VALIDATION: 'underlying' is required"})
                if not underlying.startswith("NSE:"):
                    underlying = f"NSE:{underlying}"
                    arguments["underlying"] = underlying
                
                if direction not in ("BUY", "SELL"):
                    return json.dumps({"success": False, "error": f"VALIDATION: 'direction' must be 'BUY' or 'SELL', got '{direction}'"})
                
                if not self.auto_execute:
                    result = {
                        "success": False,
                        "message": "Auto-execution disabled. Credit spread plan saved for user approval.",
                        "trade_plan": arguments
                    }
                    self.trade_plans.append(arguments)
                else:
                    result = self.tools.place_credit_spread(
                        underlying=arguments["underlying"],
                        direction=arguments["direction"],
                        spread_width=arguments.get("spread_width"),
                        rationale=arguments.get("rationale", "")
                    )
            elif tool_name == "place_debit_spread":
                # === ARGUMENT VALIDATION ===
                underlying = arguments.get("underlying", "")
                direction = arguments.get("direction", "")
                
                if not underlying or not isinstance(underlying, str):
                    return json.dumps({"success": False, "error": "VALIDATION: 'underlying' is required"})
                if not underlying.startswith("NSE:"):
                    underlying = f"NSE:{underlying}"
                    arguments["underlying"] = underlying
                
                if direction not in ("BUY", "SELL"):
                    return json.dumps({"success": False, "error": f"VALIDATION: 'direction' must be 'BUY' or 'SELL', got '{direction}'"})
                
                if not self.auto_execute:
                    result = {
                        "success": False,
                        "message": "Auto-execution disabled. Debit spread plan saved for user approval.",
                        "trade_plan": arguments
                    }
                    self.trade_plans.append(arguments)
                else:
                    result = self.tools.place_debit_spread(
                        underlying=arguments["underlying"],
                        direction=arguments["direction"],
                        spread_width=arguments.get("spread_width"),
                        rationale=arguments.get("rationale", "")
                    )
            elif tool_name == "place_iron_condor":
                # === ARGUMENT VALIDATION ===
                underlying = arguments.get("underlying", "")
                
                if not underlying or not isinstance(underlying, str):
                    return json.dumps({"success": False, "error": "VALIDATION: 'underlying' is required"})
                if not underlying.startswith("NSE:"):
                    underlying = f"NSE:{underlying}"
                    arguments["underlying"] = underlying
                
                if not self.auto_execute:
                    result = {
                        "success": False,
                        "message": "Auto-execution disabled. Iron Condor plan saved for user approval.",
                        "trade_plan": arguments
                    }
                    self.trade_plans.append(arguments)
                else:
                    result = self.tools.place_iron_condor(
                        underlying=arguments["underlying"],
                        rationale=arguments.get("rationale", "Agent requested IC")
                    )
            elif tool_name == "place_order":
                # === ARGUMENT VALIDATION ===
                order_args = arguments if isinstance(arguments, dict) else {}
                symbol = order_args.get("symbol", "")
                qty = order_args.get("quantity", 0)
                side = order_args.get("side", "")
                
                # Validate symbol format
                if not symbol or not isinstance(symbol, str):
                    return json.dumps({"success": False, "error": "VALIDATION: 'symbol' is required"})
                if not symbol.startswith("NSE:") and not symbol.startswith("NFO:"):
                    symbol = f"NSE:{symbol}"
                    order_args["symbol"] = symbol
                
                # Validate quantity is positive integer
                try:
                    qty = int(qty)
                    if qty <= 0 or qty > 50000:
                        return json.dumps({"success": False, "error": f"VALIDATION: 'quantity' must be 1-50000, got {qty}"})
                    order_args["quantity"] = qty
                except (ValueError, TypeError):
                    return json.dumps({"success": False, "error": f"VALIDATION: 'quantity' must be an integer, got '{qty}'"})
                
                # Validate side
                if side not in ("BUY", "SELL"):
                    return json.dumps({"success": False, "error": f"VALIDATION: 'side' must be 'BUY' or 'SELL', got '{side}'"})
                
                if not self.auto_execute:
                    result = {
                        "success": False,
                        "message": "Auto-execution disabled. Trade plan saved for user approval.",
                        "trade_plan": arguments
                    }
                    self.trade_plans.append(arguments)
                else:
                    result = self.tools.place_order(arguments)
            elif tool_name == "get_volume_analysis":
                result = self.tools.get_volume_analysis(arguments.get("symbols", []))
            elif tool_name == "get_oi_analysis":
                result = self.tools.get_oi_analysis(arguments.get("symbol", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def run(self, user_message: str) -> str:
        """
        Run the agent with a user message
        
        Args:
            user_message: The user's request/question
        
        Returns:
            The agent's final response
        """
        # Add user message
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        # Run the agent loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call OpenAI with sufficient output tokens
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=self.conversation,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_completion_tokens=4096
            )
            
            message = response.choices[0].message
            
            # Add assistant message to conversation
            assistant_msg = {
                "role": "assistant",
                "content": message.content or ""
            }
            
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            
            self.conversation.append(assistant_msg)
            
            # If no tool calls, check if response was truncated
            if not message.tool_calls:
                # Check if response was cut off mid-sentence (truncated)
                finish_reason = response.choices[0].finish_reason
                content = message.content or ""
                
                if finish_reason == "length" and iteration < max_iterations:
                    # Response was truncated - ask GPT to continue with tool calls
                    print("⚠️ GPT response truncated - requesting continuation...")
                    self.conversation.append({
                        "role": "user",
                        "content": "Your response was cut off. Do NOT repeat the analysis. Just call place_order() for any remaining trades you identified. One place_order() call per trade."
                    })
                    continue  # Loop back to call GPT again
                
                return content or "No response generated."
            
            # Execute tool calls
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Executing: {tool_name}({json.dumps(arguments, indent=2)[:100]}...)")
                
                result = self._execute_tool(tool_name, arguments)
                
                # Add tool result to conversation
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        return "Agent reached maximum iterations without completing."
    
    def analyze_market(self, symbols: List[str] = None) -> str:
        """
        Ask the agent to analyze the market and find opportunities
        """
        if symbols is None:
            symbols = APPROVED_UNIVERSE[:10]  # Top 10 by default
        
        prompt = f"""Analyze the following symbols for trading opportunities: {symbols}

Follow the PROCESS:
1. First get account state to check if trading is allowed
2. Get market data for these symbols
3. Identify any signals based on technical indicators (RSI, SMA)
4. If you find opportunities, generate trade plans with proper risk management
5. Validate each trade plan against hard rules

Remember: Capital preservation is the priority. If uncertain, output "NO TRADE"."""

        return self.run(prompt)
    
    def create_trade_plan(self, symbol: str, bias: str = "neutral") -> str:
        """
        Ask the agent to create a trade plan for a specific symbol
        """
        prompt = f"""Create a trade plan for {symbol}.

Current bias: {bias}

Follow all steps:
1. Get account state
2. Get market data for {symbol}
3. Calculate proper position size based on risk rules
4. Validate the trade against all hard rules
5. Output the trade plan in the required format

If any rule fails, output "NO TRADE" with reasons."""

        return self.run(prompt)
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get trade plans waiting for user approval"""
        return self.trade_plans.copy()
    
    def approve_trade(self, trade_index: int) -> str:
        """Approve and execute a pending trade plan"""
        if trade_index >= len(self.trade_plans):
            return "Invalid trade index"
        
        trade = self.trade_plans.pop(trade_index)
        
        # Execute the trade
        result = self.tools.place_order(trade)
        return json.dumps(result, indent=2)
    
    def reject_trade(self, trade_index: int) -> str:
        """Reject a pending trade plan"""
        if trade_index >= len(self.trade_plans):
            return "Invalid trade index"
        
        trade = self.trade_plans.pop(trade_index)
        return f"Trade rejected: {trade['symbol']} {trade['side']}"


def run_interactive_agent():
    """Run the agent in interactive mode"""
    print("\n" + "="*60)
    print("🤖 AGENTIC TRADING SYSTEM")
    print("="*60)
    print("""
Commands:
  analyze [symbols]  - Analyze market for opportunities
  plan <symbol>      - Create trade plan for symbol
  status             - Check account status
  risk               - Check portfolio risk
  pending            - Show pending trade approvals
  approve <index>    - Approve a pending trade
  reject <index>     - Reject a pending trade
  help               - Show this help
  quit               - Exit

Example: analyze NSE:RELIANCE NSE:TCS NSE:HDFCBANK
""")
    
    try:
        agent = TradingAgent(auto_execute=False)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nAdd your OpenAI API key to agentic_trader/config.py:")
        print('OPENAI_API_KEY = "sk-your-key-here"')
        return
    
    while True:
        try:
            user_input = input("\n🤖 > ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            
            if command == "quit":
                print("Goodbye!")
                break
            
            elif command == "help":
                print("Commands: analyze, plan, status, risk, pending, approve, reject, quit")
            
            elif command == "analyze":
                symbols = parts[1:] if len(parts) > 1 else None
                print("\n📊 Analyzing market...")
                response = agent.analyze_market(symbols)
                print(f"\n{response}")
            
            elif command == "plan":
                if len(parts) < 2:
                    print("Usage: plan <symbol>")
                    continue
                symbol = parts[1].upper()
                if ":" not in symbol:
                    symbol = f"NSE:{symbol}"
                print(f"\n📋 Creating trade plan for {symbol}...")
                response = agent.create_trade_plan(symbol)
                print(f"\n{response}")
            
            elif command == "status":
                print("\n📊 Checking account status...")
                response = agent.run("Get and summarize the current account state. Can I trade today?")
                print(f"\n{response}")
            
            elif command == "risk":
                print("\n⚠️ Checking portfolio risk...")
                response = agent.run("Get the current portfolio risk exposure and summarize it.")
                print(f"\n{response}")
            
            elif command == "pending":
                pending = agent.get_pending_approvals()
                if not pending:
                    print("No pending trade approvals.")
                else:
                    print("\n📋 Pending Trade Approvals:")
                    for i, trade in enumerate(pending):
                        print(f"  [{i}] {trade.get('side')} {trade.get('quantity')} {trade.get('symbol')}")
                        print(f"      Entry: ₹{trade.get('entry_price')} | SL: ₹{trade.get('stop_loss')} | Target: ₹{trade.get('target')}")
            
            elif command == "approve":
                if len(parts) < 2:
                    print("Usage: approve <index>")
                    continue
                try:
                    index = int(parts[1])
                    result = agent.approve_trade(index)
                    print(f"\n{result}")
                except ValueError:
                    print("Invalid index")
            
            elif command == "reject":
                if len(parts) < 2:
                    print("Usage: reject <index>")
                    continue
                try:
                    index = int(parts[1])
                    result = agent.reject_trade(index)
                    print(f"\n{result}")
                except ValueError:
                    print("Invalid index")
            
            else:
                # Free-form query
                print("\n🤔 Thinking...")
                response = agent.run(user_input)
                print(f"\n{response}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_interactive_agent()
