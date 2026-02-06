# Autonomous LLM Trading Agent

An AI-powered autonomous trading agent that uses GPT-4o for market analysis and trade decisions, with Zerodha Kite Connect for real-time data and execution.

## Features

- 🤖 **LLM-Powered Analysis**: Uses GPT-4o to analyze technicals (RSI, support/resistance, trends)
- 📊 **OI Analysis**: Futures Open Interest analysis for institutional flow detection
- ⚡ **Real-time Monitoring**: Checks positions every 3 seconds for target/stoploss
- 📈 **Trailing Stoploss**: Automatically moves SL to breakeven at 1% profit
- 🕐 **EOD Auto-Exit**: Closes all positions before 3:15 PM market close
- 📝 **Paper Trading**: Test strategies without real money
- 💾 **Trade Persistence**: Saves trades to JSON, survives restarts

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hellosandeeptiwari/MCX.git
   cd MCX
   ```

2. **Create .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate Zerodha Access Token**
   - Login to [Kite Connect](https://kite.trade)
   - Complete OAuth flow to get access token
   - Save to `zerodha_token.json`:
     ```json
     {"access_token": "your_token_here"}
     ```

5. **Run the trader**
   ```bash
   python agentic_trader/autonomous_trader.py --capital 100000 --interval 2
   ```

## Architecture

```
agentic_trader/
├── autonomous_trader.py   # Main trading loop & real-time monitor
├── config.py              # Configuration & trading rules
├── llm_agent.py           # GPT-4o agent with function calling
├── zerodha_tools.py       # Zerodha API wrapper & tools
├── dashboard.py           # Web dashboard (Flask)
└── templates/             # Dashboard HTML
```

## Trading Strategy

| Signal | Condition | Action |
|--------|-----------|--------|
| Oversold | RSI < 30 | BUY |
| Overbought | RSI > 70 | SELL (short) |
| Long Buildup | OI↑ + Price↑ | BUY |
| Short Buildup | OI↑ + Price↓ | SELL |
| Short Covering | OI↓ + Price↑ | BUY |

## Risk Management

- **Risk per trade**: 2.5% of capital
- **Max daily loss**: 5%
- **Max positions**: 10
- **Stop Loss**: 1% from entry
- **Target**: 1.5-2% from entry

## License

MIT
