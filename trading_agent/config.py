"""
TRADING AGENT CONFIGURATION
Central configuration for all modules
"""

# ========== ZERODHA API ==========
API_KEY = os.environ.get("ZERODHA_API_KEY", "")
API_SECRET = os.environ.get("ZERODHA_API_SECRET", "")

# ========== CAPITAL & RISK ==========
CAPITAL = 300000  # Rs 3 Lakh
RISK_PER_TRADE = 0.005  # 0.5% per trade (Rs 1,500)
MAX_DAILY_LOSS = 0.02  # 2% daily loss limit (Rs 6,000)
MAX_OPEN_POSITIONS = 3  # Max concurrent positions
MAX_TRADES_PER_DAY = 10  # Prevent overtrading

# ========== UNIVERSE ==========
# Liquid F&O stocks (NIFTY 50 subset)
UNIVERSE = [
    "NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:INFY", "NSE:ICICIBANK",
    "NSE:HINDUNILVR", "NSE:SBIN", "NSE:BHARTIARTL", "NSE:KOTAKBANK", "NSE:ITC",
    "NSE:LT", "NSE:AXISBANK", "NSE:ASIANPAINT", "NSE:MARUTI", "NSE:BAJFINANCE",
    "NSE:TITAN", "NSE:SUNPHARMA", "NSE:TATAMOTORS", "NSE:WIPRO", "NSE:HCLTECH",
    # Index
    "NSE:NIFTY 50", "NSE:NIFTY BANK",
    # Your focus
    "NSE:MCX"
]

# ========== STRATEGY PARAMETERS ==========
# Moving Average Crossover
MA_FAST = 9
MA_SLOW = 21
MA_TREND = 50  # Trend filter

# RSI Mean Reversion
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Breakout
BREAKOUT_LOOKBACK = 20  # Previous N candles for high/low
VOLUME_MULTIPLIER = 1.5  # Volume should be 1.5x average

# ========== TIMEFRAMES ==========
CANDLE_INTERVAL = "5minute"  # For signals
TREND_INTERVAL = "15minute"  # For trend confirmation

# ========== STOP LOSS / TARGET ==========
DEFAULT_SL_PERCENT = 1.0  # 1% stop loss
DEFAULT_TARGET_PERCENT = 2.0  # 2% target (1:2 risk-reward)
TRAILING_SL_PERCENT = 0.5  # Trail by 0.5%

# ========== TRADING HOURS ==========
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
NO_NEW_TRADES_AFTER = "15:00"  # Stop new entries after 3 PM

# ========== ALERTS (Optional) ==========
TELEGRAM_BOT_TOKEN = ""  # Add your Telegram bot token
TELEGRAM_CHAT_ID = ""  # Add your chat ID

# ========== FILES ==========
TOKEN_FILE = "zerodha_token.json"
POSITIONS_FILE = "trading_agent/positions.json"
TRADES_LOG = "trading_agent/trades_log.json"
DAILY_PNL_FILE = "trading_agent/daily_pnl.json"
