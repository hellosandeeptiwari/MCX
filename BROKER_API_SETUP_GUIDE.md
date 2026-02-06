# ============================================================
# BROKER API SETUP GUIDE FOR LIVE MCX OPTIONS DATA
# ============================================================

## OPTION 1: ZERODHA KITE CONNECT (Recommended)

### Step 1: Subscribe to Kite Connect
- Go to: https://developers.kite.trade/
- Login with your Zerodha credentials
- Subscribe to Kite Connect API (Rs. 2000/month)

### Step 2: Create an App
- Click "Create new app"
- App Name: MCX Options Analyzer
- Redirect URL: http://127.0.0.1:5000/callback
- Description: Personal trading tool
- Click "Create"

### Step 3: Get Your Credentials
- Note down: API Key
- Note down: API Secret

### Step 4: Update the Script
Open `broker_api_setup.py` and update:
```python
BROKER = "zerodha"
ZERODHA_API_KEY = "your_api_key_here"
ZERODHA_API_SECRET = "your_api_secret_here"
```

### Step 5: Run and Login
```
python broker_api_setup.py
```
- Browser will open for login
- Login with Zerodha credentials
- Copy the request_token from redirect URL
- Paste in terminal


## OPTION 2: UPSTOX API (Free)

### Step 1: Register for API Access
- Go to: https://api.upstox.com/
- Login/Register with Upstox account

### Step 2: Create an App
- Click "Create App"
- App Name: MCX Options Analyzer
- Redirect URL: http://127.0.0.1:5000/callback
- Click "Create"

### Step 3: Get Your Credentials
- Note down: API Key
- Note down: API Secret

### Step 4: Update the Script
Open `broker_api_setup.py` and update:
```python
BROKER = "upstox"
UPSTOX_API_KEY = "your_api_key_here"
UPSTOX_API_SECRET = "your_api_secret_here"
```

### Step 5: Run and Login
```
python broker_api_setup.py
```


## OPTION 3: SHOONYA (Finvasia) - FREE API

### Step 1: Open Account
- Go to: https://shoonya.finvasia.com/
- Open free trading account

### Step 2: Get API Access
- Login to Shoonya
- Go to API section
- Generate API credentials

### Step 3: Install Library
```
pip install NorenRestApiPy
```


## COMPARISON

| Feature | Zerodha | Upstox | Shoonya |
|---------|---------|--------|---------|
| Monthly Cost | Rs.2000 | Free | Free |
| Data Quality | Excellent | Good | Good |
| Speed | Fast | Fast | Moderate |
| Documentation | Excellent | Good | Good |
| Options Chain | Yes | Yes | Yes |


## QUICK START COMMANDS

```bash
# Install required libraries
pip install kiteconnect        # For Zerodha
pip install upstox-python-sdk  # For Upstox
pip install NorenRestApiPy     # For Shoonya

# Run setup
python broker_api_setup.py
```


## NEED HELP?

1. Zerodha Kite Connect docs: https://kite.trade/docs/connect/
2. Upstox API docs: https://upstox.com/developer/api-documentation/
3. Shoonya API docs: https://shoonya.finvasia.com/api-documentation/
