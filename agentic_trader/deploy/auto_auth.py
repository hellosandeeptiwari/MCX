"""
TITAN AUTO-AUTH — Automated Zerodha Kite Login
===============================================
Runs daily before market open. Uses Selenium (headless Chrome) to:
1. Open Kite login page
2. Enter user_id + password
3. Enter TOTP (generated from secret)
4. Capture access_token from redirect URL
5. Save to .env

Required .env variables:
  ZERODHA_USER_ID=AB1234
  ZERODHA_PASSWORD=your_password
  ZERODHA_TOTP_SECRET=JBSWY3DPEHPK3PXP  (base32 TOTP seed)
  ZERODHA_API_KEY=bfnlfzxt4q4m9arh
  ZERODHA_API_SECRET=88uryfi2xl0bdhpwo7ozc9krerwcq06j

Usage:
  python auto_auth.py          # Run once, get token
  python auto_auth.py --test   # Test if current token is valid
"""

import os
import sys
import json
import time
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
except ImportError:
    pass


def check_existing_token():
    """Check if current .env token is valid."""
    try:
        access_token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')
        if access_token:
            from kiteconnect import KiteConnect
            api_key = os.environ.get('ZERODHA_API_KEY', '')
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            profile = kite.profile()
            print(f"✅ Valid token in .env. User: {profile.get('user_name')}")
            return True
    except Exception as e:
        print(f"⚠️ .env token invalid: {e}")
    return False


def auto_login():
    """Automated Zerodha login via headless Chrome."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import pyotp
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install: pip install selenium pyotp")
        return False

    # Load credentials
    user_id = os.environ.get('ZERODHA_USER_ID', '')
    password = os.environ.get('ZERODHA_PASSWORD', '')
    totp_secret = os.environ.get('ZERODHA_TOTP_SECRET', '')
    api_key = os.environ.get('ZERODHA_API_KEY', '')
    api_secret = os.environ.get('ZERODHA_API_SECRET', '')

    if not all([user_id, password, totp_secret, api_key, api_secret]):
        missing = []
        if not user_id: missing.append('ZERODHA_USER_ID')
        if not password: missing.append('ZERODHA_PASSWORD')
        if not totp_secret: missing.append('ZERODHA_TOTP_SECRET')
        if not api_key: missing.append('ZERODHA_API_KEY')
        if not api_secret: missing.append('ZERODHA_API_SECRET')
        print(f"❌ Missing .env variables: {', '.join(missing)}")
        return False

    print(f"🔐 Auto-login starting for {user_id}...")

    # Setup headless Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)

        # Step 1: Open Kite login
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        driver.get(login_url)
        time.sleep(2)

        # Step 2: Enter user ID
        user_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]')))
        user_input.clear()
        user_input.send_keys(user_id)

        # Step 3: Enter password
        pass_input = driver.find_element(By.CSS_SELECTOR, 'input[type="password"]')
        pass_input.clear()
        pass_input.send_keys(password)

        # Step 4: Click login
        login_btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
        login_btn.click()
        time.sleep(3)

        # Step 5: Enter TOTP
        totp = pyotp.TOTP(totp_secret)
        totp_code = totp.now()
        print(f"   TOTP generated: {totp_code[:2]}****")

        totp_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"], input[type="number"]')))
        totp_input.clear()
        totp_input.send_keys(totp_code)

        # Some Kite versions auto-submit, others need a click
        time.sleep(2)
        try:
            submit_btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
            submit_btn.click()
        except:
            pass  # Auto-submitted

        # Step 6: Wait for redirect with request_token
        time.sleep(5)
        current_url = driver.current_url
        print(f"   Redirect URL: {current_url[:80]}...")

        if 'request_token=' not in current_url:
            # Wait longer
            for _ in range(10):
                time.sleep(1)
                current_url = driver.current_url
                if 'request_token=' in current_url:
                    break

        if 'request_token=' not in current_url:
            print(f"❌ No request_token in URL: {current_url}")
            # Save screenshot for debugging
            driver.save_screenshot('/tmp/titan_auth_fail.png')
            return False

        # Extract request_token
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(current_url)
        params = parse_qs(parsed.query)
        request_token = params.get('request_token', [None])[0]

        if not request_token:
            print(f"❌ Could not parse request_token")
            return False

        print(f"   Request token: {request_token[:8]}...")

        # Step 7: Generate access_token via API
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data['access_token']

        # Verify
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"   ✅ Logged in as: {profile.get('user_name')}")

        # Step 8: Save token
        _save_token(access_token)

        return True

    except Exception as e:
        print(f"❌ Auto-login failed: {e}")
        import traceback
        traceback.print_exc()
        if driver:
            try:
                driver.save_screenshot('/tmp/titan_auth_fail.png')
                print("   Screenshot saved: /tmp/titan_auth_fail.png")
            except:
                pass
        return False
    finally:
        if driver:
            driver.quit()


def _save_token(access_token):
    """Save access token to .env file"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Update .env
    env_path = os.path.join(base_dir, '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

        updated = False
        new_lines = []
        for line in lines:
            if line.startswith('ZERODHA_ACCESS_TOKEN='):
                new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f'ZERODHA_ACCESS_TOKEN={access_token}\n')

        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        print(f"   Updated .env ZERODHA_ACCESS_TOKEN")

    # Also set in current process environment
    os.environ['ZERODHA_ACCESS_TOKEN'] = access_token


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Titan Auto-Auth')
    parser.add_argument('--test', action='store_true', help='Just test if current token is valid')
    args = parser.parse_args()

    if args.test:
        valid = check_existing_token()
        sys.exit(0 if valid else 1)

    # Check if we already have a valid token
    if check_existing_token():
        print("   Skipping login — token already valid.")
        sys.exit(0)

    # Run auto-login
    success = auto_login()
    if not success:
        print("\n⚠️ Auto-login failed. You'll need to authenticate manually.")
        print("   Run: python agentic_trader/refresh_token.py")
        sys.exit(1)
    else:
        print("\n✅ Auto-auth complete. Titan is ready to trade!")
        sys.exit(0)
