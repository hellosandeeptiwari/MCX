"""Fetch 500 days of daily candle data for all F&O stocks."""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
from ml_models.data_fetcher import fetch_and_save_daily
fetch_and_save_daily(days=500)
