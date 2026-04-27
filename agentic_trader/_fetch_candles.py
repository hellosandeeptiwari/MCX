"""Fetch 365 days of 5-min candle data for all F&O stocks."""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '.')
from ml_models.data_fetcher import fetch_and_save_all
fetch_and_save_all(days=365)
