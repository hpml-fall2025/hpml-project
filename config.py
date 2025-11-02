"""
Configuration file for 5-minute Realized Variance Prediction System.
"""

# Stock ticker symbol
TICKER = "AAPL"

# Target date to simulate (format: YYYY-MM-DD)
# This should be a historical trading day
TARGET_DATE = "2024-01-15"

# Days of historical data to fetch (minimum 60 for HAR model)
LOOKBACK_DAYS = 60

# Data interval for yfinance
INTERVAL = "5m"

# Prediction horizon
PREDICTION_HORIZON = "5min"

# US market hours (Eastern Time)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# Approximate number of 5-minute bars per trading day
# (6.5 hours * 60 minutes / 5 minutes = 78 bars)
BARS_PER_DAY = 78

# Minimum trading days needed for training
MIN_TRAINING_DAYS = 20

