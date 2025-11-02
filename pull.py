"""
Data fetching module using yfinance for 5-minute historical stock data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz


def get_historical_data(ticker: str, target_date: str, lookback_days: int) -> pd.DataFrame:
    """Fetch historical 5-minute stock data using yfinance."""
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        
        start_date = target_dt - timedelta(days=lookback_days)
        end_date = target_dt + timedelta(days=1)
        
        print(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")
        
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="5m"
        )
        
        if data.empty:
            raise ValueError(f"No data returned from yfinance for {ticker}")
        
        data = _filter_market_hours(data)
        
        if data.empty:
            raise ValueError(f"No data available during market hours for {ticker}")
        
        _validate_data(data, target_date)
        
        data = data.sort_index()
        
        print(f"Successfully fetched {len(data)} bars of 5-minute data")
        
        return data
        
    except ValueError as e:
        raise ValueError(f"Data validation error: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to fetch data from yfinance: {str(e)}")


def _filter_market_hours(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data to regular US market hours (9:30 AM - 4:00 PM EST)."""
    est = pytz.timezone('America/New_York')
    
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    
    data.index = data.index.tz_convert(est)
    
    market_hours_mask = (
        (data.index.hour > 9) | ((data.index.hour == 9) & (data.index.minute >= 30))
    ) & (
        (data.index.hour < 16)
    )
    
    return data[market_hours_mask]


def _validate_data(data: pd.DataFrame, target_date: str) -> None:
    """Validate fetched data for completeness and quality."""
    if len(data) == 0:
        raise ValueError("No data available after filtering to market hours")
    
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    target_data = data[data.index.date == target_dt.date()]
    
    if len(target_data) == 0:
        raise ValueError(
            f"Target date {target_date} has no data. "
            f"It may be a weekend, holiday, or outside the available data range."
        )
    
    if len(target_data) < 50:
        print(
            f"Warning: Target date has only {len(target_data)} bars "
            f"(expected ~78 for a full trading day)"
        )
    
    unique_dates = data.index.date
    num_trading_days = len(set(unique_dates))
    
    if num_trading_days < 20:
        print(
            f"Warning: Only {num_trading_days} trading days available. "
            f"Minimum 20 recommended for model training."
        )
    
    if data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
        print("Warning: Data contains missing values in OHLCV columns")

