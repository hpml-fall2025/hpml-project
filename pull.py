"""
Data fetching module using yfinance for 5-minute historical stock data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from typing import List


def get_historical_data(ticker: str, data_interval: List[str], time_interval: str) -> pd.DataFrame:
    """Fetch historical 5-minute stock data using yfinance."""
    try:
        start_date = datetime.strptime(data_interval[0], "%Y-%m-%d") - timedelta(days = 7)
        
        end_date = datetime.strptime(data_interval[1], "%Y-%m-%d") + timedelta(days = 1)
        
        print(f"Fetching {ticker} data from {start_date.date()} up to {end_date.date()}...")
        
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=time_interval
        )
        
        if data.empty:
            raise ValueError(f"No available data returned from yfinance for {ticker}")
        
        _validate_data(data)
        
        data = data.sort_index()
        
        print(f"Successfully fetched {len(data)} bars of 5-minute data")
        
        return data
        
    except ValueError as e:
        raise ValueError(f"Data validation error: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to fetch data from yfinance: {str(e)}")



def _validate_data(data: pd.DataFrame) -> None:
    """Validate fetched data for completeness and quality."""
    if len(data) == 0:
        raise ValueError("No data available after filtering to market hours")
    
    if data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
        print("Warning: Data contains missing values in OHLCV columns")

