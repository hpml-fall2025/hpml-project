import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self):
        pass

    def _get_latest_data(self) -> pd.DataFrame:

        # pull initial data from yfinance
        start_date = dt.datetime.today() - dt.timedelta(days=59)
        curr_data = yf.download("SPY", period="1y", interval="5m", start=start_date)
        curr_data = curr_data[curr_data['Volume'] != 0]
        curr_data = curr_data.dropna()

        # clean 5m interval data to process best for har-rv
        curr_data_cleaned = curr_data.rename(columns={"Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low"})
        curr_data_cleaned.columns = curr_data_cleaned.columns.droplevel("Ticker")
        curr_data_cleaned.index = curr_data_cleaned.index.tz_convert("America/New_York").tz_localize(None)
        curr_data_cleaned.index.name = "date"
        curr_data_cleaned.columns.name = None

        return curr_data_cleaned
    
    def __rv_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # calculate period
        df["D"] = df.index.date
        n_periods = df.pivot_table(index = ["D"], aggfunc = 'size').values
        df.loc[df["D"] != df["D"].shift(), "Per"] = n_periods
        df.fillna(method = 'ffill', inplace = True)
        
        # calculate daily returns
        df["Ret"] = np.where(df["D"] == df["D"].shift(), ( (df["close"]-df["close"].shift()) * 1/df["Per"] ) **2, np.nan)
        
        # calculate realized variance daily
        rv = df.groupby("D")["Ret"].agg(np.sum).to_frame()
        rv.columns = ["RV_daily"]
        rv["RV_daily"] = np.sqrt(rv["RV_daily"])

        # Compute weekly and monthly RV.  
        rv["RV_weekly"] = rv["RV_daily"].rolling(5).mean()
        rv["RV_monthly"] = rv["RV_daily"].rolling(21).mean()
        rv.dropna(inplace = True)

        return rv
    
    def __get_har_rv(self, data: pd.DataFrame) -> float:
        

