import pandas as pd
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import register_matplotlib_converters
from pandas.plotting import autocorrelation_plot
from pandas_datareader import data
from scipy import stats
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


    def __get_har_rv(self, df: pd.DataFrame, rv: pd.DataFrame) -> float:

        # construct volume df for daily volume
        volume = df[["volume"]]
        volume = volume.resample('D')["volume"].sum().reset_index()
        volume = volume.set_index("date")

        rv["SPY_volume"] = volume.loc[rv.index]

        rv["Target"] = rv["RV_daily"].shift(-1)
        rv.dropna(inplace = True)

        rv_scaled = (rv - rv.min()) / (rv.max() - rv.min())
        rv_scaled = sm.add_constant(rv_scaled)

        return rv_scaled

    def main(self):

        curr_data = _get_latest_data()
        curr_rv = __rv_calculation(curr_data)
        curr_rv_scaled = __get_har_rv(curr_rv)

        model_data = pd.read_csv("SPY1min_clean.csv", parse_dates=True)
        model_rv = __rv_calculation(model_data)
        model_rv_scaled = __get_har_rv(model_rv)






        
