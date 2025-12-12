import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self):
        pass

    def get_latest_data(self):
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
    
    def har_rv_calculation(self):
        df = self.get_latest_data() 
        df["D"] = df.index.date
        n_periods = df.pivot_table(index = ["D"], aggfunc = 'size').values
        df.loc[df["D"] != df["D"].shift(), "Per"] = n_periods
        df.fillna(method = 'ffill', inplace = True)
        df["Ret"] = np.where(df["D"] == df["D"].shift(), ( (df["close"]-df["close"].shift()) * 1/df["Per"] ) **2, np.nan)
        rv = alt_df.groupby("D")["Ret"].agg(np.sum).to_frame()
        rv.columns = ["RV_daily"]
        rv["RV_daily"] = np.sqrt(rv["RV_daily"])
        date = str(rv["RV_daily"].idxmax())
        df["close"].loc[date]
        date = str(rv["RV_daily"].idxmax())
        plt.plot(df["close"].loc[date], label = f"{sample} close price")
        plt.title(f"{ticker} on {date}")
        plt.legend()
        plt.show()

    def get_har_rv(self, data: pd.DataFrame) -> float:
        pass
        

