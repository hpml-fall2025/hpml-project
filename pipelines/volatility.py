import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self):
        pass

    def get_latest_data(self):
        start_date = dt.datetime.today() - dt.timedelta(days=59)
        curr_data = yf.download("SPY", period="1y", interval="5m", start=start_date)
        curr_data = curr_data[curr_data['Volume'] != 0]
        curr_data = curr_data.dropna()
        return curr_data
    
    def get_har_rv(self, data: pd.DataFrame) -> float:
        curr_data = self.get_latest_data()
        har_rv = np.sqrt(np.mean(np.square(curr_data['Close'] - curr_data['Close'].shift(1))))
        

