import pandas as pd
import numpy as np 
import warnings
import datetime as dt
from .volatility import predict_har_vol
from .news import predict_news_vol

class DynamicWeighting:
    def __init__(self):
        rolling_har_error = 0
        rolling_news_error = 1
        H_t = 0
        N_t = 0
        lambda_har = 1
        lambda_news = 1
    
    def normalize_news_vol(N_t):
        return

    def update_rolling_har_error(H_t, RV_t):
        return 1
    
    def update_rolling_news_error(N_t, RV_t):
        return 1
    

    def predict_weighted_vol(date) -> float:
        """
        Predicted V_t = H_t * rolling_news_error/(rolling_news_error + rolling_har_error) 
                        + N_t * rolling_har_error/(rolling_news_error + rolling_har_error) 
        """
        H_t = predict_har_vol(date)
        N_t = predict_news_vol(date)

        return (H_t * rolling_news_error + N_t * rolling_har_error) / (rolling_news_error + rolling_har_error)
    