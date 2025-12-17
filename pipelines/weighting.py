import pandas as pd
import numpy as np 
import warnings
import datetime as dt
from collections import deque
from volatility import VolatilityPipeline
from news import NewsPipeline

class DynamicWeighting:
    def __init__(
        self,
        lambda_har=0.2,
        lambda_news=0.2,
        norm_window=10,
        warmup_steps=10,
        eps=1e-8
    ):
        self.har_pipe = VolatilityPipeline()
        self.news_pipe = NewsPipeline()

        self.lambda_har = float(lambda_har)
        self.lambda_news = float(lambda_news)
        self.eps = float(eps)

        self.norm_window = int(norm_window)
        self.warmup_steps = int(warmup_steps)

        self.rolling_har_error = 1.0
        self.rolling_news_error = 1.0

        self.har_hist = deque(maxlen=self.norm_window)
        self.news_hist = deque(maxlen=self.norm_window)

        self.prev_date = None
        self.prev_H = None
        self.prev_N = None

        self.step_count = 0

    def _normalize_news(self, N_t: float) -> float:
        if len(self.har_hist) < 2 or len(self.news_hist) < 2:
            return N_t

        mu_h = float(np.mean(self.har_hist))
        sd_h = float(np.std(self.har_hist, ddof=0))
        mu_n = float(np.mean(self.news_hist))
        sd_n = float(np.std(self.news_hist, ddof=0))

        return mu_h + (sd_h + self.eps) * (N_t - mu_n) / (sd_n + self.eps)

    def _update_rolling_errors(self, RV_prev: float):
        if self.prev_H is None or self.prev_N is None:
            return

        har_sq = float((RV_prev - self.prev_H) ** 2)
        news_sq = float((RV_prev - self.prev_N) ** 2)

        self.rolling_har_error = self.lambda_har * har_sq + (1.0 - self.lambda_har) * self.rolling_har_error
        self.rolling_news_error = self.lambda_news * news_sq + (1.0 - self.lambda_news) * self.rolling_news_error

    def predict_weighted_vol(self, date) -> float:
        """
        Predicted V_t = H_t * rolling_news_error/(rolling_news_error + rolling_har_error) 
                        + N_t * rolling_har_error/(rolling_news_error + rolling_har_error) 
        """

        H_t, RV_prev = self.har_pipe.predict_har_vol(date)
        H_t = float(H_t)

        if RV_prev is not None:
            self._update_rolling_errors(float(RV_prev))

        N_raw, num_headlines = self.news_pipe.predict_news_vol(date)
        N_raw = float(N_raw)

        self.har_hist.append(H_t)
        self.news_hist.append(N_raw)

        N_t = float(self._normalize_news(N_raw))

        denom = self.rolling_news_error + self.rolling_har_error + self.eps

        if self.step_count < self.warmup_steps:
            V_t = H_t
        else:
            V_t = (H_t * self.rolling_news_error + N_t * self.rolling_har_error) / denom

        self.prev_date = date
        self.prev_H = H_t
        self.prev_N = N_t
        self.step_count += 1

        return float(V_t)