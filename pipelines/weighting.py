import numpy as np
import datetime as dt
from collections import deque
from typing import Tuple, Union, Optional

from volatility import VolatilityPipeline
from news import NewsPipeline


class DynamicWeighting:
    def __init__(
        self,
        lambda_har: float = 0.2,
        lambda_news: float = 0.2,
        warmup_steps: int = 10,
        har_pipe: Optional[VolatilityPipeline] = None,
        news_pipe: Optional[NewsPipeline] = None,
        news_feature_weights: Optional[Tuple[float, float, float, float]] = (0.5, 0.3, 0.15, 0.05),
        feature_weights: Optional[Tuple[float, float, float, float]] = None,  # alias
        news_delay_hours: int = 1,
        news_k: int = 25,
        norm_window: int = 10,
        eps: float = 1e-8,
    ):
        self.har_pipe = har_pipe if har_pipe is not None else VolatilityPipeline()
        self.news_pipe = news_pipe if news_pipe is not None else NewsPipeline()

        self.lambda_har = float(lambda_har)
        self.lambda_news = float(lambda_news)
        self.warmup_steps = int(warmup_steps)

        if feature_weights is not None:
            news_feature_weights = feature_weights
        self.news_feature_weights = tuple(float(x) for x in (news_feature_weights or (0.5, 0.3, 0.15, 0.05)))

        self.news_delay_hours = int(news_delay_hours)
        self.news_k = int(news_k)

        self.norm_window = int(norm_window)
        self.eps = float(eps)

        self.har_hist = deque(maxlen=self.norm_window)
        self.news_hist = deque(maxlen=self.norm_window)

        self.rolling_har_error = 1.0
        self.rolling_news_error = 1.0

        self.prev_H = None
        self.prev_N = None
        self.step = 0

    def _normalize_news(self, N_raw: float) -> float:
        if len(self.har_hist) < 2 or len(self.news_hist) < 2:
            return float(N_raw)

        mu_h = float(np.mean(self.har_hist))
        sd_h = float(np.std(self.har_hist, ddof=0))
        mu_n = float(np.mean(self.news_hist))
        sd_n = float(np.std(self.news_hist, ddof=0))

        return mu_h + (sd_h + self.eps) * (float(N_raw) - mu_n) / (sd_n + self.eps)

    def predict_weighted_vol(self, when: Union[str, dt.datetime]) -> float:
        H_t, true_prev_rv = self.har_pipe.predict_har_vol(when)
        H_t = float(H_t)
        true_prev_rv = float(true_prev_rv)

        N_raw, _ = self.news_pipe.predict_news_vol(
            when,
            k=self.news_k,
            feature_weights=self.news_feature_weights,
            delay_hours=self.news_delay_hours,
        )
        N_raw = float(N_raw)

        self.har_hist.append(H_t)
        self.news_hist.append(N_raw)
        N_t = float(self._normalize_news(N_raw))

        if self.prev_H is not None and self.prev_N is not None:
            e_h = float((true_prev_rv - self.prev_H) ** 2)
            e_n = float((true_prev_rv - self.prev_N) ** 2)
            self.rolling_har_error = self.lambda_har * e_h + (1.0 - self.lambda_har) * self.rolling_har_error
            self.rolling_news_error = self.lambda_news * e_n + (1.0 - self.lambda_news) * self.rolling_news_error

        if self.step < self.warmup_steps:
            V_t = H_t
        else:
            denom = self.rolling_news_error + self.rolling_har_error + self.eps
            V_t = (H_t * self.rolling_news_error + N_t * self.rolling_har_error) / denom

        self.prev_H = H_t
        self.prev_N = N_t
        self.step += 1
        return float(V_t)