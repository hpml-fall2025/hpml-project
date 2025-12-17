import numpy as np
import datetime as dt
from collections import deque
from typing import Tuple, Union, Optional

from volatility import VolatilityPipeline
from news import NewsPipeline


class SimpleWeighting:
    def __init__(
        self,
        lam: float = 0.1,                 # <-- new: scalar on news
        warmup_steps: int = 10,
        har_pipe: Optional[VolatilityPipeline] = None,
        news_pipe: Optional[NewsPipeline] = None,
        feature_weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05),
        delay_hours: int = 0,
        k: int = 10,
        norm_window: int = 20,
        eps: float = 1e-8,
    ):
        self.har_pipe = har_pipe if har_pipe is not None else VolatilityPipeline()
        self.news_pipe = news_pipe if news_pipe is not None else NewsPipeline()

        self.lam = float(lam)
        self.warmup_steps = int(warmup_steps)

        self.feature_weights = tuple(float(x) for x in feature_weights)
        self.delay_hours = int(delay_hours)
        self.k = int(k)

        self.norm_window = int(norm_window)
        self.eps = float(eps)

        self.har_hist = deque(maxlen=self.norm_window)
        self.news_hist = deque(maxlen=self.norm_window)

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
        H_t, _ = self.har_pipe.predict_har_vol(when)
        H_t = float(H_t)

        N_raw, _ = self.news_pipe.predict_news_vol(
            when,
            k=self.k,
            feature_weights=self.feature_weights,
            delay_hours=self.delay_hours,
        )
        N_raw = float(N_raw)

        self.har_hist.append(H_t)
        self.news_hist.append(N_raw)

        N_t = float(self._normalize_news(N_raw))

        if self.step < self.warmup_steps:
            V_t = H_t
        else:
            V_t = H_t + self.lam * N_t

        self.step += 1
        return float(V_t)