import numpy as np
import datetime as dt
from collections import deque
from typing import Tuple, Union, Optional

from pipelines.news import NewsPipeline
from pipelines.volatility import VolatilityPipeline

mu_h = 0.04556763811529785
sd_h = 0.020110056862456746
mu_n = -0.0043576259684375545
sd_n = 0.0011150798493509803


class Weighting:
    def __init__(
        self,
        lam: float = -0.1,
        warmup_steps: int = 10,
        har_pipe: Optional[VolatilityPipeline] = None,
        news_pipe: Optional[NewsPipeline] = None,
        feature_weights: Tuple[float, float, float, float] = (0.0043954775025780105, -0.004191981733214654, -0.06672588362936503, 0.04938762749608129),
        delay_hours: int = 0,
        k: int = 0,
        eps: float = 1e-8,
    ):
        self.har_pipe = har_pipe if har_pipe is not None else VolatilityPipeline()
        self.news_pipe = news_pipe if news_pipe is not None else NewsPipeline()

        self.lam = float(lam)
        self.warmup_steps = int(warmup_steps)

        self.feature_weights = tuple(float(x) for x in feature_weights)
        self.delay_hours = int(delay_hours)
        self.k = int(k)

        self.eps = float(eps)

        self.step = 0

    def _normalize_news(self, N_raw: float) -> float:
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


        N_t = float(self._normalize_news(N_raw))

        V_t = H_t + self.lam * N_t

        self.step += 1
        return float(V_t)