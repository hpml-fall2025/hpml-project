import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "finBERT"))
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, Union

from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
from pipelines.base import Pipeline


class NewsPipeline(Pipeline):
    def __init__(
        self,
        use_gpu: bool = True,
        short_window_hours: int = 6,
        medium_window_hours: int = 30,
        long_window_hours: int = 120,
    ):
        current_file = os.path.abspath(__file__)
        pipeline_dir = os.path.dirname(current_file)
        finbert_root = os.path.join(pipeline_dir, "finBERT")

        model_path = os.path.join(finbert_root, "models", "sentiment")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3, cache_dir=None
        )

        data_path = "data/headlines.csv"
        self.df = pd.read_csv(data_path)
        self.use_gpu = use_gpu

        self.short_window_hours = int(short_window_hours)
        self.medium_window_hours = int(medium_window_hours)
        self.long_window_hours = int(long_window_hours)

        if "URL" in self.df.columns:
            self.df = self.df.drop("URL", axis=1)

        self.df["Timestamp"] = pd.to_datetime(
            self.df["Timestamp"],
            format="%Y-%m-%dT%H:%M:%SZ",
            errors="raise",
            utc=True,
        )

        self.df["Timestamp"] = (
            self.df["Timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        )

        self.df["Timestamp_hour"] = self.df["Timestamp"].dt.floor("h")
        self._hour_cache = {}

    def _to_hour(self, when: Union[str, dt.datetime, pd.Timestamp]) -> dt.datetime:
        ts = pd.to_datetime(when)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        ts = ts.to_pydatetime()
        return ts.replace(minute=0, second=0, microsecond=0)

    def _get_hour_stats(self, hour_ts: dt.datetime) -> Tuple[float, int]:
        if hour_ts in self._hour_cache:
            return self._hour_cache[hour_ts]

        rows = self.df.loc[self.df["Timestamp_hour"] == hour_ts]
        if len(rows) == 0:
            self._hour_cache[hour_ts] = (0.0, 0)
            return 0.0, 0

        headlines = rows["Headline"].tolist()
        batch = " .".join(headlines)

        res = predict(batch, self.model, use_gpu=self.use_gpu)
        scores = res["sentiment_score"].values.astype(float)

        if len(scores) == 0:
            self._hour_cache[hour_ts] = (0.0, 0)
            return 0.0, 0

        avg_ss = float(np.mean(scores ** 2))
        cnt = int(len(scores))

        self._hour_cache[hour_ts] = (avg_ss, cnt)
        return avg_ss, cnt

    def _build_hourly_series(self, start_hour: dt.datetime, end_hour: dt.datetime) -> pd.DataFrame:
        hours = pd.date_range(start_hour, end_hour, freq="h")
        vals = np.zeros(len(hours), dtype=float)
        cnts = np.zeros(len(hours), dtype=int)

        for i, h in enumerate(hours):
            avg_ss, c = self._get_hour_stats(h.to_pydatetime())
            vals[i] = avg_ss
            cnts[i] = c

        return pd.DataFrame({"N_hourly": vals, "N_cnt": cnts}, index=hours.to_pydatetime())

    def predict_news_vol(
        self,
        when: Union[str, dt.datetime],
        k: int = 25,
        feature_weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        delay_hours: int = 1,
    ) -> Tuple[float, int]:
        target_hour = self._to_hour(when)
        prev_hour = target_hour - dt.timedelta(hours=int(delay_hours))

        end_hour = prev_hour
        start_hour = end_hour - dt.timedelta(hours=int(self.long_window_hours) + 8)

        series = self._build_hourly_series(start_hour, end_hour)
        if series.empty:
            raise ValueError("No news data found in requested hour window")

        x = series.copy()
        x["N_short"] = x["N_hourly"].rolling(window=self.short_window_hours).mean()
        x["N_medium"] = x["N_hourly"].rolling(window=self.medium_window_hours).mean()
        x["N_long"] = x["N_hourly"].rolling(window=self.long_window_hours).mean()

        x = x.dropna()
        if x.empty or prev_hour not in x.index:
            return 0.0, int(series["N_cnt"].sum())

        row = x.loc[prev_hour, ["N_hourly", "N_short", "N_medium", "N_long"]].astype(float).values
        n_used = int(series.loc[series.index <= prev_hour, "N_cnt"].sum())

        w = np.array(feature_weights, dtype=float)
        s = float(np.sum(np.abs(w)))
        if s > 0:
            w = w / s

        raw_val = float(np.dot(w, row))
        conf = (n_used / (n_used + int(k))) if n_used > 0 else 0.0
        return float(conf * raw_val), int(n_used)

    def get_headline(self, when: Union[str, dt.datetime], delay_hours: int = 1) -> str:
        target_hour = self._to_hour(when)
        prev_hour = target_hour - dt.timedelta(hours=int(delay_hours))
        rows = self.df.loc[self.df["Timestamp_hour"] == prev_hour]
        if len(rows) == 0:
            return ""
        return str(rows["Headline"].iloc[0])

    def get_latest_data(
        self,
        when: Union[str, dt.datetime],
        k: int = 25,
        feature_weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05),
        delay_hours: int = 1,
    ) -> dict:
        n_vol, n_cnt = self.predict_news_vol(
            when=when,
            k=k,
            feature_weights=feature_weights,
            delay_hours=delay_hours,
        )
        headline = self.get_headline(when, delay_hours=delay_hours)
        return {
            "news_rv": float(n_vol),
            "news_cnt": int(n_cnt),
            "headline": headline,
        }