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
        medium_window_hours: int = 24,
        long_window_hours: int = 75,
        stream_lookback_hours: int | None = None,
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

        self.df["Timestamp_hour"] = pd.to_datetime(self.df["Timestamp"].dt.floor("h")).dt.floor("h")
        self._hour_cache = {}

        tmin = pd.Timestamp(self.df["Timestamp_hour"].min()).floor("h")
        tmax = pd.Timestamp(self.df["Timestamp_hour"].max()).floor("h")
        self._hour_index = pd.date_range(tmin, tmax, freq="h")

        self._max_window = int(max(self.short_window_hours, self.medium_window_hours, self.long_window_hours))
        self._lookback_len = int(self.long_window_hours) + 8 + 1
        if stream_lookback_hours is None:
            stream_lookback_hours = int(self.long_window_hours) + 8
        self.stream_lookback_hours = int(stream_lookback_hours)

        self.reset_stream()

    def reset_stream(self):
        self._stream_initialized = False
        self._stream_pos = -1

        self._q_short = []
        self._q_med = []
        self._q_long = []
        self._sum_short = 0.0
        self._sum_med = 0.0
        self._sum_long = 0.0

        self._q_cnt = []
        self._sum_cnt = 0

        self._by_hour = {}

    def _to_hour(self, when: Union[str, dt.datetime, pd.Timestamp]) -> pd.Timestamp:
        ts = pd.Timestamp(when)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        return ts.floor("h")

    def _get_hour_stats(self, hour_ts: Union[dt.datetime, pd.Timestamp]) -> Tuple[float, int]:
        h = pd.Timestamp(hour_ts).floor("h")

        if h in self._hour_cache:
            return self._hour_cache[h]

        rows = self.df.loc[self.df["Timestamp_hour"] == h]
        if len(rows) == 0:
            self._hour_cache[h] = (0.0, 0)
            return 0.0, 0

        headlines = rows["Headline"].tolist()
        batch = " .".join(headlines)

        res = predict(batch, self.model, use_gpu=self.use_gpu)
        scores = res["sentiment_score"].values.astype(float)

        if len(scores) == 0:
            self._hour_cache[h] = (0.0, 0)
            return 0.0, 0

        avg_ss = float(np.mean(scores ** 2))
        cnt = int(len(scores))

        self._hour_cache[h] = (avg_ss, cnt)
        return avg_ss, cnt

    def _push_roll(self, q, which: str, x: float, win: int):
        q.append(float(x))
        if which == "short":
            self._sum_short += float(x)
            if len(q) > win:
                self._sum_short -= float(q.pop(0))
        elif which == "med":
            self._sum_med += float(x)
            if len(q) > win:
                self._sum_med -= float(q.pop(0))
        else:
            self._sum_long += float(x)
            if len(q) > win:
                self._sum_long -= float(q.pop(0))

    def _mean_roll(self, which: str, q, win: int) -> float:
        if len(q) < win:
            return float("nan")
        if which == "short":
            return float(self._sum_short / win)
        if which == "med":
            return float(self._sum_med / win)
        return float(self._sum_long / win)

    def _push_cnt(self, c: int):
        self._q_cnt.append(int(c))
        self._sum_cnt += int(c)
        if len(self._q_cnt) > self._lookback_len:
            self._sum_cnt -= int(self._q_cnt.pop(0))

    def _process_hour(self, h: pd.Timestamp):
        v, c = self._get_hour_stats(h)

        self._push_roll(self._q_short, "short", v, self.short_window_hours)
        self._push_roll(self._q_med, "med", v, self.medium_window_hours)
        self._push_roll(self._q_long, "long", v, self.long_window_hours)
        self._push_cnt(c)

        s = self._mean_roll("short", self._q_short, self.short_window_hours)
        m = self._mean_roll("med", self._q_med, self.medium_window_hours)
        l = self._mean_roll("long", self._q_long, self.long_window_hours)

        hh = pd.Timestamp(h).floor("h")
        self._by_hour[hh] = (float(v), float(s), float(m), float(l), int(self._sum_cnt))

        keep = self._max_window + self._lookback_len + 64
        if len(self._by_hour) > keep:
            keys = sorted(self._by_hour.keys())
            for k in keys[: max(0, len(keys) - keep)]:
                del self._by_hour[k]

    def _ensure_stream_until(self, hour_ts: pd.Timestamp):
        h = pd.Timestamp(hour_ts).floor("h")

        pos = int(self._hour_index.searchsorted(h, side="left"))
        if pos < 0 or pos >= len(self._hour_index) or pd.Timestamp(self._hour_index[pos]) != h:
            raise ValueError("Requested hour not present in hour index")

        if not self._stream_initialized:
            lb = h - pd.Timedelta(hours=int(self.stream_lookback_hours))
            lb_pos = int(self._hour_index.searchsorted(lb, side="left"))
            self._stream_pos = lb_pos - 1
            self._stream_initialized = True

        while self._stream_pos < pos:
            self._stream_pos += 1
            self._process_hour(pd.Timestamp(self._hour_index[self._stream_pos]))

    def predict_news_vol(
        self,
        when: Union[str, dt.datetime],
        k: int = 25,
        feature_weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        delay_hours: int = 1,
    ) -> Tuple[float, int]:
        target_hour = self._to_hour(when)
        prev_hour = target_hour - pd.Timedelta(hours=int(delay_hours))

        self._ensure_stream_until(prev_hour)

        if prev_hour not in self._by_hour:
            return 0.0, 0

        v, s, m, l, cnt_sum = self._by_hour[prev_hour]
        if not (np.isfinite(s) and np.isfinite(m) and np.isfinite(l)):
            return 0.0, int(cnt_sum)

        row = np.array([float(v), float(s), float(m), float(l)], dtype=float)

        w = np.array(feature_weights, dtype=float)
        ws = float(np.sum(np.abs(w)))
        if ws > 0:
            w = w / ws

        raw_val = float(np.dot(w, row))
        n_used = int(cnt_sum)
        conf = (n_used / (n_used + int(k))) if n_used > 0 else 0.0
        return float(conf * raw_val), int(n_used)

    def get_headline(self, when: Union[str, dt.datetime], delay_hours: int = 1) -> str:
        target_hour = self._to_hour(when)
        prev_hour = target_hour - pd.Timedelta(hours=int(delay_hours))

        rows = self.df.loc[self.df["Timestamp_hour"] == prev_hour]
        if len(rows) == 0:
            return "No headlines in this hour."
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
        return {"news_rv": float(n_vol), "news_cnt": int(n_cnt), "headline": headline}