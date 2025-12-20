import os
import datetime as dt
from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipelines.base import Pipeline


class VolatilityPipeline(Pipeline):
    def __init__(
        self,
        short_window_hours: int = 8,
        medium_window_hours: int = 12,
        long_window_hours: int = 275,
        train_end: Union[str, dt.datetime, pd.Timestamp] = "2021-01-01 23:59:59",
        stream_lookback_days: int = 31,
    ):
        self.model = None

        self.short_window_hours = int(short_window_hours)
        self.medium_window_hours = int(medium_window_hours)
        self.long_window_hours = int(long_window_hours)
        self._max_window = int(max(self.short_window_hours, self.medium_window_hours, self.long_window_hours))

        self.demo_mode = False
        self.demo_current_hour_idx = 0
        self.demo_start_hour_idx = 0
        self.ground_truth_rv = pd.Series(dtype=float)

        self.train_end = pd.to_datetime(train_end).to_pydatetime()
        self.stream_lookback_days = int(stream_lookback_days)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "SPY1min_clean.csv")

        model_data = pd.read_csv(csv_path)
        model_data["date"] = pd.to_datetime(model_data["date"])

        model_data["date"] = (
            model_data["date"]
            .dt.tz_localize("America/Denver")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )

        model_data = model_data.set_index("date").sort_index()
        model_data = model_data[~model_data.index.duplicated(keep="last")]
        model_data = model_data.dropna(subset=["close"])

        full_hist = model_data[["open", "high", "low", "close", "volume"]].copy()
        try:
            full_hist = full_hist.between_time("09:30", "16:59:59", inclusive="both")
        except TypeError:
            full_hist = full_hist.between_time("09:30", "16:59:59")

        self.full_hist_data = full_hist

        self._hour_index = pd.Index(self.full_hist_data.index.floor("h").unique()).sort_values()

        train_hist = self.full_hist_data.loc[:pd.Timestamp(self.train_end)]
        if train_hist.empty:
            raise ValueError("Training slice is empty (check train_end).")

        self._train_model(train_hist)

        self.reset_stream()

    def reset_stream(self):
        self._stream_initialized = False
        self._stream_pos = -1

        self._short_q = []
        self._med_q = []
        self._long_q = []
        self._short_sum = 0.0
        self._med_sum = 0.0
        self._long_sum = 0.0

        self._features_by_hour = {}

    def _train_model(self, data: pd.DataFrame):
        train_df = self.__rv_calculation(data)

        self.train_min = train_df.min()
        self.train_max = train_df.max()

        denom = (self.train_max - self.train_min).replace(0.0, np.nan)
        train_scaled = (train_df - self.train_min) / denom
        train_scaled = train_scaled.dropna()

        train_scaled["Target"] = train_scaled["RV_hourly"].shift(-1)
        train_scaled = train_scaled.dropna()

        X = train_scaled[["RV_hourly", "RV_short", "RV_medium", "RV_long"]]
        X = sm.add_constant(X)
        y = train_scaled["Target"]

        self.model = sm.OLS(y, X).fit()

    def __rv_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x = x.sort_index()
        x = x[~x.index.duplicated(keep="last")]

        x["H"] = x.index.floor("h")
        x["Per"] = x.groupby("H")["close"].transform("count")

        same_hour = x["H"] == x["H"].shift(1)
        diff = x["close"] - x["close"].shift(1)

        x["Ret"] = np.nan
        term = diff * (1.0 / x["Per"])
        x.loc[same_hour, "Ret"] = term ** 2

        rv = x.groupby("H")["Ret"].sum().to_frame("RV_hourly")
        rv["RV_hourly"] = np.sqrt(rv["RV_hourly"])

        rv["RV_short"] = rv["RV_hourly"].rolling(window=self.short_window_hours).mean()
        rv["RV_medium"] = rv["RV_hourly"].rolling(window=self.medium_window_hours).mean()
        rv["RV_long"] = rv["RV_hourly"].rolling(window=self.long_window_hours).mean()

        rv = rv.dropna()
        rv = rv[~rv.index.duplicated(keep="last")].sort_index()
        return rv

    def _rv_one_hour(self, hour_ts: pd.Timestamp) -> float:
        h0 = pd.Timestamp(hour_ts).floor("h")
        h1 = h0 + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)

        dfh = self.full_hist_data.loc[h0:h1]
        if dfh.empty:
            raise ValueError("No intraday data for that hour bucket")

        close = dfh["close"].to_numpy(dtype=float)
        per = int(close.shape[0])
        if per < 2:
            return 0.0

        diffs = np.diff(close)
        term = diffs * (1.0 / per)
        rv = float(np.sqrt(np.sum(term * term)))
        return rv

    def _push_roll(self, q, sname, x, win):
        if win <= 0:
            return
        q.append(float(x))
        if sname == "short":
            self._short_sum += float(x)
            if len(q) > win:
                self._short_sum -= q.pop(0)
        elif sname == "med":
            self._med_sum += float(x)
            if len(q) > win:
                self._med_sum -= q.pop(0)
        else:
            self._long_sum += float(x)
            if len(q) > win:
                self._long_sum -= q.pop(0)

    def _roll_mean(self, sname, q, win):
        if len(q) < win:
            return float("nan")
        if sname == "short":
            return float(self._short_sum / win)
        if sname == "med":
            return float(self._med_sum / win)
        return float(self._long_sum / win)

    def _process_hour(self, hour_ts: pd.Timestamp):
        rv_h = self._rv_one_hour(hour_ts)

        self._push_roll(self._short_q, "short", rv_h, self.short_window_hours)
        self._push_roll(self._med_q, "med", rv_h, self.medium_window_hours)
        self._push_roll(self._long_q, "long", rv_h, self.long_window_hours)

        rv_s = self._roll_mean("short", self._short_q, self.short_window_hours)
        rv_m = self._roll_mean("med", self._med_q, self.medium_window_hours)
        rv_l = self._roll_mean("long", self._long_q, self.long_window_hours)

        h = pd.Timestamp(hour_ts).floor("h")
        self._features_by_hour[h] = (rv_h, rv_s, rv_m, rv_l)

        if len(self._features_by_hour) > (self._max_window + 64):
            keys = sorted(self._features_by_hour.keys())
            for k in keys[: max(0, len(keys) - (self._max_window + 64))]:
                del self._features_by_hour[k]

    def _ensure_stream_until(self, hour_ts: pd.Timestamp):
        h = pd.Timestamp(hour_ts).floor("h")
        if len(self._hour_index) == 0:
            raise ValueError("Empty hour index")

        pos = int(self._hour_index.searchsorted(h, side="left"))
        if pos >= len(self._hour_index) or pd.Timestamp(self._hour_index[pos]) != h:
            raise ValueError("Requested hour not present in data")

        if not self._stream_initialized:
            lb = h - pd.Timedelta(days=self.stream_lookback_days)
            lb_pos = int(self._hour_index.searchsorted(lb, side="left"))
            self._stream_pos = lb_pos - 1
            self._stream_initialized = True

        while self._stream_pos < pos:
            self._stream_pos += 1
            self._process_hour(pd.Timestamp(self._hour_index[self._stream_pos]))

    def predict_har_vol(self, when: Union[str, dt.datetime]) -> Tuple[float, float]:
        if self.model is None:
            raise ValueError("Model not initialized")

        if isinstance(when, str):
            target_ts = pd.to_datetime(when).to_pydatetime()
        elif isinstance(when, dt.datetime):
            target_ts = when
        else:
            raise TypeError("when must be str or datetime")

        target_hour = pd.Timestamp(target_ts).floor("h")
        prev_hour = target_hour - pd.Timedelta(hours=1)

        self._ensure_stream_until(prev_hour)

        if prev_hour not in self._features_by_hour:
            raise ValueError("No RV features found for previous hour")

        rv_h, rv_s, rv_m, rv_l = self._features_by_hour[prev_hour]
        if not np.isfinite(rv_s) or not np.isfinite(rv_m) or not np.isfinite(rv_l):
            raise ValueError("Not enough history for rolling windows")

        cols = ["RV_hourly", "RV_short", "RV_medium", "RV_long"]
        train_min = self.train_min[cols]
        train_max = self.train_max[cols]
        denom = (train_max - train_min).replace(0.0, np.nan)

        feats = pd.Series(
            {"RV_hourly": rv_h, "RV_short": rv_s, "RV_medium": rv_m, "RV_long": rv_l},
            dtype=float,
        )
        feats_scaled = (feats - train_min) / denom
        if feats_scaled.isna().any():
            raise ValueError("Degenerate scaling")

        exog_names = list(self.model.params.index)
        x = []
        for name in exog_names:
            if name == "const":
                x.append(1.0)
            else:
                x.append(float(feats_scaled[name]))
        x = np.asarray(x, dtype=float).reshape(1, -1)

        pred = float(self.model.predict(x)[0])
        true_prev_rv = float(rv_h)
        return pred, true_prev_rv

    def get_latest_data(self) -> dict:
        return {"error": "get_latest_data not used in backtest dashboard"}