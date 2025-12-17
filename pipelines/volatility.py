import os
import datetime as dt
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from .base import Pipeline
except Exception:
    from base import Pipeline


class VolatilityPipeline(Pipeline):
    def __init__(
        self,
        short_window_hours: int = 6,
        medium_window_hours: int = 30,
        long_window_hours: int = 120,
    ):
        self.model = None

        self.short_window_hours = int(short_window_hours)
        self.medium_window_hours = int(medium_window_hours)
        self.long_window_hours = int(long_window_hours)

        self.demo_mode = False
        self.demo_current_hour_idx = 0
        self.demo_start_hour_idx = 0
        self.ground_truth_rv = pd.Series(dtype=float)
        self.full_hour_index = None

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "SPY1min_clean.csv")

        try:
            model_data = pd.read_csv(csv_path)
            model_data["date"] = pd.to_datetime(model_data["date"])
            model_data = model_data.set_index("date").sort_index()

            model_data = model_data[~model_data.index.duplicated(keep="last")]
            model_data = model_data.dropna(subset=["close"])

            self.full_hist_data = model_data[["open", "high", "low", "close", "volume"]].copy()

            full_rv = self.__rv_calculation(self.full_hist_data)
            self.full_hour_index = full_rv.index.to_list()

            self._train_model(self.full_hist_data)

        except Exception as e:
            print(f"Error initializing VolatilityPipeline: {e}")
            self.model = None

    def _train_model(self, data: pd.DataFrame):
        try:
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

        except Exception as e:
            print(f"Error training model: {e}")
            self.model = None

    def setup_demo_mode(self, test_ratio: float = 0.2):
        if self.full_hour_index is None or len(self.full_hour_index) == 0:
            raise ValueError("Hourly index not available for demo mode")

        self.demo_mode = True

        n = len(self.full_hour_index)
        split_idx = int(n * (1.0 - float(test_ratio)))
        split_idx = max(0, min(split_idx, n - 1))

        self.demo_start_hour_idx = split_idx
        self.demo_current_hour_idx = split_idx

        train_end_hour = self.full_hour_index[split_idx]
        train_end_ts = train_end_hour + dt.timedelta(hours=1) - dt.timedelta(microseconds=1)
        train_data = self.full_hist_data.loc[:train_end_ts]

        self._train_model(train_data)

        full_rv = self.__rv_calculation(self.full_hist_data)
        self.ground_truth_rv = full_rv["RV_hourly"].shift(-1)

    def _get_latest_data(self) -> pd.DataFrame:
        if not self.demo_mode:
            return pd.DataFrame()

        if self.full_hour_index is None or len(self.full_hour_index) == 0:
            return pd.DataFrame()

        idx = self.demo_current_hour_idx
        if idx < 0 or idx >= len(self.full_hour_index):
            return pd.DataFrame()

        end_hour = self.full_hour_index[idx]
        end_ts = end_hour + dt.timedelta(hours=1) - dt.timedelta(microseconds=1)

        lookback_start = end_ts - dt.timedelta(days=30)

        return self.full_hist_data.loc[lookback_start:end_ts]

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

    def predict_har_vol(self, when: Union[str, dt.datetime]) -> Tuple[float, float]:
        if self.model is None:
            raise ValueError("Model not initialized")

        if isinstance(when, str):
            target_ts = pd.to_datetime(when).to_pydatetime()
        elif isinstance(when, dt.datetime):
            target_ts = when
        else:
            raise TypeError("when must be str or datetime")

        target_hour = target_ts.replace(minute=0, second=0, microsecond=0)
        prev_hour = target_hour - dt.timedelta(hours=1)

        prev_end = prev_hour + dt.timedelta(hours=1) - dt.timedelta(microseconds=1)
        lookback_start = prev_end - dt.timedelta(days=30)

        curr_data = self.full_hist_data.loc[lookback_start:prev_end]
        if curr_data.empty:
            raise ValueError("No intraday data found for requested hour window")

        curr_rv = self.__rv_calculation(curr_data)
        if curr_rv.empty:
            raise ValueError("Not enough data to compute hourly RV features")
        if prev_hour not in curr_rv.index:
            raise ValueError("No RV features found for previous hour")

        cols = ["RV_hourly", "RV_short", "RV_medium", "RV_long"]

        denom = (self.train_max[cols] - self.train_min[cols]).replace(0.0, np.nan)
        feats_scaled = (curr_rv[cols] - self.train_min[cols]) / denom
        feats_scaled = feats_scaled.dropna()
        if prev_hour not in feats_scaled.index:
            raise ValueError("Previous hour dropped during scaling (degenerate scaling)")

        last_row = feats_scaled.loc[[prev_hour], cols].copy()
        last_row["const"] = 1.0
        last_row = last_row[self.model.params.index]

        pred = float(self.model.predict(last_row).iloc[0])
        true_prev_rv = float(curr_rv.loc[prev_hour, "RV_hourly"])

        return pred, true_prev_rv

    def get_latest_data(self) -> dict:
        if self.model is None:
            return {"error": "Model not initialized"}

        curr_data = self._get_latest_data()
        if curr_data.empty:
            return {"error": "No data retrieved"}

        curr_rv = self.__rv_calculation(curr_data)
        if curr_rv.empty:
            return {"error": "Not enough data for hourly rolling windows"}

        cols = ["RV_hourly", "RV_short", "RV_medium", "RV_long"]
        denom = (self.train_max[cols] - self.train_min[cols]).replace(0.0, np.nan)
        curr_scaled = (curr_rv[cols] - self.train_min[cols]) / denom
        curr_scaled = curr_scaled.dropna()
        if curr_scaled.empty:
            return {"error": "Scaling produced empty features"}

        last_hour = curr_scaled.index[-1]
        last_row = curr_scaled.iloc[[-1]].copy()
        last_row["const"] = 1.0
        last_row = last_row[self.model.params.index]

        prediction = float(self.model.predict(last_row).iloc[0])
        result = {"volatility_prediction": prediction, "timestamp": str(last_hour)}

        if self.demo_mode:
            current_hour = last_hour
            if self.ground_truth_rv is not None and current_hour in self.ground_truth_rv.index:
                actual = self.ground_truth_rv.loc[current_hour]
                result["actual_next_hour_rv"] = float(actual) if pd.notna(actual) else None
            else:
                result["actual_next_hour_rv"] = None

            self.demo_current_hour_idx += 1
            if self.demo_current_hour_idx >= len(self.full_hour_index):
                result["demo_ended"] = True

        return result