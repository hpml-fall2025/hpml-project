import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "finBERT"))
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import pandas as pd
import datetime
import csv
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

        # Convert UTC -> NY time (so “hour” aligns with market hours better)
        # and then drop tz for easy indexing
        self.df["Timestamp"] = (
            self.df["Timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        )

        self.df["Timestamp_hour"] = self.df["Timestamp"].dt.floor("H")

        self._hour_cache = {}

    def _get_hour_stats(self, hour_ts: dt.datetime) -> Tuple[float, int]:
        """
        Returns (mean(sentiment_score^2) for that hour, headline_count).
        Cached per hour to avoid re-running FinBERT repeatedly.
        """
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

    def _build_hourly_series(
        self,
        start_hour: dt.datetime,
        end_hour: dt.datetime,
    ) -> pd.DataFrame:
        """
        Builds a continuous hourly index [start_hour, end_hour] inclusive,
        and fills each hour with (avg_ss, cnt).
        Missing hours get zeros.
        """
        hours = pd.date_range(start_hour, end_hour, freq="H")
        vals = np.zeros(len(hours), dtype=float)
        cnts = np.zeros(len(hours), dtype=int)

        for i, h in enumerate(hours):
            avg_ss, c = self._get_hour_stats(h.to_pydatetime())
            vals[i] = avg_ss
            cnts[i] = c

        out = pd.DataFrame({"N_hourly": vals, "N_cnt": cnts}, index=hours.to_pydatetime())
        return out

    def predict_news_vol(
        self,
        when: Union[str, dt.datetime],
        k: int = 25,
        feature_weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        delay_hours: int = 1,
    ) -> Tuple[float, int]:
        """
        Hourly news-vol proxy.

        For target hour H (floor to hour), we DO NOT use H’s headlines directly.
        We use up through H - delay_hours (default=1) to avoid “future” info.

        We compute HAR-style features from hourly series:
          N_hourly: value at prev_hour
          N_short:  rolling mean over last short_window_hours
          N_medium: rolling mean over last medium_window_hours
          N_long:   rolling mean over last long_window_hours

        Return:
          (confidence_shrunk_weighted_news_value, headline_count_used)

        feature_weights = (w0, wS, wM, wL) on (N_hourly, N_short, N_medium, N_long).
        """
        if isinstance(when, str):
            target_ts = pd.to_datetime(when).to_pydatetime()
        elif isinstance(when, dt.datetime):
            target_ts = when
        else:
            raise TypeError("when must be str or datetime")

        target_hour = target_ts.replace(minute=0, second=0, microsecond=0)
        prev_hour = target_hour - dt.timedelta(hours=int(delay_hours))

        end_hour = prev_hour
        start_hour = end_hour - dt.timedelta(hours=int(self.long_window_hours) + 5)

        series = self._build_hourly_series(start_hour, end_hour)
        if series.empty:
            raise ValueError("No news data found in requested hour window")

        x = series.copy()

        x["N_short"] = x["N_hourly"].rolling(window=self.short_window_hours).mean()
        x["N_medium"] = x["N_hourly"].rolling(window=self.medium_window_hours).mean()
        x["N_long"] = x["N_hourly"].rolling(window=self.long_window_hours).mean()

        x = x.dropna()
        if x.empty or prev_hour not in x.index:
            raise ValueError("Not enough history to compute rolling hourly news features")

        row = x.loc[prev_hour, ["N_hourly", "N_short", "N_medium", "N_long"]].astype(float).values
        n_used = int(series.loc[series.index <= prev_hour, "N_cnt"].sum())

        w = np.array(feature_weights, dtype=float)
        s = float(np.sum(np.abs(w)))
        if s > 0:
            w = w / s

        raw_val = float(np.dot(w, row))
        conf = (n_used / (n_used + int(k))) if n_used > 0 else 0.0

        return float(conf * raw_val), int(n_used)

            
    def get_latest_data(self, query_date=None) -> dict:
        """
        Returns news-based volatility signal for a given date.
        
        Args:
            query_date: Date to query (defaults to today for live mode)
            
        Returns:
            dict with 'news_rv' key containing the volatility signal
        """
        if query_date is None:
            query_date = datetime.date.today()
        
        day_weights = [0.5, 0.25, 0.13, 0.07, 0.03]  
        
        vol = 0
        
        for i in range(len(day_weights)):
            check_date = query_date - datetime.timedelta(days=i)
            mask = self.df["Timestamp"] == check_date
            day_rows = self.df.loc[mask]
            
            if len(day_rows)==0:
                continue
            
            day_headlines = day_rows["Headline"].tolist()
            model_batch = " .".join(day_headlines) #we do this because the model splits by "." to figure out the seperate headlines
            
            results_df = predict(model_batch, self.model, use_gpu=self.use_gpu)
            day_sentiment_scores = results_df['sentiment_score'].values
            day_avg_ss = sum(day_sentiment_scores ** 2) / len(day_sentiment_scores) #average squared sentiment score for a fixed day
            vol += day_weights[i] * day_avg_ss
            
        return {"news_rv": vol}
    
    def get_headline(self, query_date=None) -> str:
        """
        Returns a headline near the given date for demo display.
        
        Args:
            query_date: Date to find headlines near (defaults to today)
            
        Returns:
            A headline string for display
        """
        if query_date is None:
            query_date = datetime.date.today()
        
        # Look for headlines on the query date or within the past 5 days
        for days_back in range(6):
            check_date = query_date - datetime.timedelta(days=days_back)
            mask = self.df["Timestamp"] == check_date
            day_rows = self.df.loc[mask]
            
            if len(day_rows) > 0:
                # Return a random headline from this day
                return day_rows["Headline"].sample(n=1).iloc[0]
        
        # Fallback: return any random headline if no match found
        if len(self.df) > 0:
            return self.df["Headline"].sample(n=1).iloc[0]
        
        return "No headlines available"

#Example usage:
# news_pipe = NewsPipeline()
# dates = [datetime.datetime.strptime(date_string, "%Y-%m-%d").date() for date_string in ['2021-01-04', '2021-01-06', '2021-02-07']]
# news_pipe.get_latest_data(dates)
