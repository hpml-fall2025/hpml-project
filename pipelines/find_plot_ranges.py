import os, sys
import datetime as dt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from news import NewsPipeline

from volatility import VolatilityPipeline
from weighting import Weighting

CFG = {
    "lam": -0.1,
    "delay_hours": 2,
    "short_h": 1,
    "med_h": 8,
    "long_h": 48,
    "k": 10,
    "norm_window": 20,
    "feature_weights": (0.5, 0.3, 0.15, 0.05),
    "warmup_steps": 10,
}

BACKTEST_START = dt.datetime(2021, 1, 4, 10, 0, 0)
BACKTEST_END   = dt.datetime(2021, 3, 31, 16, 0, 0)

def build_hourly_rv_df(har: VolatilityPipeline) -> pd.DataFrame:
    rv_calc = har._VolatilityPipeline__rv_calculation
    rv_df = rv_calc(har.full_hist_data).copy()
    rv_df = rv_df[~rv_df.index.duplicated(keep="last")].sort_index()
    rv_df = rv_df.loc[(rv_df.index >= pd.Timestamp(BACKTEST_START)) & (rv_df.index <= pd.Timestamp(BACKTEST_END))]
    rv_df = rv_df[(rv_df.index.hour >= 9) & (rv_df.index.hour <= 16)]
    return rv_df

def scaled_true(rv_df: pd.DataFrame, t: pd.Timestamp, har: VolatilityPipeline, col: str) -> float:
    denom = float(har.train_max[col] - har.train_min[col])
    return float((float(rv_df.loc[t, col]) - float(har.train_min[col])) / denom)

def summarize(name: str, x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"{name}: no finite values")
        return
    q01, q99 = np.quantile(x, [0.01, 0.99])
    print(
        f"{name}: n={x.size}  min={x.min():.6f}  max={x.max():.6f}  "
        f"p01={q01:.6f}  p99={q99:.6f} "
        f"mean ={x.mean():.4f}  std={x.std():.4f}"
    )

def main():
    har = VolatilityPipeline(
        short_window_hours=CFG["short_h"],
        medium_window_hours=CFG["med_h"],
        long_window_hours=CFG["long_h"],
    )

    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=CFG["short_h"],
        medium_window_hours=CFG["med_h"],
        long_window_hours=CFG["long_h"],
    )

    w = Weighting(
        lam=CFG["lam"],
        warmup_steps=CFG["warmup_steps"],
        har_pipe=har,
        news_pipe=news,
        feature_weights=CFG["feature_weights"],
        delay_hours=CFG["delay_hours"],
        k=CFG["k"],
        norm_window=CFG["norm_window"],
    )

    rv_df = build_hourly_rv_df(har)
    if rv_df.empty:
        raise RuntimeError("rv_df empty for window after filtering")

    ts_list = list(rv_df.index)
    print(f"hours in window: {len(ts_list)}  ({ts_list[0]} -> {ts_list[-1]})")

    true_arr = []
    har_arr = []
    news_arr = []
    w_arr = []
    cnt_arr = []

    for t in ts_list:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t

        try:
            har_pred, _ = har.predict_har_vol(tt)
        except Exception:
            continue

        try:
            y_true = scaled_true(rv_df, t, har, "RV_hourly")
        except Exception:
            continue

        try:
            n_val, n_cnt = news.predict_news_vol(
                tt,
                k=CFG["k"],
                feature_weights=CFG["feature_weights"],
                delay_hours=CFG["delay_hours"],
            )
        except Exception:
            n_val, n_cnt = 0.0, 0

        try:
            w_pred = float(w.predict_weighted_vol(tt))
        except Exception:
            w_pred = float(har_pred)

        true_arr.append(float(y_true))
        har_arr.append(float(har_pred))
        news_arr.append(float(n_val))
        w_arr.append(float(w_pred))
        cnt_arr.append(int(n_cnt))

    true_arr = np.array(true_arr, dtype=float)
    har_arr  = np.array(har_arr, dtype=float)
    news_arr = np.array(news_arr, dtype=float)
    w_arr    = np.array(w_arr, dtype=float)
    cnt_arr  = np.array(cnt_arr, dtype=int)

    summarize("TRUE (scaled)", true_arr)
    summarize("HAR pred (scaled)", har_arr)
    summarize("WEIGHTED pred", w_arr)
    summarize("NEWS signal", news_arr)

    if cnt_arr.size:
        print(f"NEWS cnt: n={cnt_arr.size}  min={int(cnt_arr.min())}  max={int(cnt_arr.max())}  mean={cnt_arr.mean():.2f}")

if __name__ == "__main__":
    main()