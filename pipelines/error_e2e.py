import os
import sys
import time
import numpy as np
import pandas as pd
import wandb

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from pipelines.volatility import VolatilityPipeline
from pipelines.news import NewsPipeline
from pipelines.weighting import Weighting


def _build_ts_list_from_hist_data(full_hist_data: pd.DataFrame, start, end):
    idx = full_hist_data.index
    hours = pd.Index(idx.floor("h").unique()).sort_values()
    hours = hours[(hours >= pd.Timestamp(start)) & (hours <= pd.Timestamp(end))]
    hours = [pd.Timestamp(h) for h in hours if 9 <= pd.Timestamp(h).hour <= 16]
    return hours


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "n": 0}

    yt = y_true[m]
    yp = y_pred[m]

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))

    ss_res = float(np.sum(err * err))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    return {"mae": mae, "mse": mse, "r2": r2, "n": int(yt.shape[0])}


def _scaled_true_from_hour(prev_hour: pd.Timestamp, har: VolatilityPipeline) -> float:
    h0 = pd.Timestamp(prev_hour).floor("h")
    h1 = h0 + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)

    dfh = har.full_hist_data.loc[h0:h1]
    if dfh.empty:
        return float("nan")

    close = dfh["close"].to_numpy(dtype=float)
    per = int(close.shape[0])
    if per < 2:
        return float("nan")

    diffs = np.diff(close)
    term = diffs * (1.0 / per)
    rv = float(np.sqrt(np.sum(term * term)))

    denom = float(har.train_max["RV_hourly"] - har.train_min["RV_hourly"])
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")

    return float((rv - float(har.train_min["RV_hourly"])) / denom)


def main():
    backtest_start = pd.Timestamp("2021-01-05 10:00:00")
    backtest_end = pd.Timestamp("2021-03-31 16:00:00")

    cfg = {
        "lam": -0.1,
        "delay_hours": 2,
        "k": 10,
        "feature_weights": (0.5, 0.3, 0.15, 0.05),
        "norm_window": 20,
        "warmup_steps": 10,
        "har_short_h": 1,
        "har_med_h": 8,
        "har_long_h": 48,
        "news_short_h": 1,
        "news_med_h": 8,
        "news_long_h": 48,
        "train_end": "2021-01-01 23:59:59",
        "backtest_start": str(backtest_start),
        "backtest_end": str(backtest_end),
    }

    run = wandb.init(
        entity="si2449-columbia-university",
        project="Project-Runs",
        name="e2e_error_metrics",
        config=cfg,
    )

    har = VolatilityPipeline(
        short_window_hours=cfg["har_short_h"],
        medium_window_hours=cfg["har_med_h"],
        long_window_hours=cfg["har_long_h"],
        train_end=cfg["train_end"],
    )
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=cfg["news_short_h"],
        medium_window_hours=cfg["news_med_h"],
        long_window_hours=cfg["news_long_h"],
    )
    w = Weighting(
        lam=cfg["lam"],
        warmup_steps=cfg["warmup_steps"],
        har_pipe=har,
        news_pipe=news,
        feature_weights=cfg["feature_weights"],
        delay_hours=cfg["delay_hours"],
        k=cfg["k"],
        norm_window=cfg["norm_window"],
    )

    ts_list = _build_ts_list_from_hist_data(har.full_hist_data, backtest_start, backtest_end)
    if len(ts_list) == 0:
        raise RuntimeError("Empty ts_list for backtest window.")

    y_true = []
    y_har = []
    y_news = []
    y_weighted = []

    for t in ts_list:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t

        prev_hour = pd.Timestamp(tt).floor("h") - pd.Timedelta(hours=1)
        true_scaled = _safe_float(_scaled_true_from_hour(prev_hour, har))

        try:
            har_pred, _ = har.predict_har_vol(tt)
            har_pred = _safe_float(har_pred)
        except Exception:
            har_pred = float("nan")

        try:
            news_pred, _ = news.predict_news_vol(
                tt,
                k=cfg["k"],
                feature_weights=cfg["feature_weights"],
                delay_hours=cfg["delay_hours"],
            )
            news_pred = _safe_float(news_pred)
        except Exception:
            news_pred = float("nan")

        try:
            weighted_pred = _safe_float(w.predict_weighted_vol(tt))
        except Exception:
            weighted_pred = float("nan")

        y_true.append(true_scaled)
        y_har.append(har_pred)
        y_news.append(news_pred)
        y_weighted.append(weighted_pred)

    m_har = _metrics(y_true, y_har)
    m_news = _metrics(y_true, y_news)
    m_w = _metrics(y_true, y_weighted)

    wandb.log({
        "har_vs_true/mae": m_har["mae"],
        "har_vs_true/mse": m_har["mse"],
        "har_vs_true/r2": m_har["r2"],
        "har_vs_true/n": m_har["n"],

        "news_vs_true/mae": m_news["mae"],
        "news_vs_true/mse": m_news["mse"],
        "news_vs_true/r2": m_news["r2"],
        "news_vs_true/n": m_news["n"],

        "weighted_vs_true/mae": m_w["mae"],
        "weighted_vs_true/mse": m_w["mse"],
        "weighted_vs_true/r2": m_w["r2"],
        "weighted_vs_true/n": m_w["n"],
    })

    def _bar3(key, title, v1, v2, v3, l1="har-rv only", l2="news only", l3="combined"):
        t = wandb.Table(columns=["model", "value"])
        t.add_data(l1, float(v1))
        t.add_data(l2, float(v2))
        t.add_data(l3, float(v3))
        wandb.log({key: wandb.plot.bar(t, "model", "value", title=title)})

    _bar3("plot/mae", "MAE", m_har["mae"], m_news["mae"], m_w["mae"])
    _bar3("plot/mse", "MSE", m_har["mse"], m_news["mse"], m_w["mse"])
    _bar3("plot/r2", "R^2", m_har["r2"], m_news["r2"], m_w["r2"])

    wandb.finish()


if __name__ == "__main__":
    main()