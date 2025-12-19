import time
import os
import sys
import numpy as np
import pandas as pd
import wandb

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from pipelines.volatility_slow import VolatilityPipelineSlow
from pipelines.volatility import VolatilityPipeline

from pipelines.news_slow import NewsPipelineSlow 
from pipelines.news import NewsPipeline
from pipelines.weighting import Weighting


def _build_ts_list_from_data(pipe, start, end):
    idx = pipe.full_hist_data.index
    hours = pd.Index(idx.floor("h").unique()).sort_values()
    hours = hours[(hours >= pd.Timestamp(start)) & (hours <= pd.Timestamp(end))]
    hours = [pd.Timestamp(h) for h in hours if 9 <= pd.Timestamp(h).hour <= 16]
    return hours


def _run_one(w, ts_list, after_ts):
    after_ts = pd.Timestamp(after_ts)

    lat_all = []
    lat_after = []

    t0 = time.perf_counter()
    for t in ts_list:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t

        s = time.perf_counter()
        try:
            _ = w.predict_weighted_vol(tt)
        except Exception:
            pass
        e = time.perf_counter()

        dt_call = e - s
        lat_all.append(dt_call)
        if t >= after_ts:
            lat_after.append(dt_call)

    t1 = time.perf_counter()
    total_seconds = float(t1 - t0)

    lat_mean_ms = float(1000.0 * np.mean(lat_all)) if len(lat_all) else float("nan")
    lat_after_mean_ms = float(1000.0 * np.mean(lat_after)) if len(lat_after) else float("nan")
    throughput_per_s = float(len(ts_list) / total_seconds) if total_seconds > 0 else float("nan")

    return {
        "total_seconds": total_seconds,
        "lat_mean_ms": lat_mean_ms,
        "lat_after_mean_ms": lat_after_mean_ms,
        "throughput_per_s": throughput_per_s,
    }


def _log_bar(key, title, slow_val, opt_val):
    t = wandb.Table(columns=["variant", "value"])
    t.add_data("slow", float(slow_val))
    t.add_data("opt", float(opt_val))
    wandb.log({key: wandb.plot.bar(t, "variant", "value", title=title)})


def main():
    backtest_start = pd.Timestamp("2021-01-05 10:00:00")
    backtest_end = pd.Timestamp("2021-03-31 16:00:00")
    after_ts = backtest_start + pd.Timedelta(days=30)

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
        "after_ts": str(after_ts),
    }

    wandb.init(
        entity="si2449-columbia-university",
        project="Project-Runs",
        name="e2e_slow_vs_opt_metrics",
        config=cfg,
    )

    t0 = time.perf_counter()
    har_slow = VolatilityPipelineSlow(
        short_window_hours=cfg["har_short_h"],
        medium_window_hours=cfg["har_med_h"],
        long_window_hours=cfg["har_long_h"],
        train_end=cfg["train_end"],
    )
    news_slow = NewsPipelineSlow(
        use_gpu=True,
        short_window_hours=cfg["news_short_h"],
        medium_window_hours=cfg["news_med_h"],
        long_window_hours=cfg["news_long_h"],
    )
    w_slow = Weighting(
        lam=cfg["lam"],
        warmup_steps=cfg["warmup_steps"],
        har_pipe=har_slow,
        news_pipe=news_slow,
        feature_weights=cfg["feature_weights"],
        delay_hours=cfg["delay_hours"],
        k=cfg["k"],
        norm_window=cfg["norm_window"],
    )
    slow_init_seconds = float(time.perf_counter() - t0)

    ts_list = _build_ts_list_from_data(har_slow, backtest_start, backtest_end)
    if len(ts_list) == 0:
        raise RuntimeError("Empty ts_list for backtest window.")

    slow_out = _run_one(w_slow, ts_list, after_ts)

    t0 = time.perf_counter()
    har_opt = VolatilityPipeline(
        short_window_hours=cfg["har_short_h"],
        medium_window_hours=cfg["har_med_h"],
        long_window_hours=cfg["har_long_h"],
        train_end=cfg["train_end"],
    )
    news_opt = NewsPipeline(
        use_gpu=True,
        short_window_hours=cfg["news_short_h"],
        medium_window_hours=cfg["news_med_h"],
        long_window_hours=cfg["news_long_h"],
    )
    w_opt = Weighting(
        lam=cfg["lam"],
        warmup_steps=cfg["warmup_steps"],
        har_pipe=har_opt,
        news_pipe=news_opt,
        feature_weights=cfg["feature_weights"],
        delay_hours=cfg["delay_hours"],
        k=cfg["k"],
        norm_window=cfg["norm_window"],
    )
    opt_init_seconds = float(time.perf_counter() - t0)

    if hasattr(har_opt, "reset_stream"):
        har_opt.reset_stream()
    if hasattr(news_opt, "reset_stream"):
        news_opt.reset_stream()

    opt_out = _run_one(w_opt, ts_list, after_ts)

    total_speedup_x = float(slow_out["total_seconds"] / opt_out["total_seconds"]) if opt_out["total_seconds"] > 0 else float("inf")
    after_speedup_x = float(slow_out["lat_after_mean_ms"] / opt_out["lat_after_mean_ms"]) if (np.isfinite(opt_out["lat_after_mean_ms"]) and opt_out["lat_after_mean_ms"] > 0) else float("inf")

    wandb.log({
        "slow/init_seconds": slow_init_seconds,
        "slow/total_seconds": slow_out["total_seconds"],
        "slow/lat_mean_ms": slow_out["lat_mean_ms"],
        "slow/latency_after_warmup_mean_ms": slow_out["lat_after_mean_ms"],
        "slow/throughput_per_s": slow_out["throughput_per_s"],

        "opt/init_seconds": opt_init_seconds,
        "opt/total_seconds": opt_out["total_seconds"],
        "opt/lat_mean_ms": opt_out["lat_mean_ms"],
        "opt/latency_after_warmup_mean_ms": opt_out["lat_after_mean_ms"],
        "opt/throughput_per_s": opt_out["throughput_per_s"],

        "speedup/total_x": total_speedup_x,
        "speedup/after_warmup_latency_x": after_speedup_x,
    })

    _log_bar("plot/init_seconds", "Init seconds", slow_init_seconds, opt_init_seconds)
    _log_bar("plot/total_seconds", "Total runtime seconds", slow_out["total_seconds"], opt_out["total_seconds"])
    _log_bar("plot/lat_mean_ms", "Mean latency per call (ms)", slow_out["lat_mean_ms"], opt_out["lat_mean_ms"])
    _log_bar("plot/lat_after_ms", "Mean latency after warmup (ms)", slow_out["lat_after_mean_ms"], opt_out["lat_after_mean_ms"])
    _log_bar("plot/throughput", "Throughput (pred/sec)", slow_out["throughput_per_s"], opt_out["throughput_per_s"])

    t_speed = wandb.Table(columns=["metric", "value"])
    t_speed.add_data("total_speedup_x", float(total_speedup_x))
    t_speed.add_data("after_warmup_speedup_x", float(after_speedup_x))
    wandb.log({"plot/speedup": wandb.plot.bar(t_speed, "metric", "value", title="Speedup (x)")})

    wandb.finish()


if __name__ == "__main__":
    main()