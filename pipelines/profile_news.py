import time
import os, sys
import numpy as np
import pandas as pd
import wandb

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from pipelines.news_slow import NewsPipelineSlow
from pipelines.news import NewsPipeline


def _build_ts_list(start, end):
    start = pd.Timestamp(start).floor("h")
    end = pd.Timestamp(end).floor("h")
    hours = pd.date_range(start, end, freq="h")
    hours = [pd.Timestamp(h) for h in hours if 9 <= pd.Timestamp(h).hour <= 16]
    return hours


def _run_one(pipe, ts_list, after_ts, k, feature_weights, delay_hours):
    after_ts = pd.Timestamp(after_ts)

    lat_all = []
    lat_after = []

    t0 = time.perf_counter()
    for t in ts_list:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t

        s = time.perf_counter()
        try:
            pipe.predict_news_vol(
                when=tt,
                k=int(k),
                feature_weights=tuple(feature_weights),
                delay_hours=int(delay_hours),
            )
        except Exception:
            pass
        e = time.perf_counter()

        dt_call = e - s
        lat_all.append(dt_call)
        if t >= after_ts:
            lat_after.append(dt_call)

    t1 = time.perf_counter()
    total_seconds = float(t1 - t0)

    throughput_per_s = float(len(ts_list) / total_seconds) if total_seconds > 0 else float("nan")
    after_mean_ms = float(1000.0 * np.mean(lat_after)) if len(lat_after) else float("nan")

    return {
        "total_seconds": total_seconds,
        "lat_after_mean_ms": after_mean_ms,
        "throughput_per_s": throughput_per_s,
    }


def main():
    backtest_start = pd.Timestamp("2021-01-05 10:00:00")
    backtest_end = pd.Timestamp("2021-03-31 16:00:00")
    after_ts = backtest_start + pd.Timedelta(days=30)

    cfg = {
        "use_gpu": True,
        "short_h": 1,
        "med_h": 8,
        "long_h": 48,
        "delay_hours": 2,
        "k": 10,
        "feature_weights": (0.5, 0.3, 0.15, 0.05),
        "backtest_start": str(backtest_start),
        "backtest_end": str(backtest_end),
        "after_ts": str(after_ts),
    }

    wandb.init(
        entity="si2449-columbia-university",
        project="streaming-finbert",
        name="news_slow_vs_opt_metrics",
        config=cfg,
    )

    ts_list = _build_ts_list(backtest_start, backtest_end)
    if len(ts_list) == 0:
        raise RuntimeError("Empty ts_list for backtest window.")

    t0 = time.perf_counter()
    slow = NewsPipelineSlow(
        use_gpu=bool(cfg["use_gpu"]),
        short_window_hours=cfg["short_h"],
        medium_window_hours=cfg["med_h"],
        long_window_hours=cfg["long_h"],
    )
    slow_init_seconds = float(time.perf_counter() - t0)
    slow_out = _run_one(
        slow, ts_list, after_ts,
        k=cfg["k"], feature_weights=cfg["feature_weights"], delay_hours=cfg["delay_hours"]
    )

    t0 = time.perf_counter()
    opt = NewsPipeline(
        use_gpu=bool(cfg["use_gpu"]),
        short_window_hours=cfg["short_h"],
        medium_window_hours=cfg["med_h"],
        long_window_hours=cfg["long_h"],
    )
    opt_init_seconds = float(time.perf_counter() - t0)

    if hasattr(opt, "reset_stream"):
        opt.reset_stream()

    opt_out = _run_one(
        opt, ts_list, after_ts,
        k=cfg["k"], feature_weights=cfg["feature_weights"], delay_hours=cfg["delay_hours"]
    )

    total_speedup = float(slow_out["total_seconds"] / opt_out["total_seconds"]) if opt_out["total_seconds"] > 0 else float("inf")
    after_speedup = float(slow_out["lat_after_mean_ms"] / opt_out["lat_after_mean_ms"]) if (opt_out["lat_after_mean_ms"] and opt_out["lat_after_mean_ms"] > 0) else float("inf")

    wandb.log({
        "slow/init_seconds": slow_init_seconds,
        "slow/total_seconds": slow_out["total_seconds"],
        "slow/latency_after_warmup_mean_ms": slow_out["lat_after_mean_ms"],
        "slow/throughput_per_s": slow_out["throughput_per_s"],

        "opt/init_seconds": opt_init_seconds,
        "opt/total_seconds": opt_out["total_seconds"],
        "opt/latency_after_warmup_mean_ms": opt_out["lat_after_mean_ms"],
        "opt/throughput_per_s": opt_out["throughput_per_s"],

        "speedup/total_x": total_speedup,
        "speedup/after_warmup_latency_x": after_speedup,
    })

    def _log_bar(key, title, slow_val, opt_val):
        t = wandb.Table(columns=["variant", "value"])
        t.add_data("slow", float(slow_val))
        t.add_data("opt", float(opt_val))
        wandb.log({key: wandb.plot.bar(t, "variant", "value", title=title)})

    _log_bar("plot/init_seconds", "Init seconds", slow_init_seconds, opt_init_seconds)
    _log_bar("plot/total_seconds", "Total runtime seconds", slow_out["total_seconds"], opt_out["total_seconds"])
    _log_bar("plot/lat_after_ms", "Mean latency after warmup (ms)", slow_out["lat_after_mean_ms"], opt_out["lat_after_mean_ms"])
    _log_bar("plot/throughput", "Throughput (pred/sec)", slow_out["throughput_per_s"], opt_out["throughput_per_s"])

    t_speed = wandb.Table(columns=["metric", "value"])
    t_speed.add_data("total_speedup_x", float(total_speedup))
    t_speed.add_data("after_warmup_speedup_x", float(after_speedup))
    wandb.log({"plot/speedup": wandb.plot.bar(t_speed, "metric", "value", title="Speedup (x)")})

    wandb.finish()


if __name__ == "__main__":
    main()