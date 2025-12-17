import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from volatility import VolatilityPipeline
from news import NewsPipeline
from simple_weighting import SimpleWeighting

import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_metrics(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > eps else float("nan")

    bias = float(np.mean(err))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else float("nan")

    return {"n": int(len(y_true)), "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Bias": bias, "Corr": corr}


def normalize_like_code(har_series, news_series, window=20, eps=1e-8):
    h_hist = deque(maxlen=window)
    n_hist = deque(maxlen=window)
    out = []

    for h, n in zip(har_series, news_series):
        if not np.isfinite(n):
            out.append(np.nan)
            continue

        h_hist.append(float(h))
        n_hist.append(float(n))

        if len(h_hist) < 2 or len(n_hist) < 2:
            out.append(float(n))
            continue

        mu_h = float(np.mean(h_hist))
        sd_h = float(np.std(h_hist, ddof=0))
        mu_n = float(np.mean(n_hist))
        sd_n = float(np.std(n_hist, ddof=0))

        n_norm = mu_h + (sd_h + eps) * (float(n) - mu_n) / (sd_n + eps)
        out.append(float(n_norm))

    return np.array(out, dtype=float)


def plot_set(title_prefix, t, true_scaled, har_scaled, third_series=None, third_label=None):
    resid_scaled = true_scaled - har_scaled
    abs_resid_scaled = np.abs(resid_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(t, true_scaled, label="True RV_hourly (scaled)", lw = 0.5)
    plt.plot(t, har_scaled, label="HAR (scaled)", lw = 0.5)
    if third_series is not None:
        plt.plot(t, third_series, label=third_label, lw = 0.5)
    plt.title(title_prefix)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, resid_scaled, label="True - HAR (scaled residual)")
    plt.plot(t, abs_resid_scaled, label="|True - HAR| (scaled)")
    if third_series is not None:
        plt.plot(t, third_series, label=third_label)
    plt.title(title_prefix + " — residuals (scaled) + overlay")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
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

    warmup_steps = int(CFG["warmup_steps"])
    start_ts = dt.datetime(2021, 1, 4, 10, 0, 0)
    end_ts = dt.datetime(2021, 3, 31, 16, 0, 0)

    har = VolatilityPipeline()
    if getattr(har, "model", None) is None:
        raise RuntimeError("HAR model did not train (har.model is None). Fix training first.")

    rv_calc = har._VolatilityPipeline__rv_calculation
    rv_df = rv_calc(har.full_hist_data).copy()
    rv_df = rv_df[~rv_df.index.duplicated(keep="last")]

    rv_col = "RV_hourly"
    if rv_col not in rv_df.columns:
        raise RuntimeError(f"Expected '{rv_col}' in RV df. Found: {list(rv_df.columns)}")

    rv_df = rv_df.loc[(rv_df.index >= pd.Timestamp(start_ts)) & (rv_df.index <= pd.Timestamp(end_ts))]
    ts_list = list(rv_df.index)
    if len(ts_list) < 100:
        raise RuntimeError(f"Too few hourly points in window (n={len(ts_list)}).")

    rv_min = float(har.train_min[rv_col])
    rv_max = float(har.train_max[rv_col])
    denom = rv_max - rv_min
    if denom == 0:
        raise RuntimeError("Degenerate scaling for RV_hourly (train_max == train_min).")

    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=int(CFG["short_h"]),
        medium_window_hours=int(CFG["med_h"]),
        long_window_hours=int(CFG["long_h"]),
    )

    sw = SimpleWeighting(
        lam=float(CFG["lam"]),
        warmup_steps=int(CFG["warmup_steps"]),
        har_pipe=har,
        news_pipe=news,
        feature_weights=tuple(CFG["feature_weights"]),
        delay_hours=int(CFG["delay_hours"]),
        k=int(CFG["k"]),
        norm_window=int(CFG["norm_window"]),
    )

    # --- Build HAR vs TRUE using the SAME iteration pattern as sweep ---
    rows = []
    debug_fail = 0

    for t in ts_list[1:]:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        try:
            pred_scaled, _ = har.predict_har_vol(tt)
            pred_scaled = float(pred_scaled)
            true_target_scaled = float((float(rv_df.loc[t, rv_col]) - rv_min) / denom)
            rows.append({"t": t, "true_scaled": true_target_scaled, "har_scaled": pred_scaled, "tt": tt})
        except Exception as e:
            debug_fail += 1
            if debug_fail <= 5:
                print("HAR FAIL @", t, "->", repr(e))
            continue

    df = pd.DataFrame(rows).sort_values("t")
    if len(df) <= warmup_steps + 10:
        raise RuntimeError(f"Too few HAR points after filtering (n={len(df)}).")

    tplot = df["t"].tolist()
    true_plot = df["true_scaled"].values.astype(float)
    har_plot = df["har_scaled"].values.astype(float)

    print("\nHAR metrics (scaled):")
    for k, v in compute_metrics(true_plot[warmup_steps:], har_plot[warmup_steps:]).items():
        print(f"  {k}: {v}")
    print()

    # ---- SET A: HAR only ----
    plot_set("SET A: True vs HAR (scaled)", tplot, true_plot, har_plot)

    # ---- SET B: NEWS series (called EXACTLY like sweep: predict_news_vol) ----
    news_raw = np.full(len(df), np.nan, dtype=float)
    debug_fail = 0

    for i, tt in enumerate(df["tt"].tolist()):
        try:
            n_raw, _ = news.predict_news_vol(
                tt,
                k=int(CFG["k"]),
                feature_weights=tuple(CFG["feature_weights"]),
                delay_hours=int(CFG["delay_hours"]),
            )
            news_raw[i] = float(n_raw)
        except Exception as e:
            debug_fail += 1
            if debug_fail <= 5:
                print("NEWS FAIL @", df["t"].iloc[i], "tt=", tt, "->", repr(e))
            continue

    news_norm = normalize_like_code(har_plot, news_raw, window=int(CFG["norm_window"]), eps=1e-8)

    print(f"NEWS finite points: {int(np.isfinite(news_norm).sum())} / {len(news_norm)}")
    plot_set(
        title_prefix=(
            "SET B: True vs HAR vs NEWS (normalized)\n"
            f"cfg: delay={CFG['delay_hours']} short/med/long={CFG['short_h']}/{CFG['med_h']}/{CFG['long_h']} "
            f"k={CFG['k']} norm_window={CFG['norm_window']} w={CFG['feature_weights']}"
        ),
        t=tplot,
        true_scaled=true_plot,
        har_scaled=har_plot,
        third_series=news_norm,
        third_label="NEWS (normalized)",
    )

    # ---- SET C: WEIGHTED series (called EXACTLY like sweep: sw.predict_weighted_vol) ----
    weighted = np.full(len(df), np.nan, dtype=float)
    debug_fail = 0

    for i, tt in enumerate(df["tt"].tolist()):
        try:
            weighted[i] = float(sw.predict_weighted_vol(tt))
        except Exception as e:
            debug_fail += 1
            if debug_fail <= 5:
                print("WEIGHTED FAIL @", df["t"].iloc[i], "tt=", tt, "->", repr(e))
            continue

    print(f"WEIGHTED finite points: {int(np.isfinite(weighted).sum())} / {len(weighted)}")
    plot_set(
        title_prefix=(
            "SET C: True vs HAR vs WEIGHTED (SimpleWeighting)\n"
            f"cfg: lam={CFG['lam']} delay={CFG['delay_hours']} short/med/long={CFG['short_h']}/{CFG['med_h']}/{CFG['long_h']} "
            f"k={CFG['k']} norm_window={CFG['norm_window']} w={CFG['feature_weights']} warmup={CFG['warmup_steps']}"
        ),
        t=tplot,
        true_scaled=true_plot,
        har_scaled=har_plot,
        third_series=weighted,
        third_label="WEIGHTED (SimpleWeighting)",
    )


if __name__ == "__main__":
    main()