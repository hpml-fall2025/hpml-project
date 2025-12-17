import datetime as dt
import numpy as np
import pandas as pd

from news import NewsPipeline
from volatility import VolatilityPipeline
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

    mpe = float(np.mean(err / (y_true + eps)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)))
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + eps)))

    bias = float(np.mean(err))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else float("nan")

    return {
        "n": int(len(y_true)),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MPE": mpe,
        "MAPE": mape,
        "sMAPE": smape,
        "Bias": bias,
        "Corr": corr,
    }


def true_scaled(rv_df, ts, har, rv_col):
    return float(
        (rv_df.loc[ts, rv_col] - har.train_min[rv_col])
        / (har.train_max[rv_col] - har.train_min[rv_col])
    )


def run_har_baseline(har, rv_df, ts_list, warmup_steps, rv_col):
    y_true, y_pred = [], []

    for t in ts_list[1:]:
        try:
            pred_scaled, _ = har.predict_har_vol(t.to_pydatetime() if hasattr(t, "to_pydatetime") else t)
            pred_scaled = float(pred_scaled)
            y = true_scaled(rv_df, t, har, rv_col)
        except Exception:
            continue
        y_true.append(y)
        y_pred.append(pred_scaled)

    if len(y_true) <= warmup_steps:
        return None

    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_pred = np.array(y_pred[warmup_steps:], dtype=float)
    return compute_metrics(y_true, y_pred)


def run_one(
    har, rv_df, ts_list, warmup_steps, rv_col,
    lam, feature_weights, delay_hours, k, norm_window,
    short_h, med_h, long_h,
):
    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=int(short_h),
        medium_window_hours=int(med_h),
        long_window_hours=int(long_h),
    )

    dw = SimpleWeighting(
        lam=lam,
        warmup_steps=warmup_steps,
        har_pipe=har,
        news_pipe=news,
        feature_weights=feature_weights,
        delay_hours=delay_hours,
        k=k,
        norm_window=norm_window,
    )

    y_true, y_pred = [], []

    for t in ts_list[1:]:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        try:
            pred = float(dw.predict_weighted_vol(tt))
            y = true_scaled(rv_df, t, har, rv_col)
        except Exception:
            continue
        y_true.append(y)
        y_pred.append(pred)

    if len(y_true) <= warmup_steps:
        return None

    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_pred = np.array(y_pred[warmup_steps:], dtype=float)
    return compute_metrics(y_true, y_pred)


def main(
    warmup_steps=10,
    start_ts=dt.datetime(2021, 1, 4, 10, 0, 0),
    end_ts=dt.datetime(2021, 3, 31, 16, 0, 0),
    out_csv="sweep_additive_lam_windows_results.csv",
):
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

    print()
    print("HAR BASELINE (after warmup):")
    base = run_har_baseline(har, rv_df, ts_list, warmup_steps, rv_col)
    if base is None:
        print("  Not enough points.")
    else:
        for k, v in base.items():
            print(f"  {k}: {v}")
    print()

    # ---- LIGHT SWEEP ----
    lam_grid = [-0.15, -0.1]
    delays = [1, 2, 4]
    ks = [10]
    norm_windows = [20]

    feature_weight_grid = [
        #(1.0, 0.0, 0.0, 0.0),
        #(0.7, 0.2, 0.08, 0.02),
        (0.5, 0.3, 0.15, 0.05),
        #(0.4, 0.3, 0.2, 0.1),
    ]

    # short/med/long windows (keep small grid so it runs fast)
    short_grid = [1, 2]
    med_grid = [8, 16]
    long_grid = [48]

    rows = []
    total = 0

    for short_h in short_grid:
        for med_h in med_grid:
            for long_h in long_grid:
                if not (short_h < med_h < long_h):
                    continue
                for delay_hours in delays:
                    for fw in feature_weight_grid:
                        for k_conf in ks:
                            for nw in norm_windows:
                                for lam in lam_grid:
                                    total += 1
                                    m = run_one(
                                        har, rv_df, ts_list, warmup_steps, rv_col,
                                        lam, fw, delay_hours, k_conf, nw,
                                        short_h, med_h, long_h,
                                    )
                                    if m is None:
                                        continue
                                    m["lam"] = float(lam)
                                    m["delay_hours"] = int(delay_hours)
                                    m["k"] = int(k_conf)
                                    m["norm_window"] = int(nw)
                                    m["feature_weights"] = str(tuple(float(x) for x in fw))
                                    m["short_h"] = int(short_h)
                                    m["med_h"] = int(med_h)
                                    m["long_h"] = int(long_h)
                                    rows.append(m)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No configs produced results (everything failed / skipped).")
        return

    df = df.sort_values(["MAE", "MSE"], ascending=[True, True])

    print(f"Ran {total} configs, got {len(df)} valid results.")
    print()
    print("TOP 20 ADDITIVE (HAR + lam*NEWS) + WINDOW SWEEP CONFIGS:")
    print(df.head(5).to_string(index=False))

    # ---- NEW: print best config for each metric ----
    better_low = ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]
    better_high = ["R2", "Corr"]
    also_report = ["Bias", "MPE"]  # not strictly "better", but still useful

    # ensure these exist in df
    better_low = [c for c in better_low if c in df.columns]
    better_high = [c for c in better_high if c in df.columns]
    also_report = [c for c in also_report if c in df.columns]

    config_cols = ["lam", "delay_hours", "short_h", "med_h", "long_h", "k", "norm_window", "feature_weights"]
    config_cols = [c for c in config_cols if c in df.columns]

    def _print_row(title, row):
        cols = ["n"] + better_low + better_high + also_report + config_cols
        cols = [c for c in cols if c in row.index]
        print(title)
        print(row[cols].to_string())
        print()

    print()
    print("BEST CONFIG PER METRIC:")

    for m in better_low:
        row = df.loc[df[m].idxmin()]
        _print_row(f"Best (min) {m}:", row)

    for m in better_high:
        row = df.loc[df[m].idxmax()]
        _print_row(f"Best (max) {m}:", row)

    # optional: also show smallest |Bias| and |MPE|
    if "Bias" in df.columns:
        row = df.loc[(df["Bias"].abs()).idxmin()]
        _print_row("Best (min abs) Bias:", row)

    if "MPE" in df.columns:
        row = df.loc[(df["MPE"].abs()).idxmin()]
        _print_row("Best (min abs) MPE:", row)

if __name__ == "__main__":
    main()