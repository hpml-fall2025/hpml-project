import datetime as dt
import numpy as np
import pandas as pd

from news import NewsPipeline
from volatility import VolatilityPipeline
from weighting import DynamicWeighting

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


def _true_scaled_for_ts(rv_df, ts, har, rv_col):
    return float(
        (rv_df.loc[ts, rv_col] - har.train_min[rv_col])
        / (har.train_max[rv_col] - har.train_min[rv_col])
    )


def run_har_baseline(har, rv_df, ts_list, warmup_steps, rv_col):
    y_true = []
    y_pred = []

    for i in range(1, len(ts_list)):
        t = ts_list[i]
        try:
            pred_scaled, _ = har.predict_har_vol(t)
            pred_scaled = float(pred_scaled)
        except Exception:
            continue

        try:
            true_scaled = _true_scaled_for_ts(rv_df, t, har, rv_col)
        except Exception:
            continue

        y_true.append(true_scaled)
        y_pred.append(pred_scaled)

    if len(y_true) <= warmup_steps:
        return None

    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_pred = np.array(y_pred[warmup_steps:], dtype=float)
    return compute_metrics(y_true, y_pred)


def run_one(har, news, rv_df, ts_list, warmup_steps, rv_col,
            lambda_har, lambda_news, feature_weights, delay_hours, k, norm_window):
    dw = DynamicWeighting(
        lambda_har=lambda_har,
        lambda_news=lambda_news,
        warmup_steps=warmup_steps,
        har_pipe=har,
        news_pipe=news,
        news_feature_weights=feature_weights,
        news_delay_hours=delay_hours,
        news_k=k,
        norm_window=norm_window,
    )

    y_true = []
    y_pred = []

    for i in range(1, len(ts_list)):
        t = ts_list[i]

        try:
            pred = float(dw.predict_weighted_vol(t))
        except Exception:
            continue

        try:
            true_scaled = _true_scaled_for_ts(rv_df, t, har, rv_col)
        except Exception:
            continue

        y_true.append(true_scaled)
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
    out_csv="sweep_hourly_light_results.csv",
):
    har = VolatilityPipeline()

    rv_calc = har._VolatilityPipeline__rv_calculation
    rv_df = rv_calc(har.full_hist_data).copy()
    rv_df = rv_df[~rv_df.index.duplicated(keep="last")]

    rv_col = "RV_hourly"
    if rv_col not in rv_df.columns:
        raise RuntimeError(f"Expected rv_df to have column '{rv_col}'. Found: {list(rv_df.columns)}")

    rv_df = rv_df.loc[(rv_df.index >= pd.Timestamp(start_ts)) & (rv_df.index <= pd.Timestamp(end_ts))]
    ts_list = list(rv_df.index)

    print()
    print("HAR BASELINE (after warmup):")
    har_m = run_har_baseline(har, rv_df, ts_list, warmup_steps, rv_col)
    if har_m is None:
        print("  Not enough points.")
    else:
        for kk, vv in har_m.items():
            print(f"  {kk}: {vv}")
    print()

    delays = [0, 1]
    short_hours_grid = [1]
    med_hours_grid = [4, 8]
    long_hours_grid = [24, 48, 96]

    ks = [10]
    norm_windows = [20]

    lambda_har_grid = [0.3]
    lambda_news_grid = [0.1]

    feature_weights_grid = [
        #(1.0, 0.0, 0.0, 0.0),
        #(0.6, 0.3, 0.1, 0.0),
        (0.5, 0.3, 0.15, 0.05),
        (0.4, 0.3, 0.2, 0.1),
    ]

    rows = []
    total = 0

    for delay in delays:
        for sh in short_hours_grid:
            for mh in med_hours_grid:
                for lh in long_hours_grid:
                    if not (sh < mh < lh):
                        continue

                    # instantiate a NewsPipeline for THIS (short,med,long)
                    news = NewsPipeline(
                        use_gpu=True,
                        short_window_hours=sh,
                        medium_window_hours=mh,
                        long_window_hours=lh,
                    )

                    for k_conf in ks:
                        for nw in norm_windows:
                            for lhpar in lambda_har_grid:
                                for lnpar in lambda_news_grid:
                                    for fw in feature_weights_grid:
                                        total += 1
                                        m = run_one(
                                            har, news, rv_df, ts_list, warmup_steps, rv_col,
                                            lhpar, lnpar, fw, delay, k_conf, nw
                                        )
                                        if m is None:
                                            continue
                                        m["delay_hours"] = int(delay)
                                        m["short_h"] = int(sh)
                                        m["med_h"] = int(mh)
                                        m["long_h"] = int(lh)
                                        m["k"] = int(k_conf)
                                        m["norm_window"] = int(nw)
                                        m["lambda_har"] = float(lhpar)
                                        m["lambda_news"] = float(lnpar)
                                        m["feature_weights"] = str(tuple(float(x) for x in fw))
                                        rows.append(m)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No configs produced results (everything failed / skipped).")
        return

    df = df.sort_values(["MAE", "MSE"], ascending=[True, True])

    print(f"Ran {total} configs, got {len(df)} valid results.")
    print()
    print("TOP 20 HOURLY LIGHT-SWEEP CONFIGS:")
    print(df.head(20).to_string(index=False))

    df.to_csv(out_csv, index=False)
    print()
    print(f"Saved sweep results to: {out_csv}")
    print()


if __name__ == "__main__":
    main()