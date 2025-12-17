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


def run_har_baseline(har, full_rv, dates, warmup_steps):
    y_true = []
    y_pred = []

    for i in range(1, len(dates)):
        d = dates[i]
        try:
            har_pred_scaled, _ = har.predict_har_vol(d)
            har_pred_scaled = float(har_pred_scaled)
        except Exception:
            continue

        true_scaled = float(
            (full_rv.loc[d, "RV_daily"] - har.train_min["RV_daily"])
            / (har.train_max["RV_daily"] - har.train_min["RV_daily"])
        )

        y_true.append(true_scaled)
        y_pred.append(har_pred_scaled)

    if len(y_true) <= warmup_steps:
        return None

    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_pred = np.array(y_pred[warmup_steps:], dtype=float)
    return compute_metrics(y_true, y_pred)


def run_one(har, news, full_rv, dates, warmup_steps, lambda_har, lambda_news, day_weights, delay, k, norm_window):
    dw = DynamicWeighting(
        lambda_har=lambda_har,
        lambda_news=lambda_news,
        warmup_steps=warmup_steps,
        har_pipe=har,
        news_pipe=news,
        news_day_weights=day_weights,
        news_delay=delay,
        news_k=k,
        norm_window=norm_window,
    )

    y_true = []
    y_pred = []

    for i in range(1, len(dates)):
        d = dates[i]
        try:
            pred = float(dw.predict_weighted_vol(d))
        except Exception:
            continue

        true_scaled = float(
            (full_rv.loc[d, "RV_daily"] - har.train_min["RV_daily"])
            / (har.train_max["RV_daily"] - har.train_min["RV_daily"])
        )

        y_true.append(true_scaled)
        y_pred.append(pred)

    if len(y_true) <= warmup_steps:
        return None

    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_pred = np.array(y_pred[warmup_steps:], dtype=float)
    return compute_metrics(y_true, y_pred)


def main(
    warmup_steps=10,
    start_date=dt.date(2021, 1, 4),
    end_date=dt.date(2021, 3, 31),
    out_csv="sweep_specialized_results.csv",
):
    har = VolatilityPipeline()
    news = NewsPipeline(use_gpu=True)

    rv_calc = har._VolatilityPipeline__rv_calculation
    full_rv = rv_calc(har.full_hist_data)
    full_rv = full_rv.loc[(full_rv.index >= start_date) & (full_rv.index <= end_date)]
    dates = list(full_rv.index)

    print()
    print("HAR BASELINE (after warmup):")
    har_m = run_har_baseline(har, full_rv, dates, warmup_steps)
    if har_m is None:
        print("  Not enough points.")
    else:
        for k, v in har_m.items():
            print(f"  {k}: {v}")
    print()


    shapes = {
        # 12-day slow family (like slow12_exp093)
        #"slow12_exp092": [0.126515, 0.116394, 0.107083, 0.098516, 0.090635, 0.083384, 0.076713, 0.070576, 0.064930, 0.059736, 0.054957, 0.050561],
        #"slow12_exp093": [0.120398, 0.111970, 0.104132, 0.096843, 0.090064, 0.083760, 0.077896, 0.072444, 0.067373, 0.062657, 0.058271, 0.054192],
        #"slow12_exp094": [0.114486, 0.107617, 0.101160, 0.095091, 0.089385, 0.084022, 0.078981, 0.074242, 0.069787, 0.065600, 0.061664, 0.057965],
        #"slow12_exp095": [0.108781, 0.103342, 0.098175, 0.093266, 0.088603, 0.084173, 0.079964, 0.075966, 0.072167, 0.068559, 0.065131, 0.061873],

        # 15-day slow family
        #"slow15_exp092": [0.112092, 0.103124, 0.094874, 0.087284, 0.080302, 0.073877, 0.067967, 0.062530, 0.057527, 0.052925, 0.048691, 0.044796, 0.041212, 0.037915, 0.034884],
        #"slow15_exp093": [0.105533, 0.098146, 0.091276, 0.084886, 0.078944, 0.073418, 0.068279, 0.063499, 0.059054, 0.054921, 0.051076, 0.047501, 0.044176, 0.041083, 0.038208],
        "slow15_exp094": [0.099221, 0.093268, 0.087672, 0.082412, 0.077467, 0.072819, 0.068450, 0.064343, 0.060482, 0.056853, 0.053442, 0.050236, 0.047221, 0.044388, 0.041726],
        "slow15_exp095": [0.093160, 0.088502, 0.084077, 0.079873, 0.075880, 0.072086, 0.068481, 0.065057, 0.061805, 0.058714, 0.055779, 0.052990, 0.050340, 0.047823, 0.045433],
    }



    # Zoomed lambda grids around best (0.25-0.3, 0.01)
    lambda_har_grid = [0.32]
    lambda_news_grid = [0.15]

    # Keep delays small; extend k slightly since it didn’t matter much
    delays = [8]
    ks = [10]

    # News normalization window matters; try a few
    norm_windows = [20]

    rows = []
    for shape_name, w in shapes.items():
        for delay in delays:
            for k_conf in ks:
                for nw in norm_windows:
                    for lh in lambda_har_grid:
                        for ln in lambda_news_grid:
                            m = run_one(
                                har, news, full_rv, dates, warmup_steps,
                                lh, ln, w, delay, k_conf, nw
                            )
                            if m is None:
                                continue
                            m["shape"] = shape_name
                            m["delay"] = int(delay)
                            m["k"] = int(k_conf)
                            m["norm_window"] = int(nw)
                            m["lambda_har"] = float(lh)
                            m["lambda_news"] = float(ln)
                            rows.append(m)

    df = pd.DataFrame(rows).sort_values(["MAE", "MSE"], ascending=[True, True])

    print("TOP 20 SPECIALIZED CONFIGS:")
    print(df.head(20).to_string(index=False))

    df.to_csv(out_csv, index=False)
    print()
    print(f"Saved specialized sweep results to: {out_csv}")
    print()


if __name__ == "__main__":
    main()