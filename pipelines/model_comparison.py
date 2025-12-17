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


def main(
    warmup_steps=10,
    start_date=dt.date(2021, 1, 4),
    end_date=dt.date(2021, 3, 31),
):
    har = VolatilityPipeline()
    dw = DynamicWeighting()

    rv_calc = har._VolatilityPipeline__rv_calculation
    full_rv = rv_calc(har.full_hist_data)

    full_rv = full_rv.loc[(full_rv.index >= start_date) & (full_rv.index <= end_date)]
    dates = list(full_rv.index)

    if len(dates) < warmup_steps + 2:
        raise RuntimeError("Not enough trading dates in range to evaluate after warmup.")

    eval_dates = []
    y_true = []
    y_har = []
    y_w = []

    for i in range(1, len(dates)):
        d = dates[i]

        try:
            har_pred, _ = har.predict_har_vol(d)
            har_pred = float(har_pred)
        except Exception:
            continue

        try:
            w_pred = float(dw.predict_weighted_vol(d))
        except Exception:
            continue

        true_d = float(
            (full_rv.loc[d, "RV_daily"] - har.train_min["RV_daily"])
            / (har.train_max["RV_daily"] - har.train_min["RV_daily"])
        )

        eval_dates.append(d)
        y_true.append(true_d)
        y_har.append(har_pred)
        y_w.append(w_pred)

    if len(eval_dates) <= warmup_steps:
        raise RuntimeError(f"Not enough evaluation points after warmup={warmup_steps}. Got {len(eval_dates)}.")

    eval_dates = eval_dates[warmup_steps:]
    y_true = np.array(y_true[warmup_steps:], dtype=float)
    y_har = np.array(y_har[warmup_steps:], dtype=float)
    y_w = np.array(y_w[warmup_steps:], dtype=float)

    har_metrics = compute_metrics(y_true, y_har)
    w_metrics = compute_metrics(y_true, y_w)

    print("y_true range:", float(np.min(y_true)), float(np.max(y_true)))
    print("y_har range :", float(np.min(y_har)), float(np.max(y_har)))
    print("y_w range   :", float(np.min(y_w)), float(np.max(y_w)))


    print()
    print(f"Evaluation window: {eval_dates[0]} to {eval_dates[-1]}")
    print()

    print("HAR-only metrics:")
    for k, v in har_metrics.items():
        print(f"  {k}: {v}")

    print()
    print("Weighted metrics:")
    for k, v in w_metrics.items():
        print(f"  {k}: {v}")

    df_out = pd.DataFrame(
        {
            "date": eval_dates,
            "true_RV_daily": y_true,
            "har_pred": y_har,
            "weighted_pred": y_w,
            "har_err": y_har - y_true,
            "weighted_err": y_w - y_true,
        }
    )



if __name__ == "__main__":
    main()