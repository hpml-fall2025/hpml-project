import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volatility import VolatilityPipeline
from news import NewsPipeline
from weighting import Weighting

import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
    if len(ts_list) < 50:
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

    sw = Weighting(
        lam=float(CFG["lam"]),
        warmup_steps=int(CFG["warmup_steps"]),
        har_pipe=har,
        news_pipe=news,
        feature_weights=tuple(CFG["feature_weights"]),
        delay_hours=int(CFG["delay_hours"]),
        k=int(CFG["k"]),
        norm_window=int(CFG["norm_window"]),
    )

    rows = []
    har_fail = 0
    w_fail = 0

    for t in ts_list[1:]:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        try:
            har_pred_scaled, _ = har.predict_har_vol(tt)
            har_pred_scaled = float(har_pred_scaled)
            true_scaled = float((float(rv_df.loc[t, rv_col]) - rv_min) / denom)
        except Exception as e:
            har_fail += 1
            if har_fail <= 3:
                print("HAR FAIL @", t, "->", repr(e))
            continue

        try:
            weighted_scaled = float(sw.predict_weighted_vol(tt))
        except Exception as e:
            w_fail += 1
            if w_fail <= 3:
                print("WEIGHTED FAIL @", t, "->", repr(e))
            continue

        rows.append(
            {
                "t": pd.Timestamp(t),
                "true_scaled": true_scaled,
                "har_scaled": har_pred_scaled,
                "weighted_scaled": weighted_scaled,
            }
        )

    df = pd.DataFrame(rows).sort_values("t")
    if len(df) < 20:
        raise RuntimeError(f"Too few points to plot (n={len(df)}).")

    df["month"] = df["t"].dt.to_period("M").astype(str)

    print(f"Collected n={len(df)} points. Months:", sorted(df["month"].unique().tolist()))
    print(f"HAR fail count={har_fail}, WEIGHTED fail count={w_fail}")

    for m in sorted(df["month"].unique().tolist()):
        sub = df[df["month"] == m].copy()
        if len(sub) < 5:
            continue

        plt.figure()
        plt.plot(sub["t"], sub["true_scaled"], label="True RV_hourly (scaled)")
        plt.plot(sub["t"], sub["har_scaled"], label="HAR (scaled)")
        plt.plot(sub["t"], sub["weighted_scaled"], label="WEIGHTED (Weighting)")
        plt.title(
            f"{m} — True vs HAR vs Weighted (scaled)\n"
            f"lam={CFG['lam']} delay={CFG['delay_hours']} short/med/long={CFG['short_h']}/{CFG['med_h']}/{CFG['long_h']} "
            f"k={CFG['k']} norm_window={CFG['norm_window']} w={CFG['feature_weights']}"
        )
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()