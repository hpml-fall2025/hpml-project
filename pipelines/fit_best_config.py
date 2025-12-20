# pipelines/fit_best_config.py
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from pipelines.volatility import VolatilityPipeline
from pipelines.news import NewsPipeline


def _load_spy(csv_path: str) -> pd.DataFrame:
    x = pd.read_csv(csv_path)
    x["date"] = pd.to_datetime(x["date"])
    x["date"] = (
        x["date"]
        .dt.tz_localize("America/Denver")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    x = x.set_index("date").sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x.dropna(subset=["close"])
    x = x[["open", "high", "low", "close", "volume"]].copy()
    try:
        x = x.between_time("09:30", "16:59:59", inclusive="both")
    except TypeError:
        x = x.between_time("09:30", "16:59:59")
    return x


def _hourly_rv(minute_df: pd.DataFrame) -> pd.Series:
    x = minute_df.copy()
    x = x.sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x["H"] = x.index.floor("h")
    x["Per"] = x.groupby("H")["close"].transform("count")
    same_hour = x["H"] == x["H"].shift(1)
    diff = x["close"] - x["close"].shift(1)
    x["Ret"] = np.nan
    term = diff * (1.0 / x["Per"])
    x.loc[same_hour, "Ret"] = term ** 2
    rv = x.groupby("H")["Ret"].sum()
    rv = np.sqrt(rv)
    rv.name = "RV_hourly"
    rv = rv[~rv.index.duplicated(keep="last")].sort_index()
    return rv


def _build_hours(rv_hourly: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    idx = rv_hourly.index
    idx = idx[(idx >= start) & (idx <= end)]
    out = []
    for t in idx:
        tt = pd.Timestamp(t)
        if 9 <= tt.hour <= 16:
            out.append(tt)
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    X1 = np.column_stack([np.ones(len(X), dtype=float), X])
    XT = X1.T
    A = XT @ X1
    P = np.eye(A.shape[0], dtype=float)
    P[0, 0] = 0.0
    A = A + float(alpha) * P
    b = XT @ y
    beta = np.linalg.solve(A, b)
    return beta[1:].astype(float), float(beta[0])


class _Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.M2 += d * d2

    def finalize(self) -> tuple[float, float]:
        if self.n <= 1:
            return float(self.mean), float("nan")
        var = self.M2 / (self.n - 1)
        return float(self.mean), float(np.sqrt(var))


def _safe_feature_row(news: NewsPipeline, t: pd.Timestamp, delay_hours: int) -> tuple[np.ndarray, int]:
    try:
        row4, n_used = news.get_feature_row(t.to_pydatetime(), delay_hours=int(delay_hours))
        row4 = np.asarray(row4, dtype=float).reshape(4)
        return row4, int(n_used)
    except Exception:
        return np.full(4, np.nan, dtype=float), 0


def _har_params_dict(har: VolatilityPipeline) -> dict:
    if getattr(har, "model", None) is None:
        return {}
    try:
        p = har.model.params
        if hasattr(p, "to_dict"):
            d = {str(k): float(v) for k, v in p.to_dict().items()}
        else:
            d = {str(i): float(p[i]) for i in range(len(p))}
        return d
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy_csv", type=str, default="SPY1min_clean.csv")
    ap.add_argument("--out_json", type=str, default="best_config_trained.json")

    ap.add_argument("--train_start", type=str, default="2021-01-04 00:00:00")
    ap.add_argument("--train_end", type=str, default="2021-02-28 23:59:59")

    ap.add_argument("--har_short_h", type=int, default=8)
    ap.add_argument("--har_med_h", type=int, default=12)
    ap.add_argument("--har_long_h", type=int, default=275)
    ap.add_argument("--har_train_end", type=str, default="2021-01-01 23:59:59")

    ap.add_argument("--news_short_h", type=int, default=6)
    ap.add_argument("--news_med_h", type=int, default=24)
    ap.add_argument("--news_long_h", type=int, default=75)
    ap.add_argument("--delay_hours", type=int, default=0)
    ap.add_argument("--k", type=int, default=0)

    ap.add_argument("--ridge_alpha", type=float, default=0.01)
    ap.add_argument("--lambda", dest="lam", type=float, default=-0.1)
    args = ap.parse_args()

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)

    minute = _load_spy(args.spy_csv)
    rv_hourly = _hourly_rv(minute)

    hours = _build_hours(rv_hourly, train_start, train_end)
    if len(hours) == 0:
        raise RuntimeError("empty training hours (check train_start/train_end and RV index)")

    har = VolatilityPipeline(
        short_window_hours=args.har_short_h,
        medium_window_hours=args.har_med_h,
        long_window_hours=args.har_long_h,
        train_end=args.har_train_end,
    )

    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=args.news_short_h,
        medium_window_hours=args.news_med_h,
        long_window_hours=args.news_long_h,
    )

    if hasattr(har, "reset_stream"):
        har.reset_stream()
    if hasattr(news, "reset_stream"):
        news.reset_stream()

    denom_rv = float(har.train_max["RV_hourly"] - har.train_min["RV_hourly"])
    if (not np.isfinite(denom_rv)) or abs(denom_rv) < 1e-12:
        raise RuntimeError("degenerate HAR scaling for RV_hourly")

    H_pred = []
    y_true = []
    X_feat = []
    n_used = []

    for t in hours:
        tt = t.to_pydatetime()

        try:
            Ht, _ = har.predict_har_vol(tt)
            Ht = float(Ht)
        except Exception:
            Ht = float("nan")

        if t in rv_hourly.index:
            yt = float((float(rv_hourly.loc[t]) - float(har.train_min["RV_hourly"])) / denom_rv)
        else:
            yt = float("nan")

        row4, cnt = _safe_feature_row(news, t, delay_hours=int(args.delay_hours))

        H_pred.append(Ht)
        y_true.append(yt)
        X_feat.append(row4)
        n_used.append(float(cnt))

    H_pred = np.asarray(H_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    X_feat = np.asarray(X_feat, dtype=float)
    n_used = np.asarray(n_used, dtype=float)

    ok = np.isfinite(H_pred) & np.isfinite(y_true) & np.isfinite(X_feat).all(axis=1)
    if int(np.sum(ok)) < 100:
        raise RuntimeError("too few valid training rows (HAR/news missing on most hours)")

    k = int(args.k)
    conf = np.zeros_like(n_used, dtype=float)
    pos = n_used > 0
    if k == 0:
        conf[pos] = 1.0
    else:
        conf[pos] = n_used[pos] / (n_used[pos] + float(k))

    X = X_feat * conf.reshape(-1, 1)
    y_resid = y_true - H_pred

    Xtr = X[ok]
    ytr = y_resid[ok]
    w4, b = _ridge_fit(Xtr, ytr, alpha=float(args.ridge_alpha))

    N_raw = (Xtr @ w4) + float(b)

    w_h = _Welford()
    w_n = _Welford()
    for hv, nv in zip(H_pred[ok], N_raw):
        if np.isfinite(hv):
            w_h.update(float(hv))
        if np.isfinite(nv):
            w_n.update(float(nv))

    mu_h, sd_h = w_h.finalize()
    mu_n, sd_n = w_n.finalize()

    out = {
        "har": {
            "short_h": int(args.har_short_h),
            "med_h": int(args.har_med_h),
            "long_h": int(args.har_long_h),
            "train_end": str(pd.Timestamp(args.har_train_end)),
            "model_params": _har_params_dict(har),
            "train_min": {k: float(v) for k, v in har.train_min.to_dict().items()},
            "train_max": {k: float(v) for k, v in har.train_max.to_dict().items()},
        },
        "news": {
            "short_h": int(args.news_short_h),
            "med_h": int(args.news_med_h),
            "long_h": int(args.news_long_h),
            "delay_hours": int(args.delay_hours),
            "k": int(args.k),
            "ridge_alpha": float(args.ridge_alpha),
        },
        "weighting": {
            "lambda": float(args.lam),
            "w_hourly": float(w4[0]),
            "w_short": float(w4[1]),
            "w_medium": float(w4[2]),
            "w_long": float(w4[3]),
            "bias": float(b),
            "mu_h": float(mu_h),
            "sd_h": float(sd_h),
            "mu_n": float(mu_n),
            "sd_n": float(sd_n),
        },
        "fit_window": {
            "train_start": str(train_start),
            "train_end": str(train_end),
            "n_hours_total": int(len(hours)),
            "n_rows_used": int(np.sum(ok)),
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(args.out_json)
    print("har.model_params:", out["har"]["model_params"])
    print("weighting:", out["weighting"])


if __name__ == "__main__":
    main()