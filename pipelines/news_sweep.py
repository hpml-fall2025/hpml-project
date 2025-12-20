# pipelines/news_sweep.py
import os
import argparse
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, THIS_DIR)

from pipelines.news import NewsPipeline
from pipelines.volatility import VolatilityPipeline


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]

    if len(y_true) == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "R2": float("nan"), "n": 0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))

    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - float(np.sum(err ** 2)) / denom) if denom > 1e-12 else float("nan")

    return {"MAE": mae, "MSE": mse, "R2": r2, "n": int(len(y_true))}


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

    return beta[1:], float(beta[0])


class RunningStats1D:
    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = float(eps)

    def update(self, x: float):
        x = float(x)
        self.n += 1
        d = x - self.mean
        self.mean += d / float(self.n)
        d2 = x - self.mean
        self.M2 += d * d2

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        var = self.M2 / float(self.n - 1)
        if not np.isfinite(var) or var < self.eps:
            return 1.0
        return float(np.sqrt(var))


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


def _build_target_hours(rv_hourly: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    idx = rv_hourly.index
    idx = idx[(idx >= start) & (idx <= end)]
    return [pd.Timestamp(t) for t in idx if 9 <= pd.Timestamp(t).hour <= 16]


def _news_feature_row(news: NewsPipeline, when: pd.Timestamp, delay_hours: int) -> tuple[np.ndarray, int]:
    tt = when.to_pydatetime() if hasattr(when, "to_pydatetime") else when

    if hasattr(news, "get_feature_row"):
        try:
            row4, n_used = news.get_feature_row(tt, delay_hours=int(delay_hours))
            row4 = np.asarray(row4, dtype=float).reshape(4)
            return row4, int(n_used)
        except ValueError as e:
            if "Requested hour not present in hour index" in str(e):
                return np.full(4, np.nan, dtype=float), 0
            return np.full(4, np.nan, dtype=float), 0

    if hasattr(news, "_to_hour") and hasattr(news, "_ensure_stream_until") and hasattr(news, "_by_hour"):
        try:
            target_hour = news._to_hour(tt)
            prev_hour = target_hour - pd.Timedelta(hours=int(delay_hours))
            news._ensure_stream_until(prev_hour)
            if prev_hour not in news._by_hour:
                return np.full(4, np.nan, dtype=float), 0
            v, s, m, l, cnt_sum = news._by_hour[prev_hour]
            return np.asarray([v, s, m, l], dtype=float), int(cnt_sum)
        except ValueError as e:
            if "Requested hour not present in hour index" in str(e):
                return np.full(4, np.nan, dtype=float), 0
            return np.full(4, np.nan, dtype=float), 0

    return np.full(4, np.nan, dtype=float), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy_csv", type=str, default="SPY1min_clean.csv")
    ap.add_argument("--out", type=str, default="news_sweep_results.csv")

    ap.add_argument("--train_start", type=str, default="2021-01-04 00:00:00")
    ap.add_argument("--train_end", type=str, default="2021-02-15 23:59:59")
    ap.add_argument("--val_start", type=str, default="2021-02-16 00:00:00")
    ap.add_argument("--val_end", type=str, default="2021-02-28 23:59:59")

    ap.add_argument("--har_short_h", type=int, default=8)
    ap.add_argument("--har_med_h", type=int, default=12)
    ap.add_argument("--har_long_h", type=int, default=275)
    ap.add_argument("--har_train_end", type=str, default="2021-01-01 23:59:59")

    ap.add_argument("--delay_grid", type=str, default="0")
    ap.add_argument("--k_grid", type=str, default="0")
    ap.add_argument("--short_grid", type=str, default="4, 6, 8")
    ap.add_argument("--med_grid", type=str, default="20, 21, 22, 23, 24")
    ap.add_argument("--long_grid", type=str, default="70, 75, 80, 85")
    ap.add_argument("--alpha_grid", type=str, default="1e-2")
    ap.add_argument("--lambda_grid", type=str, default="-0.1, -0.09, -0.08, -0.07")
    args = ap.parse_args()

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end)

    minute = _load_spy(args.spy_csv)
    rv_hourly = _hourly_rv(minute)

    all_start = min(train_start, val_start)
    all_end = max(train_end, val_end)
    target_hours = _build_target_hours(rv_hourly, all_start, all_end)
    if len(target_hours) == 0:
        raise RuntimeError("No target hours in range (check RV series + date bounds).")

    har = VolatilityPipeline(
        short_window_hours=args.har_short_h,
        medium_window_hours=args.har_med_h,
        long_window_hours=args.har_long_h,
        train_end=args.har_train_end,
    )

    denom_rv = float(har.train_max["RV_hourly"] - har.train_min["RV_hourly"])
    if not np.isfinite(denom_rv) or abs(denom_rv) < 1e-12:
        raise RuntimeError("Degenerate HAR scaling for RV_hourly (train_max == train_min).")

    delay_grid = [int(x) for x in args.delay_grid.split(",") if x.strip() != ""]
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip() != ""]
    short_grid = [int(x) for x in args.short_grid.split(",") if x.strip() != ""]
    med_grid = [int(x) for x in args.med_grid.split(",") if x.strip() != ""]
    long_grid = [int(x) for x in args.long_grid.split(",") if x.strip() != ""]
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip() != ""]
    lambda_grid = [float(x) for x in args.lambda_grid.split(",") if x.strip() != ""]

    train_mask = np.array([(train_start <= t <= train_end) for t in target_hours], dtype=bool)
    val_mask = np.array([(val_start <= t <= val_end) for t in target_hours], dtype=bool)

    rows = []

    for sh in short_grid:
        for mh in med_grid:
            for lh in long_grid:
                if not (sh < mh < lh):
                    continue

                news = NewsPipeline(
                    use_gpu=True,
                    short_window_hours=sh,
                    medium_window_hours=mh,
                    long_window_hours=lh,
                )

                for delay in delay_grid:
                    news.reset_stream()
                    if hasattr(har, "reset_stream"):
                        har.reset_stream()

                    H_pred = []
                    y_true = []
                    X_feat = []
                    n_used_list = []

                    for t in target_hours:
                        tt = t.to_pydatetime()

                        try:
                            Ht, _ = har.predict_har_vol(tt)
                            Ht = float(Ht)
                        except Exception:
                            Ht = float("nan")

                        if t not in rv_hourly.index:
                            yt = float("nan")
                        else:
                            yt = float((float(rv_hourly.loc[t]) - float(har.train_min["RV_hourly"])) / denom_rv)

                        row4, n_used = _news_feature_row(news, pd.Timestamp(t), delay_hours=int(delay))

                        H_pred.append(Ht)
                        y_true.append(yt)
                        X_feat.append(row4.astype(float))
                        n_used_list.append(int(n_used))

                    H_pred = np.asarray(H_pred, dtype=float)
                    y_true = np.asarray(y_true, dtype=float)
                    X_feat = np.asarray(X_feat, dtype=float)
                    n_used_arr = np.asarray(n_used_list, dtype=float)

                    ok = np.isfinite(H_pred) & np.isfinite(y_true) & np.isfinite(X_feat).all(axis=1)
                    if int(np.sum(ok & train_mask)) < 50 or int(np.sum(ok & val_mask)) < 20:
                        continue

                    for k in k_grid:
                        conf = np.zeros_like(n_used_arr, dtype=float)
                        pos = n_used_arr > 0
                        conf[pos] = n_used_arr[pos] / (n_used_arr[pos] + float(k))
                        X = X_feat * conf.reshape(-1, 1)

                        y_resid = y_true - H_pred

                        train_idx = np.where(ok & train_mask)[0]
                        val_idx = np.where(ok & val_mask)[0]

                        Xtr = X[train_idx]
                        ytr = y_resid[train_idx]
                        Xva = X[val_idx]
                        yva_true = y_true[val_idx]
                        Hva = H_pred[val_idx]

                        if len(Xtr) < 50 or len(Xva) < 20:
                            continue

                        har_val_metrics = _metrics(yva_true, Hva)

                        best = None

                        for alpha in alpha_grid:
                            try:
                                w4, b0 = _ridge_fit(Xtr, ytr, alpha=float(alpha))
                            except Exception:
                                continue

                            Ntr = Xtr @ w4 + float(b0)
                            Htr = H_pred[train_idx]

                            rs_n = RunningStats1D(eps=1e-8)
                            rs_h = RunningStats1D(eps=1e-8)
                            for a, hh in zip(Ntr.tolist(), Htr.tolist()):
                                if np.isfinite(a) and np.isfinite(hh):
                                    rs_n.update(float(a))
                                    rs_h.update(float(hh))

                            mu_n = float(rs_n.mean)
                            sd_n = float(rs_n.std())
                            mu_h = float(rs_h.mean)
                            sd_h = float(rs_h.std())

                            Nva_raw = Xva @ w4 + float(b0)
                            Nva_norm = mu_h + (sd_h * (Nva_raw - mu_n) / (sd_n + 1e-8))

                            news_only_va_metrics = _metrics(yva_true, Nva_norm)

                            for lam in lambda_grid:
                                pred_va = Hva + float(lam) * Nva_norm
                                m_va = _metrics(yva_true, pred_va)
                                key = (m_va["MSE"], m_va["MAE"])

                                if best is None or key < best["key"]:
                                    best = {
                                        "key": key,
                                        "alpha": float(alpha),
                                        "lam": float(lam),
                                        "w": w4.copy(),
                                        "b0": float(b0),
                                        "mu_n": float(mu_n),
                                        "sd_n": float(sd_n),
                                        "mu_h": float(mu_h),
                                        "sd_h": float(sd_h),
                                        "val_weighted": m_va,
                                        "val_news_only": news_only_va_metrics,
                                    }

                        if best is None:
                            continue

                        w4 = best["w"]
                        b0 = best["b0"]

                        rows.append({
                            "short_h": int(sh),
                            "med_h": int(mh),
                            "long_h": int(lh),
                            "delay_hours": int(delay),
                            "k": int(k),
                            "ridge_alpha": float(best["alpha"]),
                            "lambda": float(best["lam"]),

                            "w_hourly": float(w4[0]),
                            "w_short": float(w4[1]),
                            "w_medium": float(w4[2]),
                            "w_long": float(w4[3]),
                            "bias": float(b0),

                            "mu_h_train": float(best["mu_h"]),
                            "sd_h_train": float(best["sd_h"]),
                            "mu_n_train": float(best["mu_n"]),
                            "sd_n_train": float(best["sd_n"]),

                            "val_MAE_weighted": float(best["val_weighted"]["MAE"]),
                            "val_MSE_weighted": float(best["val_weighted"]["MSE"]),
                            "val_R2_weighted": float(best["val_weighted"]["R2"]),
                            "val_n": int(best["val_weighted"]["n"]),

                            "val_MAE_news_only": float(best["val_news_only"]["MAE"]),
                            "val_MSE_news_only": float(best["val_news_only"]["MSE"]),
                            "val_R2_news_only": float(best["val_news_only"]["R2"]),
                            "val_n_news_only": int(best["val_news_only"]["n"]),

                            "val_MAE_har": float(har_val_metrics["MAE"]),
                            "val_MSE_har": float(har_val_metrics["MSE"]),
                            "val_R2_har": float(har_val_metrics["R2"]),
                            "val_n_har": int(har_val_metrics["n"]),
                        })

                        print(
                            "done",
                            f"sh={sh} mh={mh} lh={lh} delay={delay} k={k}",
                            f"val_mse={rows[-1]['val_MSE_weighted']:.6g}",
                        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No results produced.")
        return

    out = out.sort_values(["val_MAE_weighted", "val_MSE_weighted"], ascending=[True, True]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print("\nTop 10 by VAL MAE (weighted):")
    print(out.head(10).to_string(index=False))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()