import os
import argparse
import datetime as dt

import numpy as np
import pandas as pd


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


def _build_features(rv_hourly: pd.Series, short_h: int, med_h: int, long_h: int) -> pd.DataFrame:
    df = pd.DataFrame({"RV_hourly": rv_hourly.astype(float)})

    df["RV_short"] = df["RV_hourly"].rolling(window=int(short_h)).mean()
    df["RV_medium"] = df["RV_hourly"].rolling(window=int(med_h)).mean()
    df["RV_long"] = df["RV_hourly"].rolling(window=int(long_h)).mean()

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _scale_like_pipeline(df: pd.DataFrame, train_min: pd.Series, train_max: pd.Series, cols: list[str]) -> pd.DataFrame:
    denom = (train_max[cols] - train_min[cols]).replace(0.0, np.nan)
    out = (df[cols] - train_min[cols]) / denom
    out = out.dropna()
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    XT = X.T
    A = XT @ X

    P = np.eye(X.shape[1], dtype=float)
    P[0, 0] = 0.0

    A = A + float(alpha) * P
    b = XT @ y
    return np.linalg.solve(A, b)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 1e-12 else float("nan")

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Corr": corr, "n": int(len(y_true))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="SPY1min_clean.csv")
    ap.add_argument("--train_end", type=str, default="2016-12-31 23:59:59")
    ap.add_argument("--val_start", type=str, default="2017-01-01 00:00:00")
    ap.add_argument("--val_end", type=str, default="2018-12-31 23:59:59")
    ap.add_argument("--out", type=str, default="har_sweep_results.csv")
    args = ap.parse_args()

    train_end = pd.to_datetime(args.train_end)
    val_start = pd.to_datetime(args.val_start)
    val_end = pd.to_datetime(args.val_end)

    minute = _load_spy(args.csv)
    rv_hourly = _hourly_rv(minute)

    short_grid = [2, 4, 6, 8, 10]
    med_grid = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    long_grid = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
    alpha_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    rows = []
    cols = ["RV_hourly", "RV_short", "RV_medium", "RV_long"]

    total = 0
    for sh in short_grid:
        for mh in med_grid:
            for lh in long_grid:
                if not (sh < mh < lh):
                    continue
                total += len(alpha_grid)

    done = 0
    for sh in short_grid:
        for mh in med_grid:
            for lh in long_grid:
                if not (sh < mh < lh):
                    continue

                feats = _build_features(rv_hourly, sh, mh, lh)

                train_df = feats.loc[:train_end].copy()
                val_df = feats.loc[val_start:val_end].copy()

                if len(train_df) < 2000 or len(val_df) < 200:
                    continue

                train_min = train_df.min()
                train_max = train_df.max()

                train_scaled = _scale_like_pipeline(train_df, train_min, train_max, cols)
                val_scaled = _scale_like_pipeline(val_df, train_min, train_max, cols)

                train_scaled["Target"] = train_scaled["RV_hourly"].shift(-1)
                val_scaled["Target"] = val_scaled["RV_hourly"].shift(-1)

                train_scaled = train_scaled.dropna()
                val_scaled = val_scaled.dropna()

                if len(train_scaled) < 2000 or len(val_scaled) < 200:
                    continue

                Xtr = train_scaled[cols].to_numpy(dtype=float)
                ytr = train_scaled["Target"].to_numpy(dtype=float)
                Xva = val_scaled[cols].to_numpy(dtype=float)
                yva = val_scaled["Target"].to_numpy(dtype=float)

                Xtr = np.column_stack([np.ones(len(Xtr), dtype=float), Xtr])
                Xva = np.column_stack([np.ones(len(Xva), dtype=float), Xva])

                for alpha in alpha_grid:
                    try:
                        beta = _ridge_fit(Xtr, ytr, alpha=float(alpha))
                        yhat = Xva @ beta
                        m = _metrics(yva, yhat)

                        rows.append({
                            "short_h": int(sh),
                            "med_h": int(mh),
                            "long_h": int(lh),
                            "ridge_alpha": float(alpha),
                            **m
                        })
                    except Exception:
                        pass

                    done += 1
                    if done % 200 == 0:
                        print(f"progress: {done}/{total}")

    out = pd.DataFrame(rows)
    if out.empty:
        print("No results produced. (Likely too-small train/val slices after rolling windows.)")
        return

    out = out.sort_values(["MSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print("\nTop 10 by MSE:")
    print(out.head(10).to_string(index=False))

    print("\nTop 10 by MAE:")
    print(out.sort_values("MAE", ascending=True).head(10).to_string(index=False))

    print("\nTop 10 by R2:")
    print(out.sort_values("R2", ascending=False).head(10).to_string(index=False))

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()