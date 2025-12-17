import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from news import NewsPipeline
from volatility import VolatilityPipeline

import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def rv_daily_only(df):
    x = df.copy()
    x["D"] = x.index.date
    x["Per"] = x.groupby("D")["close"].transform("count")
    same_day = x["D"] == x["D"].shift(1)
    diff = x["close"] - x["close"].shift(1)
    x["Ret"] = np.nan
    term = diff * (1.0 / x["Per"])
    x.loc[same_day, "Ret"] = term ** 2
    rv = x.groupby("D")["Ret"].sum().to_frame("RV_daily")
    rv["RV_daily"] = np.sqrt(rv["RV_daily"])
    return rv


def build_day_cache(news, start_day, end_day):
    from finBERT.finbert.finbert import predict

    days = pd.date_range(start_day, end_day, freq="D").date
    ss = {}
    cnt = {}

    ts_date = pd.to_datetime(news.df["Timestamp"], errors="coerce").dt.date

    for d in days:
        rows = news.df.loc[ts_date == d]
        if len(rows) == 0:
            ss[d] = 0.0
            cnt[d] = 0
            continue

        headlines = rows["Headline"].tolist()
        batch = " .".join(headlines)
        res = predict(batch, news.model, use_gpu=news.use_gpu)
        scores = res["sentiment_score"].values.astype(float)

        if len(scores) == 0:
            ss[d] = 0.0
            cnt[d] = 0
            continue

        ss[d] = float((scores ** 2).mean())
        cnt[d] = int(len(scores))

    return ss, cnt


def compute_news_raw_series(dates, ss, cnt, day_weights, delay, k, normalize_weights=True):
    w = np.array(day_weights, dtype=float)
    if normalize_weights:
        s = float(w.sum())
        if s > 0:
            w = w / s

    out = []
    out_n = []

    for d in dates:
        v = 0.0
        n = 0
        for i in range(len(w)):
            dd = d - dt.timedelta(days=int(delay) + i)
            c = cnt.get(dd, 0)
            if c == 0:
                continue
            n += c
            v += float(w[i]) * ss.get(dd, 0.0)

        conf = (n / (n + k)) if n > 0 else 0.0
        out.append(float(conf * v))
        out_n.append(int(n))

    return np.array(out, dtype=float), np.array(out_n, dtype=int)


def normalize_like_code(har_series, news_series, window=10, eps=1e-8):
    h_hist = deque(maxlen=window)
    n_hist = deque(maxlen=window)
    out = []

    for h, n in zip(har_series, news_series):
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


def main():
    start_date = dt.date(2021, 1, 4)
    end_date = dt.date(2021, 3, 31)

    har = VolatilityPipeline()
    news = NewsPipeline()

    rv = rv_daily_only(har.full_hist_data)
    rv = rv.loc[(rv.index >= start_date) & (rv.index <= end_date)]
    dates = list(rv.index)

    rv_min = float(har.train_min["RV_daily"])
    rv_max = float(har.train_max["RV_daily"])
    true_scaled = (rv["RV_daily"].values.astype(float) - rv_min) / (rv_max - rv_min)

    har_pred_by_date = {}
    for d in dates:
        try:
            p, _ = har.predict_har_vol(d)
            har_pred_by_date[d] = float(p)
        except Exception:
            pass

    plot_dates = [d for d in dates if d in har_pred_by_date]
    plot_dates = np.array(plot_dates, dtype=object)

    mask = np.isin(np.array(dates, dtype=object), plot_dates)
    true_scaled_plot = true_scaled[mask]
    har_scaled_plot = np.array([har_pred_by_date[d] for d in plot_dates], dtype=float)

    resid = true_scaled_plot - har_scaled_plot
    abs_resid = np.abs(resid)

    configs = [
        ("d1_[1]_k25", [1.0], 0, 25),
        ("d1_[1]_-1_k25", [1.0], -1, 25),

        ("d2_[0.8,0.2]_k25", [0.8, 0.2], 0, 25),
        ("d2_[0.8,0.2]_-1_k25", [0.8, 0.2], -1, 25),
        ("d2_[0.6,0.4]_k25", [0.6, 0.4], 0, 25),

        ("d3_[0.6,0.25,0.15]_k25", [0.6, 0.25, 0.15], 0, 25),
    ]

    max_len = max(len(w) for _, w, _, _ in configs)
    max_pos_delay = max(0, max(delay for _, _, delay, _ in configs))
    min_neg_delay = min(0, min(delay for _, _, delay, _ in configs))

    need_start = start_date - dt.timedelta(days=max_pos_delay + max_len + 5)
    need_end = end_date + dt.timedelta(days=abs(min_neg_delay) + 5)

    ss, cnt = build_day_cache(news, need_start, need_end)

    plt.figure()
    plt.plot(plot_dates, true_scaled_plot, label="True RV_daily (scaled)")
    plt.plot(plot_dates, har_scaled_plot, label="HAR prediction (scaled)")
    plt.plot(plot_dates, resid, label="True - HAR (residual)")
    plt.plot(plot_dates, abs_resid, label="|True - HAR|")
    plt.title("True RV vs HAR + residuals (scaled space)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    for name, w, delay, k in configs:
        news_raw, _ = compute_news_raw_series(plot_dates, ss, cnt, w, delay, k, normalize_weights=True)
        news_norm = normalize_like_code(har_scaled_plot, news_raw, window=10, eps=1e-8)

        plt.figure()
        plt.plot(plot_dates, true_scaled_plot, label="True RV_daily (scaled)")
        plt.plot(plot_dates, har_scaled_plot, label="HAR prediction (scaled)")
        #plt.plot(plot_dates, resid, label="True - HAR (residual)")
        #plt.plot(plot_dates, abs_resid, label="|True - HAR|")
        plt.plot(plot_dates, news_norm, label=f"News vol normalized (code) [{name}]")
        plt.title(f"RV vs HAR vs residuals vs News: {name}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()