import time
import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.layout import init_page, render_header
from dashboard.sidebar import render_sidebar
from dashboard.components.metrics import render_metrics
from dashboard.components.charts import render_charts

from pipelines.news import NewsPipeline
from pipelines.volatility import VolatilityPipeline
from pipelines.weighting import Weighting
from data.store import DataStore

CFG = {
    "lam": -0.1,
    "delay_hours": 2,
    "har_short_h": 8,
    "har_med_h": 12,
    "har_long_h": 275,
    "news_short_h": 1,
    "news_med_h": 8,
    "news_long_h": 48,
    "k": 10,
    "norm_window": 20,
    "feature_weights": (0.5, 0.3, 0.15, 0.05),
    "warmup_steps": 10,
}

BACKTEST_START = dt.datetime(2021, 1, 5, 10, 0, 0)
BACKTEST_END   = dt.datetime(2021, 3, 31, 16, 0, 0)  # last plotted hour = 16:00 (3–4pm)

DEBUG_NEWS = True


def _true_scaled(rv_df: pd.DataFrame, ts: pd.Timestamp, har: VolatilityPipeline, rv_col: str) -> float:
    return float(
        (rv_df.loc[ts, rv_col] - har.train_min[rv_col])
        / (har.train_max[rv_col] - har.train_min[rv_col])
    )


def _build_hourly_rv_df(har: VolatilityPipeline) -> pd.DataFrame:
    rv_calc = har._VolatilityPipeline__rv_calculation
    rv_df = rv_calc(har.full_hist_data).copy()
    rv_df = rv_df[~rv_df.index.duplicated(keep="last")].sort_index()

    rv_df = rv_df.loc[
        (rv_df.index >= pd.Timestamp(BACKTEST_START)) &
        (rv_df.index <= pd.Timestamp(BACKTEST_END))
    ]

    # STRICT market-hour buckets: keep 09:00..16:00 only.
    # 09:00 bucket contains 09:30–09:59, 16:00 bucket contains 16:00–16:59 (4-5pm).
    rv_df = rv_df[(rv_df.index.hour >= 9) & (rv_df.index.hour <= 16)]

    return rv_df


def _init_backtest_state():
    har = VolatilityPipeline(
        short_window_hours=CFG["har_short_h"],
        medium_window_hours=CFG["har_med_h"],
        long_window_hours=CFG["har_long_h"],
    )

    news = NewsPipeline(
        use_gpu=True,
        short_window_hours=CFG["news_short_h"],
        medium_window_hours=CFG["news_med_h"],
        long_window_hours=CFG["news_long_h"],
    )

    w = Weighting(
        lam=CFG["lam"],
        warmup_steps=CFG["warmup_steps"],
        har_pipe=har,
        news_pipe=news,
        feature_weights=CFG["feature_weights"],
        delay_hours=CFG["delay_hours"],
        k=CFG["k"],
        norm_window=CFG["norm_window"],
    )

    rv_df = _build_hourly_rv_df(har)
    if rv_df.empty:
        raise RuntimeError("Hourly RV df is empty for the requested window (1/4/21–3/31/21).")

    ts_list = list(rv_df.index)

    st.session_state.har = har
    st.session_state.news = news
    st.session_state.weighting = w
    st.session_state.rv_df = rv_df
    st.session_state.ts_list = ts_list
    st.session_state.bt_i = 0
    st.session_state.store = DataStore()
    st.session_state.bt_finished = False

    try:
        st.session_state.news_tmin = pd.to_datetime(news.df["Timestamp_hour"]).min()
        st.session_state.news_tmax = pd.to_datetime(news.df["Timestamp_hour"]).max()
    except Exception:
        st.session_state.news_tmin = None
        st.session_state.news_tmax = None

    st.session_state.rv_tmin = rv_df.index.min()
    st.session_state.rv_tmax = rv_df.index.max()

    st.session_state.agg_n = 0
    st.session_state.agg_sum_abs_comb = 0.0
    st.session_state.agg_sum_abs_har = 0.0


def main():
    init_page()
    render_header()

    config = render_sidebar()
    timeframe = config["timeframe"]
    refresh_rate = config["refresh_rate"]

    if "is_running" not in st.session_state:
        st.session_state.is_running = True

    if st.session_state.get("reset_backtest", False) or ("har" not in st.session_state):
        st.session_state.reset_backtest = False
        _init_backtest_state()


    c1, c2 = st.columns([0.85, 0.15])
    with c2:
        if st.button("⏸ Pause" if st.session_state.is_running else "▶ Resume"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    news_placeholder = st.empty()
    main_placeholder = st.empty()

    def _step_once():
        har = st.session_state.har
        news = st.session_state.news
        w = st.session_state.weighting
        rv_df = st.session_state.rv_df
        ts_list = st.session_state.ts_list

        i = st.session_state.bt_i
        if i >= len(ts_list):
            st.session_state.is_running = False
            st.toast("Backtest finished.")
            st.session_state.bt_finished = True
            return

        t = ts_list[i]
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t

        rv_col = "RV_hourly"

        try:
            har_pred_scaled, _ = har.predict_har_vol(tt)
            har_pred_scaled = float(har_pred_scaled)
        except Exception:
            st.session_state.bt_i += 1
            return

        try:
            true_scaled = _true_scaled(rv_df, t, har, rv_col)
        except Exception:
            st.session_state.bt_i += 1
            return

        news_err = None
        try:
            news_val, news_cnt = news.predict_news_vol(
                tt,
                k=CFG["k"],
                feature_weights=CFG["feature_weights"],
                delay_hours=CFG["delay_hours"],
            )
            news_val = float(news_val)
            news_cnt = int(news_cnt)
        except Exception as e:
            news_val = 0.0
            news_cnt = 0
            news_err = str(e)

        if DEBUG_NEWS:
            try:
                prev_hour = news._to_hour(tt) - dt.timedelta(hours=int(CFG["delay_hours"]))
                matched = int((news.df["Timestamp_hour"] == prev_hour).sum())
            except Exception:
                pass

        headline = ""
        headline_ts = None
        try:
            headline = news.get_headline(tt, delay_hours=CFG["delay_hours"])
            if headline and "No headlines" not in headline:
                headline_ts = (pd.Timestamp(t).tz_localize(None) - pd.Timedelta(hours=CFG["delay_hours"]))
        except Exception:
            headline = ""

        if headline and "No headlines" not in headline:
            st.session_state.last_headline = headline
            st.session_state.last_headline_ts = headline_ts

        last_h = st.session_state.get("last_headline", "")
        last_ts = st.session_state.get("last_headline_ts", None)

        if last_h:
            ts_str = ""
            if last_ts is not None:
                ts_str = f" ({pd.Timestamp(last_ts).strftime('%Y-%m-%d %H:%M')})"
            news_placeholder.info(f"📰 {last_h}{ts_str}")
        else:
            news_placeholder.info("📰 (no headlines yet in this backtest window)")

        try:
            weighted_pred = float(w.predict_weighted_vol(tt))
        except Exception:
            weighted_pred = har_pred_scaled
        
        abs_err = abs(weighted_pred - true_scaled)
        har_abs_err = abs(har_pred_scaled - true_scaled)

        st.session_state.agg_n += 1
        st.session_state.agg_sum_abs_comb += float(abs_err)
        st.session_state.agg_sum_abs_har += float(har_abs_err)

        new_record = {
            "timestamp": pd.Timestamp(t),
            "true_rv": true_scaled,
            "har_rv": har_pred_scaled,
            "news_rv": float(news_val),
            "weighted_rv": float(weighted_pred),
            "news_cnt": int(news_cnt),
        }

        st.session_state.store.append_data(new_record)
        st.session_state.bt_i += 1

        df = st.session_state.store.get_data(timeframe)

        with main_placeholder.container():
            render_metrics(df, weight=0.0)
            render_charts(df, weight=0.0)

    if st.session_state.is_running:
        while True:
            _step_once()
            time.sleep(float(refresh_rate))
            if st.session_state.bt_finished:
                time.sleep(10)
    else:
        _step_once()


if __name__ == "__main__":
    main()