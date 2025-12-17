import streamlit as st
import pandas as pd

def render_metrics(df: pd.DataFrame, weight: float = 0.0):
    if df.empty:
        st.info("Waiting for data...")
        return

    latest = df.iloc[-1]

    true_rv = float(latest.get("true_rv", 0.0))
    har_rv = float(latest.get("har_rv", 0.0))
    news_rv = float(latest.get("news_rv", 0.0))
    weighted_rv = float(latest.get("weighted_rv", har_rv))
    news_cnt = int(latest.get("news_cnt", 0))

    abs_err = abs(weighted_rv - true_rv)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Weighted RV", f"{weighted_rv:.6f}", delta_color="off")
    with kpi2:
        st.metric("True RV", f"{true_rv:.6f}", delta_color="off")
    with kpi3:
        st.metric("Abs Error", f"{abs_err:.6f}", delta_color="off")
    with kpi4:
        st.metric("News Cnt", f"{news_cnt:d}", delta_color="off")

    st.caption(f"HAR={har_rv:.6f}  News={news_rv:.6f}")
    st.markdown("---")