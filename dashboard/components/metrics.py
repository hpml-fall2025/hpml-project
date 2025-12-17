import streamlit as st
import pandas as pd

def render_metrics(df: pd.DataFrame, weight: float):
    if df.empty:
        st.info("Waiting for data...")
        return

    latest = df.iloc[-1]

    true_rv = float(latest.get("true_rv", float("nan")))
    har_rv = float(latest.get("har_rv", float("nan")))
    news_rv = float(latest.get("news_rv", float("nan")))
    weighted_rv = float(latest.get("weighted_rv", float("nan")))

    pred = weighted_rv if pd.notna(weighted_rv) else har_rv
    abs_err = abs(pred - true_rv) if (pd.notna(pred) and pd.notna(true_rv)) else float("nan")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Weighted RV", f"{pred:.6f}" if pd.notna(pred) else "—", delta_color="off")
    kpi2.metric("True RV", f"{true_rv:.6f}" if pd.notna(true_rv) else "—", delta_color="off")
    kpi3.metric("Abs Error", f"{abs_err:.6f}" if pd.notna(abs_err) else "—", delta_color="inverse")
    kpi4.metric("HAR RV", f"{har_rv:.6f}" if pd.notna(har_rv) else "—", delta_color="off")

    st.markdown("---")