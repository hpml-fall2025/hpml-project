import streamlit as st
import pandas as pd

def render_metrics(df: pd.DataFrame, weight: float = 0.0):
    if df.empty:
        st.info("Waiting for data...")
        return

    latest = df.iloc[-1]

    true_rv = float(latest.get("true_rv", 0.0))
    har_rv = float(latest.get("har_rv", 0.0))
    weighted_rv = float(latest.get("weighted_rv", har_rv))

    n = int(st.session_state.get("agg_n", 0))
    sum_comb = float(st.session_state.get("agg_sum_abs_comb", 0.0))
    sum_har = float(st.session_state.get("agg_sum_abs_har", 0.0))

    agg_abs_err = (sum_comb / n) if n > 0 else 0.0
    agg_har_abs_err = (sum_har / n) if n > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("True RV", f"{true_rv:.6f}", delta_color="off")
    with kpi2:
        st.metric("HAR-RV", f"{har_rv:.6f}", delta_color="off")
    with kpi3:
        st.metric("Combined RV", f"{weighted_rv:.6f}", delta_color="off")
    with kpi4:
        st.metric("Mean Abs Error (Combined)", f"{agg_abs_err:.6f}", delta_color="off")
    with kpi5:
        st.metric("Mean Abs Error (HAR-RV)", f"{agg_har_abs_err:.6f}", delta_color="off")

    st.markdown("---")