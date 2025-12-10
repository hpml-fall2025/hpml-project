import streamlit as st
import pandas as pd

def render_metrics(df: pd.DataFrame, weight: float):
    """
    Renders the high-level metrics for the dashboard.
    """
    if df.empty:
        st.info("Waiting for data...")
        return

    latest = df.iloc[-1]
    
    # Calculate values
    news_rv = float(latest["news_rv"])
    har_rv = float(latest["har_rv"])
    actual_rv = float(latest.get("actual_rv", 0.0))
    
    # Calculate Combined RV dynamically if not in DF yet
    combined_rv = weight * news_rv + (1.0 - weight) * har_rv
    
    # Error (if available)
    abs_err = abs(combined_rv - actual_rv) if "actual_rv" in latest else 0.0

    # Render columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Combined RV", f"{combined_rv:.6f}", delta_color="off")
        
    with kpi2:
        st.metric("Actual RV", f"{actual_rv:.6f}", delta_color="off")
        
    with kpi3:
        st.metric("Abs Error", f"{abs_err:.6f}", delta=f"{abs_err:.6f}", delta_color="inverse")
        
    with kpi4:
        st.metric("News Weight", f"{weight:.2f}")
    
    st.markdown("---")
