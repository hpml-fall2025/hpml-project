import time
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.layout import init_page, render_header
from dashboard.sidebar import render_sidebar
from dashboard.components.metrics import render_metrics
from dashboard.components.charts import render_charts

from pipelines.news import NewsPipeline
from pipelines.volatility import VolatilityPipeline
from data.store import DataStore

def main():
    init_page()
    render_header()
    
    if "store" not in st.session_state:
        st.session_state.store = DataStore()

    if "news_pipe" not in st.session_state:
        st.session_state.news_pipe = NewsPipeline()
        
    if "vol_pipe" not in st.session_state:
        st.session_state.vol_pipe = VolatilityPipeline()

    if "is_running" not in st.session_state:
        st.session_state.is_running = True

    config = render_sidebar()
    weight = config["weight"]
    timeframe = config["timeframe"]
    refresh_rate = config["refresh_rate"]
    demo_mode = config["demo_mode"]

    # Handle Demo Mode State
    if demo_mode:
        if not st.session_state.get("demo_active", False):
            # Just switched ON
            with st.spinner("Initializing Demo Mode (Training Model)..."):
                st.session_state.vol_pipe.setup_demo_mode()
            st.session_state.demo_active = True
            st.toast("Experiemnt Mode Enabled: Simulating Future Data")
    else:
        if st.session_state.get("demo_active", False):
            # Just switched OFF - Reset pipeline to normal
            st.session_state.vol_pipe = VolatilityPipeline()
            st.session_state.demo_active = False
            st.toast("Experiment Mode Disabled: Returning to Live Data")

    c1, c2 = st.columns([0.85, 0.15])
    with c2:
        if st.button("⏸ Pause" if st.session_state.is_running else "▶ Resume"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    if demo_mode:
        st.markdown("### 🧪 Experiment Mode Active")

    placeholder = st.empty()

    def _update_and_render():
        if st.session_state.is_running:
            news_data = st.session_state.news_pipe.get_latest_data()
            vol_data = st.session_state.vol_pipe.get_latest_data()
            
            # hypothetical headlines for demo
            if demo_mode:
                headline = st.session_state.news_pipe.get_headline()
                st.info(f"📰 **Hypothetical News**: {headline}")
            
            # Ensure we use consistent keys for the frontend components
            # The metrics and charts expect 'har_rv'
            pred_rv = vol_data.get("volatility_prediction", vol_data.get("har_rv", 0.0))
            
            # Timestamp handling: Use timestamp from vol_data if available (demo mode), else current time
            ts = vol_data.get("timestamp", pd.Timestamp.now(tz="UTC"))
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            
            # Ensure timezone awareness (UTC)
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            new_record = {
                "timestamp": ts,
                "har_rv": pred_rv,     # Map prediction to 'har_rv' for compatibility from Refactor
                **news_data,
                **{k:v for k,v in vol_data.items() if k not in ["volatility_prediction", "timestamp"]} 
            }
            
            import numpy as np
            rng = np.random.default_rng()
            
            # Simple aggregation for display
            actual = 0.55 * news_data["news_rv"] + 0.45 * pred_rv + rng.normal(0.0, 0.02)
            new_record["actual_rv"] = float(max(actual, 1e-6))
            
            st.session_state.store.append_data(new_record)

        df = st.session_state.store.get_data(timeframe)
        
        with placeholder.container():
            render_metrics(df, weight)
            render_charts(df, weight)

    if st.session_state.is_running:
        _update_and_render()
        time.sleep(refresh_rate)
        st.rerun()
    else:
        _update_and_render()

if __name__ == "__main__":
    main()
