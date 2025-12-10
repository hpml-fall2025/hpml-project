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

    c1, c2 = st.columns([0.85, 0.15])
    with c2:
        if st.button("⏸ Pause" if st.session_state.is_running else "▶ Resume"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    placeholder = st.empty()

    def _update_and_render():
        if st.session_state.is_running:
            news_data = st.session_state.news_pipe.get_latest_data()
            vol_data = st.session_state.vol_pipe.get_latest_data()
            
            new_record = {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                **news_data,
                **vol_data
            }
            
            import numpy as np
            rng = np.random.default_rng()
            actual = 0.55 * news_data["news_rv"] + 0.45 * vol_data["har_rv"] + rng.normal(0.0, 0.02)
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
