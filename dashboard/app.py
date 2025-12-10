import time
import streamlit as st
import pandas as pd
import sys
import os

# Add root to path so we can import from pipelines and dashboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.layout import init_page, render_header
from dashboard.sidebar import render_sidebar
from dashboard.components.metrics import render_metrics
from dashboard.components.charts import render_charts

from pipelines.news import NewsPipeline
from pipelines.volatility import VolatilityPipeline
from data.store import DataStore

def main():
    # 1. Initialize Page
    init_page()
    render_header()
    
    # 2. Pipeline & Data Initialization (Session State)
    if "store" not in st.session_state:
        st.session_state.store = DataStore()
        
    if "news_pipe" not in st.session_state:
        st.session_state.news_pipe = NewsPipeline()
        
    if "vol_pipe" not in st.session_state:
        st.session_state.vol_pipe = VolatilityPipeline()

    if "is_running" not in st.session_state:
        st.session_state.is_running = True

    # 3. Sidebar Controls
    config = render_sidebar()
    weight = config["weight"]
    timeframe = config["timeframe"]
    refresh_rate = config["refresh_rate"]

    # Pause/Resume Button in main area or sidebar
    # Placing it in sidebar for cleaner UI, updating sidebar.py would be ideal but I can add it here too
    # or just use a main area button. Let's use a main area toggle for visibility.
    c1, c2 = st.columns([0.85, 0.15])
    with c2:
        if st.button("⏸ Pause" if st.session_state.is_running else "▶ Resume"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()

    # 4. Main Rendering Container
    placeholder = st.empty()

    def _update_and_render():
        # A. Fetch new data
        if st.session_state.is_running:
            news_data = st.session_state.news_pipe.get_latest_data()
            vol_data = st.session_state.vol_pipe.get_latest_data()
            
            # Combine into one record
            new_record = {
                "timestamp": pd.Timestamp.now(tz="UTC"),
                **news_data,
                **vol_data
            }
            
            # Simulate "Actual" for comparison (in valid real scenario this might come from a 3rd pipeline)
            # Using the same mock logic as the original app for continuity
            import numpy as np
            rng = np.random.default_rng()
            actual = 0.55 * news_data["news_rv"] + 0.45 * vol_data["har_rv"] + rng.normal(0.0, 0.02)
            new_record["actual_rv"] = float(max(actual, 1e-6))
            
            st.session_state.store.append_data(new_record)

        # B. Get data for view
        df = st.session_state.store.get_data(timeframe)
        
        # C. Render logic
        with placeholder.container():
            render_metrics(df, weight)
            render_charts(df, weight)

    # 5. Live Loop
    # If using st.rerun() style loop, we need to handle the sleep carefully.
    # A simple approach for Streamlit is a while loop inside the script if we want "true" streaming,
    # or relying on st.rerun.
    
    # We will use the loop approach for smoother updates without full page reloads if possible,
    # but Streamlit works best with re-runs for interactivity.
    # However, to avoid "Performance" warnings and UI locking, the best pattern often is:
    
    if st.session_state.is_running:
        _update_and_render()
        time.sleep(refresh_rate)
        st.rerun()
    else:
        _update_and_render()

if __name__ == "__main__":
    main()
