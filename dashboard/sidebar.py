import streamlit as st

def render_sidebar() -> dict:
    st.sidebar.header("Controls")
    
    weight = st.sidebar.slider(
        "News Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01,
        help="Weight assigned to the FinBERT news score."
    )
    
    timeframe = st.sidebar.selectbox(
        "Timeframe", 
        ["5 minutes", "30 minutes", "2 hours", "1 day", "All"], 
        index=1
    )
    
    st.sidebar.markdown("---")
    
    refresh_rate = st.sidebar.slider(
        "Refresh Rate (s)", 
        0.5, 5.0, 1.0, 0.5
    )
    
    demo_mode = st.sidebar.toggle("Demo Mode")
    
    return {
        "weight": weight,
        "timeframe": timeframe,
        "refresh_rate": refresh_rate,
        "demo_mode": demo_mode
    }
