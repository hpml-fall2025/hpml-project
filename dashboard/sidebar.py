import streamlit as st

def render_sidebar() -> dict:
    st.sidebar.header("Controls")

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["5 minutes", "30 minutes", "2 hours", "1 day", "All"],
        index=2,
    )

    st.sidebar.markdown("---")

    refresh_rate = st.sidebar.slider(
        "Refresh Rate (s)",
        0.1, 0.2, 5.0, 0.8
    )

    if st.sidebar.button("🔁 Reset backtest"):
        st.session_state.reset_backtest = True

    return {"timeframe": timeframe, "refresh_rate": refresh_rate}