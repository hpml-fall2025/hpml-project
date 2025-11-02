import time
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------- Page config ----------
st.set_page_config(
    page_title="Realized Variance Dashboard",
    page_icon="📈",
    layout="wide",
)


# ---------- Utilities ----------
@dataclass
class SimParams:
    mu: float
    phi: float
    sigma: float


def simulate_log_rv_series(num_points: int, params: SimParams, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.zeros(num_points, dtype=float)
    series[0] = params.mu  # start near long-run mean in log-space
    for i in range(1, num_points):
        noise = rng.normal(0.0, params.sigma)
        series[i] = params.mu + params.phi * (series[i - 1] - params.mu) + noise
    return series


def initialize_data(num_points: int = 600, freq: str = "S", seed: int = 42) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    index = pd.date_range(end=now, periods=num_points, freq=freq)

    # Simulate two pipelines in log-RV space (stationary AR(1) / OU-like), then exponentiate
    news_params = SimParams(mu=-3.2, phi=0.96, sigma=0.18)
    har_params = SimParams(mu=-3.0, phi=0.95, sigma=0.15)

    log_news = simulate_log_rv_series(num_points, news_params, seed=seed)
    log_har = simulate_log_rv_series(num_points, har_params, seed=seed + 1)

    news_rv = np.exp(log_news)
    har_rv = np.exp(log_har)

    # Actual RV as a noisy combination of both
    rng = np.random.default_rng(seed + 2)
    actual_rv = 0.55 * news_rv + 0.45 * har_rv + rng.normal(0.0, 0.02, size=num_points)
    actual_rv = np.clip(actual_rv, a_min=1e-6, a_max=None)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "news_rv": news_rv,
            "har_rv": har_rv,
            "actual_rv": actual_rv,
        }
    )
    return df


def append_next_point(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    # Evolve the processes one more step forward in time
    rng = np.random.default_rng(seed)

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    # Preserve observed freq (seconds) by default
    next_ts = last_ts + pd.Timedelta(seconds=1)

    # Infer last log values
    last_log_news = float(np.log(df["news_rv"].iloc[-1]))
    last_log_har = float(np.log(df["har_rv"].iloc[-1]))

    news_params = SimParams(mu=-3.2, phi=0.96, sigma=0.18)
    har_params = SimParams(mu=-3.0, phi=0.95, sigma=0.15)

    new_log_news = news_params.mu + news_params.phi * (last_log_news - news_params.mu) + rng.normal(0.0, news_params.sigma)
    new_log_har = har_params.mu + har_params.phi * (last_log_har - har_params.mu) + rng.normal(0.0, har_params.sigma)

    news_rv = float(np.exp(new_log_news))
    har_rv = float(np.exp(new_log_har))

    # Actual as a noisy blend
    actual_rv = 0.55 * news_rv + 0.45 * har_rv + rng.normal(0.0, 0.02)
    actual_rv = float(max(actual_rv, 1e-6))

    next_row = pd.DataFrame(
        {
            "timestamp": [next_ts],
            "news_rv": [news_rv],
            "har_rv": [har_rv],
            "actual_rv": [actual_rv],
        }
    )
    return pd.concat([df, next_row], ignore_index=True)


def filter_timeframe(df: pd.DataFrame, timeframe_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    end = pd.to_datetime(df["timestamp"].iloc[-1])
    mapping: dict[str, timedelta] = {
        "5 minutes": timedelta(minutes=5),
        "30 minutes": timedelta(minutes=30),
        "2 hours": timedelta(hours=2),
        "1 day": timedelta(days=1),
        "All": end - pd.to_datetime(df["timestamp"].iloc[0]),
    }
    window = mapping.get(timeframe_key, timedelta(minutes=30))
    start = end - window
    return df[df["timestamp"] >= start]


def plot_series(df: pd.DataFrame, traces: dict[str, dict], title: str) -> go.Figure:
    fig = go.Figure()
    for name, cfg in traces.items():
        if not cfg.get("show", True):
            continue
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[name],
                mode="lines",
                name=cfg.get("label", name),
                line=dict(width=2.5, color=cfg.get("color")),
                hovertemplate="%{y:.6f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Realized Variance",
        template="plotly_white",
    )
    return fig


def render_dashboard(df: pd.DataFrame, weight: float, show_news: bool, show_har: bool, timeframe: str) -> None:
    if df.empty:
        st.info("No data available.")
        return

    # Combine using current weight
    df = df.copy()
    df["combined_rv"] = weight * df["news_rv"] + (1.0 - weight) * df["har_rv"]

    df_win = filter_timeframe(df, timeframe)

    latest = df.iloc[-1]
    latest_combined = float(latest["combined_rv"]) 
    latest_actual = float(latest["actual_rv"]) 
    abs_err = float(abs(latest_combined - latest_actual))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Combined RV", f"{latest_combined:.6f}")
    kpi2.metric("Actual RV", f"{latest_actual:.6f}")
    kpi3.metric("Abs Error", f"{abs_err:.6f}")
    kpi4.metric("Weight (news)", f"{weight:.2f}")

    st.markdown("---")

    # Main chart: Combined vs Actual
    main_fig = plot_series(
        df_win,
        traces={
            "combined_rv": {"label": "Combined RV", "color": "#3b82f6", "show": True},
            "actual_rv": {"label": "Actual RV", "color": "#ef4444", "show": True},
        },
        title="Combined vs Actual Realized Variance",
    )
    st.plotly_chart(main_fig, use_container_width=True, theme=None)

    # Pipeline charts
    c1, c2 = st.columns(2)
    with c1:
        fig_news = plot_series(
            df_win,
            traces={
                "news_rv": {"label": "News-based RV", "color": "#22c55e", "show": show_news}
            },
            title="News-based RV",
        )
        st.plotly_chart(fig_news, use_container_width=True, theme=None)
    with c2:
        fig_har = plot_series(
            df_win,
            traces={
                "har_rv": {"label": "HAR-RV", "color": "#a855f7", "show": show_har}
            },
            title="HAR-RV",
        )
        st.plotly_chart(fig_har, use_container_width=True, theme=None)


# ---------- Session bootstrap ----------
if "rv_df" not in st.session_state:
    st.session_state.rv_df = initialize_data()


# ---------- Sidebar Controls ----------
st.sidebar.header("Controls")
weight = st.sidebar.slider("Weight on News-based RV", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
timeframe = st.sidebar.selectbox("Timeframe", ["5 minutes", "30 minutes", "2 hours", "1 day", "All"], index=1)
show_news = st.sidebar.toggle("Show News-based RV", value=True)
show_har = st.sidebar.toggle("Show HAR-RV", value=True)

st.sidebar.markdown("---")
live = st.sidebar.toggle("Live update", value=True)
refresh_every = st.sidebar.slider("Refresh every (seconds)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
steps = st.sidebar.slider("Iterations per run", min_value=30, max_value=600, value=180, step=10)


# ---------- Main Layout ----------
st.title("📈 Realized Variance Dashboard")
st.caption(
    "Prototype dashboard comparing realized variance estimates (News-based, HAR-RV) and a weighted combination vs. actual RV."
)

placeholder = st.empty()

def _render_once():
    with placeholder.container():
        render_dashboard(st.session_state.rv_df, weight, show_news, show_har, timeframe)


if live:
    for _ in range(steps):
        st.session_state.rv_df = append_next_point(st.session_state.rv_df)
        _render_once()
        time.sleep(float(refresh_every))
else:
    _render_once()


