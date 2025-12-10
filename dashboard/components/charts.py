import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def _plot_series(df: pd.DataFrame, traces: dict[str, dict], title: str, y_title: str = "Realized Variance") -> go.Figure:
    fig = go.Figure()
    for name, cfg in traces.items():
        if name not in df.columns:
            continue
            
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[name],
                mode="lines",
                name=cfg.get("label", name),
                line=dict(width=2, color=cfg.get("color")),
                hovertemplate="%{y:.6f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        title=title,
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title=y_title,
        template="plotly_dark", # Matches the "Premium" look
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def render_charts(df: pd.DataFrame, weight: float):
    """
    Renders the charts: Combined, News, and HAR.
    """
    if df.empty:
        return

    # Calculate combined on the fly for visualization
    df = df.copy()
    df["combined_rv"] = weight * df["news_rv"] + (1.0 - weight) * df["har_rv"]

    # Main Chart: Combined vs Actual
    main_fig = _plot_series(
        df,
        traces={
            "combined_rv": {"label": "Combined Prediction", "color": "#3b82f6"},
            "actual_rv": {"label": "Actual RV", "color": "#ef4444"},
        },
        title="Predictive Performance: Combined Model vs Actual",
    )
    st.plotly_chart(main_fig, use_container_width=True)

    # Sub Charts
    c1, c2 = st.columns(2)
    with c1:
        fig_news = _plot_series(
            df,
            traces={
                "news_rv": {"label": "FinBERT Sentiment (News)", "color": "#22c55e"}
            },
            title="Pipeline 1: News Sentiment Signal",
        )
        st.plotly_chart(fig_news, use_container_width=True)
        
    with c2:
        fig_vol = _plot_series(
            df,
            traces={
                "har_rv": {"label": "Statistical Prediction (HAR)", "color": "#a855f7"}
            },
            title="Pipeline 2: Volatility Estimate",
        )
        st.plotly_chart(fig_vol, use_container_width=True)
