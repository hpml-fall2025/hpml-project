import streamlit as st
import plotly.graph_objects as go
import pandas as pd

MAIN_Y_RANGE = (0.008, 0.145)   # True vs Weighted (scaled RV)
HAR_Y_RANGE  = (0.020, 0.115)   # HAR pred (scaled RV)
NEWS_Y_RANGE = (0.0, 0.30)      # News signal (always >= 0)

def _plot_series(
    df: pd.DataFrame,
    traces: dict[str, dict],
    title: str,
    y_title: str = "Value",
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16.1, 9.5], pattern="hour"),
        ]
    )

    if y_range is not None:
        fig.update_yaxes(range=[float(y_range[0]), float(y_range[1])])

    return fig

def render_charts(df: pd.DataFrame, weight: float):
    if df.empty:
        return

    main_pred_col = "weighted_rv" if "weighted_rv" in df.columns else "har_rv"
    har_pred_col = "har_rv"

    main_fig = _plot_series(
        df,
        traces={
            "true_rv": {"label": "True RV", "color": "#ef4444"},
            main_pred_col: {
                "label": "Combined Prediction" if main_pred_col == "weighted_rv" else "HAR Prediction",
                "color": "#3b82f6",
            },
            har_pred_col: {
                "label": "HAR-RV Prediction" if main_pred_col == "weighted_rv" else "HAR Prediction",
                "color": "#22c55e",
            },
        },
        title="Hourly: Prediction vs True",
        y_title="Scaled RV",
        y_range=MAIN_Y_RANGE,
    )
    st.plotly_chart(main_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_news = _plot_series(
            df,
            traces={"news_rv": {"label": "News Signal", "color": "#a855f7"}},
            title="News signal (scaled space)",
            y_title="News value",
            y_range=NEWS_Y_RANGE,
        )
        st.plotly_chart(fig_news, use_container_width=True)

    with c2:
        fig_har = _plot_series(
            df,
            traces={"har_rv": {"label": "HAR Prediction", "color": "#22c55e"}},
            title="HAR prediction (scaled space)",
            y_title="Scaled RV",
            y_range=HAR_Y_RANGE,
        )
        st.plotly_chart(fig_har, use_container_width=True)
