import streamlit as st

def init_page():
    st.set_page_config(
        page_title="Realized Variance Dashboard",
        page_icon="📈",
        layout="wide",
    )
    # Apply custom CSS for premium look (Phase 29 placeholder)
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.title("📈 Realized Variance Dashboard")
    st.caption("Real-time analysis of News Sentiment and Volatility Statistics.")
    st.markdown("---")
