import streamlit as st
from .layout import init_page, render_header
from .sidebar import render_sidebar

def main():
    # Phase 14: Page Config
    init_page()
    
    # Render layout
    render_header()
    
    # Phase 17: Sidebar
    config = render_sidebar()
    
    st.write("Dashboard Skeleton Complete. Ready for components.")

if __name__ == "__main__":
    main()
