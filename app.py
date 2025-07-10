# app.py
import streamlit as st

st.set_page_config(
    page_title="Telecom Sentinel",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Welcome to Telecom Sentinel!")
st.subheader("AI-Powered Network Anomaly Detection & Customer Impact Analysis")

st.markdown("""
This application is a demonstration of an end-to-end Machine Learning project for the telecom industry.
It showcases real-time network anomaly detection and translates technical issues into business-centric insights,
like customer churn risk and proactive retention strategies.

**Navigate through the pages on the left to explore the different modules:**

- **🌍 Global Network Overview:** A high-level dashboard of the entire network's health.
- **📡 Anomaly Deep Dive:** Investigate specific network towers and their performance over time.
- **💡 Customer Impact & Action:** Identify customers affected by anomalies and predict their churn risk.
- **🔬 About the Project:** Learn more about the technology stack and methodology.
""")