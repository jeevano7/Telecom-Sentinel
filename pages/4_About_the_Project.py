# pages/4_About_the_Project.py
import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("About the Project & Methodology")

st.markdown("""
This 'Telecom Sentinel' project is a comprehensive demonstration designed to showcase a range of skills relevant to a product-based company in the telecommunications sector. It moves beyond a simple model to a full-fledged, interactive application that bridges the gap between technical network operations and business-focused product management.
""")

st.subheader("Project Architecture")
st.caption="High-Level System Architecture" # Example image, you can create your own

st.subheader("Key Components")
st.markdown("""
1.  **Synthetic Data Generation (`data_generator.py`):**
    - **Why it's important:** Demonstrates the ability to create realistic, complex datasets when real data isn't available. This is crucial for prototyping and testing.
    - **Tools:** `Pandas`, `NumPy`, `Faker`.
    - **What was generated:** Customer profiles, cell tower locations, and time-series network logs (latency, packet drop, bandwidth) with programmatically injected anomalies.

2.  **Unsupervised Anomaly Detection (The Network's 'Immune System'):**
    - **Model:** `scikit-learn`'s **Isolation Forest**.
    - **Why this model:** It's efficient and effective at identifying outliers in multidimensional data without needing pre-labeled examples of "bad" behavior. It learns what's "normal" and flags deviations.
    - **Features Used:** `latency_ms`, `packet_drop_rate`, `total_bandwidth_mbps`.

3.  **Supervised Churn Prediction (The Business 'Crystal Ball'):**
    - **Model:** **XGBoost (Extreme Gradient Boosting)**.
    - **Why this model:** It's a powerful, industry-standard algorithm known for its high performance and interpretability in classification tasks.
    - **The 'Secret Sauce' Feature:** The model's key feature is `anomaly_experienced_count`, which directly links network performance (from the first model) to a business outcome (customer churn). This demonstrates an understanding of feature engineering and how to connect different data domains.

4.  **Interactive Streamlit Dashboard (The 'Product'):**
    - **Purpose:** To serve as the user interface for different stakeholders (executives, network engineers, product managers).
    - **Libraries:** `Streamlit`, `Plotly` (for interactive charts), `Folium` (for maps).
    - **Key Feature:** The app is designed around user **actions** and **decisions**, not just passive monitoring.
""")

st.subheader("Skills Showcased")
st.markdown("""
- **End-to-End ML Project Lifecycle:** From data creation and preprocessing to model training, evaluation, and deployment in an interactive app.
- **Business Acumen:** Framing a technical problem (network anomalies) in the context of a critical business problem (customer churn and retention).
- **Data Engineering & Feature Engineering:** Creating a robust data pipeline and engineering features that provide real predictive power.
- **Machine Learning Expertise:** Applying both unsupervised (Isolation Forest) and supervised (XGBoost) learning techniques appropriately.
- **Software Engineering & App Development:** Writing modular, clean Python code and building a multi-page, user-friendly application with Streamlit.
""")

# st.subheader("Connect with Me")
# st.markdown("""
# **I am actively seeking opportunities in Data Science and Machine Learning where I can apply my skills to solve challenging business problems.**

# - **LinkedIn:** [Your LinkedIn Profile URL](https://www.linkedin.com/in/yourprofile/)
# - **GitHub:** [Your GitHub Profile URL](https://github.com/yourusername)
# - **Project Repository:** [Link to this project's repo on GitHub](https://github.com/yourusername/telecom_sentinel)
# """, unsafe_allow_html=True)
