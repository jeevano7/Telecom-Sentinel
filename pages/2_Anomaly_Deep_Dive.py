# pages/2_Anomaly_Deep_Dive.py
import streamlit as st
from utils import load_data, load_models
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Anomaly Deep Dive", layout="wide")
st.title("Anomaly Deep Dive")

# --- Load Data and Models ---
_, towers_df, logs_df, _ = load_data()
anomaly_detector, _ = load_models()

# --- Tower Selection ---
tower_id_list = sorted(towers_df['tower_id'].unique())
selected_tower = st.selectbox("Select a Cell Tower to Investigate", tower_id_list)

if selected_tower:
    # --- Get Tower's Chronic Health Status ---
    tower_info = towers_df.loc[towers_df['tower_id'] == selected_tower].iloc[0]
    chronic_status = tower_info['health_status']
    
    st.header(f"Tower {selected_tower} | Chronic Status: {chronic_status}")
    if chronic_status == 'Unhealthy':
        st.warning("This tower is designated as chronically **Unhealthy**, meaning it consistently provides sub-par service.")
    else:
        st.success("This tower is designated as **Healthy**, with good baseline performance.")


    # --- Filter Data for Selected Tower ---
    tower_logs = logs_df[logs_df['tower_id'] == selected_tower].copy()
    tower_logs.sort_values('timestamp', inplace=True)

    # --- Predict Anomalies for the Entire History of the Tower ---
    features = ['latency_ms', 'packet_drop_rate', 'total_bandwidth_mbps']
    tower_logs['is_anomaly_event'] = anomaly_detector.predict(tower_logs[features]) == -1
    
    st.subheader(f"Performance Metrics Over Time")

    # --- Create Interactive Plots ---
    fig = px.line(tower_logs, x='timestamp', y=features, 
                  title=f'Time Series Metrics for {selected_tower}',
                  labels={'value': 'Metric Value', 'timestamp': 'Time'}, height=500)
    fig.update_layout(legend_title_text='Metrics')

    # Add shaded regions for sharp anomaly events
    anomaly_event_periods = tower_logs[tower_logs['is_anomaly_event']]
    for i in range(len(anomaly_event_periods)):
        fig.add_vrect(
            x0=anomaly_event_periods['timestamp'].iloc[i],
            x1=anomaly_event_periods['timestamp'].iloc[i] + pd.Timedelta(hours=1), # Shade the full hour
            fillcolor="red", opacity=0.3, line_width=0,
            annotation_text="Anomaly Event", annotation_position="top left"
        )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Display Problematic Log Details ---
    st.subheader("Problematic Service Logs")

    # A log is problematic if it's an anomaly OR if the tower is unhealthy
    if chronic_status == 'Unhealthy':
        problematic_logs = tower_logs[(tower_logs['is_anomaly_event']) | (towers_df.loc[towers_df['tower_id'] == selected_tower, 'health_status'].iloc[0] == 'Unhealthy')]
        # For unhealthy towers, we add a status column to explain WHY a log is shown
        problematic_logs['reason'] = problematic_logs['is_anomaly_event'].apply(
            lambda x: 'Anomaly Event' if x else 'Chronic Poor Service'
        )
        display_cols = ['timestamp', 'tower_id'] + features + ['reason']
        
    else: # If the tower is healthy, only show true anomaly events
        problematic_logs = anomaly_event_periods
        display_cols = ['timestamp', 'tower_id'] + features
        
    if not problematic_logs.empty:
        # Custom styling for the dataframe
        def style_rows(row):
            if 'reason' in row and row['reason'] == 'Anomaly Event':
                return ['background-color: #8B0000; color: white'] * len(row) # Dark red for events
            elif 'reason' in row and row['reason'] == 'Chronic Poor Service':
                 return ['background-color: #FF8C00; color: white'] * len(row) # Orange for chronic
            return [''] * len(row)
            
        st.dataframe(
            problematic_logs[display_cols].style.apply(style_rows, axis=1)
        )
    else:
        st.success("No sharp anomaly events were detected for this healthy tower during the selected period.")
