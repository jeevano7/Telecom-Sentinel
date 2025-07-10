# pages/1_🌍_Global_Network_Overview.py
import streamlit as st
from utils import load_data, load_models
import plotly.express as px
from folium import Map, Marker, Icon
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(page_title="Global Network Overview", layout="wide")
st.title("🌍 Global Network Overview")
st.markdown("Differentiating between chronic poor service (**Unhealthy Towers**) and acute, temporary events (**Anomalies**).")

# --- Load Data and Models ---
customers_df, towers_df, logs_df, churn_df = load_data()
anomaly_detector, _ = load_models()

# --- Prepare Data for Dashboard ---
latest_logs = logs_df.loc[logs_df.groupby('tower_id')['timestamp'].idxmax()]
features = ['latency_ms', 'packet_drop_rate', 'total_bandwidth_mbps']
latest_logs['is_anomaly_now'] = anomaly_detector.predict(latest_logs[features]) == -1
status_df = pd.merge(towers_df, latest_logs, on='tower_id')

# --- NEW: Define status logic ---
# Anomaly (Red): Acute, high-impact event detected right now.
# Unhealthy (Orange): Chronic poor performance, but no sharp anomaly right now.
# Healthy (Green): Good performance.
def get_status(row):
    if row['is_anomaly_now']:
        return 'Anomaly'
    elif row['health_status'] == 'Unhealthy':
        return 'Unhealthy'
    else:
        return 'Healthy'

status_df['status'] = status_df.apply(get_status, axis=1)

# --- KPIs ---
total_towers = len(towers_df)
num_unhealthy = (status_df['status'] == 'Unhealthy').sum()
num_anomalies = (status_df['status'] == 'Anomaly').sum()
healthy_towers = total_towers - num_unhealthy - num_anomalies

# Customers impacted by UNHEALTHY towers
unhealthy_tower_ids = status_df[status_df['status'] == 'Unhealthy']['tower_id'].tolist()
customers_on_unhealthy = customers_df[customers_df['home_tower_id'].isin(unhealthy_tower_ids)]

# Customers impacted by ACTIVE ANOMALY towers
anomaly_tower_ids = status_df[status_df['status'] == 'Anomaly']['tower_id'].tolist()
customers_in_anomaly = customers_df[customers_df['home_tower_id'].isin(anomaly_tower_ids)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Towers", f"{total_towers}")
col2.metric("Healthy Towers", f"{healthy_towers}")
col3.metric("Chronically Unhealthy", f"{num_unhealthy}",
             help=f"{len(customers_on_unhealthy)} customers are on these towers with consistent poor service.")
col4.metric("Active Anomalies Now", f"{num_anomalies}",
             help=f"{len(customers_in_anomaly)} customers are affected by a sharp, current event.")

st.markdown("---")

# --- Map and Chart ---
c1, c2 = st.columns([0.4, 0.6])

with c1:
    st.subheader("Live Tower Status Map")
    map_center = [status_df['latitude'].mean(), status_df['longitude'].mean()]
    m = Map(location=map_center, zoom_start=5) # Zoom out to see all of India

    # --- NEW: Updated Icons and Popups ---
    status_colors = {'Healthy': 'green', 'Unhealthy': 'orange', 'Anomaly': 'red'}
    status_icons = {'Healthy': 'signal', 'Unhealthy': 'exclamation-triangle', 'Anomaly': 'bolt'}

    for idx, row in status_df.iterrows():
        popup_text = f"""
        <b>Tower ID:</b> {row['tower_id']} ({row['city']})<br>
        <b>Status:</b> <font color='{status_colors[row['status']]}'><b>{row['status']}</b></font><br>
        <b>Chronic State:</b> {row['health_status']}<br>
        <hr>
        <b>Latency:</b> {row['latency_ms']:.2f} ms<br>
        <b>Packet Drop:</b> {row['packet_drop_rate']:.2f}%
        """
        Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            tooltip=f"{row['tower_id']}: {row['status']}",
            icon=Icon(color=status_colors[row['status']], icon=status_icons[row['status']], prefix='fa')
        ).add_to(m)
    st_folium(m, use_container_width=True)


with c2:
    st.subheader("Network-Wide Average Latency (Last 24 Hours)")
    last_24h = logs_df['timestamp'].max() - pd.Timedelta(hours=24)
    recent_logs = logs_df[logs_df['timestamp'] >= last_24h].copy()
    
    # Add status to recent logs for color coding the chart
    recent_logs_with_status = pd.merge(recent_logs, towers_df[['tower_id', 'health_status']], on='tower_id')
    
    hourly_avg_latency = recent_logs_with_status.groupby([pd.Grouper(key='timestamp', freq='H'), 'health_status'])['latency_ms'].mean().reset_index()

    fig = px.line(hourly_avg_latency, x='timestamp', y='latency_ms', color='health_status',
                  title='Avg. Latency: Healthy vs. Unhealthy Towers',
                  labels={'latency_ms': 'Latency (ms)', 'timestamp': 'Time', 'health_status': 'Tower Type'},
                  color_discrete_map={'Healthy': 'green', 'Unhealthy': 'orange'})
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text='Tower Type')
    st.plotly_chart(fig, use_container_width=True)