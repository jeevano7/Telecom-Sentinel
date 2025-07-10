# pages/3_💡_Customer_Impact_&_Action.py
import streamlit as st
from utils import load_data, load_models
import pandas as pd

st.set_page_config(page_title="Customer Impact Analysis", layout="wide")
st.title("💡 Customer Impact & Action Center")

# --- Load Data and Models ---
customers_df, towers_df, logs_df, _ = load_data()
anomaly_detector, churn_predictor = load_models()

# --- Redefine "Problematic Towers" ---
# A tower is a problem if it's chronically Unhealthy OR has an active anomaly event.

# 1. Get the latest status for all towers
latest_logs = logs_df.loc[logs_df.groupby('tower_id')['timestamp'].idxmax()]
features = ['latency_ms', 'packet_drop_rate', 'total_bandwidth_mbps']
latest_logs['is_anomaly_event'] = anomaly_detector.predict(latest_logs[features]) == -1
status_df = pd.merge(towers_df, latest_logs, on='tower_id')

# 2. Identify all problematic towers
active_anomaly_towers = status_df[status_df['is_anomaly_event'] == True]
chronically_unhealthy_towers = status_df[status_df['health_status'] == 'Unhealthy']

# Combine them into a single list of towers needing attention
problem_tower_ids = sorted(list(set(active_anomaly_towers['tower_id'].tolist() + chronically_unhealthy_towers['tower_id'].tolist())))

# --- Main Page Logic ---
if not problem_tower_ids:
    st.success("✅ All systems are operating within normal parameters. No chronic issues or active anomalies detected.")
else:
    st.warning(f"🚨 {len(problem_tower_ids)} towers require attention.")
    
    # --- Create a more informative selection box ---
    def format_tower_for_selectbox(tower_id):
        tower_info = status_df[status_df['tower_id'] == tower_id].iloc[0]
        status_tags = []
        if tower_info['health_status'] == 'Unhealthy':
            status_tags.append("Chronic Unhealthy")
        if tower_info['is_anomaly_event']:
            status_tags.append("Active Anomaly")
        return f"{tower_id} ({tower_info['city']}) - [{', '.join(status_tags)}]"

    # --- Select a Tower to Analyze ---
    selected_tower_formatted = st.selectbox(
        "Select a tower to analyze customer impact:",
        options=problem_tower_ids,
        format_func=format_tower_for_selectbox
    )
    
    # Extract the actual ID from the formatted string
    selected_tower_id = selected_tower_formatted.split(' ')[0]

    if selected_tower_id:
        # --- Get selected tower's full info ---
        tower_info = status_df[status_df['tower_id'] == selected_tower_id].iloc[0]
        
        st.subheader(f"Impact Analysis for Tower {selected_tower_id}")

        # --- Find customers connected to this tower ---
        impacted_customers = customers_df[customers_df['home_tower_id'] == selected_tower_id].copy()
        
        if impacted_customers.empty:
            st.info("No customers are registered with this tower as their home tower.")
        else:
            # --- Churn Prediction (with the CORRECT features) ---
            # Prepare the DataFrame for prediction
            X_predict = impacted_customers[['plan', 'monthly_spend', 'tenure_months']].copy()
            
            # Feature 1: Did they experience a recent sharp anomaly?
            X_predict['anomaly_experienced_count'] = 1 if tower_info['is_anomaly_event'] else 0
            
            # Feature 2: Are they on a chronically unhealthy tower? (THE FIX)
            X_predict['is_on_unhealthy_tower'] = 1 if tower_info['health_status'] == 'Unhealthy' else 0
            
            # Predict churn probabilities
            churn_probabilities = churn_predictor.predict_proba(X_predict)[:, 1]
            impacted_customers['churn_probability'] = churn_probabilities

            # Sort by highest churn risk
            impacted_customers = impacted_customers.sort_values('churn_probability', ascending=False)
            
            # --- Display Results ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers Impacted", len(impacted_customers))
                avg_churn_risk = impacted_customers['churn_probability'].mean() * 100
                st.metric("Average Churn Risk", f"{avg_churn_risk:.2f}%", help="Predicted probability of churning in the next 30 days.")

            with col2:
                st.subheader("Customer Plan Distribution")
                plan_dist = impacted_customers['plan'].value_counts()
                st.bar_chart(plan_dist)

            st.markdown("---")
            st.subheader("High-Risk Customer List")
            st.dataframe(
                impacted_customers[['customer_id', 'name', 'plan', 'tenure_months', 'churn_probability']]
                .style.background_gradient(cmap='Reds', subset=['churn_probability'])
                .format({'churn_probability': '{:.2%}'})
            )

            # --- Recommended Actions ---
            st.subheader("Actionable Retention Strategies")
            st.info("Based on the predicted churn risk, here are some automated recommendations:")

            if not impacted_customers.empty:
                high_risk_customer = impacted_customers.iloc[0]
                
                st.success("**For High-Value, High-Risk Customers:**")
                st.write(f"Consider a proactive outreach for customer **{high_risk_customer['name']} ({high_risk_customer['customer_id']})**.")
                st.text_area(
                    "Suggested SMS/Email Template:",
                    f"Hi {high_risk_customer['name'].split()[0]}, we noticed you may have experienced a service disruption in your area. We've fixed it and, as a valued customer, we've added a complimentary data pack to your account. We appreciate your business.",
                    height=150
                )

                st.warning("**For Moderate-Risk Customers:**")
                st.write("Send a bulk, less personalized message acknowledging the issue and offering a small compensation.")
                st.text_area(
                    "Bulk SMS Template:",
                    "Telecom Sentinel Alert: We recently resolved a network issue in your area. We apologize for any inconvenience. Your service quality is our priority.",
                    height=100
                )