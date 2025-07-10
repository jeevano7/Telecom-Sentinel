# data_generator.py (Version 2 - Indian Geography & Unhealthy Towers)
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os
import joblib

# --- Configuration ---
NUM_CUSTOMERS = 5000
NUM_TOWERS = 50
START_DATE = datetime(2023, 10, 1)
END_DATE = datetime(2023, 10, 31)
UNHEALTHY_TOWER_PERCENTAGE = 0.50 # 15% of towers will be chronically unhealthy

# --- Indian Geography Configuration ---
indian_cities = {
    'Mumbai': {'lat': (19.00, 19.15), 'lon': (72.80, 72.95)},
    'Delhi': {'lat': (28.50, 28.70), 'lon': (77.10, 77.30)},
    'Bengaluru': {'lat': (12.90, 13.05), 'lon': (77.55, 77.65)},
    'Chennai': {'lat': (13.00, 13.10), 'lon': (80.20, 80.30)}
}

# Initialize Faker for Indian names
fake = Faker('en_IN')

# --- 1. Create Directories ---
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('models'):
    os.makedirs('models')

# --- 2. Generate Cell Tower Data with Health Status ---
print("Generating Cell Tower Data for Indian Cities...")
tower_data = []
for i in range(NUM_TOWERS):
    city_name = random.choice(list(indian_cities.keys()))
    city_coords = indian_cities[city_name]
    
    # NEW: Assign a persistent health status
    health_status = 'Unhealthy' if random.random() < UNHEALTHY_TOWER_PERCENTAGE else 'Healthy'
    
    tower_data.append({
        'tower_id': f'T{1000+i}',
        'latitude': np.random.uniform(city_coords['lat'][0], city_coords['lat'][1]),
        'longitude': np.random.uniform(city_coords['lon'][0], city_coords['lon'][1]),
        'city': city_name,
        'health_status': health_status
    })

towers_df = pd.DataFrame(tower_data)
towers_df.to_csv('data/cell_towers.csv', index=False)
unhealthy_towers_list = towers_df[towers_df['health_status'] == 'Unhealthy']['tower_id'].tolist()
print(f"-> Saved cell_towers.csv")
print(f"-> Designated {len(unhealthy_towers_list)} towers as chronically Unhealthy: {unhealthy_towers_list}")


# --- 3. Generate Customer Profiles ---
print("Generating Customer Profiles with Indian Names...")
plans = ['Basic_5GB', 'Standard_20GB', 'Premium_100GB', 'Unlimited']
customers = {
    'customer_id': [f'C{20000+i}' for i in range(NUM_CUSTOMERS)],
    'name': [fake.name() for _ in range(NUM_CUSTOMERS)],
    'plan': [random.choice(plans) for _ in range(NUM_CUSTOMERS)],
    'monthly_spend': np.random.uniform(200, 8000, size=NUM_CUSTOMERS).round(2), # Adjusted for INR
    'tenure_months': np.random.randint(1, 72, size=NUM_CUSTOMERS),
    'home_tower_id': [random.choice(towers_df['tower_id']) for _ in range(NUM_CUSTOMERS)]
}
customers_df = pd.DataFrame(customers)
customers_df.to_csv('data/customers.csv', index=False)
print("-> Saved customers.csv")


# --- 4. Generate Network Logs with Anomalies AND Unhealthy Behavior ---
print("Generating Network Logs (this may take a moment)...")
time_series = pd.date_range(start=START_DATE, end=END_DATE, freq='H')
logs = []

# Select towers for sharp, temporary anomalies (can include unhealthy ones)
anomalous_event_towers = random.sample(towers_df['tower_id'].tolist(), 5)
print(f"Injecting sharp anomaly events into towers: {anomalous_event_towers}")

for _, tower_row in towers_df.iterrows():
    tower_id = tower_row['tower_id']
    is_unhealthy = tower_row['health_status'] == 'Unhealthy'

    for ts in time_series:
        # Normal behavior simulation
        base_latency = 50 + 20 * np.sin(2 * np.pi * ts.hour / 24)
        base_packet_drop = 0.5 + 0.4 * np.sin(2 * np.pi * ts.hour / 24)
        base_bandwidth = 300 - 150 * np.sin(2 * np.pi * ts.hour / 24)

        # NEW: Apply persistent degradation for unhealthy towers
        if is_unhealthy:
            base_latency *= 1.8  # 80% higher baseline latency
            base_packet_drop *= 2.5 # 150% higher baseline packet drop
            base_bandwidth *= 0.7 # 30% lower baseline bandwidth

        latency = base_latency + np.random.normal(0, 5)
        packet_drop = base_packet_drop + np.random.normal(0, 0.1)
        bandwidth = base_bandwidth + np.random.normal(0, 10)
        
        # Inject sharp, temporary anomalies
        is_anomaly_event = 0
        if tower_id in anomalous_event_towers and ts.day in [10, 20] and 2 <= ts.hour <= 5:
            latency *= random.uniform(2, 3) # Latency spikes
            packet_drop *= random.uniform(4, 6) # Packet drop increases
            bandwidth /= random.uniform(2, 3) # Bandwidth drops
            is_anomaly_event = 1

        logs.append({
            'timestamp': ts,
            'tower_id': tower_id,
            'latency_ms': max(10, latency),
            'packet_drop_rate': max(0, min(100, packet_drop)),
            'total_bandwidth_mbps': max(10, bandwidth),
            'is_anomaly_ground_truth': is_anomaly_event
        })

network_logs_df = pd.DataFrame(logs)
network_logs_df.to_csv('data/network_logs.csv', index=False)
print("-> Saved network_logs.csv")


# --- 5. Train Anomaly Detection Model (Isolation Forest) ---
# This model will detect BOTH sharp anomalies and the chronic bad state of unhealthy towers
from sklearn.ensemble import IsolationForest
print("Training Anomaly Detection Model...")
features = ['latency_ms', 'packet_drop_rate', 'total_bandwidth_mbps']
X_anomaly = network_logs_df[features]

# We increase contamination to account for both sharp anomalies and the baseline unhealthy data points
model_iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model_iso_forest.fit(X_anomaly)
joblib.dump(model_iso_forest, 'models/anomaly_detector.pkl')
print("-> Saved anomaly_detector.pkl")


# --- 6. Prepare Churn Data & Train Churn Model (XGBoost) ---
print("Preparing Churn Data...")
# Merge customer and tower data to know who is on an unhealthy tower
customer_tower_info = pd.merge(customers_df, towers_df, left_on='home_tower_id', right_on='tower_id')

# Create features for churn model
# Feature 1: Did the customer experience a SHARP ANOMALY EVENT?
anomaly_logs = network_logs_df[network_logs_df['is_anomaly_ground_truth'] == 1]
affected_towers_by_event = anomaly_logs['tower_id'].unique()
customer_tower_info['anomaly_experienced_count'] = customer_tower_info['home_tower_id'].apply(
    lambda x: 1 if x in affected_towers_by_event else 0
)
# Feature 2: Is the customer's home tower CHRONICALLY UNHEALTHY?
customer_tower_info['is_on_unhealthy_tower'] = (customer_tower_info['health_status'] == 'Unhealthy').astype(int)

# Create a churn label - influenced by tenure, sharp anomalies, and chronic poor service
churn_probability = 1 / (1 + np.exp(-(
    -0.05 * customer_tower_info['tenure_months'] +      # Longer tenure = less likely to churn
    2.5 * customer_tower_info['anomaly_experienced_count'] + # Sharp anomaly = more likely
    1.5 * customer_tower_info['is_on_unhealthy_tower'] - # Chronic bad service = more likely
    2.0 # Base churn tendency
)))
customer_tower_info['churned'] = (np.random.rand(NUM_CUSTOMERS) < churn_probability).astype(int)

churn_data_df = customer_tower_info.copy()
churn_data_df.to_csv('data/customer_churn_data.csv', index=False)
print("-> Saved customer_churn_data.csv")

print("Training Churn Prediction Model...")
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define features (X) and target (y) - NOW INCLUDES 'is_on_unhealthy_tower'
X_churn = churn_data_df[['plan', 'monthly_spend', 'tenure_months', 'anomaly_experienced_count', 'is_on_unhealthy_tower']]
y_churn = churn_data_df['churned']

# Preprocessing pipeline
categorical_features = ['plan']
numeric_features = ['monthly_spend', 'tenure_months', 'anomaly_experienced_count', 'is_on_unhealthy_tower']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

churn_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

churn_model_pipeline.fit(X_churn, y_churn)
joblib.dump(churn_model_pipeline, 'models/churn_predictor.pkl')
print("-> Saved churn_predictor.pkl")
print("\n--- All Done! New Indian dataset and retrained models are ready. ---")