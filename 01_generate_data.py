# 01_generate_data.py
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# --- Configuration ---
NUM_CUSTOMERS = 5000
NUM_TOWERS = 50
START_DATE = datetime(2023, 10, 1)
END_DATE = datetime(2023, 10, 31)
UNHEALTHY_TOWER_PERCENTAGE = 0.15 # 15% of towers will be chronically unhealthy

# --- Indian Geography Configuration ---
indian_cities = {
    'Mumbai': {'lat': (19.00, 19.15), 'lon': (72.80, 72.95)},
    'Delhi': {'lat': (28.50, 28.70), 'lon': (77.10, 77.30)},
    'Bengaluru': {'lat': (12.90, 13.05), 'lon': (77.55, 77.65)},
    'Chennai': {'lat': (13.00, 13.10), 'lon': (80.20, 80.30)}
}
fake = Faker('en_IN')

def generate_all_data():
    """Main function to generate and save all datasets."""
    print("--- Starting Data Generation ---")
    
    # --- 1. Create Directories ---
    if not os.path.exists('data'):
        os.makedirs('data')

    # --- 2. Generate Cell Tower Data ---
    print("Generating Cell Tower Data for Indian Cities...")
    tower_data = []
    for i in range(NUM_TOWERS):
        city_name = random.choice(list(indian_cities.keys()))
        city_coords = indian_cities[city_name]
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
    print(f"-> Saved cell_towers.csv. {len(unhealthy_towers_list)} towers are chronically Unhealthy.")

    # --- 3. Generate Customer Profiles ---
    print("Generating Customer Profiles with Indian Names...")
    customers = {
        'customer_id': [f'C{20000+i}' for i in range(NUM_CUSTOMERS)],
        'name': [fake.name() for _ in range(NUM_CUSTOMERS)],
        'plan': [random.choice(['Basic_5GB', 'Standard_20GB', 'Premium_100GB', 'Unlimited']) for _ in range(NUM_CUSTOMERS)],
        'monthly_spend': np.random.uniform(200, 8000, size=NUM_CUSTOMERS).round(2),
        'tenure_months': np.random.randint(1, 72, size=NUM_CUSTOMERS),
        'home_tower_id': [random.choice(towers_df['tower_id']) for _ in range(NUM_CUSTOMERS)]
    }
    customers_df = pd.DataFrame(customers)
    customers_df.to_csv('data/customers.csv', index=False)
    print("-> Saved customers.csv")

    # --- 4. Generate Network Logs ---
    print("Generating Network Logs...")
    time_series = pd.date_range(start=START_DATE, end=END_DATE, freq='H')
    logs = []
    anomalous_event_towers = random.sample(towers_df['tower_id'].tolist(), 5)
    print(f"Injecting sharp anomaly events into towers: {anomalous_event_towers}")
    
    for _, tower_row in towers_df.iterrows():
        tower_id = tower_row['tower_id']
        is_unhealthy = tower_row['health_status'] == 'Unhealthy'
        for ts in time_series:
            base_latency = 50 + 20 * np.sin(2 * np.pi * ts.hour / 24)
            base_packet_drop = 0.5 + 0.4 * np.sin(2 * np.pi * ts.hour / 24)
            base_bandwidth = 300 - 150 * np.sin(2 * np.pi * ts.hour / 24)

            if is_unhealthy:
                base_latency *= 1.8; base_packet_drop *= 2.5; base_bandwidth *= 0.7

            latency = base_latency + np.random.normal(0, 5)
            packet_drop = base_packet_drop + np.random.normal(0, 0.1)
            bandwidth = base_bandwidth + np.random.normal(0, 10)
            
            is_anomaly_event = 0
            if tower_id in anomalous_event_towers and ts.day in [10, 20] and 2 <= ts.hour <= 5:
                latency *= random.uniform(2, 3); packet_drop *= random.uniform(4, 6); bandwidth /= random.uniform(2, 3)
                is_anomaly_event = 1

            logs.append({
                'timestamp': ts, 'tower_id': tower_id, 'latency_ms': max(10, latency),
                'packet_drop_rate': max(0, min(100, packet_drop)), 'total_bandwidth_mbps': max(10, bandwidth),
                'is_anomaly_ground_truth': is_anomaly_event
            })
    network_logs_df = pd.DataFrame(logs)
    network_logs_df.to_csv('data/network_logs.csv', index=False)
    print("-> Saved network_logs.csv")
    
    print("\n--- Data Generation Complete! ---")

if __name__ == "__main__":
    generate_all_data()