# 02_train_models.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_all_models():
    """Main function to load data and train all models."""
    print("--- Starting Model Training ---")

    # --- 1. Load Datasets ---
    print("Loading datasets from 'data/' folder...")
    try:
        customers_df = pd.read_csv('data/customers.csv')
        towers_df = pd.read_csv('data/cell_towers.csv')
        network_logs_df = pd.read_csv('data/network_logs.csv')
    except FileNotFoundError:
        print("\nError: Data files not found.")
        print("Please run 'python 01_generate_data.py' first.")
        return

    # --- 2. Create Models Directory ---
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- 3. Train Anomaly Detection Model ---
    print("\nTraining Anomaly Detection Model (Isolation Forest)...")
    features_anomaly = ['latency_ms', 'packet_drop_rate', 'total_bandwidth_mbps']
    X_anomaly = network_logs_df[features_anomaly]
    model_iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model_iso_forest.fit(X_anomaly)
    joblib.dump(model_iso_forest, 'models/anomaly_detector.pkl')
    print("-> Saved anomaly_detector.pkl")

    # --- 4. Prepare Data for Churn Model ---
    print("\nPreparing data for Churn Prediction...")
    customer_tower_info = pd.merge(customers_df, towers_df, left_on='home_tower_id', right_on='tower_id')
    
    anomaly_logs = network_logs_df[network_logs_df['is_anomaly_ground_truth'] == 1]
    affected_towers_by_event = anomaly_logs['tower_id'].unique()
    
    customer_tower_info['anomaly_experienced_count'] = customer_tower_info['home_tower_id'].apply(
        lambda x: 1 if x in affected_towers_by_event else 0
    )
    customer_tower_info['is_on_unhealthy_tower'] = (customer_tower_info['health_status'] == 'Unhealthy').astype(int)

    # Create synthetic churn label
    churn_probability = 1 / (1 + np.exp(-(
        -0.05 * customer_tower_info['tenure_months'] +
        2.5 * customer_tower_info['anomaly_experienced_count'] +
        1.5 * customer_tower_info['is_on_unhealthy_tower'] - 2.0
    )))
    customer_tower_info['churned'] = (np.random.rand(len(customer_tower_info)) < churn_probability).astype(int)
    
    # Save the processed data with churn labels for inspection
    customer_tower_info.to_csv('data/customer_churn_data.csv', index=False)
    print("-> Saved feature-engineered customer_churn_data.csv for reference.")
    
    # --- 5. Train Churn Prediction Model ---
    print("\nTraining Churn Prediction Model (XGBoost)...")
    X_churn = customer_tower_info[['plan', 'monthly_spend', 'tenure_months', 'anomaly_experienced_count', 'is_on_unhealthy_tower']]
    y_churn = customer_tower_info['churned']

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
    
    print("\n--- Model Training Complete! ---")

if __name__ == "__main__":
    train_all_models()