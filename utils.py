# utils.py
import streamlit as st
import pandas as pd
import joblib

# Use Streamlit's caching to load data only once
@st.cache_data
def load_data():
    customers = pd.read_csv('data/customers.csv')
    towers = pd.read_csv('data/cell_towers.csv')
    logs = pd.read_csv('data/network_logs.csv', parse_dates=['timestamp'])
    churn_data = pd.read_csv('data/customer_churn_data.csv')
    return customers, towers, logs, churn_data

# Use Streamlit's resource caching for models
@st.cache_resource
def load_models():
    anomaly_detector = joblib.load('models/anomaly_detector.pkl')
    churn_predictor = joblib.load('models/churn_predictor.pkl')
    return anomaly_detector, churn_predictor

