import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter Customer Details to predict the segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (Sum of Purchase)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=4)
recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0, max_value=365, value=150) # Fix 1: Added missing )

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending], # Fix 2: Capital S to match training data
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    cluster_labels = {
        0: "High Budget, Store Visitor",
        1: "Lowest Budget, Web Purchaser",
        2: "Medium Budget, Store & Web Purchaser"
    }

    label = cluster_labels.get(cluster, f"Unknown Cluster {cluster}")

    st.success(f"Predicted Segment: Cluster {cluster}")
    st.info(f"Profile: {label}")