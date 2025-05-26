import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
import pickle

# Load Models & Scaler
delay_model = pickle.load(open("delay_model.pkl", "rb"))
cost_model = pickle.load(open("cost_model.pkl", "rb"))
carrier_model = pickle.load(open("carrier_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit App
st.title("Shipping AI Pipeline")

# User Inputs
st.subheader("Enter Shipment Details")
mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
cost_of_product = st.number_input("Cost of the Product", min_value=0.0, step=0.1)
discount_offered = st.number_input("Discount Offered", min_value=0.0, step=0.1)
weight_in_gms = st.number_input("Weight (grams)", min_value=0.0, step=1.0)
customer_rating = st.slider("Customer Rating", 1, 5, 3)
prior_purchases = st.number_input("Prior Purchases", min_value=0, step=1)
warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "E"])
product_importance = st.selectbox("Product Importance", ["Low", "Medium", "High"])

# Encoding categorical features
mode_map = {"Flight": 0, "Ship": 1, "Road": 2}
warehouse_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
importance_map = {"Low": 0, "Medium": 1, "High": 2}

mode_of_shipment = mode_map[mode_of_shipment]
warehouse_block = warehouse_map[warehouse_block]
product_importance = importance_map[product_importance]

# Prepare input for model
input_data = np.array([[mode_of_shipment, cost_of_product, discount_offered, weight_in_gms, 
                        customer_rating, prior_purchases, warehouse_block, product_importance]])
input_scaled = scaler.transform(input_data)

# Predictions
if st.button("Predict Shipping Details"):
    shipping_cost = cost_model.predict(input_scaled)[0]
    delay_probability = delay_model.predict_proba(input_scaled)[:, 1][0]
    carrier_prediction = carrier_model.predict(input_scaled)[0]
    
    carrier_map = {0: "DHL", 1: "FedEx", 2: "Maersk", 3: "Local Truck"}
    recommended_carrier = carrier_map[carrier_prediction]
    
    st.subheader("Prediction Results")
    st.write(f"Estimated Shipping Cost: **${shipping_cost:.2f}**")
    st.write(f"On-Time Delivery Probability: **{delay_probability:.2%}**")
    st.write(f"Recommended Carrier: **{recommended_carrier}**")
