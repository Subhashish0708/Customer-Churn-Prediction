import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page config (Step 42 — improved UI)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------
model = joblib.load("customer_churn_model.pkl")

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("Customer Churn Predictor")
st.sidebar.write("Machine Learning Project")

# ---------------------------------------------------------------
# App title & description
# ---------------------------------------------------------------
st.title("Customer Churn Prediction")
st.write("Enter customer details below.")

st.markdown("""
This application predicts whether a customer is likely to **stay** or **leave** based on customer details.
""")

# ---------------------------------------------------------------
# User input fields
# ---------------------------------------------------------------
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Months)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)
internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic"]
)
support = st.number_input("Support Calls", 0, 20, 1)
total = st.number_input("Total Charges", 0.0, 10000.0, 600.0)

# ---------------------------------------------------------------
# Convert text to numbers
# These values must match the encoding used when the model was trained
# (see Step 9 in customer_churn_prediction.ipynb, where LabelEncoder
# alphabetically encodes Contract as {"Month-to-month":0, "One year":1,
# "Two year":2} and InternetService as {"DSL":0, "Fiber optic":1}).
# ---------------------------------------------------------------
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
internet_map = {
    "DSL": 0,
    "Fiber optic": 1
}

# ---------------------------------------------------------------
# Prediction button
# ---------------------------------------------------------------
if st.button("Predict"):
    data = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "MonthlyCharges": [monthly],
        "Contract": [contract_map[contract]],
        "InternetService": [internet_map[internet]],
        "SupportCalls": [support],
        "TotalCharges": [total]
    })

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")
