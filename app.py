import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('loan_model.pkl')

# Page Title
st.title("Loan Default Prediction App")

st.write("Enter borrower information to predict loan default risk.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income (USD)", min_value=0, step=100, value=3000)
loan_amount = st.number_input("Loan Amount", min_value=0, step=500, value=10000)
loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"])

# Encode employment status
employment_map = {"Employed": 0, "Self-employed": 1, "Unemployed": 2}
employment_status_encoded = employment_map[employment_status]

# Prediction button
if st.button("Predict Default Risk"):
    input_data = np.array([[age, income, loan_amount, loan_term, credit_score, employment_status_encoded]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Prediction: HIGH Risk of Default ({probability:.2%} probability)")
    else:
        st.success(f"Prediction: LOW Risk of Default ({probability:.2%} probability)")