import streamlit as st
import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('models/logistic_model.pkl')
    features = joblib.load('models/top_features.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Divorce Prediction - Sample Test Run")
st.write("Please answer the following questions on a scale from 0 to 4:")

input_data = {}
for feature in features:
    input_data[feature] = st.slider(f"Question: {feature}", 0, 4, 2)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("Prediction: High Risk of Divorce")
    else:
        st.success("Prediction: Stable Marriage")
