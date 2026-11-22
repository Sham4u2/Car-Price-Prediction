"""Car Price Prediction with XGBoost — Streamlit app (fixed version).

What was fixed vs the old app.py:
  - Removed the "Price" input. The old app asked the user to enter the
    price in order to predict the price. That makes no sense, and it was
    used to build `price_per_km = price / kilometerdriven`, which leaked
    the answer into the model.
  - Removed `price_per_km` completely.
  - Added `kilometerdriven` as a real input feature, matching the
    retrained model.
  - The model is now loaded from the same folder as this file
    (`Car_Price_Prediction.pkl`), not from a hard-coded Windows path
    like `D:\Data Science\Project\...`, so it works on any computer.

To run:
  1. Run the notebook (or the retrain script) once so that
     `Car_Price_Prediction.pkl` and the `encoders/` folder exist
     in this same folder.
  2. Run:  streamlit run app.py
"""
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Car_Price_Prediction.pkl")
ENCODERS_DIR = os.path.join(BASE_DIR, "encoders")

# List of categorical columns (same as training)
categorical_cols = [
    "name", "make", "model", "city", "fueltype",
    "transmission", "bodytype", "registrationcity", "registrationstate",
]

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found: Car_Price_Prediction.pkl. "
        "Run the notebook first to train and save the model in this folder."
    )
    st.stop()
model = joblib.load(MODEL_PATH)

# Load encoders (classes_)
encoders = {}
for col in categorical_cols:
    classes_path = os.path.join(ENCODERS_DIR, f"{col}_classes.pkl")
    if not os.path.exists(classes_path):
        st.error(f"Encoder file missing for {col}: {classes_path}")
        st.stop()
    encoders[col] = joblib.load(classes_path)

st.title("Car Price Prediction with XGBoost")

# User inputs — no Price input anymore
st.header("Enter Car Details")
user_data = {}
for col in categorical_cols:
    user_data[col] = st.selectbox(f"Select {col.capitalize()}", encoders[col])

user_data["year"] = st.number_input(
    "Manufacturing Year", min_value=1980, max_value=2026, value=2020
)
user_data["kilometerdriven"] = st.number_input(
    "Kilometers Driven", min_value=0, value=50000, step=1000
)
user_data["ownernumber"] = st.number_input(
    "Owner Number", min_value=1, max_value=10, value=1
)

# Convert to DataFrame
df_input = pd.DataFrame([user_data])

# Feature engineering (same as training — no price_per_km)
current_year = datetime.now().year
df_input["car_age"] = current_year - df_input["year"]
df_input["is_1st_owner"] = (df_input["ownernumber"] == 1).astype(int)
df_input["is_automatic"] = (df_input["transmission"] == "Automatic").astype(int)

# Keep only the 13 features the retrained model was trained on,
# in the same order as training
df_input = df_input[
    [
        "name", "make", "model", "city", "fueltype", "transmission",
        "bodytype", "registrationcity", "registrationstate",
        "car_age", "kilometerdriven", "is_1st_owner", "is_automatic",
    ]
]

# Label encoding using saved classes
for col in categorical_cols:
    le = LabelEncoder()
    le.classes_ = encoders[col]
    df_input[col] = le.transform(df_input[col].astype(str))

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(df_input)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
