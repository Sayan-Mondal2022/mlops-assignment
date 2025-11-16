import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="California Housing Price Prediction", layout="centered")

st.title("🏠 California Housing Price Prediction")
st.write("Select a model and enter the house features to get predictions.")

# ----------------------------
# Load models dynamically
# ----------------------------
model_paths = {
    "Linear Regression": "saved_models/linear_regression.pkl",
    "Random Forest Regression": "saved_models/random_forest_regressor.pkl",
    "SVM Regression": "saved_models/svm_regressor.pkl"
}

model_choice = st.selectbox("Select a Prediction Model", list(model_paths.keys()))
model = joblib.load(model_paths[model_choice])

# ----------------------------
# Input Fields
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in 10k USD)", min_value=0.0, step=0.1)
    house_age = st.number_input("House Age", min_value=0.0, step=1.0)
    ave_rooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)
    ave_bedrooms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)

with col2:
    population = st.number_input("Population", min_value=0.0, step=1.0)
    ave_occupancy = st.number_input("Average Occupancy", min_value=0.0, step=0.1)
    latitude = st.number_input("Latitude", step=0.01)
    longitude = st.number_input("Longitude", step=0.01)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    input_data = np.array([
        [med_inc, house_age, ave_rooms, ave_bedrooms,
         population, ave_occupancy, latitude, longitude]
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"🏡 **Predicted Median House Value:** ${prediction * 100000:.2f}")
    st.info(f"Using Model: **{model_choice}**")
