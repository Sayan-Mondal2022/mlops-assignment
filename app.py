import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="California Housing Price Prediction", layout="centered")

st.title("🏠 California Housing Price Prediction")
st.write("Select a model and enter the house features to get predictions.")

# Short instructions
st.subheader("📋 Quick Guide")

st.write("""
### 🔢 Input Guidelines:
- **MedInc**: Income in $10,000s (e.g., $50,000 → 5.0)
- **HouseAge**: House age in years (e.g., 25 years → 25)
- **AveRooms**: Avg rooms per home (e.g., 5 rooms → 5.0)
- **AveBedrms**: Avg bedrooms (e.g., 2.5 bedrooms → 2.5)
- **Population**: Area population (e.g., 1500 people → 1500)
- **AveOccup**: People per household (e.g., 3 people → 3.0)
- **Latitude**: 32.0 to 42.0 (e.g., 34.05)
- **Longitude**: -125.0 to -114.0 (e.g., -118.25)

**Models Available:** Linear Regression, Random Forest, SVM
""")

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


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Streamlit | California Housing Price Predictor"
    "</div>",
    unsafe_allow_html=True
)