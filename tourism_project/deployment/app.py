import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model pipeline committed to the repo by the MLOps pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_v1.joblib")
model = joblib.load(MODEL_PATH)

st.title("Visit with Us — Wellness Tourism Package Predictor")
st.write(
    "This app predicts whether a customer is likely to purchase the new "
    "Wellness Tourism Package, based on customer profile and past sales-pitch "
    "interaction data."
)

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 65, 35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    NumberOfTrips = st.slider("Average Number of Trips per Year", 0, 10, 3)
    Passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting (below 5 yrs)", 0, 3, 0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=22000, step=500)

st.header("Sales Pitch Interaction")
col3, col4 = st.columns(2)
with col3:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
with col4:
    NumberOfFollowups = st.slider("Number of Follow-ups", 0, 10, 4)
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package (confidence: {probability:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (confidence: {1 - probability:.1%})")
