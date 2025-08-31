
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#=== Download and load the model ===
model_path = hf_hub_download(repo_id="naveen07garg/Tourism-Package-Prediction", filename="sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

# naveen07garg/Tourism-Package-Prediction

st.title("Visit With Us - Tour Prediction ")
st.write("Fill the details to predict:")

# Input form

# Categorical (selectbox)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# Numeric (number_input)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=1)
num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=1)
num_trips = st.number_input("Number of Trips", min_value=1, max_value=22, value=1)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)

# Binary (selectbox)
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])


#=== Making a dictionary ===
input_data = pd.DataFrame([{
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation
}])

if st.button("Predict Sales Status"):
    prediction = model.predict(input_data)[0]
    result = "Product not Sold ⚡" if prediction == 1 else "Product Sold 😄"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
