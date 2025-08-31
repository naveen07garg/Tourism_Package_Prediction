
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# naveen07garg/Tourism-Package-Prediction

st.title("🛒 Visit With Us - Tour Prediction ")
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
input_data = {
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
}

if st.button("Predict", type='primary'):
    response = requests.post("https://naveen07garg-Tourism-Package-Prediction.hf.space/v1/sales", json=input_data)    # enter user name and space name before running the cell
    if response.status_code == 200:
        prediction = response.json()['Prediction sales']
        #churn_prediction = result["Prediction"]  # Extract only the value
        st.success(f"Predicted Sales (in dollars): {prediction}")
    else:
        st.error("Error in API request")
        st.write(response.text)
        st.error("Response status code")
        st.error("==========================================")
        st.write(response.status_code)
        st.write(response.headers)
        st.write(response.request.body)
        st.write(response.request.headers)
        st.write(response.request.url)
        st.write(response.request.method)
        st.write(response.request.path_url)


# Batch Prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
file = st.file_uploader("Upload CSV file", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if file is not None:
    if st.button("Predict Batch", type='primary'):
        response = requests.post("https://naveen07garg-Tourism-Package-Prediction.hf.space/v1/salesbatch", files={"file": file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.header("Batch Prediction Results")
            st.write(predictions)
        else:
            st.error("Error in API request")
            st.write(response.text)
            st.error("Response status code")
            st.error("==========================================")
            st.write(response.status_code)
            st.write(response.headers)
            st.write(response.request.body)
            st.write(response.request.headers)
            st.write(response.request.url)
            st.write(response.request.method)
            st.write(response.request.path_url)


    # Call backend API
#    response = requests.post("https://<your-hf-backend-space-url>/predict", json=input_data)

#    if response.status_code == 200:
#        prediction = response.json()["predicted_sales"]
#        st.success(f"Predicted Sales: {round(prediction, 2)}")
#    else:
#        st.error(f"Error: {response.text}")
