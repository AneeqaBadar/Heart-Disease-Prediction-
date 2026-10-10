import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# App Title & Description
st.title("🫀 Heart Disease Risk Prediction System")
st.write("""
This interactive web app uses a Support Vector Classifier (SVC) model trained on clinical attributes 
to assess the likelihood of heart disease presence.
""")

# Load the trained Support Vector Classifier model
@st.cache_resource
def load_model():
    with open('heart_disease_svc_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Form for clinical input parameters
st.subheader("Patient Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0: Typical Angina",
            1: "1: Atypical Angina",
            2: "2: Non-anginal Pain",
            3: "3: Asymptomatic"
        }[x],
        help="Select the type of chest pain experienced by the patient."
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved (thalach)",
        min_value=60,
        max_value=220,
        value=150,
        step=1,
        help="Peak heart rate achieved during exercise testing (BPM)."
    )

with col2:
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.2,
        step=0.1,
        help="ST depression induced by exercise relative to rest."
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3, 4],
        help="Number of major blood vessels (0-4) colored by fluoroscopy."
    )

# Create input DataFrame matching exact feature schema
input_data = pd.DataFrame({
    'cp': [cp],
    'thalach': [thalach],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

st.divider()

# Prediction Action
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]

    st.subheader("Assessment Result")
    if prediction == 1:
        st.error("⚠️ **HEART DISEASE DETECTED**: The model predicts elevated risk of heart disease.")
    else:
        st.success("✅ **NO HEART DISEASE DETECTED**: The model predicts normal/low heart disease risk.")

    # Show input recap
    with st.expander("View Input Feature Summary"):
        st.dataframe(input_data)
        