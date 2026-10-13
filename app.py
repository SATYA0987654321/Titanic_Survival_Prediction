import streamlit as st
import pandas as pd
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("titanic_xgb_model.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model file: {e}")
    st.stop()

# App title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter the passenger details to predict their survival.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:
    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0
    )
    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0
    )
    embarked = st.selectbox(
        "Port of Embarkation",
        ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"]
    )

# Convert inputs to the same format used during training
sex_male = 1 if sex == "Male" else 0

embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0

# Keep the feature names and order consistent with training
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex_male": sex_male,
    "Embarked_Q": embarked_q,
    "Embarked_S": embarked_s
}])

# Prediction
if st.button("Predict Survival", type="primary"):
    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("The passenger is predicted to survive.")
        else:
            st.error("The passenger is predicted not to survive.")

        # Display probability when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.metric(
                "Estimated Survival Probability",
                f"{probabilities[1] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")