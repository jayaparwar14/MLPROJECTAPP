import joblib
import pandas as pd
import streamlit as st
import os

# lr = joblib.load("Titanic_lr_model.pkl")
# scaler = joblib.load("Titanic_scaler.pkl")


BASE_DIR = os.path.dirname(__file__)

# lr = joblib.load(os.path.join(BASE_DIR, "Titanic_lr_model.pkl"))
# scaler = joblib.load(os.path.join(BASE_DIR, "Titanic_scaler.pkl"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "Titanic_lr_model.pkl"))

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE_DIR, "Titanic_scaler.pkl"))

lr = load_model()
scaler = load_scaler()

def predict_survival () :

    st.title("🚢 Titanic Survival Predictor")

    st.subheader("Predict whether a passenger would survive the Titanic disaster")

    st.write(
        """
        Enter the passenger details below and click **Predict**
        to estimate the survival outcome using a Machine Learning model.
        """
    )    

    pclass = st.selectbox("Passengr Class " , [1,2,3])
    gender = st.selectbox("Gender " , ["Male" , "Female"])
    age = st.number_input("Age " , min_value = 0 , max_value = 100 , value = 25 )
    sibsp = st.number_input("Sibilings " , min_value = 0 , max_value = 8 , step = 1 ) 
    parch = st.number_input("Parch " , min_value = 0 , max_value = 6 , step = 1 ) 
    fare = st.number_input("Fare " , min_value = 0.0 , value = 10.0) 
    embarked_c = st.selectbox("Emabarked_c " , [0,1])
    embarked_q = st.selectbox("Embarked_q " , [0,1])
    embarked_s = st.selectbox("Embarked_s " , [0,1])

    if st.button("Predict") :

        x = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C",
           "Embarked_Q", "Embarked_S"]

        gender = 0 if gender == "Female" else 1

        data = pd.DataFrame(
            [[
            pclass,
            gender,
            age,
            sibsp,
            parch,
            fare,
            embarked_c,
            embarked_q,
            embarked_s
        ]] , columns = x)

        with st.spinner("Predicting survival..."):


            data = scaler.transform(data)

            prediction = lr.predict(data)

        if prediction[0] == 1 :

            st.success("Passenger survived ✅ ")

            with st.expander("Why was the survival chance high?"):
                st.write("""
                - Younger age may improve survival chances.
                - Higher passenger class often had better access to lifeboats.
                - Gender can influence the prediction because of historical patterns in the dataset.
                - Fare and embarkation information are also considered by the model.
                """)

        else :

            st.error("Passenger is not survived ❌ ")

            with st.expander("Why was the survival chance low?"):
                st.write("""
                - Passenger details matched patterns associated with lower survival rates.
                - Passenger class, fare, age, and family size can affect the prediction.
                - This is only a machine learning prediction, not a certainty.
                """)

    with st.expander("About this Project"):
        st.write("""
        🚢 Titanic Survival Prediction System

        This project uses a Logistic Regression model trained on the Titanic dataset.

        Features Used:
        - Passenger Class
        - Gender
        - Age
        - Siblings/Spouses
        - Parents/Children
        - Fare
        - Embarkation Details

        The model predicts whether a passenger would likely survive.
        """)

        

predict_survival()

