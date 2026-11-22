import joblib
import pandas as pd 
import streamlit as st
import os 

# rfc = joblib.load("Loan_Approval_model.pkl")
# scaler = joblib.load("Loan_Approval_scaler.pkl")

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource 
def load_model():
    return joblib.load(os.path.join(BASE_DIR , "Loan_Approval_rfc_model.pkl"))

def load_scaler():
    return joblib.load(os.path.join(BASE_DIR , "Loan_Approval_scaler.pkl"))

rfc = load_model()
scaler = load_scaler()

def LoanApprovalPredict () :

    st.title("🏦 Loan Approval Prediction System")

    st.subheader("Predict whether a loan application is likely to be approved")

    st.write("""
    Enter the applicant details below and click **Predict Loan Status**
    to estimate whether the loan application is likely to be approved
    using a Machine Learning model.
    """)

    age = st.number_input("Candidate_Age " , min_value = 0 , max_value = 100 , value = 25)
    gender = st.selectbox("Candidate_Gender " , ["Male" , "Female"])
    education = st.selectbox("Education Level " , ["Associate" , "Bachelor" , "Doctorate" , "High School" , "Master"])
    person_income = st.number_input("Annual Income" , min_value = 1)
    emmployee_experience = st.number_input("Work Experience (Years)" , min_value = 0)
    home_ownership = st.selectbox("Home Ownership Status " , ["MORTGAGE" , "OWN" , "RENT" , "OTHER"])
    loan_amount = st.number_input("Loan Ammount " , min_value = 0)
    loan_intent = st.selectbox("Loan Purpose" , ["DEBTCONSOLIDATION" , "EDUCATION" , "HOMEIMPROVEMENT" , "MEDICAL" , "PERSONAL" , "VENTURE"])
    loan_interest_rate = st.number_input("Interest Rate Offered by Bank (%) ", min_value = 0.0)
    loan_percentage = loan_amount / person_income
    credit_history = st.number_input("For how many years have you been using loans or credit cards? " , min_value = 0)
    credit_score = st.number_input("CIBIL Score (if known) " , min_value = 300 , max_value = 850)
    previous_loan = st.selectbox("Previous Loan Default " , ["No" , "Yes"])


    if st.button("🔍 Predict Loan Status") :


        gender = 0 if gender == "Female" else 1
        previous_loan = 0 if previous_loan == "No" else 1 
        Education_Associate = 1 if education == "Associate" else 0 
        Education_Bachelor = 1 if education == "Bachelor" else 0 
        Education_Doctorate = 1 if education == "Doctorate" else 0 
        Education_High = 1 if education == "High School" else 0 
        Education_Master = 1 if education == "Master" else 0 
        Home_Onwership_MORTGAGE = 1 if home_ownership == "MORTGAGE" else 0
        Home_Onwership_OTHER = 1 if home_ownership == "OTHER" else 0
        Home_Onwership_OWN = 1 if home_ownership == "OWN" else 0
        Home_Onwership_RENT = 1 if home_ownership == "RENT" else 0
        Loan_Intent_DEBTCONSOLIDATION = 1 if loan_intent == "DEBTCONSOLIDATION" else 0
        Loan_Intent_EDUCATION = 1 if loan_intent == "EDUCATION" else 0
        Loan_Intent_HOMEIMPROVEMENT = 1 if loan_intent == "HOMEIMPROVEMENT" else 0
        Loan_Intent_MEDICAL = 1 if loan_intent == "MEDICAL" else 0
        Loan_Intent_PERSONAL = 1 if loan_intent == "PERSONAL" else 0
        Loan_Intent_VENTURE = 1 if loan_intent == "VENTURE" else 0

        x = ["Age", "Gender", "Person Income", "Employee Experience", "Loan Amount",
           "Loan interest Rate", "Loan percentage", "Credit History",
           "Credit Score", "Previous Loan", "Education_Associate",
           "Education_Bachelor", "Education_Doctorate", "Education_High School",
           "Education_Master", "Home Onwership_MORTGAGE", "Home Onwership_OTHER",
           "Home Onwership_OWN", "Home Onwership_RENT",
           "Loan Intent_DEBTCONSOLIDATION", "Loan Intent_EDUCATION",
           "Loan Intent_HOMEIMPROVEMENT", "Loan Intent_MEDICAL",
           "Loan Intent_PERSONAL", "Loan Intent_VENTURE"]

        data = pd.DataFrame(
            [[
                age,
                gender,
                person_income,
                emmployee_experience,
                loan_amount,
                loan_interest_rate,
                loan_percentage,
                credit_history,
                credit_score,
                previous_loan,
                Education_Associate,
                Education_Bachelor,
                Education_Doctorate,
                Education_High,
                Education_Master,
                Home_Onwership_MORTGAGE,
                Home_Onwership_OTHER,
                Home_Onwership_OWN,
                Home_Onwership_RENT,
                Loan_Intent_DEBTCONSOLIDATION,
                Loan_Intent_EDUCATION,
                Loan_Intent_HOMEIMPROVEMENT,
                Loan_Intent_MEDICAL,
                Loan_Intent_PERSONAL,
                Loan_Intent_VENTURE,
            ]] , columns = x
        )

        with st.spinner("Analyzing applicant profile..."):

            data = scaler.transform(data)
            predict = rfc.predict(data)

        if predict[0] == 0 :

            st.success("✅ Loan Approved")

            with st.expander("Why was the loan approved?"):

                st.write("""
                - Applicant has a strong financial profile.
                - Income appears sufficient compared to the requested loan amount.
                - Credit history and credit score support loan approval.
                - Previous loan behavior indicates lower lending risk.
                - The application matches patterns commonly associated with approved loans.
                """)

        else :

            st.error("❌ Rejected")

            with st.expander("Why was the loan rejected?"):

                st.write("""
                - The applicant profile matches patterns associated with higher lending risk.
                - Income may be lower relative to the requested loan amount.
                - Credit score or credit history may not meet approval criteria.
                - Previous loan defaults can negatively affect eligibility.
                - This prediction is generated by a machine learning model and is not a final lending decision.
                """)       


    with st.expander("About this Project"):
        st.write("""
        🏦 Loan Approval Prediction System

        This project uses a Machine Learning model trained on historical
        loan application data to predict whether a loan is likely to be approved.

        Features Used:
        - Age
        - Gender
        - Education
        - Annual Income
        - Employment Experience
        - Home Ownership
        - Loan Amount
        - Loan Purpose
        - Interest Rate
        - Loan Percent Income
        - Credit History Length
        - Credit Score
        - Previous Loan Default

        The model analyzes an applicant's financial profile and predicts
        the probability of loan approval.
        """)     

        

LoanApprovalPredict()
