import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and trained model
preprocessor = joblib.load("preprocessor_pipeline.pkl")
model = joblib.load("final_model.pkl")

# Page config with bright theme
st.set_page_config(page_title="Loan Approval Predictor 💰", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏦 Loan Approval Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #FF4B4B'>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("<h2 style='color: #FF7F50;'>🔧 Applicant Details</h2>", unsafe_allow_html=True)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0","1","2","3+"])
    Education = st.selectbox("Education", ["Graduate","Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes","No"])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])

    ApplicantIncome = st.slider("Applicant Income", 0, 100000, 3000, key="income", format="$%d")
    CoapplicantIncome = st.slider("Coapplicant Income", 0, 50000, 0, key="coincome", format="$%d")
    LoanAmount = st.slider("Loan Amount (in thousands)", 0, 1000, 66, key="loanamt", format="$%d")
    Loan_Amount_Term = st.slider("Loan Amount Term (months)", 12, 480, 360)
    Property_Area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])

    if st.button("🔮 Predict Loan Approval", key="predict"):
        X_input = preprocessor.transform(input_data)
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]
        st.session_state.pred = prediction
        st.session_state.proba = proba

with col2:
    st.markdown("<h2 style='color: #32CD32;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
    if "pred" in st.session_state:
        if st.session_state.pred == 1:
            st.success(f"✅ Loan Approved! 🎉")
        else:
            st.error(f"❌ Loan Not Approved 😔")
        st.info(f"Probability of Approval: **{st.session_state.proba:.2f}**")
        
        # Fun visuals
        st.balloons()
        st.progress(min(st.session_state.proba, 1.0))
    else:
        st.info("👉 Fill in the details and click 🔮 Predict to see the result!")