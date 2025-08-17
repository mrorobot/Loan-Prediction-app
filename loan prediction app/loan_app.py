# loan_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocess import prepare_input_dataframe, FEATURE_ORDER

# model config...
MODEL_PATH = "loan_model.pkl"  
# 

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)


model = load_model()

st.title("🏦 Loan Approval — Interactive Demo")
st.write("Predict whether a loan application will be approved. Enter realistic values for better predictions.")

# --- Sidebar: info / model probability toggle
with st.sidebar:
    st.header("App Options")
    show_prob = st.checkbox("Show approval probability", value=True)
    st.markdown("---")
    st.markdown("**Notes**")
    st.caption("This demo uses a pre-trained model. Ensure your model was trained with the same preprocessing logic.")

# --- Input form layout
st.subheader("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Income (Annual)", min_value=0, value=250000, step=1000)
    
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=150, step=1)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0, step=30.0)
    credit_history = st.selectbox("Credit History (1 = good)", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")
if st.button("Predict"):
    # Build input dict and preprocess
    
    raw = {
        "Male": gender,
        "Married": married,
        
        "Non Graduate": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    X_input = prepare_input_dataframe(raw)

    try:
        # If model supports predict_proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            # assume positive class is 1 (Approved)
            prob_yes = probs[1]
        else:
            prob_yes = None

        pred = model.predict(X_input)[0]

        if pred == 1:
            st.success("✅ Loan likely to be APPROVED")
        else:
            st.error("❌ Loan likely to be NOT APPROVED")

        if show_prob and prob_yes is not None:
            st.write(f"**Approval Probability:** {prob_yes*100:.2f}%")

        # show the input features used
        st.markdown("**Model Input (processed):**")
        st.dataframe(X_input.T, use_container_width=True)

        # If model has feature_importances_
        if hasattr(model, "feature_importances_"):
            import matplotlib.pyplot as plt
            importances = model.feature_importances_
            fi = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=False)[:10]
            st.markdown("**Top feature importances**")
            fig, ax = plt.subplots()
            fi.plot.bar(ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Built by Inzayn.ai — Demo for AI Specialist Bootcamp")
