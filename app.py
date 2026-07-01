import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# Load dataset (CLEANED)
# ==============================
df = pd.read_csv("cleaned_dataset.csv")

# ==============================
# Load trained model
# ==============================
model = joblib.load("rf_model.pkl")

# ==============================
# Feature Engineering
# ==============================
df['total_assets'] = (
    df['residential_assets_value'] +
    df['commercial_assets_value'] +
    df['luxury_assets_value'] +
    df['bank_asset_value']
)

# ==============================
# Title
# ==============================
st.title("Loan Approval Prediction Dashboard")
st.write("Interactive dashboard where visualisations adapt to applicant input and support decision-making using a trained Random Forest model.")

# ==============================
# USER INPUT SECTION (MAIN DRIVER)
# ==============================
st.subheader("Enter Applicant Details")

income = st.number_input("Annual Income", min_value=0, value=5000000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=1000000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
assets = st.number_input("Total Assets", min_value=0, value=10000000)
education = st.selectbox(
    "Education Level",
    options=[0, 1],
    format_func=lambda x: "Graduate" if x == 1 else "Not Graduate"
)
self_employed = st.selectbox(
    "Self Employed",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# ==============================
# DYNAMIC FILTERING (Power BI style)
# ==============================
st.subheader("Analysis Based on Similar Applicants")


filtered_df = df[
    (df['income_annum'] >= income * 0.5) &
    (df['income_annum'] <= income * 1.5) &
    (df['cibil_score'] >= credit_score * 0.8) &
    (df['cibil_score'] <= credit_score * 1.2)
]


if len(filtered_df) == 0:
    filtered_df = df
    st.warning("No similar data found — showing full dataset.")


# ==============================
# ✅ VISUAL 1: Loan Approval Rate
# ==============================

st.subheader("Loan Status Distribution")
st.bar_chart(filtered_df['loan_status'].value_counts())

# ==============================
# ✅ VISUAL 2: Income vs Loan (matched data)
# ==============================

st.subheader("Income vs Loan Amount")
st.scatter_chart(filtered_df[['income_annum', 'loan_amount']])

# ==============================
# ✅ VISUAL 3: Credit Score Trend
# ==============================

st.subheader("Credit Score Distribution")
st.line_chart(filtered_df['cibil_score'])


# ==============================
# ✅ ANALYTICAL INSIGHT (FIXED ✅)
# ==============================

if len(filtered_df) > 0:

    # Define first
    approval_rate = filtered_df['loan_status'].mean() * 100

    # Then use it
    if approval_rate > 60:
        st.success("✅ Similar applicants have a high chance of approval")
    elif approval_rate < 40:
        st.error("❌ Similar applicants have a low chance of approval")
    else:
        st.warning("⚖️ Similar applicants have a moderate chance of approval")

# ==============================
# ✅ MODEL PREDICTION
# ==============================
st.subheader("Loan Prediction (Using Random Forest Model)")

if st.button("Predict Loan Status"):

    input_data = np.array([[ 
        1,               # loan_id (dummy)
        0,               # no_of_dependents
        education,
        self_employed,
        income,
        loan_amount,
        12,
        credit_score,
        assets/4,
        assets/4,
        assets/4,
        assets/4
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # Additional explanation

    st.write(f"Based on similar applicants, approval likelihood is approximately {approval_rate:.2f}%")

