import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Loan Approval System",
    layout="wide",
    page_icon="🏦"
)

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("models/loan_approval_model.pkl", "rb"))

# -------------------------------
# Title Section
# -------------------------------
st.title("🏦 AI-Powered Loan Approval System")
st.markdown("### Intelligent Credit Risk Assessment Dashboard")
st.markdown("---")

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([1, 1])

# ===============================
# LEFT COLUMN – USER INPUT
# ===============================
with col1:
    st.subheader("📋 Applicant Information")

    no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)

    education = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        step=100000
    )

    loan_amount = st.number_input(
        "Loan Amount Requested (₹)",
        min_value=0,
        step=100000
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=0,
        step=1
    )

    cibil_score = st.slider(
        "CIBIL Score",
        300,
        900,
        650
    )

    st.markdown("### 🏠 Asset Details")

    residential_assets_value = st.number_input(
        "Residential Assets Value (₹)",
        min_value=0,
        step=100000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value (₹)",
        min_value=0,
        step=100000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value (₹)",
        min_value=0,
        step=100000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value (₹)",
        min_value=0,
        step=100000
    )

# ===============================
# FEATURE ENGINEERING
# ===============================
total_assets = (
    residential_assets_value +
    commercial_assets_value +
    luxury_assets_value +
    bank_asset_value
)

loan_income_ratio = (
    loan_amount / income_annum
    if income_annum > 0 else 0
)

input_data = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "total_assets": total_assets,
    "loan_income_ratio": loan_income_ratio
}])

# ===============================
# RIGHT COLUMN – RESULTS
# ===============================
with col2:
    st.subheader("📊 Loan Decision Analysis")

    if st.button("🔍 Evaluate Loan Application"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("### 🎯 Approval Probability")
        st.metric(
            label="Model Confidence",
            value=f"{probability*100:.2f}%"
        )

        # Loan Decision
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # Risk Category
        st.markdown("### ⚠ Risk Assessment")

        if cibil_score >= 750:
            st.success("🟢 Low Risk Applicant")
        elif cibil_score >= 600:
            st.warning("🟡 Medium Risk Applicant")
        else:
            st.error("🔴 High Risk Applicant")

        # Financial Summary
        st.markdown("---")
        st.markdown("### 💰 Financial Summary")

        st.write(f"**Total Assets:** ₹ {total_assets:,.0f}")
        st.write(f"**Loan-to-Income Ratio:** {loan_income_ratio:.2f}")

        # -------------------------------
        # Feature Importance Chart
        # -------------------------------
        st.markdown("---")
        st.markdown("### 📈 Top Influencing Factors")

        importances = model.named_steps["classifier"].feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(6)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        st.pyplot(fig)