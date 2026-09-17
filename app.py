"""
Streamlit app: takes a customer's account details and predicts churn risk
using the trained Random Forest model (selected via model comparison in
src/train.py). Run with: streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(OUT_DIR, "best_model.joblib"))
    feature_columns = joblib.load(os.path.join(OUT_DIR, "feature_columns.joblib"))
    with open(os.path.join(OUT_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()
    return model, feature_columns, model_name


model, feature_columns, model_name = load_artifacts()

st.title("📉 Customer Churn Predictor")
st.caption(f"Model in use: **{model_name}** (selected by F1-score across Logistic Regression, Random Forest, and XGBoost — see README for full comparison)")

st.markdown("Enter a customer's account details to estimate their churn risk.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * max(tenure, 1)))

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    raw = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])

    # Recreate the same engineered features as src/data_prep.py
    raw["TenureGroup"] = pd.cut(
        raw["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    raw["NumAddonServices"] = (raw[service_cols] == "Yes").sum(axis=1)
    raw["AvgMonthlySpend"] = np.where(
        raw["tenure"] > 0, raw["TotalCharges"] / raw["tenure"], raw["MonthlyCharges"]
    )

    # One-hot encode and align to training-time columns
    encoded = pd.get_dummies(raw, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(encoded)[0, 1]
    pred = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

    st.divider()
    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{proba:.1%}")
    c2.metric("Prediction", pred)
    st.progress(min(float(proba), 1.0))

    if proba >= 0.5:
        st.warning("This customer profile matches patterns associated with higher churn risk — consider a retention offer.")
    else:
        st.success("This customer profile matches patterns associated with lower churn risk.")

st.divider()
with st.expander("Model performance (test set)"):
    import json
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    st.json(metrics)
    st.image(os.path.join(OUT_DIR, "model_comparison.png"), caption="Model comparison across all 3 algorithms")
    st.image(os.path.join(OUT_DIR, "shap_summary.png"), caption="SHAP summary — top churn drivers")
