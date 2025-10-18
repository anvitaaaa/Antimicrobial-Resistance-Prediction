#!/usr/bin/env python3
"""
Streamlit dashboard for AMR prediction demo.
Displays model prediction and top SHAP features for an uploaded isolate.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = "models/xgb_ciprofloxacin.pkl"
FEATURES_PATH = "data/processed/X_ciprofloxacin_test.parquet"

# --- PAGE SETTINGS ---
st.set_page_config(page_title="AMR Prediction Dashboard", layout="wide")
st.title("🧬 Antimicrobial Resistance Prediction Dashboard")
st.markdown("#### Using Machine Learning for Effective Drug Management")

# --- SIDEBAR ---
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("Select antibiotic and upload isolate data to predict resistance.")
antibiotic = st.sidebar.selectbox("Select Antibiotic", ["ciprofloxacin", "ampicillin", "ceftriaxone"])
st.sidebar.info("Currently demoing Ciprofloxacin model trained on isolate + CARD data.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# --- LOAD SAMPLE FEATURE TEMPLATE ---
@st.cache_data
def load_features(path):
    X = pd.read_parquet(path)
    return X

clf = load_model(MODEL_PATH)
X_template = load_features(FEATURES_PATH)
st.sidebar.success("Model and feature template loaded successfully!")

# --- MODEL UPLOADER ---
st.sidebar.markdown("### Upload a custom model (optional)")
model_upload = st.sidebar.file_uploader(
    "Upload XGBoost model (.pkl)", 
    type=["pkl"]
)

if model_upload:
    clf = joblib.load(model_upload)
    st.sidebar.success("Custom model loaded successfully!")

# --- DATA UPLOADER ---
st.markdown("### 🧫 Enter or Upload Isolate Data")
data_upload = st.file_uploader(
    "Upload isolate data file (CSV or Parquet)", 
    type=["csv", "parquet"]
)

if data_upload:
    if data_upload.name.endswith(".csv"):
        user_df = pd.read_csv(data_upload)
    elif data_upload.name.endswith(".parquet"):
        user_df = pd.read_parquet(data_upload)
    else:
        st.error("Unsupported file type!")
        st.stop()
    st.write("✅ Uploaded data preview:")
    st.dataframe(user_df.head())
else:
    st.markdown("Or select from sample isolates below (randomly loaded for demo).")
    sample = X_template.sample(5, random_state=42)
    st.dataframe(sample)
    user_df = sample


# --- PREDICTION ---
if st.button("🚀 Predict Resistance"):
    # --- SANITIZE COLUMN NAMES ---
    user_df.columns = [
        str(c).replace('[','_')
              .replace(']','_')
              .replace('<','_')
              .replace('>','_') 
        for c in user_df.columns
    ]

    # --- ALIGN FEATURES TO TEMPLATE ---
    # Fill missing columns with 0, ignore extra columns
    user_df = user_df.reindex(columns=X_template.columns, fill_value=0)

    # --- PREDICTION ---
    preds = clf.predict_proba(user_df)[:, 1]
    pred_df = pd.DataFrame({
        "Prediction Probability (Resistant)": preds,
        "Prediction Label": np.where(preds >= 0.5, "Resistant", "Susceptible")
    })

    st.markdown("### 🧠 Model Predictions")
    st.dataframe(pred_df)
    avg = np.mean(preds)
    st.metric("Average Resistance Probability", f"{avg:.2%}")

    # --- SHAP EXPLANATION ---
    st.markdown("### 🔍 Model Interpretation (SHAP Feature Importance)")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(user_df)

    shap.summary_plot(shap_values, user_df, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.success("Explanation generated successfully!")


# --- FOOTER ---
st.markdown("---")
st.caption("Developed for ML Hackathon — Predicting Antimicrobial Resistance using Machine Learning.")
