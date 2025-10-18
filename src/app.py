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
antibiotic = st.sidebar.selectbox(
    "Select Antibiotic", 
    ["ciprofloxacin", "ampicillin", "ceftriaxone", "tetracycline", 
     "gentamicin", "azithromycin", "trimethoprim-sulfamethoxazole", "chloramphenicol"]
)
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
model_features = list(clf.feature_names_in_)
X_template = load_features(FEATURES_PATH)
st.sidebar.success("Model and feature template loaded successfully!")

# --- FUNCTION TO ALIGN USER DATA TO MODEL FEATURES ---
def align_to_model(df, model_features):
    """
    Align dataframe columns to the model's features:
    - Missing columns are filled with 0
    - Extra columns are dropped
    - Duplicate columns are removed (keep first occurrence)
    """
    # Normalize column names
    df.columns = [str(c).replace('[','').replace(']','').replace(';','').replace(' ','') for c in df.columns]
    
    # Remove duplicate columns
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Reindex to model features
    aligned_df = df.reindex(columns=model_features, fill_value=0)
    return aligned_df

# --- MODEL UPLOADER ---
st.sidebar.markdown("### Upload a custom model (optional)")
model_upload = st.sidebar.file_uploader(
    "Upload XGBoost model (.pkl)", 
    type=["pkl"]
)

if model_upload:
    clf = joblib.load(model_upload)
    model_features = list(clf.feature_names_in_)
    st.sidebar.success("Custom model loaded successfully!")

# --- DATA UPLOADER ---
st.markdown("### 🧫 Upload Isolate Data")
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
else:
    # Use sample data silently (no display)
    user_df = X_template.sample(5, random_state=42)

# Align user data to model
user_df_aligned = align_to_model(user_df, model_features)

# --- PREDICTION ---
if st.button("🚀 Predict Resistance"):
    preds = clf.predict_proba(user_df_aligned)[:, 1]
    pred_labels = np.where(preds >= 0.5, "Resistant", "Susceptible")
    pred_df = pd.DataFrame({
        "Prediction Probability (Resistant)": preds,
        "Prediction Label": pred_labels
    })

    st.markdown("### 🧠 Model Predictions")
    st.dataframe(pred_df)
    st.metric("Average Resistance Probability", f"{np.mean(preds):.2%}")

    # --- SHAP EXPLANATION ---
    st.markdown("### 🔍 Model Interpretation (SHAP Feature Importance)")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(user_df_aligned)

    shap.summary_plot(shap_values, user_df_aligned, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.success("Explanation generated successfully!")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for ML Hackathon — Predicting Antimicrobial Resistance using Machine Learning.")
