#!/usr/bin/env python3
"""
explain.py
- Load an XGBoost model and a test set, compute SHAP summary plot and save top features.
"""
import argparse, joblib, pandas as pd, shap, matplotlib.pyplot as plt, numpy as np

def explain(model_path, X_test_path, out_prefix):
    clf = joblib.load(model_path)
    X_test = pd.read_parquet(X_test_path)

# --- sanitize column names for XGBoost / SHAP ---
    X_test.columns = [str(c).replace('[','_').replace(']','_').replace('<','_').replace('>','_').replace(' ','_').replace(',','_') for c in X_test.columns]

    # shap for tree model
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # summary plot
    plt.figure(figsize=(8,10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_prefix + "_shap_summary.png", dpi=150)
    # top features
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)
    feat_imp.head(50).to_csv(out_prefix + "_top50_features.csv")
    print("Saved SHAP summary and top features")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--X_test", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    explain(args.model, args.X_test, args.out)
