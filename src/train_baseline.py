#!/usr/bin/env python3
"""
train_baseline_rf.py
- Train Random Forest classifier for an antibiotic
- Save model and evaluation metrics
"""
import argparse, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

def train_and_eval(X_train_path, y_train_path, X_test_path, y_test_path, model_out, n_estimators=300):
    # Load data
    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)

    # --- sanitize column names (for safety) ---
    X_train.columns = [str(c).replace('[','_').replace(']','_').replace('<','_').replace('>','_') for c in X_train.columns]
    X_test.columns  = [str(c).replace('[','_').replace(']','_').replace('<','_').replace('>','_') for c in X_test.columns]

    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    print("Shapes:", X_train.shape, X_test.shape)
    
    # --- Random Forest classifier ---
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,         # you can tune this (default = None = fully grown trees)
        n_jobs=8,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    # Compute training accuracy
    train_probs = clf.predict_proba(X_train)[:, 1]
    y_train_pred = (train_probs >= 0.5).astype(int)
    train_acc = (y_train_pred == y_train).mean()
    test_acc = (y_pred == y_test).mean()
    
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    
    print("AUROC:", auc)
    print("AUPRC:", ap)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    joblib.dump(clf, model_out)
    print("Saved model to", model_out)
    
    # Save metrics
    metrics = {'AUROC': auc, 'AUPRC': ap, 'train_accuracy': train_acc, 'test_accuracy': test_acc}
    pd.Series(metrics).to_csv(model_out + ".metrics.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--X", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--X_test", required=True)
    p.add_argument("--y_test", required=True)
    p.add_argument("--model_out", required=True)
    args = p.parse_args()
    train_and_eval(args.X, args.y, args.X_test, args.y_test, args.model_out)
