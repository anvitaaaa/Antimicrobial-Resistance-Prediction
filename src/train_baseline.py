#!/usr/bin/env python3
"""
train_baseline_rf.py
- Train Random Forest classifier for an antibiotic
- Save model and evaluation metrics
"""
import argparse, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

def train_and_eval(X_train_path, y_train_path, X_test_path, y_test_path, model_out, n_estimators=300, n_splits=5, save_fold_models=False):
    # Load data
    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)

    # --- sanitize column names (for safety) ---
    X_train.columns = [str(c).replace('[','_').replace(']','_').replace('<','_').replace('>','_') for c in X_train.columns]
    X_test.columns  = [str(c).replace('[','_').replace(']','_').replace('<','_').replace('>','_') for c in X_test.columns]

    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    print("Shapes:", X_train.shape, X_test.shape)
    # If n_splits <= 1, keep previous behavior (train on full training set)
    if n_splits is None or int(n_splits) <= 1:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
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
        return

    # --- Stratified K-Fold training ---
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=42)
    test_probs_sum = np.zeros(len(X_test))
    fold_train_accs = []
    fold_val_aucs = []
    saved_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=8,
            random_state=42 + fold_idx
        )
        clf.fit(X_tr, y_tr)

        # Training accuracy for this fold
        train_probs = clf.predict_proba(X_tr)[:, 1]
        train_acc_fold = (train_probs >= 0.5).astype(int) == y_tr
        train_acc_fold = train_acc_fold.mean()
        fold_train_accs.append(train_acc_fold)

        # Validation AUC for this fold (informational)
        val_probs = clf.predict_proba(X_val)[:, 1]
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = float('nan')
        fold_val_aucs.append(val_auc)

        # Predict on the held-out test set and accumulate probabilities
        probs = clf.predict_proba(X_test)[:, 1]
        test_probs_sum += probs

        # Optionally save fold model
        if save_fold_models:
            fold_path = f"{model_out}.fold{fold_idx}.pkl"
            joblib.dump(clf, fold_path)
            saved_models.append(fold_path)
            print(f"Saved fold {fold_idx} model to {fold_path}")

        print(f"Fold {fold_idx}: train_acc={train_acc_fold:.4f}, val_auc={val_auc}")

    # Average probabilities across folds
    avg_test_probs = test_probs_sum / int(n_splits)
    y_pred = (avg_test_probs >= 0.5).astype(int)

    # Aggregate metrics
    train_acc = float(np.mean(fold_train_accs)) if fold_train_accs else float('nan')
    test_acc = float((y_pred == y_test).mean())
    auc = roc_auc_score(y_test, avg_test_probs)
    ap = average_precision_score(y_test, avg_test_probs)

    print("Fold train accs:", fold_train_accs)
    print("Fold val AUCs:", fold_val_aucs)
    print("Averaged AUROC:", auc)
    print("Averaged AUPRC:", ap)
    print(f"Avg train accuracy: {train_acc:.4f}")
    print(f"Test accuracy (ensemble):  {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save final model: if fold models not saved, retrain a model on full training set and save it for compatibility
    if not save_fold_models:
        final_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=8,
            random_state=42
        )
        final_clf.fit(X_train, y_train)
        joblib.dump(final_clf, model_out)
        print("Saved final model (trained on full training data) to", model_out)
    else:
        # Save a small index file listing fold model paths
        pd.Series(saved_models).to_csv(model_out + ".folds.list.csv", index=False)

    # Save aggregated metrics
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
