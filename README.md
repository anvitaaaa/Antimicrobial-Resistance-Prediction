# Antimicrobial-Resistance-Prediction
A project to predict antimicrobial resistance from genomic-derived features. This README provides a step-by-step guide (Windows-focused) to set up the environment and run training, evaluation, and the demo app in this repository.
## Repo layout (important files)
- `data/` - contains raw, processed, and test datasets.
	- `data/processed/` - prepared feature matrices (`X_*.parquet`) and labels (`y_*.csv`).
- `models/` - saved model files and metric CSVs.
- `src/` - source code:
	- `train_baseline.py` - train RandomForest baseline; supports Stratified K-Fold via `--n_splits`.
	- `app.py` - Streamlit demo app (run via `streamlit run src/app.py`).
	- other scripts: `featurize.py`, `ingest.py`, `parse_card.py`, `domain_adapt.py`, `explain.py`.
- `requirements.txt` - Python dependencies.
## Quick setup (Windows PowerShell)
1. Create a Python virtual environment and activate it (PowerShell):

```powershell
# From the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```powershell
# On some systems, you may need to allow scripts (once):
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
2. Upgrade pip and install required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Verify you can import key packages (optional):

```powershell
python -c "import pandas, sklearn, joblib; print('OK')"
```
## How to run baseline training

The baseline RandomForest trainer is `src/train_baseline.py`. It previously trained one RF on the training set and evaluated on a test set. It now supports Stratified K-Fold training to mitigate class imbalance.

Usage (single model trained on full training data — original behavior):

```powershell
python src/train_baseline.py --X data/processed/X_ampicillin_train.parquet --y data/processed/y_ampicillin_train.csv --X_test data/processed/X_ampicillin_test.parquet --y_test data/processed/y_ampicillin_test.csv --model_out models/rf_ampicillin.pkl
```

Usage (Stratified K-Fold ensemble, default 5 folds):

```powershell
python src/train_baseline.py --X data/processed/X_ampicillin_train.parquet --y data/processed/y_ampicillin_train.csv --X_test data/processed/X_ampicillin_test.parquet --y_test data/processed/y_ampicillin_test.csv --model_out models/rf_ampicillin.pkl --n_splits 5
```

Notes on flags:
- `--n_splits` (int): number of StratifiedKFold splits. If omitted or set to `1`, the script trains a single model on the full training data (old behavior).
- `--save_fold_models` (flag): if provided, each fold model will be saved as `<model_out>.fold{fold_idx}.pkl` and a list of saved models will be written to `<model_out>.folds.list.csv`.

Example saving fold models:

```powershell
python src/train_baseline.py --X data/processed/X_ampicillin_train.parquet --y data/processed/y_ampicillin_train.csv --X_test data/processed/X_ampicillin_test.parquet --y_test data/processed/y_ampicillin_test.csv --model_out models/rf_ampicillin.pkl --n_splits 5 --save_fold_models
```

Output files created by training:
- `<model_out>` — the final model (when not saving fold models) or a model file representing the final model trained on full training data.
- `<model_out>.metrics.csv` — CSV file with aggregated metrics (AUROC, AUPRC, train_accuracy, test_accuracy).
- `<model_out>.fold{i}.pkl` (optional) — per-fold model files when `--save_fold_models` is used.
- `<model_out>.folds.list.csv` (optional) — CSV listing saved fold model paths.

## Running the Streamlit app (demo)

Start the Streamlit demo to explore predictions and explanations:

```powershell
# Activate the venv first, then run:
streamlit run src/app.py
```

Streamlit will open a local URL (e.g., http://localhost:8501) in your browser.

## How to add a new antibiotic model

1. Prepare features and labels in `data/processed/` following existing naming convention: `X_<antibiotic>_train.parquet`, `y_<antibiotic>_train.csv`, and corresponding test files.
2. Run training using the commands above, substituting the antibiotic name.

## Tips for working with class imbalance

- Use `--n_splits 5` (or higher) to perform Stratified K-Fold training which preserves label ratios within folds. The script aggregates probabilities across folds before computing final metrics.
- Consider additional techniques if imbalance remains severe: class weighting in the model, oversampling (SMOTE), or threshold tuning for prediction.

## Developer notes

- The RandomForest uses `n_estimators=300` and `n_jobs=8` by default; you can change the source if you need a different configuration.
- The code sanitizes column names from the parquet feature files to avoid problematic characters.

## Troubleshooting

- If imports fail, ensure the virtual environment is active and `requirements.txt` installed.
- If you encounter permission errors when activating PowerShell scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

- If `roc_auc_score` throws an error during fold validation, it's usually because a fold's validation set contained only one class. StratifiedKFold should minimize this, but for very small datasets you may still hit this case; the script will print `nan` for that fold's AUC.

## Reproducing results & experiments

- Save the `models/*.pkl` and corresponding `*.metrics.csv` for reproducibility.
- Use `joblib` and the included model files to load models programmatically:

```python
import joblib
clf = joblib.load('models/rf_ampicillin.pkl')
probs = clf.predict_proba(X_new)[:, 1]
```





xgb_ciprofloxacin.pkl
X_ciprofloxacin_test.parquet
