python src/train_baseline.py --X data/processed/X_ciprofloxacin_train.parquet --y data/processed/y_ciprofloxacin_train.csv --X_test data/processed/X_ciprofloxacin_test.parquet --y_test data/processed/y_ciprofloxacin_test.csv --model_out models/xgb_ciprofloxacin.pkl




streamlit run src/app.py  



xgb_ciprofloxacin.pkl
X_ciprofloxacin_test.parquet