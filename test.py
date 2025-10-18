import pandas as pd

df = pd.read_csv(
    "data/raw/isolates.csv",
    low_memory=False,
    on_bad_lines='skip'   # skip malformed rows
)
print(df.columns)
print(df['AST phenotypes'].head())
