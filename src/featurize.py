#!/usr/bin/env python3
"""
featurize.py
- Build feature matrix X and target y for a chosen antibiotic.
- Saves X_train.parquet, X_test.parquet, y_train.csv, y_test.csv, plus domain column file for domain adaptation.
"""
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re

def parse_ast_cell(cell):
    """
    Parse AST phenotypes like:
    amikacin=S,amoxicillin-clavulanic acid=S,...,ciprofloxacin=S
    Returns a dict: {antibiotic: phenotype}
    """
    if pd.isna(cell): 
        return {}
    out = {}
    for pair in str(cell).split(','):
        pair = pair.strip()
        if '=' in pair:
            k,v = pair.split('=',1)
            out[k.lower()] = v.upper()
    return out

def extract_target(df, antibiotic):
    """
    Extract phenotype (S/I/R) for the target antibiotic from AST phenotypes column.
    Returns 1 if resistant, 0 if susceptible, NaN if not present.
    """
    antibiotic = antibiotic.lower()
    parsed = df['AST phenotypes'].map(parse_ast_cell)
    y = parsed.map(lambda d: 1 if (isinstance(d, dict) and antibiotic in d and d[antibiotic].startswith('R')) 
                   else (0 if (isinstance(d, dict) and antibiotic in d and d[antibiotic].startswith('S')) 
                         else np.nan))
    return y

def parse_amr_genotypes(df):
    # Normalize AMR genotypes column into set of gene tokens.
    def get_genes(cell):
        if pd.isna(cell): return []
        s = str(cell)
        tokens = re.split(r'[;,|/]+', s)
        tokens = [t.strip() for t in tokens if t.strip()]
        tokens = [t.split()[0] for t in tokens]  # keep first token part
        return tokens
    gene_lists = df['AMR genotypes'].map(get_genes)
    cnt = Counter()
    for gl in gene_lists:
        cnt.update(set(gl))
    top = [g for g,c in cnt.most_common(2000)]  # adjustable
    Xg = pd.DataFrame(0, index=df.index, columns=top, dtype='uint8')
    for idx, gl in gene_lists.items():
        for g in gl:
            if g in Xg.columns:
                Xg.at[idx,g] = 1
    return Xg

def build_metadata_features(df):
    """
    Returns:
        meta_num: DataFrame with numerical/categorical features for modeling
        domains: Series with original Location values for domain adaptation
    """
    meta = pd.DataFrame(index=df.index)

    # Safe collection_year extraction
    col_dates = pd.to_datetime(df.get('Collection date', pd.Series()), errors='coerce', infer_datetime_format=True)
    meta['collection_year'] = col_dates.dt.year.fillna(0).astype(int)

    # Preserve original Location for domains
    domains = df.get('Location', pd.Series('NA')).fillna('NA').astype(str)

    # Metadata columns
    meta['Location'] = domains.map(lambda s: s.split(':')[0])
    meta['isolation_source'] = df.get('Isolation source', pd.Series('NA')).fillna('NA')
    meta['host'] = df.get('Host', pd.Series('NA')).fillna('NA')

    # Latitude / Longitude parsing
    def parse_latlon(x):
        if pd.isna(x): return (np.nan, np.nan)
        try:
            parts = str(x).split(',')
            return float(parts[0]), float(parts[1])
        except:
            return np.nan, np.nan
    latlon = df.get('Lat/Lon', pd.Series()).apply(parse_latlon)
    meta['latitude'] = latlon.map(lambda t: t[0])
    meta['longitude'] = latlon.map(lambda t: t[1])

    # One-hot encode categorical metadata
    cat = pd.get_dummies(meta[['Location','isolation_source','host']].fillna('NA'),
                         drop_first=True, dtype='uint8')
    meta_num = pd.concat([meta[['collection_year','latitude','longitude']].fillna(0), cat], axis=1)

    return meta_num, domains

def featurize(isolates_path, card_map_csv, antibiotic, outdir, sample=None, test_size=0.2, random_state=42):
    print("Loading isolates (this can take a while)...")
    df = pd.read_csv(isolates_path, low_memory=False, on_bad_lines='skip')

    # Parse AST phenotypes
    parsed = df['AST phenotypes'].map(parse_ast_cell)
    
    # Filter rows that actually have the target antibiotic
    has_antibiotic = parsed.map(lambda d: antibiotic.lower() in d)
    df = df[has_antibiotic].reset_index(drop=True)
    parsed = parsed[has_antibiotic].reset_index(drop=True)  # align indices

    # Sample if requested
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)
        parsed = parsed.sample(n=sample, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {len(df)} rows containing '{antibiotic}' for development.")
    else:
        print(f"Using all {len(df)} rows containing '{antibiotic}'.")

    # Build target column
    y = parsed.map(lambda d: 1 if d[antibiotic.lower()].startswith('R')
                    else 0 if d[antibiotic.lower()].startswith('S')
                    else np.nan)

    # Drop rows with missing labels
    mask = y.notna()
    df = df[mask].reset_index(drop=True)
    parsed = parsed[mask].reset_index(drop=True)
    y = y[mask].astype(int).reset_index(drop=True)

    print(f"After filtering for antibiotic '{antibiotic}', {len(df)} labelled isolates remain.")

    # Genomic features
    print("Parsing AMR genotypes...")
    Xg = parse_amr_genotypes(df)
    print(f"Genomic matrix shape: {Xg.shape}")

    # Metadata features + domains
    print("Building metadata features...")
    Xm, domains = build_metadata_features(df)
    print(f"Metadata shape: {Xm.shape}")

    # Combine
    X = pd.concat([Xm.reset_index(drop=True), Xg.reset_index(drop=True)], axis=1).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test, dom_train, dom_test = train_test_split(
        X, y, domains, test_size=test_size, stratify=y, random_state=random_state
    )

    # Save
    X_train.to_parquet(f"{outdir}/X_{antibiotic}_train.parquet", index=False)
    X_test.to_parquet(f"{outdir}/X_{antibiotic}_test.parquet", index=False)
    y_train.to_csv(f"{outdir}/y_{antibiotic}_train.csv", index=False, header=True)
    y_test.to_csv(f"{outdir}/y_{antibiotic}_test.csv", index=False, header=True)
    dom_train.to_csv(f"{outdir}/domains_{antibiotic}_train.csv", index=False, header=True)
    dom_test.to_csv(f"{outdir}/domains_{antibiotic}_test.csv", index=False, header=True)
    # Save feature names for UI consistency
    X.columns.to_series().to_csv(f"{outdir}/feature_columns.csv", index=False)


    print("Saved processed files in", outdir)
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--isolates", required=True, help="path to isolates.csv")
    p.add_argument("--card-map", required=False, help="gene->aro mapping CSV (optional)", default=None)
    p.add_argument("--antibiotic", required=True, help="antibiotic name (case-insensitive) to build target for, e.g., 'ciprofloxacin'")
    p.add_argument("--outdir", required=True, help="output directory (data/processed)")
    p.add_argument("--sample", type=int, default=None, help="sample size for dev")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()
    featurize(args.isolates, args.card_map, args.antibiotic, args.outdir, sample=args.sample, test_size=args.test_size)
