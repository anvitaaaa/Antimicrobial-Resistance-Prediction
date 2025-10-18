#!/usr/bin/env python3
"""
ingest.py (fixed version)
- Robust CSV ingestion that handles inconsistent rows
- Can sample a subset for development
"""
import argparse
import pandas as pd

def sample_isolates(infile, outpath, nrows=None, sample=None):
    print(f"Reading file: {infile}")
    try:
        # Use robust parsing options
        reader = pd.read_csv(
            infile,
            chunksize=200000,
            low_memory=False,
            on_bad_lines='skip',    # skip problematic lines
            quotechar='"',
            escapechar='\\'
        )

        chunks = []
        for i, chunk in enumerate(reader):
            print(f"  -> Read chunk {i+1} ({len(chunk)} rows)")
            chunks.append(chunk)
            if sample and sum(len(c) for c in chunks) >= sample * 3:
                break

        df = pd.concat(chunks, ignore_index=True)
        if sample:
            df = df.sample(n=sample, random_state=42).reset_index(drop=True)
            print(f"Sampled {len(df)} rows from total {sum(len(c) for c in chunks)}")

        df.to_parquet(outpath, index=False)
        print(f"✅ Wrote {len(df)} rows to {outpath}")

    except Exception as e:
        print("❌ Error during ingestion:", e)
        print("Try verifying the CSV delimiter (maybe ';' instead of ',').")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to isolates.csv")
    p.add_argument("--out", required=True, help="parquet output path")
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--sample", type=int, default=None, help="random sample size for development")
    args = p.parse_args()
    sample_isolates(args.input, args.out, nrows=args.nrows, sample=args.sample)
