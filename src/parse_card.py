#!/usr/bin/env python3
"""
parse_card.py
- Parse card.json to a simplified mapping: gene_name -> aro id/name -> antibiotic classes
"""
import json, csv, argparse, os

def parse_card(card_json_path, out_csv):
    with open(card_json_path, 'r', encoding='utf-8') as f:
        card = json.load(f)
    rows = []
    # card.json is an object keyed by model id; each value contains model_name, ontology references etc.
    for mid, v in card.items():
        if not isinstance(v, dict):
            # Skip if value is not a dict
            print(f"Skipping {mid} because it's not a dict")
            continue

        name = v.get('model_name') or v.get('name') or ''
        aro = v.get('ARO_accession') or v.get('aro_accession') or (v.get('aro', {}).get('id') if isinstance(v.get('aro', {}), dict) else None)
        aro_name = v.get('ARO_name') or (v.get('aro', {}).get('name') if isinstance(v.get('aro', {}), dict) else None)

        # antibiotic class: look into 'drug_class' or 'antibiotic' keys where available
        drug_class = v.get('drug_class') or v.get('antibiotic')
        rows.append((name, aro, aro_name, drug_class))

    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gene_name','aro_id','aro_name','antibiotic_class'])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote mapping with {len(rows)} entries to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True, help="path to card.json")
    p.add_argument("--out", required=True, help="out csv mapping")
    args = p.parse_args()
    parse_card(args.card, args.out)
