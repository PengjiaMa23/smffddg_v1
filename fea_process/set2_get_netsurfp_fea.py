import os
import csv
import json
import pandas as pd
from tqdm import tqdm
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract NetSurfP features from CSV files.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--netsurfp_dir", type=str, required=True, help="Path to the NetSurfP output directory.")
    args = parser.parse_args()

    # Define the output directory and JSON file path
    output_dir = "./fea_process/netsurfp_fea"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
    output_json = os.path.join(output_dir, f"{base_name}_netsurfp.json")

    # Define the target fields to extract
    target_fields = [
        "rsa", "asa", "q3", "p[q3_H]", "p[q3_E]", "p[q3_C]", "q8",
        "p[q8_G]", "p[q8_H]", "p[q8_I]", "p[q8_B]", "p[q8_E]",
        "p[q8_S]", "p[q8_T]", "p[q8_C]", "phi", "psi", "disorder"
    ]

    # Read the input CSV file
    df = pd.read_csv(args.input_csv)
    all_results = {}

    # Function to extract features from CSV
    def extract_feature_from_csv(file_path, expected_aa, protein_json_id, pos):
        df_struct = pd.read_csv(file_path, skipinitialspace=True)  # Remove leading spaces in column names
        df_struct = df_struct[df_struct["n"] == pos]
        if df_struct.empty:
            print(f"Position {pos} not found in {file_path}")
            return

        row = df_struct.iloc[0]
        aa_in_seq = row['seq']
        if aa_in_seq != expected_aa:
            print(f"Residue mismatch at {protein_json_id}: expected {expected_aa}, got {aa_in_seq}")
            return

        all_results[protein_json_id] = {field: row[field] for field in target_fields}

    # Function to find NetSurfP file
    def find_netsurfp_file(protein_id):
        for dir_name in os.listdir(args.netsurfp_dir):
            dir_path = os.path.join(args.netsurfp_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue

            parts = dir_name.split("_", 1)
            if len(parts) < 2:
                continue

            if parts[1] == protein_id:
                for f in os.listdir(dir_path):
                    if f.endswith(".csv"):
                        file_parts = f.split("_", 1)
                        if len(file_parts) > 1 and file_parts[1].startswith(protein_id):
                            return os.path.join(dir_path, f)
        return None

    # Main loop with tqdm progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting NetSurfP features"):
        mut_id = row['id']
        wt_id = mut_id.split('_')[0]
        pos = int(row['pos_seq'])
        mut_info = row['mut_info_Seq']
        wt_aa = mut_info[0]
        mut_aa = mut_info[-1]

        # Wild-type
        wt_json_id = f"{mut_id}_wild"
        wt_path = find_netsurfp_file(wt_id)
        if wt_path:
            extract_feature_from_csv(wt_path, wt_aa, wt_json_id, pos)
        else:
            print(f"Wildtype file not found for {wt_id}")

        # Mutant
        mut_json_id = mut_id
        mut_path = find_netsurfp_file(mut_id)
        if mut_path:
            extract_feature_from_csv(mut_path, mut_aa, mut_json_id, pos)
        else:
            print(f"Mutant file not found for {mut_id}")

    # Save the results to JSON
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nStructural features saved to {output_json}")

if __name__ == "__main__":
    main()