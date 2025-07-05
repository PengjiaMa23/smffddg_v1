import pandas as pd
import argparse
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a FASTA file from a processed CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv(args.input_csv)

    # Extract the base name of the CSV file without extension
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]

    # Define the output FASTA file path
    output_fasta = os.path.join(os.path.dirname(args.input_csv), f"{base_name}.fasta")

    # Track written wild-type PDB IDs
    written_wt_ids = set()

    with open(output_fasta, "w") as fasta_file:
        for _, row in df.iterrows():
            full_id = row["id"]
            wt_seq = row["wt_seq"]
            mut_seq = row["mut_seq"]

            # Extract the wild-type ID: the part before the "_" in the ID
            wt_id = full_id.split("_")[0]

            # Write the wild-type sequence (only once)
            if wt_id not in written_wt_ids:
                fasta_file.write(f">{wt_id}\n{wt_seq}\n")
                written_wt_ids.add(wt_id)

            # Write the mutant sequence
            fasta_file.write(f">{full_id}\n{mut_seq}\n")

    print(f"Fasta file generated: {output_fasta}")

if __name__ == "__main__":
    main()