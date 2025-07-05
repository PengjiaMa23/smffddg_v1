import os
import requests
from Bio import SeqIO

# Input FASTA path
fasta_path = "/root/fssd/Thermal_stability/data/SMFFDDG_data/smffddg_wild.fasta"
# Output directory for PDB files
output_dir = "/root/fssd/Thermal_stability/data/dssp/dataset_pdb"
os.makedirs(output_dir, exist_ok=True)

# Iterate through the FASTA file to extract PDB IDs and download PDB files
for record in SeqIO.parse(fasta_path, "fasta"):
    pdb_id_full = record.id.upper()  # e.g., 1A43A
    pdb_id = pdb_id_full[:4]
    chain_id = pdb_id_full[4:] if len(pdb_id_full) > 4 else ''

    # Download PDB using the RCSB API
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id_full}.pdb")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, "w") as f:
                f.write(response.text)
            print(f"Download successful: {pdb_id_full} -> {output_path}")
        else:
            print(f"Download failed: {pdb_id_full}, HTTP status code {response.status_code}")
    except Exception as e:
        print(f"Exception: {pdb_id_full} -> {e}")