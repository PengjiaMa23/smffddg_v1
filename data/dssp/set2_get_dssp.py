import os
import subprocess

pdb_dir = "/root/fssd/Thermal_stability/data/dssp/dataset_pdb"
dssp_dir = "/root/fssd/Thermal_stability/data/dssp/dataset_dssp"

# Create the output directory if it does not exist
os.makedirs(dssp_dir, exist_ok=True)

# Iterate through the PDB files
for filename in os.listdir(pdb_dir):
    if filename.lower().endswith(".pdb"):
        pdb_path = os.path.join(pdb_dir, filename)
        dssp_filename = os.path.splitext(filename)[0] + ".dssp"
        dssp_path = os.path.join(dssp_dir, dssp_filename)

        try:
            # Run the mkdssp command
            subprocess.run(["mkdssp", "-i", pdb_path, "-o", dssp_path], check=True)
            print(f"Successfully generated: {dssp_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate: {filename}, Error: {e}")