import pandas as pd
import os

# Define the MAX_ASA dictionary
maxASA = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0,
    'C': 148.0, 'Q': 214.0, 'E': 214.0, 'G': 97.0,
    'H': 216.0, 'I': 195.0, 'L': 191.0, 'K': 230.0,
    'M': 203.0, 'F': 228.0, 'P': 154.0, 'S': 143.0,
    'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0
}

# Map secondary structure codes
def map_ss(dssp_code):
    if dssp_code in ['H', 'G', 'I']:
        return 'H'
    elif dssp_code in ['E', 'B']:
        return 'E'
    else:
        return 'C'

# Input file path
input_path = "/root/fssd/Thermal_stability/data/SMFFDDG_data/S669.csv"
df = pd.read_csv(input_path)

# Derived columns
df['pdb_id'] = df['id'].str.split('_').str[0]
df['wt_aa'] = df['mut_info_PDB'].str[0]
df['mut_resnum'] = df['mut_info_PDB'].str[1:-1].astype(int)

# Initialize result columns
df['SS'] = None
df['RASA'] = None

# DSSP directory
dssp_dir = "/root/fssd/Thermal_stability/data/dssp/dataset_dssp"

# Process by PDB file
for pdb_id, group in df.groupby('pdb_id'):
    dssp_path = os.path.join(dssp_dir, f"{pdb_id}.dssp")
    
    if not os.path.exists(dssp_path):
        print(f"DSSP file not found: {dssp_path}")
        continue

    with open(dssp_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the DSSP data
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#  RESIDUE'):
            start_idx = i + 1
            break
    residue_lines = lines[start_idx:]

    # Build DSSP index (chain ID, RESNUM) → line information
    dssp_index = {}
    for line in residue_lines:
        if len(line) < 30:
            continue
        try:
            resnum = int(line[5:10].strip())
            chain = line[11]
            dssp_index[(chain, resnum)] = line
        except:
            continue

    # Process each mutation record
    for idx, row in group.iterrows():
        mut_resnum = row['mut_resnum']
        chain = row['id'].split('_')[0][-1]
        wt_aa = row['wt_aa']

        key = (chain, mut_resnum)
        if key not in dssp_index:
            print(f"⚠️ Residue not found: {key} in {pdb_id}")
            continue

        line = dssp_index[key]
        dssp_aa = line[13]
        ss_code = line[16]
        acc_str = line[35:38].strip()
        if dssp_aa == 'd' or dssp_aa == 'b' or dssp_aa == 'a':  # Map oxidized CYS to 'C'
            dssp_aa = 'C'
        # Amino acid validation
        if dssp_aa != wt_aa:
            print(f"⚠️ Amino acid mismatch: {pdb_id} {key} Expected: {wt_aa} Actual: {dssp_aa}")
            continue

        # Relative ASA calculation
        try:
            asa = float(acc_str)
            rasa = round(asa / maxASA.get(dssp_aa, 999), 3)
        except:
            rasa = None

        ss_mapped = map_ss(ss_code)
        df.at[idx, 'SS'] = ss_mapped
        df.at[idx, 'RASA'] = rasa

# Drop unnecessary columns
df = df.drop(columns=['wt_seq', 'mut_seq'])

# Save the results
output_path = "/root/fssd/Thermal_stability/data/dssp/S669_with_dssp.csv"
df.to_csv(output_path, index=False)

print(f"\nProcessing complete, results saved to: {output_path}")
print(f"Added {df['SS'].notna().sum()} valid SS/RASA data out of {len(df)} records.")