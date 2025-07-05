import h5py
import pandas as pd
import numpy as np
import os

# Configure paths
h5_path = "/root/fssd/Thermal_stability/get_final_fea.py/S669_final_features.h5"
csv_path = "/root/fssd/Thermal_stability/data/dssp/S669_with_dssp.csv"
output_dir = "/root/fssd/Thermal_stability/data/dssp/S669"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# 1. Read H5 file and filter forward mutation data
with h5py.File(h5_path, 'r') as f:
    ids = f["ids"][:]
    local_fea = f["local_fea"][:]
    global_fea = f["global_fea"][:]
    ddg = f["ddg"][:]

# Filter forward mutation data (IDs ending with "_d")
forward_mask = [id.decode().endswith("_d") for id in ids]
forward_ids = ids[forward_mask]
forward_local = local_fea[forward_mask]
forward_global = global_fea[forward_mask]
forward_ddg = ddg[forward_mask]

# Create matching IDs (remove "_d" suffix)
match_ids = [id.decode().rstrip("_d") for id in forward_ids]
print(f"Original forward mutation data count: {len(forward_ids)}")

# 2. Read CSV file and create mappings
df_csv = pd.read_csv(csv_path)
ss_map = {}
rasa_map = {}

for _, row in df_csv.iterrows():
    ss_map[row["id"]] = row["SS"]
    rasa_map[row["id"]] = row["RASA"]

# 3. Get SS and RASA for each mutation
ss_list = [ss_map.get(mid, "NA") for mid in match_ids]
rasa_list = [rasa_map.get(mid, -1) for mid in match_ids]

# 4. Create filtering masks
mask_ss_c = np.array([ss == "C" for ss in ss_list])
mask_ss_h = np.array([ss == "H" for ss in ss_list])
mask_ss_e = np.array([ss == "E" for ss in ss_list])
mask_rasa_lt01 = np.array([rasa < 0.1 and rasa >= 0 for rasa in rasa_list])
mask_rasa_gt05 = np.array([rasa > 0.5 and rasa <= 1 for rasa in rasa_list])

# 5. Filter data and save
def save_filtered_data(mask, name):
    filtered_ids = forward_ids[mask]
    filtered_local = forward_local[mask]
    filtered_global = forward_global[mask]
    filtered_ddg = forward_ddg[mask]
    
    print(f"{name} data count: {len(filtered_ids)}")
    
    output_path = os.path.join(output_dir, f"S669_{name}.h5")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("ids", data=filtered_ids)
        f.create_dataset("local_fea", data=filtered_local)
        f.create_dataset("global_fea", data=filtered_global)
        f.create_dataset("ddg", data=filtered_ddg)
    
    return len(filtered_ids)

# Save filtered data
c_count = save_filtered_data(mask_ss_c, "SS_C")
h_count = save_filtered_data(mask_ss_h, "SS_H")
e_count = save_filtered_data(mask_ss_e, "SS_E")
rasa_lt01_count = save_filtered_data(mask_rasa_lt01, "RASA_lt0.1")
rasa_gt05_count = save_filtered_data(mask_rasa_gt05, "RASA_gt0.5")

# 6. Save all forward mutation data
all_output_path = os.path.join(output_dir, "S669_all_forward.h5")
with h5py.File(all_output_path, "w") as f:
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset("ids", data=np.array(forward_ids, dtype=object), dtype=dt)
    f.create_dataset("local_fea", data=forward_local)
    f.create_dataset("global_fea", data=forward_global)
    f.create_dataset("ddg", data=forward_ddg)
print(f"All forward mutation data count: {len(forward_ids)}")

# 7. Print results
print("\nFiltering results summary:")
print(f"Original forward mutation total: {len(forward_ids)}")
print(f"SS=C (Random Coil): {c_count} ({c_count/len(forward_ids)*100:.1f}%)")
print(f"SS=H (Alpha Helix): {h_count} ({h_count/len(forward_ids)*100:.1f}%)")
print(f"SS=E (Beta Sheet): {e_count} ({e_count/len(forward_ids)*100:.1f}%)")
print(f"RASA<0.1 (Hydrophobic Core): {rasa_lt01_count} ({rasa_lt01_count/len(forward_ids)*100:.1f}%)")
print(f"RASA>0.5 (Highly Exposed): {rasa_gt05_count} ({rasa_gt05_count/len(forward_ids)*100:.1f}%)")