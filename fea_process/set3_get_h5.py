import pandas as pd
import json
import numpy as np
import torch
import h5py
from tqdm import tqdm
import esm
import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and merge local and global features for mutations.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., Myoglobin).")
    args = parser.parse_args()

    # Define paths
    csv_path = f"./input/{args.dataset}.csv"
    netsurfp_path = f"./fea_process/netsurfp_fea/{args.dataset}_netsurfp.json"
    aap_path = f"./fea_process/parameters_list.json"
    final_output_h5 = f"./fea_process/h5_data/{args.dataset}.h5"

    # Load data
    df = pd.read_csv(csv_path)
    with open(netsurfp_path) as f:
        netsurfp_data = json.load(f)
    with open(aap_path) as f:
        aap_data = json.load(f)

    # Load ESM2 model
    def load_model():
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        return model, alphabet, device

    # Get ESM features
    def get_esm_fea(wild_seq, mut_seq, model, alphabet, device):
        rep_layer = model.num_layers
        batch_converter = alphabet.get_batch_converter()

        data = [("wt", wild_seq), ("mut", mut_seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[rep_layer])
        
        embeddings = results["representations"][rep_layer]  # shape: [2, L, 1280]
        wt_mean = embeddings[0].mean(dim=0).cpu()  # [1280]
        mut_mean = embeddings[1].mean(dim=0).cpu()

        diff_forward = mut_mean - wt_mean
        diff_reverse = wt_mean - mut_mean
        return diff_forward, diff_reverse

    # Extract local features
    def extract_local_features():
        id_list_fwd = []
        id_list_rev = []
        fea_list_fwd = []
        fea_list_rev = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            mut_id = row["id"]
            wt_id = mut_id + "_wild"
            mut_info = row["mut_info_Seq"]
            pos = int(row["pos_seq"])

            wt_aa = mut_info[0]
            mut_aa = mut_info[-1]

            if wt_id not in netsurfp_data or mut_id not in netsurfp_data:
                print(f"Skipping: {mut_id} missing netsurfp data")
                continue
            if wt_aa not in aap_data or mut_aa not in aap_data:
                print(f"Skipping: {mut_id} missing AAP data")
                continue

            def get_cos_sin(phi, psi):
                return [np.sin(np.radians(phi)), np.cos(np.radians(phi)),
                        np.sin(np.radians(psi)), np.cos(np.radians(psi))]

            def extract_feats(entry):
                d = netsurfp_data[entry]
                rsa = d["rsa"]
                disorder = d["disorder"]
                sin_phi, cos_phi, sin_psi, cos_psi = get_cos_sin(d["phi"], d["psi"])
                return [rsa, sin_phi, cos_phi, sin_psi, cos_psi, disorder]

            wt_feats = extract_feats(wt_id)
            mut_feats = extract_feats(mut_id)
            
            # Difference features
            delta_fwd = np.array(wt_feats) - np.array(mut_feats)
            delta_rev = np.array(mut_feats) - np.array(wt_feats)

            # AAP features
            aap_wt = np.array(aap_data[wt_aa])
            aap_mut = np.array(aap_data[mut_aa])
            delta_aap_fwd = aap_wt - aap_mut
            delta_aap_rev = aap_mut - aap_wt

            # Q3 probabilities
            def get_q3_probs(entry):
                d = netsurfp_data[entry]
                return [d["p[q3_H]"], d["p[q3_E]"], d["p[q3_C]"]]

            q3_wt = get_q3_probs(wt_id)
            q3_mut = get_q3_probs(mut_id)

            # Concatenate features
            local_fea_fwd = np.concatenate([delta_fwd, delta_aap_fwd, q3_wt, q3_mut], axis=0)
            local_fea_rev = np.concatenate([delta_rev, delta_aap_rev, q3_mut, q3_wt], axis=0)
            local_fea_fwd = torch.tensor(local_fea_fwd, dtype=torch.float32)
            local_fea_rev = torch.tensor(local_fea_rev, dtype=torch.float32)

            id_list_fwd.append(mut_id + "_d")  # Forward mutation
            id_list_rev.append(mut_id + "_r")  # Reverse mutation
            fea_list_fwd.append(local_fea_fwd)
            fea_list_rev.append(local_fea_rev)
        
        return id_list_fwd, fea_list_fwd, id_list_rev, fea_list_rev

    # Extract global features
    def extract_global_features(model, alphabet, device):
        id_list_fwd = []
        id_list_rev = []
        fea_list_fwd = []
        fea_list_rev = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            mut_id = row["id"]
            wt_seq = row["wt_seq"]
            mut_seq = row["mut_seq"]

            try:
                fwd_fea, rev_fea = get_esm_fea(wt_seq, mut_seq, model, alphabet, device)
            except Exception as e:
                print(f"Skipping {mut_id}: Error {e}")
                continue

            id_list_fwd.append(mut_id + "_d")  # Forward mutation
            id_list_rev.append(mut_id + "_r")  # Reverse mutation
            fea_list_fwd.append(fwd_fea)
            fea_list_rev.append(rev_fea)

        return id_list_fwd, fea_list_fwd, id_list_rev, fea_list_rev

    # Save final features
    def save_final_features(id_local_fwd, fea_local_fwd, id_global_fwd, fea_global_fwd,
                            id_local_rev, fea_local_rev, id_global_rev, fea_global_rev, output_path):
        assert list(id_local_fwd) == list(id_global_fwd), "Local and global features for forward mutations are not aligned!"
        assert list(id_local_rev) == list(id_global_rev), "Local and global features for reverse mutations are not aligned!"

        with h5py.File(output_path, "w") as f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("ids", data=np.array(id_local_fwd + id_local_rev, dtype=object), dtype=dt)
            f.create_dataset("local_fea", data=torch.stack(fea_local_fwd + fea_local_rev).numpy())
            f.create_dataset("global_fea", data=torch.stack(fea_global_fwd + fea_global_rev).numpy())

        print(f"Merging complete, total {len(id_local_fwd) + len(id_local_rev)} samples saved to: {output_path}")

    # Load ESM2 model
    model, alphabet, device = load_model()

    # Extract local features
    id_local_fwd, fea_local_fwd, id_local_rev, fea_local_rev = extract_local_features()

    # Extract global features
    id_global_fwd, fea_global_fwd, id_global_rev, fea_global_rev = extract_global_features(model, alphabet, device)

    # Save final merged features
    save_final_features(id_local_fwd, fea_local_fwd, id_global_fwd, fea_global_fwd,
                        id_local_rev, fea_local_rev, id_global_rev, fea_global_rev, final_output_h5)

if __name__ == "__main__":
    main()