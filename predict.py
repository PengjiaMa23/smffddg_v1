import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import pandas as pd
import argparse

# ==================== Model Architecture Definitions ====================
class HSwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 3) / 6, 0, 1)

class G2S_FeatureLearningModel(nn.Module):
    def __init__(self, input_size=25):
        super(G2S_FeatureLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.hswish = HSwishActivation()
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hswish(x)
        x = self.fc2(x)
        return x

class ESM_FeatureLearningModel(nn.Module):
    def __init__(self, input_size=1280):
        super(ESM_FeatureLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc3 = nn.Linear(1024, 128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        return x

class DDGPredictionModel(nn.Module):
    def __init__(self):
        super(DDGPredictionModel, self).__init__()
        self.G2S_feature_learning_model = G2S_FeatureLearningModel()
        self.ESM_feature_learning_model = ESM_FeatureLearningModel()

        self.G2S_weight = nn.Parameter(torch.randn(128))
        self.ESM_weight = nn.Parameter(torch.randn(128))

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, local_input, global_input):
        g2s_output = self.G2S_feature_learning_model(local_input)
        esm_output = self.ESM_feature_learning_model(global_input)
        g2s_weighted = g2s_output * self.G2S_weight
        esm_weighted = esm_output * self.ESM_weight
        x = g2s_weighted + esm_weighted
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze()

# ==================== Prediction Function ====================
def predict_and_save_results(h5_file_path, model_dir, output_csv_path, mutation_type="d"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature data from H5 file
    print("Loading feature data from H5 file...")
    with h5py.File(h5_file_path, "r") as f:
        ids = f["ids"][:].astype(str)
        local_fea_all = torch.tensor(f["local_fea"][:], dtype=torch.float32)
        global_fea_all = torch.tensor(f["global_fea"][:], dtype=torch.float32)

    # Move data to device
    local_fea_all = local_fea_all.to(device)
    global_fea_all = global_fea_all.to(device)

    # Initialize results list
    results = []

    # Perform model inference
    for fold_num in range(5):
        model_path = os.path.join(model_dir, f"fold_{fold_num}_best_model.pth")
        model = DDGPredictionModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            # Perform inference
            preds = model(local_fea_all, global_fea_all).cpu().numpy()
            results.append(preds)

    # Calculate ensemble prediction values
    ensemble_preds = np.mean(results, axis=0)

    # Filter forward or reverse mutations based on mutation_type
    if mutation_type == "both":
        mask = np.ones_like(ids, dtype=bool)  # Select all mutations
    else:
        mask = np.array([id_.endswith(mutation_type) for id_ in ids])
    filtered_ids = ids[mask]
    filtered_preds = ensemble_preds[mask]

    # Print prediction results
    for idx, (id_, pred) in enumerate(zip(filtered_ids, filtered_preds)):
        print(f"ID: {id_}, Predicted DDG: {pred:.4f}")

    # Save results to CSV file
    print(f"Saving prediction results to {output_csv_path}...")
    df = pd.DataFrame({
        "id": filtered_ids,
        "predicted_ddg": filtered_preds
    })
    df.to_csv(output_csv_path, index=False)
    print(f"Prediction results have been saved to {output_csv_path}")

# ==================== Execution ====================
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict DDG values using a trained model.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to the model directory.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., Myoglobin).")
    parser.add_argument("--mutation_type", type=str, default="d", choices=["d", "r", "both"], help="Type of mutation to predict (d for forward, r for reverse, both for all).")
    args = parser.parse_args()

    # Define paths
    h5_file_path = f"/root/fssd/SMFFDDG/fea_process/h5_data/{args.dataset}_final_features.h5"
    output_csv_path = f"/root/fssd/SMFFDDG/result/{args.dataset}.csv"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Call the prediction function
    predict_and_save_results(h5_file_path, args.model_dir, output_csv_path, args.mutation_type)