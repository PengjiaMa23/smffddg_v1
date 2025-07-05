import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import time
from collections import defaultdict

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ===================== Define Dataset Class =====================
class ProteinDataset(Dataset):
    def __init__(self, local_features, global_features, ddg_values, fold_indices, fold_num, mode='train'):
        """
        Parameters:
        local_features: Local feature array (n_samples, 25)
        global_features: Global feature array (n_samples, 1280)
        ddg_values: DDG value array (n_samples,)
        fold_indices: Fold assignment array (n_samples,)
        fold_num: Current fold number
        mode: 'train' or 'val', determines whether it is the training set or validation set
        """
        if mode == 'train':
            # Training set: All data not in the current fold
            mask = (fold_indices != fold_num)
        else:  # mode == 'val'
            # Validation set: Data in the current fold
            mask = (fold_indices == fold_num)
        
        self.local_features = local_features[mask]
        self.global_features = global_features[mask]
        self.ddg_values = ddg_values[mask]
        
        print(f"Creating {mode} dataset: {len(self.local_features)} samples (fold {fold_num})")

    def __len__(self):
        return len(self.local_features)

    def __getitem__(self, idx):
        return {
            'local': torch.tensor(self.local_features[idx], dtype=torch.float32),
            'global': torch.tensor(self.global_features[idx], dtype=torch.float32),
            'ddg': torch.tensor(self.ddg_values[idx], dtype=torch.float32)
        }

# ===================== Define Model Architecture =====================
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
        
        # Define two learnable weight vectors of size 128
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
        
        # Final fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, local_input, global_input):
        g2s_output = self.G2S_feature_learning_model(local_input)
        esm_output = self.ESM_feature_learning_model(global_input)
        
        # Weight the features element-wise
        g2s_weighted = g2s_output * self.G2S_weight
        esm_weighted = esm_output * self.ESM_weight
        
        # Sum the weighted features
        x = g2s_weighted + esm_weighted
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, 128)
        
        # Pass through CNN layers
        x = self.cnn_layers(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        return x.squeeze()  # Remove extra dimensions

# ===================== Training and Evaluation Functions =====================
def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics: RMSE, PCC, ACC"""
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # PCC (Pearson correlation coefficient)
    pcc = np.corrcoef(y_true, y_pred)[0, 1]
    
    # ACC (binary accuracy)
    true_binary = (y_true > 0).astype(int)
    pred_binary = (y_pred > 0).astype(int)
    acc = accuracy_score(true_binary, pred_binary)
    
    return rmse, pcc, acc

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        local_features = batch['local'].to(device)
        global_features = batch['global'].to(device)
        targets = batch['ddg'].to(device)
        
        # Forward pass
        outputs = model(local_features, global_features)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss and predictions
        total_loss += loss.item() * local_features.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate average loss and metrics
    avg_loss = total_loss / len(dataloader.dataset)
    rmse, pcc, acc = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, rmse, pcc, acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            local_features = batch['local'].to(device)
            global_features = batch['global'].to(device)
            targets = batch['ddg'].to(device)
            
            # Forward pass
            outputs = model(local_features, global_features)
            loss = criterion(outputs, targets)
            
            # Record loss and predictions
            total_loss += loss.item() * local_features.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average loss and metrics
    avg_loss = total_loss / len(dataloader.dataset)
    rmse, pcc, acc = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, rmse, pcc, acc

def train_fold(fold_num, train_loader, val_loader, hyperparams):
    model = DDGPredictionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=hyperparams['lr'], 
                          weight_decay=hyperparams['weight_decay'])

    best_val_pcc = -float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, 201):
        train_loss, train_rmse, train_pcc, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_pcc, val_acc = evaluate(
            model, val_loader, criterion, device)
        
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            best_metrics = (val_rmse, val_pcc, val_acc)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                break

    return best_val_pcc, best_epoch, best_metrics, best_model_state


# ===================== Main Training Process =====================
def main():
    # Load data
    h5_path = "/root/fssd/Thermal_stability/get_final_fea.py/S2648_final_features.h5"
    model_save_dir = "/root/fssd/Thermal_stability/model/"
    os.makedirs(model_save_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        local_fea = f["local_fea"][:]
        global_fea = f["global_fea"][:]
        ddg = f["ddg"][:]
        fold = f["fold"][:]

    hyperparam_grid = {
        'lr': [1e-2, 1e-3, 1e-4],
        'weight_decay': [1e-2, 1e-3, 1e-4, 0]
    }

    best_hyperparams = {}

    for fold_num in range(5):
        print(f"\n{'='*30}\nStarting Fold {fold_num}\n{'='*30}")
        train_dataset = ProteinDataset(local_fea, global_fea, ddg, fold, fold_num, 'train')
        val_dataset = ProteinDataset(local_fea, global_fea, ddg, fold, fold_num, 'val')
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        best_pcc = -float('inf')
        best_metrics = None
        best_hyper = None
        best_state = None

        for lr in hyperparam_grid['lr']:
            for weight_decay in hyperparam_grid['weight_decay']:
                hyper = {'lr': lr, 'weight_decay': weight_decay}
                print(f"Trying hyperparameters: lr={lr}, weight_decay={weight_decay}")
                val_pcc, epoch, metrics, state_dict = train_fold(fold_num, train_loader, val_loader, hyper)

                print(f"→ Best Epoch {epoch} | Val PCC: {metrics[1]:.4f} | RMSE: {metrics[0]:.4f} | ACC: {metrics[2]:.4f}")

                if val_pcc > best_pcc:
                    best_pcc = val_pcc
                    best_metrics = metrics
                    best_hyper = hyper
                    best_state = state_dict

        best_hyperparams[fold_num] = best_hyper
        print(f"\n===== Fold {fold_num} Training Completed =====")
        print(f"Best hyperparameters: lr={best_hyper['lr']}, weight_decay={best_hyper['weight_decay']}, PCC={best_pcc:.4f}")

    # Save best hyperparameters
    summary_path = os.path.join(model_save_dir, "hyperparameters_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Best hyperparameters for five-fold cross-validation:\n")
        for fold_num, hyper in best_hyperparams.items():
            f.write(f"Fold {fold_num}: lr={hyper['lr']}, weight_decay={hyper['weight_decay']}\n")

    print(f"\nBest hyperparameters saved to: {summary_path}")

    # Final stage: Train each fold once using the best hyperparameters and save the model
    print("\n\n========== Re-training with Best Hyperparameters and Saving Models ==========")
    for fold_num in range(5):
        print(f"\n-- Fold {fold_num} --")
        train_dataset = ProteinDataset(local_fea, global_fea, ddg, fold, fold_num, 'train')
        val_dataset = ProteinDataset(local_fea, global_fea, ddg, fold, fold_num, 'val')
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        hyper = best_hyperparams[fold_num]
        _, best_epoch, _, best_model_state = train_fold(fold_num, train_loader, val_loader, hyper)

        model = DDGPredictionModel().to(device)
        model.load_state_dict(best_model_state)
        save_path = os.path.join(model_save_dir, f"fold_{fold_num}_best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved Fold {fold_num} model to: {save_path}")

if __name__ == "__main__":
    main()