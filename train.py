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

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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


# ===================== 定义数据集类 =====================
class ProteinDataset(Dataset):
    def __init__(self, local_features, global_features, ddg_values, fold_indices, fold_num, mode='train'):
        """
        参数:
        local_features: 局部特征数组 (n_samples, 25)
        global_features: 全局特征数组 (n_samples, 1280)
        ddg_values: DDG值数组 (n_samples,)
        fold_indices: fold分配数组 (n_samples,)
        fold_num: 当前使用的fold编号
        mode: 'train' 或 'val'，决定是训练集还是验证集
        """
        if mode == 'train':
            # 训练集: 所有非当前fold的数据
            mask = (fold_indices != fold_num)
        else:  # mode == 'val'
            # 验证集: 当前fold的数据
            mask = (fold_indices == fold_num)
        
        self.local_features = local_features[mask]
        self.global_features = global_features[mask]
        self.ddg_values = ddg_values[mask]
        
        print(f"创建 {mode} 数据集: {len(self.local_features)} 个样本 (fold {fold_num})")

    def __len__(self):
        return len(self.local_features)

    def __getitem__(self, idx):
        return {
            'local': torch.tensor(self.local_features[idx], dtype=torch.float32),
            'global': torch.tensor(self.global_features[idx], dtype=torch.float32),
            'ddg': torch.tensor(self.ddg_values[idx], dtype=torch.float32)
        }

# ===================== 定义模型架构 =====================
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
        
        # 定义两个128维的可学习权重向量
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
        
        # 最终全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, local_input, global_input):
        g2s_output = self.G2S_feature_learning_model(local_input)
        esm_output = self.ESM_feature_learning_model(global_input)
        
        # 使用按元素相乘的方式进行加权
        g2s_weighted = g2s_output * self.G2S_weight
        esm_weighted = esm_output * self.ESM_weight
        
        # 将加权后的特征相加
        x = g2s_weighted + esm_weighted
        x = x.unsqueeze(1)  # 添加通道维度: (batch, 1, 128)
        
        # 通过CNN层
        x = self.cnn_layers(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        return x.squeeze()  # 移除多余的维度

# ===================== 训练和评估函数 =====================
def calculate_metrics(y_true, y_pred):
    """计算评估指标: RMSE, PCC, ACC"""
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # PCC (皮尔逊相关系数)
    pcc = np.corrcoef(y_true, y_pred)[0, 1]
    
    # ACC (二值化准确率)
    true_binary = (y_true > 0).astype(int)
    pred_binary = (y_pred > 0).astype(int)
    acc = accuracy_score(true_binary, pred_binary)
    
    return rmse, pcc, acc

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        local_features = batch['local'].to(device)
        global_features = batch['global'].to(device)
        targets = batch['ddg'].to(device)
        
        # 前向传播
        outputs = model(local_features, global_features)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失和预测
        total_loss += loss.item() * local_features.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # 计算平均损失和指标
    avg_loss = total_loss / len(dataloader.dataset)
    rmse, pcc, acc = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    return avg_loss, rmse, pcc, acc

def evaluate(model, dataloader, criterion, device):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            local_features = batch['local'].to(device)
            global_features = batch['global'].to(device)
            targets = batch['ddg'].to(device)
            
            # 前向传播
            outputs = model(local_features, global_features)
            loss = criterion(outputs, targets)
            
            # 记录损失和预测
            total_loss += loss.item() * local_features.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算平均损失和指标
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


# ===================== 主训练流程 =====================
def main():
    # 加载数据
    h5_path = "/root/fssd/Thermal_stability/get_final_fea.py/S2648_final_features.h5"
    model_save_dir = "/root/fssd/Thermal_stability/model/20250609"
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
        print(f"\n{'='*30}\n开始处理 Fold {fold_num}\n{'='*30}")
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
                print(f"尝试超参数: lr={lr}, weight_decay={weight_decay}")
                val_pcc, epoch, metrics, state_dict = train_fold(fold_num, train_loader, val_loader, hyper)

                print(f"→ 最佳Epoch {epoch} | Val PCC: {metrics[1]:.4f} | RMSE: {metrics[0]:.4f} | ACC: {metrics[2]:.4f}")

                if val_pcc > best_pcc:
                    best_pcc = val_pcc
                    best_metrics = metrics
                    best_hyper = hyper
                    best_state = state_dict

        best_hyperparams[fold_num] = best_hyper
        print(f"\n===== Fold {fold_num} 训练完成 =====")
        print(f"最优超参数: lr={best_hyper['lr']}, weight_decay={best_hyper['weight_decay']}, PCC={best_pcc:.4f}")

    # 保存最优超参数
    summary_path = os.path.join(model_save_dir, "hyperparameters_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("五折交叉验证最优超参数:\n")
        for fold_num, hyper in best_hyperparams.items():
            f.write(f"Fold {fold_num}: lr={hyper['lr']}, weight_decay={hyper['weight_decay']}\n")

    print(f"\n最优超参数已保存至: {summary_path}")

    # 最后阶段：每个fold使用最佳超参数训练一次并保存模型
    print("\n\n========== 使用最优超参数重新训练并保存模型 ==========")
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
        print(f"保存 Fold {fold_num} 模型到: {save_path}")

if __name__ == "__main__":
    main()