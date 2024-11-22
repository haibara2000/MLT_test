import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 自定义数据集类
class SingleTaskDataset(Dataset):
    def __init__(self, csv_file, target_column):
        data = pd.read_csv(csv_file)
        # 假设 1:58 列为特征，target_column 为标签
        self.features = torch.tensor(data.iloc[:, 1:59].values, dtype=torch.float32)
        self.labels = torch.tensor(data[target_column].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义单任务 MLP 模型
class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleTaskMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)
