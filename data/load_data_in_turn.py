import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 自定义数据集类
class EmotionFocusDataset(Dataset):
    def __init__(self, csv_file):
        # 加载 CSV 数据
        data = pd.read_csv(csv_file)
        # 假设前 14 列为特征，最后两列分别为表情 (emotion) 和专注度 (focus)
        self.features = torch.tensor(data.iloc[:, 1:59].values, dtype=torch.float32)
        self.emotion_labels = torch.tensor(data['emotion'].values, dtype=torch.long)  # 表情类别
        self.focus_labels = torch.tensor(data['if_focus'].values, dtype=torch.long)  # 专注度类别

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.emotion_labels[idx], self.focus_labels[idx]