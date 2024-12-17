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
        # 1:59列为特征，最后两列分别为表情 (emotion) 和专注度 (focus)
        self.features = torch.tensor(data.iloc[:, 1:59].values, dtype=torch.float32)
        # self.features = torch.tensor(data.iloc[:, 1:45].values, dtype=torch.float32)
        self.emotion_labels = torch.tensor(data['emotion'].values, dtype=torch.long)  # 表情类别
        self.focus_labels = torch.tensor(data['if_focus'].values, dtype=torch.long)  # 专注度类别

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.emotion_labels[idx], self.focus_labels[idx]

# 将数据加载到内存中（适用于 XGBoost）
def prepare_data(loader):
    features_list, emotion_labels_list, focus_labels_list = [], [], []
    for features, emotion_labels, focus_labels in loader:
        features_list.append(features.numpy())
        emotion_labels_list.append(emotion_labels.numpy())
        focus_labels_list.append(focus_labels.numpy())
    features = np.vstack(features_list)
    emotion_labels = np.concatenate(emotion_labels_list)
    focus_labels = np.concatenate(focus_labels_list)
    return features, emotion_labels, focus_labels