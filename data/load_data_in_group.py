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
        # 假设前 58 列为特征，最后两列分别为 emotion 和 focus
        features = data.iloc[:, 1:59].values
        # features = data.iloc[:, 1:45].values   # 44
        emotion_labels = data['emotion'].values
        focus_labels = data['if_focus'].values

        # 分组封装
        self.features = []
        self.emotion_labels = []
        self.focus_labels = []

        for i in range(0, len(features), 30):
            # 获取 30 个样本的特征和标签
            feature_block = features[i:i + 30]
            emotion_block = emotion_labels[i:i + 30]
            focus_block = focus_labels[i:i + 30]

            # 如果不足 30 个样本，跳过
            if len(feature_block) < 30:
                continue

            # 封装成矩阵
            self.features.append(feature_block.reshape(1, 58, 30))  # (1, 58, 30)
            # self.features.append(feature_block.reshape(1, 44, 30))  # (1, 44, 30)

            # 找到频率最高的标签
            self.emotion_labels.append(np.bincount(emotion_block.astype(int)).argmax())   # 统计表情
            self.focus_labels.append(np.bincount(focus_block.astype(int)).argmax())  # 统计专注度

        # 转换为 Tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.emotion_labels = torch.tensor(self.emotion_labels, dtype=torch.long)
        self.focus_labels = torch.tensor(self.focus_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.emotion_labels[idx], self.focus_labels[idx]
