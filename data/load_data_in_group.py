import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 自定义数据集类
class EmotionFocusDataset(Dataset):
    # 初始化数据集，以window_size为窗口大小的一组
    def __init__(self, csv_file, window_size=30):
        # 加载 CSV 数据
        data = pd.read_csv(csv_file)

        # 动态获取特征列数：假设前 1 到倒数第 4 列为特征
        num_features = data.iloc[:, 1:-4].shape[1]

        # 提取特征和标签
        features = data.iloc[:, 1:-4].values
        emotion_labels = data['emotion'].values
        focus_labels = data['if_focus'].values

        # 分组封装
        self.features = []
        self.emotion_labels = []
        self.focus_labels = []

        for i in range(0, len(features), window_size):
            # 获取窗口大小的样本
            feature_block = features[i:i + window_size]
            emotion_block = emotion_labels[i:i + window_size]
            focus_block = focus_labels[i:i + window_size]

            # 如果不足一个窗口大小，跳过
            if len(feature_block) < window_size:
                continue

            # 封装成矩阵 (1, num_features, window_size)
            self.features.append(feature_block.reshape(1, num_features, window_size))

            # 找到频率最高的标签
            self.emotion_labels.append(np.bincount(emotion_block.astype(int)).argmax())  # 统计表情
            self.focus_labels.append(np.bincount(focus_block.astype(int)).argmax())  # 统计专注度

        # 转换为 Tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.emotion_labels = torch.tensor(self.emotion_labels, dtype=torch.long)
        self.focus_labels = torch.tensor(self.focus_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.emotion_labels[idx], self.focus_labels[idx]
