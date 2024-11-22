import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 定义 shared-bottom 模型
class SharedBottomModel(nn.Module):
    def __init__(self, input_dim, shared_hidden_dim, emotion_output_dim, focus_output_dim):
        super(SharedBottomModel, self).__init__()
        # 共享底层网络
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_hidden_dim, shared_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # 表情任务分支
        self.emotion_branch = nn.Sequential(
            nn.Linear(shared_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_output_dim)
        )
        # 专注度任务分支
        self.focus_branch = nn.Sequential(
            nn.Linear(shared_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, focus_output_dim)
        )

    def forward(self, x):
        # 底层共享表示
        shared_output = self.shared_bottom(x)
        # 表情预测
        emotion_output = self.emotion_branch(shared_output)
        # 专注度预测
        focus_output = self.focus_branch(shared_output)
        return emotion_output, focus_output