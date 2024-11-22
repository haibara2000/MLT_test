import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 定义 shared-bottom MLP 模型
class SharedBottomMlpModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, emotion_output_dim, focus_output_dim):
        super(SharedBottomMlpModel, self).__init__()
        # 共享部分
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # 表情任务分支
        self.emotion_head = nn.Linear(hidden_dim // 2, emotion_output_dim)
        # 专注度任务分支
        self.focus_head = nn.Linear(hidden_dim // 2, focus_output_dim)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        emotion_output = self.emotion_head(shared_output)
        focus_output = self.focus_head(shared_output)
        return emotion_output, focus_output