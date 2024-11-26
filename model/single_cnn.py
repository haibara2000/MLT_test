import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 定义共享层为 CNN 的 Shared-Bottom 模型
class CNNModel(nn.Module):
    def __init__(self, emotion_output_dim, focus_output_dim):
        super(CNNModel, self).__init__()
        # 共享 CNN 层
        self.shared_bottom = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输入 (1, 58, 30)，输出 (16, 58, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (16, 29, 15)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出 (32, 29, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (32, 14, 7)
            nn.Flatten()  # 输出 (32 * 14 * 7 = 3136)
        )
        shared_output_dim = 32 * 14 * 7  # Flatten 后的维度

        # 表情任务分支
        self.emotion_branch = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emotion_output_dim)
        )
        # 专注度任务分支
        self.focus_branch = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, focus_output_dim)
        )

    def forward(self, x):
        shared_output = self.shared_bottom(x)  # 共享层
        emotion_output = self.emotion_branch(shared_output)  # 表情任务输出
        focus_output = self.focus_branch(shared_output)  # 专注度任务输出
        return emotion_output, focus_output