import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_features, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输入 (1, 58, 30)，输出 (16, 58, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (16, 29, 15)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出 (32, 29, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出 (32, 14, 7)
            nn.Flatten()  # 输出 (32 * 14 * 7 = 3136)
        )
        # shared_output_dim = 32 * 14 * 7

        # 计算共享层展平后的输出维度
        dummy_input = torch.zeros(1, 1, num_features, 30)  # 假设输入的形状为 (batch_size, channels, height, width)
        # 先通过共享层计算输出，再查看其展平后的大小
        shared_output_dim = self.conv(dummy_input).shape[1]  # 获取展平后的大小

        # 单任务分支
        self.task_branch = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        conv = self.conv(x)
        output = self.task_branch(conv)
        return output
