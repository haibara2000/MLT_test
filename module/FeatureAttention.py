import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    def __init__(self, num_features):
        super(FeatureAttention, self).__init__()
        # 输入特征维度是 num_features
        self.fc1 = nn.Linear(num_features, 128)  # 隐藏层，输出维度为128
        self.fc2 = nn.Linear(128, num_features)  # 输出层，输出与输入特征数相同
        self.sigmoid = nn.Sigmoid()  # 输出权重在0和1之间

    def forward(self, x):
        # x 的形状是 (batch_size, num_channel, num_features, window_size)
        batch_size, num_channel, num_features, window_size = x.shape

        # 对 x 进行展平，使其形状变为 (batch_size * num_channel * window_size, num_features)
        x_flat = x.view(batch_size * num_channel * window_size,
                        num_features)  # (batch_size * num_channel * window_size, num_features)

        # 使用全连接层计算注意力权重
        attention_weights = self.fc1(x_flat)  # (batch_size * num_channel * window_size, 128)
        attention_weights = F.relu(attention_weights)  # ReLU激活
        attention_weights = self.fc2(attention_weights)  # (batch_size * num_channel * window_size, num_features)
        attention_weights = self.sigmoid(attention_weights)  # 权重值在0到1之间

        # 重新调整形状为 (batch_size, num_channel, num_features, window_size)
        attention_weights = attention_weights.view(batch_size, num_channel, num_features, window_size)

        return attention_weights
