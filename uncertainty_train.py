import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model.shard_bottom_cnn import SharedBottomCNNModel
from data.load_data_in_group import EmotionFocusDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
print(f"Using device: {device}")
# 加载数据
# csv_file = 'data/train.csv'
csv_file = 'data/balanced_train.csv'
dataset = EmotionFocusDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
emotion_output_dim = len(pd.read_csv(csv_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_file)['if_focus'].unique())  # 专注度类别数
# 初始化模型、损失函数和优化器
model = SharedBottomCNNModel(emotion_output_dim, focus_output_dim)
emotion_criterion = nn.CrossEntropyLoss()  # 表情任务损失
focus_criterion = nn.CrossEntropyLoss()  # 专注度任务损失

# 引入可训练的不确定性参数（log_sigma^2），用于调整权重
log_sigma_emotion = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))  # log(sigma^2) for emotion task
log_sigma_focus = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))  # log(sigma^2) for focus task

# 将 log_sigma 添加到优化器中
optimizer = optim.Adam(
    list(model.parameters()) + [log_sigma_emotion, log_sigma_focus],
    lr=0.001
)

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    total_emotion_loss = 0
    total_focus_loss = 0

    for features, emotion_labels, focus_labels in dataloader:
        features, emotion_labels, focus_labels = (features.to(device),
                                                  emotion_labels.to(device),
                                                  focus_labels.to(device))  # 数据转到 GPU
        # 前向传播
        emotion_pred, focus_pred = model(features)

        # 计算损失
        emotion_loss = emotion_criterion(emotion_pred, emotion_labels)
        focus_loss = focus_criterion(focus_pred, focus_labels)

        # 自适应权重的损失计算
        weighted_emotion_loss = emotion_loss / (2 * torch.exp(log_sigma_emotion)) + log_sigma_emotion / 2
        weighted_focus_loss = focus_loss / (2 * torch.exp(log_sigma_focus)) + log_sigma_focus / 2
        total_loss = weighted_emotion_loss + weighted_focus_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_emotion_loss += emotion_loss.item()
        total_focus_loss += focus_loss.item()

    # 打印每个 epoch 的损失和权重
    print(f"Epoch [{epoch+1}/{epochs}], Emotion Loss: {total_emotion_loss:.4f}, "
          f"Focus Loss: {total_focus_loss:.4f}, "
          f"log_sigma_emotion: {log_sigma_emotion.item():.4f}, "
          f"log_sigma_focus: {log_sigma_focus.item():.4f}")

# 保存模型
torch.save(model.state_dict(), 'data/history/pth_normalized/shared_bottom_cnn_uncertainty_model1.pth')
torch.save({'log_sigma_emotion': log_sigma_emotion, 'log_sigma_focus': log_sigma_focus},
           'pth_no_normalize/loss_weights.pth')
print("模型及损失权重已保存")
