import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model.shared_bottom_mlp import SharedBottomMlpModel
from data.load_data_in_turn import EmotionFocusDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
print(f"Using device: {device}")

# 加载数据
csv_file = 'data/train.csv'  # 替换为您的 CSV 文件路径
dataset = EmotionFocusDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
input_dim = 58
shared_hidden_dim = 128
emotion_output_dim = len(pd.read_csv(csv_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_file)['if_focus'].unique())  # 专注度类别数

# 初始化模型、损失函数和优化器
model = SharedBottomMlpModel(input_dim, shared_hidden_dim, emotion_output_dim, focus_output_dim)
emotion_criterion = nn.CrossEntropyLoss()  # 表情任务的损失
focus_criterion = nn.CrossEntropyLoss()  # 专注度任务的损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 30
for epoch in range(epochs):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    total_emotion_loss = 0
    total_focus_loss = 0

    for features, emotion_labels, focus_labels in dataloader:
        features, emotion_labels, focus_labels = (features.to(device),
                                                  emotion_labels.to(device),
                                                  focus_labels.to(device)) # 数据转到 GPU
        # 前向传播
        emotion_pred, focus_pred = model(features)

        # 计算损失
        emotion_loss = emotion_criterion(emotion_pred, emotion_labels)
        focus_loss = focus_criterion(focus_pred, focus_labels)
        total_loss = emotion_loss + focus_loss  # 总损失

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_emotion_loss += emotion_loss.item()
        total_focus_loss += focus_loss.item()

    # 打印每个 epoch 的损失
    print(f"Epoch [{epoch+1}/{epochs}], Emotion Loss: {total_emotion_loss:.4f}, Focus Loss: {total_focus_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'pth/shared_bottom_mlp_model.pth')
print("模型已保存为 'shared_bottom_mlp_model.pth'")
