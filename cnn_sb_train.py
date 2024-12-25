import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model.shard_bottom_cnn import SharedBottomCNNModel
from data.load_data_in_group import EmotionFocusDataset
import random

# 设置随机种子函数
def set_random_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果有多个GPU）
    # torch.backends.cudnn.deterministic = True  # 让CUDNN算法是确定性的
    # torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动选择算法

# 设置随机种子
seed = 42  # 你可以选择任何整数作为种子
set_random_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
print(f"Using device: {device}")
# 加载数据
csv_file = 'data/history/normalized_train.csv'  # 替换为您的 CSV 文件路径
dataset = EmotionFocusDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
num_features = pd.read_csv(csv_file).iloc[:, 1:-4].shape[1]  # 特征数
emotion_output_dim = len(pd.read_csv(csv_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_file)['if_focus'].unique())  # 专注度类别数

# 初始化模型、损失函数和优化器
model = SharedBottomCNNModel(num_features, emotion_output_dim, focus_output_dim)
emotion_criterion = nn.CrossEntropyLoss()  # 表情任务损失
focus_criterion = nn.CrossEntropyLoss()  # 专注度任务损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 30
for epoch in range(epochs):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    total_emotion_loss = 0
    total_focus_loss = 0
    correct_emotion = 0
    correct_focus = 0
    total_samples = 0

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

        # 计算准确率
        _, predicted_emotion = torch.max(emotion_pred, 1)
        _, predicted_focus = torch.max(focus_pred, 1)
        correct_emotion += (predicted_emotion == emotion_labels).sum().item()
        correct_focus += (predicted_focus == focus_labels).sum().item()
        total_samples += emotion_labels.size(0)

    # 计算准确率
    emotion_accuracy = correct_emotion / total_samples
    focus_accuracy = correct_focus / total_samples

    # 打印每个 epoch 的损失和准确率
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Emotion Loss: {total_emotion_loss:.4f}, Focus Loss: {total_focus_loss:.4f}, "
          f"Emotion Accuracy: {emotion_accuracy:.4f}, Focus Accuracy: {focus_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'data/history/pth_normalized/shared_bottom_cnn_model.pth')
print("模型已保存为 'shared_bottom_cnn_model.pth'")
