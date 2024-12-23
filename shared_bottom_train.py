import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model.shared_bottom_mlp import SharedBottomMlpModel
from data.load_data_in_turn import EmotionFocusDataset
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU
print(f"Using device: {device}")

# 加载数据
# csv_file = 'data/normalized_train.csv'
csv_file = 'data/train.csv'
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
    emotion_preds, emotion_labels_list = [], []
    focus_preds, focus_labels_list = [], []

    for features, emotion_labels, focus_labels in dataloader:
        features, emotion_labels, focus_labels = (features.to(device),
                                                  emotion_labels.to(device),
                                                  focus_labels.to(device))  # 数据转到 GPU
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

        # 收集预测和标签用于计算准确率
        emotion_preds.extend(torch.argmax(emotion_pred, dim=1).cpu().numpy())
        emotion_labels_list.extend(emotion_labels.cpu().numpy())

        focus_preds.extend(torch.argmax(focus_pred, dim=1).cpu().numpy())
        focus_labels_list.extend(focus_labels.cpu().numpy())

    # 计算每个任务的准确率
    emotion_acc = accuracy_score(emotion_labels_list, emotion_preds)
    focus_acc = accuracy_score(focus_labels_list, focus_preds)

    # 打印每个 epoch 的损失和准确率
    print(f"Epoch [{epoch+1}/{epochs}], Emotion Loss: {total_emotion_loss:.4f}, Focus Loss: {total_focus_loss:.4f}")
    print(f"Epoch [{epoch+1}/{epochs}], Emotion Accuracy: {emotion_acc:.4f}, Focus Accuracy: {focus_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), 'pth_origin/shared_bottom_mlp_model.pth')
# torch.save(model.state_dict(), 'pth_normalized/shared_bottom_mlp_model.pth')
print("模型已保存为 'shared_bottom_mlp_model.pth'")
