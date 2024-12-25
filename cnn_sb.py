import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from data.load_data_in_group import EmotionFocusDataset
from model.mmoe_cnn import MMoECNNModel  # MMoE模型
from model.shard_bottom_cnn import SharedBottomCNNModel  # SharedBottomCNN模型
import pandas as pd
import random
import numpy as np
import torch.nn.functional as F

# 设置随机种子函数
def set_random_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果有多个GPU）

# 设置随机种子
seed = 42  # 你可以选择任何整数作为种子
set_random_seed(seed)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据加载
csv_train_file = 'data/normalized_train.csv'
train_dataset = EmotionFocusDataset(csv_train_file)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

csv_test_file = 'data/normalized_test.csv'
test_dataset = EmotionFocusDataset(csv_test_file)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型参数
num_features = pd.read_csv(csv_train_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(pd.read_csv(csv_train_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_train_file)['if_focus'].unique())  # 专注度类别数

# 选择模型：可以切换 `MMoECNNModel` 或 `SharedBottomCNNModel`
# 使用 MMoECNNModel
# model = MMoECNNModel(num_features, emotion_output_dim, focus_output_dim, num_experts=6).to(device)

# 使用 SharedBottomCNNModel
model = SharedBottomCNNModel(num_features, emotion_output_dim, focus_output_dim).to(device)

# 损失函数和优化器
emotion_criterion = nn.CrossEntropyLoss()
focus_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 计算准确率和F1的函数
def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def calculate_f1(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    return f1_score(labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='weighted')

# 测试函数
def evaluate(model, test_dataloader):
    model.eval()
    all_emotion_labels = []
    all_focus_labels = []
    all_emotion_preds = []
    all_focus_preds = []

    with torch.no_grad():
        for features, emotion_labels, focus_labels in test_dataloader:
            features, emotion_labels, focus_labels = features.to(device), emotion_labels.to(device), focus_labels.to(device)
            emotion_pred, focus_pred = model(features)

            # 获取预测标签
            emotion_pred_label = torch.argmax(F.softmax(emotion_pred, dim=1), dim=1)
            focus_pred_label = torch.argmax(F.softmax(focus_pred, dim=1), dim=1)

            # 保存标签和预测
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
            all_focus_labels.extend(focus_labels.cpu().numpy())
            all_emotion_preds.extend(emotion_pred_label.cpu().numpy())
            all_focus_preds.extend(focus_pred_label.cpu().numpy())

    # 计算情绪任务指标
    emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')

    # 计算专注度任务指标
    focus_acc = accuracy_score(all_focus_labels, all_focus_preds)
    focus_f1 = f1_score(all_focus_labels, all_focus_preds, average='weighted')

    return emotion_acc, emotion_f1, focus_acc, focus_f1

# 训练过程
epochs = 30
best_epoch = 0
best_avg_accuracy = 0.0
best_emotion_acc = 0.0
best_focus_acc = 0.0
best_emotion_f1 = 0.0
best_focus_f1 = 0.0

for epoch in range(epochs):
    model.train()
    model.to(device)  # 将模型部署到 GPU
    total_emotion_loss = 0.0
    total_focus_loss = 0.0
    correct_emotion = 0
    correct_focus = 0
    total_samples = 0

    for features, emotion_labels, focus_labels in train_dataloader:
        features, emotion_labels, focus_labels = (
            features.to(device),
            emotion_labels.to(device),
            focus_labels.to(device)
        )

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

    # 每个epoch后进行一次测试
    emotion_acc, emotion_f1, focus_acc, focus_f1 = evaluate(model, test_dataloader)
    avg_accuracy = (emotion_acc + focus_acc) / 2

    print(f"Epoch [{epoch + 1}/{epochs}] Test Results - "
          f"Emotion Accuracy: {emotion_acc:.4f}, Emotion F1: {emotion_f1:.4f}, "
          f"Focus Accuracy: {focus_acc:.4f}, Focus F1: {focus_f1:.4f}, "
          f"Avg Accuracy: {avg_accuracy:.4f}")

    # 保存测试集表现最好的模型
    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_emotion_acc = emotion_acc
        best_focus_acc = focus_acc
        best_emotion_f1 = emotion_f1
        best_focus_f1 = focus_f1
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'pth_normalized/best_cnn_sb.pth')

# 打印最佳结果
print(f"Best Model at Epoch [{best_epoch}] - "
      f"Emotion Accuracy: {best_emotion_acc:.4f}, Emotion F1: {best_emotion_f1:.4f}, "
      f"Focus Accuracy: {best_focus_acc:.4f}, Focus F1: {best_focus_f1:.4f}, "
      f"Avg Accuracy: {best_avg_accuracy:.4f}")

