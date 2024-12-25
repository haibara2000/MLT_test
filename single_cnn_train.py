import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
from data.load_data_in_group import EmotionFocusDataset
from model.single_cnn import SimpleCNN
import pandas as pd
import numpy as np

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 20
learning_rate = 0.001

# 数据加载
# csv_train_file = 'data/normalized_train.csv'
# csv_test_file = 'data/normalized_test.csv'  # 测试集文件路径
csv_train_file = 'data/train.csv'
csv_test_file = 'data/test.csv'  # 测试集文件路径

train_dataset = EmotionFocusDataset(csv_train_file)
test_dataset = EmotionFocusDataset(csv_test_file)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 获取输出类别数
num_features = pd.read_csv(csv_train_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(set(train_dataset.emotion_labels.numpy()))
focus_output_dim = len(set(train_dataset.focus_labels.numpy()))

# 初始化模型、损失函数和优化器
emotion_model = SimpleCNN(num_features, emotion_output_dim).to(device)
focus_model = SimpleCNN(num_features, focus_output_dim).to(device)
criterion = nn.CrossEntropyLoss()
emotion_optimizer = Adam(emotion_model.parameters(), lr=learning_rate)
focus_optimizer = Adam(focus_model.parameters(), lr=learning_rate)

# 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():  # 评估时不需要计算梯度
        for features, emotion_labels, focus_labels in dataloader:
            features = features.to(device).float()
            features = features.squeeze(2)  # 修正为 (batch_size, 1, 58, 30)
            emotion_labels = emotion_labels.to(device)
            focus_labels = focus_labels.to(device)

            # 对表情任务进行预测
            emotion_outputs = model(features)
            _, emotion_pred = torch.max(emotion_outputs, 1)

            # 对专注度任务进行预测
            focus_outputs = model(features)
            _, focus_pred = torch.max(focus_outputs, 1)

            all_labels.append(emotion_labels.cpu().numpy())
            all_preds.append(emotion_pred.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

# 训练循环
for epoch in range(epochs):
    emotion_model.train()
    focus_model.train()

    total_emotion_loss = 0
    total_focus_loss = 0
    emotion_preds, emotion_targets = [], []
    focus_preds, focus_targets = [], []

    for features, emotion_labels, focus_labels in train_loader:
        features = features.to(device).float()
        features = features.squeeze(2)  # 修正为 (batch_size, 1, 58, 30)
        emotion_labels = emotion_labels.to(device)
        focus_labels = focus_labels.to(device)

        # 表情任务训练
        emotion_optimizer.zero_grad()
        emotion_outputs = emotion_model(features)
        emotion_loss = criterion(emotion_outputs, emotion_labels)
        emotion_loss.backward()
        emotion_optimizer.step()
        total_emotion_loss += emotion_loss.item()
        emotion_preds.extend(torch.argmax(emotion_outputs, dim=1).cpu().numpy())
        emotion_targets.extend(emotion_labels.cpu().numpy())

        # 专注度任务训练
        focus_optimizer.zero_grad()
        focus_outputs = focus_model(features)
        focus_loss = criterion(focus_outputs, focus_labels)
        focus_loss.backward()
        focus_optimizer.step()
        total_focus_loss += focus_loss.item()
        focus_preds.extend(torch.argmax(focus_outputs, dim=1).cpu().numpy())
        focus_targets.extend(focus_labels.cpu().numpy())

    # 计算训练集准确率
    emotion_acc = accuracy_score(emotion_targets, emotion_preds)
    focus_acc = accuracy_score(focus_targets, focus_preds)

    # 计算测试集准确率和F1分数
    emotion_test_acc, emotion_test_f1 = evaluate_model(emotion_model, test_loader, device)
    focus_test_acc, focus_test_f1 = evaluate_model(focus_model, test_loader, device)

    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"  Emotion Loss: {total_emotion_loss:.4f}, Emotion Acc: {emotion_acc:.4f}, Emotion Test Acc: {emotion_test_acc:.4f}, Emotion Test F1: {emotion_test_f1:.4f}")
    print(f"  Focus Loss: {total_focus_loss:.4f}, Focus Acc: {focus_acc:.4f}, Focus Test Acc: {focus_test_acc:.4f}, Focus Test F1: {focus_test_f1:.4f}")

# 保存模型
torch.save({
    'emotion_model_state_dict': emotion_model.state_dict(),
    'focus_model_state_dict': focus_model.state_dict()
}, 'pth_normalized/single_cnn_models.pth')
print("模型已保存到 'single_cnn_models.pth'")
