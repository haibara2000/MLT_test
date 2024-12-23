import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from data.load_data_in_group import EmotionFocusDataset
from model.single_cnn import SimpleCNN
import pandas as pd
# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50
learning_rate = 0.001

# 数据加载
csv_file = 'data/balanced_train.csv'
dataset = EmotionFocusDataset(csv_file)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 获取输出类别数
num_features = pd.read_csv(csv_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(set(dataset.emotion_labels.numpy()))
focus_output_dim = len(set(dataset.focus_labels.numpy()))

# 初始化模型、损失函数和优化器
emotion_model = SimpleCNN(num_features, emotion_output_dim).to(device)
focus_model = SimpleCNN(num_features, focus_output_dim).to(device)
criterion = nn.CrossEntropyLoss()
emotion_optimizer = Adam(emotion_model.parameters(), lr=learning_rate)
focus_optimizer = Adam(focus_model.parameters(), lr=learning_rate)

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

    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"  Emotion Loss: {total_emotion_loss:.4f}, Emotion Acc: {emotion_acc:.4f}")
    print(f"  Focus Loss: {total_focus_loss:.4f}, Focus Acc: {focus_acc:.4f}")

# 保存模型
torch.save({
    'emotion_model_state_dict': emotion_model.state_dict(),
    'focus_model_state_dict': focus_model.state_dict()
}, 'pth_original1/single_cnn_models.pth')
print("模型已保存到 'pth1/single_cnn_models.pth'")