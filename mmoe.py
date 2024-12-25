import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from data.load_data_in_group import EmotionFocusDataset
from model.mmoe_cnn import MMoECNNModel  # 模型类在 `model/mmoe_cnn.py` 中
import torch.nn.functional as F
import pandas as pd

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
emotion_output_dim = len(pd.read_csv(csv_train_file)['emotion'].unique())
focus_output_dim = len(pd.read_csv(csv_train_file)['if_focus'].unique())

# 初始化模型
model = MMoECNNModel(num_features, emotion_output_dim, focus_output_dim, num_experts=6).to(device)

emotion_criterion = nn.CrossEntropyLoss()
focus_criterion = nn.CrossEntropyLoss()

# 引入可训练的不确定性参数（log_sigma^2），用于调整权重
log_sigma_emotion = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))  # log(sigma^2) for emotion task
log_sigma_focus = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))  # log(sigma^2) for focus task

optimizer = optim.Adam(
    list(model.parameters()) + [log_sigma_emotion, log_sigma_focus],
    lr=0.001
)

# 计算准确率和F1的函数
def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    accuracy = correct / labels.size(0)  # accuracy = correct / batch_size
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
    total_emotion_loss = 0.0
    total_focus_loss = 0.0
    total_emotion_accuracy = 0.0
    total_focus_accuracy = 0.0
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

        # 自适应权重的损失计算
        weighted_emotion_loss = emotion_loss / (2 * torch.exp(log_sigma_emotion)) + log_sigma_emotion / 2
        weighted_focus_loss = focus_loss / (2 * torch.exp(log_sigma_focus)) + log_sigma_focus / 2
        total_loss = weighted_emotion_loss + weighted_focus_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 计算准确率
        emotion_accuracy = calculate_accuracy(emotion_pred, emotion_labels)
        focus_accuracy = calculate_accuracy(focus_pred, focus_labels)

        total_emotion_loss += emotion_loss.item()
        total_focus_loss += focus_loss.item()
        total_emotion_accuracy += emotion_accuracy * features.size(0)  # 累加每个批次的样本数
        total_focus_accuracy += focus_accuracy * features.size(0)  # 累加每个批次的样本数
        total_samples += features.size(0)

    # 打印每个 epoch 的损失和准确率
    epoch_emotion_accuracy = total_emotion_accuracy / total_samples
    epoch_focus_accuracy = total_focus_accuracy / total_samples

    print(f"Epoch [{epoch+1}/{epochs}], Emotion Loss: {total_emotion_loss:.4f}, Focus Loss: {total_focus_loss:.4f}, "
          f"Emotion Accuracy: {epoch_emotion_accuracy:.4f}, Focus Accuracy: {epoch_focus_accuracy:.4f}, "
          f"log_sigma_emotion: {log_sigma_emotion.item():.4f}, log_sigma_focus: {log_sigma_focus.item():.4f}")

    # 每个epoch后进行一次测试
    emotion_acc, emotion_f1, focus_acc, focus_f1 = evaluate(model, test_dataloader)
    avg_accuracy = (emotion_acc + focus_acc) / 2

    print(f"Epoch [{epoch+1}/{epochs}] Test Results - "
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
        torch.save(model.state_dict(), 'pth_normalized/best_mmoe_cnn_model301.pth')

# 打印最佳结果
print(f"Best Model at Epoch [{best_epoch}] - "
      f"Emotion Accuracy: {best_emotion_acc:.4f}, Emotion F1: {best_emotion_f1:.4f}, "
      f"Focus Accuracy: {best_focus_acc:.4f}, Focus F1: {best_focus_f1:.4f}, "
      f"Avg Accuracy: {best_avg_accuracy:.4f}")

