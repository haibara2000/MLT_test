import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model.mmoe_cnn import MMoECNNModel  # 模型类在 `model/mmoe_cnn.py` 中
from data.load_data_in_group import EmotionFocusDataset  # 数据加载类在 `data/load_data_in_group.py` 中

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据加载
# csv_file = 'data/train.csv'
csv_file = 'data/normalized_train.csv'
dataset = EmotionFocusDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
num_features = pd.read_csv(csv_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(pd.read_csv(csv_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_file)['if_focus'].unique())  # 专注度类别数

# 初始化模型、损失函数和优化器
model = MMoECNNModel(num_features, emotion_output_dim, focus_output_dim, num_experts=6).to(device)
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

# 计算准确率的函数
def calculate_accuracy(predictions, labels):
    # 预测的标签是每个类别的概率中最大值的索引
    _, predicted_labels = torch.max(predictions, 1)
    # 计算准确率
    correct = (predicted_labels == labels).sum().item()
    accuracy = correct / labels.size(0)  # accuracy = correct / batch_size
    return accuracy

# 训练过程
epochs = 1
for epoch in range(epochs):
    model.train()
    total_emotion_loss = 0.0
    total_focus_loss = 0.0
    total_emotion_accuracy = 0.0
    total_focus_accuracy = 0.0
    total_samples = 0

    for features, emotion_labels, focus_labels in dataloader:
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
        # total_loss = emotion_loss + focus_loss
        # 自适应权重的损失计算
        weighted_emotion_loss = emotion_loss / (2 * torch.exp(log_sigma_emotion)) + log_sigma_emotion / 2
        weighted_focus_loss = focus_loss / (2 * torch.exp(log_sigma_focus)) + log_sigma_focus / 2
        total_loss = weighted_emotion_loss + weighted_focus_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # total_emotion_loss += emotion_loss.item()
        # total_focus_loss += focus_loss.item()

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

    # 打印每个 epoch 的损失和权重
    print(f"Epoch [{epoch+1}/{epochs}], Emotion Loss: {total_emotion_loss:.4f}, Focus Loss: {total_focus_loss:.4f}, "
          f"Emotion Accuracy: {epoch_emotion_accuracy:.4f}, Focus Accuracy: {epoch_focus_accuracy:.4f}, "
          f"log_sigma_emotion: {log_sigma_emotion.item():.4f}, log_sigma_focus: {log_sigma_focus.item():.4f}")

# 保存模型
# torch.save(model.state_dict(), 'pth1/mmoe_cnn_model1.pth')   # 使用原始数据训练100个epoch的结果，存在过拟合（0.28,0.83）
# torch.save(model.state_dict(), 'pth1/mmoe_cnn_model2.pth')   # 使用精简数据训练100个epoch的结果，训练准确率卡在71左右(29,70)
torch.save(model.state_dict(), 'pth_normalized/mmoe_cnn_model12232.pth')   # 使用精简数据训练100个epoch的结果，训练准确率卡在71左右(29,70)


# torch.save({'log_sigma_emotion': log_sigma_emotion, 'log_sigma_focus': log_sigma_focus},
#            'pth/loss_weights.pth')
print("模型及损失权重已保存")
