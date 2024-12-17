import torch
from torch.utils.data import DataLoader
from data.load_data_in_group import EmotionFocusDataset
from sklearn.metrics import accuracy_score, f1_score
from model.single_cnn import SimpleCNN
import pandas as pd

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 加载模型
checkpoint = torch.load('pth1/single_cnn_models.pth')
csv_test_file = 'data/balanced_test.csv'
emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())  # 专注度类别数
emotion_model = SimpleCNN(emotion_output_dim).to(device)
focus_model = SimpleCNN(focus_output_dim).to(device)
emotion_model.load_state_dict(checkpoint['emotion_model_state_dict'])
focus_model.load_state_dict(checkpoint['focus_model_state_dict'])
emotion_model.eval()
focus_model.eval()

# 数据加载
test_csv_file = 'data/test.csv'
test_dataset = EmotionFocusDataset(test_csv_file)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 测试循环
emotion_preds, emotion_targets = [], []
focus_preds, focus_targets = [], []

with torch.no_grad():
    for features, emotion_labels, focus_labels in test_loader:
        features = features.to(device).float()
        features = features.squeeze(2)  # 修正为 (batch_size, 1, 58, 30)
        emotion_labels = emotion_labels.to(device)
        focus_labels = focus_labels.to(device)

        # 表情任务测试
        emotion_outputs = emotion_model(features)
        emotion_preds.extend(torch.argmax(emotion_outputs, dim=1).cpu().numpy())
        emotion_targets.extend(emotion_labels.cpu().numpy())

        # 专注度任务测试
        focus_outputs = focus_model(features)
        focus_preds.extend(torch.argmax(focus_outputs, dim=1).cpu().numpy())
        focus_targets.extend(focus_labels.cpu().numpy())

# 计算测试集准确率和 F1 分数
emotion_acc = accuracy_score(emotion_targets, emotion_preds)
emotion_f1 = f1_score(emotion_targets, emotion_preds, average='macro')
focus_acc = accuracy_score(focus_targets, focus_preds)
focus_f1 = f1_score(focus_targets, focus_preds, average='macro')

# 输出测试结果
print("result of single_cnn:")
print(f"Emotion - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}")
print(f"Focus   - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}")