import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F
from data.load_data_in_group import  EmotionFocusDataset
from model.shard_bottom_cnn import SharedBottomCNNModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 加载测试数据
csv_test_file = 'data/test.csv'  # 替换为您的测试集 CSV 文件路径
test_dataset = EmotionFocusDataset(csv_test_file)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
# 模型参数
emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())  # 专注度类别数
model = SharedBottomCNNModel(emotion_output_dim, focus_output_dim)
model.load_state_dict(torch.load('pth/shared_bottom_cnn_uncertainty_model.pth'))
model.eval()

# 初始化指标
all_emotion_labels = []
all_focus_labels = []
all_emotion_preds = []
all_focus_preds = []
all_emotion_probs = []
all_focus_probs = []

# 测试模型
with torch.no_grad():
    for features, emotion_labels, focus_labels in test_dataloader:
        # 模型预测
        emotion_pred, focus_pred = model(features)

        # 获取概率和预测标签
        emotion_prob = F.softmax(emotion_pred, dim=1)
        focus_prob = F.softmax(focus_pred, dim=1)
        emotion_pred_label = torch.argmax(emotion_prob, dim=1)
        focus_pred_label = torch.argmax(focus_prob, dim=1)

        # 保存标签和预测
        all_emotion_labels.extend(emotion_labels.numpy())
        all_focus_labels.extend(focus_labels.numpy())
        all_emotion_preds.extend(emotion_pred_label.numpy())
        all_focus_preds.extend(focus_pred_label.numpy())
        all_emotion_probs.extend(emotion_prob.numpy())
        all_focus_probs.extend(focus_prob.numpy())

# 计算情绪任务的指标
emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
# emotion_auc = roc_auc_score(all_emotion_labels, all_emotion_probs, multi_class='ovr')

# 计算专注度任务的指标
focus_acc = accuracy_score(all_focus_labels, all_focus_preds)
focus_f1 = f1_score(all_focus_labels, all_focus_preds, average='weighted')
# focus_auc = roc_auc_score(all_focus_labels, all_focus_probs, multi_class='ovr')

# 输出测试结果
print("result of shared_bottom_cnn_uncertainty:")
print(f"Emotion Task - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}")
print(f"Focus Task - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}")
