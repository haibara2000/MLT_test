import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F
from data.load_data_in_group import  EmotionFocusDataset
from model.mmoe_cnn import MMoECNNModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 加载测试数据
# csv_test_file = 'data/test.csv'
csv_test_file = 'data/normalized_test.csv'
test_dataset = EmotionFocusDataset(csv_test_file)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
# 模型参数
num_features = pd.read_csv(csv_test_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())  # 专注度类别数
model = MMoECNNModel(num_features, emotion_output_dim, focus_output_dim, num_experts=6)
# 最优运行结果
# model.load_state_dict(torch.load('pth_best/mmoe_cnn_model.pth'))  # 0.2968, 0.8668（原始数据100epoch）

# 其他运行结果
model.load_state_dict(torch.load('pth_normalized/mmoe_cnn_model.pth'))  # 0.2968, 0.8668（原始数据100epoch）
# model.load_state_dict(torch.load('pth1/mmoe_cnn_model1.pth'))  # 0.2726, 0.8379
# model.load_state_dict(torch.load('pth_no_normalize/mmoe_cnn_uncertainty_model3.pth'))  # 0.2968, 0.8501
# model.load_state_dict(torch.load('pth/mmoe_cnn_model2.pth'))  # 0.2968, 0.8462
# model.load_state_dict(torch.load('pth/mmoe_cnn_uncertainty_model1.pth')) # 0.2968, 0.8348
# model.load_state_dict(torch.load('pth/mmoe_cnn_model1.pth'))  # 0.2997, 0.8491
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
print("result of mmoe_cnn(uncertainty):")
print(f"Emotion Task - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}")
print(f"Focus Task - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}")
