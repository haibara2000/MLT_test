import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch.nn.functional as F
from model.shared_bottom_mlp import SharedBottomMlpModel
from data.load_data_in_turn import EmotionFocusDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载测试数据
# csv_test_file = 'data/test.csv'
csv_test_file = 'data/normalized_test.csv'
test_dataset = EmotionFocusDataset(csv_test_file)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
# 模型参数
input_dim = 58
shared_hidden_dim = 128
emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())  # 表情类别数
focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())  # 专注度类别数
model = SharedBottomMlpModel(input_dim, shared_hidden_dim, emotion_output_dim, focus_output_dim)
model.load_state_dict(torch.load('pth_normalized/shared_bottom_mlp_model.pth'))
model.to(device)  # 将模型转移到 GPU
model.eval()  # 切换到评估模式

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
        features, emotion_labels, focus_labels = (features.to(device),
                                                  emotion_labels.to(device),
                                                  focus_labels.to(device)) # 数据转到 GPU
        # 模型预测
        emotion_pred, focus_pred = model(features)

        # 获取概率和预测标签
        emotion_prob = F.softmax(emotion_pred, dim=1)  # 转为概率
        focus_prob = F.softmax(focus_pred, dim=1)
        emotion_pred_label = torch.argmax(emotion_prob, dim=1)
        focus_pred_label = torch.argmax(focus_prob, dim=1)

        # 保存标签和预测
        all_emotion_labels.extend(emotion_labels.cpu().numpy())
        all_focus_labels.extend(focus_labels.cpu().numpy())
        all_emotion_preds.extend(emotion_pred_label.cpu().numpy())
        all_focus_preds.extend(focus_pred_label.cpu().numpy())
        all_emotion_probs.extend(emotion_prob.cpu().numpy())
        all_focus_probs.extend(focus_prob.cpu().numpy())

# 计算情绪任务的指标
emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
# emotion_auc = roc_auc_score(all_emotion_labels, all_emotion_probs, multi_class='ovr')

# 计算专注度任务的指标
all_focus_labels_one_hot = label_binarize(all_focus_labels, classes=range(focus_output_dim))
focus_acc = accuracy_score(all_focus_labels, all_focus_preds)
focus_f1 = f1_score(all_focus_labels, all_focus_preds, average='weighted')
# focus_auc = roc_auc_score(all_focus_labels_one_hot, all_focus_probs, multi_class='ovr')

# 输出测试结果
# print(f"Emotion Task - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}, AUC: {emotion_auc:.4f}")
# print(f"Focus Task - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}, AUC: {focus_auc:.4f}")
print("result of shared_bottom_mlp:")
print(f"Emotion Task - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}")
print(f"Focus Task - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}")
