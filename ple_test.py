import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from data.load_data_in_group import EmotionFocusDataset
from model.ple2 import PLEModel
import torch.nn.functional as F
import pandas as pd

# 加载测试数据
csv_test_file = 'data/normalized_test.csv'
test_dataset = EmotionFocusDataset(csv_test_file)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
num_features = pd.read_csv(csv_test_file).iloc[:, 1:-4].shape[1]
emotion_output_dim = len(pd.read_csv(csv_test_file)['emotion'].unique())
focus_output_dim = len(pd.read_csv(csv_test_file)['if_focus'].unique())
# 初始化模型
# ple_cnn_model2的参数: 27,73
# model = PLEModel(num_CGC_layers=2,
#                  input_size=58,
#                  emotion_output_dim=emotion_output_dim,
#                  focus_output_dim=focus_output_dim,
#                  num_specific_experts=1,
#                  num_shared_experts=1,
#                  experts_out=32,
#                  experts_hidden=64,
#                  towers_hidden=128)
# 初始化模型
model = PLEModel(num_CGC_layers=2,
                 input_size=num_features,
                 emotion_output_dim=emotion_output_dim,
                 focus_output_dim=focus_output_dim,
                 num_specific_experts=4,
                 num_shared_experts=2,
                 experts_out=32,
                 experts_hidden=64,
                 towers_hidden=128)
# model.load_state_dict(torch.load('pth/ple_cnn_model5.pth'))   # 0.2601, 0.7266
model.load_state_dict(torch.load('pth_normalized/best_ple_uncertainty_model.pth'))  # 0.2968, 0.7013
# model.load_state_dict(torch.load('pth/ple_uncertainty_model2.pth'))  # 0.2968, 0.7013
model.eval()

# 初始化指标
all_emotion_labels = []
all_focus_labels = []
all_emotion_preds = []
all_focus_preds = []

with torch.no_grad():
    for features, emotion_labels, focus_labels in test_dataloader:
        emotion_pred, focus_pred = model(features)

        # 获取预测标签
        emotion_pred_label = torch.argmax(F.softmax(emotion_pred, dim=1), dim=1)
        focus_pred_label = torch.argmax(F.softmax(focus_pred, dim=1), dim=1)

        # 保存标签和预测
        all_emotion_labels.extend(emotion_labels.numpy())
        all_focus_labels.extend(focus_labels.numpy())
        all_emotion_preds.extend(emotion_pred_label.numpy())
        all_focus_preds.extend(focus_pred_label.numpy())

# 计算情绪任务指标
emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')

# 计算专注度任务指标
focus_acc = accuracy_score(all_focus_labels, all_focus_preds)
focus_f1 = f1_score(all_focus_labels, all_focus_preds, average='weighted')

print("result of ple_cnn:")
print(f"Emotion Task - Accuracy: {emotion_acc:.4f}, F1 Score: {emotion_f1:.4f}")
print(f"Focus Task - Accuracy: {focus_acc:.4f}, F1 Score: {focus_f1:.4f}")
