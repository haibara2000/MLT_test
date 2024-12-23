import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from data.load_data_in_turn import  EmotionFocusDataset, prepare_data
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb

# 定义数据加载器
test_csv = 'data/test.csv'
# test_csv = 'data/normalized_test.csv'
# test_csv = 'data/balanced_test.csv'
test_dataset = EmotionFocusDataset(test_csv)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# 准备测试集数据
X_test, y_test_emotion, y_test_focus = prepare_data(test_loader)

# 转换为 DMatrix 格式
dtest_emotion = xgb.DMatrix(X_test, label=y_test_emotion)
dtest_focus = xgb.DMatrix(X_test, label=y_test_focus)

# 加载模型
# checkpoint = torch.load('pth/xgboost_models.pth')
checkpoint = torch.load('pth_origin_1220/xgboost_models.pth')
bst_emotion = xgb.Booster()
bst_emotion.load_model(checkpoint['emotion_model'])

bst_focus = xgb.Booster()
bst_focus.load_model(checkpoint['focus_model'])
print("Models loaded from 'xgboost_models.pth'")

# 预测
y_pred_emotion = bst_emotion.predict(dtest_emotion)
y_pred_focus = (bst_focus.predict(dtest_focus) > 0.5).astype(int)

# 评估
accuracy_emotion = accuracy_score(y_test_emotion, y_pred_emotion)
f1_emotion = f1_score(y_test_emotion, y_pred_emotion, average='weighted')

accuracy_focus = accuracy_score(y_test_focus, y_pred_focus)
f1_focus = f1_score(y_test_focus, y_pred_focus)

# 输出测试结果
print("result of XGBoost:")
print(f"Emotion Classification - Accuracy: {accuracy_emotion:.4f}, F1 Score: {f1_emotion:.4f}")
print(f"Focus Classification - Accuracy: {accuracy_focus:.4f}, F1 Score: {f1_focus:.4f}")
