import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data.load_data_in_turn import EmotionFocusDataset

# 定义数据加载器
train_csv = 'data/train.csv'
test_csv = 'data/test.csv'
batch_size = 32

train_dataset = EmotionFocusDataset(train_csv)
test_dataset = EmotionFocusDataset(test_csv)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# 将数据加载到内存中（适用于 XGBoost）
def prepare_data(loader):
    features_list, emotion_labels_list, focus_labels_list = [], [], []
    for features, emotion_labels, focus_labels in loader:
        features_list.append(features.numpy())
        emotion_labels_list.append(emotion_labels.numpy())
        focus_labels_list.append(focus_labels.numpy())
    features = np.vstack(features_list)
    emotion_labels = np.concatenate(emotion_labels_list)
    focus_labels = np.concatenate(focus_labels_list)
    return features, emotion_labels, focus_labels

# 准备训练集和测试集数据
import numpy as np
X_train, y_train_emotion, y_train_focus = prepare_data(train_loader)
X_test, y_test_emotion, y_test_focus = prepare_data(test_loader)

# 转换为 DMatrix 格式
dtrain_emotion = xgb.DMatrix(X_train, label=y_train_emotion)
dtest_emotion = xgb.DMatrix(X_test, label=y_test_emotion)

dtrain_focus = xgb.DMatrix(X_train, label=y_train_focus)
dtest_focus = xgb.DMatrix(X_test, label=y_test_focus)

# 设置 XGBoost 参数
params_emotion = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train_emotion)),
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 6,
    'seed': 42
}

params_focus = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'seed': 42
}

# 训练模型
print("Training emotion classification model...")
bst_emotion = xgb.train(params_emotion, dtrain_emotion, num_boost_round=100)

print("Training focus classification model...")
bst_focus = xgb.train(params_focus, dtrain_focus, num_boost_round=100)

# 预测
y_pred_emotion = bst_emotion.predict(dtest_emotion)
y_pred_focus = (bst_focus.predict(dtest_focus) > 0.5).astype(int)

# 评估
accuracy_emotion = accuracy_score(y_test_emotion, y_pred_emotion)
f1_emotion = f1_score(y_test_emotion, y_pred_emotion, average='weighted')

accuracy_focus = accuracy_score(y_test_focus, y_pred_focus)
f1_focus = f1_score(y_test_focus, y_pred_focus)

print(f"Emotion Classification - Accuracy: {accuracy_emotion:.4f}, F1 Score: {f1_emotion:.4f}")
print(f"Focus Classification - Accuracy: {accuracy_focus:.4f}, F1 Score: {f1_focus:.4f}")