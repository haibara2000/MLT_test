import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset
from data.load_data_in_turn import  EmotionFocusDataset, prepare_data
import joblib  # 用于保存和加载模型
import os

# 定义数据加载器
train_csv = 'data/normalized_train.csv'
batch_size = 32

train_dataset = EmotionFocusDataset(train_csv)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 准备训练集数据
X_train, y_train_emotion, y_train_focus = prepare_data(train_loader)

# 定义随机森林模型
emotion_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
focus_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 训练模型
print("Training emotion classification model...")
emotion_model.fit(X_train, y_train_emotion)

print("Training focus classification model...")
focus_model.fit(X_train, y_train_focus)

# 计算训练集的准确率
y_pred_emotion_train = emotion_model.predict(X_train)
y_pred_focus_train = focus_model.predict(X_train)

emotion_train_acc = accuracy_score(y_train_emotion, y_pred_emotion_train)
focus_train_acc = accuracy_score(y_train_focus, y_pred_focus_train)

print(f"Emotion Model Training Accuracy: {emotion_train_acc:.4f}")
print(f"Focus Model Training Accuracy: {focus_train_acc:.4f}")

# 保存模型到一个 `.pth` 文件
os.makedirs('models', exist_ok=True)
torch.save({'emotion_model': emotion_model, 'focus_model': focus_model}, 'pth/rf_models.pth')
print("Combined model saved as 'pth/rf_models.pth'")