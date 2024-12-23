import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Dataset
from data.load_data_in_turn import  EmotionFocusDataset, prepare_data
import numpy as np
import os

# 定义数据加载器
train_csv = 'data/train.csv'
# train_csv = 'data/normalized_train.csv'
batch_size = 32

train_dataset = EmotionFocusDataset(train_csv)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 准备训练集数据
X_train, y_train_emotion, y_train_focus = prepare_data(train_loader)

# 定义决策树模型
emotion_model = DecisionTreeClassifier(max_depth=10, random_state=42)
focus_model = DecisionTreeClassifier(max_depth=10, random_state=42)

# 训练模型
print("Training emotion classification model...")
emotion_model.fit(X_train, y_train_emotion)

print("Training focus classification model...")
focus_model.fit(X_train, y_train_focus)

# 保存模型到一个 `.pth` 文件
os.makedirs('models', exist_ok=True)
torch.save({'emotion_model': emotion_model, 'focus_model': focus_model}, 'pth_origin/decision_tree_models.pth')
print("Combined model saved as 'pth/dt_models.pth'")
