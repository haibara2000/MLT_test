import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset
from data.load_data_in_turn import  EmotionFocusDataset, prepare_data
import numpy as np
import os

# 定义数据加载器
train_csv = 'data/normalized_train.csv'
# train_csv = 'data/balanced_train.csv'
# train_csv = 'data/train.csv'
batch_size = 32

train_dataset = EmotionFocusDataset(train_csv)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 准备训练集数据
X_train, y_train_emotion, y_train_focus = prepare_data(train_loader)

# 转换为 DMatrix 格式
dtrain_emotion = xgb.DMatrix(X_train, label=y_train_emotion)
dtrain_focus = xgb.DMatrix(X_train, label=y_train_focus)

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

# 保存模型为字典
os.makedirs('models', exist_ok=True)
torch.save({'emotion_model': bst_emotion.save_raw(),
            'focus_model': bst_focus.save_raw()}, 'pth_normalized/xgboost_models.pth')
print("Models saved as 'xgboost_models.pth'")
