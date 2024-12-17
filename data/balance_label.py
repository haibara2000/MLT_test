import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 读取数据
data = pd.read_csv('normalized_train.csv')

# 提取特征和标签
X = data.drop(columns=['if_focus'])  # 删除 'if_focus' 列，剩下的就是特征
y = data['if_focus']  # 'if_focus' 列作为目标标签

#
# # 使用SMOTE生成少数类样本，并控制生成的样本数量（少数类样本数量是多数类的 50%）
# smote = SMOTE(sampling_strategy=0.5)  # 少数类样本生成到多数类的 50%
# X_resampled, y_resampled = smote.fit_resample(X, y)

# 使用欠采样对多数类进行采样
under_sampler = RandomUnderSampler()
# X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# 打印平衡后各类别的数量
print("Balanced class distribution:")
print(pd.Series(y_resampled).value_counts())

# 合并平衡后的数据
balanced_data = pd.concat([data.iloc[:, :1], X_resampled, data.iloc[:, -1], y_resampled], axis=1)

# 保存到新的CSV文件
balanced_data.to_csv('balanced_train.csv', index=False)
