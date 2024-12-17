import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('train.csv')  # 请替换为你的数据路径

# 假设学生ID列为 'student_number'，特征从 'eye_open' 开始到第59列
features = data.columns[1:59]  # 假设特征列从第2列开始，到第59列

# 创建一个新的 DataFrame 来保存归一化后的数据
normalized_data = data.copy()

# 按学生进行 min-max 归一化
for feature in features:
    # 对每个特征，按学生进行归一化
    for student_id in data['student_number'].unique():
        # 获取该学生的该特征值
        student_feature_value = data.loc[data['student_number'] == student_id, feature].values[0]

        # 计算该学生的该特征的 min 和 max
        student_min = data.loc[data['student_number'] == student_id, feature].min()
        student_max = data.loc[data['student_number'] == student_id, feature].max()

        # 进行 min-max 归一化
        if student_max != student_min:  # 防止除零错误
            normalized_value = (student_feature_value - student_min) / (student_max - student_min)
        else:
            normalized_value = 0  # 如果最大值等于最小值，则设为0（或者你也可以选择其他处理方式）

        # 保存归一化后的值到新的 DataFrame
        normalized_data.loc[normalized_data['student_number'] == student_id, feature] = normalized_value

# 保存归一化后的数据到新的 CSV 文件
normalized_data.to_csv('normalized_train.csv', index=False)

print("Min-Max normalization complete. Data saved as 'normalized_student_data.csv'.")
