import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('test.csv')  # 请替换为你的数据路径

# 假设学生ID列为 'student_number'，特征从 'eye_open' 开始到第59列
features = data.columns[1:59]  # 假设特征列从第2列开始，到第59列
student_ids = data['student_number']

# 存储每个特征在不同学生上的极差
feature_range_diff = {}

# 存储每个特征的最小极差、最大极差及相应学生ID
feature_range_extremes = {}

# 按学生计算每个特征的极差
for feature in features:
    # 计算每个学生该特征的极差（最大值 - 最小值）
    student_feature_ranges = data.groupby('student_number')[feature].agg(lambda x: x.max() - x.min())

    # 计算该特征在不同学生之间的极差差异性（标准差）
    std_dev = np.std(student_feature_ranges)
    feature_range_diff[feature] = std_dev

    # 获取最小极差和最大极差及相应的学生ID
    min_range_student = student_feature_ranges.idxmin()  # 极差最小的学生ID
    max_range_student = student_feature_ranges.idxmax()  # 极差最大的学生ID
    min_range_value = student_feature_ranges.min()  # 最小极差值
    max_range_value = student_feature_ranges.max()  # 最大极差值

    # 存储结果
    feature_range_extremes[feature] = {
        'min_range_value': min_range_value,
        'min_range_student': min_range_student,
        'max_range_value': max_range_value,
        'max_range_student': max_range_student
    }

# 打印每个特征的极差差异性大小
for feature, std_dev in feature_range_diff.items():
    print(f"Feature: {feature}, Range Standard Deviation: {std_dev}")

# 打印每个特征的最小极差和最大极差，以及相应学生ID
for feature, extremes in feature_range_extremes.items():
    print(f"\nFeature: {feature}")
    print(f"  Min Range: {extremes['min_range_value']}, Student ID: {extremes['min_range_student']}")
    print(f"  Max Range: {extremes['max_range_value']}, Student ID: {extremes['max_range_student']}")

# 计算所有特征差异性的标准差
diff_values = list(feature_range_diff.values())
overall_std_dev = np.std(diff_values)

# 打印所有特征差异性的标准差
print(f"\nStandard Deviation of Feature Range Variabilities: {overall_std_dev}")

# # 如果差异性标准差较大，可能表示按学生做归一化有意义
# some_threshold = 0.5  # 设置一个合理的阈值，这个值需要根据你的实际数据进行调整
# if overall_std_dev > some_threshold:
#     print("\nIt may be necessary to normalize the data per student.")
# else:
#     print("\nNormalization per student may not be necessary.")
