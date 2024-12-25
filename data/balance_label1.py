import pandas as pd
import numpy as np

# 读取数据
file_path = 'history/normalized_test.csv'  # 这里替换为你数据文件的路径
df = pd.read_csv(file_path)

# 提取标签列（假设if_focus是标签列之一，最后一列）
label_column = 'if_focus'
student_id_column = 'student_number'  # 假设学生ID列为student_number

# 按学生ID分组
df_grouped = df.groupby(student_id_column)

# 创建一个空的DataFrame用于存储删除后的数据
df_balanced = pd.DataFrame()

# 对每个学生的数据进行处理
for _, group in df_grouped:
    # 分离if_focus为1和if_focus为0的数据
    group_minority = group[group[label_column] == 1]
    group_majority = group[group[label_column] == 0]

    # 计算每个学生删除的数据数量差异
    minority_count = len(group_minority)
    majority_count = len(group_majority)

    # 如果minority_count大于majority_count，进行欠采样
    if minority_count > majority_count:
        # 随机删除一部分if_focus=1的数据，使两者数量一致
        group_minority_undersampled = group_minority.sample(n=majority_count, random_state=42)
    else:
        # 如果minority_count<=majority_count，不做删除
        group_minority_undersampled = group_minority

    # 合并欠采样后的数据
    balanced_group = pd.concat([group_majority, group_minority_undersampled])

    # 将当前处理后的学生数据加入到平衡后的DataFrame中
    df_balanced = pd.concat([df_balanced, balanced_group])

# 打乱数据顺序（可选）
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 打印平衡后的各类别数量
print("平衡后各类别数量：")
print(df_balanced[label_column].value_counts())

# 保存处理后的数据到新的CSV文件
output_file_path = 'balanced_test.csv'  # 新文件路径
df_balanced.to_csv(output_file_path, index=False)

print(f"处理后的数据已保存到 {output_file_path}")
