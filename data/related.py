# 探索label之间的相关性
import pandas as pd
from scipy.stats import chi2_contingency

# 加载数据
data = pd.read_csv('train.csv')

# 获取 if_focus=0 和 if_focus=1 的样本
focus_0 = data[data['if_focus'] == 0]
focus_1 = data[data['if_focus'] == 1]

# 找到较小类别的数量
min_size = min(len(focus_0), len(focus_1))

# 对较大类别进行采样，选择与较小类别相同数量的样本
focus_0 = focus_0.sample(n=min_size, random_state=42)
focus_1 = focus_1.sample(n=min_size, random_state=42)

# 合并这两个类别的样本
balanced_data = pd.concat([focus_0, focus_1])

# 打乱数据顺序
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 查看平衡后的数据
print(balanced_data['if_focus'].value_counts())

# 创建交叉表
contingency_table = pd.crosstab(balanced_data['emotion'], balanced_data['if_focus'])

# 计算卡方检验
chi2, p_value, _, _ = chi2_contingency(contingency_table)

print("交叉表：")
print(contingency_table)
print(f"\n卡方检验 p-value: {p_value}")
