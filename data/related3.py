import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# 计算百分比
contingency_percentage = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

# 绘制热力图显示百分比
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.title("Relevance Search (Percentage)")
plt.xlabel('if_focus')
plt.ylabel('emotion')
plt.show()

# 保存图像
plt.savefig('relevance_percentage.png', format='png')
