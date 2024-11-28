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

# 计算每个表情在不同 if_focus 条件下的数量以及百分比
focus_0_counts = focus_0['emotion'].value_counts()
focus_1_counts = focus_1['emotion'].value_counts()

# 计算每个表情的出现百分比
focus_0_percentage = (focus_0_counts / focus_0_counts.sum()) * 100
focus_1_percentage = (focus_1_counts / focus_1_counts.sum()) * 100

# 合并结果
focus_0_stats = pd.DataFrame({
    'count_focus_0': focus_0_counts,
    'percentage_focus_0': focus_0_percentage
}).reset_index().rename(columns={'index': 'emotion'})

focus_1_stats = pd.DataFrame({
    'count_focus_1': focus_1_counts,
    'percentage_focus_1': focus_1_percentage
}).reset_index().rename(columns={'index': 'emotion'})

# 合并两部分统计信息
merged_stats = pd.merge(focus_0_stats, focus_1_stats, on='emotion', how='outer').fillna(0)

# 输出统计信息
print(merged_stats)

# 可视化：绘制每个表情在不同专注度条件下的数量
plt.figure(figsize=(12, 6))
merged_stats.set_index('emotion')[['count_focus_0', 'count_focus_1']].plot(kind='bar', stacked=False)
plt.title("Emotion Counts for Each Focus Condition")
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='if_focus', loc='upper right')
plt.tight_layout()
plt.show()

# 可视化：绘制每个表情在不同专注度条件下的百分比
plt.figure(figsize=(12, 6))
merged_stats.set_index('emotion')[['percentage_focus_0', 'percentage_focus_1']].plot(kind='bar', stacked=False)
plt.title("Emotion Percentage for Each Focus Condition")
plt.xlabel('Emotion')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='if_focus', loc='upper right')
plt.tight_layout()
plt.show()

# 保存图像
plt.savefig('relevance_percentage.png', format='png')
