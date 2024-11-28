# 可视化方法
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

# 如果 'emotion' 和 'if_focus' 是类别标签，可以先进行编码
balanced_data['emotion_encoded'] = balanced_data['emotion'].astype('category').cat.codes
balanced_data['focus_encoded'] = balanced_data['if_focus'].astype('category').cat.codes

# 计算皮尔逊相关系数
correlation = balanced_data['emotion_encoded'].corr(balanced_data['focus_encoded'])

print(f"表情和专注度标签之间的相关系数: {correlation}")
