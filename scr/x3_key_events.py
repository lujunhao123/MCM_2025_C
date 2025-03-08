import pandas as pd
import matplotlib.pyplot as plt
from utils import get_code
"""
好的，我现在需要解释什么是“依赖国家”，根据用户提供的代码和上下文来看。首先，用户在处理奥运会奖牌数据，
筛选出那些奖牌集中在少数项目或赛事上的国家。我需要理解代码的逻辑，然后结合统计方法来解释这个概念。

首先，代码中定义了一个阈值threshold=0.5，然后筛选出Medal Proportion超过这个阈值的国家。
这里的Medal Proportion是某个国家在特定体育项目或赛事中获得的奖牌数占总奖牌数的比例。
所以，当这个比例超过60%时，意味着该国一半以上的奖牌都来自这个项目或赛事，这样的国家就被视为“依赖国家”。
"""
# 设置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 数据准备
country_mapping, noc_mapping, country_codes = get_code()
data = pd.read_csv("./2025_Problem_C_Data/summerOly_athletes.csv")
print(data.info())
medal_data = data[data['Medal'] != 'No medal']
medal_data["NOC"] = medal_data["NOC"].map(country_codes)

# 数据处理
medal_counts = medal_data.groupby(['NOC', 'Sport', 'Event'])['Medal'].count().reset_index(name='Medal Count')
total_medals_per_country = medal_counts.groupby('NOC')['Medal Count'].sum().reset_index(name='Total Medals')
medal_data_with_totals = pd.merge(medal_counts, total_medals_per_country, on='NOC')
medal_data_with_totals['Medal Proportion'] = medal_data_with_totals['Medal Count'] / medal_data_with_totals['Total Medals']

# 筛选依赖型国家
threshold = 0.6
reliant_countries = medal_data_with_totals[medal_data_with_totals['Medal Proportion'] > threshold]
reliant_countries_top_sports = reliant_countries.groupby('NOC').apply(lambda x: x.nlargest(2, 'Medal Proportion')).reset_index(drop=True)

# 创建事件编码映射
sport_events = reliant_countries_top_sports['Sport'] + ' - ' + reliant_countries_top_sports['Event']
unique_events = sport_events.unique()
event_labels = [f"{chr(65+i)}" for i in range(len(unique_events))]
event_mapping = dict(zip(unique_events, event_labels))
reliant_countries_top_sports['Event Code'] = sport_events.map(event_mapping)


country_max_prop = reliant_countries_top_sports.groupby('NOC')['Medal Proportion'].max().sort_values(ascending=False)
# 按排序后的国家顺序处理数据
sorted_countries = country_max_prop.index.tolist()
reliant_countries_top_sports = reliant_countries_top_sports.set_index('NOC').loc[sorted_countries].reset_index()


# 可视化设置
plt.figure(figsize=(10, 6))  # 进一步增大画布尺寸
ax = plt.gca()

# 绘制柱状图（保持相同）
for country in reliant_countries_top_sports['NOC'].unique():
    country_data = reliant_countries_top_sports[reliant_countries_top_sports['NOC'] == country]
    plt.bar(country_data['Event Code'],
            country_data['Medal Proportion'],
            label=country,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.2)

# 坐标轴标签设置
plt.xticks(range(len(unique_events)), event_labels)
plt.xlabel('Sport-Event Code', labelpad=20, fontsize=12)
plt.ylabel('Medal Proportion', labelpad=20, fontsize=12)
plt.title('National Medal Dependency Profile',
         fontsize=16,
         fontweight='bold',
         pad=25)

# 调整图例布局
# ---------------------------
# 项目图例（底部）
legend_text = [f"{label}: {event}" for event, label in event_mapping.items()]
max_per_line = 4  # 减少每行显示数量
lines = ["    ".join(legend_text[i:i+max_per_line])
        for i in range(0, len(legend_text), max_per_line)]
import os
# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# 生成图例文本
legend_text = [f"{label}: {event}" for event, label in event_mapping.items()]

# 保存到txt文件
with open('./output/event_legend.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(legend_text))

print(f"图例已保存至：{os.path.abspath('./output/event_legend.txt')}")

# 国家图例（右侧外部）
plt.legend(title='Country Code',
          bbox_to_anchor=(1.03, 0.5),  # 向右调整锚点位置
          loc='center left',
          ncol=2,  # 关键参数：设置两列
          columnspacing=0.8,  # 列间距
          frameon=True,
          framealpha=0.9,
          edgecolor='gray',
          title_fontsize=10,
          fontsize=8)

# 调整布局参数
plt.subplots_adjust()

plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig("National dependence on specific sports.pdf")  # 保存图表