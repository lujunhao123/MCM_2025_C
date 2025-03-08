import pandas as pd
import matplotlib.pyplot as plt
from utils import get_code

# 设置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12


# 数据准备
def load_and_preprocess():
    # 加载国家代码映射
    country_mapping, noc_mapping, country_codes = get_code()

    # 读取原始数据
    data = pd.read_csv("./2025_Problem_C_Data/summerOly_athletes.csv")
    print("原始数据摘要：")
    print(data.info())

    # 数据过滤与清洗
    medal_data = data[data['Medal'] != 'No medal'].copy()
    medal_data["NOC"] = medal_data["NOC"].map(country_codes)
    medal_data = medal_data[medal_data['NOC'].notna()]  # 过滤无效国家代码

    # 奖牌类型布尔值转换
    medal_data['Gold'] = medal_data['Gold'].astype(int)
    medal_data['Silver'] = medal_data['Silver'].astype(int)
    medal_data['Bronze'] = medal_data['Bronze'].astype(int)

    return medal_data


# 统治力分析核心逻辑
def analyze_dominance(medal_data):
    # 计算运动员级指标
    athlete_stats = medal_data.groupby(['NOC', 'Name']).agg(
        Gold=('Gold', 'sum'),
        Silver=('Silver', 'sum'),
        Bronze=('Bronze', 'sum'),
        First_Year=('Year', 'min'),
        Last_Year=('Year', 'max'),
        Medal_Count=('Medal', 'count')
    ).reset_index()

    # 计算时间跨度
    athlete_stats['Year_Span'] = athlete_stats['Last_Year'] - athlete_stats['First_Year'] + 1

    # 计算国家级指标
    country_stats = medal_data.groupby('NOC').agg(
        Country_Medals=('Medal', 'count'),
        Country_Gold=('Gold', 'sum'),
        Country_Silver=('Silver', 'sum'),
        Country_Bronze=('Bronze', 'sum')
    ).reset_index()

    # 合并数据
    merged = pd.merge(athlete_stats, country_stats, on='NOC')

    # 计算核心指标
    merged['Medal_Proportion'] = merged['Medal_Count'] / merged['Country_Medals']
    merged['Weighted_Score'] = merged['Gold'] * 3 + merged['Silver'] * 2 + merged['Bronze'] * 1
    merged['Dominance_Index'] = merged['Weighted_Score'] * merged['Year_Span'] * merged['Medal_Proportion']

    # 筛选每个国家最具统治力的运动员
    dominant_athletes = merged.loc[merged.groupby('NOC')['Dominance_Index'].idxmax()]

    return dominant_athletes.sort_values('Dominance_Index', ascending=False)


# 可视化与输出
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(dominant_df, top_n=10):
    # 筛选前N个运动员
    plt.style.use('default')
    plot_data = dominant_df.nlargest(top_n, 'Dominance_Index')

    # 创建一个颜色映射（比如，基于 Dominance_Index）
    norm = plt.Normalize(plot_data['Dominance_Index'].min(), plot_data['Dominance_Index'].max())
    cmap = plt.get_cmap('viridis')  # 选择一个合适的色彩图，比如 'coolwarm', 'viridis', 'plasma' 等

    # 创建画布
    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # 绘制条形图，每个条形的颜色根据 Dominance_Index 的值变化
    bars = ax.barh(plot_data['Name'],  # 将 NOC 替换为 Name，按运动员名字绘制
                   plot_data['Dominance_Index'],
                   color=cmap(norm(plot_data['Dominance_Index'])),  # 根据颜色映射设置条形图颜色
                   edgecolor='black',  # 边框颜色
                   height=0.8,
                   linewidth=2.5)  # 增加边框的粗细

    # 添加数据标签，显示运动员名字和对应的国家
    for bar, country in zip(bars, plot_data['NOC']):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(width * 1.02, y,
                f'{country}',  # 显示运动员名字和国家代码
                va='center', ha='left',
                fontsize=15, color='black')  # 设置数据标签的颜色为黑色

    # 装饰图表
    ax.set_xlabel('Dominance Index', labelpad=17)
    ax.set_ylabel('Athlete Name (Country Code)', labelpad=17)  # 修改 Y 轴标签为运动员名字和国家代码
    ax.tick_params(axis='x', labelsize=15)  # 设置 x 轴标签字体大小
    ax.tick_params(axis='y', labelsize=15)  # 设置 y 轴标签的样式
    ax.invert_yaxis()  # 反转 Y 轴，使得条形图按 Dominance_Index 从高到低排列
    ax.grid(axis='x', alpha=0.3)

    # 添加颜色条（Colorbar）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 设置空的数组来创建颜色条
    plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, label='Dominance Index')

    # 显示图表
    plt.tight_layout()
    plt.savefig("National dependence on top athletes.pdf")  # 保存图表




# 主程序
if __name__ == "__main__":
    # 数据加载
    medal_data = load_and_preprocess()

    # 核心分析
    dominant_athletes = analyze_dominance(medal_data)

    # 结果输出
    print("\n国家运动员依赖分析结果：")
    result_table = dominant_athletes[['NOC', 'Name', 'Dominance_Index', 'Medal_Proportion']]
    result_table['Dominance_Index'] = result_table['Dominance_Index'].round(1)
    result_table['Medal_Proportion'] = result_table['Medal_Proportion'].apply(lambda x: f"{x:.1%}")
    result_table.to_csv(r"./2025_Problem_C_Data/国家运动员依赖分析结果", index=False)
    print(result_table.to_markdown(index=False))

    # 可视化
    visualize_results(dominant_athletes)