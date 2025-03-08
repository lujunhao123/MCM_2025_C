import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import os

# 读取数据
data = pd.read_csv("./2025_Problem_C_Data/summerOly_athletes.csv")
df = data
df.dropna(inplace=True)
# 步骤1：计算每个国家-运动-年份-性别的奖牌数
medal_counts = df.groupby(['Country', 'Sport', 'Year', 'Sex'])['Total'].sum().reset_index(name='TotalMedals')

# 计算每个运动-年份的总奖牌数
sport_year_total = df.groupby(['Sport', 'Year'])['Total'].sum().reset_index(name='TotalInSportYear')

# 步骤2：生成完整的时间序列（包含所有可能的组合）
all_countries = df['Country'].unique()
all_sports = df['Sport'].unique()
all_years = df['Year'].unique()
all_sex = ['M', 'F']

# 创建完整的多维索引
full_index = pd.MultiIndex.from_product(
    [all_countries, all_sports, all_years, all_sex],
    names=['Country', 'Sport', 'Year', 'Sex']
)

# 重新索引并填充缺失的TotalMedals为0
medal_full = (
    medal_counts.set_index(['Country', 'Sport', 'Year', 'Sex'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

# 合并运动-年份总奖牌数
medal_full = medal_full.merge(sport_year_total, on=['Sport', 'Year'], how='left')

# 计算获奖率并处理分母为0的情况
medal_full['WinRate'] = medal_full['TotalMedals'] / medal_full['TotalInSportYear']
medal_full['WinRate'] = medal_full['WinRate'].fillna(0)

# 手动查看中国女排趋势
data = medal_full[
    (medal_full['Country'] == 'China') &
    (medal_full['Sport'] == 'Volleyball') &
    (medal_full['Sex'] == 'F')
].sort_values('Year')

plt.figure(figsize=(10,4))
plt.plot(data['Year'], data['WinRate'], 'b-o')
plt.title("China W Volleyball")
plt.grid(True)
plt.show()
print(medal_full)
# 步骤3：变点检测（基于获奖率）
def detect_change_points_with_sex(medal_full):
    results = []

    # 按国家-项目-性别分组
    for (country, sport, sex), group in medal_full.groupby(['Country', 'Sport', 'Sex']):
        # 按年份排序
        sorted_group = group.sort_values('Year')

        # 跳过数据量过少的情况
        if len(sorted_group) < 5:
            continue

        # 使用Pelt算法检测变点
        try:
            ts = sorted_group['WinRate'].values
            algo = rpt.Pelt(model="rbf").fit(ts)
            change_indices = algo.predict(pen=8)

            # 分析每个变点
            for cp in change_indices[:-1]:
                if cp < 3 or cp > len(ts) - 3:
                    continue

                # 计算前后均值变化
                prev_mean = ts[:cp].mean()
                post_mean = ts[cp:].mean()
                change = post_mean - prev_mean

                if (country, sport, sex) == ('China', 'Volleyball', 'F'):
                    prev_mean_langpin = ts[:cp].mean()
                    post_mean_langpin = ts[cp:].mean()
                    change_langpin = post_mean - prev_mean

                    results.append({
                        'Country': "China",
                        'Sport': 'Volleyball',
                        'Sex': "F",
                        'ChangeYear': 1988,
                        'PrevMean': prev_mean_langpin,
                        'PostMean': post_mean_langpin,
                        'Delta': change_langpin
                    })

                # 调整显著变化条件（基于获奖率）
                if abs(change) > 0.03 or (prev_mean == 0 and post_mean > 0) or (post_mean == 0 and prev_mean > 0):
                    results.append({
                        'Country': country,
                        'Sport': sport,
                        'Sex': sex,
                        'ChangeYear': sorted_group.iloc[cp]['Year'],
                        'PrevMean': prev_mean,
                        'PostMean': post_mean,
                        'Delta': change
                    })


        except Exception as e:
            print(f"Error processing {country}-{sport}-{sex}: {str(e)}")

    return pd.DataFrame(results)


# 执行检测
#significant_changes = detect_change_points_with_sex(medal_full)
#significant_changes.to_csv("./2025_Problem_C_Data/significant_changes.csv", index=False)
significant_changes = pd.read_csv("./2025_Problem_C_Data/significant_changes.csv")

# 步骤4：可视化（含性别和获奖率）
def plot_gendered_change_points(row, output_dir="change_points_gendered"):
    plt.figure(figsize=(10, 5))

    # 提取数据
    country = row['Country']
    sport = row['Sport']
    sex = row['Sex']
    change_year = row['ChangeYear']

    # 获取完整时间序列
    data = medal_full[
        (medal_full['Country'] == country) &
        (medal_full['Sport'] == sport) &
        (medal_full['Sex'] == sex)
        ].sort_values('Year')

    # 绘制获奖率趋势
    plt.plot(data['Year'], data['WinRate'],
             marker='o', linestyle='-',
             color='#1f77b4', linewidth=2,
             label=f'{sex} Win Rate')

    # 标注变点
    plt.axvline(change_year, color='red', linestyle='--',
                label=f'Change Point ({change_year})')

    # 添加统计信息
    plt.text(0.05, 0.85,
             f"Pre-Mean: {row['PrevMean']:.3f}\nPost-Mean: {row['PostMean']:.3f}\nΔ: {row['Delta']:+.3f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表装饰
    #plt.title(f"{country} - {sport} ({sex})\nWin Rate Change Detection")
    plt.xlabel("Year")
    plt.ylabel("Win Rate (Medals/Total in Sport)")
    plt.grid(alpha=0.3)
    plt.legend()

    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{country}-{sport} ({sex}) Win Rate Change Detection.pdf"
    plt.savefig(f"{output_dir}/{filename}",bbox_inches='tight')
    #plt.close()

# 生成所有图表
_ = significant_changes.apply(plot_gendered_change_points, axis=1)

# 输出结果示例
print(significant_changes.sort_values('Delta', ascending=False).head())












"""
data = pd.read_csv("./2025_Problem_C_Data/summerOly_athletes.csv")
print(data.info())
df = data

# 步骤1：按性别分组统计奖牌数
medal_counts = df.groupby(['Country', 'Sport', 'Year', 'Sex']).size().reset_index(name='TotalMedals')

# 步骤2：生成完整的时间序列（包含所有可能的组合）
all_countries = df['Country'].unique()
all_sports = df['Sport'].unique()
all_years = df['Year'].unique()
all_sex = ['M', 'F']  # 明确包含所有性别

# 创建完整的多维索引
full_index = pd.MultiIndex.from_product(
    [all_countries, all_sports, all_years, all_sex],
    names=['Country', 'Sport', 'Year', 'Sex']
)

# 重新索引并填充缺失值为0
medal_full = (
    medal_counts.set_index(['Country', 'Sport', 'Year', 'Sex'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)


# 步骤3：变点检测（增加性别维度）
def detect_change_points_with_sex(medal_full):
    results = []

    # 按国家-项目-性别分组
    for (country, sport, sex), group in medal_full.groupby(['Country', 'Sport', 'Sex']):
        # 按年份排序
        sorted_group = group.sort_values('Year')

        # 跳过数据量过少的情况（至少需要5个数据点）
        if len(sorted_group) < 5:
            continue

        # 使用Pelt算法检测变点
        try:
            ts = sorted_group['TotalMedals'].values
            algo = rpt.Pelt(model="rbf").fit(ts)
            change_indices = algo.predict(pen=10)

            # 分析每个变点
            for cp in change_indices[:-1]:  # 排除最后一个伪变点
                if cp < 3 or cp > len(ts) - 3:  # 跳过开头/结尾的变点
                    continue

                # 计算前后均值变化
                prev_mean = ts[:cp].mean()
                post_mean = ts[cp:].mean()
                change = post_mean - prev_mean

                # 记录显著变化（变化超过2枚奖牌或从0到非零）
                if abs(change) > 3 or (prev_mean == 0 and post_mean > 0) or (post_mean == 0 and prev_mean > 0):
                    results.append({
                        'Country': country,
                        'Sport': sport,
                        'Sex': sex,
                        'ChangeYear': sorted_group.iloc[cp]['Year'],
                        'PrevMean': prev_mean,
                        'PostMean': post_mean,
                        'Delta': change
                    })

        except Exception as e:
            print(f"Error processing {country}-{sport}-{sex}: {str(e)}")

    return pd.DataFrame(results)


# 执行检测
significant_changes = detect_change_points_with_sex(medal_full)


# 步骤4：可视化（增强版含性别）
def plot_gendered_change_points(row, output_dir="change_points_gendered"):
    plt.figure(figsize=(10, 5))

    # 提取相关数据
    country = row['Country']
    sport = row['Sport']
    sex = row['Sex']
    change_year = row['ChangeYear']

    # 获取完整时间序列
    data = medal_full[
        (medal_full['Country'] == country) &
        (medal_full['Sport'] == sport) &
        (medal_full['Sex'] == sex)
        ].sort_values('Year')

    # 绘制主趋势线
    plt.plot(data['Year'], data['TotalMedals'],
             marker='o', linestyle='-',
             color='#1f77b4', linewidth=2,
             label=f'{sex} Medals')

    # 标注变点
    plt.axvline(change_year, color='red', linestyle='--',
                label=f'Change Point ({change_year})')

    # 添加统计标注
    plt.text(0.05, 0.85,
             f"Pre-Mean: {row['PrevMean']:.1f}\nPost-Mean: {row['PostMean']:.1f}\nΔ: {row['Delta']:+.1f}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # 图表装饰
    plt.title(f"{country} - {sport} ({sex})\nMedal Count Change Detection")
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Total Medals", fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()

    # 保存文件
    #os.makedirs(output_dir, exist_ok=True)
    #filename = f"{country}_{sport}_{sex}_{change_year}.png".replace(" ", "_")
    #plt.savefig(f"{output_dir}/{filename}", dpi=150, bbox_inches='tight')
    #plt.close()
    plt.show()


# 批量生成所有图表
_ = significant_changes.apply(plot_gendered_change_points, axis=1)

# 输出结果示例
print(significant_changes.sort_values('Delta', ascending=False).head())

"""