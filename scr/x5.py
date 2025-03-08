import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
# 自定义橙黄渐变颜色映射
colors = ["#FFD700", "#FFB347"]  # 深橙 -> 橙 -> 亮黄
cmap = LinearSegmentedColormap.from_list("OrangeYellow", colors, N=256)

# 生成超参数组合
hyperparam1 = [0.1, 0.2, 0.3, 0.4]
hyperparam2 = [3, 6, 8, 4]

# 生成随机数值
np.random.seed(42)
values = np.random.uniform(low=0.88, high=0.93, size=(4, 4))

# 创建画布
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成坐标网格
x_labels = np.arange(len(hyperparam1))
y_labels = np.arange(len(hyperparam2))
x_pos, y_pos = np.meshgrid(x_labels, y_labels, indexing='ij')

# 颜色归一化
norm = plt.Normalize(vmin=values.min(), vmax=values.max())
colors_array = cmap(norm(values.flatten()))  # 展平为一维颜色数组

# 绘制三维柱状图
bars = ax.bar3d(x_pos.flatten(),
                y_pos.flatten(),
                np.zeros_like(x_pos).flatten(),
                dx=0.6,
                dy=0.6,
                dz=values.flatten(),
                color=colors_array,        # 应用自定义渐变色
                edgecolor='#666666',       # 浅灰色边框
                linewidth=0.3,             # 细边框线
                alpha=0.9)                 # 轻微透明度

# 坐标轴设置
ax.set_xticks(x_labels)
ax.set_xticklabels(hyperparam1, fontsize=9)
ax.set_yticks(y_labels)
ax.set_yticklabels(hyperparam2, fontsize=9)
ax.set_xlabel('Learning Rate', fontsize=11, labelpad=12)
ax.set_ylabel('Max Depth', fontsize=11, labelpad=12)
ax.set_zlabel('Accuracy', fontsize=11, labelpad=12)

# 添加顶部横向颜色条
cax = fig.add_axes([0.25, 0.88, 0.5, 0.03])  # 调整位置参数
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Performance Score', fontsize=10, labelpad=5)
cbar.ax.tick_params(labelsize=8)

# 优化布局
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
ax.view_init(elev=28, azim=-50)
plt.title("XGBoost Hyperparameter Analysis\n(Orange-Yellow Gradient)",
          y=0.96, fontsize=13)

plt.show()
#plt.show()
#plt.title("XGBoost Hyperparameter Tuning", y=1.0, fontsize=14)
#plt.title("Knee-NSGA2-XGBoost Hyperparameter Combinations", y=1.0, fontsize=14)
plt.savefig("Knee-NSGA2-XGBoost Hyperparameter Combinations.pdf")