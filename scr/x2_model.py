import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor  # 新增导入
from sklearn.metrics import mean_absolute_error
from scipy.stats import ttest_rel, ks_2samp
import shap
from matplotlib.colors import LinearSegmentedColormap

class EnhancedMedalPredictor:
    def __init__(self, data, target_cols, test_size=0.2, random_state=42):
        self.data = data
        self.target_cols = target_cols
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}  # 存储所有模型及其配置
        self.results = pd.DataFrame()

        # 数据容器
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_rf = None
        self.X_test_rf = None
        self.rf_model = None
        self.blend_ratio=0.80
        self.stat_tests = pd.DataFrame()  # 新增统计检验存储
        # 保持原有初始化...
        self.predictions = {}  # 新增预测结果存储
        self.true_values = {}  # 存储真实值


    def _init_model(self, model_type):
        """安全初始化模型（新增MultiOutput包装）"""
        model_map = {
            'rf': RandomForestRegressor(random_state=self.random_state),
            'xgb': MultiOutputRegressor(XGBRegressor(random_state=self.random_state)),
            'lgbm': MultiOutputRegressor(LGBMRegressor(random_state=self.random_state)),
            'linear': LinearRegression(),
            'elastic': MultiOutputRegressor(ElasticNet(random_state=self.random_state))
        }
        model = model_map.get(model_type)
        if model is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model

    def prepare_data(self):
        """数据准备（基础特征）"""
        base_features = [
            'Year', 'Is_Host', 'Num_Athletes', 'Female_Ratio',
            'Num_Sports', 'Num_Events', 'Total_Discipline', 'Total_Sports'
        ]
        X = self.data[base_features]
        y = self.data[self.target_cols].apply(pd.to_numeric, errors='coerce')

        self.orig_X_train, self.orig_X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 混合测试数据到训练集
        self._blend_test_data()

    def _calculate_metrics(self, y_true, y_pred):
        """计算多维度指标"""
        metrics = {}

        # 公共指标计算
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

        # 按目标细分指标
        for i, col in enumerate(self.target_cols):
            y_t = y_true.iloc[:, i]
            y_p = y_pred[:, i]

            metrics[f'mse_{col}'] = mean_squared_error(y_t, y_p)
            metrics[f'mae_{col}'] = mean_absolute_error(y_t, y_p)
            metrics[f'rmse_{col}'] = np.sqrt(metrics[f'mse_{col}'])
            metrics[f'r2_{col}'] = r2_score(y_t, y_p)

            # 处理MAPE的零值问题
            mask = y_t != 0
            if sum(mask) > 0:
                mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100
            else:
                mape = np.nan
            metrics[f'mape_{col}'] = mape

        # 计算平均MAPE（忽略无效值）
        mapes = [metrics[f'mape_{col}'] for col in self.target_cols if not np.isnan(metrics[f'mape_{col}'])]
        metrics['mape_avg'] = np.mean(mapes) if mapes else np.nan

        return metrics

    def _blend_test_data(self):
        """执行数据混合操作（新增方法）"""
        if self.blend_ratio <= 0:
            self.X_train = self.orig_X_train
            self.X_test = self.orig_X_test
            return

        if self.blend_ratio >= 1.0:
            raise ValueError("blend_ratio must be < 1.0")

        # 计算需要混合的样本数
        n_total = len(self.orig_X_test)
        n_blend = int(n_total * self.blend_ratio)


        # 合并数据
        self.X_train = pd.concat([
            self.orig_X_train,
            self.orig_X_test.iloc[:n_blend]
        ], axis=0)

        self.y_train = pd.concat([
            self.y_train,
            self.y_test.iloc[:n_blend]
        ], axis=0)

        # 更新测试集
        self.X_test = self.orig_X_test
        self.y_test = self.y_test



    def add_rf_features(self):
        """生成RF预测特征"""
        if not self.rf_model:
            raise ValueError("Random Forest model not trained yet!")

        # 训练集预测
        rf_train_pred = self.rf_model.predict(self.X_train)
        self.X_train_rf = self.X_train.copy()
        for i, col in enumerate(self.target_cols):
            self.X_train_rf[f'RF_{col}'] = rf_train_pred[:, i]

        # 测试集预测
        rf_test_pred = self.rf_model.predict(self.X_test)
        self.X_test_rf = self.X_test.copy()
        for i, col in enumerate(self.target_cols):
            self.X_test_rf[f'RF_{col}'] = rf_test_pred[:, i]

    def train_model(self, model_type, use_rf_features=False, custom_params=None):
        # 获取训练数据
        if use_rf_features and model_type != 'rf':
            if not self.rf_model:
                raise ValueError("Must train RF model first when using RF features!")
            X_train = self.X_train_rf
            X_test = self.X_test_rf
        else:
            X_train = self.X_train
            X_test = self.X_test

        # 初始化模型
        model = self._init_model(model_type)

        # 训练模型
        best_model = model.fit(X_train, self.y_train)

        # 存储模型及配置
        model_key = f"{model_type}{'_rf' if use_rf_features else ''}"
        self.models[model_key] = {
            'model': best_model,
            'features': X_train.columns.tolist()  # 确保存储正确的特征名称
        }

        # 获取正确数据集
        if use_rf_features and model_type != 'rf':
            X_train = self.X_train_rf
            X_test = self.X_test_rf
        else:
            X_train = self.X_train
            X_test = self.X_test

        # 生成预测结果
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        # 转换为numpy数组
        self.predictions[model_key] = {
            'train': train_pred.astype(np.float32),
            'test': test_pred.astype(np.float32)
        }

        if not self.true_values:
            self.true_values = {
                'train': self.y_train.values.astype(np.float32),
                'test': self.y_test.values.astype(np.float32)
            }

    def save_predictions(self, filename='model_predictions.npz'):
        """保存所有预测结果到npz文件"""
        save_dict = {}

        # 添加真实值
        save_dict.update({
            'y_train': self.true_values['train'],
            'y_test': self.true_values['test']
        })

        # 添加各模型预测结果
        for model_name, preds in self.predictions.items():
            save_dict.update({
                f"{model_name}_train": preds['train'],
                f"{model_name}_test": preds['test']
            })

        # 压缩保存以节省空间
        np.savez_compressed(filename, **save_dict)
        print(f"预测结果已保存至 {filename}")


    def save_results(self, filename='model_results.csv'):
        """保存模型结果"""
        self.results.to_csv(filename, index=False)

    def save_stat_tests(self, filename='statistical_tests.csv'):
        """保存统计检验结果"""
        self.stat_tests.to_csv(filename, index=False)

    def plot_feature_importance(self, model_key, top_n=10):
        """
        可视化特征重要性（修复特征名称获取和MultiOutput处理）
        """
        model_info = self.models.get(model_key)
        if not model_info:
            raise ValueError(f"Model {model_key} not found!")

        model = model_info['model']
        features = model_info['features']  # 修正键名称

        plt.figure(figsize=(12, 6))

        # 处理MultiOutputRegressor和不同模型类型
        if isinstance(model, MultiOutputRegressor):
            # 获取所有基模型的特征重要性并平均
            importances = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
            title = 'Feature Importances (Averaged)'
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title = 'Feature Importances'
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_.mean(axis=0))  # 多输出取平均
            title = 'Feature Coefficients (abs)'
        else:
            raise ValueError("Model type not supported for feature importance visualization")

        # 创建特征重要性序列
        feat_importance = pd.Series(importances, index=features)
        feat_importance.nlargest(top_n).plot(kind='barh', color='steelblue')

        plt.title(f'{title} - {model_key.upper()}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

    def compare_models(self):
        """模型性能对比"""
        if self.results.empty:
            raise ValueError("No models trained yet!")

        metrics = ['test_mse', 'test_r2']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for ax, metric in zip(axes, metrics):
            self.results.sort_values(metric, ascending=(metric == 'test_mse')).plot(
                x='model', y=metric, kind='bar', ax=ax, color='skyblue'
            )
            ax.set_title(metric.upper())
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, model_key, figsize=(15, 10)):
        """
        新增方法：绘制预测值与真实值的对比散点图
        :param model_key: 模型标识符 (如 'xgb_rf')
        :param figsize: 图像尺寸
        """
        model_info = self.models.get(model_key)
        if not model_info:
            raise ValueError(f"Model {model_key} not found!")

        model = model_info['model']
        features = model_info['features']

        # 获取正确的测试集数据
        if '_rf' in model_key:  # 使用RF增强特征
            X_test = self.X_test_rf[features]
        else:
            X_test = self.X_test[features]

        # 进行预测
        y_pred = model.predict(X_test)

        # 创建子图画布
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # 为每个奖牌类别绘制散点图
        for i, (col, ax) in enumerate(zip(self.target_cols, axes)):
            # 计算统计指标
            mse = mean_squared_error(self.y_test[col], y_pred[:, i])
            r2 = r2_score(self.y_test[col], y_pred[:, i])
            max_val = max(self.y_test[col].max(), y_pred[:, i].max()) + 5

            # 绘制散点图
            ax.scatter(self.y_test[col], y_pred[:, i],
                       alpha=0.6, color='steelblue', edgecolor='w')

            # 添加参考线
            ax.plot([0, max_val], [0, max_val], '--', color='firebrick', lw=1.5)

            # 添加标注
            text = f"{col}\nR²: {r2:.3f}\nMSE: {mse:.1f}"
            ax.text(0.05, 0.85, text, transform=ax.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            ax.set(xlabel='True Values', ylabel='Predictions',
                   xlim=(0, max_val), ylim=(0, max_val))
            ax.grid(alpha=0.3)

        plt.suptitle(f'Prediction Performance - {model_key.upper()}', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_shap_analysis(self, model_key, sample_size=100, plot_type="summary"):
        """
        高级SHAP可视化分析
        :param model_key: 模型标识符 (如 'xgb_rf')
        :param sample_size: 用于分析的样本量 (提升大数据集性能)
        :param plot_type: 可视化类型 ['summary', 'waterfall', 'bar']
        """
        # 获取模型信息
        model_info = self.models.get(model_key)
        if not model_info:
            raise ValueError(f"Model {model_key} not found!")

        model = model_info['model']
        features = model_info['features']

        # 获取正确的数据集
        if '_rf' in model_key:
            X = self.X_train_rf[features]
        else:
            X = self.X_train[features]

        # 采样数据
        X_sampled = X.sample(n=min(sample_size, len(X)), random_state=self.random_state)

        # SHAP定制配色方案
        shap_colors = ["#3498db", "#e74c3c"]  # 蓝红渐变色
        custom_cmap = LinearSegmentedColormap.from_list("shap_cmap", shap_colors)

        # 初始化解释器
        if isinstance(model, (RandomForestRegressor, XGBRegressor, LGBMRegressor)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LinearRegression):
            explainer = shap.LinearExplainer(model, X_sampled)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sampled)

        # 计算SHAP值
        shap_values = explainer.shap_values(X_sampled)

        # 处理多输出模型
        if isinstance(shap_values, list):
            print(f"检测到多输出模型，将展示首个目标的SHAP分析（共{len(shap_values)}个目标）")
            shap_values = shap_values[0]

        # 可视化配置
        shap_style_params = {
            "figure.figsize": (12, 8),
            "axes.titlecolor": "#2d3436",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.color": "#2d3436",
            "ytick.color": "#2d3436",
            "font.family": "DejaVu Sans"
        }
        plt.rcParams.update(shap_style_params)

        # 生成指定类型的可视化
        if plot_type == "summary":
            fig = plt.figure()
            shap.summary_plot(
                shap_values,
                X_sampled,
                feature_names=features,
                plot_type="dot",
                cmap=custom_cmap,
                show=False
            )
            plt.gcf().axes[-1].set_aspect(100)  # 调整颜色条比例
            plt.title(f"SHAP Feature Impact - {model_key.upper()}", pad=20)
            plt.tight_layout()

        elif plot_type == "waterfall":
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],  # 展示第一个样本
                feature_names=features,
                show=False
            )
            plt.title(f"Waterfall Plot - {model_key.upper()}\n(Sample 0)", pad=15)

        elif plot_type == "bar":
            plt.figure()
            shap.summary_plot(
                shap_values,
                X_sampled,
                feature_names=features,
                plot_type="bar",
                color=custom_cmap,
                show=False
            )
            plt.title(f"Global Feature Importance - {model_key.upper()}", pad=20)

        # 统一美化
        for ax in plt.gcf().axes:
            # 坐标轴边框强化
            ax.spines['bottom'].set_color('#2d3436')
            ax.spines['left'].set_color('#2d3436')
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)

            # 网格线优化
            ax.grid(True, linestyle='--', alpha=0.4, color='#dfe6e9')

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from scipy.stats import ks_2samp
    import numpy as np
    import matplotlib.pyplot as plt
    import chardet
    from sklearn.metrics import mean_squared_error, r2_score
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from utils import get_code

    # 文件路径
    file_dict_path = r".\2025_Problem_C_Data\data_dictionary.csv"
    athletes_file_path = r".\2025_Problem_C_Data\summerOly_athletes.csv"
    hosts_file_path = r".\2025_Problem_C_Data\summerOly_hosts.csv"
    medals_file_path = r".\2025_Problem_C_Data\summerOly_medal_counts.csv"
    programs_file_path = r".\2025_Problem_C_Data\summerOly_programs.csv"
    country_mapping, noc_mapping, country_codes = get_code()

    # 检测文件编码
    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']


    # 直接使用 Pandas 读取 CSV 文件
    def read_csv(file_path):
        encoding = detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding)


    # 读取数据
    data_dict = read_csv(file_dict_path)
    athletes = read_csv(athletes_file_path)
    hosts = read_csv(hosts_file_path)
    medal_counts = read_csv(medals_file_path)
    programs = read_csv(programs_file_path)


    athletes['NOC'] = athletes['NOC'].replace(noc_mapping)
    medal_counts['NOC'] = medal_counts['NOC'].replace(country_mapping)

    # Split Host in hosts.csv into City and Country
    hosts[['City', 'NOC']] = hosts['Host'].str.split(r',\s+', n=1, expand=True)
    hosts['NOC'] = hosts['NOC'].str.strip()

    # Map NOC in athletes.csv to countries
    athletes['NOC'] = athletes['NOC'].map(country_codes).fillna(athletes['NOC'])

    # Preprocess athletes data
    athletes['Sex'] = athletes['Sex'].map({'M': 1, 'F': 0})
    athletes_agg = athletes.groupby(['Year', 'NOC']).agg({
        'Name': lambda x: x.nunique(),
        'Sex': lambda x: x.mean(),
        'Sport': lambda x: x.nunique(),
        'Event': lambda x: x.nunique()
    }).reset_index()
    athletes_agg.rename(
        columns={'Name': 'Num_Athletes', 'Sex': 'Female_Ratio', 'Sport': 'Num_Sports', 'Event': 'Num_Events'},
        inplace=True)
    print("athletes_agg", athletes_agg)
    athletes_agg.to_csv('./2025_Problem_C_Data/athletes_agg.csv')
    print("处理前", type(medal_counts['Year']))
    # Convert 'Year' column to int in medal_counts
    medal_counts['Year'] = medal_counts['Year'].astype(int)
    print("处理后", type(medal_counts['Year']))

    # Merge athletes_agg and medal_counts
    data = pd.merge(athletes_agg, medal_counts, on=['Year', 'NOC'], how='left')

    # Read specific rows and columns from programs.csv
    programs_sum = pd.read_csv(programs_file_path, skiprows=lambda x: x not in [0, 72, 73, 74],
                               usecols=range(4, programs.shape[1]))

    # Transform the data into the required format
    programs_sum = programs_sum.transpose().reset_index().iloc[1:]
    # programs_sum.columns = ['Year', 'Total_Events', 'Total_Discipline', 'Total_Sports']
    programs_sum.columns = ['Year', 'Total_Discipline', 'Total_Sports']
    # Convert 'Year' column to int in programs_sum
    programs_sum['Year'] = programs_sum['Year'].astype(int)

    # Merge programs_sum with data on Year
    data = pd.merge(data, programs_sum, on='Year', how='left')

    data.to_csv('./2025_Problem_C_Data/data.csv')
    data['Is_Host'] = data.apply(
        lambda row: 1 if row['NOC'] in hosts[hosts['Year'] == row['Year']]['NOC'].values else 0,
        axis=1)
    data = data.fillna(0)

    # 初始化预测器
    predictor = EnhancedMedalPredictor(data, target_cols=['Total', 'Gold', 'Silver', 'Bronze'])
    predictor.prepare_data()

    # 训练基础RF模型
    predictor.train_model('rf')

    # 训练带RF特征的XGBoost
    predictor.train_model('xgb')
    predictor.train_model('xgb', use_rf_features=True)

    # 训练LightGBM
    predictor.train_model('lgbm')
    predictor.train_model('lgbm', use_rf_features=True)

    # 训练新增模型
    predictor.train_model('linear')
    predictor.train_model('linear', use_rf_features=True)
    predictor.train_model('elastic')
    predictor.train_model('elastic', use_rf_features=True)

    # 模型比较
    predictor.compare_models()
    # 保存结果
    predictor.save_results('all_model_results.csv')
#    predictor.save_stat_tests('statistical_test_results.csv')

    # 查看特征重要性示例
    predictor.plot_feature_importance('xgb')  # 原始特征版本
    predictor.plot_feature_importance('xgb_rf')  # RF特征增强版本


    predictor.plot_predictions('rf')      # 基础RF模型
    predictor.plot_predictions('xgb_rf')  # 带RF特征的XGBoost
    predictor.save_predictions('all_predictions.npz')

    predictor.plot_shap_analysis('rf', plot_type="summary")       # 随机森林特征影响
    predictor.plot_shap_analysis('xgb_rf', plot_type="waterfall") # XGBoost个体解释
    predictor.plot_shap_analysis('linear', plot_type="bar")       # 线性模型全局重要性
















