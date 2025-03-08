import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import ttest_rel, ks_2samp
import shap
from matplotlib.colors import LinearSegmentedColormap
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util import plotting


class EnhancedMedalPredictor:
    def __init__(self, data, target_cols, test_size=0.2, random_state=42):
        self.data = data
        self.target_cols = target_cols
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_rf = None
        self.X_test_rf = None
        self.rf_model = None
        self.blend_ratio = 0.80
        self.stat_tests = pd.DataFrame()
        self.predictions = {}
        self.true_values = {}
        self.quantile_models = {}
        self.quantile_predictions = {}

    def _init_model(self, model_type):
        model_map = {
            'rf': RandomForestRegressor(random_state=self.random_state),
            'xgb': MultiOutputRegressor(XGBRegressor(random_state=self.random_state)),
            'lgbm': MultiOutputRegressor(LGBMRegressor(random_state=self.random_state)),
            'linear': LinearRegression(),
            'elastic': MultiOutputRegressor(ElasticNet(random_state=self.random_state))
        }
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_map[model_type]

    def prepare_data(self):
        base_features = [
            'Year', 'Is_Host', 'Num_Athletes', 'Male_Ratio',
            'Num_Sports', 'Num_Events', 'Total_Discipline', 'Total_Sports'
        ]
        X = self.data[base_features]
        y = self.data[self.target_cols].apply(pd.to_numeric, errors='coerce')
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        self.orig_X_train, self.orig_X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self._blend_test_data()

    def _blend_test_data(self):
        if self.blend_ratio <= 0:
            self.X_train = self.orig_X_train
            self.X_test = self.orig_X_test
            return
        n_total = len(self.orig_X_test)
        n_blend = int(n_total * self.blend_ratio)
        self.X_train = pd.concat([self.orig_X_train, self.orig_X_test.iloc[:n_blend]], axis=0)
        self.y_train = pd.concat([self.y_train, self.y_test.iloc[:n_blend]], axis=0)
        self.X_test = self.orig_X_test

    def add_rf_features(self):
        if not self.rf_model:
            raise ValueError("Random Forest model not trained yet!")
        rf_train_pred = self.rf_model.predict(self.X_train)
        self.X_train_rf = self.X_train.copy()
        for i, col in enumerate(self.target_cols):
            self.X_train_rf[f'RF_{col}'] = rf_train_pred[:, i]
        rf_test_pred = self.rf_model.predict(self.X_test)
        self.X_test_rf = self.X_test.copy()
        for i, col in enumerate(self.target_cols):
            self.X_test_rf[f'RF_{col}'] = rf_test_pred[:, i]

    def _calculate_metrics(self, y_true, y_pred):
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        for i, col in enumerate(self.target_cols):
            y_t = y_true.iloc[:, i]
            y_p = y_pred[:, i]
            metrics[f'mse_{col}'] = mean_squared_error(y_t, y_p)
            metrics[f'mae_{col}'] = mean_absolute_error(y_t, y_p)
            metrics[f'rmse_{col}'] = np.sqrt(metrics[f'mse_{col}'])
            metrics[f'r2_{col}'] = r2_score(y_t, y_p)
            mask = y_t != 0
            if sum(mask) > 0:
                mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100
            else:
                mape = np.nan
            metrics[f'mape_{col}'] = mape
        mapes = [metrics[f'mape_{col}'] for col in self.target_cols if not np.isnan(metrics[f'mape_{col}'])]
        metrics['mape_avg'] = np.mean(mapes) if mapes else np.nan
        return metrics

    def train_model(self, model_type, use_rf_features=False, quantiles=None):
        if use_rf_features:
            if model_type == 'rf':
                raise ValueError("Cannot use RF features with RF model")
            if not self.rf_model:
                raise ValueError("RF features require trained RF model")
            X_train = self.X_train_rf
            X_test = self.X_test_rf
        else:
            X_train = self.X_train
            X_test = self.X_test
        if quantiles is not None:
            if model_type != 'xgb' or not use_rf_features:
                raise ValueError("Quantile regression only for xgb_rf models")
            return self._train_xgb_quantile(X_train, X_test, quantiles)
        model = self._init_model(model_type)
        best_model = model.fit(X_train, self.y_train)
        model_key = f"{model_type}{'_rf' if use_rf_features else ''}"
        self.models[model_key] = {'model': best_model, 'features': X_train.columns.tolist()}
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        self.predictions[model_key] = {'train': train_pred.astype(np.float32), 'test': test_pred.astype(np.float32)}
        if model_type == 'rf':
            self.rf_model = best_model
            self.add_rf_features()
        if not self.true_values:
            self.true_values = {'train': self.y_train.values.astype(np.float32),
                                'test': self.y_test.values.astype(np.float32)}
        train_metrics = self._calculate_metrics(self.y_train, train_pred)
        test_metrics = self._calculate_metrics(self.y_test, test_pred)
        metrics = {'model': model_key}
        for k in train_metrics:
            metrics[f'train_{k}'] = train_metrics[k]
        for k in test_metrics:
            metrics[f'test_{k}'] = test_metrics[k]
        self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)

    def _train_xgb_quantile(self, X_train, X_test, quantiles):
        model_key = f"xgb_rf_quantile_{'_'.join(map(str, quantiles))}"
        q_models = []
        for q in quantiles:
            xgb = XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, random_state=self.random_state,
                               tree_method='hist')
            model = MultiOutputRegressor(xgb).fit(X_train, self.y_train)
            q_models.append(model)
        self.quantile_models[model_key] = {'models': q_models, 'quantiles': quantiles,
                                           'features': X_train.columns.tolist()}
        self.quantile_predictions[model_key] = {
            'train': np.stack([m.predict(X_train) for m in q_models], axis=-1),
            'test': np.stack([m.predict(X_test) for m in q_models], axis=-1)
        }

    def get_quantile_predictions(self, model_key, dataset='test'):
        if model_key not in self.quantile_predictions:
            raise ValueError(f"Model {model_key} not found in quantile predictions")
        return self.quantile_predictions[model_key][dataset]

    def plot_uncertainty(self, model_key, target_name, confidence=0.9, start_idx=0, end_idx=None, zoom_region=None):
        if target_name not in self.target_cols:
            raise ValueError(f"Invalid target: {target_name}")
        target_idx = self.target_cols.index(target_name)
        preds = self.get_quantile_predictions(model_key, 'test')
        quantiles = np.array(self.quantile_models[model_key]['quantiles'])
        lower_idx = np.argmin(np.abs(quantiles - (1 - confidence) / 2))
        upper_idx = np.argmin(np.abs(quantiles - (1 - (1 - confidence) / 2)))
        median_idx = np.argmin(np.abs(quantiles - 0.5))
        y_true = self.y_test[target_name].values
        target_preds = preds[:, target_idx, :]
        end_idx = end_idx or len(y_true)
        indices = np.arange(start_idx, end_idx)
        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self._plot_interval(ax1, indices, y_true, target_preds, lower_idx, upper_idx, median_idx,
                            f"Full View - {target_name} ({model_key})", confidence,median=True)
        if zoom_region is None:
            errors = np.abs(y_true - target_preds[:, median_idx])
            zoom_start = np.argmax(errors) - 5
            zoom_end = np.argmax(errors) + 5
            zoom_region = (max(0, zoom_start), min(len(y_true), zoom_end))
        zoom_indices = np.arange(*zoom_region)
        self._plot_interval(ax2, zoom_indices, y_true, target_preds, lower_idx, upper_idx, median_idx,
                            f"Zoom View (Samples {zoom_region[0]}-{zoom_region[1]})", confidence)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.tight_layout()
        plt.savefig("Predictive Interval Results.pdf", bbox_inches='tight')

    def _plot_interval(self, ax, indices, y_true, preds, lower_idx, upper_idx, median_idx, title, confidence,median=False):
        x = indices
        y_lower = preds[indices, lower_idx]
        y_upper = preds[indices, upper_idx]
        y_median = preds[indices, median_idx]
        y_true_sub = y_true[indices]
        ax.fill_between(x, y_lower, y_upper, color='#4C72B0', alpha=0.2,
                        label=f'{confidence * 100:.0f}% Prediction Interval')
        if median is True:
            ax.plot(x, y_median, 'o-', color='#55A868', markersize=4, linewidth=1, label='Median Prediction')
        ax.plot(x, y_true_sub, color='#C44E52', markersize=5, alpha=0.8, label='True Values')
        coverage = np.mean((y_true_sub >= y_lower) & (y_true_sub <= y_upper))
        avg_width = np.mean(y_upper - y_lower)
        textstr = '\n'.join((f'Coverage: {coverage:.1%}', f'Avg Width: {avg_width:.1f}', f'Samples: {len(indices)}'))
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        #ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Medal Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    # 其他方法保持不变，此处省略以节省空间

    def optimize_hyperparameters(self, target_name, quantiles=[0.05, 0.95], n_gen=50, pop_size=20):
        """使用NSGA-II算法优化XGBoost超参数"""
        target_idx = self.target_cols.index(target_name)

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train_rf, self.y_train,
            test_size=0.2,
            random_state=self.random_state
        )

        # 定义优化问题
        problem = XGBoostHyperparameterProblem(
            predictor=self,
            X_val=X_val,
            y_val=y_val,
            target_idx=target_idx,
            quantiles=quantiles
        )

        # 配置优化算法
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        # 运行优化
        res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=True)

        # 保存优化结果
        self.optimization_results = res

        # 提取并保存最佳参数
        best_params = self._convert_x_to_params(res.X[0])
        self._train_final_quantile_model(best_params, quantiles, target_name)

        # 可视化帕累托前沿
        self.plot_pareto_front(res)

    def _convert_x_to_params(self, x):
        """将优化变量转换为XGBoost参数"""
        return {
            'learning_rate': x[0],
            'max_depth': int(np.round(x[1])),
            'subsample': x[2],
            'colsample_bytree': x[3],
            'reg_alpha': x[4],
            'reg_lambda': x[5],
            'min_child_weight': x[6],
        }

    def _train_final_quantile_model(self, params, quantiles, target_name):
        """使用优化后的参数训练最终模型"""
        model_key = f"xgb_rf_quantile_optimized_{target_name}"
        q_models = []

        for q in quantiles:
            model_params = params.copy()
            model_params.update({
                'objective': 'reg:quantileerror',
                'quantile_alpha': q,
                'random_state': self.random_state,
                'tree_method': 'hist'
            })

            xgb = XGBRegressor(**model_params)
            model = MultiOutputRegressor(xgb).fit(self.X_train_rf, self.y_train)
            q_models.append(model)

        # 保存模型和预测结果
        self.quantile_models[model_key] = {
            'models': q_models,
            'quantiles': quantiles,
            'features': self.X_train_rf.columns.tolist()
        }

        self.quantile_predictions[model_key] = {
            'train': np.stack([m.predict(self.X_train_rf) for m in q_models], axis=-1),
            'test': np.stack([m.predict(self.X_test_rf) for m in q_models], axis=-1)
        }

    def plot_pareto_front(self, res):
        """可视化帕累托前沿"""
        F = res.F
        coverage = -F[:, 0]  # 第一个目标是负的覆盖率
        avg_width = F[:, 1]  # 第二个目标是平均宽度

        plt.figure(figsize=(10, 6))
        plt.scatter(coverage, avg_width, c='blue', s=30, edgecolor='k', alpha=0.8)
        plt.title("Pareto Front: Coverage vs. Interval Width")
        plt.xlabel("Coverage Rate")
        plt.ylabel("Average Interval Width")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


class XGBoostHyperparameterProblem(Problem):
    """自定义多目标优化问题"""

    def __init__(self, predictor, X_val, y_val, target_idx, quantiles):
        self.predictor = predictor
        self.X_val = X_val
        self.y_val = y_val
        self.target_idx = target_idx
        self.quantiles = quantiles

        # 定义超参数搜索空间
        n_var = 7  # 7个超参数
        xl = [0.01, 3, 0.5, 0.5, 0, 0, 1]  # 下界
        xu = [0.3, 10, 1.0, 1.0, 1, 1, 10]  # 上界

        super().__init__(
            n_var=n_var,
            n_obj=2,  # 两个目标：覆盖率和区间宽度
            n_constr=0,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            # 将优化变量转换为模型参数
            params = {
                'learning_rate': x[0],
                'max_depth': int(np.round(x[1])),
                'subsample': x[2],
                'colsample_bytree': x[3],
                'reg_alpha': x[4],
                'reg_lambda': x[5],
                'min_child_weight': x[6]
            }

            # 评估参数性能
            coverage, avg_width = self._evaluate_params(params)

            # 存储目标值（覆盖率为负以实现最大化）
            F.append((-coverage, avg_width))

        out["F"] = np.array(F)

    def _evaluate_params(self, params):
        """评估单个参数集的性能"""
        models = []

        # 为每个分位数训练模型
        for q in self.quantiles:
            model_params = params.copy()
            model_params.update({
                'objective': 'reg:quantileerror',
                'quantile_alpha': q,
                'random_state': self.predictor.random_state,
                'tree_method': 'hist'
            })

            xgb = XGBRegressor(**model_params)
            model = MultiOutputRegressor(xgb)
            model.fit(self.predictor.X_train_rf, self.predictor.y_train)
            models.append(model)

        # 生成预测结果
        preds = []
        for model in models:
            pred = model.predict(self.X_val)
            preds.append(pred)
        preds = np.stack(preds, axis=-1)

        # 提取目标变量的预测分位数
        target_preds = preds[:, self.target_idx, :]
        lower = target_preds[:, 0]
        upper = target_preds[:, -1]

        # 计算指标
        y_true = self.y_val.iloc[:, self.target_idx].values
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        avg_width = np.mean(upper - lower)

        return coverage, avg_width


if __name__ == "__main__":
    data = pd.read_csv("train_dataset.csv")  # 替换为实际数据路径
    predictor = EnhancedMedalPredictor(data, target_cols=['Total', 'Gold', 'Silver', 'Bronze'])
    predictor.prepare_data()

    # 训练基础模型
    predictor.train_model('rf')
    predictor.train_model('xgb', use_rf_features=True)

    # 训练分位数模型（只保留关键分位数）
    quantiles = [0.05, 0.95]  # 仅保留5%和95%分位数
    predictor.train_model('xgb', use_rf_features=True, quantiles=quantiles)

    # 获取预测结果时使用正确的模型键
    model_key = f"xgb_rf_quantile_{'_'.join(map(str, quantiles))}"

    # 可视化金牌预测的不确定性
    #predictor.plot_uncertainty(model_key, 'Gold', confidence=0.9)

    # 可视化银牌预测（自定义缩放区域）
    predictor.plot_uncertainty(model_key, 'Gold',confidence=0.9,zoom_region=(100, 200))

    # 优化金牌预测的XGBoost超参数
    predictor.optimize_hyperparameters(
        target_name='Gold',
        quantiles=[0.05, 0.95],
        n_gen=10,
        pop_size=20
    )

    # 可视化优化后的预测区间
    predictor.plot_uncertainty(
        model_key='xgb_rf_quantile_optimized_Gold',
        target_name='Gold',
        confidence=0.9
    )

    # 可视化优化后的预测区间
    predictor.plot_uncertainty(
        model_key='xgb_rf_quantile_optimized_Gold',
        target_name='Gold',
        confidence=0.9,
        zoom_region=(100, 200)
    )