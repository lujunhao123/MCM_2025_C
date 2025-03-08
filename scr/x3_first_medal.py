import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import get_code
country_mapping,noc_mapping,country_codes = get_code()
# 数据加载和预处理
class DataPreprocessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.country_mapping = {
            'Soviet Union': 'Russia',
            'West Germany': 'Germany',
            'East Germany': 'Germany',
            'Yugoslavia': 'Serbia',
            'Czechoslovakia': 'Czech Republic',
            'Bohemia': 'Czech Republic',
            'Russian Empire': 'Russia',
            'United Team of Germany': 'Germany',
            'Unified Team': 'Russia',
            'Serbia and Montenegro': 'Serbia',
            'Netherlands Antilles': 'Netherlands',
            'Virgin Islands': 'United States',
            'ROC': 'Russia',
        }
        self.preprocess_data()

    def preprocess_data(self):
        # 国家名称映射
        self.data['NOC'] = self.data['NOC'].replace(self.country_mapping)

        # 过滤掉1896年获奖的国家
        data_1896_winners = self.data[(self.data['Year'] == 1896) & (self.data['Total'] > 0)]['NOC'].unique()
        self.data = self.data[~self.data['NOC'].isin(data_1896_winners)]

        # 创建特征集
        self.country_features = self.data.groupby('NOC').agg({
            'Year': ['count', 'min', 'max'],
            'Female_Ratio': 'mean',
            'Num_Athletes': 'sum',
            'Num_Sports': 'sum',
            'Num_Events': 'sum',
            'Total': 'sum'
        }).reset_index()

        self.country_features.columns = [
            'NOC', 'Participations', 'First_Year', 'Last_Year', 'Total_Athletes',
            'Female_Ratio', 'Total_Sports', 'Total_Events', 'Total_Medals'
        ]

        # 添加是否获奖的列
        self.country_features['Has_Won'] = self.country_features['Total_Medals'] > 0

        # 编码国家名称
        self.le = LabelEncoder()
        self.country_features['NOC'] = self.le.fit_transform(self.country_features['NOC'])

        # 划分训练集和测试集
        self.X = self.country_features.drop(columns=['Has_Won', 'Total_Medals'])
        self.y = self.country_features['Has_Won']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )


# 模型训练和预测
class ModelComparator:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=1000, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        self.results = {}
        self.metrics = {}

    def train_and_predict(self):
        for name, model in self.models.items():
            if name == 'SVM':
                # 标准化数据
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # 存储预测结果
            self.results[name] = y_pred_proba

            # 计算分类指标
            self.metrics[name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred),
                'ROC AUC': roc_auc_score(self.y_test, y_pred_proba)
            }

    def get_results(self):
        return self.results

    def get_metrics(self):
        return self.metrics


# 可视化结果
class ResultVisualizer:
    def __init__(self, results, metrics, country_features, le, X_test):
        self.results = results
        self.metrics = metrics
        self.country_features = country_features
        self.le = le
        self.X_test = X_test  # 添加 X_test 属性

    def save_results_to_csv(self, file_path='model_results.csv'):
        # 将预测结果保存到 CSV 文件
        results_df = self.country_features.copy()
        results_df['NOC'] = self.le.inverse_transform(results_df['NOC'])  # 解码国家名称
        for model_name, probabilities in self.results.items():
            results_df[f'Win_Probability_{model_name}'] = pd.Series(index=self.X_test.index, data=probabilities)
        results_df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

    def plot_svm_histogram(self):
        # 只绘制 SVM 模型的直方图
        svm_probabilities = self.results['SVM']
        plt.figure(figsize=(8, 6))
        sns.set_palette(sns.color_palette('muted'))  # 使用淡雅配色
        sns.histplot(svm_probabilities, bins=20, color='skyblue', kde=True, edgecolor='black',linewidth=1.6)
        #plt.title('Win Probability Distribution (SVM)', fontsize=16)
        plt.xlabel('Win Probability', fontsize=14)
        plt.ylabel('Number of Countries', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("Win Probability Distribution (SVM).pdf")

    def plot_top_10_countries(self):
        # 获取 SVM 模型的预测结果
        svm_probabilities = self.results['SVM']

        # 将预测结果扩展到整个数据集
        full_probabilities = pd.Series(index=self.country_features.index, dtype=float)
        full_probabilities[self.X_test.index] = svm_probabilities  # 仅填充测试集的预测结果
        self.country_features['Win_Probability_SVM'] = full_probabilities

        # 筛选未获奖的国家并排序
        first_time_winners = self.country_features[self.country_features['Has_Won'] == False].sort_values(
            by='Win_Probability_SVM', ascending=False
        )
        first_time_winners['NOC'] = self.le.inverse_transform(first_time_winners['NOC'])  # 解码国家名称
        top_10_countries = first_time_winners.head(10)

        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        sns.set_palette(sns.color_palette('pastel'))  # 使用淡雅配色
        sns.barplot(x=top_10_countries['NOC'], y=top_10_countries['Win_Probability_SVM'], edgecolor='black', linewidth=1.2)
        plt.axhline(y=0.5, color='darkred', linestyle='--', linewidth=1.5, label='Win Probability = 0.5')
        plt.legend(loc='upper right', fontsize=12)  # 添加图例
        #plt.title('Top 10 Countries with Highest Probability of Winning a Medal for the First Time (SVM)', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Win Probability', fontsize=14)
        #plt.xticks(rotation=45)  # 旋转国家名称以便更好地显示
        plt.xticks(fontsize=11)  # 旋转国家名称以便更好地显示
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("Top 10 Countries with Highest Probability of Winning a Medal for the First Time (SVM).pdf")

        # 打印前十名国家及其概率
        print("Top 10 Countries with Highest Probability of Winning a Medal for the First Time:")
        print(top_10_countries[['NOC', 'Win_Probability_SVM']].reset_index(drop=True))


if __name__ == "__main__":
    # 数据预处理
    preprocessor = DataPreprocessor('./2025_Problem_C_Data/data.csv')

    # 模型训练和预测
    comparator = ModelComparator(preprocessor.X_train, preprocessor.y_train, preprocessor.X_test, preprocessor.y_test)
    comparator.train_and_predict()
    results = comparator.get_results()
    metrics = comparator.get_metrics()

    # 可视化结果
    visualizer = ResultVisualizer(results, metrics, preprocessor.country_features, preprocessor.le, preprocessor.X_test)
    visualizer.save_results_to_csv()  # 保存结果到 CSV
    visualizer.plot_svm_histogram()  # 绘制 SVM 模型的直方图
    visualizer.plot_top_10_countries()  # 绘制前十名国家的柱状图