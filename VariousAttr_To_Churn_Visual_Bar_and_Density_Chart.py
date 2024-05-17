# ----- completed by Yu Zheng ----
# ----- 2024年4月10日17点42分 ------
# ----- have a nice day ! --------
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 各个属性与流失之间的关系 可视化 VariousAttr_To_Churn_Visual_Bar_and_Density_Chart
# -----------------数据集预处理---------------------------------------------
# 读取数据集
# 此处使用相对路径 “./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv” 便于项目移动
df = pd.read_csv('./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 控制台打印所有列
# 取消列数的限制 显示所有列
pd.set_option('display.max_columns', None)
# head:n 默认值为5
print(df.head())
# 检视数据集
df.info()
# 转换数据格式
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# 数据清洗
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
# 获取数字列
print(df.describe().T)
# 确定唯一值 循环检查数据集中不含 int 或 float 的唯一值
for col in df.columns:
    if df[col].dtype != 'int64' and df[col].dtype != 'float64':
        print(f'{col} : {df[col].unique()}')
# 检查缺失值
print(df.isnull().sum())
# 缺失值的热力图绘制 Visual_Fig_1
print("--------df-Null-values-heatmap--------------")
sns.heatmap(df.isnull())
# 绘制缺失值热力图
plt.show()
# -----------------数据集预处理---------------------------------------------


# -----------------------------------可视化分析----------------------------
# 定义 Yes:red, No:blue
colors = {'Yes': 'red', 'No': 'blue'}
palette = {0:'blue', 1:'red'}
# 条形图 各个属性内部 不同的值与 流失与否之间的可视化 结果
for i, predictor in enumerate(df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure'])):
    plt.figure(i, figsize=(12, 5))
    sns.countplot(data=df, x=predictor, hue='Churn', palette=colors)
    plt.title(predictor)
    for rect in plt.gca().patches:
        height = rect.get_height()
        plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
plt.show()

# 绘制 tenure (使用期)图
# Churn 是一个英文词汇,主要用于描述客户流失或变动的现象。 电信行业: 定义:在电信行业,“churn” 表示用户切换到其他电信运营商的行为。
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']
plt.figure(figsize=(10, 6))
plt.hist([churned['tenure'], not_churned['tenure']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])

plt.title(' Tenure by Churn')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在 Bar 顶部添加文字说明
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

# 绘制 MonthlyCharges(月消费) 图
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']
plt.figure(figsize=(10, 6))
plt.hist([churned['MonthlyCharges'], not_churned['MonthlyCharges']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])
plt.title('MonthlyCharges by Churn')
plt.xlabel('MonthlyCharges')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 在 Bar 顶部添加文字说明
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

# 绘制 TotalCharges(累计消费)图
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']
plt.figure(figsize=(10, 6))
plt.hist([churned['TotalCharges'], not_churned['TotalCharges']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])
plt.title(' TotalCharges by Churn')
plt.xlabel('TotalCharges')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 在 Bar 顶部添加文字说明
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
# 可视化
plt.show()

# Density 图分析
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 绘制 MonthlyCharges
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, alpha=0.5, ax=axes[0])
axes[0].set_title('Density Plot of Monthly Charges by Churn Status')
axes[0].set_xlabel('Monthly Charges')
axes[0].set_ylabel('Density')

# 绘制 TotalCharges
sns.kdeplot(data=df, x="TotalCharges", hue="Churn", fill=True, alpha=0.5, ax=axes[1])
axes[1].set_title('Density Plot of Total Charges by Churn Status')
axes[1].set_xlabel('Total Charges')
axes[1].set_ylabel('Density')
# 可视化
plt.tight_layout()
plt.show()

# 结论(月费较低的客户更容易流失)
