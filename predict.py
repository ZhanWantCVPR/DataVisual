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


# 此处使用相对路径 “./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv” 便于项目移动
df = pd.read_csv('./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 指定分类 类别 (churn 0:代表不流失 and 1:代表流失)
categorical_cols = df.select_dtypes(include=['category', 'object']).columns
# OneHot编码
encoder = OneHotEncoder(sparse=False)  # Setting drop='first' to avoid multicollinearity
encoded_data = encoder.fit_transform(df[categorical_cols])
# 根据编码数据创建DataFrame
# DataFrame 是 Pandas 库中最常用的数据结构之一，用于存储和操作二维标签化的数据，类似于电子表格或数据库中的表格。
# DataFrame 可以看作是由多个 Series 对象组成的数据表，每一列都是一个 Series，而每一行代表一个样本或观察。
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
# 删除原始类别
df.drop(columns=categorical_cols, inplace=True)
# 重置索引
df.reset_index(drop=True, inplace=True)
# 链接原始数据和编码数据
df = pd.concat([df, encoded_df], axis=1)
df.drop('Churn_No', axis=1, inplace=True)
print(df.head())
# 重命名 churn_yes to churn
df.rename(columns={'Churn_Yes': 'Churn'}, inplace=True)

# 切分数据 X and y
X = df.drop('Churn', axis=1)
y = df['Churn']
# data into train and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 初始化一个空列表，用于存储模型评估分数
model_scores = []

# 用于评估的模型 列表
models = [
    ('Random Forest', RandomForestClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__max_depth': [None, 10, 20]}),  # Add hyperparameters for Random Forest
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Gradient Boosting
    # ('Support Vector Machine', SVC(random_state=42, class_weight='balanced'),
    #     {'model__C': [0.1, 1, 10],
    #      'model__gamma': ['scale', 'auto']}),  # Add hyperparameters for SVM
    ('Logistic Regression', LogisticRegression(random_state=42, class_weight='balanced'),
        {'model__C': [0.1, 1, 10],
         'model__penalty': ['l1', 'l2']}),  # Add hyperparameters for Logistic Regression
    ('K-Nearest Neighbors', KNeighborsClassifier(),
        {'model__n_neighbors': [3, 5, 7],
         'model__weights': ['uniform', 'distance']}),  # Add hyperparameters for KNN
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {'model__max_depth': [None, 10, 20],
         'model__min_samples_split': [2, 5, 10]}),  # Add hyperparameters for Decision Tree
    ('Ada Boost', AdaBoostClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Ada Boost
    # ('XG Boost', XGBClassifier(random_state=42),
    #     {'model__n_estimators': [50, 100, 200],
    #      'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for XG Boost
    ('Naive Bayes', GaussianNB(), {})  # No hyperparameters for Naive Bayes
]

best_model = None
best_accuracy = 0.0

# 评估上述模型
for name, model, param_grid in models:
    # 为每个模型创建 pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', model)
    ])

    # 使用 GridSearchCV 调整超参数
    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)
        grid_search.fit(X_train, y_train)
        pipeline = grid_search.best_estimator_

    # 训练数据对接 pipeline
    pipeline.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = pipeline.predict(X_test)

    # 计算精度分数
    accuracy = accuracy_score(y_test, y_pred)

    # 将模型name和精度添加到列表中
    model_scores.append({'Model': name, 'Accuracy': accuracy})

    # 将列表转换为 DataFrame
    scores_df = pd.DataFrame(model_scores)
    # 输出评估指标
    print("Model:", name)
    print("Test Accuracy:", accuracy.round(3),"%")
    print()

    # 判断是否是最佳结果
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# 检索并输出最佳模型
print("Best Model:")
print("Test Accuracy:", best_accuracy)
print("Model Pipeline:", best_model, "with accuracy", best_accuracy.round(2), "%")

# 绘制结果图
colors = sns.color_palette('pastel', n_colors=len(scores_df))

# 获取 Bar 和 Score
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model', y='Accuracy', data=scores_df, palette=colors)

# 添加文本于Bar之上
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# 可视化
plt.title('Model Scores')
plt.xlabel('Models')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()