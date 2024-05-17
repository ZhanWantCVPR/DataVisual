# ----- completed by Yu Zheng ----
# ----- 2024年4月10日17点42分 ------
# ----- have a nice day ! --------
# import imblearn
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
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

# 上采样
sm = SMOTEENN()
X_res, y_res = sm.fit_resample(X, y)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_res, y_res, test_size=0.2)

# 初始化一个空列表，用于存储模型评估分数
model_scores = []

# 模型列表
models = [
    ('Random Forest', RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100, 200],
         'max_depth': [None, 10, 20]}),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {'n_estimators': [50, 100, 200],
         'learning_rate': [0.05, 0.1, 0.5]}),
    # ('Support Vector Machine', SVC(random_state=42, class_weight='balanced'),
    #     {'C': [0.1, 1, 10],
    #      'gamma': ['scale', 'auto']}),
    ('Logistic Regression', LogisticRegression(random_state=42, class_weight='balanced'),
        {'C': [0.1, 1, 10],
         'penalty': ['l1', 'l2']}),
    ('K-Nearest Neighbors', KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7],
         'weights': ['uniform', 'distance']}),
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5, 10]}),
    ('Ada Boost', AdaBoostClassifier(random_state=42),
        {'n_estimators': [50, 100, 200],
         'learning_rate': [0.05, 0.1, 0.5]}),
    # ('XG Boost', XGBClassifier(random_state=42),
    #     {'max_depth': randint(3, 6),
    #      'learning_rate': uniform(0.01, 0.2),
    #      'n_estimators': randint(100, 300),
    #      'subsample': uniform(0.8, 0.2)}),
    ('Naive Bayes', GaussianNB(), {})
]

best_model = None
best_accuracy = 0.0

for name, model, param_grid in models:
    # 创建 pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Feature Scaling
        ('model', model)
    ])

    # 超参数调整 XG Boost
    if name == 'XG Boost':
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                           n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1)
        random_search.fit(Xr_train, yr_train)
        pipeline = random_search.best_estimator_
    # 超参数调整 其他模型
    elif param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=2, verbose=0)
        grid_search.fit(Xr_train, yr_train)
        pipeline = grid_search.best_estimator_

    # pipeline
    pipeline.fit(Xr_train, yr_train)

    # 测试集 测试
    y_pred = pipeline.predict(Xr_test)

    # 计算 accuracy
    accuracy = accuracy_score(yr_test, y_pred)

    # name 和 accuracy 添加列表
    model_scores.append({'Model': name, 'Accuracy': accuracy})

    # 转换 DataFrame
    scores_df = pd.DataFrame(model_scores)

    # 性能指标
    print("Model:", name)
    print("Test Accuracy:", accuracy.round(3),"%")
    print()

    # 判断最值
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# 检索最佳效果模型
print("Best Model:")
print("Test Accuracy:", best_accuracy)
print("Model Pipeline:", best_model, "with accuracy", best_accuracy.round(2), "%")

# 颜色 Bar
colors = sns.color_palette('pastel', n_colors=len(scores_df))

# Bar plot 和 Score
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model', y='Accuracy', data=scores_df, palette=colors)

# 添加 Bar name
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Model Scores')
plt.xlabel('Models')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


