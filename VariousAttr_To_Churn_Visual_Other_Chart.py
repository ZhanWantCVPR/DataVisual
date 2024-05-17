# ----- completed by Yu Zheng ----
# ----- 2024年4月10日17点42分 ------
# ----- have a nice day ! --------
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -->数据集预处理
# 数据集位置
data = pd.read_csv('./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 数据集信息
data.info()
# 清理和转换 "TotalCharges" 列的数据，将其中的非数值型数据转换为数值型数据，
# 并将转换后的结果更新到原始的列中，以便后续的数据处理和分析。
data["TotalCharges"] = (pd.to_numeric(data["TotalCharges"],errors="coerce"))
data.isnull().sum()
# 识别分类变量的唯一值数量
data.nunique()
# 清理包含缺失值的数据，以便后续的数据分析和建模。在数据分析和机器学习中，
# 经常会遇到缺失值的情况，如果不处理缺失值，可能会影响数据分析的准确性。
data.dropna(inplace=True)
data.drop("customerID",axis="columns",inplace=True)

# -->分析
# 类数组创建
categ_columns = list(data.select_dtypes(include = ['object']).columns)
# 饼图 gender Partner Dependents PhoneService
fig, axes = plt.subplots(1, 4, figsize=(16, 8), facecolor="lightgray")
for i, column in enumerate(categ_columns[:4]):
    ax = axes[i]
    d = data[column].value_counts()
    ax.pie(d, labels=d.values,autopct="%1.1f%%",shadow=True)
    ax.set_title(column,size=18)
    ax.legend(d.index)

# 饼图 MultipleLines InternetService OnlineSecurity OnlineBackup
fig, axes = plt.subplots(1, 4, figsize=(16, 8), facecolor="lightgray")
for i, column in enumerate(categ_columns[4:8]):
    ax = axes[i]
    d = data[column].value_counts()
    ax.pie(d, labels=d.values,autopct="%1.1f%%")
    ax.set_title(column,size=18)
    ax.legend(d.index,loc="best")

# 饼图 DeviceProtection TechSupport StreamingTV StreamingMovies
fig, axes = plt.subplots(1, 4, figsize=(16, 8), facecolor="lightgray")
for i, column in enumerate(categ_columns[8:12]):
    ax = axes[i]
    d = data[column].value_counts()
    ax.pie(d, labels=d.values,autopct="%1.1f%%")
    ax.set_title(column,size=18)
    ax.legend(d.index)

# 饼图 Contract PaperlessBilling PaymentMethod Churn
fig, axes = plt.subplots(1, 4, figsize=(16, 8), facecolor="lightgray")
for i, column in enumerate(categ_columns[12:]):
        ax = axes[i]
        d = data[column].value_counts()
        ax.pie(d, labels=d.values,autopct="%1.1f%%")
        ax.set_title(column,size=18)
        ax.legend(d.index)
# 可视化上述饼图
plt.show()

# 分布图 (Churn)
fig, ax = plt.subplots(facecolor="lightblue")
d = data["Churn"].value_counts()
ax.pie(d,autopct='%1.1f%%', startangle=90,labels=d.values)
ax.legend(d.index)
ax.set_title("Distribution of Churn")
centre_circle = plt.Circle((0,0),0.4,fc='lightblue')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal');

# 交叉表 Contract-Count-churn
fig , ax = plt.subplots(facecolor="lightgray")
pd.crosstab(data["Contract"],data["Churn"]).plot(kind="barh",ax=ax)
ax.set(xlabel="Count")


# 绘制箱线图 Churn-tenure
sns.boxplot(data=data,x="Churn",y="tenure",width=0.3)

# 交叉表 用于展示两个或多个因素之间的关系 进一步理解和呈现数据的关系和分布。PaymentMethod-Count-churn
fig , ax = plt.subplots(facecolor="lightgray")
pd.crosstab(data["Churn"],data["PaymentMethod"]).plot(kind="bar",color=["m","b","green","brown"],ax=ax,stacked=True,)
ax.set(xlabel="Count")
# 可视化
plt.show()

# 分布图(Tenure)
fig , ax = plt.subplots(1,2,figsize=(12,5))
sns.histplot(data["tenure"],ax=ax[0])
sns.kdeplot(data["tenure"],fill=True,ax=ax[1])
sns.rugplot(data["tenure"],height=0.05)
fig.suptitle("Distribution of Tenure",size=18);

# 分布图(TotalCharges)
fig , ax = plt.subplots(1,2,figsize=(12,5))
sns.histplot(data["TotalCharges"],ax=ax[0],color='Green')
sns.kdeplot(data["TotalCharges"],fill=True,ax=ax[1],color='Green')
sns.rugplot(data["TotalCharges"],height=0.05,color='Green')
fig.suptitle("Distribution of TotalCharges",size=18)

# 分布图(MonthlyCharges)
fig , ax = plt.subplots(1,2,figsize=(12,5))
sns.histplot(data["MonthlyCharges"],ax=ax[0],color='Red')
sns.kdeplot(data["MonthlyCharges"],fill=True,ax=ax[1],color='Red')
sns.rugplot(data["MonthlyCharges"],height=0.05,color='Red')
fig.suptitle("Distribution of MonthlyCharges",size=18)

# 分布图可视化
plt.show()