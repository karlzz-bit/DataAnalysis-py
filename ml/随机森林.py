import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集，指定分隔符为分号
student_mat = pd.read_csv('student-mat.csv', sep=';')
student_por = pd.read_csv('student-por.csv', sep=';')

# 初步查看数据
print(student_mat.head())
print(student_mat.info())
print(student_mat.describe())

# 检查前几行数据
print(student_mat.head())

# 检查数据列是否包含 'G3'
print(student_mat.columns)

# 描述性统计分析
print(student_mat.describe())

# 检查缺失值
missing_values = student_mat.isnull().sum()
print(missing_values[missing_values > 0])

# 类别变量的分布
categorical_columns = student_mat.select_dtypes(include='object').columns
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=student_mat)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# 数值变量的分布
numeric_columns = student_mat.select_dtypes(include='number').columns

# 对每个数值型列绘制直方图
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(student_mat[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 数值变量之间的关系
sns.pairplot(student_mat[numeric_columns])
plt.show()

# 类别变量和数值变量的关系
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y='G3', data=student_mat)
    plt.title(f'Final Grades (G3) by {col}')
    plt.xticks(rotation=45)
    plt.show()
# 只选择数值型列用于计算相关矩阵
numeric_columns = student_mat.select_dtypes(include='number').columns
correlation_matrix = student_mat[numeric_columns].corr()
# 计算相关矩阵
correlation_matrix = student_mat[numeric_columns].corr()

# 可视化相关矩阵
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 探索性数据分析 - 可视化
# 绘制成绩的分布图
sns.histplot(student_mat['G3'], kde=True)
plt.title('Distribution of Final Grades (G3) in Math')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.show()

# 性别与成绩的关系
sns.boxplot(x='sex', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Gender in Math')
plt.xlabel('Gender')
plt.ylabel('Final Grade (G3)')
plt.show()

# 年龄与成绩的关系
sns.scatterplot(x='age', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Age in Math')
plt.xlabel('Age')
plt.ylabel('Final Grade (G3)')
plt.show()

# 家庭住址类型与成绩的关系
sns.boxplot(x='address', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Address in Math')
plt.xlabel('Address')
plt.ylabel('Final Grade (G3)')
plt.show()

# 父母教育程度与成绩的关系

sns.boxplot(x='Medu', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Mother\'s Education in Math')
plt.xlabel('Mother\'s Education Level')
plt.ylabel('Final Grade (G3)')
plt.show()

sns.boxplot(x='Fedu', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Father\'s Education in Math')
plt.xlabel('Father\'s Education Level')
plt.ylabel('Final Grade (G3)')
plt.show()
# 学习时间与成绩的关系
sns.boxplot(x='studytime', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Study Time in Math')
plt.xlabel('Study Time')
plt.ylabel('Final Grade (G3)')
plt.show()

# 缺勤次数与成绩的关系
sns.scatterplot(x='absences', y='G3', data=student_mat)
plt.title('Final Grades (G3) by Absences in Math')
plt.xlabel('Absences')
plt.ylabel('Final Grade (G3)')
plt.show()

#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#
# # 初步数据分析和可视化
numeric_columns = student_mat.select_dtypes(include='number').columns

# 绘制数值变量的分布
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(student_mat[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# 数据预处理
# 处理分类变量
student_mat = pd.get_dummies(student_mat, drop_first=True)

# 定义特征和目标变量
X = student_mat.drop(['G1', 'G2', 'G3'], axis=1)
y = student_mat['G3']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建和训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 模型预测
y_pred = rf.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 可视化真实值与预测值的比较
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(15, 5))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 假设 rf 是你的随机森林模型
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]  # 特征重要性排序的索引

# 获取特征名称或索引作为 x 轴坐标
feature_names = X.columns  # 假设 X 是你的特征数据的 DataFrame
sorted_feature_names = feature_names[indices]

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[indices])
plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.title('Feature Importances in Random Forest Model')
plt.tight_layout()
plt.show()
