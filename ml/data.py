import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 假设你的数据文件是CSV格式，且位于当前工作目录下，名为'data.csv'
# 第一列是索引列（如果有），最后一列是目标变量

# 读取数据
data = pd.read_csv('HousingData.csv')

# 假设除了最后一列是目标变量'target'外，其他都是特征
X = data.iloc[:, :-14]
y = data.iloc[:, 14]

# 数据预处理：标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 岭回归
ridge = Ridge(alpha=1.0)  # 你可以通过交叉验证来找到最佳的alpha值
ridge.fit(X_train, y_train)
y_ridge_pred = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
print(f"Ridge Regression MSE: {ridge_mse}")

# Lasso回归
lasso = Lasso(alpha=0.1)  # 同样，你可以通过交叉验证来找到最佳的alpha值
lasso.fit(X_train, y_train)
y_lasso_pred = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_lasso_pred)
print(f"Lasso Regression MSE: {lasso_mse}")

# 你还可以进一步查看模型的系数等属性
print("Ridge coefficients:", ridge.coef_)
print("Lasso coefficients:", lasso.coef_)