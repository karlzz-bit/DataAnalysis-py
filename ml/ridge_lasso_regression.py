import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 假设你的数据文件是CSV格式，且位于当前工作目录下，名为'data.csv'
# 第一列是索引列（如果有），最后一列是目标变量

# 读取数据
data = pd.read_csv('bostan.csv')

# 假设除了最后一列是目标变量'target'外，其他都是特征
X = data.iloc[:, :-14]
y = data.iloc[:, 14]

# 数据预处理：标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 添加交叉验证来选择最佳alpha值
from sklearn.model_selection import GridSearchCV

# 岭回归参数网格搜索
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)

# Lasso参数网格搜索
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
lasso_cv = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train, y_train)

# 使用最佳参数训练模型
ridge = Ridge(alpha=ridge_cv.best_params_['alpha'])
lasso = Lasso(alpha=lasso_cv.best_params_['alpha'])

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# 预测和评估
y_ridge_pred = ridge.predict(X_test)
y_lasso_pred = lasso.predict(X_test)

# 评估指标
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} 评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

print_metrics(y_test, y_ridge_pred, "岭回归")
print_metrics(y_test, y_lasso_pred, "Lasso回归")

# 特征重要性分析
feature_names = X.columns
def print_feature_importance(model, model_name):
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': abs(model.coef_)
    })
    importance = importance.sort_values('coefficient', ascending=False)
    print(f"\n{model_name} 特征重要性 (前5个):")
    print(importance.head())

print_feature_importance(ridge, "岭回归")
print_feature_importance(lasso, "Lasso回归")

# 保存模型结果
results = pd.DataFrame({
    'True Values': y_test,
    'Ridge Predictions': y_ridge_pred,
    'Lasso Predictions': y_lasso_pred
})
results.to_csv('model_predictions.csv', index=False)
