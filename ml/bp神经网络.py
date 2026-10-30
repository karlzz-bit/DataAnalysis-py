import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 读取 .arff 文件并转换为 DataFrame
from scipy.io import arff

data_path = r'Rice_Cammeo_Osmancik.arff'
data, meta = arff.loadarff(data_path)
df = pd.DataFrame(data)

# 将 byte 类型的字符串转换为正常的字符串类型
df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))
# 分离特征和目标变量
X = df.drop(columns=['Class'])
y = df['Class'].apply(lambda x: 1 if x == 'Cammeo' else 0)  # 将类标签转换为二进制

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 构建模型
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 二分类问题使用 sigmoid 激活函数

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
# 获取预测概率
y_pred_prob = model.predict(X_test)

# 获取预测标签（0 或 1）
y_pred = (y_pred_prob > 0.5).astype(int)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 计算 ROC AUC 分数
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC AUC Score: {roc_auc}')
