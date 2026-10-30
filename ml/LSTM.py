import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('wldata.csv')

# 检查列名
print("数据列名：", data.columns)

# 统一列名格式
data.columns = data.columns.str.strip().str.lower()

# 检查必需列
required_columns = ['date', '场地1', '场地2', 'volume']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"数据集中缺少必要列：{col}")

# 数据处理
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data['day'] = data['date'].dt.day
data['route'] = data['场地1'] + '→' + data['场地2']
data = data.groupby(['route', 'day'])['volume'].sum().reset_index()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['volume']])

# 创建监督学习数据集
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# 设置 look_back 参数
look_back = 3
X, y = create_dataset(scaled_data, look_back)

# 划分训练集和测试集
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 调整输入数据的形状 [样本数, 时间步长, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建 LSTM 模型
model = Sequential([
    Input(shape=(look_back, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# 预测部分优化
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-01-31')
days = (end_date - start_date).days + 1
dates = pd.date_range(start_date, end_date)
routes = ['DC14→DC10', 'DC20→DC35', 'DC25→DC62']

# 生成预测数据
predictions = []
for route in routes:
    last_days = scaled_data[-look_back:].copy()
    last_days = np.reshape(last_days, (1, look_back, 1))

    for day in range(1, days + 1):
        prediction = model.predict(last_days)
        predicted_volume = scaler.inverse_transform(prediction)[0][0]

        date = start_date + pd.Timedelta(days=day - 1)
        predictions.append({
            'date': date,
            'route': route,
            'predicted_volume': predicted_volume
        })

        # 更新输入数据
        new_input = np.array([[prediction[0][0]]])
        last_days = np.append(last_days[:, 1:, :], [new_input], axis=1)

# 转为 DataFrame
predictions_df = pd.DataFrame(predictions)
print(predictions_df)
