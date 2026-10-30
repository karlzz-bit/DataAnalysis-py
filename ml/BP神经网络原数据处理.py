import pandas as pd
from scipy.io import arff
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

data_path = r'Rice_Cammeo_Osmancik.arff'

try:
    # 读取 .arff 文件
    data, meta = arff.loadarff(data_path)

    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 将 byte 类型的字符串转换为正常的字符串类型
    if 'Class' in df.columns:
        df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))

    print(df.head())
except Exception as e:
    print("An error occurred while reading the .arff file:")
    traceback.print_exc()

# 数据基本信息和描述性统计
print(df.info())
print(df.describe())

# 面积和周边的直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Area'], kde=True, bins=30)
plt.title('Distribution of Area')

plt.subplot(1, 2, 2)
sns.histplot(df['Perimeter'], kde=True, bins=30)
plt.title('Distribution of Perimeter')
plt.show()

# 各品种的大米特征分布
plt.figure(figsize=(16, 10))
features = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df, x=feature, hue='Class', kde=True)
    plt.title(f'Distribution of {feature} by Class')
plt.tight_layout()
plt.show()

# 散点图矩阵
sns.pairplot(df, hue='Class')
plt.show()

# 热力图（相关矩阵）
plt.figure(figsize=(10, 8))

# 选择数值列计算相关性矩阵
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
