import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取出生率数据和元数据
data_folder = r"D:\_Program\_codes\zz\Fertility rate, total (births per woman) [WB Gender Portal]"
fertility_data_path = os.path.join(data_folder, "Fertility rate, total (births per woman).csv")
metadata_path = os.path.join(data_folder, "Metadata.csv")

fertility_data = pd.read_csv(fertility_data_path)
metadata = pd.read_csv(metadata_path)

# 打印列名，确认是否有 'Country Name'
print("Columns in dataset:", fertility_data.columns)

# 如果 'Country Name' 不存在，可能是 'Economy'
if 'Country Name' not in fertility_data.columns and 'Economy' in fertility_data.columns:
    fertility_data.rename(columns={'Economy': 'Country Name'}, inplace=True)

# 再次检查
print(fertility_data.head())

# 从fertility_data中筛选出中国的数据
china_data = fertility_data[fertility_data['Country Name'] == 'China']

# 确保列为年份，并筛选出从1960年到2020年的出生率数据
years = [str(year) for year in range(1960, 2021)]
china_fertility = china_data[years].transpose()  # 转置，使年份在横轴
china_fertility.columns = ['Fertility Rate']
china_fertility.index = china_fertility.index.astype(int)  # 将索引转为整数

# 处理缺失值（如果有）
china_fertility.dropna(inplace=True)

# 绘制出生率曲线
plt.figure(figsize=(10, 6))
plt.plot(china_fertility.index, china_fertility['Fertility Rate'], marker='o', color='b', linestyle='-')
plt.title('China Fertility Rate (Births per Woman) Over Time')
plt.xlabel('Year')
plt.ylabel('Fertility Rate (Births per Woman)')
plt.grid(True)
plt.xticks(range(1960, 2021, 5))  # 设置x轴为5年一个刻度
plt.show()
