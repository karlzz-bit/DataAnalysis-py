import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

data = pd.read_excel("D:/Code/codes/python/Online Retail.xlsx", sheet_name='Sheet1')


# 数据预处理
data.dropna(inplace=True)
transactions = data.groupby('InvoiceNo')['Description'].apply(list)

# 将数据转换为适合关联规则挖掘的格式
transaction_encoded = transactions.str.join('|').str.get_dummies()

# 使用Apriori算法挖掘频繁项集
frequent_itemsets = apriori(transaction_encoded, min_support=0.01, use_colnames=True)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 输出频繁项集和关联规则
print("Frequent Itemsets:")
print(frequent_itemsets.head())
print("\nAssociation Rules:")
print(rules.head())

# 可视化频繁项集
plt.figure(figsize=(10,6))
frequent_itemsets.support.plot(kind='bar', title='Frequent Itemsets')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.show()

# 可视化关联规则
plt.figure(figsize=(10,6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()
