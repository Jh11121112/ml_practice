import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv')
#fOR CATEGORICAL COLUMNS
# 1. Countplot
sns.countplot(df['gender'])
plt.show()

#2. Pie chart
df['gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()