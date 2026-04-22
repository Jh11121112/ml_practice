import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport



df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv')
#fOR CATEGORICAL COLUMNS
# 1. Countplot
sns.countplot(df['gender'])
plt.show()

#2. Pie chart
df['gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

prof = ProfileReport(df, title='Customer Data Profile Report', explorative=True)
prof.to_file(output_file='customer_data_profile_report.html')
