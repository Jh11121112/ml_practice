import pandas as pd

df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv')

#How big is the data?
print(df.shape) #(rows, columns)

#what is data types of each col?
print(df.info())

#Are there any missing values
print(df.isnull().sum())

#how does data look mathematically?
print(df.describe())

#are there any duplicate rows?
print(df.duplicated().sum())

#how is the correlation between numerical columns?
print(df.corr())
