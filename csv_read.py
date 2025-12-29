import pandas as pd

df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv')
print(df.head())

#use_cols paramenter to read specific columns
df2 = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv', usecols =['customer_id', 'age', 'account_balance'])
print(df2.head())

#nrows parameter to read specific number of rows: eg nrows = 5 then 1 to 5 row will be read

