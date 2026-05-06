#Standardization

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/sample_customer_data.csv')
df = df.drop(['customer_id','gender', 'city', 'account_type'] , axis = 1)
#Train - Test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.drop('is_active', axis = 1), df['is_active'], test_size = 0.2, random_state = 0)

print(X_train.shape)
 #Standard Scaler
 #fit() → learns mean & std from data - usually do on train dataset, so it will learn, we learn always from train but transform we do on train and test both
#transform() → applies scaling
#fit_transform() → both together

from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
x_train_scaled = scl.fit_transform(X_train)
x_test_scaled = scl.transform(X_test)
print(x_train_scaled)

#so when we scale data it gets converted into numpy array and we have to again convert it into dataframe. You lose column names also thats wh use x_train.columns

x_train_scaled = pd.DataFrame(x_train_scaled, columns = X_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = X_test.columns)


#Normalization - usually using minmax scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)