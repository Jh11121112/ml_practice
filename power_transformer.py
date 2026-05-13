
#for making data normally distributed - for regression models usually
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/concrete_data.csv')

X = df.drop(columns = ['Strength'])
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Applying Regression without any transformation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

r2_score(y_test,y_pred)

# Plotting the distplots without any transformation

# Loop through every column in X_train dataframe
# Example columns could be Age, Fare, Salary etc.
for col in X_train.columns:

    # Create a figure/window for plots
    # width = 14, height = 4
    plt.figure(figsize=(14,4))


    # Create first graph position
    # 1 row, 2 columns, first graph
    plt.subplot(121)


    # Create distribution plot for current column
    # Shows how values are distributed/spread
    sns.distplot(X_train[col])


    # Add current column name as title
    plt.title(col)


    # Create second graph position
    # 1 row, 2 columns, second graph
    plt.subplot(122)


    # Create QQ plot for current column
    # Checks whether data follows normal distribution
    stats.probplot(X_train[col], dist="norm", plot=plt)


    # Add current column name as title
    plt.title(col)


    # Display both graphs
    plt.show()
    
    # Applying Box-Cox Transform

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')

X_train_transformed = pt.fit_transform(X_train+0.000001) #added this value since box cox dont work for data which have 0 values, so in this way by adding no value of data will be 0 and box cox will work
X_test_transformed = pt.transform(X_test+0.000001)

pd.DataFrame({'cols':X_train.columns,'box_cox_lambdas':pt.lambdas_}) #for each col what will be the value of lambda

# Applying linear regression on transformed data

lr = LinearRegression()
lr.fit(X_train_transformed,y_train)

y_pred2 = lr.predict(X_test_transformed)

r2_score(y_test,y_pred2)

# Apply Yeo-Johnson transform

pt1 = PowerTransformer()

X_train_transformed2 = pt1.fit_transform(X_train)
X_test_transformed2 = pt1.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_transformed2,y_train)

y_pred3 = lr.predict(X_test_transformed2)

print(r2_score(y_test,y_pred3))

pd.DataFrame({'cols':X_train.columns,'Yeo_Johnson_lambdas':pt1.lambdas_})
