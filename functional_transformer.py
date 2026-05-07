import pandas as pd
import numpy as np

import scipy.stats as stats # for q-q plot

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/train.csv', usecols=['Age', 'Fare', 'Survived'])

df.isnull().sum # to find count of null values in each columns

df['Age'].fillna(df['Age'].mean(), inplace=True)

X = df.iloc[:,1:3]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create a figure/window for plots with width=14 and height=4
plt.figure(figsize=(14,4))

# Create first plot position
# 1 row, 2 columns, first graph
plt.subplot(121)

# Plot distribution of Age column
# Shows how Age values are spread
sns.distplot(X_train['Age'])

# Add title to first graph
plt.title('Age PDF')


# Create second plot position
# 1 row, 2 columns, second graph
plt.subplot(122)

# Create QQ plot to check if Age follows normal distribution
stats.probplot(X_train['Age'], dist="norm", plot=plt)

# Add title to second graph
plt.title('Age QQ Plot')

plt.show()

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.distplot(X_train['Fare'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(X_train['Fare'], dist="norm", plot=plt)
plt.title('Age QQ Plot')

plt.show()

#log transformer
trf = FunctionTransformer(func=np.log1p)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed,y_train)
clf2.fit(X_train_transformed,y_train)
    
y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)
    
print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))
