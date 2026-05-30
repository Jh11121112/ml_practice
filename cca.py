import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/janhvihaldwania/ml_practice/data_folder/train.csv')

df.isnull().mean()*100

# Create an empty list called cols and fill it with column names
# that satisfy the condition inside the if statement
cols = [

    # Take one column name at a time from df.columns
    var for var in df.columns

    # Check if the percentage of missing values is less than 5%
    # AND greater than 0%
    if df[var].isnull().mean() < 0.05
    and df[var].isnull().mean() > 0
]

# Display the final list of selected column names
cols

# Count rows remaining after removing rows with missing values in cols,
# then divide by total original rows
len(df[cols].dropna()) / len(df)

new_df = df[cols].dropna()
df.shape, new_df.shape