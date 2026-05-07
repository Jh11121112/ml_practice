# =========================================================
# COMPLETE END-TO-END MACHINE LEARNING PIPELINE
# WITH VERY SIMPLE BEGINNER-FRIENDLY COMMENTS
# =========================================================


# =========================================================
# STEP 1: IMPORT LIBRARIES
# =========================================================

# pandas -> used to work with tables/dataframes
import pandas as pd

# train_test_split -> used to divide data into train and test
from sklearn.model_selection import train_test_split

# ColumnTransformer -> apply different preprocessing to different columns
from sklearn.compose import ColumnTransformer

# Pipeline -> combine preprocessing + model in one workflow
from sklearn.pipeline import Pipeline

# SimpleImputer -> fill missing values
from sklearn.impute import SimpleImputer

# OneHotEncoder -> convert text categories into numbers
# StandardScaler -> scale numeric data to similar range
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LogisticRegression -> machine learning model
from sklearn.linear_model import LogisticRegression

# accuracy_score -> checks model performance
from sklearn.metrics import accuracy_score


# =========================================================
# STEP 2: CREATE SAMPLE DATA
# =========================================================

# Creating a dataframe manually
# Think of dataframe like Excel sheet

df = pd.DataFrame({

    # Numeric column
    'age': [25, 30, 35, 40, 28, 32, None, 45],

    # Numeric column
    'salary': [50000, 60000, 80000, 90000, 52000, None, 75000, 100000],

    # Categorical/text column
    'gender': ['Male', 'Female', 'Female', 'Male',
               'Female', 'Male', 'Female', 'Male'],

    # Categorical/text column
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore',
             'Mumbai', 'Delhi', 'Bangalore', 'Delhi'],

    # Target column
    # This is what model will predict
    'is_active': [1, 0, 1, 1, 0, 1, 0, 1]
})

# Print full dataframe
print("ORIGINAL DATAFRAME")
print(df)


# =========================================================
# STEP 3: SEPARATE INPUTS (X) AND OUTPUT (y)
# =========================================================

# X -> input features
# We remove target column because model should predict it

X = df.drop('is_active', axis=1)

# y -> target/output column
y = df['is_active']

print("\nINPUT FEATURES")
print(X)

print("\nTARGET COLUMN")
print(y)


# =========================================================
# STEP 4: SPLIT DATA INTO TRAIN AND TEST
# =========================================================

# train data -> model learns from this
# test data -> model is tested on this

# test_size=0.2 means:
# 20% data for testing
# 80% data for training

# random_state=42 means:
# keep same split every time code runs

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN DATA SHAPE")
print(X_train.shape)

print("\nTEST DATA SHAPE")
print(X_test.shape)


# =========================================================
# STEP 5: DEFINE WHICH COLUMNS ARE NUMERIC/TEXT
# =========================================================

# Numeric columns
numeric_features = ['age', 'salary']

# Categorical/text columns
categorical_features = ['gender', 'city']


# =========================================================
# STEP 6: CREATE PIPELINE FOR NUMERIC COLUMNS
# =========================================================

# Numeric pipeline will:
# 1. Fill missing numeric values
# 2. Scale numeric values

numeric_pipeline = Pipeline([

    # Fill missing values with column mean
    ('imputer', SimpleImputer(strategy='mean')),

    # Scale numbers to similar range
    ('scaler', StandardScaler())
])


# =========================================================
# STEP 7: CREATE PIPELINE FOR CATEGORICAL COLUMNS
# =========================================================

# Categorical pipeline will:
# 1. Fill missing text values
# 2. Convert text into numbers

categorical_pipeline = Pipeline([

    # Fill missing values with most frequent category
    ('imputer', SimpleImputer(strategy='most_frequent')),

    # Convert categories into numeric columns
    ('encoder', OneHotEncoder(
        drop='first',            # avoid duplicate info
        sparse_output=False      # return normal array
    ))
])


# =========================================================
# STEP 8: COMBINE BOTH PIPELINES
# =========================================================

# Apply numeric pipeline to numeric columns
# Apply categorical pipeline to categorical columns

preprocessor = ColumnTransformer([

    # num -> name of transformation
    # numeric_pipeline -> what to apply
    # numeric_features -> which columns

    ('num', numeric_pipeline, numeric_features),

    # cat -> name of transformation
    ('cat', categorical_pipeline, categorical_features)
])


# =========================================================
# STEP 9: CREATE FINAL MACHINE LEARNING PIPELINE
# =========================================================

# Final pipeline will:
# 1. Preprocess data
# 2. Train model

pipe = Pipeline([

    # preprocessing step
    ('preprocessing', preprocessor),

    # machine learning model
    ('model', LogisticRegression())
])


# =========================================================
# STEP 10: TRAIN THE MODEL
# =========================================================

# fit() means:
# learn patterns from training data

pipe.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")


# =========================================================
# STEP 11: MAKE PREDICTIONS
# =========================================================

# predict() means:
# use trained model on unseen test data

y_pred = pipe.predict(X_test)

print("\nPREDICTED VALUES")
print(y_pred)

print("\nACTUAL VALUES")
print(y_test.values)


# =========================================================
# STEP 12: CHECK MODEL ACCURACY
# =========================================================

# Compare actual vs predicted values

acc = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(acc)


# =========================================================
# STEP 13: PREDICT ON NEW CUSTOMER DATA
# =========================================================

# Imagine a new customer came to company

new_customer = pd.DataFrame({

    'age': [29],

    'salary': [65000],

    'gender': ['Female'],

    'city': ['Delhi']
})

# Model predicts whether customer is active or not
prediction = pipe.predict(new_customer)

print("\nNEW CUSTOMER PREDICTION")
print(prediction)

# If output is:
# 1 -> active customer
# 0 -> inactive customer