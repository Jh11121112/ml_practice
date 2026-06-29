# creating dataset with 2 classes and 3 features, where class 1 has mean 0 and class 2 has mean 1. The covariance matrix is identity matrix for both classes. The dataset is then shuffled to create a random dataset with 40 samples.
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

np.random.seed(23)

# Class 1
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

df = pd.DataFrame(class1_sample,
                  columns=['feature1', 'feature2', 'feature3'])
df['target'] = 1

# Class 2
mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample,
                   columns=['feature1', 'feature2', 'feature3'])
df1['target'] = 0

# Combine both DataFrames
df = pd.concat([df, df1], ignore_index=True)

# Shuffle rows
df = df.sample(frac=1, random_state=23).reset_index(drop=True)

# Step 1 - Apply standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df.iloc[:,0:3] = scaler.fit_transform(df.iloc[:,0:3])

# Step 2 - Find Covariance Matrix
covariance_matrix = np.cov([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]])
print('Covariance Matrix:\n', covariance_matrix)

# Step 3 - Finding EV and EVs
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

pc = eigen_vectors[0:2] # selecting the first 2 principal components
pc

transformed_df = np.dot(df.iloc[:,0:3],pc.T)
# 40,3 - 3,2
new_df = pd.DataFrame(transformed_df,columns=['PC1','PC2']) #new dataframe with 2 principal components
new_df['target'] = df['target'].values
new_df.head()


     

     