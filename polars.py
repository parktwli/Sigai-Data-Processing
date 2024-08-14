import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())  # show the first few rows
print(df.info())  # show data types and summary statistics
print(df.describe())  # show summary statistics

import matplotlib.pyplot as plt

plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset')
plt.show()

# Calculate the mean and standard deviation of each feature
means = df.mean()
stds = df.std()

print(means)
print(stds)

# Calculate the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Calculate the covariance matrix
cov_matrix = df.cov()
print(cov_matrix)


# Filter rows where sepal length is greater than 5
filtered_df = df[df['sepal length (cm)'] > 5]

print(filtered_df.head())

# Group by target variable and calculate mean of each feature
grouped_df = df.groupby('target').mean()
print(grouped_df)