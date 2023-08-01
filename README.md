# applied-data-science

# -*- coding: utf-8 -*-
"""Applied-Data-Science.ipynb

"""

# Load the Drive module:
from google.colab import drive

drive.mount("/content/drive")
! ls "/content/drive/MyDrive"

import os
print(os.getcwd())
print(os.listdir())
print(os.listdir("drive/MyDrive"))

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")
df.head()

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

#Classification of datatypes
df.dtypes

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

# Count the number of columns
num_columns = df.shape[1]

# Print the result
print("Number of columns:", num_columns)

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

# Get the dimensions (number of rows and columns)
dimensions = df.shape

# Print the result
print("Dimensions:", dimensions)

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df["Make"]

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df['Make'].unique()

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df['Make'].value_counts()

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df.info()

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df.corr(numeric_only=True)

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

df.describe()

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

# Compute the distribution for all columns
distributions = {}
for column in df.columns:
    distribution = df[column].value_counts()
    distributions[column] = distribution

# Print the distributions
for column, distribution in distributions.items():
    print("Distribution of", column)
    print(distribution)
    print()

!pip install pandas
!pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = '/content/drive/MyDrive/cars_dataset.csv'
df = pd.read_csv(file_path)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')

# Save the heatmap as a PDF file
plt.savefig('/content/drive/MyDrive/correlation_heatmap.pdf', format='pdf')

import pandas as pd
import seaborn as sns

sns.pairplot(df)

# Save the pair plot as a PNG file
plt.savefig('/content/drive/MyDrive/pairplot.png', dpi=300)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/cars_dataset.csv")

# --------------
# Import the data
# --------------
df = pd.read_csv('/content/drive/MyDrive/cars_dataset.csv')
print(df.info())

# Get Cylinders count
dMean_cylinders = df['Cylinders'].mean()
print('// complete ........ data model cylinders mean: ', dMean_cylinders)

# Generate a small image for slides ;-)
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
sns.scatterplot(x='Fuel Consumption Comb (L/100 km)',
                y='CO2 Emissions(g/km)',
                data=df, alpha=0.5, color='grey')# palette='Grays')
plt.savefig('fig_scatter_Fuel_CO2.pdf')
plt.close()

# --------------
# Import the data
# --------------
df = pd.read_csv('/content/drive/MyDrive/cars_dataset.csv')
print(df.info())

# Get Cylinders count
dMean_cylinders = df['Cylinders'].mean()
print('// complete ........ data model cylinders mean: ', dMean_cylinders)

# Generate a small image for slides ;-)
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
sns.scatterplot(x='CO2 Emissions(g/km)',
                y='Cylinders',
                data=df, alpha=0.5, color='grey')# palette='Grays')
plt.savefig('fig_scatter_Cylinders_Fuel_CO2.pdf')
plt.close()

# Selecting the desired columns
columns_of_interest = ['Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
subset_df = df[columns_of_interest]

# Applying the describe() method on the subset DataFrame
description = subset_df.describe()

# Printing the description
print(description)

from sklearn.linear_model import LinearRegression

# Import the data
df = pd.read_csv('/content/drive/MyDrive/cars_dataset.csv')
print(df.info())

# Get Cylinders count
dMean_cylinders = df['Cylinders'].mean()
print('// complete ........ data model cylinders mean: ', dMean_cylinders)

# Generate a scatter plot
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
sns.scatterplot(x='CO2 Emissions(g/km)', y='Cylinders', data=df, alpha=0.5, color='grey')

# Perform linear regression
regression_model = LinearRegression()
X = df[['CO2 Emissions(g/km)']]  # Input feature
y = df['Cylinders']  # Target variable
regression_model.fit(X, y)

# Plot the linear regression line
plt.plot(X, regression_model.predict(X), color='red')

plt.savefig('fig_scatter_Linear_Regression_2_Cylinders_Fuel_CO2.pdf')
plt.close()

# Import the data
df = pd.read_csv('/content/drive/MyDrive/cars_dataset.csv')
print(df.info())

# Get Cylinders count
dMean_cylinders = df['Cylinders'].mean()
print('// complete ........ data model cylinders mean: ', dMean_cylinders)

# Generate a scatter plot
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
sns.scatterplot(x='Fuel Consumption Comb (L/100 km)', y='CO2 Emissions(g/km)', data=df, alpha=0.5, color='grey')

# Perform linear regression
regression_model = LinearRegression()
X = df[['Fuel Consumption Comb (L/100 km)']]  # Input feature
y = df['CO2 Emissions(g/km)']  # Target variable
regression_model.fit(X, y)

# Plot the linear regression line
plt.plot(X, regression_model.predict(X), color='red')

plt.savefig('fig_scatter_Linear_Regression_2CO2_Emissions_Fuel_CO2.pdf')
plt.close()

#Residual Plot, r-squared and RMSE for Evaluation Metrics (Capstone)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/drive/MyDrive/cars_dataset.csv')

# Load data
X = df[['Cylinders']].values
y = df['CO2 Emissions(g/km)'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation metrics
r_squared = model.score(X_test, y_test)
print(f"R-Squared: {r_squared}")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted CO2")
plt.ylabel("Residuals")

# Feature importance
print("Slope", model.coef_)
print("Intercept", model.intercept_)

# ------------------------------------------
# Pre-Processing
# ------------------------------------------
# Data pre-processing
# Let's check the dataset for missing values.
# @code: 0, or ‘index’: Drop rows which contain missing values.
# Let's see where the Null values are.
# Let's see the data shape and NaN values.
# This will give number of NaN values in every column.
df_null_values = df.isnull().sum()
print('NANs?', df_null_values)

# Show missing values in a figure
# plt.figure(figsize=(15,5))
# sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='Greys')
# plt.xticks(rotation=45, fontsize=6)
# plt.tight_layout()
# plt.savefig('fig_MissingValues.pdf')
# plt.close()

# Drop all rows with NaN.
df = df.dropna(axis=0)
df_null_values = df.isnull().sum()
print('NANs_After_Update?', df_null_values)
print('// complete ........ Pre-Processing')

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Check datatypes
datatypes = df.dtypes
print("Datatypes:\n", datatypes)

