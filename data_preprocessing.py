# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:39:20 2024

@author: no67wuwu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the data set
dataset = pd.read_csv(r"C:\Machine-Learning-A-Z-course\Machine Learning A-Z (Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Python\Data.csv")

x = dataset.iloc[:, :-1].values # independet variables
y = dataset.iloc[:, -1].value # dependet variable 
y = dataset['Purchased'].values # intead of the idex we can also use the name of the column to extract the dependet variable

# Print
print(x)
print(y)

# we create those entities because the maching leraning model to obtain those entities, independet variables and dependet variables
# Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # create the imputer
imputer.fit(x[:, 1:3]) # selest the columns that we want to 
x[:, 1:3] = imputer.transform(x[:, 1:3])


## Encoding
# Encoding the idependent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough') # if there is not passth does not go throu rows
X = np.array(ct.fit_transform(x)) # create the array with the transormed. The column country has been trasformed to 01 for example France is coding 1 0 0 each nunber is a column
print(x)


# Encoding the dependet variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(y)
print(Y)

## Splitting the dataser into tha Training set and Test set

# We have Y and X (mayuscula) that is transformed for the model and be able
# to split the data as follow

from sklearn.model_selection import train_test_split

# we define the training data ans test data with the function train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 1)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# test 20%
# training 80$

## Feature Scaling
# Standardisation: it is used almost in all the cases
# Normalisation: it is used for data that has a nomal distribution

# Import library
from sklearn.preprocessing import StandardScaler

# Creat an object of the class
sc = StandardScaler()
# Apply the the scaling just to the numerical features, so we don't scaling for the columns that represent the country
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # is taking the column from the index 3 onwards as well as all the rows
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

print(X_train)
print(X_test)
































