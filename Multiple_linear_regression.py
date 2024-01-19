# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:10:17 2024

@author: no67wuwu
wrok: Multiple linear regresion

No need to apply the Feature scaling 

"""
# import the libraries
import pandas as pd
import numpy as np

# Import the data set
data_startup = pd.read_csv(r"C:\Machine-Learning-A-Z-course\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")

x = data_startup.iloc[:, :-1].values
y = data_startup.iloc[:, -1].values # Select the dependet variable 


# Encoding the categorical variable in this case is the State
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough') # if there is not passth does not go throu rows
X = np.array(ct.fit_transform(x)) # create the array with the transormed. The column country has been trasformed to 01 for example France is coding 1 0 0 each nunber is a column
print(X)


## Splitting the dataser into tha Training set and Test set
from sklearn.model_selection import train_test_split
# we define the training data ans test data with the function train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)

print(x_train)
print(x_test)
print(y_train)
print(y_test )


# Trining the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# With the fit method we train our model
regressor.fit(x_train, y_train)

# Predict the Test set result
y_pred = regressor.predict(x_test) # we added just the year of Experience because we want to predic the salary
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


























