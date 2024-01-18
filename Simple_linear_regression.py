# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:09:01 2024

@author: no67wuwu

work: Regression models: Simple linear regression

"""
# Predict the new salary for a new person base on the salary data, the set has just two variables year of experience and the salary

# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset 
data_salary = pd.read_csv(r"C:\Machine-Learning-A-Z-course\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")

# Separate the independet and dependet variable
x = data_salary.iloc[:, :-1].values # independet variables
y = data_salary.iloc[:, -1].values # dependet variable 

# Splitting the dataset into the teining data ans test data 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.2, random_state = 1)

# Training the Simple linear regression model on the trining set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
 
# With the fit method we train our model
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test) # we added just the year of Experience because we want to predic the salary

# Visualize the training set result
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the tes set result
plt.scatter(X_test, Y_test , color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)



























