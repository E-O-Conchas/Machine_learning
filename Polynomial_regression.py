# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:05:04 2024

@author: no67wuwu

work: Polinomial regression with Python

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the data set
data = pd.read_csv(r"C:\Machine-Learning-A-Z-course\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")

# Filter the independet and dependet variables from the data set 
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Trining the linear regression model on the whole data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)


# Trining the polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# Visualising the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Ttuth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Ttuth or Bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicta new result with Linear regression
lin_reg.predict([[6.5]]) # it is wrong

# Predicting a new result with polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))





