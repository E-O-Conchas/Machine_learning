# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:34:29 2024

@author: no67wuwu

work: Support Vector regression model to understand the position level of the relation 

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv(r"C:\Machine-Learning-A-Z-course\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python\Position_Salaries.csv")

# Filter the independet and dependet variables from the data set 
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

print(x)
print(y)

# Reshape the y to obtain the sata in array 
y = y.reshape(len(y),1)
print(y)


# print(x) 
# [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

# print(y) Salaries
# [  45000   50000   60000   80000  110000  150000  200000  300000  500000
#  1000000]


# Feature scaling 
# We dont have the split of the data set. We will apply the feature scaling in the whole data set

# Import library
from sklearn.preprocessing import StandardScaler

# Creat an object of the class for each feature for x and y
sc_x = StandardScaler()
sc_y = StandardScaler()

# Apply the the scaling 
x_trans = sc_x.fit_transform(x)
y_trans = sc_y.fit_transform(y)

# Print the results
print(x_trans)
print(y_trans)

# Trining the SVR model on the whole dataset
from sklearn.svm import SVR

# Creat an object of the class
regressor = SVR(kernel = "rbf")
regressor.fit(x_trans, y_trans)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

# inverse_transform()

# Visualisingthe SVR results
plt.scatter(sc_x.inverse_transform(x_trans), sc_y.inverse_transform(y_trans), color = 'red')
plt.plot(sc_x.inverse_transform(x_trans), sc_y.inverse_transform(regressor.predict(x_trans).reshape(-1,1)), color = 'blue')
plt.title('Ttuth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the SVR result ( for higher resolution and smoother curve)

x_grid = np.arange(min(sc_x.inverse_transform(x_trans)), max(sc_x.inverse_transform(x_trans)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x_trans), sc_y.inverse_transform(y_trans), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')
plt.title('Ttuth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()












