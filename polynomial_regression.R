## Poplynomial Regression ##

# Remove all objects from the workspace (memory) of the R session
rm(list=ls())
gc()

# Import libraries
# install.packages('caTools')
# install.packages('ggplot2')
library(caTools)
library(ggplot2)

# Set working directory
setwd('C:/Machine-Learning-A-Z-course/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/R')
getwd()

# Import data
data_salary <- read.csv('Position_Salaries.csv')
data_salary <- data_salary[2:3]


# Splitting the data set into the training set and test set
# set.seed(123)
# split <- sample.split(data_salary$Salary,
#                       SplitRatio = 0.8) # percetage of the training set so 80% 
# 
# # Create the training and test set
# training_set <- subset(dataset, split == TRUE)
# test_set <- subset(dataset, split == FALSE)


# Fittinng Linear Regresion to the dataset
lin_reg <- lm(formula = Salary ~ .,
              data = data_salary)
summary(lin_reg)

# Fitting Polynomial regression to the dataset
# We Create the potetial to the level to create a polynomial regresion model
data_salary$Level2 <- data_salary$Level^2
data_salary$Level3 <- data_salary$Level^3
data_salary$Level4 <- data_salary$Level^4

pol_reg <- lm(formula = Salary ~ .,
              data = data_salary)

summary(pol_reg)

# Visualising the Linear Regression results
library(ggplot2)

ggplot()+
  geom_point(aes(x = data_salary$Level , y = data_salary$Salary),
             colour = 'red')+
  geom_line(aes(x = data_salary$Level, y = predict(object = lin_reg, newdata = data_salary)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Linear regression)') +
  xlab('Level') +
  ylab('Salary')
  
# Visualising the Polynomial Regression results
library(ggplot2)

ggplot()+
  geom_point(aes(x = data_salary$Level , y = data_salary$Salary),
             colour = 'red')+
  geom_line(aes(x = data_salary$Level, y = predict(object = pol_reg, newdata = data_salary)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Polynomial regression)') +
  xlab('Level') +
  ylab('Salary')


# Predicting a new result with the Linear regression model

# Single prediction value
y_pred <- predict(lin_reg, data.frame(Level = 6.5))


# Predict a new result with the Polynomial regression model
y_pred <- predict(pol_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))

# Precited variable 
# y_pred
# 1 
# 158862.5 



