## Multiple Linear Regression ##

# Remove all objects from the workspace (memory) of the R session
rm(list=ls())
gc()

# Import libraries
# install.packages('caTools')
# install.packages('ggplot2')
library(caTools)
library(ggplot2)

# Set working directory
setwd('C:/Machine-Learning-A-Z-course/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/R')
getwd()

# Importing the data set
data_startup <- read.csv("50_Startups.csv")


# Encoding Categorical data
data_startup$State <- factor(data_startup$State,
                          levels = c('New York', 'California', 'Florida'),
                          labels = c(1,2,3))


# Splitting the data set into the training set and test set
set.seed(123)
split <- sample.split(data_startup$Profit,
                      SplitRatio = 0.8) # percentage of the training set so 80% 
# Create the training and test set
training_set <- subset(data_startup, split == TRUE)
test_set <- subset(data_startup, split == FALSE)

names(training_set)
# Fitting multiple linear regression to the training set
# . means all independent variables
regressor <- lm(formula = Profit ~ .,
                data = training_set)


summary(regressor)

# Predicting the Test set result
y_pred <- predict(regressor, newdata = test_set)


