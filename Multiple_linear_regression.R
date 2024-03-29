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

# Fitting multiple linear regression to the training set
# . means all independent variables
regressor <- lm(formula = Profit ~ .,
                data = training_set)


summary(regressor)

# Predicting the Test set result
y_pred <- predict(regressor, newdata = test_set)

# Backward elimination to predict better the model
# Eliminate variable if the p values is higher than the Significant value 0.05-5%
names(data_startup)
regressor <- lm(formula = Profit ~ R.D.Spend,
                data = data_startup)
summary(regressor)

# eliminated <- + State
# eliminated <- + Administration
# eliminated <-  + Marketing.Spend

# Function to perform the backward Elimination
backwardElimination <- function(x, sl){
  numVars <-  length(x)
  for (i in c(1:numVars)){
    regressor <-  lm(formula = Profit ~ ., data = x)
    maxVar  <- max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j <-  which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x <- x[, -j]
    }
    numVars = numVars -1
  }
  return(summary((regressor)))
}


# sl <- 0.05
# dataset <- dataset[, c(1,2,3,4,5)]

backwardElimination(training_set, sl)



