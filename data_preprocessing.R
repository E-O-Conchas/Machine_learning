
# Remove all objects from the workspace (memory) of the R session
rm(list=ls())
gc()

# Set working directory
setwd('C:/Machine-Learning-A-Z-course/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/R')

# In each machine learning model we use dependent variables and independent variables, 
# we use independent variables to predict dependent variables

# Importing the data set
dataset <-  read.csv('Data.csv')

# Taking care of missing data
# Take the mean and replace the missing data with the mean
dataset$Age <-  ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x, na.rm= TRUE)), 
                     dataset$Age)

dataset$Salary <-  ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN= function(x) mean(x, na.rm= TRUE)), 
                     dataset$Salary)

# Encoding Categorical data
dataset$Country <- factor(dataset$Country,
                          levels = c('France', 'Spain', 'Germany'),
                          labels = c(1,2,3))

dataset$Purchased <- factor(dataset$Purchased,
                          levels = c('No', 'Yes'),
                          labels = c(0,1))

# Splitting the data set into the training set and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split <- sample.split(dataset$Purchased,
                     SplitRatio = 0.8) # percetage of the training set so 80% 

# Create the training and test set
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)


# Feature scaling

# Standardization
# Normalization
training_set[,2:3] <- scale(training_set[,2:3])
test_set[,2:3] <- scale(test_set[,2:3])
















