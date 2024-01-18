
## Simple Linear Regression ##

# Remove all objects from the workspace (memory) of the R session
rm(list=ls())
gc()

# Import libraries
# install.packages('caTools')
# install.packages('ggplot2')
library(caTools)
library(ggplot2)

# Set working directory
setwd('C:/Machine-Learning-A-Z-course/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python')
getwd()

# Importing the data set
df_salary <-  read.csv('Salary_Data.csv')

# Splitting the data set into the training set and test set
set.seed(123)
split <- sample.split(df_salary$Salary,
                      SplitRatio = 2/3) # percentage of the training set so 80% 
# Create the training and test set
training_set <- subset(df_salary, split == TRUE)
test_set <- subset(df_salary, split == FALSE)


# Fitting simple linear regression to the training set
                          # Salary is proportional to the years of experience
regressor <- lm(formula = Salary ~ YearsExperience,
                data = training_set)
summary(regressor)

# Predicting the Test set result
y_pred <- predict(regressor, newdata = test_set)



# Visualizing the Training  set result
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experiance (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')
  

# Visualizing the Test  set result
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experiance (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')




