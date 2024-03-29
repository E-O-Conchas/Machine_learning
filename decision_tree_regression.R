# Remove all objects from the environment
rm(list=ls())
gc()

## Decision Tree regression


# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling # We dont need apply feature scaling


# Fitting the Decision Tree Regression to the dataset
# import library
install.packages('rpart')
library(rpart)

regressor <- rpart(formula = Salary ~ ., 
                   data = dataset,
                   control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# # Visualising the Decision Tree Regression results
# # install.packages('ggplot2')
# library(ggplot2)
# ggplot() +
#   geom_point(aes(x = dataset$Level, y = dataset$Salary),
#              colour = 'red') +
#   geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
#             colour = 'blue') +
#   ggtitle('Truth or Bluff (Decision Tree Regresion)') +
#   xlab('Level') +
#   ylab('Salary')
# 
# 


# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.0001)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regresion)') +
  xlab('Level') +
  ylab('Salary')
