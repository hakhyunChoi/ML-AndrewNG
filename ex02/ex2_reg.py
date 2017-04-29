## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np

## User defined functions
import plotData as pd
import mapFeature as mf
import costFunctionReg as cfr
## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data    = np.loadtxt('ex2data2.txt', delimiter=',')
X       = data[:,0:2]; y = data[:, 2:3];

plt, p1, p2     = pd.plotData(X, y)
# Put some labels 
 
# # Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
# 
# # Specified in plot order
plt.legend((p1, p2), ('y = 1', 'y = 0'), numpoints=1, handlelength=0)

 
## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
# Add Polynomial Features
 
# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mf.mapFeature(X[:,0], X[:,1])
 
# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))
 
# Set regularization parameter lambda to 1
lambda_reg = 1
 
# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = cfr.costFunctionReg(initial_theta, X, y, lambda_reg)
 
print('Cost at initial theta (zeros): {:f}\n'.format(cost))
input('Program paused. Press enter to continue.\n')

 
# ## ============= Part 2: Regularization and Accuracies =============
# #  Optional Exercise:
# #  In this part, you will get to try different values of lambda and 
# #  see how regularization affects the decision coundart
# #
# #  Try the following values of lambda (0, 1, 10, 100).
# #
# #  How does the decision boundary change when you vary lambda? How does
# #  the training set accuracy vary?
# #
# 
# # Initialize fitting parameters
# initial_theta = zeros(size(X, 2), 1)
# 
# # Set regularization parameter lambda to 1 (you should vary this)
# lambda = 1
# 
# # Set Options
# options = optimset('GradObj', 'on', 'MaxIter', 400)
# 
# # Optimize
# [theta, J, exit_flag] = ...
#     fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options)
# 
# # Plot Boundary
# plotDecisionBoundary(theta, X, y)
# hold on
# title(sprintf('lambda = #g', lambda))
# 
# # Labels and Legend
# xlabel('Microchip Test 1')
# ylabel('Microchip Test 2')
# 
# legend('y = 1', 'y = 0', 'Decision boundary')
# hold off
# 
# # Compute accuracy on our training set
# p = predict(theta, X)
# 
# print('Train Accuracy: #f\n', mean(double(p == y)) * 100)