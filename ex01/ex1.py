#  Machine Learning Online Class - Exercise 1: Linear Regression
# 
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
# 
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
# 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
# 
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
# 
# 
#  Initialization
import numpy as np
import matplotlib.pyplot as plt 

import warmUpExercise as wue
import plotData as pd
import computeCost as cc
import gradientDescent as gd

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m 
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(wue.warmUpExercise())

input('Program paused. Press enter to continue.\n');

 
# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=",")
X = data[:, 0]
y = data[:, 1]
m = len(y)  # number of training examples
 
# Plot Data
# Note: You have to complete the code in plotData.m
pd.plotData(X, y);
input('Program paused. Press enter to continue.\n')
 
# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

X = np.column_stack((np.ones((m, 1)), X))  # Add a column of ones to x
theta = np.zeros((2, 1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01
 
# compute and display initial cost
print(cc.computeCost(X, y, theta))

# run gradsient descent
theta = gd.gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent: ')
print('{:f} {:f} \n'.format(theta[0, 0], theta[1, 0]))
 
# Plot the linear fit
plt.plot(X[:, 1], np.dot(X, theta), '-', label='Linear regression')
plt.legend(loc='lower right')
plt.draw()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:f}\n'.format(float(predict1 * 10000)))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:f}\n'.format(float(predict2 * 10000)))
 
input('Program paused. Press enter to continue.\n')
