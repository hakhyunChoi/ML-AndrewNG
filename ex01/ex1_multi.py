# # Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).

import numpy as np
import matplotlib.pyplot as plt 

import featureNormalize as fn
import gradientDescentMulti as gdm
import normalEqn as ne
## ================ Part 1: Feature Normalization ================

# # Clear and Close Figures
print('Loading data ...\n')

# # Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
# print(' x = [#.0f #.0f], y = #.0f \n', [X(1:10,:) y(1:10,:)]')
# Phyton으로 하는 방법을 몰라서 loop를 돌림
for i in range(10):
    print('x = [{:f},{:f}] y = {:f}\n'.format(X[i, 0], X[i, 1], y[i]))
 
input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = fn.featureNormalize(X)

# Add intercept term to X
X = np.column_stack((np.ones((m, 1)), X))



## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta, J_history = gdm.gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(J_history.size), J_history, '-b', LineWidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show(block=False)

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print("{:f}, {:f}, {:f}".format(theta[0,0], theta[1,0], theta[2,0]))
print('\n')
 
# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE =============+++++++++++++++++++++++++++=========
# Recall that the first column of X is all-ones. Thus, it does not need to be normalized.
price       = 0  # You should change this
area_norm   = ( (1650 - mu[0]) / sigma[0] )  
br_norm     = ( (   3 - mu[1]) / sigma[1] )
added_norm  = [1, area_norm, br_norm]
price = np.dot(added_norm,theta)
# ============================================================
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}'.format(price[0]))
input('Program paused. Press enter to continue.\n')

 
## ================ Part 3: Normal Equations ================
  
print('Solving with normal equations...\n')
# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form solution for linear regression using the normal
#               equations. You should complete the code in normalEqn.m
#               After doing so, you should complete this code to predict the price of a 1650 sq-ft, 3 br house.
  
# # Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,0:2]
y = data[:,  2]
m = len(y)
  
# Add intercept term to X
X = np.column_stack((np.ones((m, 1)), X))
  
# Calculate the parameters from the normal equation
theta = ne.normalEqn(X, y)
  
# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' {:f} {:f} {:f}\n'.format(theta[0], theta[1], theta[2]))
print('\n')
  
# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price       = 0  # You should change this
area        = 1650
br          = 3
added       = np.array([1, area, br])
theta       = np.array(theta)
price       = np.dot(added, theta)
# ============================================================
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n {:f}\n'.format(price))
