## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
# import matplotlib.pyplot as plt

import plotData as pd
import costFunction as cf
import plotDecisionBoundary as pdb
import sigmoid as sig
import predict as pr
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:,0:2]; y = data[:, 2:3]; 
## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plt, p1, p2 = pd.plotData(X, y)

# Put some labels 

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend((p1, p2), ('Admitted', 'Not Admitted'), numpoints=1, handlelength=0)
plt.show(block=False)
input('Program paused. Press enter to continue.\n')

 
## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m
#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape 
 
# Add intercept term to x and X_test
X = np.column_stack((np.ones((m, 1)), X))
 
# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))
 
# Compute and display initial cost and gradient
cost, grad = cf.costFunction(initial_theta, X, y, return_grad=True)
 
# print('Cost at initial theta (zeros): {:f}\n'.format(cost))
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):')
# print(' {:f} \n'.format(grad))
print(grad)
 
input('Program paused. Press enter to continue.\n')

 
## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
 
#  Set options for fminunc
myargs = (X, y)
# options = fmin(cf.costFunction, x0=initial_theta, args=(X, y), maxiter=100)
option = fmin(cf.costFunction, x0=initial_theta, args=myargs)

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
theta, cost, _, _, _, _, _ = fmin_bfgs(cf.costFunction, x0=option, args=myargs, full_output=True)
  
# Print theta to screen
print('Cost at theta found by fminunc: {:f}\n'.format(cost))
print('theta:')
print(theta)
  
plt.close('all')
# Plot Boundary
plt, p1, p2 = pdb.plotDecisionBoundary(theta, X, y)
  
# Put some labels 
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
  
# Specified in plot order
plt.show(block=False)  
input('Program paused. Press enter to continue.\n')
 
## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m
 
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
 
prob = sig.sigmoid(np.dot([[1, 45, 85]], theta.T))
print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(float(prob)))

#  Compute accuracy on our training set
p = pr.predict(theta, X)
print('Train Accuracy: {:f}'.format(np.mean((p == y))*100))
input('Program paused. Press enter to continue.\n')