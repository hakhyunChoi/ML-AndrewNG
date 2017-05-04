import numpy as np

def computeCost(X, y, theta):
# COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
#
# Initialize some useful values
    m = len(y)  # number of training examples

# You need to return the following variables correctly 
    J = 0
    
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
 
    # According to the book
    # J = (np.dot(theta.transpose(), X.transpose()) - y)**2 
    # But I think this is better
    J = (np.dot(X, theta).transpose() - y) ** 2
    return (1.0 / (2 * m)) * J.sum()
# =========================================================================
