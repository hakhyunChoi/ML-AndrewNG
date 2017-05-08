import numpy as np
import sys

## User define function
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lambda_reg, return_grad = False):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    #regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 
    
    # Initialize some useful values
    m       = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J       = 0
    y       = y.reshape(y.size,1)
    theta   = theta.reshape(theta.size,1)
    grad    = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
    # Hint: When computing the gradient of the regularized cost function, 
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta; 
    #           temp(1) = 0;   # because we don't add anything for j = 0  
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    one_y   = np.ones((y.shape))
    z       = np.dot(X, theta)
    sig_z   = sigmoid(z)
    one     = np.dot( (-one_y*y).T, np.log(sig_z) )
    two     = np.dot( (one_y-y).T, np.log(1-sig_z) )
    reg     = lambda_reg * np.dot(theta.T, theta) / (2 * m)

    J       = ( one - two ) / m + reg     
#     sys.stdout.write("Cost: %f   \r" % (J) )
#     sys.stdout.flush()
        
    grad                    = ( np.dot((sig_z - y).T, X) + lambda_reg * theta.T) / m
    grad_no_regularization  = np.dot((sig_z - y).T, X) / m

    grad[0]                 = grad_no_regularization[0]    
    
    # =============================================================
    if return_grad:
        return J, grad.flatten()
    else:
        return J