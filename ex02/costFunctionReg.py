import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, lambda_reg, return_grad=False):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros((theta.shape[0]))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    z       = np.dot(X, theta)
    sig_z   = sigmoid(z)
    one     = np.dot( -y.T, np.log(sig_z) )
    two     = np.dot( (1-y).T, np.log(1-sig_z) ) 
    J       = ( one - two ) / m 
    
    grad        = ( np.dot((sig_z - y).T, X) + lambda_reg * grad ) / m
    grad_zero   = np.dot((sig_z - y).T, X) / m
    
    grad[0]     = grad_zero[0]
    
    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J
# =============================================================