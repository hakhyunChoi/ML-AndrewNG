import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y, return_grad=False):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
    
    # Initialize some useful values
    m       = len(y) # number of training examples

    # You need to return the following variables correctly 
    J       = 0
    grad    = np.zeros((theta.shape))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    z       = np.dot(X, theta)
    sig_z   = sigmoid(z)
    one     = np.dot( -y.T, np.log(sig_z) )
    two     = np.dot( (1-y).T, np.log(1-sig_z) ) 
    J       = ( one - two ) / m

#    Error if I  use below code.
#     for j in range(grad.shape[0]):
#         grad[j]    =  np.dot(X[:,j:j+1].T, sig_z-y) / m

    grad = (1./m) * np.dot((sig_z - y).T, X)

    if return_grad == True:
        return J, grad
    elif return_grad == False:
        return J # for use in fmin/fmin_bfgs optimization function
# =============================================================