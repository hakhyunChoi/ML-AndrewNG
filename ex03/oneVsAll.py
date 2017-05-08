import numpy as np
from scipy.optimize import minimize, fmin, fmin_bfgs

# User define function
import lrCostFunction as lrcf

def oneVsAll(X, y, num_labels, lambda_reg):
	# ONEVSALL trains multiple logistic regression classifiers and returns all
	# the classifiers in a matrix all_theta, where the i-th row of all_theta 
	# corresponds to the classifier for label i
	#    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
	#    logistic regression classifiers and returns each of these classifiers
	#    in a matrix all_theta, where the i-th row of all_theta corresponds 
	#    to the classifier for label i
	
	#  Some useful variables
	m, n 		= X.shape
	
	# You need to return the following variables correctly 
	all_theta 	= np.zeros((num_labels, n + 1))
	
	# Add ones to the X data matrix
	X 			= np.column_stack((np.ones((m,1)), X))
	
	# ====================== YOUR CODE HERE ======================
	# Instructions: You should complete the following code to train num_labels logistic regression classifiers
	#               with regularization parameter lambda. 

	# Hint: theta[:] will return a column vector.

	# Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
	#       whether the ground truth is true/false for this class.

	# Note: For this assignment, we recommend using fmincg to optimize the cost
	#       function. It is okay to use a for-loop (for c = 1:num_labels) to
	#       loop over the different classes.
	#
	#       fmincg works similarly to fminunc, but is more efficient when we
	#       are dealing with large number of parameters.
	#
	# Example Code for fmincg:

	for i in range(num_labels):
		initial_theta = np.zeros((n + 1, 1))

		#  Set options for fminunc
		myargs = (X, (y%10==i).astype(int).flatten(), lambda_reg, True)
		## functions WITH gradient/jac parameter
		# from https://github.com/tansaku/py-coursera/issues/9#issuecomment-8801160
		theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':13}, method="Newton-CG", jac=True)
		# left other methods in case we want to tinker later on
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="CG", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="BFGS", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="L-BFGS-B", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="TNC", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="SLSQP", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="dogleg", jac=True)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="trust-ncg", jac=True)
		
		## functions WITHOUT gradient/jac parameter
		# myargs = (X, (y%10==c).astype(int), lambda_reg)
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="Nelder-Mead")
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="Powell") #maybe
		# theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="COBYLA")

		
		# assign row of all_theta corresponding to current c/class
		all_theta[i,:] = theta["x"]  	
	
	return all_theta
# =========================================================================