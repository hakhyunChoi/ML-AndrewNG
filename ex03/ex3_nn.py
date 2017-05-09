## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io

##User define funciton
import displayData as dd
import predict as pr

## Setup the parameters you will use for this exercise
input_layer_size  = 400     # 20x20 Input Images of Digits
hidden_layer_size = 25      # 25 hidden units
num_labels = 10             # 10 labels, from 1 to 10   
                            # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat     = scipy.io.loadmat('ex3data1.mat')
X, y    = mat['X'], mat['y']
m       = X.shape[0]

# Randomly select 100 data points to display
# Randomly select 100 data points to display
rand_indices    = np.random.permutation(m)
sel             = X[rand_indices[0:100], :]

dd.displayData(sel)
input('Program paused. Press enter to continue.')

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.
 
print('\nLoading Saved Neural Network Parameters ...')
 
# Load the weights into variables Theta1 and Theta2
# environment
# Theta1 has size 25 X 401
# Theta2 has size 10 X 26
mat             = scipy.io.loadmat('ex3weights.mat')
Theta1, Theta2  = mat['Theta1'], mat['Theta2']


## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
 
pred    = pr.predict(Theta1, Theta2, X)
pred    = pred.reshape(pred.shape[0],1) 
print('\nTraining Set Accuracy: {:f}'.format(np.mean(pred == y) * 100))
input('Program paused. Press enter to continue.')

 
#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.
 
#  Randomly permute examples
rp = np.random.permutation(m)

for i in range(m):
    # Display 
    print('\nDisplaying Example Image\n')
    dd.displayData(X[rp[i],:])
 
    pred = pr.predict(Theta1, Theta2, X[rp[i]].reshape(1,X.shape[1]))
    print('\nNeural Network Prediction: {0} (digit {0})'.format(pred, np.mod(pred, 10)))
     
    input('Program paused. Press enter to continue.')