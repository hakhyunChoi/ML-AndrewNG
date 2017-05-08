# 해당 프로그램은 Logistic Regression을 통한 learning machine Learning program이다
# 총 20X20의 그림의 숫자를 학습하고 이를 바탕으로 추정하는데
# 이에 따라서 Theta가 총 (10X41) 개의 값을 갖게 된다. --> 숫자가 0-9 10개 pixel(20X20)+1 때문이다.
#Theta 값을 구한뒤 Sigmoid(Pixel X Theta) 하게 되면 10개의 값을 취하게 되고 이중 가장 큰값이 추정값이다.     


import numpy as np
import scipy.io


## User defined functions
import displayData as dd
import oneVsAll as ova
import predictOneVsAll as poa

## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

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

## Setup the parameters you will use for this part of the exercise
input_layer_size    = 400   # 20x20 Input Images of Digits
num_labels          = 10    # 10 labels, from 1 to 10   
# (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

mat             = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X, y            = mat['X'], mat['y']
m               = X.shape[0]

# Randomly select 100 data points to display
rand_indices    = np.random.permutation(m)
sel             = X[rand_indices[0:100], :]

dd.displayData(sel)

input('Program paused. Press enter to continue.')

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#
 
print('\nTraining One-vs-All Logistic Regression...')
 
lambda_reg = 0.1
all_theta = ova.oneVsAll(X, y, num_labels, lambda_reg)
input('Program paused. Press enter to continue.')

# ## ================ Part 3: Predict for One-Vs-All ================
# #  After ...
pred = poa.predictOneVsAll(all_theta, X)

print('Training Set Accuracy: {:f}'.format(np.mean(pred == y%10)*100))
print('Training Set Accuracy for 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Training Set Accuracy for 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Training Set Accuracy for 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Training Set Accuracy for 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Training Set Accuracy for 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Training Set Accuracy for 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Training Set Accuracy for 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Training Set Accuracy for 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Training Set Accuracy for 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Training Set Accuracy for 0:  {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500]%10)     * 100))

