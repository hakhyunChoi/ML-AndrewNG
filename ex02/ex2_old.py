import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))


def plotSignoidFunction():
    z = np.arange(-10,10, step=0.01)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(z, sigmoid(z), 'r')
    plt.grid(True)
    plt.show()
   
   
def plotData(data):
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')  
    plt.show()
    

def costFunction(theta, X, y):
    
    theta       = np.matrix(theta)
    X           = np.matrix(X)
    y           = np.matrix(y)
    
    first       = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second      = np.multiply((1-y), np.log(1-sigmoid(X * theta.T)))
    J           = np.sum(first - second) / len(X)
 
    return J
 
 
def gradient(theta, X, y):
    
    theta       = np.matrix(theta)
    X           = np.matrix(X)
    y           = np.matrix(y)
    
    parameters  = int(theta.ravel().shape[1])
    grad        = np.zeros(parameters)
    error       = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term        = np.multiply(error, X[:,i])
        grad[i]     = np.sum(term) / len(X)
    
    return grad


def predict(theta, X):
    X               = np.matrix(X)
    probability     = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability] 

# Load Data
# The first two columns contains the exam scores and the third column contains the label.
data    = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1','Exam 2', 'Admitted'])

# add a ones column - this makes the matrix multiplication work our easier
data.insert(0,'Ones',1)
cols    = data.shape[1]
# X       = np.matrix( data[:, [0,1]] )
X       = data.iloc[:,0:cols-1]
y       = data.iloc[:,cols-1:cols]
theta   = np.zeros(3)
print(costFunction(theta, X, y))

result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y))  
print(costFunction(result[0], X, y))

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y.values)]
accuracy = (sum(map(int, correct)) % len(correct))  

print ('accuracy = {0}%'.format(accuracy))  