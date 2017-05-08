import numpy as np

##User define funciton
import sigmoid as sig

def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)
    
    # Useful values
    m           = X.shape[0]
    num_labels  = Theta2.shape[0]
    
    # You need to return the following variables correctly 
    p           = np.zeros((X.shape[0], 1))
    X           = np.column_stack((np.ones((m,1)), X))
    a2          = np.ones((m, Theta1.shape[0]))
    a3          = np.ones((m, Theta2.shape[0]))
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    for i in range(m):
        for j in range (Theta1.shape[0]):
            a2[i,j] = sig.sigmoid( np.dot(X[i], Theta1[j].T) )

    a2      = np.column_stack((np.ones((m,1)), a2))
    
    for i in range(m):
        for j in range (Theta2.shape[0]):
            a3[i,j] = sig.sigmoid( np.dot(a2[i], Theta2[j].T) )
            
    p       = np.argmax(a3, axis=1) 

    return p + 1
# =========================================================================