import numpy as np

# WARMUPEXERCISE Example function in python3.6
#   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix
def warmUpExercise():
    A = []
    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix 
    #               In octave, we return values by defining which variables
    #               represent the return values (at the top of the file)
    #               and then set them accordingly. 
    A = np.identity(5)
    return A
# ===========================================
