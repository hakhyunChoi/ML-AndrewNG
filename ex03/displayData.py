import numpy as np
import math
import matplotlib.pyplot as plt

def displayData(X, example_width=None):
    #DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.
    
    # Set example_width automatically if not passed in
    if not example_width:
        example_width = int( round(np.sqrt(X.shape[1])) )
    
    plt.close()
    plt.figure()
    # Gray Image
    plt.set_cmap('gray')
    
    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)
    
    # Compute number of items to display
    display_rows = int( math.floor(math.sqrt(m)) )
    display_cols = int( math.ceil(m / display_rows) )
    
    # Between images padding
    pad = 1
    
    # Setup blank display
    display_array = - np.ones( (pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)) )

    # Copy each example into a patch on the display array
    curr_ex = 1
    for j in range(1, display_rows+1):
        for i in range(1, display_cols+1):
            if curr_ex > m:
                break 
            # Copy the patch
            
            # Get the max value of the patch
            max_val = max(abs(X[curr_ex-1, :]))
            rows    = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols    = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))
            
#             display_array[rows, cols] = np.reshape( X[curr_ex, :], example_height, example_width ) / max_val
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            curr_ex += 1
        
        if curr_ex > m:
            break
    # Display Image
    h = plt.imshow(display_array, vmin=-1, vmax=1 )
    
    # Do not show axis
    plt.axis('off')
    plt.show(block=False)
    
    return h, display_array