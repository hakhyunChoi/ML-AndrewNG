import plotData as pd
import numpy as np

def plotDecisionBoundary(theta, X, y):
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    plt, p1, p2 = pd.plotData(X[:,1:3], y)
    
    if X.shape[1] <= 3: 
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])
     
        # Calculate the decision boundary line
        plot_y = (-1.0/theta[2]) * (theta[1]*plot_x + theta[0])
     
        # Plot, and adjust axes for better viewing
        p3 = plt.plot(plot_x, plot_y)
         
        # Legend, specific for the exercise
        plt.legend((p1, p2, p3[0]), ('Admitted', 'Not Admitted', 'Decision Boundary'), numpoints=1, handlelength=0.5)
        plt.axis([30, 100, 30, 100])
#     else:
#         # Here is the grid range
#         u = linspace(-1, 1.5, 50);
#         v = linspace(-1, 1.5, 50);
#      
#         z = zeros(length(u), length(v));
#         # Evaluate z = theta*x over the grid
#         for i = 1:length(u)
#             for j = 1:length(v)
#                 z(i,j) = mapFeature(u(i), v(j))*theta;
#             end
#         end
#         z = z'; # important to transpose z before calling contour
#      
#         # Plot z = 0
#         # Notice you need to specify the range [0, 0]
#         contour(u, v, z, [0, 0], 'LineWidth', 2)
    return plt, p1, p2