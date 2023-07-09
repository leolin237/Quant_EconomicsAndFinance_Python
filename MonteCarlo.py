#Let us estimate the value of pi using Monte Carlo method.
#We will use the fact that the area of a circle is pi*r^2 and the area of a square is 4*r^2.
#We will generate random points in the square and see how many of them fall in the circle.
#The ratio of the number of points in the circle to the total number of points will be equal to the ratio of the area of the circle to the area of the square.
#We will use this fact to estimate the value of pi.
#We will repeat this experiment a large number of times and take the average of the estimates to get a better estimate of pi.
#We will also plot the estimates to see how the estimates converge to the true value of pi as the number of experiments increases.

import numpy as np
import matplotlib.pyplot as plt

#Number of experiments
N = 1000

#Number of points in each experiment
n = 10000

#Radius of the circle
r = 1

#Area of the circle
area_circle = np.pi*r**2

#Area of the square
area_square = 4*r**2

#Estimates of pi
pi_estimates = []

#Number of points in the circle
n_circle = 0

#Number of points in the square
n_square = 0

#all the points x and y chosen randomly from the interval [-r,r] to plot the circle at thr iteration k
x_in = []
x_out = []
y_in = []
y_out = []

#The iteration at which we want to plot the circle
k = N-1

#Generate random points in the square and see how many of them fall in the circle
for i in range(N):
    for j in range(n):
        x = np.random.uniform(-r,r)
        y = np.random.uniform(-r,r)
        if x**2 + y**2 <= r**2:
            n_circle += 1
            if(i==k):
                x_in.append(x)
                y_in.append(y)
        else:
            n_square += 1
            if(i==k):
                x_out.append(x)
                y_out.append(y)
    pi_estimates.append(4*r**2*n_circle/(n_circle+n_square))
    n_circle = 0
    n_square = 0

#Plot the estimates of pi
plt.plot(pi_estimates)
plt.xlabel('Number of experiments')
plt.ylabel('Estimate of pi')
plt.savefig('MonteCarlo.png')
plt.show()

#Plot all the point in the circle and the square at the iteration k
plt.plot(x_in,y_in,'ro')
plt.plot(x_out,y_out,'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('MonteCarlo_circle.png')
plt.show()

#Print the final estimate of pi
print('Final estimate of pi = ',pi_estimates[40])

#Print the average of the estimates of pi
print('Average of the estimates of pi = ',np.mean(pi_estimates))

#Print the standard deviation of the estimates of pi
print('Standard deviation of the estimates of pi = ',np.std(pi_estimates))

#Print the error in the estimate of pi
print('Error in the estimate of pi = ',np.abs(np.mean(pi_estimates)-np.pi))

#Print the relative error in the estimate of pi
print('Relative error in the estimate of pi = ',np.abs(np.mean(pi_estimates)-np.pi)/np.pi)

#Print the percentage error in the estimate of pi
print('Percentage error in the estimate of pi = ',np.abs(np.mean(pi_estimates)-np.pi)/np.pi*100,'%')