# test the function that constucts polynomials from points and extrema
import numpy as np
import matplotlib.pyplot as plt
import pymbvi.util as ut

# example for double well
points = (np.array([-5.0, 5.0, 0.0]), np.array([0.0, 0.0, 100]))
extrema = np.array([-5.0, 0.0, 5.0])

# evaluate
param, A = ut.construct_polynomial(points, extrema)

# set up function
def fun(x):
    degree = len(param)-1
    ind = np.array([degree-i for i in range(degree+1)])
    y = (np.expand_dims(x, axis=1)**np.expand_dims(ind, axis=0))*np.expand_dims(param, axis=0)
    return(y.sum(axis=1))
    

# test with plot
x = np.linspace(-10.0, 10.0, 100)
y = fun(x)


# plot
plt.plot(x, y)
plt.plot(points[0], points[1], 'xk')
plt.show()



#np.array([[ 1., -1.,  1., -1.,  1.], [ 1.,  1.,  1.,  1.,  1.], [-4.,  3., -2.,  1.,  0.],[ 0.,  0.,  0.,  1.,  0.], [ 4.,  3.,  2.,  1.,  0.]])