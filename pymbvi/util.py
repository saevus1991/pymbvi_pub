# helper functions required for the vi procedure

import numpy as np


def interp_pwc(time, time_grid, grid_val):
    """
    Function that evaluates a piece-wise constant function given by the grid forward_time, forward at time
    Note that if len(grid_val) = n, we have len(time_grid) = n+1 and f(t) = grid_val[i] for t in [time_grid[i], time_grid[i+1] )
    If the time-step is fixed, the computation is simplified
    """
    if (time >= time_grid[-1]):
        return grid_val[-1]
    elif (time <= time_grid[0]):
        return grid_val[0]
    else:
        tmp = time-time_grid[0]
        ind = int(tmp/(time_grid[1]-time_grid[0]))
        return(grid_val[ind])


def num_derivative(fun, x, h=1e-6):
    """
    Compute jacobian of fun at value x with symmetric finite differences
    """
    # set up output
    m = len(x)
    y = fun(x)
    if isinstance(y, np.ndarray):
        n = len(fun(x))
    elif isinstance(y, np.generic):
        n = 1
    else:
        raise ValueError('Unsupported output type {} of fun'.format(type(y)))
    jacobian = np.zeros((n, m))
    # evaluate
    for i in range(m):
        x_up = x.copy()
        x_up[i] += h
        x_low = x.copy()
        x_low[i] -= h
        jacobian[:, i] = fun(x_up)-fun(x_low)
    return(jacobian/(2*h))


def num_hessian(fun, x, h=1e-6):
    """
    Compute hessian of fun at value x with symmetric finite differences
    """
    # set up output
    m = len(x)
    y = fun(x)
    if isinstance(y, np.ndarray):
        n = len(fun(x))
    elif isinstance(y, np.generic):
        n = 1
    else:
        raise ValueError('Unsupported output type {} of fun'.format(type(y)))
    hessian = np.zeros((n, m, m))
    # evaluate
    for i in range(m):
        # diagonal elements
        x1 = x.copy()
        x1[i] += 2*h
        x2 = x.copy()
        x2[i] += h  
        x3 = x.copy()
        x3[i] += -h
        x4 = x.copy()
        x4[i] -= 2*h
        res = (-fun(x1)+16*fun(x2)-30*fun(x)+16*fun(x3)-fun(x4))/(12*h**2)
        hessian[:, i, i] = res
        # off diagonal elements
        for j in range(i+1, m):
            x1 = x.copy()
            x1[i] += h
            x1[j] += h
            x2 = x.copy()
            x2[i] += h
            x2[j] -= h
            x3 = x.copy()
            x3[i] -= h
            x3[j] += h
            x4 = x.copy()
            x4[i] -= h
            x4[j] -= h
            res = (fun(x1)-fun(x2)-fun(x3)+fun(x4))/(4*h**2)
            hessian[:, i, j] = res
            hessian[:, j, i] = res
    return(hessian)

def quad_trap(time, val):
    """
    trapezoidal integrator for a function known on a grid
    """
    delta_t = time[1:]-time[0:-1]
    res = 0.5*(val[1:]+val[0:-1])
    return(np.tensordot(delta_t, res, axes=1))


def construct_polynomial(points, extrema=None):
    """
    Construct polynomial from a list of points and extrema
    points is a tuple (x, y) containing the coordinates of the points
    extrema is a vector of the extrema location
    """
    # extract stuff
    x = points[0]
    y = points[1]
    # copmute degree
    if extrema is not None:
        degree = len(points[0]) + len(extrema) - 1
    else:
        degree = len(points[0]) -1
    ind = np.array([degree-i for i in range(degree+1)])
    ind2 = ind-1
    ind2[-1] = 0
    # set up equation systems
    if extrema is not None:
        A1 = np.expand_dims(x, axis=1)**np.expand_dims(ind, axis=0)
        A2 = ind * np.expand_dims(extrema, axis=1)**np.expand_dims(ind2, axis=0)
        A = np.concatenate([A1, A2], axis=0)
        b = np.concatenate([y, np.zeros(len(extrema))])
    else:
        A = np.expand_dims(x, axis=1)**np.expand_dims(ind, axis=0)
        b = y
    # solve system
    param = np.linalg.solve(A, b)
    return(param, A)


def integrate(time, arg):
    """
    Integrate over all but last dimension
    """
    dim = arg.shape
    arg = arg.reshape((-1, np.prod(dim[3:])))
    delta = time.flatten()[1:]-time.flatten()[:-1]
    tmp = 0.5*(arg[1:, :]+arg[:-1, :])
    out = (delta*tmp.T).T
    out = out.sum(axis=0)
    return(out)


def integrate_subsamples(time, arg, dim=None):
    """
    Integrate out subsampling intervals
    """
    if dim is None:
        dim = arg.shape
        arg = arg.reshape((-1, np.prod(dim[3:])))
    out = np.zeros(arg.shape)
    delta = time.flatten()[1:]-time.flatten()[:-1]
    tmp = 0.5*(arg[1:, :]+arg[:-1, :])
    out[:-1, :] = (delta*tmp.T).T
    out = out.reshape(dim).sum(axis=2)
    return(out)


def lstsq_reg(a, b, k=1-5):
    """
    Compute regularized least squaes solution
    min_x ||a*x - b||**2 + k * ||x||**2
    """
    # construct extended system
    a_ex = np.concatenate([a, k*np.eye(a.shape[0])])
    b_ex = np.concatenate([b, np.zeros(a.shape[0])])
    # solve system
    x = np.linalg.lstsq(a_ex, b_ex, rcond=None)
    return(x[0])
