import numpy as np

# optimizers for the variational inference

def robust_gradient_descent(fun, initial, projector=None, iter=1, h=1e-2):
    """
    fun is a function that ouptus a gradient and a value
    """
    # initialization
    val, grad = fun(initial)
    arg = initial
    for i in range(iter):
        print('Elbo in iteration {0} is {1}.'.format(i, val))
        # update argument
        arg_test = arg-h*grad
        if projector is not None:
            projector(arg_test)
        # evaluate objective function for new argument
        val_test, grad_test = fun(arg_test)
        if val_test < val:
            val = val_test
            grad = grad_test
            arg = arg_test
            h = 1.2*h
        else:
            h = 0.5*h
    return(arg, val, grad)

def multi_arg_gradient_descent(fun, initial, projector=None, iter=1, h=1e-2):
    """
    This is a version in which a joint gradient descent for several variables is performed
    initial is the list of arguments
    """
    # initialization
    val, grad = fun(initial)
    arg = initial
    for i in range(iter):
        print('Elbo in iteration {0} is {1}.'.format(i, val))
        # update argument
        arg_test = [arg[i]-h*grad[i] for i in range(len(arg))]
        if projector is not None:
            projector(arg_test)
        # evaluate objective function for new argument
        val_test, grad_test = fun(arg_test)
        if val_test < val:
            val = val_test
            grad = grad_test
            arg = arg_test
            h = 1.2*h
        else:
            h = 0.5*h
    return(arg, val, grad)

def robust_autograd_descent(fun, initial, projector=None, iter=1, h=1e-2):
    """
    fun is a function that ouptus a gradient and a value
    initial is a tuple of arguments
    """
    # initialization
    val = fun(initial)
    arg = initial
    for i in range(iter):
        print('Elbo in iteration {0} is {1}.'.format(i, val))
        # update argument
        arg_test = arg-h*grad
        if projector is not None:
            projector(arg_test)
        # evaluate objective function for new argument
        val_test, grad_test = fun(arg_test)
        if val_test < val:
            val = val_test
            grad = grad_test
            arg = arg_test
            h = 1.2*h
        else:
            h = 0.5*h
    return(arg, val, grad)


# helper functions

def bound_projector(arg):
    """
    Cut of values above or below boundary
    """
    # set bounds
    bounds = np.array([1e-6, 1e6])
    # cut off larger values
    arg[arg > bounds[1]] = bounds[1]
    # cut off lower values
    arg[arg < bounds[0]] = bounds[0]