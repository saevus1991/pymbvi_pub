# This files contains a class of variational models that can be fully expressed in terms of tensor math operations

import numpy as np
import torch
from pymbvi.models.model import Model
import pymbvi.util as ut

# global options
torch.set_default_dtype(torch.float64)


class LinearModel(Model):
    """
    Base class for models implemented using torch autograd
    The derived models require a custom implmentation of the set_... functions
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # store inputs
        self.initial = torch.tensor(initial)
        self.rates = torch.tensor(rates)
        self.tspan = torch.tensor(tspan)
        # derived quantities
        self.num_moments = len(initial)
        self.num_reactions = len(rates)
        # stuff for ode functions
        self.forward_mat = self.set_forward_mat()
        self.forward_vec = self.set_forward_vec()
        self.natural_moment_mat = self.set_natural_moment_mat()
        self.natural_moment_vec = self.set_natural_moment_vec()

    # interface functions required by abstract Model class

    def num_states(self):
        return(len(self.initial.numpy()))

    def num_controls(self):
        return(len(self.rates.numpy()))

    def forward(self, time, state, control):
        """
        Identical to forward but expecting torch tensor input
        """
        # evaluate the derivative
        dydt = (self.forward_mat@control)@state + self.forward_vec@control
        return(dydt)

    def backward(self, time, state, control, forward_time, forward, rates):
        # perform interpolation
        forward = ut.interp_pwc(time, forward_time, forward)
        # compute control part
        alpha_eff = self.kl_equation(control, rates)
        # compute Jacobian
        pj = self.natural_moment_mat
        fj = self.forward_mat@control
        # evaluae result
        dydt = pj.T@alpha_eff-fj.T@state
        return(dydt)

    def control_gradient(self, forward, backward):
        """
        Evaluates the control gradient  on a number of support points
        """
        # compute jacobian
        fj_stack = np.tensordot(self.forward_mat, forward, ([1], [1]))+np.expand_dims(self.forward_vec, -1)
        # rearrange for further processing
        grad = fj_stack.swapaxes(0, 2) @ np.expand_dims(backward, -1)
        return(grad.squeeze())

    def control_gradient_old(self, forward, backward):
        """
        Evaluates the control gradient  on a number of support points
        """
        # number of time_steps
        num_steps = forward.shape[0]
        # initialize
        grad = np.zeros((num_steps, 6))
        # contributions lagrange multpliers
        for i in range(num_steps):
            fj = np.tensordot(self.forward_mat, forward[i, :], ([1], [0]))+self.forward_vec
            grad[i, :] = fj.T@backward[i, :]
        return(grad)

    def natural_moments(self, forward):
        """
        Same as natural)moments but in torch version
        """
        prop = (self.natural_moment_mat@(forward.T)).T + self.natural_moment_vec
        return(prop)

    def kl_equation(self, control, rates):
        #print('Kl equation stuff: {0}'.format(tmp))
        control_eff = rates[0, :]-control+control*np.log(control/rates[1, :])
        return(control_eff)

    # functions that have to be provided by derived classes

    def set_forward_mat(self):
        """
        matrix that transforms controls to state
        """
        raise NotImplementedError

    def set_forward_vec(self):
        """
        ofset depending on control
        """
        raise NotImplementedError

    def set_natural_moment_mat(self):
        raise NotImplementedError

    def set_natural_moment_vec(self):
        raise NotImplementedError

    def kl_prior(self, state, get_gradient=False):
        if get_gradient:
            return(0.0, np.zeros(state.size))
        else:
            return(0.0)

    def get_initial(self):
        return(self.initial.numpy())


class SimpleGeneExpression(LinearModel):
    """
    Specific implementation of a 3 species gene expression model
    """

    def set_forward_mat(self):
        """
        matrix that transforms controls to state
        """
        # initialize
        forward_mat = np.zeros((9, 9, 6))
        # first row
        forward_mat[0, 0, [0, 1]] = np.array([-1.0, -1.0])
        # second row
        forward_mat[1, 0, 2] = 1.0
        forward_mat[1, 1, 3] = -1.0
        # third row
        forward_mat[2, 1, 4] = 1.0
        forward_mat[2, 2, 5] = -1.0
        # fourth row
        forward_mat[3, 0, [0, 1]] = np.array([-1.0, 1.0])
        forward_mat[3, 3, [0, 1]] = torch.tensor([-2.0, -2.0])
        # fifth row
        forward_mat[4, 3, 2] = 1.0
        forward_mat[4, 4, [0, 1, 3]] = np.array([-1.0, -1.0, -1.0])
        # sixth row 
        forward_mat[5, 4, 4] = 1.0
        forward_mat[5, 5, [0, 1, 5]] = np.array([-1.0, -1.0, -1.0])
        # eventh row
        forward_mat[6, 0, 2] = 1.0
        forward_mat[6, 1, 3] = 1.0
        forward_mat[6, 4, 2] = 2.0
        forward_mat[6, 6, 3] = -2.0
        # eighth row
        forward_mat[7, 5, 2] = 1.0
        forward_mat[7, 6, 4] = 1.0
        forward_mat[7, 7, [3, 5]] = np.array([-1.0, -1.0])
        # nineth row
        forward_mat[8, 1, 4] = 1.0
        forward_mat[8, 2, 5] = 1.0
        forward_mat[8, 7, 4] = 2.0
        forward_mat[8, 8, 5] = -2.0
        return(forward_mat)

    def set_forward_vec(self):
        """
        ofset depending on control
        """
        # initialize
        forward_vec = np.zeros((9, 6))
        # set nonzero elements
        forward_vec[0, 0] = 1
        forward_vec[3, 0] = 1
        return(forward_vec)

    def set_natural_moment_mat(self):
        prop = np.zeros((6, 9))
        # gene on
        prop[0, 0] = -1.0
        # gene off
        prop[1, 0] = 1.0
        # translation
        prop[2, 0] = 1.0
        # mrna degradation
        prop[3, 1] = 1.0
        # transcription
        prop[4, 1] = 1.0
        # protein degradation
        prop[5, 2] = 1.0
        return(prop)

    def set_natural_moment_vec(self):
        natural_moment_vec = np.zeros(6)
        natural_moment_vec[0] = 1.0
        return(natural_moment_vec)
