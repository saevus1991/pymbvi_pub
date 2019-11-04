# This files contains a class of variational models that use the torch autograd framework to compute required derivatives

import numpy as np
import torch
from pymbvi.models.model import Model
import pymbvi.util as ut

# global options
torch.set_default_dtype(torch.float64)


class AutogradModel(Model):
    """
    Base class for models implemented using torch autograd
    The derived models require a custom implmentation of the set_... functions
    """

    # constructor and setup

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # store inputs
        self.initial = AutogradModel.init_tensor(initial)
        self.rates = AutogradModel.init_tensor(rates)
        self.tspan = AutogradModel.init_tensor(tspan)
        # derived quantities
        self.num_moments = len(initial)
        self.num_reactions = len(rates)

    @staticmethod
    def init_tensor(array):
        """
        initialize depending on input
        """
        if type(array) is np.ndarray:
            new_array = torch.tensor(array)
        elif type(array) is list:
            new_array = torch.tensor(array)
        elif type(array) is torch.Tensor:
            new_array = array.clone().detach()
        else:
            raise ValueError('Invalid input type for input {}'.format(array))
        return(new_array)

    # interface functions required by abstract Model class

    def num_states(self):
        return(len(self.initial.numpy()))

    def num_controls(self):
        return(len(self.rates.numpy()))

    def num_param(self):
        return(len(self.rates.numpy()))       

    def forward(self, time, state, control, rates):
        """ 
        Compute r.h.s of the forward differential equations 
        """
        # evaluate the derivative
        dydt = self.forward_torch(time, torch.from_numpy(state), torch.from_numpy(control), torch.from_numpy(rates))
        return(dydt)

    def backward(self, time, state, control, forward_time, forward, rates):
        # perform interpolation
        exp_control = np.exp(control)
        forward = ut.interp_pwc(time, forward_time, forward)
        # compute control part
        control_eff = 1.0-exp_control+exp_control*control
        # convert to torch stuff
        forward_torch = torch.from_numpy(forward)
        forward_torch.requires_grad = True
        control_torch = torch.from_numpy(control)
        rates_torch = torch.from_numpy(rates)
        # compoute contribution of the natural moments
        tmp = self.natural_moments_torch(forward_torch, rates_torch)
        tmp.backward(torch.from_numpy(control_eff))
        dydt = np.array(forward_torch.grad)
        forward_torch.grad.zero_()
        # add contribution from the forward equation
        tmp = self.forward_torch(time, forward_torch, control_torch, rates_torch)
        tmp.backward(torch.from_numpy(state))
        dydt -= np.array(forward_torch.grad)
        return(dydt)

    def constraint_gradient(self, control, forward, backward, rates):
        """
        Evaluates the gradient of the constraint with respect to the controls
        """
        # preparations
        dim = forward.shape
        grad = np.zeros(dim[:3]+control.shape[-1:])
        rates_torch = torch.from_numpy(rates)
        # contributions lagrange multpliers
        for i in range(dim[0]):
            for j in range(dim[1]):
                control_torch = torch.tensor(control[i, j], requires_grad=True)
                for k in range(dim[2]):
                    forward_torch = torch.from_numpy(forward[i, j, k])
                    backward_torch = torch.from_numpy(backward[i, j, k])
                    tmp = self.forward_torch(i, forward_torch, control_torch, rates_torch)
                    tmp.backward(backward_torch)
                    grad[i, j, k] = np.array(control_torch.grad)
                    control_torch.grad.zero_()
        return(grad)

    def constraint_rates_gradient(self, control, forward, backward, rates):
        """
        Evaluates the gradient of the constraint with respect to the controls
        """
        # preparations
        dim = forward.shape
        grad = np.zeros(dim[:3]+control.shape[-1:])
        rates_torch = torch.tensor(rates, requires_grad=True)
        # contributions lagrange multpliers
        for i in range(dim[0]):
            for j in range(dim[1]):
                control_torch = torch.tensor(control[i, j])
                for k in range(dim[2]):
                    forward_torch = torch.from_numpy(forward[i, j, k])
                    backward_torch = torch.from_numpy(backward[i, j, k])
                    tmp = self.forward_torch(i, forward_torch, control_torch, rates_torch)
                    tmp.backward(backward_torch)
                    grad[i, j, k] = np.array(rates_torch.grad)
                    rates_torch.grad.zero_()
        return(grad)

    def constraint_full_gradient(self, control, forward, backward, rates):
        """
        Evaluates the gradient of the constraint with respect to the controls
        """
        # preparations
        dim = forward.shape
        grad_control = np.zeros(dim[:3]+control.shape[-1:])
        grad_rates = np.zeros(dim[:3]+control.shape[-1:])
        rates_torch = torch.tensor(rates, requires_grad=True)
        # contributions lagrange multpliers
        for i in range(dim[0]):
            for j in range(dim[1]):
                control_torch = torch.tensor(control[i, j], requires_grad=True)
                for k in range(dim[2]):
                    forward_torch = torch.from_numpy(forward[i, j, k])
                    backward_torch = torch.from_numpy(backward[i, j, k])
                    tmp = self.forward_torch(i, forward_torch, control_torch, rates_torch)
                    tmp.backward(backward_torch)
                    grad_control[i, j, k] = np.array(control_torch.grad)
                    grad_rates[i, j, k] = np.array(rates_torch.grad)
                    rates_torch.grad.zero_()
                    control_torch.grad.zero_()
        return(grad_control, grad_rates)

    def natural_moments(self, forward, rates):
        """
        Map forward moment function to the natural moments
        Forward is expected to be of shape (num_steps, control_dim)
        """
        prop = self.natural_moments_torch(torch.from_numpy(forward).T, torch.from_numpy(rates))
        return(prop.numpy())

    def get_initial(self):
        return(self.initial.numpy())

    # helper functions to be implmented by derived classes

    def forward_torch(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        raise NotImplementedError

    def natural_moments_torch(self, forward, rates):
        """
        Same as natural)moments but in torch version
        """
        raise NotImplementedError


class LinearAutogradModel(AutogradModel):
    """
    Linear model sublcass of the autograd model
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # base class constructor
        super().__init__(initial, rates, tspan)
        # stuff for ode functions
        self.forward_mat = self.set_forward_mat()
        self.forward_vec = self.set_forward_vec()
        self.natural_moment_mat = self.set_natural_moment_mat()
        self.natural_moment_vec = self.set_natural_moment_vec()

    def forward_torch(self, time, state, control):
        """
        Identical to forward but expecting torch tensor input
        """
        # evaluate the derivative
        dydt = torch.matmul(torch.matmul(self.forward_mat, control), state) + torch.matmul(self.forward_vec, control)
        return(dydt)

    def forward_batch(self, time, state, control):
        """
        Takes a hole batch of state and control and propagates it forward in time
        """
        # transform the control to forward matrix
        forward_mat = torch.matmul(self.forward_mat, control.T)
        # apply to state
        dydt = torch.matmul(forward_mat.permute(2, 0, 1), state.unsqueeze(-1)).squeeze()
        # add state independent contribution
        dydt += torch.matmul(control, self.forward_vec.T)
        return(dydt)

    def natural_moments_torch(self, forward):
        """
        Same as natural)moments but in torch version
        """
        prop = torch.matmul(self.natural_moment_mat, forward).T+self.natural_moment_vec
        return(prop)

    # helper functions to set the required matrices, to be implemented by derived models

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


# specific model implementations

class SimpleGeneExpression(LinearAutogradModel):
    """
    Specific implementation of a 3 species gene expression model
    """

    def set_forward_mat(self):
        """
        matrix that transforms controls to state
        """
        # initialize
        forward_mat = torch.zeros((9, 9, 6))
        # first row
        forward_mat[0, 0, [0, 1]] = torch.tensor([-1.0, -1.0])
        # second row
        forward_mat[1, 0, 2] = 1.0
        forward_mat[1, 1, 3] = -1.0
        # third row
        forward_mat[2, 1, 4] = 1.0
        forward_mat[2, 2, 5] = -1.0
        # fourth row
        forward_mat[3, 0, [0, 1]] = torch.tensor([-1.0, 1.0])
        forward_mat[3, 3, [0, 1]] = torch.tensor([-2.0, -2.0])
        # fifth row
        forward_mat[4, 3, 2] = 1.0
        forward_mat[4, 4, [0, 1, 3]] = torch.tensor([-1.0, -1.0, -1.0])
        # sixth row 
        forward_mat[5, 4, 4] = 1.0
        forward_mat[5, 5, [0, 1, 5]] = torch.tensor([-1.0, -1.0, -1.0])
        # eventh row
        forward_mat[6, 0, 2] = 1.0
        forward_mat[6, 1, 3] = 1.0
        forward_mat[6, 4, 2] = 2.0
        forward_mat[6, 6, 3] = -2.0
        # eighth row
        forward_mat[7, 5, 2] = 1.0
        forward_mat[7, 6, 4] = 1.0
        forward_mat[7, 7, [3, 5]] = torch.tensor([-1.0, -1.0])
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
        forward_vec = torch.zeros((9, 6))
        # set nonzero elements
        forward_vec[0, 0] = 1
        forward_vec[3, 0] = 1
        return(forward_vec)

    def set_natural_moment_mat(self):
        prop = torch.zeros(6, 9)
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
        natural_moment_vec = torch.zeros(6)
        natural_moment_vec[0] = 1.0
        return(natural_moment_vec)

    def kl_prior(self, initial):
        return(0.0)


class BernoulliTasep(AutogradModel):
    """
    Implementaiton of tasep model with Bernoulli closure and rate parametrization
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # base class constructor
        super().__init__(initial, rates, tspan)
        # stuff for ode functions
        self.moment_selection = self.set_moment_selection()

    def forward_torch(self, time, state, control):
        """
        Identical to forward but expecting torch tensor input
        """
        dydt = torch.zeros(self.num_moments)
        dydt[0] = control[0]*(1.0-state[0]) - control[1]*state[0]*(1-state[1])
        dydt[1:-1] = control[1]*(state[0:-2]*(1-state[1:-1]) - state[1:-1]*(1-state[2:]))
        dydt[-1] = control[1]*state[-2]*(1-state[-1]) - control[2]*state[-1]
        return(dydt)

    def natural_moments_torch(self, forward):
        """
        Same as natural)moments but in torch version
        """
        # preparations
        if forward.dim() == 2:
            pad_one = torch.ones(1, forward.shape[1])
            pad_zero = torch.zeros(1, forward.shape[1])
        else:
            pad_one = torch.ones(1)
            pad_zero = torch.zeros(1)
        tmp1 = torch.cat([pad_one, forward])
        tmp2 = 1.0-torch.cat([forward, pad_zero])
        # compute raw phi
        phi = tmp1*tmp2
        # sumarize reactions with same parameter
        phi_reduced = torch.matmul(phi.T, self.moment_selection)
        return(phi_reduced)

    def set_moment_selection(self):
        """
        Construct a matrix for producing phi reduces
        """
        moment_selection = torch.zeros(self.num_moments+1, 3)
        moment_selection[0, 0] = 1.0
        moment_selection[-1, -1] = 1.0
        moment_selection[1:-1, 1] = 1.0
        return(moment_selection)


class BernoulliTasep2(AutogradModel):
    """
    Implementaiton of tasep model with Bernoulli closure and rate parametrization
    allows one control for each reaction and is thus more flexible
    """

    def forward_torch(self, time, state, control):
        """
        Identical to forward but expecting torch tensor input
        """
        dydt = torch.zeros(self.num_moments)
        dydt[0] = control[0]*(1.0-state[0]) - control[1]*state[0]*(1-state[1])
        dydt[1:-1] = control[1:-2]*state[0:-2]*(1-state[1:-1]) - control[2:-1]*state[1:-1]*(1-state[2:])
        dydt[-1] = control[-2]*state[-2]*(1-state[-1]) - control[-1]*state[-1]
        return(dydt)

    def natural_moments_torch(self, forward):
        """
        Same as natural)moments but in torch version
        """
        # preparations
        if forward.dim() == 2:
            pad_one = torch.ones(1, forward.shape[1])
            pad_zero = torch.zeros(1, forward.shape[1])
        else:
            pad_one = torch.ones(1)
            pad_zero = torch.zeros(1)
        tmp1 = torch.cat([pad_one, forward])
        tmp2 = 1.0-torch.cat([forward, pad_zero])
        # compute raw phi
        phi = tmp1*tmp2
        return(phi)

