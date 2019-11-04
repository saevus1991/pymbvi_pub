# This files contains special model implementations for the variational inference

import numpy as np
import torch
import pymbvi.util as ut
from pymbvi.models.mjp.partition_control import PartitionControl
from pymbvi.models.mjp.autograd_model import AutogradModel, LinearAutogradModel


class SimpleGeneExpression(AutogradModel, PartitionControl):
    """
    Specific implementation of a 3 species gene expression model
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # base class constructor
        AutogradModel.__init__(self, initial, rates, tspan)
        # stuff for ode functions
        self.forward_mat = self.set_forward_mat()
        self.forward_vec = self.set_forward_vec()
        self.natural_moment_mat = self.set_natural_moment_mat()
        self.natural_moment_vec = self.set_natural_moment_vec()

    def forward_torch(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        # evaluate the derivative
        eff_control = torch.exp(control+rates)
        dydt = torch.matmul(torch.matmul(self.forward_mat, eff_control), state) + torch.matmul(self.forward_vec, eff_control)
        return(dydt)

    def natural_moments_torch(self, forward, rates):
        """
        Same as natural)moments but in torch version
        """
        prop = torch.matmul(self.natural_moment_mat, forward).T+self.natural_moment_vec
        return(prop*torch.exp(rates))

    # helper functions to set the required matrices, to be implemented by derived models

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


class BernoulliTasep1(AutogradModel, PartitionControl):
    """
    Implementaiton of tasep model with Bernoulli closure and rate parametrization
    allows one control for each different parameter 
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # base class constructor
        super().__init__(initial, rates, tspan)
        # stuff for ode functions
        self.rate_mat = self.set_rate_mat()

    def forward_torch(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        eff_control = self.rate_mat @ torch.exp(rates + control)
        dydt = torch.zeros(self.num_moments)
        dydt[0] = eff_control[0]*(1.0-state[0]) - eff_control[1]*state[0]*(1-state[1])
        dydt[1:-1] = eff_control[1:-2]*state[0:-2]*(1-state[1:-1]) - eff_control[2:-1]*state[1:-1]*(1-state[2:])
        dydt[-1] = eff_control[-2]*state[-2]*(1-state[-1]) - eff_control[-1]*state[-1]
        return(dydt)

    def natural_moments_torch(self, forward, rates):
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
        tmp1 = torch.cat([pad_one, forward]).T
        tmp2 = 1.0-torch.cat([forward, pad_zero]).T
        # compute raw phi
        phi = torch.exp(rates) * ((tmp1 * tmp2) @self.rate_mat)
        # if forward.dim() == 2:
        #     phi = eff_rates.unsqueeze(1) * tmp1 * tmp2
        # else:
        #     phi = eff_rates * tmp1 * tmp2
        return(phi)

    def set_rate_mat(self):
        """
        matrix that allows to obtain effective rates by matrix multiplication
        """
        rate_mat = torch.zeros(len(self.initial)+1, 3)
        rate_mat[0, 0] = 1
        rate_mat[1:-1, 1] = 1
        rate_mat[-1, -1] = 1
        return(rate_mat)

    def num_controls(self):
        return(3)


class BernoulliTasep2(AutogradModel, PartitionControl):
    """
    Implementaiton of tasep model with Bernoulli closure and rate parametrization
    allows one control for each reaction 
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # base class constructor
        super().__init__(initial, rates, tspan)
        # stuff for ode functions
        self.rate_mat = self.set_rate_mat()

    def forward_torch(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        eff_control = (self.rate_mat @ rates) * torch.exp(control)
        dydt = torch.zeros(self.num_moments)
        dydt[0] = eff_control[0]*(1.0-state[0]) - eff_control[1]*state[0]*(1-state[1])
        dydt[1:-1] = eff_control[1:-2]*state[0:-2]*(1-state[1:-1]) - eff_control[2:-1]*state[1:-1]*(1-state[2:])
        dydt[-1] = eff_control[-2]*state[-2]*(1-state[-1]) - eff_control[-1]*state[-1]
        return(dydt)

    def natural_moments_torch(self, forward, rates):
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
        tmp1 = torch.cat([pad_one, forward]).T
        tmp2 = 1.0-torch.cat([forward, pad_zero]).T
        # compute raw phi
        eff_rates = self.rate_mat @ rates
        phi = eff_rates * tmp1 * tmp2
        # if forward.dim() == 2:
        #     phi = eff_rates.unsqueeze(1) * tmp1 * tmp2
        # else:
        #     phi = eff_rates * tmp1 * tmp2
        return(phi)

    def set_rate_mat(self):
        """
        matrix that allows to obtain effective rates by matrix multiplication
        """
        rate_mat = torch.zeros(len(self.initial)+1, 3)
        rate_mat[0, 0] = 1
        rate_mat[1:-1, 1] = 1
        rate_mat[-1, -1] = 1
        return(rate_mat)

    def num_controls(self):
        return(len(self.initial)+1)


class PredatorPrey(AutogradModel, PartitionControl):
    """
    Specific implementation of a 3 species gene expression model
    """

    def forward_torch(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        # compute effecitve control
        eff_control = torch.exp(control+rates)
        hom = self.lognorm_cosure(state)
        # construct derivative
        dydt = torch.zeros(5)
        dydt[0] = eff_control[0]*state[0]-eff_control[1]*(state[3]+state[0]*state[1])
        dydt[1] = eff_control[2]*(state[3]+state[0]*state[1])-eff_control[3]*state[1]
        dydt[2] = 2*eff_control[0]*state[2]+eff_control[0]*state[0]+eff_control[1]*(state[3]+state[0]*state[1])-2*eff_control[1]*(hom[0]-(state[3]+state[0]*state[1])*state[0])
        dydt[3] = eff_control[0]*state[3]-eff_control[1]*hom[1]+eff_control[1]*(state[3]+state[0]*state[1])*state[1]+eff_control[2]*hom[0]-eff_control[2]*(state[3]+state[0]*state[1])*state[0]-eff_control[3]*state[3]
        dydt[4] = 2*eff_control[2]*(hom[1]-(state[3]+state[0]*state[1])*state[1])+eff_control[2]*(state[3]+state[0]*state[1])-2*eff_control[3]*state[4]+eff_control[3]*state[1]
        return(dydt)

    def forward_torch_new(self, time, state, control, rates):
        """
        Identical to forward but expecting torch tensor input
        """
        # compute effecitve control
        eff_control = torch.exp(control+rates)
        hom = self.lognorm_cosure(state)
        # construct derivative
        dm = torch.zeros(2)
        dm[0] = eff_control[0]*state[0]-eff_control[1]*(state[3]+state[0]*state[1])
        dm[1] = eff_control[2]*(state[3]+state[0]*state[1])-eff_control[3]*state[1]
        # second order
        dydt[2] = 2*eff_control[0]*state[2]+eff_control[0]*state[0]+eff_control[1]*(state[3]+state[0]*state[1])-2*eff_control[1]*(hom[0]-(state[3]+state[0]*state[1])*state[0])
        dydt[3] = eff_control[0]*state[3]-eff_control[1]*hom[1]+eff_control[1]*(state[3]+state[0]*state[1])*state[1]+eff_control[2]*hom[0]-eff_control[2]*(state[3]+state[0]*state[1])*state[0]-eff_control[3]*state[3]
        dydt[4] = 2*eff_control[2]*(hom[1]-(state[3]+state[0]*state[1])*state[1])+eff_control[2]*(state[3]+state[0]*state[1])-2*eff_control[3]*state[4]+eff_control[3]*state[1]
        return(dydt)

    def natural_moments_torch(self, forward, rates):
        """
        Compute phi's from the moment system
        """
        if forward.dim() == 1:
            prop = torch.zeros(4)
            prop[0] = forward[0]
            prop[1] = forward[3]+forward[0]*forward[1]
            prop[2] = forward[3]+forward[0]*forward[1]
            prop[3] = forward[1]
        else:
            prop = torch.zeros((4,)+forward.shape[1:])
            prop[0, :] = forward[0, :]
            prop[1, :] = forward[3, :]+forward[0, :]*forward[1, :]
            prop[2, :] = forward[3, :]+forward[0, :]*forward[1, :]
            prop[3, :] = forward[1, :]        
        return(prop.T*torch.exp(rates))

    def lognorm_cosure(self, state):
        """
        lognormal closure for the moments E[X1^2 * X2] and E[X1* X2^2]
        """
        hom = torch.zeros(2)
        hom[0] = ((state[2]+state[0]*state[0])/state[1])*((state[3]+state[0]*state[1])/state[0])**2
        hom[1] = ((state[4]+state[1]*state[1])/state[0])*((state[3]+state[0]*state[1])/state[1])**2
        return(hom)
