# This files contains special model implementations for the variational inference

from pymbvi.models.model import Model
import numpy as np
import pymbvi.util as ut


class SimpleGeneExpression(Model):
    """
    Specific implementation of a 3 species gene expression model
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # store inputs
        self.initial = initial
        self.rates = rates
        self.tspan = tspan
        # derived quantities
        self.num_moments = len(initial)
        self.num_reactions = len(rates)

    # main functions requires by abstract Model class

    def num_states(self):
        return(len(self.initial))

    def num_controls(self):
        return(len(self.rates))

    def forward(self, time, state, control):
        """ 
        Compute r.h.s of the forward differential equations 
        """
        # evaluate the derivative
        dydt = np.zeros(9)
        dydt[0] = control[0]*(1-state[0])-control[1]*state[0]
        dydt[1] = control[2]*state[0]-control[3]*state[1]
        dydt[2] = control[4]*state[1]-control[5]*state[2]
        dydt[3] = -control[0]*state[0]+control[0]-2*control[0]*state[3]-2*control[1]*state[3]+control[1]*state[0]
        dydt[4] = -control[0]*state[4]-control[1]*state[4]+control[2]*state[3]-control[3]*state[4]
        dydt[5] = -control[0]*state[5]-control[1]*state[5]+control[4]*state[4]-control[5]*state[5]
        dydt[6] = 2*control[2]*state[4]+control[2]*state[0]-2*control[3]*state[6]+control[3]*state[1]
        dydt[7] = control[2]*state[5]-control[3]*state[7]+control[4]*state[6]-control[5]*state[7]
        dydt[8] = 2*control[4]*state[7]+control[4]*state[1]-2*control[5]*state[8]+control[5]*state[2]
        return(dydt)

    def backward(self, time, state, control, forward_time, forward, rates):
        # perform interpolation
        forward = ut.interp_pwc(time, forward_time, forward)
        # compute control part
        alpha_eff = self.kl_equation(control, rates)
        # compute Jacobian
        pj = self.prop_jacobian(forward)
        fj = self.forward_jacobian(control)
        # evaluae result
        dydt = pj.T@alpha_eff-fj.T@state
        return(dydt)

    def control_gradient(self, forward, backward):
        """
        Evaluates the control gradient  on a number of support points
        """
        # number of time_steps
        num_steps = forward.shape[0]
        # initialize
        grad = np.zeros((num_steps, 6))
        # contributions lagrange multpliers
        for i in range(num_steps):
            fj = self.forward_jacobian_control(forward[i, :])
            grad[i, :] = backward[i, :].reshape(1, -1)@fj
        return(grad)

    def natural_moments(self, forward):
        """
        Map forward moment function to the natural moments
        Forward is expected to be of shape (num_steps, control_dim)
        """
        # get number of time steps
        num_steps = forward.shape[0]
        prop = np.zeros((num_steps, 6))
        # gene on
        prop[:, 0] = 1-forward[:, 0]
        # gene off
        prop[:, 1] = forward[:, 0]
        # translation
        prop[:, 2] = forward[:, 0]
        # mrna degradation
        prop[:, 3] = forward[:, 1]
        # transcription
        prop[:, 4] = forward[:, 1]
        # protein degradation
        prop[:, 5] = forward[:, 2]
        return(prop)

    # helper functions 

    def kl_equation(self, control, rates):
        #print('Kl equation stuff: {0}'.format(tmp))
        control_eff = rates[0, :]-control+control*np.log(control/rates[1, :])
        return(control_eff)

    def prop_jacobian(self, state):
        # initialize
        jacobian = np.zeros((6, 9))
        # first row
        jacobian[0, 0] = -1
        # second row
        jacobian[1, 0] = 1
        # third row
        jacobian[2, 0] = 1
        # fourth row
        jacobian[3, 1] = 1
        # fifth row
        jacobian[4, 1] = 1
        # sixth row
        jacobian[5, 2] = 1
        return(jacobian)

    def forward_jacobian(obj, control):
        # initialize
        jacobian = np.zeros((9, 9))
        # first row
        jacobian[0, 0] = -control[0]-control[1]
        # second row
        jacobian[1, 0] = control[2]
        jacobian[1, 1] = -control[3]
        # third row
        jacobian[2, 1] = control[4]
        jacobian[2, 2] = -control[5]
        # fourth row
        jacobian[3, 0] = -control[0]+control[1]
        jacobian[3, 3] = -2*control[0]-2*control[1]
        # fifth row
        jacobian[4, 3] = control[2]
        jacobian[4, 4] = -control[0]-control[1]-control[3]
        # sixth row 
        jacobian[5, 4] = control[4]
        jacobian[5, 5] = -control[0]-control[1]-control[5]
        # eventh row
        jacobian[6, 0] = control[2]
        jacobian[6, 1] = control[3]
        jacobian[6, 4] = 2*control[2]
        jacobian[6, 6] = -2*control[3]
        # eighth row
        jacobian[7, 5] = control[2]
        jacobian[7, 6] = control[4]
        jacobian[7, 7] = -control[3]-control[5]
        # nineth row
        jacobian[8, 1] = control[4]
        jacobian[8, 2] = control[5]
        jacobian[8, 7] = 2*control[4]
        jacobian[8, 8] = -2*control[5]
        return(jacobian)

    def forward_jacobian_control(self, state):
        # initialize
        jacobian = np.zeros((9, 6))
        # first row
        jacobian[0, 0] = (1-state[0])
        jacobian[0, 1] = -state[0]
        # second row
        jacobian[1, 2] = state[0]
        jacobian[1, 3] = -state[1]
        # third row
        jacobian[2, 4] = state[1]
        jacobian[2, 5] = -state[2]
        # fourth row
        jacobian[3, 0] = -state[0]+1-2*state[3]
        jacobian[3, 1] = -2*state[3]+state[0]
        # fifth row
        jacobian[4, 0] = -state[4]
        jacobian[4, 1] = -state[4]
        jacobian[4, 2] = state[3]
        jacobian[4, 3] = -state[4]
        # sixth row 
        jacobian[5, 0] = -state[5]
        jacobian[5, 1] = -state[5]
        jacobian[5, 4] = state[4]
        jacobian[5, 5] = -state[5]
        # seventh row
        jacobian[6, 2] = 2*state[4]+state[0]
        jacobian[6, 3] = -2*state[6]+state[1]
        # eighth row
        jacobian[7, 2] = state[5]
        jacobian[7, 3] = -state[7]
        jacobian[7, 4] = state[6]
        jacobian[7, 5] = -state[7]
        # nineth row
        jacobian[8, 4] = 2*state[7]+state[1]
        jacobian[8, 5] = -2*state[8]+state[2]
        return(jacobian)

    def get_initial(self):
        return(self.initial.copy())

    def kl_prior(self, initial):
        """
        Compute the contribution of the initial kl
        """
        return(0.0)

  
class BasicTasep(Model):
    """
    Realization of the tasep model
    """

    def __init__(self, initial, rates, tspan):
        """
        Constructor method
        """
        # store inputs
        self.initial = initial
        self.rates = rates
        self.tspan = tspan
        # derived quantities
        self.num_moments = len(initial)
        self.num_reactions = len(rates)