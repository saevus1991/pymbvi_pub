# Gradient based variational engine vor moment-based approximation of Markov processes 

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pymbvi.util as ut

from matplotlib import pyplot as plt


class VariationalEngine(object):
    """
    Class for running the variational inference procedure
    """

    # construction and setup

    def __init__(self, model, obs_model, obs_times, obs_data, num_controls=10, subsample=100, tspan=None, options=None):
        """ 
        Constructor method
        Input
            model: underlying ctmc model
            obs_model: observation model
            obs_times: array of observation times
            obs_data: array of observation vectors 
            options: a dictionary with options. currently supported options
                    'mode': 'smoothing', 'inference'
        """
        # store stuff
        self.model = model
        self.obs_model = obs_model
        self.obs_times = obs_times
        self.obs_data = obs_data
        self.num_intervals = len(obs_times)+1
        self.options = self.set_options(options)
        # set additional stuff
        self.num_controls = num_controls
        self.subsample = subsample
        self.tspan = self.set_tspan(tspan)
        # set up containers
        self.initial = model.get_initial()
        self.rates = np.zeros((self.model.num_param()))
        self.control = np.zeros((self.num_intervals, self.num_controls, self.model.num_controls()))
        self.control_gradient = np.zeros((self.num_intervals, self.num_controls, self.model.num_controls()))
        self.rates_gradient = None #np.zeros((self.model.num_param()))
        #self.stats = np.zeros((self.num_intervals, self.num_controls, self.model.num_controls()))
        self.time = self.set_time()
        self.forward = np.zeros((self.num_intervals, self.num_controls, self.subsample+1, self.model.num_states()))
        self.backward = np.zeros((self.num_intervals, self.num_controls, self.subsample+1, self.model.num_states()))
        self.kl = np.zeros((self.num_intervals, self.num_controls, self.subsample+1))
        self.residual = np.zeros(len(obs_times))
        self.rates = self.initialize_rates()

    def set_options(self, options):
        if options is None:
            options = {'mode': 'smoothing'}
        return(options)

    def set_tspan(self, tspan):
        """
        Compute a default time step depending on the observation times
        """
        if (tspan is None):
            delta = (self.obs_times.max()-self.obs_times.min())/self.num_intervals
            t_min = self.obs_times.min()-delta
            t_max = self.obs_times.max()+delta
            tspan = np.array([t_min, t_max])
        return tspan

    def set_time(self):
        """
        Construct timte grid
        """
        start = np.concatenate([self.tspan[[0]], self.obs_times])
        end = np.concatenate([self.obs_times, self.tspan[[1]]])
        time = np.zeros((self.num_intervals, self.num_controls, self.subsample+1))
        for i in range(self.num_intervals):
            delta = np.linspace(start[i], end[i], self.num_controls+1)
            #print(delta)
            for j in range(self.num_controls):
                time[i, j, :] = np.linspace(delta[j], delta[j+1], self.subsample+1)
        return(time)

    def initialize_rates(self):
        """
        Initialize the rate vector
        """
        rates = np.array(self.model.rates)
        return(rates)

    def initialize_control(self, time, control):
        """
        Initialize the control to the prior rates
        If no control is provided, use prior rates
        """
        tmp = interp1d(time, control, axis=0)(self.time[:, :, 0].flatten())
        self.control = tmp.reshape(self.control.shape)
        return

    def initialize(self, time=None, control=None):
        if time is None and control is None:
            time = self.tspan
            control = np.zeros((2, self.model.num_controls()))
        # initialize from rates
        self.initialize_control(time, control)
        return
    
    # main dynamic functions

    def forward_update(self):
        # preparations
        initial = self.initial.copy()
        # iterate over the rest
        for i in range(self.num_intervals):
            for j in range(self.num_controls):
                # wrap function for fixed control
                def odefun(time, state):
                    return(self.model.forward(time, state, self.control[i, j], self.rates))
                tspan = self.time[i, j, [0, -1]]
                sol = solve_ivp(odefun, tspan, initial, t_eval=self.time[i, j])
                self.forward[i, j] = sol['y'].transpose()
                initial = sol['y'][:, -1]

    def backward_update(self):
        # set constraint in last interval 
        terminal = np.zeros(self.model.num_states())
        # iterate backward
        for i in reversed(range(self.num_intervals-1)):
            # update terminal state with observations
            terminal += self.obs_model.get_terminal(self.forward[i, -1, -1], self.obs_data[i], self.obs_times[i])
            for j in reversed(range(self.num_controls)):
                # wrap function for fixed control
                #print('Interval {0} subinterval {1}'.format(i, j))
                def odefun(time, state):
                    #print(time)
                    #print(self.time[i, j, [-1, 0]])
                    return(self.model.backward(time, state, self.control[i, j], self.time[i, j], self.forward[i, j], self.rates))
                tspan = self.time[i, j, [-1, 0]]
                sol = solve_ivp(odefun, tspan, terminal, t_eval=self.time[i, j, ::-1])
                self.backward[i, j] = sol['y'].transpose()[::-1, :]
                terminal = sol['y'][:, -1]

    def gradient_update(self):
        if self.options['mode'] == 'smoothing':
            self.control_gradient = self.model.control_gradient(self.time, self.control, self.forward, self.backward, self.rates)
        elif self.options['mode'] == 'inference':
            #self.control_gradient = self.model.control_gradient(self.time, self.control, self.forward, self.backward, self.rates)
            #self.rates_gradient = self.model.rates_gradient(self.time, self.control, self.forward, self.backward, self.rates)
            self.control_gradient, self.rates_gradient = self.model.joint_gradient(self.time, self.control, self.forward, self.backward, self.rates)
        else:
            raise ValueError('Invalid value ' + self.options['mode'] + ' for option mode')
        return

    # objective function evaluation

    def objective_function(self, control=None, get_gradient=True):
        # set control
        if control is not None:
            self.control = control
        # evaluate backward integration only if gradient information is required 
        if get_gradient:
            # update all sub-components
            self.forward_update()
            self.backward_update()
            # update objective function
            self.evaluate_kl()
            self.evaluate_residuals()
            #self.compute_statistics()
            elbo = self.kl.sum()+self.residual.sum()
            # update the gradient (uses stats)
            self.gradient_update()
            grad = self.control_gradient.copy()
            grad[np.isnan(grad)] = 0.0
            return(elbo, grad)
        else:
            # update only the required sub-components
            self.moment_update()
            # compute the elbo
            self.evaluate_kl()
            self.evaluate_residuals()
            #self.compute_statistics()
            elbo = self.kl.sum()+self.residual.sum()
            return(elbo)

    def objective_function_inference(self, arg=None, get_gradient=True):
        # set stuf
        if arg is not None:
            self.control = arg[0]
            self.rates = arg[1]
        # evaluate backward integration only if gradient information is required 
        if get_gradient:
            # update all sub-components
            self.forward_update()
            self.backward_update()
            # update objective function
            self.evaluate_kl()
            self.evaluate_residuals()
            #self.compute_statistics()
            elbo = self.kl.sum()+self.residual.sum()
            # update the gradient (uses stats)
            self.gradient_update()
            grad = [self.control_gradient.copy(), self.rates_gradient.copy()]
            grad[0][np.isnan(grad[0])] = 0.0
            grad[1][np.isnan(grad[1])] = 0.0
            return(elbo, grad)
        else:
            # update only the required sub-components
            self.moment_update()
            # compute the elbo
            self.evaluate_kl()
            self.evaluate_residuals()
            #self.compute_statistics()
            elbo = self.kl.sum()+self.residual.sum()
            return(elbo)

    def evaluate_residuals(self):
        """
        For each observation point, compute the residual contribution 
        corresonding to the expected negative log likliehood of the observation model
        """
        for i in range(len(self.obs_data)):
            self.residual[i] = self.obs_model.get_residual(self.forward[i, -1, -1, :], self.obs_data[i], self.obs_times[i])

    # def compute_statistics(self):
    #     """
    #     Compute sufficient statistics to evaluate the objective function.
    #     For the piece-wise constant controls, it is sufficient to integrate the natural moments
    #     over the subsampling interval
    #     """
    #     # get propensities
    #     propensities = self.model.natural_moments(self.forward.reshape((-1, self.forward.shape[-1])))
    #     propensities = propensities.reshape((self.num_intervals, self.num_controls, self.subsample+1, -1))
    #     # get time interval
    #     delta = self.time[:, :, 1:]-self.time[:, :, :-1]
    #     # integral of the natural moments
    #     val = 0.5*(propensities[:, :, 1:, :] + propensities[:, :, :-1, :])
    #     self.stats = np.einsum('ijm,ijmk->ijk', delta, val)
        return

    # evaluate the evidence lower bound
    def evaluate_kl(self):
        """
        Evaluate objective function based on stored values in stats
        """
        # prior kl contribution
        self.kl = self.model.kl_prior(self.time, self.control, self.forward, self.rates)
        return

    # getters 

    def get_time(self):
        """
        Return time as an array of shape (time_steps,)
        """
        return(self.time.flatten())

    def get_control(self):
        """
        Return forward states as an array of shape (time_steps, num_states)
        """
        return(self.control.reshape(-1, self.model.num_controls()))

    def get_forward(self):
        """
        Return forward states as an array of shape (time_steps, num_states)
        """
        return(self.forward.reshape(-1, self.model.num_states()))

    def get_backward(self):
        """
        Return backward states as an array of shape (time_steps, num_states)
        """
        return(self.backward.reshape(-1, self.model.num_states()))

    def get_gradient(self):
        """
        Return gradients as an array of shape (time_steps, num_controls)
        """
        return(self.control_gradient.reshape(-1, self.model.num_controls()))