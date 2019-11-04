import pymbvi.variational_engine as vi
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pymbvi.models.observation.tasep_obs_model import Gauss, LognormGauss
from pymbvi.models.observation.kinetic_obs_model import SingleGaussObs
#from pymbvi.models.linear_model import SimpleGeneExpression
#from pymbvi.models.special_models import SimpleGeneExpression as CompareModel
from pymbvi.models.mjp.autograd_model import BernoulliTasep2
from pymbvi.variational_engine import VariationalEngine
from pymbvi.optimize import multi_arg_gradient_descent

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.special_models import TASEP

# fix seed
#np.random.seed(9134874)

# activate plotting
plotting = True

# prepare  model for simulation
num_stems = 14
alpha, rates, obs_param = sm.get_standard_model("tasep")
model = TASEP(len(alpha), np.array(rates))

# prepare initial conditions
initial = np.zeros(48)
tspan = np.array([0.0, 120*10])

# set up simulator
simulator = ssa.Simulator(model, initial)

# set up an observation model
obs_param[4] = 0.1
obs_model = LognormGauss(np.array(obs_param), np.array(alpha))
#obs_model = Gauss(np.array(obs_param), np.array(alpha))

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# produce observations 
delta_t = 10.0
t_obs = np.arange(delta_t, tspan[1], delta_t)
observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

# set up variational tasep model
moment_initial = initial
rates = [rates[0]] + [rates[1] for i in range(len(alpha)-1)] + [rates[2]]
model = BernoulliTasep2(moment_initial, np.array(rates), tspan)

# get forward  mean and stuff
def ode_fun(time, state):#, control_time, control):
    return(model.forward(time, state, np.array(rates)))

# solve 
test = solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
t_prior = test['t']
states_prior = test['y'].T

# set up variational engine
vi_engine = VariationalEngine(model, obs_model, t_obs, observations, num_controls=2, subsample=10, tspan=tspan)
vi_engine.initialize()

#print(vi_engine.time)

# control_old = vi_engine.control.copy()
# # compute forward  and backward stuff
# vi_engine.forward_update()
# forward_old = vi_engine.forward.copy()
# vi_engine.backward_update()
# backward_old = vi_engine.backward.copy()
# vi_engine.gradient_update()
# vi_engine.evaluate_residuals()
# vi_engine.compute_statistics()
# elbo = vi_engine.evaluate_kl()

# control_new = vi_engine.control.copy()
# forward_new = vi_engine.forward.copy()
# backward_new = vi_engine.backward.copy()
# assert(np.linalg.norm(control_new-control_old)==0.0)
# assert(np.linalg.norm(forward_new-forward_old)==0.0)
# assert(np.linalg.norm(backward_new-backward_old)==0.0)

# elbo = vi_engine.objective_function()[0]

# project to a given domain
def bound_projector(arg):
    """
    Cut of values above or below boundary
    """
    # extract relevant component (control)
    control = arg[0]
    # set bounds
    bounds = np.array([1e-6, 1e6])
    # cut off larger values
    control[control > bounds[1]] = bounds[1]
    # cut off lower values
    control[control < bounds[0]] = bounds[0]
    # extract initial
    initial = arg[1]
    # set bounds
    bounds = np.array([0.0, 1.0])
    # # cut off larger values
    # initial[initial > bounds[1]] = bounds[1]
    # # cut of lower values
    # initial[initial < bounds[0]] = bounds[0]

# objective function wrapper
def objective_fun(arg):
    sol = vi_engine.objective_function_free_initial(arg[0], arg[1])
    return(sol[0], sol[1:])


# test optimization
initial_control = vi_engine.control.copy()
initial_moment = np.ones(vi_engine.initial.shape)*1e-6
optimal_control = multi_arg_gradient_descent(objective_fun, [initial_control, initial_moment], projector=bound_projector, iter=20, h=1e-7)[0]
objective_fun(optimal_control)

print(optimal_control[1])

# get stuff for plotting
t_posterior = vi_engine.get_time()
states_posterior = vi_engine.get_forward()

# # time = vi_engine.get_time()
# # grad = vi_engine.get_gradient()
# # forward = vi_engine.get_forward()
# # backward = vi_engine.get_backward()

# # for i in range(0, 9):
# #     plt.subplot(3, 3, i+1)
# #     plt.plot(time, forward[:, i], '-b')
# #     plt.ylabel(str(i+1))
# # plt.show()

# # for i in range(0, 9):
# #     plt.subplot(3, 3, i+1)
# #     plt.plot(time, backward[:, i], '-b')
# # plt.show()

# # for i in range(0, 6):
# #     plt.subplot(3, 2, i+1)
# #     plt.plot(grad[:, i], '-b')
# # plt.show()


# t_prior = vi_engine.get_time()
# states_prior = vi_engine.get_forward()


# plot result 
if plotting:
    plt.subplot(3, 1, 1)
    plt.plot(t_plot, states_plot@alpha, '-r')
    plt.plot(t_prior, states_prior@alpha, '-b')
    plt.plot(t_posterior, states_posterior@alpha, '-k')

    plt.subplot(3, 1, 2)
    intensity = obs_model.intensity(states_plot, t_plot)
    prior_intensity = obs_model.intensity(states_prior, t_prior)
    posterior_intensity = obs_model.intensity(states_posterior, t_posterior)
    plt.plot(t_plot, intensity, '-r')
    plt.plot(t_prior, prior_intensity, '-b')
    plt.plot(t_posterior, posterior_intensity, '-k')

    plt.subplot(3, 1, 3)
    plt.plot(t_obs, observations, '-k')

    #plt.plot(t_prior, states_prior@alpha, '-r')
    #plt.plot(t_prior, states_prior@alpha, '-r')

    # plt.subplot(2, 1, 2)
    # plt.plot(t_prior, 100*states_prior[:, 0], '-k')
    # plt.plot(t_prior, states_prior[:, 1], '-b')
    # plt.plot(t_prior, states_prior[:, 1] + np.sqrt(states_prior[:, 6]), '--b')
    # plt.plot(t_prior, states_prior[:, 1] - np.sqrt(states_prior[:, 6]), '--b')
    # plt.plot(t_prior, states_prior[:, 2], '-r')
    # plt.plot(t_prior, states_prior[:, 2] + np.sqrt(states_prior[:, 8]), '--r')
    # plt.plot(t_prior, states_prior[:, 2] - np.sqrt(states_prior[:, 8]), '--r')
    # plt.plot(t_obs, observations, 'xk')

    plt.show()
