import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pymbvi.models.observation.kinetic_obs_model import LognormObs
from pymbvi.models.observation.kinetic_obs_model import SingleGaussObs
from pymbvi.models.mjp.specific_models import SimpleGeneExpression
from pymbvi.optimize import robust_gradient_descent
from pymbvi.variational_engine import VariationalEngine

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.kinetic_model import KineticModel 

# fix seed
np.random.seed(191029950)

# activate plotting
plotting = True

# prepare  model for simulation
pre, post, rates = sm.get_standard_model("simple_gene_expression")
model = KineticModel(np.array(pre), np.array(post), np.array(rates))

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])

# set up simulator
simulator = ssa.Simulator(model, initial)

# set up an observation model
sigma = np.array([0.15])
num_species = 4
obs_species = 3
obs_model = LognormObs(sigma, num_species, obs_species, num_species-1, obs_species-1)

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# produce observations 
delta_t = 300.0
t_obs = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)
observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

# set up gene expression model
moment_initial = np.zeros(9)
moment_initial[0] = 1
model = SimpleGeneExpression(moment_initial, np.log(np.array(rates)), tspan)

# get forward  mean and stuff
def ode_fun(time, state):#, control_time, control):
    return(model.forward(time, state, np.zeros(len(rates)), np.log(np.array(rates))))

# solve 
test = solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
t_prior = test['t']
states_prior = test['y'].T

# set up variational engine
vi_engine = VariationalEngine(model, obs_model, t_obs, observations, subsample=30, tspan=tspan)
vi_engine.initialize()

# project to a given domain
def bound_projector(arg):
    """
    Cut of values above or below boundary
    """
    # set bounds
    bounds = np.array([-13, 13])
    # cut off larger values
    arg[arg > bounds[1]] = bounds[1]
    # cut off lower values
    arg[arg < bounds[0]] = bounds[0]

# test optimization
initial_control = vi_engine.control.copy()
optimal_control = robust_gradient_descent(vi_engine.objective_function, initial_control, projector=bound_projector, iter=10, h=1e-3)[0]
vi_engine.objective_function(optimal_control)

print(np.exp(vi_engine.rates))

#print(vi_engine.residual.sum())
#print(vi_engine.kl.sum())

# time = vi_engine.get_time()
# grad = vi_engine.get_gradient()
# forward = vi_engine.get_forward()
# backward = vi_engine.get_backward()

# for i in range(0, 9):
#     plt.subplot(3, 3, i+1)
#     plt.plot(time, forward[:, i], '-b')
#     plt.ylabel(str(i+1))
# plt.show()

# for i in range(0, 9):
#     plt.subplot(3, 3, i+1)
#     plt.plot(time, backward[:, i], '-b')
# plt.show()

# for i in range(0, 6):
#     plt.subplot(3, 2, i+1)
#     plt.plot(grad[:, i], '-b')
# plt.show()


t_post = vi_engine.get_time()
states_post = vi_engine.get_forward()


# plot result 
if plotting:
    plt.subplot(3, 1, 1)
    plt.plot(t_plot, 100*states_plot[:, 1], '-k')
    plt.plot(t_plot, states_plot[:, 2], '-b')
    plt.plot(t_plot, states_plot[:, 3], '-r')
    plt.plot(t_obs, observations, 'xk')

    plt.subplot(3, 1, 2)
    plt.plot(t_prior, 100*states_prior[:, 0], '-k')
    plt.plot(t_prior, states_prior[:, 1], '-b')
    plt.plot(t_prior, states_prior[:, 2], '-r')
    plt.plot(t_obs, observations, 'xk')

    plt.subplot(3, 1, 3)
    plt.plot(t_post, 100*states_post[:, 0], '-k')
    plt.plot(t_post, states_post[:, 1], '-b')
    plt.plot(t_post, states_post[:, 2], '-r')
    plt.plot(t_obs, observations, 'xk')


    plt.show()
