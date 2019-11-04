import pymbvi.variational_engine as vi
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pymbvi.models.observation.kinetic_obs_model import SingleGaussObs
from pymbvi.models.mjp.specific_models import SimpleGeneExpression
from pymbvi.variational_engine import VariationalEngine

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.kinetic_model import KineticModel 

# class TestVariationalEngine(unittest.TestCase):


def robst_gradient_descent(fun, initial, projector=None, iter=1, h=1e-2):
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
sigma = np.array([15.0])
num_species = 4
obs_species = 3
obs_model = SingleGaussObs(sigma, num_species, obs_species, num_species-1, obs_species-1)

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
model = SimpleGeneExpression(moment_initial, rates, tspan)

# get forward  mean and stuff
def ode_fun(time, state):#, control_time, control):
    return(model.forward(time, state, np.ones(len(rates))))

# solve 
test = solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
t_prior = test['t']
states_prior = test['y']

# set up variational engine
vi_engine = VariationalEngine(model, obs_model, t_obs, observations, subsample=30, tspan=tspan)
vi_engine.initialize(tspan, np.ones((2, len(rates))))

control_old = vi_engine.control.copy()
# compute forward  and backward stuff
vi_engine.forward_update()
forward_old = vi_engine.forward.copy()
vi_engine.backward_update()
backward_old = vi_engine.backward.copy()
vi_engine.gradient_update()
vi_engine.evaluate_residuals()
elbo = vi_engine.evaluate_kl()

control_new = vi_engine.control.copy()
forward_new = vi_engine.forward.copy()
backward_new = vi_engine.backward.copy()
assert(np.linalg.norm(control_new-control_old)==0.0)
assert(np.linalg.norm(forward_new-forward_old)==0.0)
assert(np.linalg.norm(backward_new-backward_old)==0.0)

elbo = vi_engine.objective_function()[0]

# project to a given domain
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


# test optimization
initial_control = vi_engine.control.copy()
robst_gradient_descent(vi_engine.objective_function, initial_control, projector=bound_projector, iter=100, h=1e-2)


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


t_prior = vi_engine.get_time()
states_prior = vi_engine.get_forward()


# plot result 
if plotting:
    plt.subplot(2, 1, 1)
    plt.plot(t_plot, 100*states_plot[:, 1], '-k')
    plt.plot(t_plot, states_plot[:, 2], '-b')
    plt.plot(t_plot, states_plot[:, 3], '-r')
    plt.plot(t_obs, observations, 'xk')

    plt.subplot(2, 1, 2)
    plt.plot(t_prior, 100*states_prior[:, 0], '-k')
    plt.plot(t_prior, states_prior[:, 1], '-b')
    plt.plot(t_prior, states_prior[:, 2], '-r')
    plt.plot(t_obs, observations, 'xk')

    plt.show()
