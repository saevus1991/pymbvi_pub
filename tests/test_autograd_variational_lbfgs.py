import pymbvi.variational_engine as vi
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from pymbvi.models.observation.kinetic_obs_model import LognormObs
from pymbvi.models.observation.kinetic_obs_model import SingleGaussObs
from pymbvi.models.mjp.autograd_model import SimpleGeneExpression
from pymbvi.optimize import robust_gradient_descent, bound_projector
#from pymbvi.models.special_models import SimpleGeneExpression as CompareModel
#from pymbvi.models.autograd_model import AutogradModel
from pymbvi.variational_engine import VariationalEngine
from pymbvi.odeint import OneStepMethod, solve_ode

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.kinetic_model import KineticModel

# fix seed
#np.random.seed(9134874)

# forward class

def trapezoid_integrator(time, val):
    """
    trapezoid rule integrator
    Input
        time: increasing tensor of shape (n_t,)
        array: tensor of shape (n_t, dim_val)
    Output
        output" tensor of shape (n_t-1, dim_val)
    """
    # get time interval
    delta_t = time[1:]-time[:-1]
    # interpolate grid values
    val_interp = 0.5*(val[1:, :]+val[:-1, :])
    # produce output
    output = val_interp*delta_t.unsqueeze(1)
    return(output)


def rk(odefun, time, state, stepsize):
    """
    Classical Runge Kutta method
    """
    # calculate support points
    k1 = odefun(time, state)
    k2 = odefun(time+0.5*stepsize, state+0.5*stepsize.unsqueeze(1)*k1)
    k3 = odefun(time+0.5*stepsize, state+0.5*stepsize.unsqueeze(1)*k2)
    k4 = odefun(time+stepsize, state+stepsize.unsqueeze(1)*k3)
    # get increment
    delta = (k1+2*k2+2*k3+k4)/6.0
    return(delta)

def euler(odefun, time, state, stepsize):
    """
    Classical Runge Kutta method
    """
    # calculate support points
    delta = odefun(time, state)
    return(delta)

def forward_contribution(time, natural_moments, kl_eff):
    """
    Forward contribution of the KL divergence
    """
    integral = trapezoid_integrator(time, natural_moments*kl_eff)
    return(integral.sum())


def backward_contribution(time, forward, forward_new, backward):
    """
    Contribution of the lagrange multipliers
    """
    tmp = (forward[1:, :] - forward_new)*backward[:-1, :]
    return(tmp.sum())


def observation_contribution(time, forward, obs_model, obs_data):
    # get observed values
    obs_times = time[:-1, -1, -1]
    obs_states = forward[:-1, -1, -1, :]
    # compute residuals
    residuals = obs_model.get_residual_torch(obs_states, obs_data, obs_times)
    return(residuals.sum())


def forward_update(odefun, time, forward):
    """
    one step update
    """
    stepsize = time[1:]-time[:-1]
    delta = rk(odefun, time[:-1], forward[:-1, :], stepsize)
    new_forward = forward[:-1, :]+stepsize.unsqueeze(1)*delta
    return(new_forward)

def constraint(time, state, control):
    def odefun(time, state):
        return(model.forward_batch(time, state, control_flat[:-1, :]))
    new_forward = forward_update(odefun, time.flatten(), forward)
    return(forward[1:, :] - forward_new)

def objective_function(time, control, forward, model, obs_model, obs_data):
    # compute model specific stuff
    natural_moments = model.natural_moments_torch(forward_flat.T)
    kl_eff = model.rates-control_flat+control_flat*torch.log(control_flat/model.rates)
    # forward contribution
    forward_loss = forward_contribution(time.flatten(), natural_moments, kl_eff)
    # observation contribution
    obs_loss = observation_contribution(time, torch.exp(forward), obs_model, obs_data)
    # combined loss
    loss = forward_loss+obs_loss
    return(grad_loss)

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
moment_initial = torch.zeros(9)
moment_initial[0] = 1
model = SimpleGeneExpression(moment_initial, rates, tspan)

# # get forward  mean and stuff
# def ode_fun(time, state):#, control_time, control):
#     dydt = model.forward(time, state.numpy(), np.array(rates))
#     return(torch.from_numpy(dydt))

# for i in range(10):
#     time, sol = solve_ode(ode_fun, moment_initial, num_steps=3000, stepsize=1, method='rk')

# plt.plot(time, sol[:, 1], '-b')
# plt.plot(time, sol[:, 2], '-r')
# plt.show()

# # solve
# test = solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
# t_prior = test['t']
# states_prior = test['y']

# set up variational engine
vi_engine = VariationalEngine(model, obs_model, t_obs, observations, subsample=30, tspan=tspan)
vi_engine.initialize()
vi_engine.objective_function()

print(vi_engine.residual.sum())
print(vi_engine.evaluate_forward_kl())

optimal_control = robust_gradient_descent(vi_engine.objective_function, vi_engine.control, projector=bound_projector, iter=1, h=1e-2)[0]


# test forward batch
time = torch.tensor(vi_engine.time)
# forward = moment_initial.repeat([torch.numel(time), 1]).reshape(time.shape+(-1, ))
# backward = torch.ones(forward.size())
# dim = vi_engine.control.shape[0]*vi_engine.control.shape[1]
# control = model.rates.repeat([dim, 1]).reshape(vi_engine.control.shape)
# forward = moment_initial.repeat([torch.numel(time), 1]).reshape(time.shape+(-1, ))
# forward[forward<1e-6] = 1e-6
# forward = torch.log(forward)
# backward = torch.ones(forward.size())
# dim = vi_engine.control.shape[0]*vi_engine.control.shape[1]
# control = model.rates.repeat([dim, 1]).reshape(vi_engine.control.shape)
# control = torch.log(control)
forward = torch.log(torch.tensor(vi_engine.forward))
forward[forward<-10.0] = -10.0
backward = torch.tensor(vi_engine.backward)
control = torch.log(torch.tensor(vi_engine.control))
obs_data = torch.from_numpy(observations).flatten()

# learning rate
alpha = 1e-6
n_iter = 1

# prepare gradients
forward.requires_grad = True
backward.requires_grad = True
control.requires_grad = True

minimize(objective, initial, 

# initalize
loss = objective_function(time, control, forward, backward, model, obs_model, obs_data)
loss.backward()



# test = moment_initial.repeat([torch.numel(time), 1])
# control_flat = torch.exp(control.unsqueeze(2).repeat(1, 1, forward.shape[2], 1).reshape(-1, control.shape[-1]))
# def odefun(time, state):
#     return(model.forward_batch(time, state, control_flat[:-1, :]))
# for i in range(1000):
#     tmp = forward_update(odefun, time.flatten(), test)
#     test = torch.cat([moment_initial.unsqueeze(0), tmp])

# otpimization via lbfgs

# optimizer = torch.optim.LBFGS((forward, backward, control))
# def closure():
#     optimizer.zero_grad()
#     loss = objective_function(time, control, forward, backward, model, obs_model, obs_data)
#     print(loss)
#     loss.backward()
#     return loss

# optimizer.step(closure)

# elbo = vi_engine.objective_function()[0]

# # project to a given domain
# def bound_projector(arg):
#     """
#     Cut of values above or below boundary
#     """
#     # set bounds
#     bounds = np.array([1e-6, 1e6])
#     # cut off larger values
#     arg[arg > bounds[1]] = bounds[1]
#     # cut off lower values
#     arg[arg < bounds[0]] = bounds[0]


# # test optimization
# initial_control = vi_engine.control.copy()
# optimal_control = robust_gradient_descent(vi_engine.objective_function, initial_control, projector=bound_projector, iter=100, h=1e-2)[0]
# vi_engine.objective_function(optimal_control)

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


t_prior = vi_engine.get_time()
states_prior = np.exp(forward.detach().reshape(-1, forward.shape[-1]).numpy())
t_intermediate = t_prior
states_intermediate = vi_engine.get_forward()

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
    plt.plot(t_prior, states_prior[:, 1] + np.sqrt(states_prior[:, 6]), '--b')
    plt.plot(t_prior, states_prior[:, 1] - np.sqrt(states_prior[:, 6]), '--b')
    plt.plot(t_prior, states_prior[:, 2], '-r')
    plt.plot(t_prior, states_prior[:, 2] + np.sqrt(states_prior[:, 8]), '--r')
    plt.plot(t_prior, states_prior[:, 2] - np.sqrt(states_prior[:, 8]), '--r')
    plt.plot(t_obs, observations, 'xk')

    plt.subplot(3, 1, 3)
    plt.plot(t_intermediate, 100*states_intermediate[:, 0], '-k')
    plt.plot(t_intermediate, states_intermediate[:, 1], '-b')
    plt.plot(t_intermediate, states_intermediate[:, 1] + np.sqrt(states_intermediate[:, 6]), '--b')
    plt.plot(t_intermediate, states_intermediate[:, 1] - np.sqrt(states_intermediate[:, 6]), '--b')
    plt.plot(t_intermediate, states_intermediate[:, 2], '-r')
    plt.plot(t_intermediate, states_intermediate[:, 2] + np.sqrt(states_intermediate[:, 8]), '--r')
    plt.plot(t_intermediate, states_intermediate[:, 2] - np.sqrt(states_intermediate[:, 8]), '--r')
    plt.plot(t_obs, observations, 'xk')

    plt.show()
