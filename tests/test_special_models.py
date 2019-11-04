import pymbvi.variational_engine as vi
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pymbvi.models.observation.kinetic_obs_model import SingleGaussObs
from pymbvi.models.mjp.special_models import SimpleGeneExpression

pyssa_path = '/Users/christian/Documents/Code/pyssa'
sys.path.append(pyssa_path)
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.kinetic_model import KineticModel 

# prepare  model for simulation
pre, post, rates = sm.get_standard_model("simple_gene_expression")

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)


# set up gene expression model
moment_initial = np.zeros(9)
moment_initial[0] = 1
model = SimpleGeneExpression(moment_initial, rates, tspan)

fj_control = np.array([[-0.002000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.150000, -0.001000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.040000, -0.008000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.000000, -0.004000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.000000, 0.150000, -0.003000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.000000, 0.000000, 0.040000, -0.010000, 0.000000, 0.000000, 0.000000, ],
[0.150000, 0.001000, 0.000000, 0.000000, 0.300000, 0.000000, -0.002000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.150000, 0.040000, -0.009000, 0.000000, ],
[0.000000, 0.040000, 0.008000, 0.000000, 0.000000, 0.000000, 0.000000, 0.080000, -0.016000, ],
])

fj_test = model.forward_jacobian(np.array(rates))

err = np.linalg.norm((fj_control-fj_test).flatten(), np.inf)
print(err)

fjc_control = np.array([[0.498761, -0.501239, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.501239, -74.814083, 0.000000, 0.000000, ],
[0.000000, 0.000000, 0.000000, 0.000000, 74.814083, -373.760556, ],
[-0.001236, 0.001243, 0.000000, 0.000000, 0.000000, 0.000000, ],
[-12.494056, -12.494056, 0.249998, -12.494056, 0.000000, 0.000000, ],
[-49.966256, -49.966256, 0.000000, 0.000000, 12.494056, -49.966256, ],
[0.000000, 0.000000, 25.489352, -3772.675057, 0.000000, 0.000000, ],
[0.000000, 0.000000, 49.966256, -9350.818080, 1923.744570, -9350.818080, ],
[0.000000, 0.000000, 0.000000, 0.000000, 18776.450243, -93633.775844, ],
])

state = np.array([5.01239444e-01, 7.48140834e01, 3.73760556e02, 2.49998453e-01,
 1.24940561e01, 4.99662556e01, 1.92374457e03, 9.35081808e03,
 4.70037682e04])
fjc_test = model.forward_jacobian_control(state)

err = np.linalg.norm((fjc_control-fjc_test).flatten(), np.inf)
print(err)

pj_control = np.array([[-1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
[0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ],
])

pj_test = model.prop_jacobian(state)

err = np.linalg.norm((pj_control-pj_test).flatten(), np.inf)
print(err)

deta_control = np.array([-2.7732, -1.2165, -0.1490, -1.5564, -5.3060, -2.6109, -0.8075, -1.7256, 0.4000])

m_t = np.array([2000, 2010])
eta = np.linspace(0.1, 20, 9)
m = np.array([state*0.95+moment_initial*0.05, state])
c = np.array([rates, rates])
alpha = np.array(rates)*np.array([0.9, 0.85, 1.05, 1.1, 1.2, 1.25])
t = 2003.0

deta_test = model.backward(t, eta, alpha, m_t, m, c)

err = np.linalg.norm((deta_control-deta_test).flatten(), np.inf)
print(err)