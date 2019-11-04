import unittest
import pymbvi.models.observation.kinetic_obs_model as kom
import numpy as np
from scipy.stats import norm
from pymbvi.util import num_derivative


class TestSingleGaussObs(unittest.TestCase):

    def test_construction(self):
        # set up model
        sigma = 5
        num_species = 3
        obs_species = 2
        obs_model = kom.SingleGaussObs(sigma, num_species, obs_species)
        # perform checks
        self.assertEqual(sigma, obs_model.sigma)
        self.assertEqual(num_species, obs_model.num_species)
        self.assertEqual(obs_species, obs_model.obs_species)
        self.assertEqual(obs_model.obs_second_moment, 8)

    def test_sample(self):
        # set up model
        sigma = 0.0
        num_species = 3
        obs_species = 2
        obs_model = kom.SingleGaussObs(sigma, num_species, obs_species)
        # check 
        state = np.round(np.abs(10*np.random.randn(num_species)))
        obs = obs_model.sample(state)
        self.assertAlmostEqual(state[2], obs)

    def test_lhh(self):
        # set up model
        sigma = 5
        num_species = 3
        obs_species = 2
        obs_model = kom.SingleGaussObs(sigma, num_species, obs_species)
        # chck trivial values
        state = np.round(np.abs(10*np.random.randn(num_species)))
        obs = state[2]
        self.assertAlmostEqual(-np.log(np.sqrt(2*np.pi)*sigma), obs_model.llh(state, obs))
        # check additional random value
        obs = state[2]+sigma*np.random.randn()
        llh_test = norm.logpdf(obs, state[2], sigma)
        self.assertAlmostEqual(llh_test, obs_model.llh(state, obs))

    def test_get_residual(self):
        # set up model
        sigma = 5
        num_species = 3
        obs_species = 2
        obs_model = kom.SingleGaussObs(sigma, num_species, obs_species)
        # get values
        state = np.round(np.abs(10*np.random.randn(num_species)))
        obs = state[2]+sigma*np.random.randn()
        sample_time = 1.0
        dim = int(num_species*(num_species+3)/2)
        moments = np.round(np.abs(10*np.random.randn(dim)))
        # compute test values
        test_val = obs_model.get_residual(moments, obs, sample_time)
        second_moment = moments[obs_species]**2+moments[-1]
        contr_val = np.log(np.sqrt(2*np.pi)*sigma)+(second_moment-2*moments[obs_species]*obs+obs**2)/(2*sigma**2)
        self.assertAlmostEqual(test_val, contr_val)

    def test_get_terminal(self):#, moments, observation, sample_time):
        # set up model
        sigma = 5
        num_species = 3
        obs_species = 2
        obs_model = kom.SingleGaussObs(sigma, num_species, obs_species)
        # get values
        state = np.round(np.abs(10*np.random.randn(num_species)))
        obs = state[2]+sigma*np.random.randn()
        sample_time = 1.0
        dim = int(num_species*(num_species+3)/2)
        moments = np.round(np.abs(10*np.random.randn(dim)))
        # compute test values
        test_val = obs_model.get_terminal(moments, obs, sample_time)
        def fun(x): return(obs_model.get_residual(x, obs, sample_time))
        contr_val = -num_derivative(fun, moments)
        err = np.linalg.norm(test_val-contr_val, np.inf)
        self.assertAlmostEqual(err, 0.0)

if __name__ == '__main__':
    unittest.main()