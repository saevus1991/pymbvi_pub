# This file collects several obsrevation model classes relevant for kinetic models

from pymbvi.models.observation.obs_model import ObservationModel
import numpy as np
import torch

class SingleGaussObs(ObservationModel):
    """
    This model implements observations of a single species with Gaussian noise
    """
    
    def __init__(self, sigma, num_species, obs_species, num_moments=None, obs_moment=None):
        """
        The model requires two inputs
            sigma:  std of the gaussian noise
            selection: species that is observed encoded in a one-hot array
            num_species/num_moments and obs_species/obs_moment account for the fact that
            the number of moments may differ from the number of species
        """
        self.sigma = sigma
        self.num_species = num_species
        self.obs_species = obs_species
        if num_moments is None:
            self.num_moments = num_species
        else:
            self.num_moments = num_moments
        if obs_moment is None:
            self.obs_moment = obs_species
        else:
            self.obs_moment = obs_moment
        self.obs_second_moment = self.get_second_moment()

    def sample(self, state, time=None):
        return(state[self.obs_species]+self.sigma*np.random.randn())

    def llh(self, state, obs):
        llh = -0.5*np.log(2*np.pi)-np.log(self.sigma)-0.5*(obs-state[self.obs_species])**2/self.sigma**2
        return(llh)

    def get_residual(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # contribution of the normalizer
        residual = 0.5*np.log(2*np.pi)+np.log(self.sigma)
        # central moment contribution
        residual += 0.5/self.sigma**2*moments[self.obs_second_moment]
        # observation contribution
        residual += 0.5/self.sigma**2*(observation-moments[self.obs_moment])**2
        return(residual)

    def get_terminal(self, moments, observation, sample_time):
        """
        Compute the reset condition depending on moments and observations
        """
        # set up update
        terminal = np.zeros(moments.shape)
        # update observed species
        terminal[self.obs_moment] = (observation-moments[self.obs_moment])/self.sigma**2
        terminal[self.obs_second_moment] = -0.5/self.sigma**2
        return(terminal)

    # additional functions

    def get_second_moment(self):
        """
        Get the indices of the second order moments corresponding
        to selection
        """
        second_moment = int(self.num_moments+self.obs_moment*(self.num_moments-0.5*self.obs_moment+0.5))
        return(second_moment)


class MultiGaussObs(ObservationModel):
    """
    This model implements observations of multiple species with gaussian observations
    we assume uncorrelated noise for different species, i.e. sigma is vector
    """
    
    def __init__(self, sigma):
        """
        The model requires two inputs
            sigma:  std of the gaussian noise
            selection: species that is observed encoded in a one-hot array
            num_species/num_moments and obs_species/obs_moment account for the fact that
            the number of moments may differ from the number of species
        """
        self.sigma = sigma
        self.num_species = len(sigma)
        self.obs_second_moment = self.get_second_moment()

    def sample(self, state, time=None):
        output = state[:self.num_species]+self.sigma*np.random.randn(self.num_species)
        return(output)

    def llh(self, state, obs):
        llh = -0.5*self.num_species*np.log(2*np.pi)-np.log(self.sigma).sum()-0.5*np.sum((obs-state[:self.num_species])**2/self.sigma**2)
        return(llh)

    def get_residual(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # contribution of the normalizer
        residual = 0.5*self.num_species*np.log(2*np.pi)+np.log(self.sigma).sum()
        # central moment contribution
        residual += 0.5*(moments[self.obs_second_moment]/self.sigma**2).sum()
        # observation contribution
        residual += 0.5*((observation-moments[:self.num_species])**2/self.sigma**2).sum()
        return(residual)

    def get_terminal(self, moments, observation, sample_time):
        """
        Compute the reset condition depending on moments and observations
        """
        # set up update
        terminal = np.zeros(moments.shape)
        # update observed species
        terminal[:self.num_species] = (observation-moments[:self.num_species])/self.sigma**2
        terminal[self.obs_second_moment] = -0.5/self.sigma**2
        return(terminal)

    # additional functions

    def get_second_moment(self):
        """
        Get the indices of the second order moments corresponding
        to selection
        """
        ind = np.array([i for i in range(1, self.num_species+1)])
        second_moment = ind*(ind+1)/2-1+self.num_species
        return(second_moment.astype(int))


class LognormObs(ObservationModel):
    """
    This model implements observations of a single species with lognormal noise
    uses an approximation for residual/terminal evaluation
    """
    
    def __init__(self, sigma, num_species, obs_species, num_moments=None, obs_moment=None):
        """
        The model requires two inputs
            sigma:  std of the gaussian noise
            selection: species that is observed encoded in a one-hot array
            num_species/num_moments and obs_species/obs_moment account for the fact that
            the number of moments may differ from the number of species
        """
        self.sigma = sigma
        self.num_species = num_species
        self.obs_species = obs_species
        if num_moments is None:
            self.num_moments = num_species
        else:
            self.num_moments = num_moments
        if obs_moment is None:
            self.obs_moment = obs_species
        else:
            self.obs_moment = obs_moment
        self.obs_second_moment = self.get_second_moment()

    def sample(self, state, time=None):
        sample = state[self.obs_species]*np.exp(self.sigma*np.random.randn())
        return(sample)

    def llh(self, state, obs):
        llh = -0.5*np.log(2*np.pi)-np.log(self.sigma*obs)-0.5*(np.log(obs)-np.log(state[self.obs_species]))**2/self.sigma**2
        return(llh)

    def get_residual(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # aproximate evaluation by scaling noise with observation
        sigma = self.sigma*observation
        # contribution of the normalizer
        residual = 0.5*np.log(2*np.pi)+np.log(sigma)
        # central moment contribution
        residual += 0.5/sigma**2*moments[self.obs_second_moment]
        # observation contribution
        residual += 0.5/sigma**2*(observation-moments[self.obs_moment])**2
        return(residual)

    def get_residual_torch(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # aproximate evaluation by scaling noise with observation
        sigma = torch.tensor(self.sigma)*observation
        # contribution of the normalizer
        residual = 0.5*torch.log(2*torch.tensor(np.pi))+torch.log(sigma)
        # central moment contribution
        residual += 0.5/sigma**2*moments[:, self.obs_second_moment]
        # observation contribution
        residual += 0.5/sigma**2*(observation-moments[:, self.obs_moment])**2
        return(residual)

    def get_terminal(self, moments, observation, sample_time):
        """
        Compute the reset condition depending on moments and observations
        """
        # aproximate evaluation by scaling noise with observation
        sigma = self.sigma*observation
        # set up update
        terminal = np.zeros(moments.shape)
        # update observed species
        terminal[self.obs_moment] = (observation-moments[self.obs_moment])/sigma**2
        terminal[self.obs_second_moment] = -0.5/sigma**2
        return(terminal)


    # additional functions

    def get_second_moment(self):
        """
        Get the indices of the second order moments corresponding
        to selection
        """
        second_moment = int(self.num_moments+self.obs_moment*(self.num_moments-0.5*self.obs_moment+0.5))
        return(second_moment)