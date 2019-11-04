# implementation of a specific form of the observation model used for the tasep process as model for transcripiton

from pymbvi.models.observation.obs_model import ObservationModel
import numpy as np


class Gauss(ObservationModel):
    """
    This model implements observations of the tasep model. Observations are obtained from
    a weighted sum of polymerases. 
    """
    
    def __init__(self, obs_param, alpha):
        """
        The model requires two inputs
            obs_param:  np.array([b0, b1, lambda, gamma, sigma]) where
                b0: non-bleachable background
                b1: bleachable background
                ldambda: rate of exponential bleaching model
                gamma: scaling factor between active stemloop and observed intensity
                sigma: noise level of the mulitplicative noise
        """
        self.b0, self.b1, self.lamb, self.gamma, self.sigma = obs_param.reshape(-1, 1)
        self.alpha = alpha

    def sample(self, state, time):
        # compute raw intensity
        I = self.intensity(state, time)
        # apply multiplicative noise
        sample = I+self.sigma*np.random.standard_normal(I.shape)
        return(sample)

    def llh(self, state, time, obs):
        # compute intensity
        I = self.intensity(state, time)
        # compute log likelihood
        llh = 0.5*np.sum((obs-I)**2)/self.sigma**2
        llh += (-0.5*np.log(2*np.pi)-np.log(self.sigma))*len(I)
        return(llh)

    def get_residual(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # get required summary statistics
        exp_intensity = self.intensity(moments, sample_time)
        var_intensity = self.var_intensity(moments, sample_time)
        # contribution of the normalizer
        residual = 0.5*np.log(2*np.pi)+np.log(self.sigma)
        # central moment contribution
        residual += 0.5*var_intensity/self.sigma**2
        # observation contribution
        residual += 0.5*(observation-exp_intensity)**2/self.sigma**2
        return(residual)

    def get_terminal(self, moments, observation, sample_time):
        """
        Compute the reset condition depending on moments and observations
        """
        # get required summary statistics
        exp_intensity = self.intensity(moments, sample_time)
        grad_exp_intensity = self.grad_exp_intensity(moments, sample_time)
        grad_var_intensity = self.grad_var_intensity(moments, sample_time)
        # get updates
        terminal = (observation-exp_intensity)/self.sigma**2*grad_exp_intensity
        terminal += -0.5/self.sigma**2*grad_var_intensity
        return(terminal)

    # helper functions to calculate some intermediate values

    def intensity(self, state, time):
        """
        compute intensity (without stochstic noise part)
            state is either an array of shape (L, ) or (num_steps, L)
            time shold be of shape (1, ) ore (num_steps, ) according to the form of state
        """
        # get number of stemloops
        N = state @ self.alpha
        # add background and scaling
        I = self.b0 + np.exp(-self.lamb*time)*(self.b1+self.gamma*N)
        return(I)

    def grad_exp_intensity(self, moments, time):
        """
        Calculate expected intensity under product bernoulli approximation
        """
        # get number of stemloops
        N = moments @ self.alpha
        # compute gradient
        grad = np.exp(-self.lamb*time)*(self.b1+self.gamma*N)*self.alpha
        return(grad)

    def var_intensity(self, moments, time):
        """ 
        Calculate variance of intensity under product bernoulli approximation
        """
        prefactor = np.exp(-2*self.lamb*time)*self.gamma**2
        N_sq = np.sum(self.alpha**2*moments*(1-moments)**2)
        var = prefactor*N_sq
        return(var)

    def grad_var_intensity(self, moments, time):
        prefactor = np.exp(-2*self.lamb*time)*self.gamma**2
        grad = prefactor*self.alpha**2*(1-2*moments)
        return(grad)


class LognormGauss(ObservationModel):
    """
    This model implements observations of the tasep model. Observations are obtained from
    a weighted sum of polymerases. 
    """
    
    def __init__(self, obs_param, alpha):
        """
        The model requires two inputs
            obs_param:  np.array([b0, b1, lambda, gamma, sigma]) where
                b0: non-bleachable background
                b1: bleachable background
                ldambda: rate of exponential bleaching model
                gamma: scaling factor between active stemloop and observed intensity
                sigma: noise level of the mulitplicative noise
        """
        self.b0, self.b1, self.lamb, self.gamma, self.sigma = obs_param.reshape(-1, 1)
        self.alpha = alpha

    def sample(self, state, time):
        # compute raw intensity
        I = self.intensity(state, time)
        # apply multiplicative noise
        sample = I*np.exp(self.sigma*np.random.standard_normal(I.shape))
        return(sample)

    def llh(self, state, time, obs):
        # compute intensity
        I = self.intensity(state, time)
        # compute log likelihood
        llh = 0.5*np.sum((np.log(obs)-np.log(I)))**2/self.sigma**2
        llh += -np.sum(np.log(obs))
        llh += (-0.5*np.log(2*np.pi)-np.log(self.sigma*obs))*len(I)
        return(llh)

    def get_residual(self, moments, observation, sample_time):
        """
        Compute contribution of the obseravtions to the objective functions
        """
        # aproximate evaluation by scaling noise with observation
        sigma = self.sigma*observation
        # get required summary statistics
        exp_intensity = self.intensity(moments, sample_time)
        var_intensity = self.var_intensity(moments, sample_time)
        # contribution of the normalizer
        residual = 0.5*np.log(2*np.pi)+np.log(sigma)
        # central moment contribution
        residual += 0.5*var_intensity/sigma**2
        # observation contribution
        residual += 0.5*(observation-exp_intensity)**2/sigma**2
        return(residual)

    def get_terminal(self, moments, observation, sample_time):
        """
        Compute the reset condition depending on moments and observations
        """
        # aproximate evaluation by scaling noise with observation
        sigma = self.sigma*observation
        # get required summary statistics
        exp_intensity = self.intensity(moments, sample_time)
        grad_exp_intensity = self.grad_exp_intensity(moments, sample_time)
        grad_var_intensity = self.grad_var_intensity(moments, sample_time)
        # get updates
        terminal = (observation-exp_intensity)/sigma**2*grad_exp_intensity
        terminal += -0.5/sigma**2*grad_var_intensity
        return(terminal)

    # helper functions to calculate some intermediate values

    def intensity(self, state, time):
        """
        compute intensity (without stochstic noise part)
            state is either an array of shape (L, ) or (num_steps, L)
            time shold be of shape (1, ) ore (num_steps, ) according to the form of state
        """
        # get number of stemloops
        N = state @ self.alpha
        # add background and scaling
        I = self.b0 + np.exp(-self.lamb*time)*(self.b1+self.gamma*N)
        return(I)

    def grad_exp_intensity(self, moments, time):
        """
        Calculate expected intensity under product bernoulli approximation
        """
        # get number of stemloops
        N = moments @ self.alpha
        # compute gradient
        grad = np.exp(-self.lamb*time)*(self.b1+self.gamma*N)*self.alpha
        return(grad)

    def var_intensity(self, moments, time):
        """ 
        Calculate variance of intensity under product bernoulli approximation
        """
        prefactor = np.exp(-2*self.lamb*time)*self.gamma**2
        N_sq = np.sum(self.alpha**2*moments*(1-moments)**2)
        var = prefactor*N_sq
        return(var)

    def grad_var_intensity(self, moments, time):
        prefactor = np.exp(-2*self.lamb*time)*self.gamma**2
        grad = prefactor*self.alpha**2*(1-2*moments)
        return(grad)