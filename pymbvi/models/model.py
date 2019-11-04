from abc import ABC, abstractmethod


class Model(ABC):
    """
    ABstract model class. Methods are required by the variational engine. 
    """

    @abstractmethod
    def forward(self, *args):
        pass
    
    @abstractmethod
    def backward(self, *args):
        pass

    @abstractmethod
    def control_gradient(self, *args):
        pass
    
    @abstractmethod
    def rates_gradient(self, *arg):
        pass

    # method no longer part of the interface to variational engine
    # @abstractmethod
    # def natural_moments(self, *args):
    #     pass

    @abstractmethod
    def get_initial(self, *args):
        pass

    @abstractmethod
    def kl_prior(self, *args):
        pass

    @abstractmethod
    def num_controls():
        pass

    @abstractmethod
    def num_states():
        pass

    @abstractmethod
    def num_param():
        pass