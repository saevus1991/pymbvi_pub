from abc import ABC, abstractmethod


class ObservationModel(ABC):
    """
    ABstract observationmodel class. Methods are required by the variational engine. 
    """

    @abstractmethod
    def sample(self, *args):
        pass

    def llh(self, *args):
        pass
    
    @abstractmethod
    def get_residual(self, *args):
        pass

    @abstractmethod
    def get_terminal(self, *args):
        pass