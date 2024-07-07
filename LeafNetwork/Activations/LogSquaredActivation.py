from .Activation import Activation
import numpy as np

class LogSquaredActivation(Activation):
    """
    Experimental activation function. Just for testing.
    Log-Squared Activation. 
    """
    
    def __init__(self):
        super().__init__(self.log_squared, self.log_squared_prime)
    
    def log_squared(self, x: float) -> float:
        return np.sign(x) * np.log(1 + x**2)
    
    def log_squared_prime(self, x: float) -> float:
        return 2 * x / (1 + x**2)