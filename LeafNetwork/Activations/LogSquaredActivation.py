from .Activation import Activation
import numpy as np

class LogSquaredActivation(Activation):
    """
    Experimental activation function. Just for testing.
    Log-Squared Activation. 
    """
    
    def __init__(self):
        super().__init__()
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log(1 + x**2)
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        return 2 * x / (1 + x**2)
    
            
    
    