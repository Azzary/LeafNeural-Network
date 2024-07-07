import numpy as np
from .Activation import Activation

class ELU(Activation):
    """
    Exponential Linear Unit. Allows negative outputs, pushing mean activations
    closer to zero. Can speed up learning in deep networks.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__(self.elu, self.elu_prime)
    
    def elu(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def elu_prime(self, x):
        return np.where(x > 0, 1, self.alpha * np.exp(x))