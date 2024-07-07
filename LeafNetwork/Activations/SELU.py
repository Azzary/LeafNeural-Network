import numpy as np
from .Activation import Activation

class SELU(Activation):
    """
    Scaled Exponential Linear Unit. Self-normalizing variant of ELU.
    Useful for deep networks, helps maintain mean 0 and variance 1.
    """
    
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        super().__init__(self.selu, self.selu_prime)
    
    def selu(self, x):
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def selu_prime(self, x):
        return self.scale * np.where(x > 0, 1, self.alpha * np.exp(x))