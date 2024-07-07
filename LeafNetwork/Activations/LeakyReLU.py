import numpy as np
from .Activation import Activation

class LeakyReLU(Activation):
    """
    Leaky ReLU allows small negative gradients, preventing "dead neurons".
    Good for hidden layers in deep networks, especially unsupervised learning.
    """
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super().__init__(self.leaky_relu, self.leaky_relu_prime)
    
    def leaky_relu(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def leaky_relu_prime(self, x):
        return np.where(x > 0, 1, self.alpha)