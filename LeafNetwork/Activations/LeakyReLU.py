import numpy as np
from .Activation import Activation

class LeakyReLU(Activation):
    """
    Leaky ReLU allows small negative gradients, preventing "dead neurons".
    Good for hidden layers in deep networks, especially unsupervised learning.
    """
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super().__init__()
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)