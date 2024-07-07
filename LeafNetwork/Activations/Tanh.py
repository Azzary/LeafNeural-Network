from .Activation import Activation
import numpy as np

class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation. Maps inputs to [-1, 1]. Often used
    in recurrent neural networks. Can suffer from vanishing gradients in very
    deep networks.
    """
    
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)
    
    def tanh(self, x: float) -> float:
        x = np.clip(x, -500, 500)
        return 2 / (1 + np.exp(-2*x)) - 1
    
    def tanh_prime(self, x: float) -> float:
        x = np.clip(x, -500, 500)
        return 1 - self.tanh(x)**2