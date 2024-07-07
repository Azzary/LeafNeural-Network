from .Activation import Activation
import numpy as np

class Softmax(Activation):
    """
    Softmax activation. Converts raw scores to probabilities. Commonly used
    in the output layer for multi-class classification problems.
    """
    
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.activation(x)
        return s * (1 - s)