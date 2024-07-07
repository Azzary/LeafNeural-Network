import numpy as np
from .Activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function. Maps input to [0,1]. Useful for binary
    classification and output layer probabilities. Can suffer from vanishing
    gradients in deep networks.
    """
    
    def __init__(self):
        super().__init__()
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.activation(x)
        return s * (1 - s)
    
