import numpy as np
from .Activation import Activation

class ReLU(Activation):
    
    def __init__(self):
        super().__init__()
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    
