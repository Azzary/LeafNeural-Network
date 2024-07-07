import numpy as np
from .Activation import Activation

class ReLU(Activation):
    
    def __init__(self):
        super().__init__(self.relu, self.relu_prime)
    
    def relu(self, x: float) -> float:
        return np.maximum(0, x)
    
    def relu_prime(self, x: float) -> float:
        return np.where(x > 0, 1, 0)
