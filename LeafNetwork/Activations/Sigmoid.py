import numpy as np
from .Activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function. Maps input to [0,1]. Useful for binary
    classification and output layer probabilities. Can suffer from vanishing
    gradients in deep networks.
    """
    
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)