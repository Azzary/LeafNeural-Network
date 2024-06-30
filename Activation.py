from LeafLayer import LeafLayer
import numpy as np

class Activation(LeafLayer):
    
    def __init__(self, activation: callable, activation_prime: callable):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input: float = None
        
    def forward(self, input: float):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))