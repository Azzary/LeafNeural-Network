from abc import abstractmethod
from ..Layers.LeafLayer import LeafLayer
import numpy as np

class Activation(LeafLayer):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    
    @abstractmethod
    def activation(self, input: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def activation_derivative(self, input: np.ndarray) -> np.ndarray:
        pass