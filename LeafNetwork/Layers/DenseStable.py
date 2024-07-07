import numpy as np
from .LeafLayer import LeafLayer

class DenseStable(LeafLayer):
    """
    A stable implementation of a dense (fully connected) layer for neural networks.
    This layer includes protections against exploding gradients and NaN values:
    - Weights are initialized using He initialization for better gradient flow
    - Gradients are normalized if they exceed a certain threshold
    - Bias is initialized to zero to start from a neutral position
    
    This implementation aims to provide a more robust training process,
    especially for deep networks or when dealing with varied input distributions.
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.input: np.ndarray = np.array([])
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        # Gradient normalization to prevent exploding gradients
        weights_gradient_norm = np.linalg.norm(weights_gradient)
        if weights_gradient_norm > 1:
            weights_gradient /= weights_gradient_norm
        
        bias_gradient_norm = np.linalg.norm(bias_gradient)
        if bias_gradient_norm > 1:
            bias_gradient /= bias_gradient_norm

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return np.dot(self.weights.T, output_gradient)