import numpy as np
from .LeafLayer import LeafLayer

class DenseAdam(LeafLayer):
    """
    NOT WORKING AS EXPECTED
    A dense (fully connected) layer for neural networks with Adam optimizer.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.input: np.ndarray = np.array([])
        
        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        self.t += 1
        
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        
        # Update biased first moment estimate
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * weights_gradient
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_gradient
        
        # Update biased second raw moment estimate
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(weights_gradient)
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * np.square(bias_gradient)
        
        # Compute bias-corrected first moment estimate
        m_weights_corrected = self.m_weights / (1 - self.beta1**self.t)
        m_bias_corrected = self.m_bias / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_weights_corrected = self.v_weights / (1 - self.beta2**self.t)
        v_bias_corrected = self.v_bias / (1 - self.beta2**self.t)
        
        # Update parameters
        self.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        self.bias -= learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.epsilon)
        
        return np.dot(self.weights.T, output_gradient)