from .Activation import Activation
import numpy as np

class Softmax(Activation):
    
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Soustraction du maximum pour la stabilité numérique
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def softmax_prime(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient * self.output * (1 - self.output)