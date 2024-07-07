from .Activation import Activation
import numpy as np

class Softmax(Activation):
    
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Soustraction du maximum pour la stabilité numérique
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def softmax_prime(self, x: np.ndarray) -> np.ndarray:
        s = self.softmax(x)
        return s * (1 - s)