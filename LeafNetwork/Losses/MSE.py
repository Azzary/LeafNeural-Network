import numpy as np
from .Loss import Loss

class MSE(Loss):
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_true - y_pred
        diff = np.clip(diff, -1e15, 1e15)
        return np.mean(np.square(diff), dtype=np.float64)

    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        diff = np.clip(diff, -1e15, 1e15)
        return 2 * diff / np.size(y_true)