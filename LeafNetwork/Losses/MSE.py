import numpy as np
from .Loss import Loss

class MSE(Loss):
    EPSILON = 1e-8

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        diff = y_true - y_pred
        diff = np.clip(diff, -1e5, 1e5)
        squared_diff = np.square(diff)
        
        if np.all(np.isnan(squared_diff)):
            print("Warning: All values are NaN in MSE computation")
            return np.float64(self.EPSILON)
        
        mean_squared_diff = np.nanmean(squared_diff)
        
        if np.isnan(mean_squared_diff):
            print("Warning: Mean is NaN in MSE computation")
            return np.float64(self.EPSILON)
        
        return np.float64(mean_squared_diff + self.EPSILON)

    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        diff = np.clip(diff, -1e5, 1e5)
        gradient = 2 * diff / (np.size(y_true) + self.EPSILON)
        
        if np.any(np.isnan(gradient)):
            print("Warning: NaN values in gradient computation")
            gradient = np.nan_to_num(gradient, nan=0.0)
        
        return gradient