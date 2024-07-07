import numpy as np

def mse(y_true, y_pred):
    diff = y_true - y_pred
    diff = np.clip(diff, -1e15, 1e15)
    return np.mean(np.square(diff), dtype=np.float64)

def mse_prime(y_true, y_pred):
    diff = y_pred - y_true
    diff = np.clip(diff, -1e15, 1e15)
    return 2 * diff / np.size(y_true)

        
def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / y_pred.shape[0]