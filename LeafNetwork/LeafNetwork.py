import numpy as np
import json
import importlib
from typing import Callable, List, Type
from .Layers.LeafLayer import LeafLayer
from .Losses import Loss, MSE
adjusment_func = Callable[[float, float, float, int, float], float]






class LeafNetwork:
    def __init__(self, input_size: int, loss: Loss = MSE()):
        self.layers: List[LeafLayer] = []
        self.input_size = input_size
        self.error_history: List[np.float64] = []
        self.loss = loss

    def add(self, layer: LeafLayer) -> None:
        self.layers.append(layer)

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = self._ensure_2d(input)
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad: np.ndarray, learning_rate: float) -> None:
        output_grad = self._ensure_2d(output_grad)
        i = 0
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, learning_rate)
            i += 1
        
    @staticmethod
    def adaptive_lr(current_lr: float, current_error: float, previous_error: float, epoch: int, min_lr: float = 0.00001) -> float:
        if epoch == 0 or previous_error == 0:
            return current_lr
        
        error_ratio = current_error / previous_error
        
        if error_ratio > 1:
            increase_percentage = (error_ratio - 1) * 100
            decay_rate = min(0.5, increase_percentage / 100)
            new_lr = max(current_lr * (1 - decay_rate), min_lr)
        elif error_ratio < 0.99:
            decrease_percentage = (1 - error_ratio) * 100
            increase_rate = min(0.1, decrease_percentage / 200)
            new_lr = min(current_lr * (1 + increase_rate), current_lr * 1.5)
        else:
            new_lr = current_lr
        
        return max(new_lr, min_lr)
        
    @staticmethod
    def no_lr_adjustment(current_lr: float, current_error: float, 
                            previous_error: float, epoch: int, 
                            min_lr: float = 1e-6) -> float:
        return current_lr 

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, 
              lr_adjustment_func: adjusment_func = no_lr_adjustment) -> List[np.float64]:
        X = X.reshape(X.shape[0], X.shape[1], 1) if X.ndim == 2 else X
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1) if Y.ndim == 2 else Y

        current_lr = learning_rate
        previous_error = None

        for epoch in range(epochs):
            error: np.float64 = np.float64(0)
            for x, y in zip(X, Y):
                output = self.forward(x)
                error += self.loss.compute_loss(y, output)
                grad = self.loss.compute_gradient(y, output)
                self.backward(grad, current_lr)

            error /= len(X)
            self.error_history.append(error)
            
            # Ajuster le taux d'apprentissage
            current_lr = lr_adjustment_func(current_lr, error, previous_error, epoch, 1e-6)
            
            print(f"Epoch: {epoch} - Error: {error:.6f} - Learning Rate: {current_lr:.6f}")

            previous_error = error

        return self.error_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(X.shape[0], X.shape[1], 1) if X.ndim == 2 else X
        return np.array([self.forward(x).flatten() for x in X])

    def save(self, filename: str) -> None:
        model_data = {
            "input_size": self.input_size,
            "layers": [layer.save() for layer in self.layers],
            "loss": {
                "type": self.loss.__class__.__name__,
                "module": self.loss.__class__.__module__
            }
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filename: str) -> 'LeafNetwork':
        with open(filename, 'r') as f:
            model_data = json.load(f)

        def load_class(module_name: str, class_name: str) -> Type:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        layers = [load_class(layer_data['module'], layer_data['type']).load(layer_data)
                  for layer_data in model_data['layers']]

        loss_class = load_class(model_data['loss']['module'], model_data['loss']['type'])
        loss_instance = loss_class()

        nn_instance = cls(model_data['input_size'], loss_instance)
        nn_instance.layers = layers
        return nn_instance