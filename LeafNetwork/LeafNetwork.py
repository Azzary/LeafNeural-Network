import numpy as np
import json
import importlib
from typing import Callable, List, Type
from .Layers.LeafLayer import LeafLayer
from .Losses import Loss, MSE
from .Utils.LearningRateUtils import adjusment_func, no_lr_adjustment, adaptive_lr





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


    def train_with_rollback(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, 
                            lr_adjustment_func: Callable = no_lr_adjustment) -> List[np.float64]:
        """
        Trains a neural network model with the ability to rollback to previous weights and biases if the error increases between last epochs and current epoch.
        
        Parameters:
        X (np.ndarray): Input data, reshaped to three dimensions if it is two-dimensional.
        Y (np.ndarray): Target data, reshaped to three dimensions if it is two-dimensional.
        epochs (int): Number of training iterations.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_adjustment_func (Callable): Function to adjust the learning rate based on current training progress.

        Returns:
        List[np.float64]: List of recorded errors after each epoch.
        """
        X = X.reshape(X.shape[0], X.shape[1], 1) if X.ndim == 2 else X
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1) if Y.ndim == 2 else Y

        current_lr = learning_rate
        previous_error = None
        previous_weights = [layer.weights.copy() if hasattr(layer, 'weights') else None for layer in self.layers]
        previous_biases = [layer.bias.copy() if hasattr(layer, 'bias') else None for layer in self.layers]

        for epoch in range(epochs):
            error: np.float64 = np.float64(0)
            for x, y in zip(X, Y):
                output = self.forward(x)
                error += self.loss.compute_loss(y, output)
                grad = self.loss.compute_gradient(y, output)
                self.backward(grad, current_lr)

            error /= len(X)

            if previous_error is not None and error > previous_error:
                # Rollback to previous weights and biases
                for i, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        layer.weights = previous_weights[i].copy()
                    if hasattr(layer, 'bias'):
                        layer.bias = previous_biases[i].copy()
                
                current_lr *= 0.5
                print(f"Epoch: {epoch} - Error increased. Rolling back. New Learning Rate: {current_lr:.6f}")
            else:
                self.error_history.append(error)
                current_lr = lr_adjustment_func(current_lr, error, previous_error, epoch, 1e-6)
                
                # Update previous weights and biases
                previous_weights = [layer.weights.copy() if hasattr(layer, 'weights') else None for layer in self.layers]
                previous_biases = [layer.bias.copy() if hasattr(layer, 'bias') else None for layer in self.layers]
                
                print(f"Epoch: {epoch} - Error: {error:.6f} - Learning Rate: {current_lr:.6f}")

            previous_error = error

        return self.error_history
    
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, 
              lr_adjustment_func: adjusment_func = no_lr_adjustment) -> List[np.float64]:
        
        """
        Trains a neural network model without rollback functionality.

        Parameters:
        X (np.ndarray): Input data, reshaped to three dimensions if it is two-dimensional.
        Y (np.ndarray): Target data, reshaped to three dimensions if it is two-dimensional.
        epochs (int): Number of training iterations.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_adjustment_func (Callable): Function to adjust the learning rate based on current training progress.

        Returns:
        List[np.float64]: List of recorded errors after each epoch.
        """
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