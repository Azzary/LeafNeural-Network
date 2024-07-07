
from .Layers.Dense import Dense
from .Activations.ReLU import ReLU
from .Activations.Softmax import Softmax
from .Activations.Tanh import Tanh
import numpy as np
from .Layers.LeafLayer import LeafLayer
import json
from .Losses import Loss, MSE
import importlib

class LeafNetwork:
    
    def __init__(self, input_size: int, loss: Loss = MSE()):
        self.layers = []
        self.input_size = input_size
        self.error_history: list = []
        self.loss = loss
        
    def add(self, layer: LeafLayer):
        self.layers.append(layer)

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            input = input.reshape(-1, 1)
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad: np.ndarray, learning_rate: float):
        if output_grad.ndim == 1:
            output_grad = output_grad.reshape(-1, 1)
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, learning_rate)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float) -> list:
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

        error_history = []

        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.forward(x)
                error += self.loss.compute_loss(y, output)
                grad = self.loss.compute_gradient(y, output)
                self.backward(grad, learning_rate)

            error /= len(X)
            error_history.append(error)
            print(f"Epoch: {epoch} - Error: {error:.6f}")

        self.error_history = error_history 
        return error_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return np.array([self.forward(x).flatten() for x in X])

    def save(self, filename: str):
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

        input_size = model_data['input_size']
        
        layers = []
        for layer_data in model_data['layers']:
            module_name = layer_data['module']
            class_name = layer_data['type']
            module = importlib.import_module(module_name)
            layer_class: LeafLayer = getattr(module, class_name)
            layer = layer_class.load(layer_data)
            layers.append(layer)
        
        loss_module_name = model_data['loss']['module']
        loss_class_name = model_data['loss']['type']
        loss_module = importlib.import_module(loss_module_name)
        loss_class = getattr(loss_module, loss_class_name)
        loss_instance = loss_class()

        nn_instance = cls(input_size, loss_instance)
        nn_instance.layers = layers
        return nn_instance