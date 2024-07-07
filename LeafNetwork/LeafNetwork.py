
from .Layers.Dense import Dense
from .Activations.ReLU import ReLU
from .Activations.Softmax import Softmax
from .Activations.Tanh import Tanh
import numpy as np
from .Layers.LeafLayer import LeafLayer
import json
from .Losses import Loss, MSE


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
            "layers": []
        }
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer_data = {
                    "type": "Dense",
                    "input_size": layer.weights.shape[1],
                    "output_size": layer.weights.shape[0],
                    "weights": layer.weights.tolist(),
                    "bias": layer.bias.tolist()
                }
            elif isinstance(layer, ReLU):
                layer_data = {"type": "ReLU"}
            elif isinstance(layer, Softmax):
                layer_data = {"type": "Softmax"}
            elif isinstance(layer, Tanh):
                layer_data = {"type": "Tanh"}
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            model_data["layers"].append(layer_data)
        
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filename: str) -> 'LeafNetwork':
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        nn = cls(model_data["input_size"])
        for layer_data in model_data["layers"]:
            if layer_data["type"] == "Dense":
                layer = Dense(layer_data["input_size"], layer_data["output_size"])
                layer.weights = np.array(layer_data["weights"])
                layer.bias = np.array(layer_data["bias"])
                nn.add(layer)
            elif layer_data["type"] == "ReLU":
                nn.add(ReLU())
            elif layer_data["type"] == "Softmax":
                nn.add(Softmax())
            elif layer_data["type"] == "Tanh":
                nn.add(Tanh())
        
        return nn