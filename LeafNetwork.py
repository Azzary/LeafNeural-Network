
from Dense import Dense
from ReLU import ReLU
from Softmax import Softmax
from Tanh import Tanh
from LossUtils import categorical_cross_entropy, categorical_cross_entropy_prime, mse, mse_prime
import numpy as np
from LeafLayer import LeafLayer
import json

import tensorflow as tf
class LeafNetwork:
    
    def __init__(self, input_size: int):
        self.layers = []
        self.input_size = input_size
        self.error_history: list = []
        
    def add(self, layer: LeafLayer):
        self.layers.append(layer)

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Assurez-vous que l'entrée a la forme attendue, ajoutez des dimensions si nécessaire.
        if input.ndim == 1:
            input = input.reshape(-1, 1)
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad: np.ndarray, learning_rate: float):
        # Assurez-vous que le gradient a la forme correcte
        if output_grad.ndim == 1:
            output_grad = output_grad.reshape(-1, 1)
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, learning_rate)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

        error_history = []

        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.forward(x)
                error += mse(y, output)
                grad = mse_prime(y, output)
                self.backward(grad, learning_rate)

            error /= len(X)
            error_history.append(error)
            print(f"Epoch: {epoch} - Error: {error:.6f}")

        self.error_history = error_history 
        return error_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Ajout d'une dimension aux données si nécessaire
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return np.array([self.forward(x).flatten() for x in X])

    def save(self, filename):
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
    def load(cls, filename):
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
    
    
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

if __name__ == "__main__":
    # Charger les données MNIST
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Prétraitement des données
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    y_train_one_hot = to_one_hot(y_train)
    
    # Initialiser le réseau
    nn = LeafNetwork(784)
    nn.add(Dense(784, 128))
    nn.add(ReLU())
    nn.add(Dense(128, 64))
    nn.add(ReLU())
    nn.add(Dense(64, 10))
    # Entraîner le réseau
    # Vous devrez convertir y_train en une forme appropriée pour la classification, par exemple en utilisant one-hot encoding
    nn.train(X_train[:10000], y_train_one_hot[:10000], epochs=2, learning_rate=0.0006)
    nn.save("mnist_model.json")
    # Tester le réseau
    predictions = nn.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    print("")
    accuracy = np.mean(predicted_labels == y_test)
    print(f"Test Accuracy: {accuracy}")

