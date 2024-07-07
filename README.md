# LeafNetwork

LeafNetwork is a custom neural network library built from scratch in Python, designed for educational purposes. It provides a simple framework to create, train, and evaluate neural networks using basic layers and activation functions.

## Features

- Dense (fully connected) layers with stability improvements
- Activation functions: ReLU, Tanh, Softmax, ... (check LeafNetwork/activations/ folder)
- Loss functions: Mean Squared Error (MSE), Categorical Cross Entropy
- Customizable neural network architecture
- Dynamic learning rate adjustment
- Train with rollback mechanism to prevent divergence
- Evaluation on the MNIST dataset

## Performance

The LeafNetwork model has been tested on the MNIST dataset provided by TensorFlow (`tf.keras.datasets.mnist`).

### Training and Testing Details

- **Dataset:** MNIST
- **Training Samples:** 40,000
- **Number of Epochs:** 25
- **Initial Learning Rate:** 0.006
- **Learning Rate Adjustment:** Adaptive percentage-based
- **Test Accuracy:** 0.9666 (96.66%)

With a neural network implemented entirely from scratch, without using any deep learning framework

### Visualization

![Confusion Matrix](https://github.com/Azzary/LeafNeural-Network/blob/main/images/confusion_matrix.png)

The confusion matrix provides a detailed view of the model's performance across different digits, helping to identify any patterns in misclassifications.

## Future Directions

While the current model performs well, there's always room for improvement. Here are some ideas for future enhancements:

1. Add new types of layers to handle different kinds of data better.
2. Implement techniques to prevent overfitting and improve the model's ability to generalize.
3. Try out different methods for adjusting the model's parameters during training.
4. Explore ways to make the training process more stable and efficient.

## Extensibility

LeafNetwork is designed to be easily extensible. You can add custom components as follows:

### Custom Activation Functions

To add a custom activation function, create a class that inherits from `Activation` and implement the following methods:

```python
class CustomActivation(Activation):
    @abstractmethod
    def activation(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def activation_derivative(self, input: np.ndarray) -> np.ndarray:
        pass
```

### Custom Layers

To add a custom layer same as activation function, create a class that inherits from `Layer` and implement the following methods:

```python
class CustomLayer(LeafLayer):
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        pass
```

### Custom Loss Functions

Once again to have cost functions, create a class that inherits from `Loss` and implement the following methods:

```python
class CustomLoss(Loss):
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
```

### Usage Custom Components

To use your custom components in the network:

```python
from CustomLayer import CustomLayer
from CustomActivation import CustomActivation
from CustomLoss import CustomLoss
from LeafNetwork import LearningRateUtils
from LeafNetwork import *

# Create a custom layer
input_size = 10

leaf_network = LeafNetwork(input_size)

leaf_network.add_layer(CustomLayer(10, 5))
leaf_network.add_activation(CustomActivation())
leaf_network.add_layer(DenseStable(5, 5))
leaf_network.add_activation(ReLU())
leaf_network.add_layer(DenseStable(5, 1))

history = leaf_network.train(X_train[:100], y_train_one_hot[:100], epochs=10, learning_rate=0.001)
# or 
history = nn.train(X_train[:100], y_train_one_hot[:100], epochs=10, learning_rate=0.001, lr_adjustment_func= LearningRateUtils.adaptive_percentage)
# or 
history = nn.train_with_rollback(X_train[:100], y_train_one_hot[:100], epochs=10, learning_rate=0.001, lr_adjustment_func= LearningRateUtils.adaptive_percentage)

res = leaf_network.predict(input_data)
```
