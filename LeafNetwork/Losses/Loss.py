from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    
    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss between the true labels and the predicted labels.

        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Computed loss as a float
        """
        pass

    @abstractmethod
    def compute_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.

        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Computed gradient as a numpy array
        """
        pass